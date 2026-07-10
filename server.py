"""FastAPI 后端：把 Agent 的事件流通过 SSE 推给前端。

接口（详见规格 §6）：
  POST /api/chat            {session_id, message} -> SSE 事件流
  POST /api/upload          上传文档进知识库（仅 .txt/.md）
  GET  /api/memory/{sid}    查看长期记忆
  POST /api/reset/{sid}     清空会话
  GET  /api/capabilities    当前已启用能力

SSE 事件类型见 agent.chat_stream（tool_call/token/sources/done）；
此外服务端在流式过程中出错时会补发一个 {"type":"error","message"} 事件。
"""
import json
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from factory import build_agent
from memory.base import NoOpLongTermMemory
from memory.conversation_store import ConversationStore

BASE_DIR = pathlib.Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "docs"
ALLOWED_EXT = {".txt", ".md"}
MAX_UPLOAD_BYTES = 1 * 1024 * 1024  # 1MB

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时把 data/docs 下的种子文档入库；知识库依赖不可用时静默跳过（优雅降级）
    try:
        from rag.ingest import ingest_dir
        n = ingest_dir()
        if n:
            print(f"[启动] 已从 data/docs 入库 {n} 个文本块")
    except (ImportError, NotImplementedError):
        pass
    except Exception as e:  # noqa: BLE001 种子入库失败不应阻止服务启动
        print(f"[启动] 种子文档入库失败（不影响启动）: {e}")
    yield


app = FastAPI(title="智能多工具助手", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORE = ConversationStore()

_agents: dict = {}


def get_agent(session_id: str):
    if session_id not in _agents:
        _agents[session_id] = build_agent(session_id, store=STORE)
    return _agents[session_id]


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str = ""


@app.post("/api/chat")
async def chat(req: ChatRequest):
    agent = get_agent(req.session_id)

    def event_stream():
        completed = False
        try:
            for ev in agent.chat_stream(req.message):
                # 以「agent 产出 done」为落盘信号（而非「客户端已收到」）：
                # 即使最后一帧 yield 因客户端断开而失败，本轮也应持久化。
                if ev.get("type") == "done":
                    completed = True
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001 流中出错也要让前端感知
            err = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            if completed:
                try:
                    history = agent.short_term.history
                    assistant_text = history[-1]["content"] if history else ""
                    STORE.save(
                        req.session_id,
                        short_term=agent.short_term,
                        user_msg=req.message,
                        assistant_text=assistant_text,
                    )
                except Exception as e:  # noqa: BLE001 保存失败不应破坏已完成的回复
                    print(f"[保存对话失败（不影响对话）]: {e}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "缺少文件名")
    safe_name = pathlib.Path(file.filename).name  # 去掉任何目录，防路径穿越
    if not safe_name:
        raise HTTPException(400, "文件名不合法")
    if pathlib.Path(safe_name).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, f"只支持 {sorted(ALLOWED_EXT)} 格式")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "文件过大（上限 1MB）")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / safe_name
    path.write_text(raw.decode("utf-8", errors="ignore"), encoding="utf-8")

    try:
        from rag.ingest import ingest_file
        n = ingest_file(str(path))
    except (ImportError, NotImplementedError):
        n = 0
    return {"chunks_indexed": n}


@app.get("/api/memory/{session_id}")
async def get_memory(session_id: str):
    """返回全局长期记忆中的所有事实（与 session_id 无关；session_id 仅用于获取 agent 实例）。"""
    agent = get_agent(session_id)
    return {"facts": agent.long_term.all_facts()}


@app.post("/api/reset/{session_id}")
async def reset(session_id: str):
    _agents.pop(session_id, None)
    STORE.delete(session_id)
    return {"ok": True}


@app.get("/api/capabilities")
async def capabilities(session_id: str = "default"):
    agent = get_agent(session_id)
    names = agent.registry.available_names()
    return {
        "tools": names,
        "rag_enabled": "knowledge_base" in names,
        "long_term_enabled": not isinstance(agent.long_term, NoOpLongTermMemory),
    }


class RenameRequest(BaseModel):
    title: str = ""


@app.get("/api/conversations")
async def list_conversations():
    return {"conversations": STORE.list()}


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    rec = STORE.get(session_id)
    if rec is None:
        raise HTTPException(404, "对话不存在")
    return {"title": rec.get("title", ""), "messages": rec.get("transcript", [])}


@app.patch("/api/conversations/{session_id}")
async def rename_conversation(session_id: str, req: RenameRequest):
    if not STORE.rename(session_id, req.title):
        raise HTTPException(404, "对话不存在")
    return {"ok": True}


@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str):
    _agents.pop(session_id, None)
    STORE.delete(session_id)
    return {"ok": True}


# 托管前端（必须放在所有 /api 路由之后，避免根挂载遮蔽 API）。
# 这样 `uvicorn server:app` 即可在 http://localhost:8000 同时提供 API 与界面。
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
