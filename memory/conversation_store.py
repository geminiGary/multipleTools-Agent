"""对话持久化（老师基础设施）：每条对话一个 JSON 文件落盘。

职责单一——只负责一条对话状态的存取，且只读写 ShortTermMemory 的公有属性
（.history / .summary），因此 short_term.py 范例保持一行不动。

文件结构 data/conversations/<session_id>.json：
  {
    "session_id", "title", "created_at", "updated_at",
    "transcript": [{"role","content"}, ...],            # append-only 完整逐字记录（界面渲染）
    "short_term": {"history": [...], "summary": "..."}   # 快照（恢复 LLM 上下文）
  }
"""
import json
import os
import pathlib
import re
from datetime import datetime, timezone

TITLE_MAX = 30
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(session_id: str) -> str:
    """去掉非 [A-Za-z0-9_-] 字符，防路径穿越。"""
    return _UNSAFE.sub("", session_id or "")


def _make_title(msg: str) -> str:
    msg = (msg or "").strip().replace("\n", " ")
    if len(msg) > TITLE_MAX:
        return msg[:TITLE_MAX] + "…"
    return msg or "新对话"


class ConversationStore:
    def __init__(self, dir: str = "data/conversations"):
        self.dir = pathlib.Path(dir)

    def _path(self, session_id: str) -> pathlib.Path:
        safe = _safe_id(session_id)
        if not safe:
            raise ValueError(f"非法 session_id（净化后为空）: {session_id!r}")
        return self.dir / f"{safe}.json"

    def load(self, session_id: str) -> dict | None:
        try:
            return json.loads(self._path(session_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def get(self, session_id: str) -> dict | None:
        """语义别名：供「切回渲染」使用。"""
        return self.load(session_id)

    def save(self, session_id: str, *, short_term, user_msg: str, assistant_text: str) -> None:
        rec = self.load(session_id)
        now = _now()
        if rec is None:
            rec = {
                "session_id": session_id,
                "title": _make_title(user_msg),
                "created_at": now,
                "transcript": [],
                "short_term": {},
            }
        rec["transcript"].append({"role": "user", "content": user_msg})
        rec["transcript"].append({"role": "assistant", "content": assistant_text})
        rec["short_term"] = {
            "history": list(short_term.history),
            "summary": short_term.summary,
        }
        rec["updated_at"] = now
        self._write(session_id, rec)

    def list(self) -> list[dict]:
        if not self.dir.exists():
            return []
        items = []
        for path in self.dir.glob("*.json"):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # 跳过损坏 / 不可读文件，不影响列表
            items.append({
                "session_id": path.stem,
                "title": rec.get("title", ""),
                "updated_at": rec.get("updated_at", ""),
                "message_count": len(rec.get("transcript", [])),
            })
        items.sort(key=lambda it: it["updated_at"], reverse=True)
        return items

    def rename(self, session_id: str, title: str) -> bool:
        rec = self.load(session_id)
        if rec is None:
            return False
        rec["title"] = title
        self._write(session_id, rec)
        return True

    def delete(self, session_id: str) -> bool:
        try:
            path = self._path(session_id)
        except ValueError:
            return False
        if not path.exists():
            return False
        path.unlink()
        return True

    def _write(self, session_id: str, rec: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(session_id)
        tmp = self.dir / f"{_safe_id(session_id)}.json.tmp"
        try:
            tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)  # 原子替换，避免半截 JSON
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
