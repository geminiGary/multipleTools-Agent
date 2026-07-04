"""服务端端点测试：用 FastAPI TestClient，不需要 API key。

with TestClient(app) 会触发 lifespan 启动钩子——也顺便验证了
RAG 未实现时种子文档入库会被优雅跳过、不会让服务崩溃。
"""
from fastapi.testclient import TestClient

import server
from memory.conversation_store import ConversationStore
from memory.short_term import ShortTermMemory
from tests.fakes import FakeLLM
from server import app


def test_capabilities_baseline():
    with TestClient(app) as client:
        resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "get_weather" in data["tools"]
    assert data["rag_enabled"] is False
    assert data["long_term_enabled"] is True


def test_upload_rejects_bad_extension():
    with TestClient(app) as client:
        resp = client.post(
            "/api/upload",
            files={"file": ("x.exe", b"hello", "application/octet-stream")},
        )
    assert resp.status_code == 400


def test_reset_returns_ok():
    with TestClient(app) as client:
        resp = client.post("/api/reset/some-session")
    assert resp.json() == {"ok": True}


def test_conversations_list_get_rename_delete(tmp_path, monkeypatch):
    test_store = ConversationStore(dir=str(tmp_path))
    monkeypatch.setattr(server, "STORE", test_store)

    st = ShortTermMemory(FakeLLM())
    test_store.save("c1", short_term=st, user_msg="第一条消息", assistant_text="回复")

    with TestClient(server.app) as client:
        listed = client.get("/api/conversations").json()
        assert any(c["session_id"] == "c1" for c in listed["conversations"])

        one = client.get("/api/conversations/c1").json()
        assert one["title"] == "第一条消息"
        assert one["messages"][0] == {"role": "user", "content": "第一条消息"}

        assert client.patch("/api/conversations/c1", json={"title": "改名"}).json() == {"ok": True}
        assert client.get("/api/conversations/c1").json()["title"] == "改名"

        assert client.get("/api/conversations/missing").status_code == 404
        assert client.patch("/api/conversations/missing", json={"title": "x"}).status_code == 404

        assert client.delete("/api/conversations/c1").json() == {"ok": True}
        assert client.get("/api/conversations/c1").status_code == 404
        # 幂等：删除从未存在的会话也返回 ok
        assert client.delete("/api/conversations/never-existed").json() == {"ok": True}
