import json
from factory import build_agent
from memory.base import NoOpLongTermMemory
from memory.conversation_store import ConversationStore


def test_baseline_capabilities():
    agent = build_agent("test-session")
    names = agent.registry.available_names()
    # 基线：两个范例工具可用；未实现的工具被隐藏
    assert "get_weather" in names
    assert "calculator" in names
    assert "web_search" not in names
    assert "knowledge_base" not in names
    # 长期记忆默认关闭 -> NoOp
    assert isinstance(agent.long_term, NoOpLongTermMemory)


def test_build_agent_seeds_short_term_from_store(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    (tmp_path / "s1.json").write_text(json.dumps({
        "session_id": "s1", "title": "T", "updated_at": "2026-01-01T00:00:00+00:00",
        "transcript": [{"role": "user", "content": "hi"}],
        "short_term": {
            "history": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ],
            "summary": "早先摘要",
        },
    }), encoding="utf-8")
    agent = build_agent("s1", store=store)
    assert agent.short_term.summary == "早先摘要"
    assert agent.short_term.history[-1] == {"role": "assistant", "content": "yo"}


def test_build_agent_no_record_starts_empty(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    agent = build_agent("new", store=store)
    assert agent.short_term.history == []
    assert agent.short_term.summary == ""


def test_long_term_keyed_by_global_user_id(monkeypatch, tmp_path):
    """启用长期记忆时，无论 session_id 是什么，都按固定全局 USER_ID 构造。"""
    from memory import long_term as lt_mod
    from config import USER_ID

    captured = {}

    class SpyLTM:
        def __init__(self, user_id="local", path="data/memory_store.json"):
            captured["id"] = user_id

        def recall(self, query):
            return []

        def remember(self, user_msg, reply):
            pass

        def all_facts(self):
            return []

    monkeypatch.setattr(lt_mod, "ENABLED", True)
    monkeypatch.setattr(lt_mod, "FileLongTermMemory", SpyLTM)
    build_agent("session-XYZ", store=ConversationStore(dir=str(tmp_path)))
    assert captured["id"] == USER_ID
    assert captured["id"] != "session-XYZ"


def test_build_agent_tolerates_malformed_short_term(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    # short_term 字段类型错误（应为 dict，这里给 list）——不应让构建崩溃
    (tmp_path / "bad.json").write_text(json.dumps({
        "session_id": "bad", "title": "x", "updated_at": "2026-01-01T00:00:00+00:00",
        "transcript": [],
        "short_term": ["不是 dict"],
    }), encoding="utf-8")
    agent = build_agent("bad", store=store)
    assert agent.short_term.history == []
    assert agent.short_term.summary == ""
