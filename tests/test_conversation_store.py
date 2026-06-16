import json
import pytest

from memory.conversation_store import ConversationStore, TITLE_MAX
from memory.short_term import ShortTermMemory
from tests.fakes import FakeLLM


def _short_term(history, summary=""):
    """构造一个带指定状态的 ShortTermMemory（store 只读取 .history / .summary）。"""
    m = ShortTermMemory(FakeLLM())
    m.history = list(history)
    m.summary = summary
    return m


def test_save_then_load_roundtrip(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    st = _short_term(
        [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "在的"}],
        summary="早先摘要",
    )
    store.save("c1", short_term=st, user_msg="你好", assistant_text="在的")
    rec = store.load("c1")
    assert rec["transcript"] == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "在的"},
    ]
    assert rec["short_term"]["summary"] == "早先摘要"
    assert rec["short_term"]["history"] == st.history
    assert rec["created_at"] and rec["updated_at"]


def test_title_from_first_user_message(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    st = _short_term([])
    store.save("c1", short_term=st, user_msg="北京天气", assistant_text="晴")
    assert store.load("c1")["title"] == "北京天气"


def test_title_truncated(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    long_msg = "字" * 50
    st = _short_term([])
    store.save("c1", short_term=st, user_msg=long_msg, assistant_text="ok")
    title = store.load("c1")["title"]
    assert title.startswith("字字")
    assert len(title) == TITLE_MAX + 1  # 30 个字符 + 1 个省略号


def test_title_stable_across_turns_and_appends(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    st = _short_term([])
    store.save("c1", short_term=st, user_msg="第一句", assistant_text="回一")
    store.save("c1", short_term=st, user_msg="第二句", assistant_text="回二")
    rec = store.load("c1")
    assert rec["title"] == "第一句"
    assert len(rec["transcript"]) == 4


def test_list_sorted_desc_with_count(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    (tmp_path / "a.json").write_text(json.dumps({
        "session_id": "a", "title": "A", "updated_at": "2026-01-01T00:00:00+00:00",
        "transcript": [{"role": "user", "content": "x"}],
    }), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({
        "session_id": "b", "title": "B", "updated_at": "2026-02-01T00:00:00+00:00",
        "transcript": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    }), encoding="utf-8")
    items = store.list()
    assert [it["session_id"] for it in items] == ["b", "a"]
    assert items[0]["message_count"] == 2
    assert items[1]["message_count"] == 1


def test_list_empty_when_dir_absent(tmp_path):
    store = ConversationStore(dir=str(tmp_path / "nope"))
    assert store.list() == []


def test_rename(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    st = _short_term([])
    store.save("c1", short_term=st, user_msg="原标题", assistant_text="ok")
    assert store.rename("c1", "新标题") is True
    assert store.load("c1")["title"] == "新标题"
    assert store.rename("missing", "x") is False


def test_delete(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    st = _short_term([])
    store.save("c1", short_term=st, user_msg="hi", assistant_text="yo")
    assert store.delete("c1") is True
    assert store.load("c1") is None
    assert store.delete("c1") is False


def test_load_corrupt_returns_none(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert store.load("bad") is None


def test_invalid_session_id(tmp_path):
    store = ConversationStore(dir=str(tmp_path))
    st = _short_term([])
    # 读取 / 删除 / 重命名：非法 id 优雅返回，不抛
    assert store.load("!!!") is None
    assert store.delete("!!!") is False
    assert store.rename("!!!", "x") is False
    # save：非法 id 无法持久化，显式报错而非静默写入隐藏文件
    with pytest.raises(ValueError):
        store.save("!!!", short_term=st, user_msg="x", assistant_text="y")
