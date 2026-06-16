from memory.short_term import ShortTermMemory
from tests.fakes import FakeLLM


def test_context_starts_with_system_and_keeps_turn():
    m = ShortTermMemory(FakeLLM(), max_messages=10)
    m.add("user", "你好")
    ctx = m.get_context()
    assert ctx[0]["role"] == "system"
    assert ctx[-1] == {"role": "user", "content": "你好"}


def test_compresses_when_exceeding_limit():
    m = ShortTermMemory(FakeLLM(summary="历史摘要X"), max_messages=4)
    for i in range(6):
        m.add("user", f"消息{i}")
    assert len(m.history) <= 4
    assert m.summary == "历史摘要X"
    ctx = m.get_context()
    assert any(msg["role"] == "system" and "历史摘要X" in msg["content"] for msg in ctx)


def test_compress_handles_tiny_limit():
    # 回归：max_messages//2 可能为 0，曾导致 history[-0:] 取全量、压缩静默失效
    m = ShortTermMemory(FakeLLM(summary="s"), max_messages=1)
    for i in range(6):
        m.add("user", f"m{i}")
    assert len(m.history) <= m.max_messages + 1
