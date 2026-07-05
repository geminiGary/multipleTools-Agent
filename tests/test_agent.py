from agent import Agent
from tools.registry import ToolRegistry
from memory.short_term import ShortTermMemory
from memory.base import NoOpLongTermMemory
from tests.fakes import FakeLLM, EchoTool


def _build(turns):
    llm = FakeLLM(turns=turns)
    registry = ToolRegistry()
    registry.register(EchoTool())
    return Agent(
        llm=llm,
        registry=registry,
        short_term=ShortTermMemory(llm),
        long_term=NoOpLongTermMemory(),
    )


def test_plain_answer_streams_tokens_and_done():
    agent = _build([{"content": "你好呀"}])
    events = list(agent.chat_stream("hi"))
    tokens = "".join(e["delta"] for e in events if e["type"] == "token")
    assert tokens == "你好呀"
    assert events[-1]["type"] == "done"


def test_tool_call_then_answer():
    agent = _build([
        {"tool_calls": [{"id": "c1", "name": "echo", "arguments": '{"text": "hi"}'}]},
        {"content": "最终回答"},
    ])
    events = list(agent.chat_stream("调用echo"))
    types = [e["type"] for e in events]
    assert "tool_call" in types
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["name"] == "echo"
    assert call["arguments"] == {"text": "hi"}
    tokens = "".join(e["delta"] for e in events if e["type"] == "token")
    assert tokens == "最终回答"
    assert events[-1]["type"] == "done"


from tools.base import Tool


class _SourceTool(Tool):
    name = "src"
    description = "返回带来源的结果"
    parameters = {"type": "object", "properties": {}}

    def run(self, **kwargs) -> str:
        self.last_sources = [{"doc": "a.md", "score": 0.9, "snippet": "片段"}]
        return "检索结果"


class _KnowledgeTool(Tool):
    name = "knowledge_base"
    description = "本地知识库"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def run(self, query: str) -> str:
        self.last_sources = [{"doc": "知简-价格与套餐.md", "score": 0.9, "snippet": "专业版"}]
        return "【本地知识库检索结果】知简专业版 ¥18/月。"


class _NoAnswerKnowledgeTool(_KnowledgeTool):
    def run(self, query: str) -> str:
        self.last_sources = []
        return f"【本地知识库检索结果：未找到可回答依据】知识库未提及“{query}”的答案。"


class _FactMemory(NoOpLongTermMemory):
    def recall(self, query: str) -> list[str]:
        return ["用户叫小明"]


def test_sources_event_emitted_when_tool_reports_sources():
    llm = FakeLLM(turns=[
        {"tool_calls": [{"id": "c1", "name": "src", "arguments": "{}"}]},
        {"content": "好的"},
    ])
    registry = ToolRegistry()
    registry.register(_SourceTool())
    agent = Agent(llm=llm, registry=registry,
                  short_term=ShortTermMemory(llm), long_term=NoOpLongTermMemory())
    events = list(agent.chat_stream("查一下"))
    src_events = [e for e in events if e["type"] == "sources"]
    assert len(src_events) == 1
    assert src_events[0]["items"][0]["doc"] == "a.md"


def test_knowledge_base_questions_are_prefetched():
    llm = FakeLLM(turns=[{"content": "知识库未提及。"}])
    registry = ToolRegistry()
    registry.register(_KnowledgeTool())
    agent = Agent(llm=llm, registry=registry,
                  short_term=ShortTermMemory(llm), long_term=NoOpLongTermMemory())

    events = list(agent.chat_stream("知简笔记的创始人是谁"))

    call = next(e for e in events if e["type"] == "tool_call")
    assert call["name"] == "knowledge_base"
    assert any("本轮已经预先检索本地知识库" in m["content"] for m in llm.seen_messages[0])
    assert any("知简专业版" in m["content"] for m in llm.seen_messages[0])


def test_unanswered_knowledge_base_prefetch_returns_directly():
    llm = FakeLLM(turns=[{"content": "不应该进入模型"}])
    registry = ToolRegistry()
    registry.register(_NoAnswerKnowledgeTool())
    agent = Agent(llm=llm, registry=registry,
                  short_term=ShortTermMemory(llm), long_term=NoOpLongTermMemory())

    events = list(agent.chat_stream("知简笔记的创始人是谁"))
    text = "".join(e["delta"] for e in events if e["type"] == "token")

    assert "知识库未提及" in text
    assert llm.seen_messages == []


def test_long_term_facts_injected_into_context():
    llm = FakeLLM(turns=[{"content": "好的"}])
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=llm, registry=registry,
                  short_term=ShortTermMemory(llm), long_term=_FactMemory())
    list(agent.chat_stream("hi"))
    first_msgs = llm.seen_messages[0]
    assert any(m["role"] == "system" and "用户叫小明" in m["content"] for m in first_msgs)


def test_stops_at_max_tool_rounds():
    # 工具调用永不停止时，必须在 MAX_TOOL_ROUNDS 处终止而非死循环
    looping_turns = [
        {"tool_calls": [{"id": "c", "name": "echo", "arguments": '{"text":"x"}'}]}
        for _ in range(50)
    ]
    llm = FakeLLM(turns=looping_turns)
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=llm, registry=registry,
                  short_term=ShortTermMemory(llm), long_term=NoOpLongTermMemory())
    events = list(agent.chat_stream("loop"))
    assert events[-1]["type"] == "done"
