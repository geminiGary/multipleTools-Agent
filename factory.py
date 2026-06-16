"""装配真实依赖，产出一个可用的 Agent。

优雅降级在此体现：
- 所有工具都 register，但骨架工具 is_available()=False，对模型隐藏。
- 长期记忆按 long_term.ENABLED 决定用学生实现还是 NoOp。
"""
from llm import LLMClient
from agent import Agent
from tools.registry import ToolRegistry
from tools.weather import WeatherTool
from tools.calculator import CalculatorTool
from tools.web_search import WebSearchTool
from tools.knowledge_base import KnowledgeBaseTool
from memory.short_term import ShortTermMemory
from memory.base import NoOpLongTermMemory
from memory import long_term as lt_mod
from config import USER_ID
from memory.conversation_store import ConversationStore


def build_agent(session_id: str = "default", store: "ConversationStore | None" = None) -> Agent:
    llm = LLMClient()

    registry = ToolRegistry()
    registry.register(WeatherTool())
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(KnowledgeBaseTool())

    short_term = ShortTermMemory(llm)
    # 从持久化记录灌回短期记忆（cache-miss / 服务重启时自动恢复 LLM 上下文）
    if store is None:
        store = ConversationStore()
    record = store.load(session_id)
    snapshot = record.get("short_term") if record else None
    if isinstance(snapshot, dict):
        history = snapshot.get("history")
        short_term.history = list(history) if isinstance(history, list) else []
        summary = snapshot.get("summary")
        short_term.summary = summary if isinstance(summary, str) else ""

    # 长期记忆按全局 USER_ID 存储（与 session_id 分离），所有窗口共享一份
    if getattr(lt_mod, "ENABLED", False):
        long_term = lt_mod.FileLongTermMemory(USER_ID)
    else:
        long_term = NoOpLongTermMemory()

    return Agent(llm=llm, registry=registry, short_term=short_term, long_term=long_term)
