"""长期记忆的抽象接口 + 空实现。

短期记忆和长期记忆是两种不同角色，接口分开：
- 短期记忆见 short_term.py（维护对话历史）。
- 长期记忆见此处接口；学生在 long_term.py 中实现，未实现时用 NoOp 兜底。
"""
from abc import ABC, abstractmethod


class LongTermMemory(ABC):
    @abstractmethod
    def recall(self, query: str) -> list[str]:
        """根据当前输入召回相关的历史事实，作为上下文注入。"""
        ...

    @abstractmethod
    def remember(self, user_msg: str, reply: str) -> None:
        """从一轮对话中抽取并持久化值得长期记住的事实。"""
        ...

    @abstractmethod
    def all_facts(self) -> list[str]:
        """返回当前存储的全部事实（供 /api/memory 展示）。"""
        ...


class NoOpLongTermMemory(LongTermMemory):
    """长期记忆未实现时的默认实现：什么都不做，保证整体可运行。"""

    def recall(self, query: str) -> list[str]:
        return []

    def remember(self, user_msg: str, reply: str) -> None:
        pass

    def all_facts(self) -> list[str]:
        return []
