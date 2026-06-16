"""所有工具的统一接口。学生新增工具只需继承 Tool 并实现 run()。"""
from abc import ABC, abstractmethod


class Tool(ABC):
    name: str = ""           # function calling 里的 name（唯一）
    description: str = ""     # 给模型看的功能描述，决定模型何时调用
    parameters: dict = {"type": "object", "properties": {}}  # 参数的 JSON Schema

    def __init__(self):
        # 子类若自定义 __init__，记得调用 super().__init__()。
        # 用实例属性而非类属性，避免多个工具共享同一个 list 互相污染。
        self.last_sources: list = []  # run() 可写入本轮引用来源（如 RAG 出处）

    def is_available(self) -> bool:
        """是否启用。未实现的骨架工具返回 False，从而对模型隐藏（优雅降级）。"""
        return True

    @abstractmethod
    def run(self, **kwargs) -> str:
        """执行工具，返回文本结果。参数与 parameters 中声明的一致。"""
        ...

    def openai_schema(self) -> dict:
        """转成 OpenAI tools 格式（基类已实现，子类无需改动）。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
