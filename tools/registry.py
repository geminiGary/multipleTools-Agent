"""工具注册表：收集 schema、按名分发，并实现优雅降级。

- schemas()/available_names() 只暴露 is_available() 为真的工具，
  不可用工具对模型不可见。
- dispatch() 兜底捕获异常，单个工具出错不会让整轮对话崩溃。
"""
from tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def available_names(self) -> list[str]:
        return [name for name, t in self._tools.items() if t.is_available()]

    def schemas(self) -> list[dict]:
        return [t.openai_schema() for t in self._tools.values() if t.is_available()]

    def dispatch(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"[错误] 工具 {name} 不存在"
        try:
            return tool.run(**args)
        except NotImplementedError:
            return f"[提示] 工具 {name} 尚未实现"
        except Exception as e:  # noqa: BLE001 兜底，保证对话不崩
            return f"[错误] 工具 {name} 执行失败: {e}"
