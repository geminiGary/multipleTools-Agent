from tools.base import Tool
from tools.registry import ToolRegistry


class OkTool(Tool):
    name = "ok"
    description = "总是可用的测试工具"

    def run(self, **kwargs) -> str:
        return "ok-result"


class HiddenTool(Tool):
    name = "hidden"
    description = "未实现，应被隐藏"

    def is_available(self) -> bool:
        return False

    def run(self, **kwargs) -> str:
        raise NotImplementedError


def test_available_filters_hidden():
    r = ToolRegistry()
    r.register(OkTool())
    r.register(HiddenTool())
    assert r.available_names() == ["ok"]
    assert len(r.schemas()) == 1


def test_dispatch_runs():
    r = ToolRegistry()
    r.register(OkTool())
    assert r.dispatch("ok", {}) == "ok-result"


def test_dispatch_missing_tool():
    assert "不存在" in ToolRegistry().dispatch("nope", {})


def test_dispatch_not_implemented():
    r = ToolRegistry()
    r.register(HiddenTool())
    assert "尚未实现" in r.dispatch("hidden", {})


def test_last_sources_isolated_between_instances():
    a = OkTool()
    b = OkTool()
    a.last_sources.append("x")
    assert b.last_sources == []
