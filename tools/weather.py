"""天气查询工具。

使用内置示例数据，保证没有外部天气 API 时也能跑通工具调用链路。
如需接入真实服务，可在 run() 中替换为 HTTP 请求并保留相同的返回契约。
"""
from tools.base import Tool


class WeatherTool(Tool):
    name = "get_weather"
    description = "查询指定城市的当前天气。当用户询问某地天气时调用。"
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，例如 北京"}
        },
        "required": ["city"],
    }

    _MOCK = {"北京": "晴，22°C", "上海": "多云，25°C", "广州": "小雨，28°C"}

    def run(self, city: str) -> str:
        return f"{city}当前天气：{self._MOCK.get(city, '晴，20°C（示例数据）')}"
