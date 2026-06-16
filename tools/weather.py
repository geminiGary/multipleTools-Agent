"""范例工具 1：天气查询。

教学说明：用内置 mock 数据，保证没有 API key、不联网也能跑通整条链路。
学生写真实工具时，把 run() 里换成真实 HTTP 请求即可。
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
