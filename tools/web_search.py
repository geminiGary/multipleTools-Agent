"""【学生任务】网络搜索工具。

教学提示：可以接真实搜索 API，也可以先用 mock 数据跑通流程
（与 weather.py 同理）。完成 run() 后把 is_available() 改为 True。
"""
from tools.base import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = "根据关键词进行网络搜索，返回相关结果摘要。需要实时信息时调用。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["query"],
    }

    def is_available(self) -> bool:
        # TODO: 实现 run() 后改为 return True
        return False

    def run(self, query: str) -> str:
        """执行搜索并返回结果摘要字符串。

        TODO: 调用真实搜索 API，或先返回 mock 结果跑通链路。
        """
        raise NotImplementedError("TODO: 实现网络搜索")
