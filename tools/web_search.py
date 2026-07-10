"""网络搜索工具。

优先调用博查 Web Search API；未配置 API Key 或请求失败时，
使用内置示例结果兜底，确保智能体主流程可继续运行。
"""
from config import BOCHA_SEARCH_API_KEY, BOCHA_SEARCH_ENDPOINT
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

    _MOCK = {
        "Python": "Python 是一种广泛使用的高级编程语言，具有简洁的语法和强大的库支持。",
        "AI": "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
        "天气": "天气是指地球大气层在某一时间和地点的状态，包括温度、湿度、风速等因素。",
    }

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 8.0,
        http_post=None,
    ):
        super().__init__()
        self.api_key = api_key if api_key is not None else BOCHA_SEARCH_API_KEY
        self.endpoint = endpoint or BOCHA_SEARCH_ENDPOINT
        self.timeout = timeout
        self.http_post = http_post

    def is_available(self) -> bool:
        return True

    def run(self, query: str) -> str:
        """执行搜索并返回结果摘要字符串。

        设置 WEB_SEARCH_API_KEY 后调用博查 Web Search API；未设置或请求失败时返回 mock 结果。
        """
        query = query.strip()
        if not query:
            return "搜索关键词不能为空"

        if self.api_key:
            try:
                return self._run_bocha_search(query)
            except Exception as e:  # noqa: BLE001
                return f"真实搜索暂时不可用，已返回示例搜索结果。错误：{e}\n\n{self._mock_result(query)}"

        return self._mock_result(query)

    def _run_bocha_search(self, query: str) -> str:
        http_post = self.http_post
        if http_post is None:
            import httpx

            http_post = httpx.post

        response = http_post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "freshness": "oneYear",
                "summary": True,
                "count": 5,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = self._extract_results(data)
        if not results:
            return f"未找到与 '{query}' 相关的真实搜索结果。"

        lines = [f"网络搜索结果：{query}"]
        self.last_sources = []
        for i, item in enumerate(results[:5], start=1):
            title = item.get("name") or "无标题"
            url = item.get("url") or ""
            site_name = item.get("siteName") or ""
            description = item.get("summary") or item.get("snippet") or "无摘要"
            date_published = item.get("datePublished") or ""
            lines.append(
                f"{i}. {title}\n"
                f"   摘要：{description}\n"
                f"   来源：{site_name} {url}\n"
                f"   时间：{date_published}"
            )
            if url:
                self.last_sources.append({"title": title, "url": url})
        return "\n".join(lines)

    def _extract_results(self, data: dict) -> list:
        payload = data.get("data", data)
        if not isinstance(payload, dict):
            return []
        return payload.get("webPages", {}).get("value", [])

    def _mock_result(self, query: str) -> str:
        results = self._MOCK.get(query)
        if results is None:
            return f"未找到与 '{query}' 相关的搜索结果（示例搜索结果）"
        lines = [f"网络搜索结果：{query}"]
        for i, item in enumerate(results.split("。"), start=1):
            if not item:
                continue
            lines.append(f"{i}. {item}")
        return "\n".join(lines)
