from tools.web_search import WebSearchTool


BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"


def test_default_constructor_uses_project_config():
    tool = WebSearchTool()
    assert tool.is_available() is True


def test_known_query_uses_mock_without_api_key():
    result = WebSearchTool(api_key="", endpoint=BOCHA_ENDPOINT).run(query="Python")

    assert "Python" in result
    assert "网络搜索结果" in result


def test_unknown_query_has_mock_fallback():
    result = WebSearchTool(api_key="", endpoint=BOCHA_ENDPOINT).run(query="量子计算")

    assert "量子计算" in result
    assert "示例搜索结果" in result


def test_empty_query():
    result = WebSearchTool(api_key="", endpoint=BOCHA_ENDPOINT).run(query="   ")

    assert "不能为空" in result


def test_schema_shape():
    schema = WebSearchTool(api_key="", endpoint=BOCHA_ENDPOINT).openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert "query" in schema["function"]["parameters"]["properties"]


def test_bocha_api_response_is_formatted():
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "code": 200,
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": "示例标题",
                                "summary": "示例摘要",
                                "snippet": "备用摘要",
                                "url": "https://example.com",
                                "siteName": "Example",
                                "datePublished": "2026-06-20T00:00:00+08:00",
                            }
                        ]
                    }
                }
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    tool = WebSearchTool(
        api_key="fake-key",
        endpoint=BOCHA_ENDPOINT,
        http_post=fake_post,
    )
    result = tool.run(query="测试关键词")

    assert captured["url"] == BOCHA_ENDPOINT
    assert captured["headers"]["Authorization"] == "Bearer fake-key"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["json"]["query"] == "测试关键词"
    assert captured["json"]["freshness"] == "oneYear"
    assert captured["json"]["summary"] is True
    assert captured["json"]["count"] == 5
    assert captured["timeout"] == tool.timeout
    assert "示例标题" in result
    assert "示例摘要" in result
    assert "Example https://example.com" in result
    assert "2026-06-20T00:00:00+08:00" in result
    assert tool.last_sources == [{"title": "示例标题", "url": "https://example.com"}]


def test_bocha_api_uses_snippet_when_summary_is_missing():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "code": 200,
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": "只有片段",
                                "snippet": "这是 snippet 摘要",
                                "url": "https://example.com/snippet",
                            }
                        ]
                    }
                }
            }

    def fake_post(url, headers, json, timeout):
        return FakeResponse()

    tool = WebSearchTool(
        api_key="fake-key",
        endpoint=BOCHA_ENDPOINT,
        http_post=fake_post,
    )
    result = tool.run(query="测试关键词")

    assert "只有片段" in result
    assert "这是 snippet 摘要" in result
    assert "https://example.com/snippet" in result
