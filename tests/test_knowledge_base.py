from rag.vector_store import STORE
from tools.knowledge_base import KnowledgeBaseTool


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def setup_function():
    STORE.clear()


def teardown_function():
    STORE.clear()


def test_unanswered_founder_question_does_not_fabricate():
    STORE.add(
        ["# 知简笔记 · 产品概览\n知简笔记是一款跨平台笔记应用。当前最新版本为 v4.2。"],
        [[1.0, 0.0]],
        [{"doc": "知简-产品概览.md"}],
    )
    tool = KnowledgeBaseTool()
    tool._llm = FakeEmbedder()

    result = tool.run("知简笔记的创始人是谁")

    assert "知识库未提及" in result
    assert "王子浩" not in result
    assert tool.last_sources == []


def test_rerank_finds_team_history_policy():
    STORE.add(
        [
            "## 常用快捷键\n全局搜索 | Ctrl+Shift+F",
            "## 团队版\n版本历史保留 1 年。团队版按年付费。",
        ],
        [[1.0, 0.0], [1.0, 0.0]],
        [{"doc": "知简-功能与快捷键.md"}, {"doc": "知简-价格与套餐.md"}],
    )
    tool = KnowledgeBaseTool()
    tool._llm = FakeEmbedder()

    result = tool.run("团队版的历史记录能保存多久")

    assert "1 年" in result
    assert tool.last_sources[0]["doc"] == "知简-价格与套餐.md"
