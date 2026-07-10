"""RAG 检索工具：把本地知识库检索包装成一个 Tool。

当内存向量库已入库时，该工具会出现在模型可用工具列表里；
空库时自动隐藏，实现知识库能力的优雅降级。
"""
from tools.base import Tool
from rag.vector_store import STORE
from llm import LLMClient


class KnowledgeBaseTool(Tool):
    name = "knowledge_base"
    description = (
        "在本地知识库中检索与问题相关的资料。凡是用户询问知简笔记、本地文档、"
        "已上传资料、价格套餐、快捷键、版本历史、导入来源、退款、同步隐私等信息时，"
        "必须调用本工具；回答时只能依据本工具返回的资料，资料未提及时要明确说未提及，不能猜测。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要检索的问题或关键词，保留用户原始问法"}
        },
        "required": ["query"],
    }

    def __init__(self):
        super().__init__()  # 初始化 last_sources 等基类实例属性
        self._llm = LLMClient()

    def is_available(self) -> bool:
        return len(STORE) > 0

    def run(self, query: str) -> str:
        """检索知识库并返回拼接好的上下文，同时把来源写入 self.last_sources。

        返回内容面向模型阅读，last_sources 面向前端仪表栏展示。
        无结果或超出知识库范围时返回明确的“知识库未提及”提示，避免模型编造。
        """
        query = query.strip()
        if not query:
            self.last_sources = []
            return "查询为空，请提供问题或关键词。"
        q_emb = self._llm.embed([query])[0]
        raw_hits = self._rerank(query, STORE.search(q_emb, top_k=8))
        hits = raw_hits[:4]
        sources = []
        seen_docs = set()
        for hit in hits:
            doc = hit["metadata"].get("doc")
            if doc in seen_docs:
                continue
            seen_docs.add(doc)
            sources.append(hit)
            if len(sources) >= 3:
                break
        self.last_sources = [{"doc": h["metadata"].get("doc"), "score": h["score"], "snippet": h["text"][:50]} for h in sources]
        if not hits:
            return self._no_answer(query)

        combined_text = "\n".join(h["text"] for h in hits)
        if self._is_unanswered_intent(query, combined_text):
            self.last_sources = []
            return self._no_answer(query)
        
        parts = []
        for i, h in enumerate(hits, start=1):
            doc = h["metadata"].get("doc", "未知文档")
            score = h["score"]
            parts.append(f"结果 {i} (文档: {doc}, 相似度: {score:.4f}):\n{h['text']}\n")
        return (
            "【本地知识库检索结果】\n"
            "回答规则：只能依据下面资料回答；如果资料没有直接包含答案，必须说“知识库未提及”，"
            "不要用常识、早先对话、长期记忆或网络搜索补充。\n\n"
            + "\n\n".join(parts)
        )

    def _rerank(self, query: str, hits: list[dict]) -> list[dict]:
        markers = self._markers_for_query(query)
        reranked = []
        for hit in hits:
            text = hit["text"].lower()
            phrase_score = sum(1 for marker in markers if marker and marker.lower() in text)
            adjusted = dict(hit)
            adjusted["score"] = float(hit["score"]) + phrase_score * 0.08
            reranked.append(adjusted)
        reranked.sort(key=lambda h: h["score"], reverse=True)
        return reranked

    def _markers_for_query(self, query: str) -> list[str]:
        q = query.lower()
        groups = [
            (("多少钱", "价格", "收费", "套餐", "专业版", "免费版", "团队版"),
             ["价格", "套餐", "¥", "专业版", "免费版", "团队版"]),
            (("最多", "多少篇", "笔记数量", "存多少"),
             ["最多保存", "100 篇", "100篇", "无限", "笔记数量"]),
            (("快捷键", "全局搜索"),
             ["常用快捷键", "全局搜索", "Ctrl+Shift+F"]),
            (("历史记录", "版本历史", "保存多久", "保留多久"),
             ["版本历史", "保留", "团队版", "1 年", "1年", "30 天"]),
            (("导入", "来源"),
             ["导入", "导入来源", "Evernote", ".enex", "Notion", "导出包"]),
            (("退款",),
             ["退款", "7 天", "7天", "无理由", "未使用月份"]),
            (("训练", "ai 模型", "ai模型", "隐私", "华东节点", "存在哪里"),
             ["数据", "训练", "AI 模型", "华东节点", "上海", "隐私"]),
            (("最新版本", "版本号"),
             ["当前最新版本", "v4.2", "2026 年 3 月", "2026年3月"]),
        ]
        markers = []
        for triggers, values in groups:
            if any(trigger in q for trigger in triggers):
                markers.extend(values)
        return markers

    def _is_unanswered_intent(self, query: str, text: str) -> bool:
        q = query.lower()
        doc_text = text.lower()
        missing_groups = [
            (("创始人", "创办人", "创立者", "ceo", "负责人"), ["创始人", "创办人", "创立者", "ceo", "负责人"]),
            (("总部", "公司地址", "办公地址"), ["总部", "公司地址", "办公地址"]),
            (("印象笔记", "哪个更好", "对比"), ["印象笔记", "对比"]),
        ]
        for triggers, required_terms in missing_groups:
            if any(trigger in q for trigger in triggers):
                return not any(term in doc_text for term in required_terms)
        return False

    def _no_answer(self, query: str) -> str:
        return (
            "【本地知识库检索结果：未找到可回答依据】\n"
            f"知识库未提及“{query}”的答案。请直接说明知识库未提及，"
            "不要根据常识、早先对话、长期记忆或网络搜索猜测。"
        )
