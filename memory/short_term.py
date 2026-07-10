"""短期记忆：维护多轮对话历史；超过上限时把较旧的部分摘要压缩。

该模块负责一次会话内的上下文管理：近期原文直接保留，
更早的对话由 LLM 压缩成滚动摘要后注入上下文。
"""

DEFAULT_SYSTEM_PROMPT = (
    "你是一个乐于助人的智能助手，可以调用工具来更好地回答问题。"
    "当用户询问本地知识库、已上传文档、知简笔记或文档内事实时，必须使用 knowledge_base，"
    "或依据系统已经提供的本地知识库检索结果回答。回答这类问题时只能依据检索结果；"
    "检索结果未直接提及答案时，必须说知识库未提及，不要猜测、不要用早先对话摘要或长期记忆补足。"
    "实时新闻、赛程、当前事件等才使用 web_search。"
)


class ShortTermMemory:
    def __init__(self, llm, max_messages: int = 20, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.llm = llm
        self.max_messages = max_messages
        self.system_prompt = system_prompt
        self.history: list[dict] = []  # [{"role","content"}, ...]
        self.summary: str = ""          # 较旧对话的滚动摘要

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self._maybe_compress()

    def get_context(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            msgs.append({"role": "system", "content": f"早先对话摘要：{self.summary}"})
        msgs.extend(self.history)
        return msgs

    def _maybe_compress(self) -> None:
        if len(self.history) <= self.max_messages:
            return
        keep = max(1, self.max_messages // 2)
        old, recent = self.history[:-keep], self.history[-keep:]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        prefix = f"已有摘要：{self.summary}\n" if self.summary else ""
        resp = self.llm.chat([
            {"role": "system", "content": "用简洁中文总结以下对话要点，保留关键事实，去掉寒暄。"},
            {"role": "user", "content": prefix + text},
        ])
        self.summary = resp.choices[0].message.content
        self.history = recent
