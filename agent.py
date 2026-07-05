"""Agent 主循环：function-calling + 结构化事件流。

chat_stream() 是一个生成器，按顺序产出事件：
  {"type":"tool_call","name","arguments"}  模型调用了某工具
  {"type":"token","delta"}                  回复的增量文本（覆盖所有回合：
                                             工具调用回合若模型带前导文本也会产出）
  {"type":"sources","items"}                RAG 引用来源（无则不产出）
  {"type":"done"}                            本轮结束

流式细节被封装在这里和 LLMClient.stream_turn 中
"""
import json

from memory.base import NoOpLongTermMemory

# 工具调用最大轮次：防止工具/模型异常导致无限循环
MAX_TOOL_ROUNDS = 10
KNOWLEDGE_BASE_HINTS = (
    "知简",
    "知识库",
    "本地文档",
    "文档里",
    "资料里",
    "调用rag",
    "rag",
    "专业版",
    "免费版",
    "团队版",
    "全局搜索",
    "快捷键",
    "版本历史",
    "历史记录",
    "导入",
    "退款",
    "最新版本",
    "版本号",
    "同步",
    "隐私",
    "训练",
    "华东节点",
    "客服",
    "创始人",
    "总部",
    "印象笔记",
)


class Agent:
    def __init__(self, llm, registry, short_term, long_term):
        self.llm = llm
        self.registry = registry
        self.short_term = short_term
        self.long_term = long_term

    def chat_stream(self, user_msg: str):
        self.short_term.add("user", user_msg)

        messages = self.short_term.get_context()
        insert_at = 1
        if not isinstance(self.long_term, NoOpLongTermMemory):
            messages.insert(insert_at, {
                "role": "system",
                "content": (
                    "你有一个后台长期记忆模块。用户表达稳定事实或偏好、"
                    "或明确要求记住个人信息时，简短确认即可；系统会在本轮回复后自动保存。"
                ),
            })
            insert_at += 1

        facts = self.long_term.recall(user_msg)
        if facts:
            fact_text = "已知用户信息：\n" + "\n".join(f"- {f}" for f in facts)
            messages.insert(insert_at, {"role": "system", "content": fact_text})

        turn_sources: list = []
        knowledge_context = self._prefetch_knowledge_base(user_msg)
        tools_for_model = self.registry.schemas()
        if knowledge_context is not None:
            yield {"type": "tool_call", "name": "knowledge_base", "arguments": {"query": user_msg}}
            turn_sources.extend(knowledge_context["sources"])
            tools_for_model = [
                schema for schema in tools_for_model
                if schema["function"]["name"] != "knowledge_base"
            ]
            if knowledge_context["unanswered"]:
                final_text = f"知识库未提及“{user_msg}”的答案，因此我不能根据文档回答这个问题。"
                yield {"type": "token", "delta": final_text}
                self.short_term.add("assistant", final_text)
                self.long_term.remember(user_msg, final_text)
                yield {"type": "done"}
                return
            messages.insert(insert_at, {
                "role": "system",
                "content": (
                    "本轮已经预先检索本地知识库。对于与知简、本地文档或知识库有关的问题，"
                    "下面检索结果的优先级高于早先对话摘要、长期记忆和模型常识。"
                    "只能依据检索结果回答；如果检索结果说知识库未提及，必须直接说明未提及，"
                    "不得猜测、不得沿用早先错误回答、不得改用网络搜索补充。\n\n"
                    + knowledge_context["result"]
                ),
            })
            insert_at += 1

        final_text = ""
        text = ""
        for _round in range(MAX_TOOL_ROUNDS):
            text, tool_calls = yield from self._run_turn(messages, tools_for_model)
            if not tool_calls:
                final_text = text
                break
            messages.append({
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}  # 模型偶尔返回非法 JSON，降级为空参数而非崩溃
                yield {"type": "tool_call", "name": tc["name"], "arguments": args}
                tool = self.registry.get(tc["name"])
                if tool is not None:
                    tool.last_sources = []  # 重置，避免本次失败时残留上一轮的陈旧来源
                result = self.registry.dispatch(tc["name"], args)
                if tool is not None:
                    turn_sources.extend(getattr(tool, "last_sources", []) or [])
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        else:
            # 达到工具调用轮次上限：用最后一轮文本兜底收尾，避免无限循环
            final_text = text or "（已达到最大工具调用轮次，提前结束）"

        if turn_sources:
            yield {"type": "sources", "items": turn_sources}

        self.short_term.add("assistant", final_text)
        self.long_term.remember(user_msg, final_text)
        yield {"type": "done"}

    def _prefetch_knowledge_base(self, user_msg: str) -> dict | None:
        if not self._should_prefetch_knowledge_base(user_msg):
            return None
        tool = self.registry.get("knowledge_base")
        if tool is None or not tool.is_available():
            return None
        tool.last_sources = []
        result = self.registry.dispatch("knowledge_base", {"query": user_msg})
        return {
            "result": result,
            "sources": getattr(tool, "last_sources", []) or [],
            "unanswered": result.startswith("【本地知识库检索结果：未找到可回答依据】"),
        }

    def _should_prefetch_knowledge_base(self, user_msg: str) -> bool:
        text = (user_msg or "").lower()
        return any(hint.lower() in text for hint in KNOWLEDGE_BASE_HINTS)

    def _run_turn(self, messages, tools=None):
        """流式跑一个回合：把文本增量转成 token 事件 yield 出去，
        返回 (完整文本, 工具调用列表)。

        注意：这里手动 next() 而非 `for piece in gen`，因为 for 循环会
        丢弃生成器的 return 值；我们需要用 StopIteration.value 取回工具调用列表。
        """
        chunks: list[str] = []
        gen = self.llm.stream_turn(messages, self.registry.schemas() if tools is None else tools)
        tool_calls: list = []
        try:
            while True:
                piece = next(gen)
                chunks.append(piece)
                yield {"type": "token", "delta": piece}
        except StopIteration as stop:
            tool_calls = stop.value or []
        return "".join(chunks), tool_calls
