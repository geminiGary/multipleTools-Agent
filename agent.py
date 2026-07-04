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
        final_text = ""
        text = ""
        for _round in range(MAX_TOOL_ROUNDS):
            text, tool_calls = yield from self._run_turn(messages)
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

    def _run_turn(self, messages):
        """流式跑一个回合：把文本增量转成 token 事件 yield 出去，
        返回 (完整文本, 工具调用列表)。

        注意：这里手动 next() 而非 `for piece in gen`，因为 for 循环会
        丢弃生成器的 return 值；我们需要用 StopIteration.value 取回工具调用列表。
        """
        chunks: list[str] = []
        gen = self.llm.stream_turn(messages, self.registry.schemas())
        tool_calls: list = []
        try:
            while True:
                piece = next(gen)
                chunks.append(piece)
                yield {"type": "token", "delta": piece}
        except StopIteration as stop:
            tool_calls = stop.value or []
        return "".join(chunks), tool_calls
