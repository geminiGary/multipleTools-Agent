"""OpenAI 兼容的模型客户端封装。

设计要点：
- chat(): 一次性返回（用于工具决策回合、短期记忆摘要）。
- stream_turn(): 流式回合——逐 token yield 文本增量，并在结束时
  return 组装好的工具调用列表。Agent 用 `yield from` 消费它，
- embed(): 文本嵌入（RAG 用）。
"""
from typing import Generator
from openai import OpenAI
import config


class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.chat_model = config.CHAT_MODEL
        self.embedding_model = config.EMBEDDING_MODEL

    def chat(self, messages, tools=None, tool_choice="auto"):
        kwargs = {"model": self.chat_model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)

    def stream_turn(self, messages, tools=None) -> Generator[str, None, list]:
        """流式生成一个回合。

        yield: 回复文本的增量片段（str）
        return: 本回合模型请求的工具调用 list[dict]，每个含 id/name/arguments
        """
        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            tools=tools or None,
            tool_choice="auto" if tools else None,
            stream=True,
        )
        acc: dict[int, dict] = {}
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
            for tc in (delta.tool_calls or []):
                slot = acc.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        return [acc[i] for i in sorted(acc)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [d.embedding for d in resp.data]
