"""OpenAI 兼容的模型客户端封装。

设计要点：
- chat(): 一次性返回（用于工具决策回合、短期记忆摘要）。
- stream_turn(): 流式回合——逐 token yield 文本增量，并在结束时
  return 组装好的工具调用列表。Agent 用 `yield from` 消费它，
- embed(): 文本嵌入（RAG 用）。
"""
from typing import Generator
import hashlib
import math

from openai import OpenAI
import config


class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.chat_model = config.CHAT_MODEL
        self.embedding_model = config.EMBEDDING_MODEL
        self._embedding_remote_disabled = False

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
        if not texts:
            return []
        if not self._embedding_remote_disabled:
            try:
                resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
                return [d.embedding for d in resp.data]
            except Exception as exc:  # noqa: BLE001
                if not getattr(config, "LOCAL_EMBEDDING_FALLBACK", True):
                    raise
                self._embedding_remote_disabled = True
                print(f"[embedding] 远程 embedding 不可用，改用本地兜底向量: {exc}")
        return self._local_embed(texts)

    def _local_embed(self, texts: list[str], dim: int = 256) -> list[list[float]]:
        """本地轻量哈希向量，供不支持 embeddings 的模型服务兜底使用。"""
        return [self._local_embed_one(text, dim) for text in texts]

    def _local_embed_one(self, text: str, dim: int) -> list[float]:
        vec = [0.0] * dim
        cleaned = "".join(ch.lower() for ch in text if not ch.isspace())
        tokens: list[tuple[str, float]] = []

        for ch in cleaned:
            tokens.append((ch, 1.0))
        for i in range(max(0, len(cleaned) - 1)):
            tokens.append((cleaned[i:i + 2], 1.4))
        for i in range(max(0, len(cleaned) - 2)):
            tokens.append((cleaned[i:i + 3], 1.8))

        expansions = {
            "__price__": ("多少钱", "价格", "收费", "套餐", "¥", "元"),
            "__shortcut__": ("快捷键", "ctrl", "shift"),
            "__privacy__": ("隐私", "数据", "训练", "模型", "节点"),
            "__refund__": ("退款", "退还"),
            "__import__": ("导入", "来源", "evernote", "notion", "enex"),
            "__version__": ("最新版本", "版本号", "v4.2", "2026年3月"),
        }
        for topic, markers in expansions.items():
            if any(marker in cleaned for marker in markers):
                tokens.append((topic, 6.0))

        word = ""
        for ch in text.lower():
            if ch.isascii() and ch.isalnum():
                word += ch
            elif word:
                tokens.append((word, 2.0))
                word = ""
        if word:
            tokens.append((word, 2.0))

        for token, weight in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            vec[idx] += weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]
