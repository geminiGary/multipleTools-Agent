"""内存向量库：用 numpy 实现余弦相似度检索。

实现要点：
- add(texts, embeddings): 把文本块和对应向量存起来。
- search(query_embedding, top_k): 返回相似度最高的 top_k 个块。
- 用模块级共享实例 STORE，让 ingest.py 入库、knowledge_base.py 检索
  访问的是同一个库。
"""
import numpy as np


class VectorStore:
    def __init__(self):
        self.texts: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadatas: list[dict] = []

    def __len__(self) -> int:
        return len(self.texts)

    def clear(self) -> None:
        """清空内存库，便于服务重启或测试时重新入库。"""
        self.texts.clear()
        self.embeddings.clear()
        self.metadatas.clear()

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict] | None = None) -> None:
        """把若干文本块及其向量加入库中。

        metadatas 为 None 时用空 dict 占位，长度需与 texts 对齐。
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("texts, embeddings, metadatas must have the same length")
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """检索最相似的 top_k 个块。

        返回: [{"text": str, "score": float, "metadata": dict}, ...]，按 score 降序。
        """
        if not self.texts:
            return []

        query_embedding = np.array(query_embedding, dtype=float)
        embeddings = np.array(self.embeddings, dtype=float)

        # 计算余弦相似度
        dot_product = np.dot(embeddings, query_embedding)
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = dot_product / (norms + 1e-8)

        # 获取 top_k 个最相似的索引
        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        # 组装结果
        results = []
        for i in top_k_indices:
            results.append({
                "text": self.texts[i],
                "score": float(similarities[i]),
                "metadata": self.metadatas[i]
            })

        return results


# 共享实例：ingest 入库与 knowledge_base 检索都用它
STORE = VectorStore()
