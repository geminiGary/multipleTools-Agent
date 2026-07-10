"""文档切块 + 嵌入入库。

把 data/docs 下的文本文件读入，切成小块，调用 embedding API 得到向量，
写入 rag.vector_store.STORE。
"""
from rag.vector_store import STORE
from llm import LLMClient
from pathlib import Path

_llm = LLMClient()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """把长文本切成带重叠的小块，返回非空块列表。"""
    if (chunk_size <= 0) or (overlap < 0) or (overlap >= chunk_size):
        raise ValueError("chunk_size must be > 0, overlap must be >= 0 and < chunk_size")
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return [chunk for chunk in chunks if chunk.strip()]  # 过滤空块


def ingest_file(path: str) -> int:
    """读取单个文本文件，切块、嵌入、入库，返回入库的块数。

    元数据中保留来源路径，便于知识库工具在回答时展示引用来源。
    """
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = _llm.embed(chunks)
    metadatas = [{"doc": path} for _ in chunks]
    STORE.add(chunks, embeddings, metadatas)
    return len(chunks)


def ingest_dir(dir_path: str = "data/docs") -> int:
    """把目录下所有 .md/.txt 文件入库，返回总块数。"""
    total_chunks = 0
    root = Path(dir_path)
    if not root.exists() or not root.is_dir():
        return 0
    STORE.clear()
    for filename in root.iterdir():
        if filename.is_file() and filename.suffix.lower() in (".md", ".txt"):
            total_chunks += ingest_file(filename)
    return total_chunks
