"""向量库（Milvus Lite）+ 中文 embedding（bge）封装。

设计：
- 默认 Milvus Lite（文件存储，URI 类似 ./data/vector.db），无 Docker。
- 切生产：把 MILVUS_URI 改成 "http://milvus-host:19530"，代码不动。
- Embedding 默认 BAAI/bge-small-zh-v1.5（95MB），可改 large（1.3GB，更准）。
- 按 BGE 推荐：query 端自动加 "为这个句子生成表示以用于检索相关文章：" 前缀。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymilvus import DataType, MilvusClient

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "vector.db"

DEFAULT_EMBED_MODEL = os.getenv("XIMING_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
DEFAULT_COLLECTION = os.getenv("XIMING_MILVUS_COLLECTION", "ximing_kb")
DEFAULT_URI = os.getenv("MILVUS_URI", str(DEFAULT_DB))
EMBED_DIM = {"BAAI/bge-small-zh-v1.5": 512, "BAAI/bge-base-zh-v1.5": 768, "BAAI/bge-large-zh-v1.5": 1024}

# bge-zh 系列 query 检索前缀（论文推荐，提升 1-2 个点）
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


@dataclass
class Embedder:
    """懒加载 sentence-transformers 模型。"""
    model_name: str = DEFAULT_EMBED_MODEL
    _model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        return EMBED_DIM.get(self.model_name, 512)

    def encode_docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self._ensure()
        return self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)

    def encode_query(self, query: str) -> np.ndarray:
        self._ensure()
        return self._model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False)[0]


class VectorStore:
    """Milvus Lite 客户端封装。schema：doc_id (PK) / vector / text / 元数据。"""

    def __init__(self, uri: str = DEFAULT_URI, collection: str = DEFAULT_COLLECTION, embedder: Embedder | None = None):
        Path(uri).parent.mkdir(parents=True, exist_ok=True) if not uri.startswith("http") else None
        self.client = MilvusClient(uri=uri)
        self.collection = collection
        self.embedder = embedder or Embedder()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="doc_id",     datatype=DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field(field_name="vector",     datatype=DataType.FLOAT_VECTOR, dim=self.embedder.dim)
        schema.add_field(field_name="text",       datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name="category",   datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="node_type",  datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="trust_level", datatype=DataType.INT8)
        schema.add_field(field_name="source_url", datatype=DataType.VARCHAR, max_length=512)

        index = self.client.prepare_index_params()
        index.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")
        self.client.create_collection(collection_name=self.collection, schema=schema, index_params=index)

    # ------------------------------------------------------------------
    def upsert(self, nodes: Iterable[dict], batch_size: int = 64) -> int:
        self.ensure_collection()
        nodes = list(nodes)
        n = 0
        for i in range(0, len(nodes), batch_size):
            chunk = nodes[i : i + batch_size]
            texts = [n.get("text", "") for n in chunk]
            vecs = self.embedder.encode_docs(texts).tolist()
            rows = []
            for node, vec in zip(chunk, vecs):
                rows.append({
                    "doc_id":     str(node.get("doc_id"))[:64],
                    "vector":     vec,
                    "text":       (node.get("text") or "")[:8000],
                    "category":   (node.get("category") or "")[:32],
                    "node_type":  (node.get("node_type") or "")[:32],
                    "trust_level": int(node.get("trust_level") or 3),
                    "source_url": (node.get("source_url") or "")[:512],
                })
            self.client.upsert(collection_name=self.collection, data=rows)
            n += len(rows)
        # Milvus Lite 写完后无显式 flush；查询前 load 一次
        self.client.load_collection(self.collection)
        return n

    def search(
        self,
        query: str,
        *,
        k: int = 20,
        category: str | None = None,
        node_types: Iterable[str] | None = None,
    ) -> list[dict]:
        self.ensure_collection()
        if self.count() == 0:
            return []
        qv = self.embedder.encode_query(query).tolist()
        # 元数据过滤——Milvus 标量字段 boolean expression
        clauses: list[str] = []
        if category:
            clauses.append(f'(category == "{category}" or category == "通用" or category == "")')
        if node_types:
            quoted = ",".join(f'"{t}"' for t in node_types)
            clauses.append(f"node_type in [{quoted}]")
        expr = " and ".join(clauses) if clauses else None

        results = self.client.search(
            collection_name=self.collection,
            data=[qv],
            limit=k,
            output_fields=["doc_id", "text", "category", "node_type", "trust_level", "source_url"],
            filter=expr,
        )
        return [
            {
                "doc_id": h["entity"]["doc_id"],
                "text": h["entity"]["text"],
                "category": h["entity"]["category"],
                "node_type": h["entity"]["node_type"],
                "trust_level": h["entity"]["trust_level"],
                "source_url": h["entity"]["source_url"],
                "score": float(h["distance"]),
            }
            for h in (results[0] if results else [])
        ]

    def count(self) -> int:
        if not self.client.has_collection(self.collection):
            return 0
        try:
            stats = self.client.get_collection_stats(self.collection)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def all_texts(self) -> list[tuple[str, str, dict]]:
        """供 BM25 离线构建：返回 [(doc_id, text, meta), ...]。"""
        self.ensure_collection()
        if self.count() == 0:
            return []
        out: list[tuple[str, str, dict]] = []
        cursor = ""
        while True:
            page = self.client.query(
                collection_name=self.collection,
                filter="",
                output_fields=["doc_id", "text", "category", "node_type", "trust_level", "source_url"],
                limit=1000,
                offset=len(out),
            )
            if not page:
                break
            for r in page:
                meta = {k: r[k] for k in ("category", "node_type", "trust_level", "source_url") if k in r}
                meta["doc_id"] = r["doc_id"]
                out.append((r["doc_id"], r["text"], meta))
            if len(page) < 1000:
                break
        return out
