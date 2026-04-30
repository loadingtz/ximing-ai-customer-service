"""混合检索：Milvus（bge-zh 稠密向量）+ BM25（中文关键词）+ 元数据过滤 + 同义词扩展。

向量库已升级为生产可换的 Milvus Lite（默认）/ Milvus（生产，URI 切换）。
BM25 仍用 rank_bm25 做"词面"召回，对中文专名（"肉桂""老枞"）抗歧义至关重要。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .vector_store import VectorStore

# 茶叶领域同义词/别名词典——召回时 query 扩展
SYNONYMS: dict[str, list[str]] = {
    "大红袍": ["DHP", "岩茶之王"],
    "肉桂": ["牛肉", "马肉", "桂皮香"],
    "水仙": ["老枞", "老丛", "丛香"],
    "白毫银针": ["银针", "牙茶", "白茶芽头"],
    "白牡丹": ["牡丹"],
    "正山小种": ["桐木小种", "小种"],
    "金骏眉": ["骏眉"],
    "盖碗": ["三才碗"],
    "出汤": ["倒茶", "分茶"],
    "退货": ["退款", "退换", "退回"],
    "发霉": ["霉点", "受潮", "霉变"],
}


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip() and not re.fullmatch(r"\s+", t)]


def _expand_query(query: str) -> str:
    extras: list[str] = []
    for term, syns in SYNONYMS.items():
        if term in query:
            extras.extend(syns)
        else:
            for s in syns:
                if s in query:
                    extras.append(term)
                    break
    return query + (" " + " ".join(extras) if extras else "")


@dataclass
class Node:
    doc_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> str | None:
        return self.meta.get("category")

    @property
    def node_type(self) -> str | None:
        return self.meta.get("node_type")


@dataclass
class Hit:
    node: Node
    score: float
    score_vector: float
    score_bm25: float


class HybridIndex:
    """读取 Milvus 中的全部文本构建 BM25 内存索引；查询时双路并打分融合。"""

    def __init__(self, store: VectorStore | None = None, vector_weight: float = 0.6):
        self.store = store or VectorStore()
        self.vector_weight = vector_weight
        rows = self.store.all_texts()
        self.nodes = [Node(doc_id=did, text=txt, meta=meta) for did, txt, meta in rows]
        self._tokenized = [_tokenize(n.text) for n in self.nodes]
        self._bm25 = BM25Okapi(self._tokenized) if self.nodes else None
        self._id_to_idx = {n.doc_id: i for i, n in enumerate(self.nodes)}

    @classmethod
    def load(cls) -> "HybridIndex":
        return cls()

    @property
    def empty(self) -> bool:
        return not self.nodes

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        category: str | None = None,
        node_types: Iterable[str] | None = None,
    ) -> list[Hit]:
        if self.empty:
            return []
        q = _expand_query(query)

        # 1) 向量召回 (top 50, 远多于 k，方便后融合)
        v_hits = self.store.search(q, k=max(k * 5, 30), category=category, node_types=node_types)
        v_score = {h["doc_id"]: h["score"] for h in v_hits}

        # 2) BM25 召回（全集打分，再按同样的 category/node_type 过滤）
        bm25_arr = self._bm25.get_scores(_tokenize(q))
        node_types_set = set(node_types) if node_types else None

        # 收集候选并打分
        cand_ids: set[str] = set(v_score.keys())
        # 加上 BM25 top 30
        bm25_top = np.argsort(-bm25_arr)[:30]
        for i in bm25_top:
            cand_ids.add(self.nodes[i].doc_id)

        # 归一化
        def _normalize(values: dict) -> dict:
            if not values:
                return {}
            mn = min(values.values()); mx = max(values.values())
            if mx <= mn:
                return {k: 0.0 for k in values}
            return {k: (v - mn) / (mx - mn) for k, v in values.items()}

        bm25_score = {self.nodes[i].doc_id: float(bm25_arr[i]) for i in range(len(self.nodes))}
        v_score_n = _normalize(v_score)
        bm25_score_n = _normalize(bm25_score)

        results: list[Hit] = []
        for did in cand_ids:
            n = self.nodes[self._id_to_idx[did]]
            if category and n.category and n.category not in (category, "通用", ""):
                continue
            if node_types_set and n.node_type and n.node_type not in node_types_set:
                continue
            sv = v_score_n.get(did, 0.0)
            sb = bm25_score_n.get(did, 0.0)
            score = self.vector_weight * sv + (1 - self.vector_weight) * sb
            results.append(Hit(node=n, score=score, score_vector=sv, score_bm25=sb))

        results.sort(key=lambda h: -h.score)
        return results[:k]

    def confidence(self, hits: list[Hit]) -> float:
        if not hits:
            return 0.0
        top = hits[0].score
        gap = top - (hits[1].score if len(hits) > 1 else 0.0)
        return max(0.0, min(1.0, 0.7 * top + 0.3 * (1 - math.exp(-gap * 5))))
