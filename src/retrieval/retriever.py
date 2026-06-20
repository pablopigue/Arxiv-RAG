"""Two-stage retrieval: dense recall, then cross-encoder reranking.

Stage 1 (bi-encoder) is fast and pulls top_k candidates. Stage 2 (cross-encoder)
is slower but far more accurate because it reads the query and passage together;
it re-scores the candidates and we keep top_n.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.vector_store import VectorStore


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


class Retriever:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore()

    def retrieve(self, query: str, top_k: int | None = None, top_n: int | None = None) -> list[dict]:
        top_k = top_k or settings.top_k
        top_n = top_n or settings.top_n

        candidates = self.store.query(query, k=top_k)
        if not candidates:
            return []

        if settings.use_reranker:
            pairs = [(query, c["text"]) for c in candidates]
            scores = _reranker().predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        return candidates[:top_n]
