"""Two-stage retrieval as a LangChain `BaseRetriever`: recall, then reranking.

Stage 1 pulls top_k candidates: dense (bi-encoder) search, optionally fused with
BM25 keyword search (hybrid). Stage 2 (cross-encoder) is slower but far more
accurate because it reads the query and passage together; it re-scores the
candidates and we keep top_n.

Subclassing `BaseRetriever` makes this a Runnable: it supports `.invoke()`,
`.batch()`, composition with `|` and tracing, like any built-in retriever.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.vector_store import chunk_id, get_vector_store


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int) -> list[Document]:
    """Merge ranked lists: each doc scores sum(1 / (k + rank)) over the lists it appears in.

    RRF only uses ranks, so it can combine scores on incompatible scales
    (cosine similarity vs. BM25) without any normalisation.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            cid = chunk_id(doc.metadata)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            docs.setdefault(cid, doc)
    fused = sorted(docs, key=scores.__getitem__, reverse=True)
    for cid in fused:
        docs[cid].metadata["rrf_score"] = scores[cid]
    return [docs[cid] for cid in fused]


class RerankingRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: Chroma = Field(default_factory=get_vector_store)
    # Read at construction time, so eval can flip settings between runs.
    top_k: int = Field(default_factory=lambda: settings.top_k)
    top_n: int = Field(default_factory=lambda: settings.top_n)
    use_reranker: bool = Field(default_factory=lambda: settings.use_reranker)
    use_hybrid: bool = Field(default_factory=lambda: settings.use_hybrid)

    # Private state: built lazily on the first hybrid query, not part of the config.
    _bm25: BM25Index | None = PrivateAttr(default=None)

    def refresh(self) -> None:
        """Call after adding documents to the store so BM25 sees them."""
        self._bm25 = None

    def _dense(self, query: str) -> list[Document]:
        hits = self.vector_store.similarity_search_with_relevance_scores(query, k=self.top_k)
        docs = []
        for doc, score in hits:
            doc.metadata["score"] = score
            docs.append(doc)
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs = self._dense(query)  # stage 1
        if self.use_hybrid:
            if self._bm25 is None:
                self._bm25 = BM25Index(self.vector_store)
            keyword = self._bm25.search(query, k=self.top_k)
            docs = reciprocal_rank_fusion([docs, keyword], k=settings.rrf_k)[: self.top_k]

        if self.use_reranker and docs:  # stage 2
            scores = _reranker().predict([(query, d.page_content) for d in docs])
            for d, s in zip(docs, scores, strict=True):
                d.metadata["rerank_score"] = float(s)
            docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)

        return docs[: self.top_n]
