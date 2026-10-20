"""BM25 keyword search over the chunks already stored in Chroma.

Dense embeddings capture meaning but can miss exact tokens (model names, acronyms,
numbers like "ms-marco-MiniLM-L-6"). BM25 scores literal term overlap, so the two
fail on different queries, which is why fusing them usually helps.
"""

from __future__ import annotations

import re

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """In-memory BM25 index built from every chunk in the vector store."""

    def __init__(self, store: Chroma) -> None:
        raw = store.get(include=["documents", "metadatas"])
        self._texts: list[str] = raw["documents"]
        self._metas: list[dict] = raw["metadatas"]
        self._bm25 = BM25Okapi([_tokenize(t) for t in self._texts]) if self._texts else None

    def search(self, query: str, k: int) -> list[Document]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top = np.argsort(-scores)[:k]
        return [
            Document(
                page_content=self._texts[i],
                metadata={**self._metas[i], "bm25_score": float(scores[i])},
            )
            for i in top
        ]
