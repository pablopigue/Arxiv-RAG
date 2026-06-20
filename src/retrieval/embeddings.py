"""Local embedding model.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    vecs = _model().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    prefixed = settings.query_instruction + text
    vec = _model().encode([prefixed], normalize_embeddings=True)[0]
    return vec.tolist()
