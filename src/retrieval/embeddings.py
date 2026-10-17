"""Local embedding model, exposed through LangChain's `Embeddings` interface.

Anything that takes a LangChain `Embeddings` (vector stores, retrievers, eval
tools) can use this object, and swapping to a hosted model is a one-line change.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        # bge expects an instruction prefix on the *query* only, never on passages.
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": settings.query_instruction,
        },
    )
