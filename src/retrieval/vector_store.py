"""Persistent ChromaDB collection via LangChain's `Chroma` vector store.

The embedding function is attached to the store, so the same model is guaranteed
at index and query time.
"""

from __future__ import annotations

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.retrieval.embeddings import get_embeddings


def get_vector_store() -> Chroma:
    settings.ensure_dirs()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )


def chunk_id(metadata: dict) -> str:
    """Stable id for a chunk; the eval set uses it as the gold label."""
    return f"{metadata['arxiv_id']}::p{metadata['page']}::c{metadata['chunk_on_page']}"


def add_documents(store: Chroma, docs: list[Document]) -> int:
    if not docs:
        return 0
    store.add_documents(docs, ids=[chunk_id(d.metadata) for d in docs])
    return len(docs)


def count(store: Chroma) -> int:
    return len(store.get(include=[])["ids"])
