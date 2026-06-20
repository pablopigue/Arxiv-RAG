"""Thin wrapper around a persistent ChromaDB collection.

Compute embeddings (see embeddings.py) and pass them in, rather than
letting Chroma pick a default model, so the same model is guaranteed at index and
query time.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.ingest.chunker import Chunk
from src.retrieval.embeddings import embed_passages, embed_query


class VectorStore:
    def __init__(self) -> None:
        settings.ensure_dirs()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # write
    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = embed_passages(texts)
        ids = [
            f"{c.metadata['arxiv_id']}::p{c.metadata['page']}::c{c.metadata['chunk_on_page']}"
            for c in chunks
        ]
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    # read
    def query(self, text: str, k: int | None = None) -> list[dict]:
        k = k or settings.top_k
        res = self._collection.query(
            query_embeddings=[embed_query(text)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({"text": doc, "metadata": meta, "score": 1.0 - dist})
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name, metadata={"hnsw:space": "cosine"}
        )
