"""Tests for the chunker."""

from __future__ import annotations

from langchain_core.documents import Document

from src.ingest.chunker import chunk_documents

BASE = {"arxiv_id": "1234.5678", "title": "Test", "authors": "X", "published": "2024-01-01"}


def _page(n: int, text: str) -> Document:
    return Document(page_content=text, metadata={**BASE, "page": n})


def test_chunks_respect_max_size():
    chunks = chunk_documents([_page(1, "a" * 5000)], size=1000, overlap=0)
    assert all(len(c.page_content) <= 1000 for c in chunks)
    assert "".join(c.page_content for c in chunks) == "a" * 5000


def test_split_prefers_paragraph_boundaries():
    text = ("Para one is long enough to keep around. " * 2).strip()
    pages = [_page(1, f"{text}\n\n{text.replace('one', 'two')}")]
    chunks = chunk_documents(pages, size=len(text) + 5, overlap=0)
    assert chunks[0].page_content == text


def test_chunks_carry_metadata():
    pages = [_page(1, "Sentence one. " * 200), _page(2, "Other content. " * 200)]
    chunks = chunk_documents(pages, size=500, overlap=50)
    assert chunks
    for c in chunks:
        assert c.metadata["arxiv_id"] == "1234.5678"
        assert c.metadata["page"] in (1, 2)
        assert "chunk_on_page" in c.metadata


def test_tiny_fragments_dropped():
    assert chunk_documents([_page(1, "short")]) == []
