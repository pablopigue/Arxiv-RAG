"""Tests for the chunker."""
from __future__ import annotations

from src.ingest.chunker import _split_recursive, chunk_pages


def test_split_respects_max_size():
    text = "a" * 5000
    pieces = _split_recursive(text, size=1000, seps=["\n\n", "\n", ". ", " "])
    assert all(len(p) <= 1000 for p in pieces)
    assert "".join(pieces) == text


def test_split_prefers_paragraph_boundaries():
    text = "Para one.\n\nPara two.\n\nPara three."
    pieces = _split_recursive(text, size=15, seps=["\n\n", "\n", ". ", " "])
    assert "Para one." in pieces


def test_chunk_pages_carries_metadata():
    pages = [(1, "Sentence one. " * 200), (2, "Other content. " * 200)]
    base = {"arxiv_id": "1234.5678", "title": "Test", "authors": "X", "published": "2024-01-01"}
    chunks = chunk_pages(pages, base, size=500, overlap=50)
    assert chunks
    for c in chunks:
        assert c.metadata["arxiv_id"] == "1234.5678"
        assert c.metadata["page"] in (1, 2)
        assert "chunk_on_page" in c.metadata


def test_tiny_fragments_dropped():
    pages = [(1, "short")]
    base = {"arxiv_id": "1", "title": "t", "authors": "a", "published": "2024-01-01"}
    assert chunk_pages(pages, base) == []
