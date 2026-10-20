"""Download, parse, chunk and store one arXiv paper. Shared by build_index.py and the agent."""

from __future__ import annotations

from langchain_chroma import Chroma

from src.ingest.arxiv_loader import Paper, download_pdf
from src.ingest.chunker import chunk_documents
from src.ingest.pdf_parser import load_pages
from src.retrieval.vector_store import add_documents


def is_indexed(store: Chroma, arxiv_id: str) -> bool:
    return bool(store.get(where={"arxiv_id": arxiv_id}, limit=1, include=[])["ids"])


def index_paper(paper: Paper, store: Chroma) -> int:
    """Returns the number of chunks added (0 if the paper was already indexed)."""
    if is_indexed(store, paper.arxiv_id):
        return 0
    paper = download_pdf(paper)
    base_meta = {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.short_authors,
        "published": paper.published,
    }
    pages = load_pages(paper.pdf_path, base_meta)
    return add_documents(store, chunk_documents(pages))
