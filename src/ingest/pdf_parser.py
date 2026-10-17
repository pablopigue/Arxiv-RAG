"""Extract clean text from a PDF, one LangChain Document per page, using PyMuPDF."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def _clean(text: str) -> str:
    # Collapse hyphenation at line breaks ("represen-\ntation" -> "representation")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pages(pdf_path: Path, metadata: dict | None = None) -> list[Document]:
    """Return one Document per non-empty page. Page numbers are 1-indexed."""
    pages: list[Document] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))
            if text:
                pages.append(Document(page_content=text, metadata={**(metadata or {}), "page": i}))
    return pages
