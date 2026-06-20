"""Extract clean text from a PDF, page by page, using fitz."""
from __future__ import annotations

import re
from pathlib import Path

import fitz


def _clean(text: str) -> str:
    # Collapse hyphenation at line breaks ("represen-\ntation" -> "representation")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text). Page numbers are 1-indexed."""
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))
            if text:
                pages.append((i, text))
    return pages


def extract_text(pdf_path: Path) -> str:
    """Convenience: the whole document as one string."""
    return "\n\n".join(text for _, text in extract_pages(pdf_path))
