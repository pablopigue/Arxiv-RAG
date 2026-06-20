"""Split documents into overlapping chunks while preserving source metadata.

A from-scratch recursive splitter so the mechanics are explicit:
it tries to break on the largest natural boundary that fits (paragraph -> line ->
sentence -> word) before falling back to a hard character cut.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import settings

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    metadata: dict


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        # No separator left: hard cut.
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, *rest = seps
    parts = text.split(sep)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        candidate = part if not buf else f"{buf}{sep}{part}"
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if len(part) > size:
                chunks.extend(_split_recursive(part, size, rest))
                buf = ""
            else:
                buf = part
    if buf:
        chunks.append(buf)
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        out.append(f"{tail} {cur}".strip())
    return out


def chunk_pages(
    pages: list[tuple[int, str]],
    base_metadata: dict,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk a list of (page_number, text) tuples into Chunk objects.

    Each chunk carries the source paper metadata plus its page number, which is
    what lets the LLM cite.
    """
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    chunks: list[Chunk] = []
    for page_no, page_text in pages:
        pieces = _add_overlap(_split_recursive(page_text, size, _SEPARATORS), overlap)
        for j, piece in enumerate(pieces):
            piece = piece.strip()
            if len(piece) < 50:  # drop near-empty fragments
                continue
            meta = dict(base_metadata)
            meta.update({"page": page_no, "chunk_on_page": j})
            chunks.append(Chunk(text=piece, metadata=meta))
    return chunks
