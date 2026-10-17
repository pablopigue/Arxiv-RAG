"""Split page Documents into overlapping chunks while preserving source metadata.

Uses LangChain's RecursiveCharacterTextSplitter: it tries to break on the largest
natural boundary that fits (paragraph -> line -> sentence -> word) before falling
back to a hard character cut.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
MIN_CHUNK_CHARS = 50  # drop near-empty fragments


def chunk_documents(
    pages: list[Document],
    size: int | None = None,
    overlap: int | None = None,
) -> list[Document]:
    """Chunk one Document per page into chunk Documents.

    Each chunk carries the source paper metadata plus its page number, which is
    what lets the LLM cite.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size or settings.chunk_size,
        chunk_overlap=overlap if overlap is not None else settings.chunk_overlap,
        separators=SEPARATORS,
    )

    chunks: list[Document] = []
    for page in pages:
        for j, piece in enumerate(splitter.split_text(page.page_content)):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Document(page_content=piece, metadata={**page.metadata, "chunk_on_page": j})
            )
    return chunks
