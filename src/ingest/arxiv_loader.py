"""Fetch paper metadata and PDFs from the arXiv API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import arxiv

from src.config import settings


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_path: Path | None = None
    categories: list[str] = field(default_factory=list)

    @property
    def short_authors(self) -> str:
        if not self.authors:
            return "Unknown"
        if len(self.authors) <= 2:
            return ", ".join(self.authors)
        return f"{self.authors[0]} et al."


def search_papers(query: str, max_results: int = 25) -> list[Paper]:
    """Search arXiv and return metadata.

    Example query: 'cat:cs.LG AND abs:retrieval augmented generation'
    """
    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: list[Paper] = []
    for result in client.results(search):
        papers.append(
            Paper(
                arxiv_id=result.get_short_id(),
                title=result.title.strip().replace("\n", " "),
                authors=[a.name for a in result.authors],
                abstract=result.summary.strip().replace("\n", " "),
                published=result.published.date().isoformat(),
                categories=list(result.categories),
            )
        )
    return papers


def download_pdf(paper: Paper, raw_dir: Path | None = None) -> Paper:
    """Download a paper's PDF and attach the local path to the Paper object."""
    raw_dir = raw_dir or settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_id = paper.arxiv_id.replace("/", "_")
    dest = raw_dir / f"{safe_id}.pdf"

    if not dest.exists():
        client = arxiv.Client()
        result = next(client.results(arxiv.Search(id_list=[paper.arxiv_id])))
        result.download_pdf(dirpath=str(raw_dir), filename=dest.name)

    paper.pdf_path = dest
    return paper
