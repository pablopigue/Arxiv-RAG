"""Build the vector index from arXiv papers.

Example:
    uv run scripts/build_index.py --max 30 \
        --query "cat:cs.LG AND abs:retrieval augmented generation"
    uv run scripts/build_index.py --query "cat:cs.CL AND abs:transformer" --max 20 --reset
"""

from __future__ import annotations

# Allow running as `python scripts/build_index.py` from the project root.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from rich.console import Console
from tqdm import tqdm

from src.ingest.arxiv_loader import search_papers
from src.ingest.indexer import index_paper
from src.retrieval.vector_store import count, get_vector_store

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Index arXiv papers for RAG.")
    parser.add_argument("--query", required=True, help="arXiv search query")
    parser.add_argument("--max", type=int, default=25, help="max papers to fetch")
    parser.add_argument("--reset", action="store_true", help="wipe the index first")
    args = parser.parse_args()

    store = get_vector_store()
    if args.reset:
        console.print("[yellow]Resetting collection...[/]")
        store.reset_collection()

    console.print(f"[bold]Searching arXiv:[/] {args.query}")
    papers = search_papers(args.query, max_results=args.max)
    console.print(f"Found [green]{len(papers)}[/] papers.")

    total_chunks = 0
    for paper in tqdm(papers, desc="Indexing papers"):
        try:
            total_chunks += index_paper(paper, store)
        except Exception as exc:  # noqa: BLE001 - keep going on a bad PDF
            console.print(f"[red]Skipped {paper.arxiv_id}: {exc}[/]")

    console.print(
        f"[bold green]Done.[/] Indexed {total_chunks} chunks. "
        f"Collection now holds {count(store)} chunks."
    )


if __name__ == "__main__":
    main()
