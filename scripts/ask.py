"""Ask a question from the terminal.

Example:
    python scripts/ask.py "What is the difference between a bi-encoder and a cross-encoder?"
"""
from __future__ import annotations

# Allow running as `python scripts/ask.py` from the project root.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from src.generation.rag_pipeline import RAGPipeline

console = Console()


def main() -> None:
    if len(sys.argv) < 2:
        console.print("[red]Usage:[/] python scripts/ask.py \"<your question>\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = RAGPipeline().answer(question)

    console.print(Panel(result.answer, title="Answer", border_style="green"))
    console.print("\n[bold]Sources[/]")
    for s in result.sources:
        console.print(f"  [cyan][{s.n}][/] {s.title} (arXiv:{s.arxiv_id}, p.{s.page}) — {s.score:.3f}")


if __name__ == "__main__":
    main()
