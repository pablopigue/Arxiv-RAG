"""Ask the corrective-RAG agent; it fetches new arXiv papers when the index can't answer.

Examples:
    uv run scripts/agent.py "What is speculative decoding?"
    uv run scripts/agent.py --max-fetches 0 "..."      # never fetch, plain RAG with a grader
    uv run scripts/agent.py --graph                    # print the graph as Mermaid

Note: fetched papers are added to the main index, which changes eval results.
"""

from __future__ import annotations

# Allow running as `python scripts/agent.py` from the project root.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from rich.console import Console
from rich.panel import Panel

from src.agent.graph import build_agent, initial_state

console = Console()


def describe(node: str, changes: dict) -> str:
    """One-line summary of what a graph node just did."""
    match node:
        case "retrieve":
            return f"{len(changes['sources'])} sources"
        case "grade":
            verdict = "sufficient" if changes["sufficient"] else "NOT sufficient"
            return f"{verdict} ({changes['grade_reason']})"
        case "rewrite_query":
            return f"phrases {changes['search_phrases']}"
        case "search_arxiv":
            added = changes["added_papers"] or "nothing new"
            return f"query {changes['search_query']!r} -> indexed {added}"
        case _:
            return "done"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--max-fetches", type=int, default=1)
    parser.add_argument("--papers", type=int, default=3, help="papers to index per fetch")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args()

    agent = build_agent(max_fetches=args.max_fetches, papers_per_fetch=args.papers)
    if args.graph:
        print(agent.get_graph().draw_mermaid())
        return
    if not args.question:
        parser.error("a question is required")

    # stream_mode="updates" yields {node_name: what_that_node_returned} after each step.
    state = initial_state(" ".join(args.question))
    for update in agent.stream(state, stream_mode="updates"):
        for node, changes in update.items():
            state.update(changes)
            console.print(f"[cyan]→ {node}[/] {describe(node, changes)}")

    console.print(Panel(state["answer"], title="Answer", border_style="green"))
    for s in state["sources"]:
        console.print(
            f"  [cyan][{s.n}][/] {s.title} (arXiv:{s.arxiv_id}, p.{s.page}) — {s.score:.3f}"
        )


if __name__ == "__main__":
    main()
