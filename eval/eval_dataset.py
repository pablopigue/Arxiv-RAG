"""Build a synthetic evaluation set.

Manual gold labels are expensive, so we bootstrap: sample indexed chunks, ask the
local LLM to write a specific question answerable *only* from that chunk, and record
the chunk id as the gold passage.

Output: eval/qa_set.jsonl  ({"question", "gold_id", "gold_arxiv_id"})
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.generation.llm import get_llm
from src.retrieval.vector_store import VectorStore

OUT_PATH = Path(__file__).resolve().parent / "qa_set.jsonl"

GEN_SYSTEM = (
    "You write one short, specific question that can be answered ONLY using the given "
    "passage from a machine-learning paper. Do not answer it. Avoid pronouns like 'this' "
    "or 'the paper'. Output just the question."
)


def build(n: int = 50, seed: int = 0) -> Path:
    random.seed(seed)
    store = VectorStore()
    # Pull the raw collection so we have ids + documents to sample from.
    raw = store._collection.get(include=["documents", "metadatas"])
    ids, docs, metas = raw["ids"], raw["documents"], raw["metadatas"]
    if not ids:
        raise RuntimeError("Index is empty - run scripts/build_index.py first.")

    idxs = random.sample(range(len(ids)), k=min(n, len(ids)))
    llm = get_llm()

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i in idxs:
            passage = docs[i][:1500]
            question = llm.generate(GEN_SYSTEM, f"Passage:\n{passage}").strip().strip('"')
            if len(question) < 10:
                continue
            f.write(
                json.dumps(
                    {
                        "question": question,
                        "gold_id": ids[i],
                        "gold_arxiv_id": metas[i].get("arxiv_id"),
                    }
                )
                + "\n"
            )
    print(f"Wrote eval set to {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    build()
