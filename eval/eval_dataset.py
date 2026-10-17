"""Build a synthetic evaluation set.

Manual gold labels are expensive, so we bootstrap: sample indexed chunks, ask the
local LLM to write a specific question answerable *only* from that chunk, and record
the chunk id as the gold passage.

Output: eval/qa_set.jsonl  ({"question", "gold_id", "gold_arxiv_id"})
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.generation.llm import get_llm
from src.retrieval.vector_store import get_vector_store

OUT_PATH = Path(__file__).resolve().parent / "qa_set.jsonl"

GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write one short, specific question that can be answered ONLY using the "
            "given passage from a machine-learning paper. Do not answer it. Avoid pronouns "
            "like 'this' or 'the paper'. Output just the question.",
        ),
        ("human", "Passage:\n{passage}"),
    ]
)


def build(n: int = 50, seed: int = 0, out: Path = OUT_PATH, max_concurrency: int = 4) -> Path:
    random.seed(seed)
    # Pull the raw collection so we have ids + documents to sample from.
    raw = get_vector_store().get(include=["documents", "metadatas"])
    ids, docs, metas = raw["ids"], raw["documents"], raw["metadatas"]
    if not ids:
        raise RuntimeError("Index is empty - run scripts/build_index.py first.")

    idxs = random.sample(range(len(ids)), k=min(n, len(ids)))
    gen_question = GEN_PROMPT | get_llm() | StrOutputParser()
    questions = gen_question.batch(
        [{"passage": docs[i][:1500]} for i in idxs], config={"max_concurrency": max_concurrency}
    )

    with out.open("w", encoding="utf-8") as f:
        for i, question in zip(idxs, questions, strict=True):
            question = question.strip().strip('"')
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
    print(f"Wrote eval set to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of questions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    build(args.n, args.seed, args.out)
