"""Evaluate the RAG system.

Two families of metrics:

1. Retrieval quality:
   - Hit@k  : fraction of questions where the gold passage is in the top-k.
   - MRR    : mean reciprocal rank of the gold passage.
   We report these both with and without the reranker so you can quantify its lift.

2. Faithfulness (LLM-as-judge): does the generated answer stay grounded in the
   retrieved sources? Reported as the share of answers judged grounded.

Run:
    python eval/evaluate.py            # retrieval only
    python eval/evaluate.py --faithfulness   # also run LLM-as-judge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.generation.llm import get_llm
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever

QA_PATH = Path(__file__).resolve().parent / "qa_set.jsonl"

JUDGE_SYSTEM = (
    "You are a strict grader. Given a QUESTION, the SOURCES shown to an assistant, and "
    "its ANSWER, reply with a single word: GROUNDED if every claim in the answer is "
    "supported by the sources, or UNGROUNDED otherwise."
)


def load_qa() -> list[dict]:
    if not QA_PATH.exists():
        raise RuntimeError("No qa_set.jsonl - run eval/eval_dataset.py first.")
    return [json.loads(line) for line in QA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _retrieved_ids(hits: list[dict]) -> list[str]:
    out = []
    for h in hits:
        m = h["metadata"]
        out.append(f"{m['arxiv_id']}::p{m['page']}::c{m['chunk_on_page']}")
    return out


def retrieval_metrics(use_reranker: bool, k: int = 5) -> dict:
    settings.use_reranker = use_reranker
    retriever = Retriever()
    qa = load_qa()

    hits_at_k = 0
    rr_sum = 0.0
    for ex in qa:
        results = retriever.retrieve(ex["question"], top_n=k)
        ids = _retrieved_ids(results)
        if ex["gold_id"] in ids:
            hits_at_k += 1
            rr_sum += 1.0 / (ids.index(ex["gold_id"]) + 1)

    n = len(qa)
    return {"n": n, f"hit@{k}": hits_at_k / n, "mrr": rr_sum / n}


def faithfulness_score(use_reranker: bool) -> dict:
    settings.use_reranker = use_reranker
    pipeline = RAGPipeline()
    judge = get_llm()
    qa = load_qa()

    grounded = 0
    abstained = 0
    for ex in qa:
        result = pipeline.answer(ex["question"])
        if result.answer.lower().startswith("i cannot answer"):
            abstained += 1
            continue
        sources = "\n\n".join(f"[{s.n}] {s.text}" for s in result.sources)
        verdict = judge.generate(
            JUDGE_SYSTEM,
            f"QUESTION: {ex['question']}\n\nSOURCES:\n{sources}\n\nANSWER: {result.answer}",
        )
        if "GROUNDED" in verdict.upper() and "UNGROUNDED" not in verdict.upper():
            grounded += 1

    n = len(qa)
    answered = n - abstained
    return {
        "n": n,
        "abstained": abstained,
        "faithfulness": grounded / answered if answered else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faithfulness", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print("=== Retrieval metrics ===")
    base = retrieval_metrics(use_reranker=False, k=args.k)
    reranked = retrieval_metrics(use_reranker=True, k=args.k)
    print(f"  Dense only     : hit@{args.k}={base[f'hit@{args.k}']:.3f}  mrr={base['mrr']:.3f}")
    print(f"  + Reranker     : hit@{args.k}={reranked[f'hit@{args.k}']:.3f}  mrr={reranked['mrr']:.3f}")
    lift = reranked["mrr"] - base["mrr"]
    print(f"  Reranker MRR lift: {lift:+.3f}")

    if args.faithfulness:
            print("\n=== Faithfulness (LLM-as-judge) ===")
            base_f = faithfulness_score(use_reranker=False)
            reranked_f = faithfulness_score(use_reranker=True)
            print(f"  Dense only     : faithfulness={base_f['faithfulness']:.3f}  "
                f"(abstained on {base_f['abstained']}/{base_f['n']})")
            print(f"  + Reranker     : faithfulness={reranked_f['faithfulness']:.3f}  "
                f"(abstained on {reranked_f['abstained']}/{reranked_f['n']})")
            f_lift = reranked_f["faithfulness"] - base_f["faithfulness"]
            print(f"  Reranker faithfulness lift: {f_lift:+.3f}")


if __name__ == "__main__":
    main()
