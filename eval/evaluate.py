"""Evaluate the RAG system.

Two families of metrics:

1. Retrieval quality:
   - Hit@k  : fraction of questions where the gold passage is in the top-k.
   - MRR    : mean reciprocal rank of the gold passage.
   Reported for dense vs. hybrid (dense + BM25) recall, each with and without the
   reranker, so you can quantify each component's lift.

2. Faithfulness (LLM-as-judge): does the generated answer stay grounded in the
   retrieved sources? Reported as the share of answers judged grounded.

Every metric comes with a 95% bootstrap confidence interval, and every lift with a
*paired* bootstrap interval (same questions, both systems), which tells you whether
a difference is real or just which questions happened to be in the set.

Run:
    uv run eval/evaluate.py                         # retrieval only
    uv run eval/evaluate.py --faithfulness          # also run LLM-as-judge
    uv run eval/evaluate.py --hybrid --qa eval/qa_set_200.jsonl --save eval/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.generation.llm import get_llm
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.retriever import RerankingRetriever
from src.retrieval.vector_store import chunk_id

QA_PATH = Path(__file__).resolve().parent / "qa_set.jsonl"

# name -> RerankingRetriever settings. The first one is the baseline for lifts.
CONFIGS = {
    "dense": {"use_hybrid": False, "use_reranker": False},
    "dense+rerank": {"use_hybrid": False, "use_reranker": True},
    "hybrid": {"use_hybrid": True, "use_reranker": False},
    "hybrid+rerank": {"use_hybrid": True, "use_reranker": True},
}

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict grader. Given a QUESTION, the SOURCES shown to an assistant, "
            "and its ANSWER, decide whether every claim in the answer is supported by the "
            "sources. First check the claims, then give your verdict.",
        ),
        ("human", "QUESTION: {question}\n\nSOURCES:\n{sources}\n\nANSWER: {answer}"),
    ]
)


class Verdict(BaseModel):
    """Whether an answer is fully supported by its sources."""

    # Field order matters: the model writes `reasoning` before it commits to `grounded`.
    reasoning: str = Field(
        description="One or two sentences checking the answer's claims against the sources."
    )
    grounded: bool = Field(
        description="True only if every claim in the answer is supported by the sources."
    )


def load_qa(path: Path = QA_PATH) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"No {path.name} - run eval/eval_dataset.py first.")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ---------------------------------------------------------------- statistics


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean: resample the questions with replacement many times."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0)
    means = np.random.default_rng(seed).choice(v, size=(n_boot, len(v))).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _fmt(values) -> str:
    lo, hi = bootstrap_ci(values)
    return f"{np.mean(values):.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_lift(a, b) -> str:
    """Paired lift b - a. If the interval excludes 0, the difference is significant."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    lo, hi = bootstrap_ci(diff)
    flag = "  *" if lo > 0 or hi < 0 else ""
    return f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{flag}"


# ---------------------------------------------------------------- metrics


def retrieval_metrics(qa: list[dict], k: int = 5, **retriever_kwargs) -> dict:
    """Per-question hit (0/1) and reciprocal rank, so they can be bootstrapped."""
    retriever = RerankingRetriever(top_n=k, **retriever_kwargs)
    hits, rrs = [], []
    for ex in qa:
        ids = [chunk_id(d.metadata) for d in retriever.invoke(ex["question"])]
        found = ex["gold_id"] in ids
        hits.append(int(found))
        rrs.append(1.0 / (ids.index(ex["gold_id"]) + 1) if found else 0.0)
    return {"hit": hits, "rr": rrs}


def faithfulness_score(
    qa: list[dict], max_concurrency: int = 4, tags: list[str] | None = None, **retriever_kwargs
) -> dict:
    pipeline = RAGPipeline(retriever=RerankingRetriever(**retriever_kwargs))
    # The output is a validated Verdict object, so there are no strings to parse.
    judge = (JUDGE_PROMPT | get_llm().with_structured_output(Verdict)).with_config(
        run_name="faithfulness_judge"
    )
    # Every Runnable accepts this config. max_concurrency caps .batch()'s thread pool;
    # tags and metadata show up in LangSmith, so you can filter traces per eval run.
    config = RunnableConfig(
        max_concurrency=max_concurrency, tags=["eval", *(tags or [])], metadata=retriever_kwargs
    )
    start = time.perf_counter()

    # 1) Answer all questions concurrently. Each output is the chain's dict:
    #    {"question", "sources", "context", "answer"}.
    outputs = pipeline.chain.batch([ex["question"] for ex in qa], config=config)

    answered = [o for o in outputs if not o["answer"].strip().lower().startswith("i cannot answer")]

    # 2) Judge all non-abstained answers concurrently.
    verdicts = judge.batch(
        [
            {
                "question": o["question"],
                "sources": "\n\n".join(f"[{s.n}] {s.text}" for s in o["sources"]),
                "answer": o["answer"],
            }
            for o in answered
        ],
        config=config,
        return_exceptions=True,  # one malformed reply shouldn't crash the whole batch
    )

    return {
        "n": len(qa),
        "abstained": len(outputs) - len(answered),
        "judge_failures": sum(isinstance(v, Exception) for v in verdicts),
        "grounded": [int(isinstance(v, Verdict) and v.grounded) for v in verdicts],
        "seconds": time.perf_counter() - start,
    }


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faithfulness", action="store_true")
    parser.add_argument("--hybrid", action="store_true", help="also evaluate dense+BM25 configs")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--qa", type=Path, default=QA_PATH, help="QA set (jsonl)")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel LLM calls in .batch()")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N questions")
    parser.add_argument(
        "--save", type=Path, default=None, help="write all per-question results as JSON"
    )
    args = parser.parse_args()

    qa = load_qa(args.qa)[: args.limit]
    configs = {name: c for name, c in CONFIGS.items() if args.hybrid or not c["use_hybrid"]}
    base = next(iter(configs))
    results: dict = {
        "qa": str(args.qa),
        "n": len(qa),
        "k": args.k,
        "retrieval": {},
        "faithfulness": {},
    }

    print(f"=== Retrieval (n={len(qa)}, mean [95% bootstrap CI]) ===")
    for name, cfg in configs.items():
        r = results["retrieval"][name] = retrieval_metrics(qa, args.k, **cfg)
        print(f"  {name:<14} hit@{args.k} {_fmt(r['hit'])}   mrr {_fmt(r['rr'])}")
    print(f"  Paired lift vs {base} (* = interval excludes 0):")
    for name in list(configs)[1:]:
        a, b = results["retrieval"][base], results["retrieval"][name]
        hit_lift, mrr_lift = _fmt_lift(a["hit"], b["hit"]), _fmt_lift(a["rr"], b["rr"])
        print(f"    {name:<14} hit@{args.k} {hit_lift}   mrr {mrr_lift}")

    if args.faithfulness:
        print(f"\n=== Faithfulness (LLM-as-judge, concurrency={args.concurrency}) ===")
        for name, cfg in configs.items():
            f = results["faithfulness"][name] = faithfulness_score(
                qa, args.concurrency, [name], **cfg
            )
            print(
                f"  {name:<14} faithfulness {_fmt(f['grounded'])}  "
                f"(abstained {f['abstained']}/{f['n']}, judge failures {f['judge_failures']})"
                f"  [{f['seconds']:.0f}s]"
            )

    if args.save:
        args.save.write_text(json.dumps(results, indent=2))
        print(f"\nSaved per-question results to {args.save}")


if __name__ == "__main__":
    main()
