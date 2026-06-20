"""The RAG pipeline: retrieve -> build a grounded prompt -> generate with citations.

Two design decisions reduce hallucination and make the system demoable:
  1. The system prompt forbids using outside knowledge and requires inline [n]
     citations that map to the numbered sources.
  2. If the answer isn't in the retrieved context, the model is told to say so.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.generation.llm import LLM, get_llm
from src.retrieval.retriever import Retriever

SYSTEM_PROMPT = (
    "You are a precise research assistant answering questions about machine-learning "
    "papers. Use ONLY the numbered sources provided. Cite every claim with inline "
    "markers like [1] or [2][3] that refer to those sources. If the sources do not "
    "contain the answer, reply exactly: 'I cannot answer this from the provided papers.' "
    "Be concise and technical."
)


@dataclass
class Source:
    n: int
    title: str
    arxiv_id: str
    page: int
    text: str
    score: float


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[Source]


def _format_context(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        blocks.append(f"[{s.n}] {s.title} (arXiv:{s.arxiv_id}, p.{s.page})\n{s.text}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm()

    def answer(self, question: str) -> RAGAnswer:
        hits = self.retriever.retrieve(question)
        if not hits:
            return RAGAnswer(question, "I cannot answer this from the provided papers.", [])

        sources = [
            Source(
                n=i + 1,
                title=h["metadata"].get("title", "Unknown"),
                arxiv_id=h["metadata"].get("arxiv_id", "?"),
                page=h["metadata"].get("page", 0),
                text=h["text"],
                score=h.get("rerank_score", h.get("score", 0.0)),
            )
            for i, h in enumerate(hits)
        ]

        user_prompt = (
            f"Question: {question}\n\n"
            f"Sources:\n{_format_context(sources)}\n\n"
            "Answer using only the sources above, with inline [n] citations."
        )
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        return RAGAnswer(question=question, answer=answer, sources=sources)
