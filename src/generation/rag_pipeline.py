"""The RAG pipeline as an LCEL chain: retrieve -> grounded prompt -> generate with citations.

Two design decisions reduce hallucination and make the system demoable:
  1. The system prompt forbids using outside knowledge and requires inline [n]
     citations that map to the numbered sources.
  2. If the answer isn't in the retrieved context, the model is told to say so.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableGenerator,
    RunnablePassthrough,
)

from src.generation.llm import get_llm
from src.retrieval.retriever import RerankingRetriever

NO_ANSWER = "I cannot answer this from the provided papers."

SYSTEM_PROMPT = (
    "You are a precise research assistant answering questions about machine-learning "
    "papers. Use ONLY the numbered sources provided. Cite every claim with inline "
    "markers like [1] or [2][3] that refer to those sources. If the sources do not "
    f"contain the answer, reply exactly: '{NO_ANSWER}' "
    "Be concise and technical."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Question: {question}\n\n"
            "Sources:\n{context}\n\n"
            "Answer using only the sources above, with inline [n] citations.",
        ),
    ]
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


def _to_sources(docs: list[Document]) -> list[Source]:
    return [
        Source(
            n=i + 1,
            title=d.metadata.get("title", "Unknown"),
            arxiv_id=d.metadata.get("arxiv_id", "?"),
            page=d.metadata.get("page", 0),
            text=d.page_content,
            # Best score available: reranker > hybrid fusion > dense similarity.
            score=next(
                (d.metadata[k] for k in ("rerank_score", "rrf_score", "score") if k in d.metadata),
                0.0,
            ),
        )
        for i, d in enumerate(docs)
    ]


def _format_context(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[{s.n}] {s.title} (arXiv:{s.arxiv_id}, p.{s.page})\n{s.text}" for s in sources
    )


def _lstrip_stream(chunks: Iterator[str]) -> Iterator[str]:
    """Drop leading whitespace while still streaming token by token.

    A plain `lambda s: s.strip()` would have to wait for the full answer, which
    silently turns `.stream()` into one big chunk at the end.
    """
    started = False
    for chunk in chunks:
        if not started:
            chunk = chunk.lstrip()
            started = bool(chunk)
        if chunk:
            yield chunk


def build_generate(llm: BaseChatModel) -> Runnable:
    """{"question", "context"} -> answer text (streamable)."""
    return PROMPT | llm | StrOutputParser() | RunnableGenerator(_lstrip_stream)


def build_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    """question (str) -> {"question", "sources", "context", "answer"}.

    `.stream()` yields partial dicts: first {"question"}, {"sources"}, {"context"},
    then many {"answer": "<token>"} chunks.
    """
    return (
        {"question": RunnablePassthrough(), "sources": retriever | _to_sources}
        | RunnablePassthrough.assign(context=lambda x: _format_context(x["sources"]))
        | RunnablePassthrough.assign(
            # Skip the LLM call entirely when retrieval came back empty.
            answer=RunnableBranch(
                (lambda x: not x["sources"], lambda _: NO_ANSWER), build_generate(llm)
            )
        )
    ).with_config(run_name="rag_chain")  # the name shown in LangSmith traces


class RAGPipeline:
    def __init__(
        self, retriever: BaseRetriever | None = None, llm: BaseChatModel | None = None
    ) -> None:
        self.chain = build_chain(retriever or RerankingRetriever(), llm or get_llm())

    def answer(self, question: str) -> RAGAnswer:
        out = self.chain.invoke(question)
        return RAGAnswer(question=question, answer=out["answer"].strip(), sources=out["sources"])
