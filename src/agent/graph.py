r"""Corrective RAG agent built with LangGraph.

    START -> retrieve -> grade --(sufficient, or out of fetches)--> generate -> END
                ^            \--(insufficient)--> rewrite_query -> search_arxiv
                |__________________________________________________________|

If the indexed papers can't answer the question, the agent turns the question into
an arXiv search query, downloads and indexes the top results, and retrieves again.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.generation.llm import get_llm
from src.generation.rag_pipeline import (
    NO_ANSWER,
    Source,
    _format_context,
    _to_sources,
    build_generate,
)
from src.ingest.arxiv_loader import search_papers
from src.ingest.indexer import index_paper
from src.retrieval.retriever import RerankingRetriever


class AgentState(TypedDict):
    question: str
    sources: list[Source]
    sufficient: bool
    search_query: str
    fetches: int
    # A reducer: nodes return only the *new* ids and LangGraph appends them
    # (the default for other keys is to overwrite).
    added_papers: Annotated[list[str], operator.add]
    answer: str


class Grade(BaseModel):
    """Whether retrieved sources are enough to answer a question."""

    reasoning: str = Field(description="One sentence on what the sources do or don't cover.")
    sufficient: bool = Field(
        description="True if the sources contain the information needed to answer."
    )


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether the numbered SOURCES contain the information needed to "
            "answer the QUESTION. Be strict: related but non-answering sources are not enough.",
        ),
        ("human", "QUESTION: {question}\n\nSOURCES:\n{context}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Turn the user's question into a short arXiv search query: 3-6 technical "
            "keywords, no boolean operators, no quotes. Output only the query.",
        ),
        ("human", "{question}"),
    ]
)


def build_agent(
    retriever: RerankingRetriever | None = None,
    llm: BaseChatModel | None = None,
    max_fetches: int = 1,
    papers_per_fetch: int = 3,
):
    retriever = retriever or RerankingRetriever()
    llm = llm or get_llm()
    grade_chain = GRADE_PROMPT | llm.with_structured_output(Grade)
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    generate_chain = build_generate(llm)

    # Each node receives the full state and returns only the keys it updates.
    def retrieve(state: AgentState) -> dict:
        return {"sources": _to_sources(retriever.invoke(state["question"]))}

    def grade(state: AgentState) -> dict:
        if not state["sources"]:
            return {"sufficient": False}
        g = grade_chain.invoke(
            {"question": state["question"], "context": _format_context(state["sources"])}
        )
        return {"sufficient": g.sufficient}

    def rewrite_query(state: AgentState) -> dict:
        query = rewrite_chain.invoke({"question": state["question"]})
        return {"search_query": query.strip().strip('"')}

    def search_arxiv(state: AgentState) -> dict:
        added = []
        for paper in search_papers(state["search_query"], max_results=papers_per_fetch):
            try:
                if index_paper(paper, retriever.vector_store):
                    added.append(paper.arxiv_id)
            except Exception:  # noqa: BLE001 - one bad PDF shouldn't stop the agent
                continue
        retriever.refresh()  # BM25 must see the new chunks
        return {"fetches": state["fetches"] + 1, "added_papers": added}

    def generate(state: AgentState) -> dict:
        if not state["sources"]:
            return {"answer": NO_ANSWER}
        answer = generate_chain.invoke(
            {"question": state["question"], "context": _format_context(state["sources"])}
        )
        return {"answer": answer.strip()}

    # A conditional edge: a function that reads the state and names the next node.
    def route_after_grade(state: AgentState) -> Literal["generate", "rewrite_query"]:
        if state["sufficient"] or state["fetches"] >= max_fetches:
            return "generate"
        return "rewrite_query"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("search_arxiv", search_arxiv)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade)
    graph.add_edge("rewrite_query", "search_arxiv")
    graph.add_edge("search_arxiv", "retrieve")  # the cycle
    graph.add_edge("generate", END)
    return graph.compile()


def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "sources": [],
        "sufficient": False,
        "search_query": "",
        "fetches": 0,
        "added_papers": [],
        "answer": "",
    }
