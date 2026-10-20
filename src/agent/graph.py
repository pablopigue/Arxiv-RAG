r"""Corrective RAG agent built with LangGraph.

    START -> retrieve -> grade --(sufficient, or out of fetches)--> generate -> END
                ^            \--(insufficient)--> rewrite_query -> search_arxiv
                |__________________________________________________________|

If the indexed papers can't answer the question, the agent turns the question into
arXiv key phrases, downloads and indexes the top results, and retrieves again.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.config import settings
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
    grade_reason: str  # why grade decided, shown in the CLI
    search_phrases: list[str]
    search_query: str  # the arXiv query that actually returned papers
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


class SearchPhrases(BaseModel):
    """Key phrases for an arXiv search."""

    phrases: list[str] = Field(
        description="1-3 short technical key phrases (1-3 words each) that a relevant paper's "
        "abstract would contain, most specific first, e.g. ['FlashAttention', 'GPU memory'].",
        min_length=1,
        max_length=3,
    )


# "Be strict" made Llama 3.1 8B reject 7/13 answerable cases (fetching papers for nothing);
# this wording rejected 4/13 while still catching 4/6 on-topic-but-unanswerable ones.
GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether the numbered SOURCES contain enough information to answer "
            "the main point of the QUESTION. Sources that only mention the topic in passing "
            "(for example in a reference list or a related-work sentence) are not enough.",
        ),
        ("human", "QUESTION: {question}\n\nSOURCES:\n{context}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract the key technical phrases needed to find arXiv papers that answer the "
            "user's question. Use the exact names of methods, models or concepts.",
        ),
        ("human", "{question}"),
    ]
)


def arxiv_queries(phrases: list[str]) -> list[str]:
    """Candidate arXiv queries, from most to least specific.

    Unquoted keywords match each word anywhere ("flash" finds flash-memory physics), so
    the phrases are quoted, searched in abstracts, and limited to computer science.
    """
    quoted = [f'abs:"{p}"' for p in phrases]
    queries = [" AND ".join([*quoted, "cat:cs.*"])]
    if len(phrases) > 1:
        queries.append(f"{quoted[0]} AND cat:cs.*")
    queries.append(" ".join(phrases))  # last resort: plain keywords
    return queries


def build_agent(
    retriever: RerankingRetriever | None = None,
    llm: BaseChatModel | None = None,
    max_fetches: int = 1,
    papers_per_fetch: int = 3,
):
    retriever = retriever or RerankingRetriever()
    llm = llm or get_llm()
    grade_chain = GRADE_PROMPT | llm.with_structured_output(Grade)
    rewrite_chain = REWRITE_PROMPT | llm.with_structured_output(SearchPhrases)
    generate_chain = build_generate(llm)

    # Each node receives the full state and returns only the keys it updates.
    def retrieve(state: AgentState) -> dict:
        return {"sources": _to_sources(retriever.invoke(state["question"]))}

    def grade(state: AgentState) -> dict:
        if not state["sources"]:
            return {"sufficient": False, "grade_reason": "no sources"}
        # Cheap deterministic check first: the cross-encoder says whether the best chunk is
        # even on topic. It can't tell whether an on-topic chunk *answers* the question
        # (a passing mention scores high too), so above the threshold the LLM decides.
        top = state["sources"][0].score
        if retriever.use_reranker and top < settings.agent_min_rerank_score:
            return {
                "sufficient": False,
                "grade_reason": f"top rerank {top:.2f} < {settings.agent_min_rerank_score}",
            }
        g = grade_chain.invoke(
            {"question": state["question"], "context": _format_context(state["sources"])}
        )
        return {"sufficient": g.sufficient, "grade_reason": f"LLM, top rerank {top:.2f}"}

    def rewrite_query(state: AgentState) -> dict:
        result = rewrite_chain.invoke({"question": state["question"]})
        return {"search_phrases": [p.strip().strip('"') for p in result.phrases]}

    def search_arxiv(state: AgentState) -> dict:
        query, papers = "", []
        for query in arxiv_queries(state["search_phrases"]):
            papers = search_papers(query, max_results=papers_per_fetch)
            if papers:
                break
        added = []
        for paper in papers:
            try:
                if index_paper(paper, retriever.vector_store):
                    added.append(paper.arxiv_id)
            except Exception:  # noqa: BLE001 - one bad PDF shouldn't stop the agent
                continue
        retriever.refresh()  # BM25 must see the new chunks
        return {"search_query": query, "fetches": state["fetches"] + 1, "added_papers": added}

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
        "grade_reason": "",
        "search_phrases": [],
        "search_query": "",
        "fetches": 0,
        "added_papers": [],
        "answer": "",
    }
