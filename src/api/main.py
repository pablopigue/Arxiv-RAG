"""FastAPI service exposing the RAG pipeline.

Run: uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.vector_store import count, get_vector_store

app = FastAPI(title="arXiv RAG", version="1.0.0")


@lru_cache(maxsize=1)
def _pipeline() -> RAGPipeline:
    return RAGPipeline()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What problem does RAG solve?"])


class SourceOut(BaseModel):
    n: int
    title: str
    arxiv_id: str
    page: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": count(get_vector_store())}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = _pipeline().answer(req.question)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceOut(
                n=s.n,
                title=s.title,
                arxiv_id=s.arxiv_id,
                page=s.page,
                score=round(s.score, 4),
                snippet=s.text[:300] + ("…" if len(s.text) > 300 else ""),
            )
            for s in result.sources
        ],
    )
