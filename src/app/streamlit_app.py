"""Minimal Streamlit demo.

Run: streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.vector_store import count, get_vector_store


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


st.set_page_config(page_title="arXiv RAG", page_icon="📚")
st.title("📚 Ask the ML papers")
st.caption("Retrieval-augmented Q&A over an arXiv corpus — answers cite their sources.")

n_chunks = count(get_vector_store())
if n_chunks == 0:
    st.warning("The index is empty. Run `uv run scripts/build_index.py` first.")
    st.stop()

st.sidebar.metric("Indexed chunks", n_chunks)

question = st.text_input(
    "Your question", placeholder="e.g. How does a cross-encoder reranker work?"
)
if st.button("Ask", type="primary") and question:
    sources = []

    def answer_tokens():
        # chain.stream() yields partial dicts; "sources" arrives once, before the
        # LLM starts, and "answer" arrives as many small token chunks.
        for chunk in load_pipeline().chain.stream(question):
            if "sources" in chunk:
                sources.extend(chunk["sources"])
            if "answer" in chunk:
                yield chunk["answer"]

    st.markdown("### Answer")
    with st.spinner("Retrieving…"):
        st.write_stream(answer_tokens())

    st.markdown("### Sources")
    for s in sources:
        with st.expander(
            f"[{s.n}] {s.title} (arXiv:{s.arxiv_id}, p.{s.page}) — score {s.score:.3f}"
        ):
            st.write(s.text)
