"""Minimal Streamlit demo.

Run: streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.vector_store import VectorStore


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


st.set_page_config(page_title="arXiv RAG", page_icon="📚")
st.title("📚 Ask the ML papers")
st.caption("Retrieval-augmented Q&A over an arXiv corpus — answers cite their sources.")

n_chunks = VectorStore().count()
if n_chunks == 0:
    st.warning("The index is empty. Run `python scripts/build_index.py` first.")
    st.stop()

st.sidebar.metric("Indexed chunks", n_chunks)

question = st.text_input("Your question", placeholder="e.g. How does a cross-encoder reranker work?")
if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating…"):
        result = load_pipeline().answer(question)

    st.markdown("### Answer")
    st.write(result.answer)

    st.markdown("### Sources")
    for s in result.sources:
        with st.expander(f"[{s.n}] {s.title} (arXiv:{s.arxiv_id}, p.{s.page}) — score {s.score:.3f}"):
            st.write(s.text)
