"""Central configuration.

Every tunable lives here so the rest of the code never hard-codes a model name
or a path.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "arxiv_papers"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge models recommend prefixing the *query* with this instruction.
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # Reranking
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    chunk_size: int = 1000          # characters (~250 tokens)
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 20                 # candidates pulled from the vector store
    top_n: int = 5                  # passages kept after reranking and fed to the LLM

    # Generation
    llm_backend: str = "ollama"     # the LLM layer is swappable: see src/generation/llm.py
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"  
    temperature: float = 0.1 

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.raw_dir, self.chroma_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
