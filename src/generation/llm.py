"""LLM factory.

The rest of the app only knows about LangChain's `BaseChatModel`, so swapping
Ollama for Claude or OpenAI is a config change: set RAG_LLM_PROVIDER and
RAG_LLM_MODEL and install the matching `langchain-<provider>` package.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.config import settings


def get_llm() -> BaseChatModel:
    provider = settings.llm_provider.lower()
    kwargs = {"base_url": settings.ollama_host} if provider == "ollama" else {}
    return init_chat_model(
        settings.llm_model,
        model_provider=provider,
        temperature=settings.temperature,
        **kwargs,
    )
