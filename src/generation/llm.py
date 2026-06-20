"""LLM abstraction.

The rest of the app only knows about the `LLM` protocol, so swapping Ollama for
Claude or OpenAI later means adding one class and one line in `get_llm()` - no
changes to the pipeline.
"""
from __future__ import annotations

from typing import Protocol

import ollama

from src.config import settings


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or settings.ollama_model
        self._client = ollama.Client(host=host or settings.ollama_host)

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": settings.temperature},
        )
        return resp["message"]["content"].strip()



def get_llm() -> LLM:
    backend = settings.llm_backend.lower()
    if backend == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM backend: {settings.llm_backend}")
