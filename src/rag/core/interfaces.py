from __future__ import annotations

from typing import Protocol, List, Callable
from langchain_core.documents import Document


class VectorStoreProtocol(Protocol):
    def similarity_search_with_score(self, query: str, k: int = 4) -> list: ...


class RerankerProtocol(Protocol):
    def compress_documents(self, documents: List[Document], query: str) -> List[Document]: ...


class LLMProtocol(Protocol):
    def invoke(self, prompt: str) -> str: ...


PromptBuilder = Callable[[str, List[str]], str]


class RetrievalStrategyProtocol(Protocol):
    def retrieve(self, query: str) -> List[Document]: ...
