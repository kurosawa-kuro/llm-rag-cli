from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import SEARCH_K, RERANK_TOP_K
from app.interfaces import (
    VectorStoreProtocol,
    RerankerProtocol,
    LLMProtocol,
    PromptBuilder,
    RetrievalStrategyProtocol,
)


@dataclass(frozen=True)
class RagSettings:
    search_k: int = SEARCH_K
    rerank_top_k: int = RERANK_TOP_K


class AppContainer:
    def __init__(
        self,
        *,
        settings=None,
        embeddings=None,
        vectorstore: VectorStoreProtocol | None = None,
        reranker: RerankerProtocol | None = None,
        llm: LLMProtocol | None = None,
        prompt_builder: PromptBuilder | None = None,
        retrieval_strategy: RetrievalStrategyProtocol | None = None,
    ):
        self.settings = settings or RagSettings()
        self._embeddings = embeddings
        self._vectorstore = vectorstore
        self._reranker = reranker
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._retrieval_strategy = retrieval_strategy

    @property
    def embeddings(self):
        if self._embeddings is None:
            from app.embeddings import create_embeddings
            self._embeddings = create_embeddings()
        return self._embeddings

    @property
    def vectorstore(self) -> VectorStoreProtocol:
        if self._vectorstore is None:
            from app.db import create_vectorstore
            self._vectorstore = create_vectorstore(self.embeddings)
        return self._vectorstore

    @property
    def reranker(self) -> RerankerProtocol:
        if self._reranker is None:
            from app.reranker import create_reranker
            self._reranker = create_reranker()
        return self._reranker

    @property
    def llm(self) -> LLMProtocol:
        if self._llm is None:
            from app.llm import create_llm
            self._llm = create_llm()
        return self._llm

    @property
    def prompt_builder(self) -> PromptBuilder:
        if self._prompt_builder is None:
            from app.prompting import build_prompt
            self._prompt_builder = build_prompt
        return self._prompt_builder

    @property
    def retrieval_strategy(self) -> RetrievalStrategyProtocol:
        if self._retrieval_strategy is None:
            from app.retrieval import TwoStageRetrieval

            self._retrieval_strategy = TwoStageRetrieval(
                vectorstore=self.vectorstore,
                reranker=self.reranker,
                search_k=self.settings.search_k,
                rerank_top_k=self.settings.rerank_top_k,
            )
        return self._retrieval_strategy


_container = None


def get_container():
    global _container
    if _container is None:
        _container = AppContainer()
    return _container
