from dataclasses import dataclass
from app.config import SEARCH_K, RERANK_TOP_K


@dataclass
class RagSettings:
    search_k: int = SEARCH_K
    rerank_top_k: int = RERANK_TOP_K


class AppContainer:
    def __init__(self, *, settings=None, embeddings=None, vectorstore=None,
                 reranker=None, llm=None, prompt_builder=None):
        self.settings = settings or RagSettings()
        self._embeddings = embeddings
        self._vectorstore = vectorstore
        self._reranker = reranker
        self._llm = llm
        self._prompt_builder = prompt_builder

    @property
    def embeddings(self):
        if self._embeddings is None:
            from app.embeddings import create_embeddings
            self._embeddings = create_embeddings()
        return self._embeddings

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            from app.db import create_vectorstore
            self._vectorstore = create_vectorstore(self.embeddings)
        return self._vectorstore

    @property
    def reranker(self):
        if self._reranker is None:
            from app.reranker import create_reranker
            self._reranker = create_reranker()
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            from app.llm import create_llm
            self._llm = create_llm()
        return self._llm

    @property
    def prompt_builder(self):
        if self._prompt_builder is None:
            from app.prompting import build_prompt
            self._prompt_builder = build_prompt
        return self._prompt_builder


_container = None


def get_container():
    global _container
    if _container is None:
        _container = AppContainer()
    return _container
