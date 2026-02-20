from dataclasses import dataclass
from app.config import SEARCH_K, RERANK_TOP_K


@dataclass
class RagSettings:
    search_k: int = SEARCH_K
    rerank_top_k: int = RERANK_TOP_K


class AppContainer:
    """
    依存（DB/モデル/設定/関数）を集約し、遅延初期化 & キャッシュするコンテナ。
    テストでは AppContainer を差し替えるだけで全依存を入れ替え可能。
    """

    def __init__(self, *, settings=None, vectorstore=None, reranker=None,
                 llm=None, prompt_builder=None):
        self.settings = settings or RagSettings()
        self._vectorstore = vectorstore
        self._reranker = reranker
        self._llm = llm
        self._prompt_builder = prompt_builder

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            from app.db import get_vectorstore
            self._vectorstore = get_vectorstore()
        return self._vectorstore

    @property
    def reranker(self):
        if self._reranker is None:
            from app.reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            from app.llm import get_llm
            self._llm = get_llm()
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
