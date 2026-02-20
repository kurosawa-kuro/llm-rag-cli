from __future__ import annotations

from dataclasses import dataclass
from typing import List
from langchain_core.documents import Document

from rag.core.interfaces import VectorStoreProtocol, RerankerProtocol


@dataclass(frozen=True)
class TwoStageRetrieval:
    vectorstore: VectorStoreProtocol
    reranker: RerankerProtocol
    search_k: int
    rerank_top_k: int
    score_threshold: float = 0.5

    def retrieve(self, query: str) -> List[Document]:
        # 1st stage: vector search with score filtering
        results = self.vectorstore.similarity_search_with_score(
            query, k=self.search_k,
        )
        docs = [doc for doc, score in results if score <= self.score_threshold]
        if not docs:
            return []

        # 2nd stage: rerank
        reranked = self.reranker.compress_documents(list(docs), query) or []
        return list(reranked[: self.rerank_top_k])
