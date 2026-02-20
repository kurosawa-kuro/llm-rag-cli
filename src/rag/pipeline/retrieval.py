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

    def retrieve(self, query: str) -> List[Document]:
        # 1st stage: vector search
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.search_k})
        docs = retriever.invoke(query) or []
        if not docs:
            return []

        # 2nd stage: rerank
        reranked = self.reranker.compress_documents(list(docs), query) or []
        return list(reranked[: self.rerank_top_k])
