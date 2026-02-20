from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from rag.core.config import RERANKER_MODEL, RERANK_TOP_K


class CrossEncoderReranker:
    """HuggingFaceCrossEncoder を使った reranker。RerankerProtocol を満たす。"""

    def __init__(self, model_name: str = RERANKER_MODEL, top_n: int = RERANK_TOP_K):
        self._model = HuggingFaceCrossEncoder(model_name=model_name)
        self.top_n = top_n

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.score(pairs)
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.top_n]]


def create_reranker(top_n: int = RERANK_TOP_K) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=RERANKER_MODEL, top_n=top_n)


def get_compression_retriever(base_retriever):
    from langchain.retrievers import ContextualCompressionRetriever
    return ContextualCompressionRetriever(
        base_compressor=create_reranker(),
        base_retriever=base_retriever,
    )


def rerank(query, docs, top_k=3):
    if not docs:
        return []
    lc_docs = [Document(page_content=d["content"], metadata={"source": d.get("source", "")}) for d in docs]
    compressor = create_reranker(top_n=top_k)
    reranked = compressor.compress_documents(lc_docs, query)
    return [{"content": doc.page_content, "source": doc.metadata.get("source", "")} for doc in reranked[:top_k]]
