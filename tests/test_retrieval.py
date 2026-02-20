from unittest.mock import MagicMock
from langchain_core.documents import Document
import pytest

from app.retrieval import TwoStageRetrieval


@pytest.fixture
def mock_vectorstore():
    vs = MagicMock()
    retriever = MagicMock()
    vs.as_retriever.return_value = retriever
    return vs


@pytest.fixture
def mock_reranker():
    return MagicMock()


class TestTwoStageRetrieval:
    def test_retrieve_calls_vector_search_then_rerank(self, mock_vectorstore, mock_reranker):
        docs = [
            Document(page_content=f"doc{i}", metadata={"source": f"s{i}"})
            for i in range(5)
        ]
        mock_vectorstore.as_retriever.return_value.invoke.return_value = docs
        mock_reranker.compress_documents.return_value = docs[:3]

        strategy = TwoStageRetrieval(
            vectorstore=mock_vectorstore,
            reranker=mock_reranker,
            search_k=10,
            rerank_top_k=3,
        )
        result = strategy.retrieve("test query")

        mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 10})
        mock_vectorstore.as_retriever.return_value.invoke.assert_called_once_with("test query")
        mock_reranker.compress_documents.assert_called_once_with(docs, "test query")
        assert len(result) == 3

    def test_retrieve_limits_to_rerank_top_k(self, mock_vectorstore, mock_reranker):
        docs = [
            Document(page_content=f"doc{i}", metadata={"source": f"s{i}"})
            for i in range(5)
        ]
        mock_vectorstore.as_retriever.return_value.invoke.return_value = docs
        mock_reranker.compress_documents.return_value = docs

        strategy = TwoStageRetrieval(
            vectorstore=mock_vectorstore,
            reranker=mock_reranker,
            search_k=10,
            rerank_top_k=2,
        )
        result = strategy.retrieve("test")

        assert len(result) == 2

    def test_retrieve_empty_vector_results(self, mock_vectorstore, mock_reranker):
        mock_vectorstore.as_retriever.return_value.invoke.return_value = []

        strategy = TwoStageRetrieval(
            vectorstore=mock_vectorstore,
            reranker=mock_reranker,
            search_k=10,
            rerank_top_k=3,
        )
        result = strategy.retrieve("test")

        assert result == []
        mock_reranker.compress_documents.assert_not_called()

    def test_retrieve_none_vector_results(self, mock_vectorstore, mock_reranker):
        mock_vectorstore.as_retriever.return_value.invoke.return_value = None

        strategy = TwoStageRetrieval(
            vectorstore=mock_vectorstore,
            reranker=mock_reranker,
            search_k=10,
            rerank_top_k=3,
        )
        result = strategy.retrieve("test")

        assert result == []
        mock_reranker.compress_documents.assert_not_called()

    def test_retrieve_fewer_than_top_k(self, mock_vectorstore, mock_reranker):
        docs = [Document(page_content="doc1", metadata={"source": "s1"})]
        mock_vectorstore.as_retriever.return_value.invoke.return_value = docs
        mock_reranker.compress_documents.return_value = docs

        strategy = TwoStageRetrieval(
            vectorstore=mock_vectorstore,
            reranker=mock_reranker,
            search_k=10,
            rerank_top_k=3,
        )
        result = strategy.retrieve("test")

        assert len(result) == 1

    def test_is_frozen_dataclass(self):
        vs = MagicMock()
        rr = MagicMock()
        strategy = TwoStageRetrieval(vectorstore=vs, reranker=rr, search_k=10, rerank_top_k=3)
        with pytest.raises(AttributeError):
            strategy.search_k = 20
