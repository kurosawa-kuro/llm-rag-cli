from unittest.mock import MagicMock
from langchain_core.documents import Document
import pytest

from rag.pipeline.retrieval import TwoStageRetrieval


@pytest.fixture
def mock_vectorstore():
    return MagicMock()


@pytest.fixture
def mock_reranker():
    return MagicMock()


class TestTwoStageRetrieval:
    def test_retrieve_calls_vector_search_then_rerank(self, mock_vectorstore, mock_reranker):
        docs = [
            Document(page_content=f"doc{i}", metadata={"source": f"s{i}"})
            for i in range(5)
        ]
        mock_vectorstore.similarity_search_with_score.return_value = [
            (doc, 0.3) for doc in docs
        ]
        mock_reranker.compress_documents.return_value = docs[:3]

        strategy = TwoStageRetrieval(
            vectorstore=mock_vectorstore,
            reranker=mock_reranker,
            search_k=10,
            rerank_top_k=3,
        )
        result = strategy.retrieve("test query")

        mock_vectorstore.similarity_search_with_score.assert_called_once_with(
            "test query", k=10,
        )
        mock_reranker.compress_documents.assert_called_once_with(docs, "test query")
        assert len(result) == 3

    def test_retrieve_limits_to_rerank_top_k(self, mock_vectorstore, mock_reranker):
        docs = [
            Document(page_content=f"doc{i}", metadata={"source": f"s{i}"})
            for i in range(5)
        ]
        mock_vectorstore.similarity_search_with_score.return_value = [
            (doc, 0.3) for doc in docs
        ]
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
        mock_vectorstore.similarity_search_with_score.return_value = []

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
        mock_vectorstore.similarity_search_with_score.return_value = []

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
        mock_vectorstore.similarity_search_with_score.return_value = [
            (docs[0], 0.3),
        ]
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

    def test_filters_by_score_threshold(self, mock_vectorstore, mock_reranker):
        good_doc = Document(page_content="relevant", metadata={"source": "s1"})
        bad_doc = Document(page_content="irrelevant", metadata={"source": "s2"})
        mock_vectorstore.similarity_search_with_score.return_value = [
            (good_doc, 0.3),
            (bad_doc, 0.95),
        ]
        mock_reranker.compress_documents.return_value = [good_doc]

        strategy = TwoStageRetrieval(
            vectorstore=mock_vectorstore,
            reranker=mock_reranker,
            search_k=10,
            rerank_top_k=3,
            score_threshold=0.8,
        )
        result = strategy.retrieve("test")

        mock_reranker.compress_documents.assert_called_once_with([good_doc], "test")
        assert len(result) == 1

    def test_all_below_threshold_returns_empty(self, mock_vectorstore, mock_reranker):
        doc = Document(page_content="irrelevant", metadata={"source": "s1"})
        mock_vectorstore.similarity_search_with_score.return_value = [
            (doc, 0.95),
        ]

        strategy = TwoStageRetrieval(
            vectorstore=mock_vectorstore,
            reranker=mock_reranker,
            search_k=10,
            rerank_top_k=3,
            score_threshold=0.8,
        )
        result = strategy.retrieve("test")

        assert result == []
        mock_reranker.compress_documents.assert_not_called()
