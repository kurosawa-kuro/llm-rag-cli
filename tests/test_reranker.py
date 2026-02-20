import sys
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
import pytest


@pytest.fixture(autouse=True)
def reset_reranker():
    import app.reranker
    app.reranker._reranker = None
    yield
    app.reranker._reranker = None


class TestGetReranker:
    @patch.dict(sys.modules, {
        "langchain_community.cross_encoders": MagicMock(),
        "langchain_classic": MagicMock(),
        "langchain_classic.retrievers": MagicMock(),
        "langchain_classic.retrievers.document_compressors": MagicMock(),
    })
    def test_loads_correct_model(self):
        import app.reranker
        app.reranker._reranker = None

        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        app.reranker.get_reranker()
        HuggingFaceCrossEncoder.assert_called_once_with(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    @patch.dict(sys.modules, {
        "langchain_community.cross_encoders": MagicMock(),
        "langchain_classic": MagicMock(),
        "langchain_classic.retrievers": MagicMock(),
        "langchain_classic.retrievers.document_compressors": MagicMock(),
    })
    def test_creates_reranker_with_model(self):
        import app.reranker
        app.reranker._reranker = None

        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        app.reranker.get_reranker()
        CrossEncoderReranker.assert_called_once_with(
            model=HuggingFaceCrossEncoder.return_value,
            top_n=3,
        )

    @patch.dict(sys.modules, {
        "langchain_community.cross_encoders": MagicMock(),
        "langchain_classic": MagicMock(),
        "langchain_classic.retrievers": MagicMock(),
        "langchain_classic.retrievers.document_compressors": MagicMock(),
    })
    def test_returns_reranker_instance(self):
        import app.reranker
        app.reranker._reranker = None

        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        result = app.reranker.get_reranker()
        assert result == CrossEncoderReranker.return_value

    @patch.dict(sys.modules, {
        "langchain_community.cross_encoders": MagicMock(),
        "langchain_classic": MagicMock(),
        "langchain_classic.retrievers": MagicMock(),
        "langchain_classic.retrievers.document_compressors": MagicMock(),
    })
    def test_singleton_only_loads_once(self):
        import app.reranker
        app.reranker._reranker = None

        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        app.reranker.get_reranker()
        app.reranker.get_reranker()
        HuggingFaceCrossEncoder.assert_called_once()


class TestRerank:
    @patch("app.reranker.get_reranker")
    def test_returns_reranked_documents(self, mock_get_reranker):
        reranked = [
            Document(page_content="high", metadata={"source": "s1"}),
            Document(page_content="mid", metadata={"source": "s2"}),
            Document(page_content="low", metadata={"source": "s0"}),
        ]
        mock_get_reranker.return_value.compress_documents.return_value = reranked
        from app.reranker import rerank

        result = rerank("query", [
            {"content": "high", "source": "s1"},
            {"content": "mid", "source": "s2"},
            {"content": "low", "source": "s0"},
        ], top_k=3)

        assert len(result) == 3
        assert result[0]["content"] == "high"

    @patch("app.reranker.get_reranker")
    def test_returns_top_k_documents(self, mock_get_reranker):
        reranked = [
            Document(page_content="high", metadata={"source": "s1"}),
            Document(page_content="mid", metadata={"source": "s2"}),
            Document(page_content="low", metadata={"source": "s0"}),
        ]
        mock_get_reranker.return_value.compress_documents.return_value = reranked
        from app.reranker import rerank

        result = rerank("query", [
            {"content": f"doc{i}", "source": f"s{i}"} for i in range(5)
        ], top_k=3)

        assert len(result) == 3

    @patch("app.reranker.get_reranker")
    def test_empty_docs_returns_empty(self, mock_get_reranker):
        from app.reranker import rerank

        result = rerank("query", [], top_k=3)
        assert result == []

    @patch("app.reranker.get_reranker")
    def test_preserves_all_dict_fields(self, mock_get_reranker):
        reranked = [Document(page_content="text", metadata={"source": "file.pdf:p1"})]
        mock_get_reranker.return_value.compress_documents.return_value = reranked
        from app.reranker import rerank

        result = rerank("query", [{"content": "text", "source": "file.pdf:p1"}], top_k=1)
        assert result[0]["content"] == "text"
        assert result[0]["source"] == "file.pdf:p1"

    @patch("app.reranker.get_reranker")
    def test_single_doc_returns_that_doc(self, mock_get_reranker):
        reranked = [Document(page_content="only doc", metadata={"source": "src1"})]
        mock_get_reranker.return_value.compress_documents.return_value = reranked
        from app.reranker import rerank

        result = rerank("query", [{"content": "only doc", "source": "src1"}], top_k=3)
        assert len(result) == 1
        assert result[0]["content"] == "only doc"

    @patch("app.reranker.get_reranker")
    def test_top_k_1_returns_single(self, mock_get_reranker):
        reranked = [
            Document(page_content="high", metadata={"source": "s1"}),
            Document(page_content="low", metadata={"source": "s0"}),
        ]
        mock_get_reranker.return_value.compress_documents.return_value = reranked
        from app.reranker import rerank

        result = rerank("query", [
            {"content": "low", "source": "s0"},
            {"content": "high", "source": "s1"},
        ], top_k=1)
        assert len(result) == 1

    @patch("app.reranker.get_reranker")
    def test_returns_list_type(self, mock_get_reranker):
        reranked = [Document(page_content="doc", metadata={"source": "s"})]
        mock_get_reranker.return_value.compress_documents.return_value = reranked
        from app.reranker import rerank

        result = rerank("query", [{"content": "doc", "source": "s"}], top_k=1)
        assert isinstance(result, list)

    @patch("app.reranker.get_reranker")
    def test_calls_compress_documents(self, mock_get_reranker):
        reranked = [Document(page_content="doc", metadata={"source": "s"})]
        mock_get_reranker.return_value.compress_documents.return_value = reranked
        from app.reranker import rerank

        rerank("my query", [{"content": "doc A", "source": "s0"}, {"content": "doc B", "source": "s1"}], top_k=2)
        mock_get_reranker.return_value.compress_documents.assert_called_once()
        call_args = mock_get_reranker.return_value.compress_documents.call_args
        assert call_args[0][1] == "my query"

    @patch("app.reranker.get_reranker")
    def test_converts_dicts_to_documents(self, mock_get_reranker):
        reranked = [Document(page_content="doc A", metadata={"source": "s0"})]
        mock_get_reranker.return_value.compress_documents.return_value = reranked
        from app.reranker import rerank

        rerank("query", [{"content": "doc A", "source": "s0"}], top_k=1)
        lc_docs = mock_get_reranker.return_value.compress_documents.call_args[0][0]
        assert isinstance(lc_docs[0], Document)
        assert lc_docs[0].page_content == "doc A"


class TestGetCompressionRetriever:
    @patch.dict(sys.modules, {
        "langchain_classic": MagicMock(),
        "langchain_classic.retrievers": MagicMock(),
    })
    @patch("app.reranker.get_reranker")
    def test_wraps_base_retriever(self, mock_get_reranker):
        import app.reranker
        from langchain_classic.retrievers import ContextualCompressionRetriever

        base = MagicMock()
        app.reranker.get_compression_retriever(base)
        ContextualCompressionRetriever.assert_called_once_with(
            base_compressor=mock_get_reranker.return_value,
            base_retriever=base,
        )

    @patch.dict(sys.modules, {
        "langchain_classic": MagicMock(),
        "langchain_classic.retrievers": MagicMock(),
    })
    @patch("app.reranker.get_reranker")
    def test_returns_compression_retriever(self, mock_get_reranker):
        import app.reranker
        from langchain_classic.retrievers import ContextualCompressionRetriever

        base = MagicMock()
        result = app.reranker.get_compression_retriever(base)
        assert result == ContextualCompressionRetriever.return_value
