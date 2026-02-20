import sys
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
import pytest


class TestCreateReranker:
    @patch("rag.components.reranker.HuggingFaceCrossEncoder")
    def test_loads_correct_model(self, mock_hf):
        from rag.components.reranker import CrossEncoderReranker
        CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        mock_hf.assert_called_once_with(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    @patch("rag.components.reranker.HuggingFaceCrossEncoder")
    def test_creates_reranker_with_defaults(self, mock_hf):
        from rag.components.reranker import create_reranker
        result = create_reranker()
        assert result.top_n == 3
        mock_hf.assert_called_once_with(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    @patch("rag.components.reranker.HuggingFaceCrossEncoder")
    def test_returns_reranker_instance(self, mock_hf):
        from rag.components.reranker import create_reranker, CrossEncoderReranker
        result = create_reranker()
        assert isinstance(result, CrossEncoderReranker)


class TestRerank:
    @patch("rag.components.reranker.create_reranker")
    def test_returns_reranked_documents(self, mock_create_reranker):
        reranked = [
            Document(page_content="high", metadata={"source": "s1"}),
            Document(page_content="mid", metadata={"source": "s2"}),
            Document(page_content="low", metadata={"source": "s0"}),
        ]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        result = rerank("query", [
            {"content": "high", "source": "s1"},
            {"content": "mid", "source": "s2"},
            {"content": "low", "source": "s0"},
        ], top_k=3)

        assert len(result) == 3
        assert result[0]["content"] == "high"

    @patch("rag.components.reranker.create_reranker")
    def test_returns_top_k_documents(self, mock_create_reranker):
        reranked = [
            Document(page_content="high", metadata={"source": "s1"}),
            Document(page_content="mid", metadata={"source": "s2"}),
            Document(page_content="low", metadata={"source": "s0"}),
        ]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        result = rerank("query", [
            {"content": f"doc{i}", "source": f"s{i}"} for i in range(5)
        ], top_k=3)

        assert len(result) == 3

    @patch("rag.components.reranker.create_reranker")
    def test_empty_docs_returns_empty(self, mock_create_reranker):
        from rag.components.reranker import rerank

        result = rerank("query", [], top_k=3)
        assert result == []

    @patch("rag.components.reranker.create_reranker")
    def test_preserves_all_dict_fields(self, mock_create_reranker):
        reranked = [Document(page_content="text", metadata={"source": "file.pdf:p1"})]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        result = rerank("query", [{"content": "text", "source": "file.pdf:p1"}], top_k=1)
        assert result[0]["content"] == "text"
        assert result[0]["source"] == "file.pdf:p1"

    @patch("rag.components.reranker.create_reranker")
    def test_single_doc_returns_that_doc(self, mock_create_reranker):
        reranked = [Document(page_content="only doc", metadata={"source": "src1"})]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        result = rerank("query", [{"content": "only doc", "source": "src1"}], top_k=3)
        assert len(result) == 1
        assert result[0]["content"] == "only doc"

    @patch("rag.components.reranker.create_reranker")
    def test_top_k_1_returns_single(self, mock_create_reranker):
        reranked = [
            Document(page_content="high", metadata={"source": "s1"}),
            Document(page_content="low", metadata={"source": "s0"}),
        ]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        result = rerank("query", [
            {"content": "low", "source": "s0"},
            {"content": "high", "source": "s1"},
        ], top_k=1)
        assert len(result) == 1

    @patch("rag.components.reranker.create_reranker")
    def test_returns_list_type(self, mock_create_reranker):
        reranked = [Document(page_content="doc", metadata={"source": "s"})]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        result = rerank("query", [{"content": "doc", "source": "s"}], top_k=1)
        assert isinstance(result, list)

    @patch("rag.components.reranker.create_reranker")
    def test_calls_compress_documents(self, mock_create_reranker):
        reranked = [Document(page_content="doc", metadata={"source": "s"})]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        rerank("my query", [{"content": "doc A", "source": "s0"}, {"content": "doc B", "source": "s1"}], top_k=2)
        mock_create_reranker.return_value.compress_documents.assert_called_once()
        call_args = mock_create_reranker.return_value.compress_documents.call_args
        assert call_args[0][1] == "my query"

    @patch("rag.components.reranker.create_reranker")
    def test_converts_dicts_to_documents(self, mock_create_reranker):
        reranked = [Document(page_content="doc A", metadata={"source": "s0"})]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        rerank("query", [{"content": "doc A", "source": "s0"}], top_k=1)
        lc_docs = mock_create_reranker.return_value.compress_documents.call_args[0][0]
        assert isinstance(lc_docs[0], Document)
        assert lc_docs[0].page_content == "doc A"


class TestRerankEdgeCases:
    @patch("rag.components.reranker.create_reranker")
    def test_none_docs_returns_empty(self, mock_create_reranker):
        from rag.components.reranker import rerank

        result = rerank("query", None, top_k=3)
        assert result == []

    @patch("rag.components.reranker.create_reranker")
    def test_docs_missing_source_key_uses_empty_default(self, mock_create_reranker):
        reranked = [Document(page_content="text", metadata={})]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        result = rerank("query", [{"content": "text"}], top_k=1)
        assert result[0]["source"] == ""

    @patch("rag.components.reranker.create_reranker")
    def test_top_k_zero_returns_empty(self, mock_create_reranker):
        reranked = [Document(page_content="text", metadata={"source": "s"})]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        result = rerank("query", [{"content": "text", "source": "s"}], top_k=0)
        assert result == []

    @patch("rag.components.reranker.create_reranker")
    def test_empty_query_still_calls_compressor(self, mock_create_reranker):
        reranked = [Document(page_content="text", metadata={"source": "s"})]
        mock_create_reranker.return_value.compress_documents.return_value = reranked
        from rag.components.reranker import rerank

        result = rerank("", [{"content": "text", "source": "s"}], top_k=1)
        mock_create_reranker.return_value.compress_documents.assert_called_once()
        assert len(result) == 1


class TestGetCompressionRetriever:
    @patch.dict(sys.modules, {
        "langchain": MagicMock(),
        "langchain.retrievers": MagicMock(),
    })
    @patch("rag.components.reranker.create_reranker")
    def test_wraps_base_retriever(self, mock_create_reranker):
        import rag.components.reranker
        from langchain.retrievers import ContextualCompressionRetriever

        base = MagicMock()
        rag.components.reranker.get_compression_retriever(base)
        ContextualCompressionRetriever.assert_called_once_with(
            base_compressor=mock_create_reranker.return_value,
            base_retriever=base,
        )

    @patch.dict(sys.modules, {
        "langchain": MagicMock(),
        "langchain.retrievers": MagicMock(),
    })
    @patch("rag.components.reranker.create_reranker")
    def test_returns_compression_retriever(self, mock_create_reranker):
        import rag.components.reranker
        from langchain.retrievers import ContextualCompressionRetriever

        base = MagicMock()
        result = rag.components.reranker.get_compression_retriever(base)
        assert result == ContextualCompressionRetriever.return_value
