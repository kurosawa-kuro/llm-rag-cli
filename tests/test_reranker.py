from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


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
