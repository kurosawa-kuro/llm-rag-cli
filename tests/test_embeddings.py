import sys
from unittest.mock import patch, MagicMock


class TestCreateEmbeddings:
    @patch.dict(sys.modules, {"langchain_huggingface": MagicMock()})
    def test_loads_correct_model(self):
        import rag.components.embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        result = rag.components.embeddings.create_embeddings()
        HuggingFaceEmbeddings.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")

    @patch.dict(sys.modules, {"langchain_huggingface": MagicMock()})
    def test_returns_embeddings_instance(self):
        import rag.components.embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        result = rag.components.embeddings.create_embeddings()
        assert result == HuggingFaceEmbeddings.return_value
