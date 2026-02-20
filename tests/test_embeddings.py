import sys
from unittest.mock import patch, MagicMock
import pytest


class TestCreateEmbeddings:
    @patch.dict(sys.modules, {"langchain_huggingface": MagicMock()})
    def test_loads_correct_model(self):
        import app.embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        result = app.embeddings.create_embeddings()
        HuggingFaceEmbeddings.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")

    @patch.dict(sys.modules, {"langchain_huggingface": MagicMock()})
    def test_returns_embeddings_instance(self):
        import app.embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        result = app.embeddings.create_embeddings()
        assert result == HuggingFaceEmbeddings.return_value


class TestEmbed:
    @patch("app.embeddings.create_embeddings")
    def test_calls_embed_documents(self, mock_create_emb):
        mock_create_emb.return_value.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]
        from app.embeddings import embed

        result = embed(["text1", "text2"])
        mock_create_emb.return_value.embed_documents.assert_called_once_with(["text1", "text2"])

    @patch("app.embeddings.create_embeddings")
    def test_returns_embed_documents_result(self, mock_create_emb):
        expected = [[0.1] * 384, [0.2] * 384]
        mock_create_emb.return_value.embed_documents.return_value = expected
        from app.embeddings import embed

        result = embed(["text1", "text2"])
        assert result == expected

    @patch("app.embeddings.create_embeddings")
    def test_multiple_texts(self, mock_create_emb):
        expected = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        mock_create_emb.return_value.embed_documents.return_value = expected
        from app.embeddings import embed

        result = embed(["a", "b", "c"])
        assert len(result) == 3

    @patch("app.embeddings.create_embeddings")
    def test_single_text(self, mock_create_emb):
        expected = [[0.1] * 384]
        mock_create_emb.return_value.embed_documents.return_value = expected
        from app.embeddings import embed

        result = embed(["hello"])
        assert len(result) == 1

    @patch("app.embeddings.create_embeddings")
    def test_embed_japanese_text(self, mock_create_emb):
        expected = [[0.1] * 384, [0.2] * 384]
        mock_create_emb.return_value.embed_documents.return_value = expected
        from app.embeddings import embed

        result = embed(["パスワードを忘れた", "料金プラン"])
        mock_create_emb.return_value.embed_documents.assert_called_once_with(["パスワードを忘れた", "料金プラン"])
        assert len(result) == 2


class TestEmbedEdgeCases:
    @patch("app.embeddings.create_embeddings")
    def test_empty_list(self, mock_create_emb):
        mock_create_emb.return_value.embed_documents.return_value = []
        from app.embeddings import embed

        result = embed([])
        mock_create_emb.return_value.embed_documents.assert_called_once_with([])
        assert result == []

    @patch("app.embeddings.create_embeddings")
    def test_embed_raises_propagates(self, mock_create_emb):
        mock_create_emb.return_value.embed_documents.side_effect = ValueError("invalid input")
        from app.embeddings import embed

        with pytest.raises(ValueError, match="invalid input"):
            embed(["text"])
