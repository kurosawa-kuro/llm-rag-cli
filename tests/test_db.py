from unittest.mock import patch, MagicMock
import pytest


class TestCreateVectorstore:
    @patch("rag.infra.db.PGVector")
    def test_creates_pgvector_with_correct_params(self, mock_pgvector_class):
        from rag.infra.db import create_vectorstore
        from rag.core.config import CONNECTION_STRING, COLLECTION_NAME

        mock_embeddings = MagicMock()
        create_vectorstore(mock_embeddings)

        mock_pgvector_class.assert_called_once_with(
            embeddings=mock_embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )

    @patch("rag.infra.db.PGVector")
    def test_returns_pgvector_instance(self, mock_pgvector_class):
        from rag.infra.db import create_vectorstore

        mock_embeddings = MagicMock()
        result = create_vectorstore(mock_embeddings)
        assert result == mock_pgvector_class.return_value

    @patch("rag.infra.db.PGVector")
    def test_uses_jsonb(self, mock_pgvector_class):
        from rag.infra.db import create_vectorstore

        mock_embeddings = MagicMock()
        create_vectorstore(mock_embeddings)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["use_jsonb"] is True

    @patch("rag.infra.db.PGVector")
    def test_uses_documents_collection(self, mock_pgvector_class):
        from rag.infra.db import create_vectorstore

        mock_embeddings = MagicMock()
        create_vectorstore(mock_embeddings)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["collection_name"] == "documents"

    @patch("rag.infra.db.PGVector")
    def test_passes_embeddings_instance(self, mock_pgvector_class):
        from rag.infra.db import create_vectorstore

        mock_embeddings = MagicMock()
        create_vectorstore(mock_embeddings)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["embeddings"] is mock_embeddings

    @patch("rag.infra.db.PGVector")
    def test_connection_string_format(self, mock_pgvector_class):
        from rag.infra.db import create_vectorstore

        mock_embeddings = MagicMock()
        create_vectorstore(mock_embeddings)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert "postgresql+psycopg://" in call_kwargs["connection"]
