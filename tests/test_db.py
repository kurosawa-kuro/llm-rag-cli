import sys
from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture(autouse=True)
def reset_vectorstore():
    import app.db
    app.db._vectorstore = None
    yield
    app.db._vectorstore = None


class TestGetVectorstore:
    @patch("app.db.get_embeddings")
    @patch("app.db.PGVector")
    def test_creates_pgvector_with_correct_params(self, mock_pgvector_class, mock_get_embeddings):
        from app.db import get_vectorstore
        from app.config import CONNECTION_STRING, COLLECTION_NAME

        get_vectorstore()

        mock_pgvector_class.assert_called_once_with(
            embeddings=mock_get_embeddings.return_value,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )

    @patch("app.db.get_embeddings")
    @patch("app.db.PGVector")
    def test_returns_pgvector_instance(self, mock_pgvector_class, mock_get_embeddings):
        from app.db import get_vectorstore

        result = get_vectorstore()
        assert result == mock_pgvector_class.return_value

    @patch("app.db.get_embeddings")
    @patch("app.db.PGVector")
    def test_singleton_returns_same_instance(self, mock_pgvector_class, mock_get_embeddings):
        from app.db import get_vectorstore

        vs1 = get_vectorstore()
        vs2 = get_vectorstore()
        assert vs1 is vs2
        mock_pgvector_class.assert_called_once()

    @patch("app.db.get_embeddings")
    @patch("app.db.PGVector")
    def test_uses_jsonb(self, mock_pgvector_class, mock_get_embeddings):
        from app.db import get_vectorstore

        get_vectorstore()

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["use_jsonb"] is True

    @patch("app.db.get_embeddings")
    @patch("app.db.PGVector")
    def test_uses_documents_collection(self, mock_pgvector_class, mock_get_embeddings):
        from app.db import get_vectorstore

        get_vectorstore()

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["collection_name"] == "documents"

    @patch("app.db.get_embeddings")
    @patch("app.db.PGVector")
    def test_passes_embeddings_instance(self, mock_pgvector_class, mock_get_embeddings):
        from app.db import get_vectorstore

        get_vectorstore()

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["embeddings"] == mock_get_embeddings.return_value

    @patch("app.db.get_embeddings")
    @patch("app.db.PGVector")
    def test_connection_string_format(self, mock_pgvector_class, mock_get_embeddings):
        from app.db import get_vectorstore

        get_vectorstore()

        call_kwargs = mock_pgvector_class.call_args[1]
        assert "postgresql+psycopg://" in call_kwargs["connection"]


class TestInitDb:
    @patch("app.db.get_vectorstore")
    def test_init_db_calls_get_vectorstore(self, mock_get_vs):
        from app.db import init_db

        init_db()

        mock_get_vs.assert_called_once()

    @patch("app.db.get_vectorstore")
    def test_init_db_returns_none(self, mock_get_vs):
        from app.db import init_db

        result = init_db()

        assert result is None
