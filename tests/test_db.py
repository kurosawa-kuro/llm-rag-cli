from unittest.mock import patch, MagicMock, call


class TestGetConn:
    @patch("app.db.psycopg2.connect")
    def test_calls_connect_with_db_config(self, mock_connect):
        from app.db import get_conn
        from app.config import DB_CONFIG

        get_conn()
        mock_connect.assert_called_once_with(**DB_CONFIG)

    @patch("app.db.psycopg2.connect")
    def test_returns_connection(self, mock_connect):
        from app.db import get_conn

        conn = get_conn()
        assert conn == mock_connect.return_value


class TestInitDb:
    @patch("app.db.get_conn")
    def test_creates_vector_extension(self, mock_get_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_get_conn.return_value = conn
        from app.db import init_db

        init_db()

        calls = [str(c) for c in cur.execute.call_args_list]
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in c for c in calls)

    @patch("app.db.get_conn")
    def test_creates_documents_table(self, mock_get_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_get_conn.return_value = conn
        from app.db import init_db

        init_db()

        calls = [str(c) for c in cur.execute.call_args_list]
        assert any("CREATE TABLE IF NOT EXISTS documents" in c for c in calls)
        assert any("VECTOR(384)" in c for c in calls)

    @patch("app.db.get_conn")
    def test_creates_source_column(self, mock_get_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_get_conn.return_value = conn
        from app.db import init_db

        init_db()

        calls = [str(c) for c in cur.execute.call_args_list]
        assert any("source TEXT" in c for c in calls)

    @patch("app.db.get_conn")
    def test_creates_chunk_index_column(self, mock_get_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_get_conn.return_value = conn
        from app.db import init_db

        init_db()

        calls = [str(c) for c in cur.execute.call_args_list]
        assert any("chunk_index INT" in c for c in calls)

    @patch("app.db.get_conn")
    def test_commits(self, mock_get_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_get_conn.return_value = conn
        from app.db import init_db

        init_db()

        conn.commit.assert_called_once()
