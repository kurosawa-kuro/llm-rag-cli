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

    @patch("app.db.get_conn")
    def test_creates_id_serial_primary_key(self, mock_get_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_get_conn.return_value = conn
        from app.db import init_db

        init_db()

        calls = [str(c) for c in cur.execute.call_args_list]
        assert any("id SERIAL PRIMARY KEY" in c for c in calls)

    @patch("app.db.get_conn")
    def test_creates_content_text_column(self, mock_get_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_get_conn.return_value = conn
        from app.db import init_db

        init_db()

        calls = [str(c) for c in cur.execute.call_args_list]
        assert any("content TEXT" in c for c in calls)

    @patch("app.db.get_conn")
    def test_executes_extension_before_table(self, mock_get_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_get_conn.return_value = conn
        from app.db import init_db

        init_db()

        calls = [str(c) for c in cur.execute.call_args_list]
        ext_idx = next(i for i, c in enumerate(calls) if "CREATE EXTENSION" in c)
        tbl_idx = next(i for i, c in enumerate(calls) if "CREATE TABLE" in c)
        assert ext_idx < tbl_idx

    @patch("app.db.get_conn")
    def test_idempotent_uses_if_not_exists(self, mock_get_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_get_conn.return_value = conn
        from app.db import init_db

        init_db()

        calls = [str(c) for c in cur.execute.call_args_list]
        for c in calls:
            if "CREATE" in c:
                assert "IF NOT EXISTS" in c
