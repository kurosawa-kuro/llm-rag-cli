from unittest.mock import patch, MagicMock
import numpy as np
import pytest


class TestSearch:
    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_embeds_query_and_searches(self, mock_embed, mock_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = [("doc1",), ("doc2",), ("doc3",)]
        from app.ask import search

        results = search("test query")

        mock_embed.assert_called_once_with(["test query"])
        assert len(results) == 3
        assert results == ["doc1", "doc2", "doc3"]

    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_uses_pgvector_distance_operator(self, mock_embed, mock_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = []
        from app.ask import search

        search("query", k=3)

        sql = cur.execute.call_args[0][0]
        assert "<->" in sql
        assert "LIMIT" in sql

    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_respects_k_parameter(self, mock_embed, mock_conn, mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = []
        from app.ask import search

        search("query", k=5)

        params = cur.execute.call_args[0][1]
        assert 5 in params


class TestMain:
    @patch("builtins.print")
    @patch("app.ask.generate", return_value="回答テスト")
    @patch("app.ask.search", return_value=["context1", "context2"])
    @patch("app.ask.sys")
    def test_builds_japanese_prompt(self, mock_sys, mock_search, mock_gen, mock_print):
        mock_sys.argv = ["ask.py", "質問テスト"]
        from app.ask import main

        main()

        prompt = mock_gen.call_args[0][0]
        assert "以下の情報を基に回答してください" in prompt
        assert "質問:" in prompt
        assert "回答:" in prompt

    @patch("builtins.print")
    @patch("app.ask.generate", return_value="回答テスト")
    @patch("app.ask.search", return_value=["context1"])
    @patch("app.ask.sys")
    def test_prints_answer(self, mock_sys, mock_search, mock_gen, mock_print):
        mock_sys.argv = ["ask.py", "テスト"]
        from app.ask import main

        main()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "回答テスト" in printed
