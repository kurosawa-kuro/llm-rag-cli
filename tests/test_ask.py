from unittest.mock import patch, MagicMock
import numpy as np
import pytest


class TestSearch:
    @patch("app.ask.rerank", side_effect=lambda q, docs, top_k: docs[:top_k])
    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_embeds_query_and_searches(self, mock_embed, mock_conn, mock_rerank,
                                       mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = [("doc1", "src1"), ("doc2", "src2"), ("doc3", "src3")]
        from app.ask import search

        results = search("test query")

        mock_embed.assert_called_once_with(["test query"])
        assert len(results) == 3

    @patch("app.ask.rerank", side_effect=lambda q, docs, top_k: docs[:top_k])
    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_uses_pgvector_distance_operator(self, mock_embed, mock_conn, mock_rerank,
                                              mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = []
        from app.ask import search

        search("query")

        sql = cur.execute.call_args[0][0]
        assert "<->" in sql
        assert "LIMIT" in sql

    @patch("app.ask.rerank", side_effect=lambda q, docs, top_k: docs[:top_k])
    @patch("app.ask.SEARCH_K", 10)
    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_uses_search_k_for_initial_retrieval(self, mock_embed, mock_conn,
                                                  mock_rerank,
                                                  mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = [("doc", "src")] * 10
        from app.ask import search

        search("query")

        params = cur.execute.call_args[0][1]
        assert 10 in params

    @patch("app.ask.rerank", side_effect=lambda q, docs, top_k: docs[:top_k])
    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_select_includes_source(self, mock_embed, mock_conn, mock_rerank,
                                     mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = []
        from app.ask import search

        search("query")

        sql = cur.execute.call_args[0][0]
        assert "source" in sql


class TestSearchWithReranking:
    @patch("app.ask.rerank")
    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_search_calls_rerank(self, mock_embed, mock_conn, mock_rerank,
                                  mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = [("doc1", "src1"), ("doc2", "src2")]
        mock_rerank.return_value = [{"content": "doc1", "source": "src1"}]
        from app.ask import search

        search("test query")

        mock_rerank.assert_called_once()
        call_args = mock_rerank.call_args[0]
        assert call_args[0] == "test query"
        assert len(call_args[1]) == 2

    @patch("app.ask.RERANK_TOP_K", 3)
    @patch("app.ask.rerank")
    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_search_passes_rerank_top_k(self, mock_embed, mock_conn, mock_rerank,
                                         mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = [("doc", "src")] * 10
        mock_rerank.return_value = [{"content": "doc", "source": "src"}] * 3
        from app.ask import search

        search("query")

        call_args = mock_rerank.call_args[0]
        assert call_args[2] == 3

    @patch("app.ask.rerank")
    @patch("app.ask.get_conn")
    @patch("app.ask.embed")
    def test_search_returns_reranked_results(self, mock_embed, mock_conn, mock_rerank,
                                              mock_db_connection):
        conn, cur = mock_db_connection
        mock_conn.return_value = conn
        mock_embed.return_value = np.array([[0.1] * 384])
        cur.fetchall.return_value = [("doc1", "src1"), ("doc2", "src2"), ("doc3", "src3")]
        reranked = [
            {"content": "doc2", "source": "src2"},
            {"content": "doc1", "source": "src1"},
        ]
        mock_rerank.return_value = reranked
        from app.ask import search

        results = search("query")

        assert results == reranked


class TestMain:
    @patch("builtins.print")
    @patch("app.ask.generate", return_value="回答テスト")
    @patch("app.ask.search", return_value=[
        {"content": "context1", "source": "doc.pdf:p1"},
        {"content": "context2", "source": "data.csv:r1"},
    ])
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
    @patch("app.ask.search", return_value=[
        {"content": "context1", "source": "doc.pdf:p1"},
    ])
    @patch("app.ask.sys")
    def test_prints_answer(self, mock_sys, mock_search, mock_gen, mock_print):
        mock_sys.argv = ["ask.py", "テスト"]
        from app.ask import main

        main()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "回答テスト" in printed

    @patch("builtins.print")
    @patch("app.ask.generate", return_value="回答テスト")
    @patch("app.ask.search", return_value=[
        {"content": "context1", "source": "doc.pdf:p1"},
        {"content": "context2", "source": "data.csv:r1"},
    ])
    @patch("app.ask.sys")
    def test_prints_sources(self, mock_sys, mock_search, mock_gen, mock_print):
        mock_sys.argv = ["ask.py", "テスト"]
        from app.ask import main

        main()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Sources" in printed
        assert "doc.pdf:p1" in printed
        assert "data.csv:r1" in printed
