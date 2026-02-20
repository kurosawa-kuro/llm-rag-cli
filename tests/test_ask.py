from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
import sys
import pytest


class TestMain:
    @patch("builtins.print")
    @patch("app.ask.get_container")
    @patch("app.ask.get_graph")
    @patch("app.ask.sys")
    def test_uses_graph_invoke(self, mock_sys, mock_get_graph, mock_get_container, mock_print):
        mock_sys.argv = ["ask.py", "質問テスト"]
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "回答テスト",
            "sources": ["doc.pdf:p1", "data.csv:r1"],
        }
        from app.ask import main

        main()

        mock_get_graph.return_value.invoke.assert_called_once_with({"query": "質問テスト"})

    @patch("builtins.print")
    @patch("app.ask.get_container")
    @patch("app.ask.get_graph")
    @patch("app.ask.sys")
    def test_prints_answer(self, mock_sys, mock_get_graph, mock_get_container, mock_print):
        mock_sys.argv = ["ask.py", "テスト"]
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "回答テスト",
            "sources": ["doc.pdf:p1"],
        }
        from app.ask import main

        main()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "回答テスト" in printed

    @patch("builtins.print")
    @patch("app.ask.get_container")
    @patch("app.ask.get_graph")
    @patch("app.ask.sys")
    def test_prints_sources(self, mock_sys, mock_get_graph, mock_get_container, mock_print):
        mock_sys.argv = ["ask.py", "テスト"]
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "回答テスト",
            "sources": ["doc.pdf:p1", "data.csv:r1"],
        }
        from app.ask import main

        main()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Sources" in printed
        assert "doc.pdf:p1" in printed
        assert "data.csv:r1" in printed

    @patch("builtins.print")
    @patch("app.ask.get_container")
    @patch("app.ask.get_graph")
    @patch("app.ask.sys")
    def test_passes_query_from_argv(self, mock_sys, mock_get_graph, mock_get_container, mock_print):
        mock_sys.argv = ["ask.py", "テスト質問"]
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "回答",
            "sources": [],
        }
        from app.ask import main

        main()

        mock_get_graph.return_value.invoke.assert_called_once_with({"query": "テスト質問"})

    @patch("builtins.print")
    @patch("app.ask.get_container")
    @patch("app.ask.get_graph")
    @patch("app.ask.sys")
    def test_prints_answer_and_sources_sections(self, mock_sys, mock_get_graph, mock_get_container, mock_print):
        mock_sys.argv = ["ask.py", "テスト"]
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "回答テスト",
            "sources": ["doc.pdf:p1"],
        }
        from app.ask import main

        main()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Answer" in printed
        assert "Sources" in printed


class TestMainEdgeCases:
    @patch("app.ask.sys")
    def test_main_missing_argv_raises_index_error(self, mock_sys):
        mock_sys.argv = ["ask.py"]  # queryなし
        from app.ask import main

        with pytest.raises(IndexError):
            main()

    @patch("builtins.print")
    @patch("app.ask.get_container")
    @patch("app.ask.get_graph")
    @patch("app.ask.sys")
    def test_main_empty_sources(self, mock_sys, mock_get_graph, mock_get_container, mock_print):
        mock_sys.argv = ["ask.py", "テスト"]
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "回答",
            "sources": [],
        }
        from app.ask import main

        main()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "回答" in printed

    @patch("builtins.print")
    @patch("app.ask.get_container")
    @patch("app.ask.get_graph")
    @patch("app.ask.sys")
    def test_main_missing_sources_key(self, mock_sys, mock_get_graph, mock_get_container, mock_print):
        mock_sys.argv = ["ask.py", "テスト"]
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "回答",
            # "sources" キーなし → .get() で [] がデフォルト
        }
        from app.ask import main

        main()  # クラッシュしない
