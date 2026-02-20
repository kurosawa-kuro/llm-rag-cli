from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
import pytest


class TestSearch:
    @patch("app.ask.get_compression_retriever")
    @patch("app.ask.get_vectorstore")
    def test_returns_list_of_dicts(self, mock_vs, mock_ccr):
        mock_ccr.return_value.invoke.return_value = [
            Document(page_content="doc1", metadata={"source": "src1"}),
            Document(page_content="doc2", metadata={"source": "src2"}),
        ]
        from app.ask import search

        results = search("query")

        assert isinstance(results, list)
        assert len(results) == 2

    @patch("app.ask.get_compression_retriever")
    @patch("app.ask.get_vectorstore")
    def test_returns_dicts_with_content_and_source(self, mock_vs, mock_ccr):
        mock_ccr.return_value.invoke.return_value = [
            Document(page_content="text1", metadata={"source": "src1"}),
        ]
        from app.ask import search

        results = search("query")

        for r in results:
            assert "content" in r
            assert "source" in r
            assert isinstance(r["content"], str)
            assert isinstance(r["source"], str)

    @patch("app.ask.get_compression_retriever")
    @patch("app.ask.get_vectorstore")
    def test_calls_retriever_with_query(self, mock_vs, mock_ccr):
        mock_ccr.return_value.invoke.return_value = []
        from app.ask import search

        search("テスト質問")

        mock_ccr.return_value.invoke.assert_called_once_with("テスト質問")

    @patch("app.ask.get_compression_retriever")
    @patch("app.ask.get_vectorstore")
    def test_creates_base_retriever_with_search_k(self, mock_vs, mock_ccr):
        mock_ccr.return_value.invoke.return_value = []
        from app.ask import search

        search("query")

        mock_vs.return_value.as_retriever.assert_called_once_with(search_kwargs={"k": 10})

    @patch("app.ask.get_compression_retriever")
    @patch("app.ask.get_vectorstore")
    def test_wraps_with_compression_retriever(self, mock_vs, mock_ccr):
        mock_ccr.return_value.invoke.return_value = []
        from app.ask import search

        search("query")

        base_retriever = mock_vs.return_value.as_retriever.return_value
        mock_ccr.assert_called_once_with(base_retriever)

    @patch("app.ask.get_compression_retriever")
    @patch("app.ask.get_vectorstore")
    def test_search_with_single_result(self, mock_vs, mock_ccr):
        mock_ccr.return_value.invoke.return_value = [
            Document(page_content="single doc", metadata={"source": "src1"}),
        ]
        from app.ask import search

        results = search("query")
        assert len(results) == 1
        assert results[0]["content"] == "single doc"

    @patch("app.ask.get_compression_retriever")
    @patch("app.ask.get_vectorstore")
    def test_search_empty_results(self, mock_vs, mock_ccr):
        mock_ccr.return_value.invoke.return_value = []
        from app.ask import search

        results = search("query")
        assert results == []

    @patch("app.ask.get_compression_retriever")
    @patch("app.ask.get_vectorstore")
    def test_extracts_content_from_page_content(self, mock_vs, mock_ccr):
        mock_ccr.return_value.invoke.return_value = [
            Document(page_content="actual content", metadata={"source": "doc.pdf:p1"}),
        ]
        from app.ask import search

        results = search("query")
        assert results[0]["content"] == "actual content"
        assert results[0]["source"] == "doc.pdf:p1"


class TestMain:
    @patch("builtins.print")
    @patch("app.graph.get_graph")
    @patch("app.ask.sys")
    def test_uses_graph_invoke(self, mock_sys, mock_get_graph, mock_print):
        mock_sys.argv = ["ask.py", "質問テスト"]
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "回答テスト",
            "sources": ["doc.pdf:p1", "data.csv:r1"],
        }
        from app.ask import main

        main()

        mock_get_graph.return_value.invoke.assert_called_once_with({"query": "質問テスト"})

    @patch("builtins.print")
    @patch("app.graph.get_graph")
    @patch("app.ask.sys")
    def test_prints_answer(self, mock_sys, mock_get_graph, mock_print):
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
    @patch("app.graph.get_graph")
    @patch("app.ask.sys")
    def test_prints_sources(self, mock_sys, mock_get_graph, mock_print):
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
    @patch("app.graph.get_graph")
    @patch("app.ask.sys")
    def test_passes_query_from_argv(self, mock_sys, mock_get_graph, mock_print):
        mock_sys.argv = ["ask.py", "テスト質問"]
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "回答",
            "sources": [],
        }
        from app.ask import main

        main()

        mock_get_graph.return_value.invoke.assert_called_once_with({"query": "テスト質問"})

    @patch("builtins.print")
    @patch("app.graph.get_graph")
    @patch("app.ask.sys")
    def test_prints_answer_and_sources_sections(self, mock_sys, mock_get_graph, mock_print):
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
