import json
import os
import re
from unittest.mock import patch, MagicMock, mock_open
import pytest


class TestEvalQuestions:
    def test_eval_questions_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        assert os.path.exists(path)

    def test_eval_questions_is_valid_json_list(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_eval_questions_has_minimum_count(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data) >= 10

    def test_each_question_has_required_fields(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        for i, q in enumerate(data):
            assert "query" in q, f"Question {i} missing 'query'"
            assert "expected_source" in q, f"Question {i} missing 'expected_source'"
            assert "expected_keywords" in q, f"Question {i} missing 'expected_keywords'"

    def test_expected_source_matches_naming_convention(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        pattern = re.compile(r"^.+\.(csv|pdf):(r|p)\d+$")
        for i, q in enumerate(data):
            assert pattern.match(q["expected_source"]), \
                f"Question {i} source '{q['expected_source']}' doesn't match pattern"

    def test_expected_keywords_is_nonempty_list(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
        with open(path) as f:
            data = json.load(f)
        for i, q in enumerate(data):
            assert isinstance(q["expected_keywords"], list)
            assert len(q["expected_keywords"]) >= 1


class TestLoadQuestions:
    def test_loads_json_from_file(self):
        questions = [{"query": "q", "expected_source": "s", "expected_keywords": ["k"]}]
        m = mock_open(read_data=json.dumps(questions))
        with patch("builtins.open", m):
            from app.evaluate import load_questions

            result = load_questions("dummy.json")
        assert result == questions

    def test_returns_list(self):
        m = mock_open(read_data="[]")
        with patch("builtins.open", m):
            from app.evaluate import load_questions

            result = load_questions("dummy.json")
        assert isinstance(result, list)


class TestEvaluateSingle:
    def test_returns_retrieval_hit_when_source_found(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "パスワードをリセットしてください",
            "sources": ["faq.csv:r1"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single(
            "パスワードを忘れた", "faq.csv:r1", ["パスワード", "リセット"],
            mock_graph,
        )
        assert result["retrieval_hit"] is True

    def test_returns_retrieval_miss_when_source_not_found(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "関係ない回答",
            "sources": ["products.csv:r1"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single(
            "パスワードを忘れた", "faq.csv:r1", ["パスワード"],
            mock_graph,
        )
        assert result["retrieval_hit"] is False

    def test_returns_faithfulness_score(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "パスワードをリセット",
            "sources": ["faq.csv:r1"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single(
            "q", "faq.csv:r1", ["パスワード", "リセット", "メール"],
            mock_graph,
        )
        assert result["faithfulness"] == pytest.approx(2.0 / 3.0)

    def test_returns_exact_match_true_when_all_keywords_present(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "パスワードをリセットするにはメールアドレスを入力",
            "sources": ["faq.csv:r1"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single(
            "パスワードを忘れた", "faq.csv:r1", ["パスワード", "リセット", "メールアドレス"],
            mock_graph,
        )
        assert result["exact_match"] is True

    def test_returns_exact_match_false_when_keyword_missing(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "パスワードの変更",
            "sources": ["faq.csv:r1"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single(
            "パスワードを忘れた", "faq.csv:r1", ["パスワード", "リセット", "メールアドレス"],
            mock_graph,
        )
        assert result["exact_match"] is False

    def test_returns_latency_as_float(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "answer",
            "sources": ["s"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single("q", "s", ["answer"], mock_graph)
        assert isinstance(result["latency"], float)
        assert result["latency"] >= 0

    def test_calls_graph_invoke_with_query(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "a",
            "sources": ["s"],
        }
        from app.evaluate import evaluate_single

        evaluate_single("my query", "s", ["a"], mock_graph)
        mock_graph.invoke.assert_called_once_with({"query": "my query"})

    def test_uses_answer_from_graph_result(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "specific answer text",
            "sources": ["s"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single("my query", "s", ["specific"], mock_graph)
        assert result["answer"] == "specific answer text"


class TestRunEvaluation:
    def test_returns_list_of_results(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "パスワード リセット",
            "sources": ["faq.csv:r1"],
        }
        questions = [
            {"query": "q1", "expected_source": "faq.csv:r1", "expected_keywords": ["パスワード"]},
            {"query": "q2", "expected_source": "faq.csv:r2", "expected_keywords": ["アカウント"]},
        ]
        from app.evaluate import run_evaluation

        results = run_evaluation(questions, mock_graph)
        assert len(results) == 2

    def test_aggregates_retrieval_hits(self):
        call_count = {"n": 0}

        def mock_invoke(input_dict):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"answer": "keyword1", "sources": ["faq.csv:r1"]}
            return {"answer": "keyword1", "sources": ["other"]}

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = mock_invoke
        questions = [
            {"query": "q1", "expected_source": "faq.csv:r1", "expected_keywords": ["keyword1"]},
            {"query": "q2", "expected_source": "faq.csv:r2", "expected_keywords": ["keyword1"]},
        ]
        from app.evaluate import run_evaluation

        results = run_evaluation(questions, mock_graph)
        hits = sum(1 for r in results if r["retrieval_hit"])
        assert hits == 1


class TestPrintReport:
    @patch("builtins.print")
    def test_prints_config_header(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        results = [{"query": "q", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"}]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "500" in printed
        assert "100" in printed

    @patch("builtins.print")
    def test_prints_retrieval_percentage(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        results = [
            {"query": "q1", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"},
            {"query": "q2", "retrieval_hit": False, "faithfulness": 0.5, "exact_match": False, "latency": 0.3, "answer": "a"},
        ]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "50.0%" in printed

    @patch("builtins.print")
    def test_prints_faithfulness_percentage(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        results = [
            {"query": "q1", "retrieval_hit": True, "faithfulness": 0.8, "exact_match": False, "latency": 0.5, "answer": "a"},
            {"query": "q2", "retrieval_hit": True, "faithfulness": 0.6, "exact_match": False, "latency": 0.3, "answer": "a"},
        ]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "70.0%" in printed

    @patch("builtins.print")
    def test_prints_average_latency(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        results = [
            {"query": "q1", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 1.0, "answer": "a"},
            {"query": "q2", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 3.0, "answer": "a"},
        ]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "2.0" in printed

    @patch("builtins.print")
    def test_prints_rerank_status(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        results = [{"query": "q", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"}]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Re-rank" in printed


class TestEvaluateMain:
    @patch("app.evaluate.print_report")
    @patch("app.evaluate.run_evaluation")
    @patch("app.evaluate.load_questions")
    @patch("app.evaluate.get_container")
    @patch("app.evaluate.get_graph")
    def test_main_calls_pipeline(self, mock_get_graph, mock_get_container,
                                  mock_load, mock_run, mock_print):
        mock_load.return_value = [
            {"query": "q", "expected_source": "s", "expected_keywords": ["k"]}
        ]
        mock_run.return_value = [
            {"query": "q", "retrieval_hit": True, "faithfulness": 1.0,
             "exact_match": True, "latency": 0.5, "answer": "a"}
        ]
        from app.evaluate import main

        main()

        mock_load.assert_called_once()
        mock_run.assert_called_once()
        mock_print.assert_called_once()

    @patch("app.evaluate.print_report")
    @patch("app.evaluate.run_evaluation")
    @patch("app.evaluate.load_questions")
    @patch("app.evaluate.get_container")
    @patch("app.evaluate.get_graph")
    def test_main_passes_graph(self, mock_get_graph, mock_get_container,
                                mock_load, mock_run, mock_print):
        mock_load.return_value = []
        mock_run.return_value = []
        from app.evaluate import main

        main()

        call_args = mock_run.call_args[0]
        assert call_args[1] == mock_get_graph.return_value


class TestRunEvaluationExtended:
    def test_preserves_order(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "answer",
            "sources": ["s"],
        }
        questions = [
            {"query": "q1", "expected_source": "s", "expected_keywords": ["answer"]},
            {"query": "q2", "expected_source": "s", "expected_keywords": ["answer"]},
            {"query": "q3", "expected_source": "s", "expected_keywords": ["answer"]},
        ]
        from app.evaluate import run_evaluation

        results = run_evaluation(questions, mock_graph)
        assert results[0]["query"] == "q1"
        assert results[1]["query"] == "q2"
        assert results[2]["query"] == "q3"

    def test_each_result_has_all_fields(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "answer",
            "sources": ["faq.csv:r1"],
        }
        questions = [
            {"query": "q", "expected_source": "faq.csv:r1", "expected_keywords": ["answer"]},
        ]
        from app.evaluate import run_evaluation

        results = run_evaluation(questions, mock_graph)
        r = results[0]
        assert "query" in r
        assert "retrieval_hit" in r
        assert "faithfulness" in r
        assert "exact_match" in r
        assert "latency" in r
        assert "answer" in r

    def test_calls_graph_for_each_question(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "a",
            "sources": ["s"],
        }
        questions = [
            {"query": "q1", "expected_source": "s", "expected_keywords": ["a"]},
            {"query": "q2", "expected_source": "s", "expected_keywords": ["a"]},
        ]
        from app.evaluate import run_evaluation

        run_evaluation(questions, mock_graph)
        assert mock_graph.invoke.call_count == 2


class TestEvaluateSingleExtended:
    def test_returns_answer_field(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "generated answer",
            "sources": ["s"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single("q", "s", ["answer"], mock_graph)
        assert result["answer"] == "generated answer"

    def test_with_multiple_sources(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "answer with keyword1",
            "sources": ["s1", "s2", "s3"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single("q", "s2", ["keyword1"], mock_graph)
        assert result["retrieval_hit"] is True
        assert result["faithfulness"] == 1.0

    def test_query_field_in_result(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "a",
            "sources": ["s"],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single("my specific query", "s", ["a"], mock_graph)
        assert result["query"] == "my specific query"


class TestPrintReportExtended:
    @patch("builtins.print")
    def test_prints_question_count(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        results = [
            {"query": "q1", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"},
            {"query": "q2", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"},
            {"query": "q3", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"},
        ]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "3" in printed

    @patch("builtins.print")
    def test_prints_100_percent_retrieval(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        results = [
            {"query": "q1", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"},
            {"query": "q2", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"},
        ]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "100.0%" in printed

    @patch("builtins.print")
    def test_prints_rerank_on_when_top_k_positive(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        results = [{"query": "q", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"}]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "ON" in printed

    @patch("builtins.print")
    def test_prints_rerank_off_when_top_k_zero(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 0}
        results = [{"query": "q", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"}]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "OFF" in printed

    @patch("builtins.print")
    def test_prints_search_k(self, mock_print):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        results = [{"query": "q", "retrieval_hit": True, "faithfulness": 1.0, "exact_match": True, "latency": 0.5, "answer": "a"}]
        print_report(results, config)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "10" in printed


class TestEvaluateEdgeCases:
    def test_load_questions_nonexistent_file_raises_error(self):
        from app.evaluate import load_questions

        with pytest.raises(FileNotFoundError):
            load_questions("/nonexistent/path.json")

    def test_load_questions_invalid_json_raises_error(self):
        from app.evaluate import load_questions

        m = mock_open(read_data="not valid json{{{")
        with patch("builtins.open", m):
            with pytest.raises(json.JSONDecodeError):
                load_questions("dummy.json")

    def test_print_report_empty_results_raises_zero_division(self):
        from app.evaluate import print_report

        config = {"CHUNK_SIZE": 500, "CHUNK_OVERLAP": 100, "SEARCH_K": 10, "RERANK_TOP_K": 3}
        with pytest.raises(ZeroDivisionError):
            print_report([], config)

    def test_run_evaluation_empty_questions_returns_empty(self):
        mock_graph = MagicMock()
        from app.evaluate import run_evaluation

        results = run_evaluation([], mock_graph)
        assert results == []
        mock_graph.invoke.assert_not_called()

    def test_evaluate_single_empty_sources(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "no context answer",
            "sources": [],
        }
        from app.evaluate import evaluate_single

        result = evaluate_single("q", "faq.csv:r1", ["keyword"], mock_graph)
        assert result["retrieval_hit"] is False
        assert result["answer"] == "no context answer"

    def test_evaluate_single_graph_raises_propagates(self):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("LLM error")
        from app.evaluate import evaluate_single

        with pytest.raises(RuntimeError, match="LLM error"):
            evaluate_single("q", "s", ["k"], mock_graph)


class TestMakefileTarget:
    def test_makefile_has_evaluate_target(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Makefile")
        with open(path) as f:
            content = f.read()
        assert "evaluate:" in content
        assert "evaluate.py" in content
