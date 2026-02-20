import time
import pytest


class TestRetrievalAtK:
    def test_correct_source_in_results_returns_true(self):
        from app.metrics import retrieval_at_k

        results = [
            {"content": "...", "source": "faq.csv:r1"},
            {"content": "...", "source": "faq.csv:r2"},
        ]
        assert retrieval_at_k(results, "faq.csv:r1") is True

    def test_correct_source_not_in_results_returns_false(self):
        from app.metrics import retrieval_at_k

        results = [
            {"content": "...", "source": "faq.csv:r2"},
            {"content": "...", "source": "faq.csv:r3"},
        ]
        assert retrieval_at_k(results, "faq.csv:r1") is False

    def test_empty_results_returns_false(self):
        from app.metrics import retrieval_at_k

        assert retrieval_at_k([], "faq.csv:r1") is False

    def test_no_false_positive_on_partial_match(self):
        from app.metrics import retrieval_at_k

        results = [{"content": "...", "source": "faq.csv:r10"}]
        assert retrieval_at_k(results, "faq.csv:r1") is False


class TestFaithfulness:
    def test_all_keywords_present_returns_1(self):
        from app.metrics import faithfulness

        answer = "パスワードをリセットするにはメールアドレスを入力してください"
        keywords = ["パスワード", "リセット", "メールアドレス"]
        assert faithfulness(answer, keywords) == 1.0

    def test_no_keywords_present_returns_0(self):
        from app.metrics import faithfulness

        assert faithfulness("こんにちは", ["パスワード", "リセット"]) == 0.0

    def test_partial_keywords_returns_fraction(self):
        from app.metrics import faithfulness

        answer = "パスワードの変更が必要です"
        keywords = ["パスワード", "リセット", "メールアドレス"]
        assert faithfulness(answer, keywords) == pytest.approx(1.0 / 3.0)

    def test_empty_keywords_returns_1(self):
        from app.metrics import faithfulness

        assert faithfulness("any answer", []) == 1.0

    def test_empty_answer_returns_0(self):
        from app.metrics import faithfulness

        assert faithfulness("", ["keyword"]) == 0.0

    def test_case_sensitive_match(self):
        from app.metrics import faithfulness

        assert faithfulness("PostgreSQL is used", ["PostgreSQL"]) == 1.0
        assert faithfulness("postgresql is used", ["PostgreSQL"]) == 0.0


class TestExactMatch:
    def test_all_keywords_present_returns_true(self):
        from app.metrics import exact_match

        answer = "パスワードをリセットするにはメールアドレスを入力してください"
        keywords = ["パスワード", "リセット", "メールアドレス"]
        assert exact_match(answer, keywords) is True

    def test_missing_keyword_returns_false(self):
        from app.metrics import exact_match

        answer = "パスワードの変更が必要です"
        keywords = ["パスワード", "リセット", "メールアドレス"]
        assert exact_match(answer, keywords) is False

    def test_empty_keywords_returns_true(self):
        from app.metrics import exact_match

        assert exact_match("any answer", []) is True

    def test_empty_answer_returns_false(self):
        from app.metrics import exact_match

        assert exact_match("", ["keyword"]) is False

    def test_single_keyword_present(self):
        from app.metrics import exact_match

        assert exact_match("PostgreSQL is used", ["PostgreSQL"]) is True

    def test_single_keyword_missing(self):
        from app.metrics import exact_match

        assert exact_match("MySQL is used", ["PostgreSQL"]) is False


class TestMeasureLatency:
    def test_returns_function_result(self):
        from app.metrics import measure_latency

        result, _ = measure_latency(lambda: 42)
        assert result == 42

    def test_returns_elapsed_time(self):
        from app.metrics import measure_latency

        _, elapsed = measure_latency(lambda: time.sleep(0.1))
        assert elapsed >= 0.1
        assert elapsed < 0.5

    def test_elapsed_is_float(self):
        from app.metrics import measure_latency

        _, elapsed = measure_latency(lambda: None)
        assert isinstance(elapsed, float)

    def test_propagates_exception(self):
        from app.metrics import measure_latency

        def failing_fn():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            measure_latency(failing_fn)

    def test_returns_complex_result(self):
        from app.metrics import measure_latency

        result, _ = measure_latency(lambda: {"key": "value", "list": [1, 2, 3]})
        assert result == {"key": "value", "list": [1, 2, 3]}

    def test_elapsed_is_non_negative(self):
        from app.metrics import measure_latency

        _, elapsed = measure_latency(lambda: None)
        assert elapsed >= 0


class TestRetrievalAtKExtended:
    def test_multiple_results_with_match_at_end(self):
        from app.metrics import retrieval_at_k

        results = [
            {"content": "...", "source": "a.csv:r1"},
            {"content": "...", "source": "b.csv:r2"},
            {"content": "...", "source": "target.pdf:p1"},
        ]
        assert retrieval_at_k(results, "target.pdf:p1") is True

    def test_duplicate_sources_in_results(self):
        from app.metrics import retrieval_at_k

        results = [
            {"content": "...", "source": "faq.csv:r1"},
            {"content": "...", "source": "faq.csv:r1"},
        ]
        assert retrieval_at_k(results, "faq.csv:r1") is True

    def test_source_with_high_page_number(self):
        from app.metrics import retrieval_at_k

        results = [{"content": "...", "source": "doc.pdf:p100"}]
        assert retrieval_at_k(results, "doc.pdf:p100") is True

    def test_single_result_matches(self):
        from app.metrics import retrieval_at_k

        results = [{"content": "...", "source": "faq.csv:r1"}]
        assert retrieval_at_k(results, "faq.csv:r1") is True


class TestFaithfulnessExtended:
    def test_single_keyword_found(self):
        from app.metrics import faithfulness

        assert faithfulness("パスワードをリセット", ["パスワード"]) == 1.0

    def test_keyword_as_substring_in_longer_word(self):
        from app.metrics import faithfulness

        # "パスワード" is substring of "パスワードリセット"
        assert faithfulness("パスワードリセットが必要", ["パスワード"]) == 1.0

    def test_japanese_keywords(self):
        from app.metrics import faithfulness

        answer = "PostgreSQLとDockerを使用しています"
        keywords = ["PostgreSQL", "Docker"]
        assert faithfulness(answer, keywords) == 1.0

    def test_many_keywords_partial_match(self):
        from app.metrics import faithfulness

        answer = "料金プランの変更はダッシュボードから"
        keywords = ["料金プラン", "変更", "ダッシュボード", "メール"]
        assert faithfulness(answer, keywords) == pytest.approx(3.0 / 4.0)
