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
