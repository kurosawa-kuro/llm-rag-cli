import time
import pytest
from langchain_core.documents import Document


class TestRetrievalAtK:
    def test_correct_source_in_results_returns_true(self):
        from rag.evaluation.metrics import retrieval_at_k

        sources = ["faq.csv:r1", "faq.csv:r2"]
        assert retrieval_at_k(sources, "faq.csv:r1") is True

    def test_correct_source_not_in_results_returns_false(self):
        from rag.evaluation.metrics import retrieval_at_k

        sources = ["faq.csv:r2", "faq.csv:r3"]
        assert retrieval_at_k(sources, "faq.csv:r1") is False

    def test_empty_results_returns_false(self):
        from rag.evaluation.metrics import retrieval_at_k

        assert retrieval_at_k([], "faq.csv:r1") is False

    def test_no_false_positive_on_partial_match(self):
        from rag.evaluation.metrics import retrieval_at_k

        sources = ["faq.csv:r10"]
        assert retrieval_at_k(sources, "faq.csv:r1") is False


class TestFaithfulness:
    def test_all_keywords_present_returns_1(self):
        from rag.evaluation.metrics import faithfulness

        answer = "パスワードをリセットするにはメールアドレスを入力してください"
        keywords = ["パスワード", "リセット", "メールアドレス"]
        assert faithfulness(answer, keywords) == 1.0

    def test_no_keywords_present_returns_0(self):
        from rag.evaluation.metrics import faithfulness

        assert faithfulness("こんにちは", ["パスワード", "リセット"]) == 0.0

    def test_partial_keywords_returns_fraction(self):
        from rag.evaluation.metrics import faithfulness

        answer = "パスワードの変更が必要です"
        keywords = ["パスワード", "リセット", "メールアドレス"]
        assert faithfulness(answer, keywords) == pytest.approx(1.0 / 3.0)

    def test_empty_keywords_returns_1(self):
        from rag.evaluation.metrics import faithfulness

        assert faithfulness("any answer", []) == 1.0

    def test_empty_answer_returns_0(self):
        from rag.evaluation.metrics import faithfulness

        assert faithfulness("", ["keyword"]) == 0.0

    def test_case_sensitive_match(self):
        from rag.evaluation.metrics import faithfulness

        assert faithfulness("PostgreSQL is used", ["PostgreSQL"]) == 1.0
        assert faithfulness("postgresql is used", ["PostgreSQL"]) == 0.0


class TestExactMatch:
    def test_all_keywords_present_returns_true(self):
        from rag.evaluation.metrics import exact_match

        answer = "パスワードをリセットするにはメールアドレスを入力してください"
        keywords = ["パスワード", "リセット", "メールアドレス"]
        assert exact_match(answer, keywords) is True

    def test_missing_keyword_returns_false(self):
        from rag.evaluation.metrics import exact_match

        answer = "パスワードの変更が必要です"
        keywords = ["パスワード", "リセット", "メールアドレス"]
        assert exact_match(answer, keywords) is False

    def test_empty_keywords_returns_true(self):
        from rag.evaluation.metrics import exact_match

        assert exact_match("any answer", []) is True

    def test_empty_answer_returns_false(self):
        from rag.evaluation.metrics import exact_match

        assert exact_match("", ["keyword"]) is False

    def test_single_keyword_present(self):
        from rag.evaluation.metrics import exact_match

        assert exact_match("PostgreSQL is used", ["PostgreSQL"]) is True

    def test_single_keyword_missing(self):
        from rag.evaluation.metrics import exact_match

        assert exact_match("MySQL is used", ["PostgreSQL"]) is False


class TestMeasureLatency:
    def test_returns_function_result(self):
        from rag.evaluation.metrics import measure_latency

        result, _ = measure_latency(lambda: 42)
        assert result == 42

    def test_returns_elapsed_time(self):
        from rag.evaluation.metrics import measure_latency

        _, elapsed = measure_latency(lambda: time.sleep(0.1))
        assert elapsed >= 0.1
        assert elapsed < 0.5

    def test_elapsed_is_float(self):
        from rag.evaluation.metrics import measure_latency

        _, elapsed = measure_latency(lambda: None)
        assert isinstance(elapsed, float)

    def test_propagates_exception(self):
        from rag.evaluation.metrics import measure_latency

        def failing_fn():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            measure_latency(failing_fn)

    def test_returns_complex_result(self):
        from rag.evaluation.metrics import measure_latency

        result, _ = measure_latency(lambda: {"key": "value", "list": [1, 2, 3]})
        assert result == {"key": "value", "list": [1, 2, 3]}

    def test_elapsed_is_non_negative(self):
        from rag.evaluation.metrics import measure_latency

        _, elapsed = measure_latency(lambda: None)
        assert elapsed >= 0


class TestRetrievalAtKEdgeCases:
    def test_none_results_raises_type_error(self):
        from rag.evaluation.metrics import retrieval_at_k

        with pytest.raises(TypeError):
            retrieval_at_k(None, "faq.csv:r1")

    def test_none_expected_source(self):
        from rag.evaluation.metrics import retrieval_at_k

        sources = ["faq.csv:r1"]
        assert retrieval_at_k(sources, None) is False


class TestFaithfulnessEdgeCases:
    def test_none_answer_raises_type_error(self):
        from rag.evaluation.metrics import faithfulness

        with pytest.raises(TypeError):
            faithfulness(None, ["keyword"])

    def test_none_keywords_returns_1(self):
        from rag.evaluation.metrics import faithfulness

        # None is falsy, so `not None` is True
        assert faithfulness("answer", None) == 1.0


class TestExactMatchEdgeCases:
    def test_none_answer_raises_type_error(self):
        from rag.evaluation.metrics import exact_match

        with pytest.raises(TypeError):
            exact_match(None, ["keyword"])

    def test_none_keywords_returns_true(self):
        from rag.evaluation.metrics import exact_match

        assert exact_match("answer", None) is True


class TestMeasureLatencyEdgeCases:
    def test_none_func_raises_type_error(self):
        from rag.evaluation.metrics import measure_latency

        with pytest.raises(TypeError):
            measure_latency(None)

    def test_func_returning_none(self):
        from rag.evaluation.metrics import measure_latency

        result, elapsed = measure_latency(lambda: None)
        assert result is None
        assert elapsed >= 0


class TestRetrievalAtKExtended:
    def test_multiple_results_with_match_at_end(self):
        from rag.evaluation.metrics import retrieval_at_k

        sources = ["a.csv:r1", "b.csv:r2", "target.pdf:p1"]
        assert retrieval_at_k(sources, "target.pdf:p1") is True

    def test_duplicate_sources_in_results(self):
        from rag.evaluation.metrics import retrieval_at_k

        sources = ["faq.csv:r1", "faq.csv:r1"]
        assert retrieval_at_k(sources, "faq.csv:r1") is True

    def test_source_with_high_page_number(self):
        from rag.evaluation.metrics import retrieval_at_k

        sources = ["doc.pdf:p100"]
        assert retrieval_at_k(sources, "doc.pdf:p100") is True

    def test_single_result_matches(self):
        from rag.evaluation.metrics import retrieval_at_k

        sources = ["faq.csv:r1"]
        assert retrieval_at_k(sources, "faq.csv:r1") is True


class TestFaithfulnessExtended:
    def test_single_keyword_found(self):
        from rag.evaluation.metrics import faithfulness

        assert faithfulness("パスワードをリセット", ["パスワード"]) == 1.0

    def test_keyword_as_substring_in_longer_word(self):
        from rag.evaluation.metrics import faithfulness

        # "パスワード" is substring of "パスワードリセット"
        assert faithfulness("パスワードリセットが必要", ["パスワード"]) == 1.0

    def test_japanese_keywords(self):
        from rag.evaluation.metrics import faithfulness

        answer = "PostgreSQLとDockerを使用しています"
        keywords = ["PostgreSQL", "Docker"]
        assert faithfulness(answer, keywords) == 1.0

    def test_many_keywords_partial_match(self):
        from rag.evaluation.metrics import faithfulness

        answer = "料金プランの変更はダッシュボードから"
        keywords = ["料金プラン", "変更", "ダッシュボード", "メール"]
        assert faithfulness(answer, keywords) == pytest.approx(3.0 / 4.0)


class TestContextRelevance:
    def test_all_keywords_in_documents(self):
        from rag.evaluation.metrics import context_relevance

        docs = [Document(page_content="パスワードをリセットするにはメールアドレスが必要")]
        assert context_relevance(docs, ["パスワード", "リセット", "メールアドレス"]) == 1.0

    def test_no_keywords_in_documents(self):
        from rag.evaluation.metrics import context_relevance

        docs = [Document(page_content="こんにちは")]
        assert context_relevance(docs, ["パスワード", "リセット"]) == 0.0

    def test_partial_keywords_in_documents(self):
        from rag.evaluation.metrics import context_relevance

        docs = [Document(page_content="パスワードの変更")]
        assert context_relevance(docs, ["パスワード", "リセット", "メールアドレス"]) == pytest.approx(1.0 / 3.0)

    def test_keywords_spread_across_documents(self):
        from rag.evaluation.metrics import context_relevance

        docs = [
            Document(page_content="パスワードの管理"),
            Document(page_content="リセット手順"),
        ]
        assert context_relevance(docs, ["パスワード", "リセット"]) == 1.0

    def test_empty_documents(self):
        from rag.evaluation.metrics import context_relevance

        assert context_relevance([], ["keyword"]) == 0.0

    def test_empty_keywords(self):
        from rag.evaluation.metrics import context_relevance

        docs = [Document(page_content="any content")]
        assert context_relevance(docs, []) == 1.0


class TestRetrievalMrr:
    def test_expected_source_at_first_position(self):
        from rag.evaluation.metrics import retrieval_mrr

        docs = [
            Document(page_content="a", metadata={"source": "faq.csv:r1"}),
            Document(page_content="b", metadata={"source": "faq.csv:r2"}),
        ]
        assert retrieval_mrr(docs, "faq.csv:r1") == 1.0

    def test_expected_source_at_second_position(self):
        from rag.evaluation.metrics import retrieval_mrr

        docs = [
            Document(page_content="a", metadata={"source": "faq.csv:r2"}),
            Document(page_content="b", metadata={"source": "faq.csv:r1"}),
        ]
        assert retrieval_mrr(docs, "faq.csv:r1") == 0.5

    def test_expected_source_at_third_position(self):
        from rag.evaluation.metrics import retrieval_mrr

        docs = [
            Document(page_content="a", metadata={"source": "other1"}),
            Document(page_content="b", metadata={"source": "other2"}),
            Document(page_content="c", metadata={"source": "faq.csv:r1"}),
        ]
        assert retrieval_mrr(docs, "faq.csv:r1") == pytest.approx(1.0 / 3.0)

    def test_expected_source_not_found(self):
        from rag.evaluation.metrics import retrieval_mrr

        docs = [Document(page_content="a", metadata={"source": "other"})]
        assert retrieval_mrr(docs, "faq.csv:r1") == 0.0

    def test_empty_documents(self):
        from rag.evaluation.metrics import retrieval_mrr

        assert retrieval_mrr([], "faq.csv:r1") == 0.0
