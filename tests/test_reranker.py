import sys
from unittest.mock import patch, MagicMock
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_model():
    import app.reranker
    app.reranker._model = None
    yield
    app.reranker._model = None


class TestGetModel:
    @patch.dict(sys.modules, {"sentence_transformers": MagicMock()})
    def test_loads_correct_model(self):
        import app.reranker
        app.reranker._model = None

        from sentence_transformers import CrossEncoder
        app.reranker.get_model()
        CrossEncoder.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @patch.dict(sys.modules, {"sentence_transformers": MagicMock()})
    def test_returns_model_instance(self):
        import app.reranker
        app.reranker._model = None

        from sentence_transformers import CrossEncoder
        model = app.reranker.get_model()
        assert model == CrossEncoder.return_value

    @patch.dict(sys.modules, {"sentence_transformers": MagicMock()})
    def test_singleton_only_loads_once(self):
        import app.reranker
        app.reranker._model = None

        from sentence_transformers import CrossEncoder
        app.reranker.get_model()
        app.reranker.get_model()
        CrossEncoder.assert_called_once()


class TestRerank:
    @patch("app.reranker.get_model")
    def test_returns_top_k_documents(self, mock_get_model):
        mock_get_model.return_value.predict.return_value = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        docs = [{"content": f"doc{i}", "source": f"src{i}"} for i in range(5)]
        from app.reranker import rerank

        result = rerank("query", docs, top_k=3)

        assert len(result) == 3

    @patch("app.reranker.get_model")
    def test_returns_documents_sorted_by_score_descending(self, mock_get_model):
        mock_get_model.return_value.predict.return_value = np.array([0.1, 0.9, 0.5])
        docs = [
            {"content": "low", "source": "s0"},
            {"content": "high", "source": "s1"},
            {"content": "mid", "source": "s2"},
        ]
        from app.reranker import rerank

        result = rerank("query", docs, top_k=3)

        assert result[0]["content"] == "high"
        assert result[1]["content"] == "mid"
        assert result[2]["content"] == "low"

    @patch("app.reranker.get_model")
    def test_calls_predict_with_query_doc_pairs(self, mock_get_model):
        mock_get_model.return_value.predict.return_value = np.array([0.5, 0.3])
        docs = [
            {"content": "doc A", "source": "s0"},
            {"content": "doc B", "source": "s1"},
        ]
        from app.reranker import rerank

        rerank("my query", docs, top_k=2)

        pairs = mock_get_model.return_value.predict.call_args[0][0]
        assert pairs == [["my query", "doc A"], ["my query", "doc B"]]

    @patch("app.reranker.get_model")
    def test_top_k_greater_than_docs_returns_all(self, mock_get_model):
        mock_get_model.return_value.predict.return_value = np.array([0.5, 0.3])
        docs = [
            {"content": "doc A", "source": "s0"},
            {"content": "doc B", "source": "s1"},
        ]
        from app.reranker import rerank

        result = rerank("query", docs, top_k=10)

        assert len(result) == 2

    @patch("app.reranker.get_model")
    def test_empty_docs_returns_empty(self, mock_get_model):
        from app.reranker import rerank

        result = rerank("query", [], top_k=3)

        assert result == []

    @patch("app.reranker.get_model")
    def test_preserves_all_dict_fields(self, mock_get_model):
        mock_get_model.return_value.predict.return_value = np.array([0.9])
        docs = [{"content": "text", "source": "file.pdf:p1"}]
        from app.reranker import rerank

        result = rerank("query", docs, top_k=1)

        assert result[0]["content"] == "text"
        assert result[0]["source"] == "file.pdf:p1"

    @patch("app.reranker.get_model")
    def test_single_doc_returns_that_doc(self, mock_get_model):
        mock_get_model.return_value.predict.return_value = np.array([0.8])
        docs = [{"content": "only doc", "source": "src1"}]
        from app.reranker import rerank

        result = rerank("query", docs, top_k=3)
        assert len(result) == 1
        assert result[0]["content"] == "only doc"

    @patch("app.reranker.get_model")
    def test_top_k_1_returns_highest_scored(self, mock_get_model):
        mock_get_model.return_value.predict.return_value = np.array([0.2, 0.8, 0.5])
        docs = [
            {"content": "low", "source": "s0"},
            {"content": "high", "source": "s1"},
            {"content": "mid", "source": "s2"},
        ]
        from app.reranker import rerank

        result = rerank("query", docs, top_k=1)
        assert len(result) == 1
        assert result[0]["content"] == "high"

    @patch("app.reranker.get_model")
    def test_identical_scores_returns_top_k(self, mock_get_model):
        mock_get_model.return_value.predict.return_value = np.array([0.5, 0.5, 0.5, 0.5])
        docs = [{"content": f"doc{i}", "source": f"s{i}"} for i in range(4)]
        from app.reranker import rerank

        result = rerank("query", docs, top_k=2)
        assert len(result) == 2

    @patch("app.reranker.get_model")
    def test_returns_list_type(self, mock_get_model):
        mock_get_model.return_value.predict.return_value = np.array([0.5])
        docs = [{"content": "doc", "source": "s"}]
        from app.reranker import rerank

        result = rerank("query", docs, top_k=1)
        assert isinstance(result, list)
