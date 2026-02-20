import sys
from unittest.mock import patch, MagicMock
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_model():
    import app.embeddings
    app.embeddings._model = None
    yield
    app.embeddings._model = None


class TestGetModel:
    @patch.dict(sys.modules, {"sentence_transformers": MagicMock()})
    def test_loads_correct_model(self):
        import app.embeddings
        app.embeddings._model = None

        from sentence_transformers import SentenceTransformer
        app.embeddings.get_model()
        SentenceTransformer.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")

    @patch.dict(sys.modules, {"sentence_transformers": MagicMock()})
    def test_returns_model_instance(self):
        import app.embeddings
        app.embeddings._model = None

        from sentence_transformers import SentenceTransformer
        model = app.embeddings.get_model()
        assert model == SentenceTransformer.return_value


class TestEmbed:
    @patch("app.embeddings.get_model")
    def test_returns_encode_result(self, mock_get_model):
        fake_vectors = np.random.rand(2, 384).astype(np.float32)
        mock_get_model.return_value.encode.return_value = fake_vectors
        from app.embeddings import embed

        result = embed(["text1", "text2"])
        assert result.shape == (2, 384)

    @patch("app.embeddings.get_model")
    def test_calls_encode_with_texts(self, mock_get_model):
        mock_get_model.return_value.encode.return_value = np.zeros((1, 384))
        from app.embeddings import embed

        texts = ["hello"]
        embed(texts)
        mock_get_model.return_value.encode.assert_called_once_with(texts)

    @patch("app.embeddings.get_model")
    def test_multiple_texts(self, mock_get_model):
        fake_vectors = np.random.rand(3, 384).astype(np.float32)
        mock_get_model.return_value.encode.return_value = fake_vectors
        from app.embeddings import embed

        result = embed(["a", "b", "c"])
        assert result.shape == (3, 384)

    @patch("app.embeddings.get_model")
    def test_single_text_returns_correct_shape(self, mock_get_model):
        fake_vectors = np.random.rand(1, 384).astype(np.float32)
        mock_get_model.return_value.encode.return_value = fake_vectors
        from app.embeddings import embed

        result = embed(["hello"])
        assert result.shape == (1, 384)

    @patch("app.embeddings.get_model")
    def test_returns_numpy_array(self, mock_get_model):
        fake_vectors = np.random.rand(2, 384).astype(np.float32)
        mock_get_model.return_value.encode.return_value = fake_vectors
        from app.embeddings import embed

        result = embed(["a", "b"])
        assert isinstance(result, np.ndarray)

    @patch("app.embeddings.get_model")
    def test_embed_japanese_text(self, mock_get_model):
        fake_vectors = np.random.rand(2, 384).astype(np.float32)
        mock_get_model.return_value.encode.return_value = fake_vectors
        from app.embeddings import embed

        result = embed(["パスワードを忘れた", "料金プラン"])
        mock_get_model.return_value.encode.assert_called_once_with(["パスワードを忘れた", "料金プラン"])
        assert result.shape == (2, 384)


class TestSingleton:
    @patch.dict(sys.modules, {"sentence_transformers": MagicMock()})
    def test_get_model_returns_same_instance_on_second_call(self):
        import app.embeddings
        app.embeddings._model = None

        model1 = app.embeddings.get_model()
        model2 = app.embeddings.get_model()
        assert model1 is model2

    @patch.dict(sys.modules, {"sentence_transformers": MagicMock()})
    def test_get_model_only_loads_once(self):
        import app.embeddings
        app.embeddings._model = None

        from sentence_transformers import SentenceTransformer
        app.embeddings.get_model()
        app.embeddings.get_model()
        app.embeddings.get_model()
        SentenceTransformer.assert_called_once()
