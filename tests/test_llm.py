import sys
from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture(autouse=True)
def reset_llm():
    import app.llm
    app.llm._llm = None
    yield
    app.llm._llm = None


class TestGetLlm:
    @patch.dict(sys.modules, {"llama_cpp": MagicMock()})
    def test_initializes_with_correct_params(self):
        import app.llm
        app.llm._llm = None

        from llama_cpp import Llama
        app.llm.get_llm()
        Llama.assert_called_once_with(
            model_path="./models/llama-2-7b.Q4_K_M.gguf",
            n_ctx=2048,
        )

    @patch.dict(sys.modules, {"llama_cpp": MagicMock()})
    def test_returns_llm_instance(self):
        import app.llm
        app.llm._llm = None

        from llama_cpp import Llama
        llm = app.llm.get_llm()
        assert llm == Llama.return_value


class TestGenerate:
    @patch("app.llm.get_llm")
    def test_passes_prompt_and_max_tokens(self, mock_get_llm, mock_llm_response):
        mock_get_llm.return_value.return_value = mock_llm_response
        from app.llm import generate

        generate("test prompt")
        mock_get_llm.return_value.assert_called_once_with("test prompt", max_tokens=300)

    @patch("app.llm.get_llm")
    def test_returns_text_from_choices(self, mock_get_llm, mock_llm_response):
        mock_get_llm.return_value.return_value = mock_llm_response
        from app.llm import generate

        result = generate("test prompt")
        assert result == "テスト回答です。"
