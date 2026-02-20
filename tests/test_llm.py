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
    @patch.dict(sys.modules, {"langchain_community": MagicMock(), "langchain_community.llms": MagicMock()})
    def test_initializes_with_correct_params(self):
        import app.llm
        app.llm._llm = None

        from langchain_community.llms import LlamaCpp
        app.llm.get_llm()
        LlamaCpp.assert_called_once_with(
            model_path="./models/llama-2-7b.Q4_K_M.gguf",
            n_ctx=2048,
            max_tokens=300,
            verbose=False,
        )

    @patch.dict(sys.modules, {"langchain_community": MagicMock(), "langchain_community.llms": MagicMock()})
    def test_returns_llm_instance(self):
        import app.llm
        app.llm._llm = None

        from langchain_community.llms import LlamaCpp
        llm = app.llm.get_llm()
        assert llm == LlamaCpp.return_value


class TestGenerate:
    @patch("app.llm.get_llm")
    def test_calls_invoke_with_prompt(self, mock_get_llm):
        mock_get_llm.return_value.invoke.return_value = "テスト回答です。"
        from app.llm import generate

        generate("test prompt")
        mock_get_llm.return_value.invoke.assert_called_once_with("test prompt")

    @patch("app.llm.get_llm")
    def test_returns_text_from_invoke(self, mock_get_llm):
        mock_get_llm.return_value.invoke.return_value = "テスト回答です。"
        from app.llm import generate

        result = generate("test prompt")
        assert result == "テスト回答です。"

    @patch("app.llm.get_llm")
    def test_returns_string_type(self, mock_get_llm):
        mock_get_llm.return_value.invoke.return_value = "テスト回答です。"
        from app.llm import generate

        result = generate("test")
        assert isinstance(result, str)

    @patch("app.llm.get_llm")
    def test_generate_with_japanese_prompt(self, mock_get_llm):
        mock_get_llm.return_value.invoke.return_value = "回答結果"
        from app.llm import generate

        prompt = "以下の情報を基に回答してください:\n\nコンテキスト\n\n質問:テスト\n回答:"
        result = generate(prompt)
        mock_get_llm.return_value.invoke.assert_called_once_with(prompt)
        assert result == "回答結果"

    @patch("app.llm.get_llm")
    def test_generate_with_long_prompt(self, mock_get_llm):
        mock_get_llm.return_value.invoke.return_value = "answer"
        from app.llm import generate

        long_prompt = "word " * 500
        generate(long_prompt)
        mock_get_llm.return_value.invoke.assert_called_once_with(long_prompt)


class TestSingleton:
    @patch.dict(sys.modules, {"langchain_community": MagicMock(), "langchain_community.llms": MagicMock()})
    def test_get_llm_returns_same_instance_on_second_call(self):
        import app.llm
        app.llm._llm = None

        llm1 = app.llm.get_llm()
        llm2 = app.llm.get_llm()
        assert llm1 is llm2

    @patch.dict(sys.modules, {"langchain_community": MagicMock(), "langchain_community.llms": MagicMock()})
    def test_get_llm_only_loads_once(self):
        import app.llm
        app.llm._llm = None

        from langchain_community.llms import LlamaCpp
        app.llm.get_llm()
        app.llm.get_llm()
        app.llm.get_llm()
        LlamaCpp.assert_called_once()


class TestGenerateEdgeCases:
    @patch("app.llm.get_llm")
    def test_empty_prompt(self, mock_get_llm):
        mock_get_llm.return_value.invoke.return_value = ""
        from app.llm import generate

        result = generate("")
        mock_get_llm.return_value.invoke.assert_called_once_with("")
        assert result == ""

    @patch("app.llm.get_llm")
    def test_llm_invoke_raises_propagates(self, mock_get_llm):
        mock_get_llm.return_value.invoke.side_effect = RuntimeError("model failed")
        from app.llm import generate

        with pytest.raises(RuntimeError, match="model failed"):
            generate("test")
