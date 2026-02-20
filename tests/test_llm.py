import sys
from unittest.mock import patch, MagicMock
import pytest


class TestCreateLlm:
    @patch.dict(sys.modules, {"langchain_community": MagicMock(), "langchain_community.llms": MagicMock()})
    def test_initializes_with_correct_params(self):
        import rag.components.llm
        from langchain_community.llms import LlamaCpp
        rag.components.llm.create_llm()
        LlamaCpp.assert_called_once_with(
            model_path="./models/llama-2-7b.Q4_K_M.gguf",
            n_ctx=2048,
            max_tokens=300,
            verbose=False,
        )

    @patch.dict(sys.modules, {"langchain_community": MagicMock(), "langchain_community.llms": MagicMock()})
    def test_returns_llm_instance(self):
        import rag.components.llm
        from langchain_community.llms import LlamaCpp
        llm = rag.components.llm.create_llm()
        assert llm == LlamaCpp.return_value


class TestGenerate:
    @patch("rag.components.llm.create_llm")
    def test_calls_invoke_with_prompt(self, mock_create_llm):
        mock_create_llm.return_value.invoke.return_value = "テスト回答です。"
        from rag.components.llm import generate

        generate("test prompt")
        mock_create_llm.return_value.invoke.assert_called_once_with("test prompt")

    @patch("rag.components.llm.create_llm")
    def test_returns_text_from_invoke(self, mock_create_llm):
        mock_create_llm.return_value.invoke.return_value = "テスト回答です。"
        from rag.components.llm import generate

        result = generate("test prompt")
        assert result == "テスト回答です。"

    @patch("rag.components.llm.create_llm")
    def test_returns_string_type(self, mock_create_llm):
        mock_create_llm.return_value.invoke.return_value = "テスト回答です。"
        from rag.components.llm import generate

        result = generate("test")
        assert isinstance(result, str)

    @patch("rag.components.llm.create_llm")
    def test_generate_with_japanese_prompt(self, mock_create_llm):
        mock_create_llm.return_value.invoke.return_value = "回答結果"
        from rag.components.llm import generate

        prompt = "以下の情報を基に回答してください:\n\nコンテキスト\n\n質問:テスト\n回答:"
        result = generate(prompt)
        mock_create_llm.return_value.invoke.assert_called_once_with(prompt)
        assert result == "回答結果"

    @patch("rag.components.llm.create_llm")
    def test_generate_with_long_prompt(self, mock_create_llm):
        mock_create_llm.return_value.invoke.return_value = "answer"
        from rag.components.llm import generate

        long_prompt = "word " * 500
        generate(long_prompt)
        mock_create_llm.return_value.invoke.assert_called_once_with(long_prompt)


class TestGenerateEdgeCases:
    @patch("rag.components.llm.create_llm")
    def test_empty_prompt(self, mock_create_llm):
        mock_create_llm.return_value.invoke.return_value = ""
        from rag.components.llm import generate

        result = generate("")
        mock_create_llm.return_value.invoke.assert_called_once_with("")
        assert result == ""

    @patch("rag.components.llm.create_llm")
    def test_llm_invoke_raises_propagates(self, mock_create_llm):
        mock_create_llm.return_value.invoke.side_effect = RuntimeError("model failed")
        from rag.components.llm import generate

        with pytest.raises(RuntimeError, match="model failed"):
            generate("test")
