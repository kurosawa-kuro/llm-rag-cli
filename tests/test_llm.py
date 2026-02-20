import sys
from unittest.mock import patch, MagicMock


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
            stop=["質問:", "\n\n"],
            verbose=False,
        )

    @patch.dict(sys.modules, {"langchain_community": MagicMock(), "langchain_community.llms": MagicMock()})
    def test_returns_llm_instance(self):
        import rag.components.llm
        from langchain_community.llms import LlamaCpp
        llm = rag.components.llm.create_llm()
        assert llm == LlamaCpp.return_value
