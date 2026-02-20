from unittest.mock import MagicMock
from langchain_core.documents import Document

from rag.core.interfaces import (
    VectorStoreProtocol,
    RerankerProtocol,
    LLMProtocol,
    RetrievalStrategyProtocol,
)


class TestVectorStoreProtocol:
    def test_duck_type_compatible(self):
        obj = MagicMock()
        obj.similarity_search_with_score.return_value = [(Document(page_content="a"), 0.3)]
        result = obj.similarity_search_with_score("query", k=10)
        assert isinstance(result, list)


class TestRerankerProtocol:
    def test_duck_type_compatible(self):
        obj = MagicMock()
        docs = [Document(page_content="a")]
        obj.compress_documents.return_value = docs
        result = obj.compress_documents(docs, "query")
        assert result == docs


class TestLLMProtocol:
    def test_duck_type_compatible(self):
        obj = MagicMock()
        obj.invoke.return_value = "answer"
        result = obj.invoke("prompt")
        assert result == "answer"


class TestRetrievalStrategyProtocol:
    def test_duck_type_compatible(self):
        obj = MagicMock()
        obj.retrieve.return_value = [Document(page_content="a")]
        result = obj.retrieve("query")
        assert isinstance(result, list)
