from unittest.mock import MagicMock
from langchain_core.documents import Document

from rag.core.interfaces import (
    RetrieverProtocol,
    VectorStoreProtocol,
    RerankerProtocol,
    LLMProtocol,
    RetrievalStrategyProtocol,
)


class TestRetrieverProtocol:
    def test_duck_type_compatible(self):
        obj = MagicMock()
        obj.invoke.return_value = [Document(page_content="a")]
        result = obj.invoke("query")
        assert isinstance(result, list)


class TestVectorStoreProtocol:
    def test_duck_type_compatible(self):
        obj = MagicMock()
        retriever = MagicMock()
        obj.as_retriever.return_value = retriever
        result = obj.as_retriever(search_kwargs={"k": 10})
        assert result is retriever


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
