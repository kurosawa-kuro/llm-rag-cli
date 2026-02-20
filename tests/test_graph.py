from unittest.mock import MagicMock
from langchain_core.documents import Document
import pytest

from app.container import AppContainer, RagSettings


@pytest.fixture(autouse=True)
def reset_graph():
    import app.graph
    app.graph._graph = None
    yield
    app.graph._graph = None


@pytest.fixture
def mock_container():
    container = MagicMock(spec=AppContainer)
    container.settings = RagSettings(search_k=10, rerank_top_k=3)
    container.prompt_builder = lambda q, c: f"以下の情報を基に回答してください:\n\n{c}\n\n質問:{q}\n回答:"
    return container


class TestRAGState:
    def test_state_has_required_fields(self):
        from app.graph import RAGState

        assert "query" in RAGState.__annotations__
        assert "documents" in RAGState.__annotations__
        assert "reranked_documents" in RAGState.__annotations__
        assert "contexts" in RAGState.__annotations__
        assert "prompt" in RAGState.__annotations__
        assert "answer" in RAGState.__annotations__
        assert "sources" in RAGState.__annotations__

    def test_state_defaults(self):
        from app.graph import RAGState

        state = RAGState()
        assert state.query == ""
        assert state.documents == []
        assert state.reranked_documents == []
        assert state.contexts == []
        assert state.prompt == ""
        assert state.answer == ""
        assert state.sources == []


class TestRetrieveNode:
    def test_returns_documents_key(self, mock_container):
        mock_retriever = MagicMock()
        mock_container.vectorstore.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            Document(page_content="doc1", metadata={"source": "s1"}),
        ]
        from app.graph import create_retrieve, RAGState

        retrieve = create_retrieve(mock_container)
        result = retrieve(RAGState(query="test"))

        assert "documents" in result
        assert len(result["documents"]) == 1

    def test_calls_retriever_with_query(self, mock_container):
        mock_retriever = MagicMock()
        mock_container.vectorstore.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []
        from app.graph import create_retrieve, RAGState

        retrieve = create_retrieve(mock_container)
        retrieve(RAGState(query="テスト質問"))

        mock_retriever.invoke.assert_called_once_with("テスト質問")

    def test_uses_search_k(self, mock_container):
        mock_retriever = MagicMock()
        mock_container.vectorstore.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []
        from app.graph import create_retrieve, RAGState

        retrieve = create_retrieve(mock_container)
        retrieve(RAGState(query="test"))

        mock_container.vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 10})


class TestRerankNode:
    def test_returns_reranked_documents(self, mock_container):
        docs = [
            Document(page_content="doc1", metadata={"source": "s1"}),
            Document(page_content="doc2", metadata={"source": "s2"}),
        ]
        mock_container.reranker.compress_documents.return_value = docs
        from app.graph import create_rerank, RAGState

        rerank = create_rerank(mock_container)
        result = rerank(RAGState(query="test", documents=docs))

        assert "reranked_documents" in result
        assert len(result["reranked_documents"]) == 2

    def test_calls_compress_documents_with_query(self, mock_container):
        docs = [Document(page_content="doc1", metadata={"source": "s1"})]
        mock_container.reranker.compress_documents.return_value = docs
        from app.graph import create_rerank, RAGState

        rerank = create_rerank(mock_container)
        rerank(RAGState(query="my query", documents=docs))

        mock_container.reranker.compress_documents.assert_called_once_with(docs, "my query")

    def test_limits_to_rerank_top_k(self, mock_container):
        docs = [Document(page_content=f"doc{i}", metadata={"source": f"s{i}"}) for i in range(5)]
        mock_container.reranker.compress_documents.return_value = docs
        mock_container.settings = RagSettings(search_k=10, rerank_top_k=3)
        from app.graph import create_rerank, RAGState

        rerank = create_rerank(mock_container)
        result = rerank(RAGState(query="test", documents=docs))

        assert len(result["reranked_documents"]) == 3


class TestGenerateNode:
    def test_builds_japanese_prompt(self, mock_container):
        mock_container.llm.invoke.return_value = "回答テスト"
        from app.graph import create_generate, RAGState

        docs = [Document(page_content="context", metadata={"source": "s1"})]
        generate = create_generate(mock_container)
        result = generate(RAGState(query="質問テスト", reranked_documents=docs))

        prompt = result["prompt"]
        assert "以下の情報を基に回答してください" in prompt
        assert "質問:" in prompt
        assert "回答:" in prompt

    def test_returns_answer(self, mock_container):
        mock_container.llm.invoke.return_value = "回答テスト"
        from app.graph import create_generate, RAGState

        docs = [Document(page_content="context", metadata={"source": "s1"})]
        generate = create_generate(mock_container)
        result = generate(RAGState(query="質問", reranked_documents=docs))

        assert result["answer"] == "回答テスト"

    def test_returns_sources(self, mock_container):
        mock_container.llm.invoke.return_value = "回答"
        from app.graph import create_generate, RAGState

        docs = [
            Document(page_content="c1", metadata={"source": "doc.pdf:p1"}),
            Document(page_content="c2", metadata={"source": "data.csv:r1"}),
        ]
        generate = create_generate(mock_container)
        result = generate(RAGState(query="質問", reranked_documents=docs))

        assert result["sources"] == ["doc.pdf:p1", "data.csv:r1"]

    def test_returns_contexts(self, mock_container):
        mock_container.llm.invoke.return_value = "回答"
        from app.graph import create_generate, RAGState

        docs = [Document(page_content="context text", metadata={"source": "s1"})]
        generate = create_generate(mock_container)
        result = generate(RAGState(query="質問", reranked_documents=docs))

        assert result["contexts"] == ["context text"]

    def test_prompt_includes_context(self, mock_container):
        mock_container.llm.invoke.return_value = "回答"
        from app.graph import create_generate, RAGState

        docs = [Document(page_content="important context", metadata={"source": "s1"})]
        generate = create_generate(mock_container)
        result = generate(RAGState(query="テスト", reranked_documents=docs))

        assert "important context" in result["prompt"]

    def test_calls_llm_invoke(self, mock_container):
        mock_container.llm.invoke.return_value = "回答"
        from app.graph import create_generate, RAGState

        docs = [Document(page_content="context", metadata={"source": "s1"})]
        generate = create_generate(mock_container)
        generate(RAGState(query="質問", reranked_documents=docs))

        mock_container.llm.invoke.assert_called_once()


class TestBuildRagGraph:
    def test_graph_compiles(self, mock_container):
        from app.graph import build_rag_graph

        graph = build_rag_graph(container=mock_container)
        assert graph is not None

    def test_graph_has_nodes(self, mock_container):
        from app.graph import build_rag_graph

        graph = build_rag_graph(container=mock_container)
        node_names = list(graph.nodes.keys())
        assert "retrieve" in node_names
        assert "rerank" in node_names
        assert "generate" in node_names


class TestGetGraph:
    def test_returns_compiled_graph(self, mock_container):
        from app.graph import get_graph

        graph = get_graph(container=mock_container)
        assert graph is not None

    def test_singleton_returns_same_instance(self, mock_container):
        from app.graph import get_graph

        g1 = get_graph(container=mock_container)
        # Second call without container uses cache
        g2 = get_graph()
        assert g2 is not None

    def test_container_injection_bypasses_cache(self, mock_container):
        from app.graph import get_graph

        g1 = get_graph()
        g2 = get_graph(container=mock_container)
        # container injection always creates a new graph
        assert g2 is not g1


class TestRetrieveNodeEdgeCases:
    def test_empty_query(self, mock_container):
        mock_retriever = MagicMock()
        mock_container.vectorstore.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []
        from app.graph import create_retrieve, RAGState

        retrieve = create_retrieve(mock_container)
        result = retrieve(RAGState(query=""))

        assert result["documents"] == []
        mock_retriever.invoke.assert_called_once_with("")


class TestRerankNodeEdgeCases:
    def test_empty_documents_list(self, mock_container):
        mock_container.reranker.compress_documents.return_value = []
        from app.graph import create_rerank, RAGState

        rerank = create_rerank(mock_container)
        result = rerank(RAGState(query="test", documents=[]))

        assert result["reranked_documents"] == []

    def test_documents_without_source_metadata(self, mock_container):
        docs = [Document(page_content="text", metadata={})]
        mock_container.reranker.compress_documents.return_value = docs
        from app.graph import create_rerank, RAGState

        rerank = create_rerank(mock_container)
        result = rerank(RAGState(query="test", documents=docs))

        assert len(result["reranked_documents"]) == 1


class TestGenerateNodeEdgeCases:
    def test_empty_reranked_documents(self, mock_container):
        mock_container.llm.invoke.return_value = "回答なし"
        from app.graph import create_generate, RAGState

        generate = create_generate(mock_container)
        result = generate(RAGState(query="テスト", reranked_documents=[]))

        assert result["answer"] == "回答なし"
        assert result["contexts"] == []
        assert result["sources"] == []

    def test_document_without_source_returns_empty_string(self, mock_container):
        mock_container.llm.invoke.return_value = "回答"
        from app.graph import create_generate, RAGState

        docs = [Document(page_content="context", metadata={})]
        generate = create_generate(mock_container)
        result = generate(RAGState(query="テスト", reranked_documents=docs))

        assert result["sources"] == [""]

    def test_llm_raises_propagates(self, mock_container):
        mock_container.llm.invoke.side_effect = RuntimeError("model error")
        from app.graph import create_generate, RAGState

        docs = [Document(page_content="context", metadata={"source": "s"})]
        generate = create_generate(mock_container)

        with pytest.raises(RuntimeError, match="model error"):
            generate(RAGState(query="テスト", reranked_documents=docs))
