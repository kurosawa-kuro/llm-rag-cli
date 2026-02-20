from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
import pytest


@pytest.fixture(autouse=True)
def reset_graph():
    import app.graph
    app.graph._graph = None
    yield
    app.graph._graph = None


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


class TestRetrieveNode:
    @patch("app.graph.get_vectorstore")
    def test_returns_documents_key(self, mock_vs):
        mock_retriever = MagicMock()
        mock_vs.return_value.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            Document(page_content="doc1", metadata={"source": "s1"}),
        ]
        from app.graph import retrieve

        result = retrieve({"query": "test"})

        assert "documents" in result
        assert len(result["documents"]) == 1

    @patch("app.graph.get_vectorstore")
    def test_calls_retriever_with_query(self, mock_vs):
        mock_retriever = MagicMock()
        mock_vs.return_value.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []
        from app.graph import retrieve

        retrieve({"query": "テスト質問"})

        mock_retriever.invoke.assert_called_once_with("テスト質問")

    @patch("app.graph.get_vectorstore")
    def test_uses_search_k(self, mock_vs):
        mock_retriever = MagicMock()
        mock_vs.return_value.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []
        from app.graph import retrieve

        retrieve({"query": "test"})

        mock_vs.return_value.as_retriever.assert_called_once_with(search_kwargs={"k": 10})


class TestRerankNode:
    @patch("app.graph.get_reranker")
    def test_returns_reranked_documents(self, mock_reranker):
        docs = [
            Document(page_content="doc1", metadata={"source": "s1"}),
            Document(page_content="doc2", metadata={"source": "s2"}),
        ]
        mock_reranker.return_value.compress_documents.return_value = docs
        from app.graph import rerank_node

        state = {"query": "test", "documents": docs}
        result = rerank_node(state)

        assert "reranked_documents" in result
        assert len(result["reranked_documents"]) == 2

    @patch("app.graph.get_reranker")
    def test_calls_compress_documents_with_query(self, mock_reranker):
        docs = [Document(page_content="doc1", metadata={"source": "s1"})]
        mock_reranker.return_value.compress_documents.return_value = docs
        from app.graph import rerank_node

        state = {"query": "my query", "documents": docs}
        rerank_node(state)

        mock_reranker.return_value.compress_documents.assert_called_once_with(docs, "my query")

    @patch("app.graph.RERANK_TOP_K", 3)
    @patch("app.graph.get_reranker")
    def test_limits_to_rerank_top_k(self, mock_reranker):
        docs = [Document(page_content=f"doc{i}", metadata={"source": f"s{i}"}) for i in range(5)]
        mock_reranker.return_value.compress_documents.return_value = docs
        from app.graph import rerank_node

        state = {"query": "test", "documents": docs}
        result = rerank_node(state)

        assert len(result["reranked_documents"]) == 3


class TestGenerateNode:
    @patch("app.graph.get_llm")
    def test_builds_japanese_prompt(self, mock_llm):
        mock_llm.return_value.invoke.return_value = "回答テスト"
        from app.graph import generate_node

        docs = [Document(page_content="context", metadata={"source": "s1"})]
        state = {"query": "質問テスト", "reranked_documents": docs}
        result = generate_node(state)

        prompt = result["prompt"]
        assert "以下の情報を基に回答してください" in prompt
        assert "質問:" in prompt
        assert "回答:" in prompt

    @patch("app.graph.get_llm")
    def test_returns_answer(self, mock_llm):
        mock_llm.return_value.invoke.return_value = "回答テスト"
        from app.graph import generate_node

        docs = [Document(page_content="context", metadata={"source": "s1"})]
        state = {"query": "質問", "reranked_documents": docs}
        result = generate_node(state)

        assert result["answer"] == "回答テスト"

    @patch("app.graph.get_llm")
    def test_returns_sources(self, mock_llm):
        mock_llm.return_value.invoke.return_value = "回答"
        from app.graph import generate_node

        docs = [
            Document(page_content="c1", metadata={"source": "doc.pdf:p1"}),
            Document(page_content="c2", metadata={"source": "data.csv:r1"}),
        ]
        state = {"query": "質問", "reranked_documents": docs}
        result = generate_node(state)

        assert result["sources"] == ["doc.pdf:p1", "data.csv:r1"]

    @patch("app.graph.get_llm")
    def test_returns_contexts(self, mock_llm):
        mock_llm.return_value.invoke.return_value = "回答"
        from app.graph import generate_node

        docs = [Document(page_content="context text", metadata={"source": "s1"})]
        state = {"query": "質問", "reranked_documents": docs}
        result = generate_node(state)

        assert result["contexts"] == ["context text"]

    @patch("app.graph.get_llm")
    def test_prompt_includes_context(self, mock_llm):
        mock_llm.return_value.invoke.return_value = "回答"
        from app.graph import generate_node

        docs = [Document(page_content="important context", metadata={"source": "s1"})]
        state = {"query": "テスト", "reranked_documents": docs}
        result = generate_node(state)

        assert "important context" in result["prompt"]

    @patch("app.graph.get_llm")
    def test_calls_llm_invoke(self, mock_llm):
        mock_llm.return_value.invoke.return_value = "回答"
        from app.graph import generate_node

        docs = [Document(page_content="context", metadata={"source": "s1"})]
        state = {"query": "質問", "reranked_documents": docs}
        generate_node(state)

        mock_llm.return_value.invoke.assert_called_once()


class TestBuildRagGraph:
    def test_graph_compiles(self):
        from app.graph import build_rag_graph

        graph = build_rag_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        from app.graph import build_rag_graph

        graph = build_rag_graph()
        node_names = list(graph.nodes.keys())
        assert "retrieve" in node_names
        assert "rerank" in node_names
        assert "generate" in node_names


class TestGetGraph:
    def test_returns_compiled_graph(self):
        from app.graph import get_graph

        graph = get_graph()
        assert graph is not None

    def test_singleton_returns_same_instance(self):
        from app.graph import get_graph

        g1 = get_graph()
        g2 = get_graph()
        assert g1 is g2


class TestRetrieveNodeEdgeCases:
    @patch("app.graph.get_vectorstore")
    def test_empty_query(self, mock_vs):
        mock_retriever = MagicMock()
        mock_vs.return_value.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []
        from app.graph import retrieve

        result = retrieve({"query": ""})

        assert result["documents"] == []
        mock_retriever.invoke.assert_called_once_with("")


class TestRerankNodeEdgeCases:
    @patch("app.graph.get_reranker")
    def test_empty_documents_list(self, mock_reranker):
        mock_reranker.return_value.compress_documents.return_value = []
        from app.graph import rerank_node

        state = {"query": "test", "documents": []}
        result = rerank_node(state)

        assert result["reranked_documents"] == []

    @patch("app.graph.get_reranker")
    def test_documents_without_source_metadata(self, mock_reranker):
        docs = [Document(page_content="text", metadata={})]
        mock_reranker.return_value.compress_documents.return_value = docs
        from app.graph import rerank_node

        state = {"query": "test", "documents": docs}
        result = rerank_node(state)

        assert len(result["reranked_documents"]) == 1


class TestGenerateNodeEdgeCases:
    @patch("app.graph.get_llm")
    def test_empty_reranked_documents(self, mock_llm):
        mock_llm.return_value.invoke.return_value = "回答なし"
        from app.graph import generate_node

        state = {"query": "テスト", "reranked_documents": []}
        result = generate_node(state)

        assert result["answer"] == "回答なし"
        assert result["contexts"] == []
        assert result["sources"] == []

    @patch("app.graph.get_llm")
    def test_document_without_source_returns_empty_string(self, mock_llm):
        mock_llm.return_value.invoke.return_value = "回答"
        from app.graph import generate_node

        docs = [Document(page_content="context", metadata={})]
        state = {"query": "テスト", "reranked_documents": docs}
        result = generate_node(state)

        assert result["sources"] == [""]

    @patch("app.graph.get_llm")
    def test_llm_raises_propagates(self, mock_llm):
        mock_llm.return_value.invoke.side_effect = RuntimeError("model error")
        from app.graph import generate_node

        docs = [Document(page_content="context", metadata={"source": "s"})]
        state = {"query": "テスト", "reranked_documents": docs}

        with pytest.raises(RuntimeError, match="model error"):
            generate_node(state)
