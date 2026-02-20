import pytest
from langchain_core.documents import Document


@pytest.mark.integration
@pytest.mark.heavy
class TestRealEmbeddingSimilarity:
    def test_fruit_query_returns_fruit_docs(self, real_vectorstore):
        """fruit クエリで果物関連ドキュメントが上位に来ることを確認"""
        docs = [
            Document(page_content="apple is a red fruit", metadata={"source": "a"}),
            Document(page_content="banana is a yellow fruit", metadata={"source": "b"}),
            Document(page_content="car engine maintenance guide", metadata={"source": "c"}),
        ]
        real_vectorstore.add_documents(docs)

        results = real_vectorstore.similarity_search("fruit", k=2)

        top_texts = [r.page_content for r in results]
        assert any("fruit" in text for text in top_texts)
        assert "car engine maintenance guide" not in top_texts[:1]

    def test_unrelated_query_does_not_match_fruit(self, real_vectorstore):
        """無関係クエリで果物ドキュメントが1位にならないことを確認"""
        docs = [
            Document(page_content="apple is a red fruit", metadata={"source": "a"}),
            Document(page_content="banana is a yellow fruit", metadata={"source": "b"}),
            Document(page_content="car engine maintenance guide", metadata={"source": "c"}),
        ]
        real_vectorstore.add_documents(docs)

        results = real_vectorstore.similarity_search("engine repair", k=1)

        assert results[0].page_content == "car engine maintenance guide"

    def test_retriever_with_search_k(self, real_vectorstore):
        """as_retriever 経由でも実Embeddingsで検索できることを確認"""
        docs = [
            Document(page_content="python programming language", metadata={"source": "p"}),
            Document(page_content="java programming language", metadata={"source": "j"}),
            Document(page_content="recipe for chocolate cake", metadata={"source": "r"}),
        ]
        real_vectorstore.add_documents(docs)

        retriever = real_vectorstore.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke("programming")

        top_texts = [r.page_content for r in results]
        assert any("programming" in text for text in top_texts)
