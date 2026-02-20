import pytest
from langchain_core.documents import Document


@pytest.mark.integration
class TestDatabaseConnection:
    def test_vectorstore_connects(self, test_vectorstore):
        """PGVector が PostgreSQL に接続できることを確認"""
        assert test_vectorstore is not None

    def test_add_documents(self, test_vectorstore):
        """ドキュメントを追加できることを確認"""
        docs = [
            Document(page_content="テスト文書1", metadata={"source": "test.pdf:p1", "chunk_index": 0}),
            Document(page_content="テスト文書2", metadata={"source": "test.csv:r1", "chunk_index": 0}),
        ]
        ids = test_vectorstore.add_documents(docs)
        assert len(ids) == 2

    def test_similarity_search_returns_results(self, test_vectorstore):
        """追加したドキュメントが検索で返されることを確認"""
        docs = [
            Document(page_content="RAGシステムの概要", metadata={"source": "doc.pdf:p1", "chunk_index": 0}),
        ]
        test_vectorstore.add_documents(docs)
        results = test_vectorstore.similarity_search("RAG", k=1)
        assert len(results) >= 1

    def test_similarity_search_k_parameter(self, test_vectorstore):
        """k パラメータで取得件数を制御できることを確認"""
        docs = [
            Document(page_content=f"文書{i}", metadata={"source": f"doc.pdf:p{i}", "chunk_index": 0})
            for i in range(5)
        ]
        test_vectorstore.add_documents(docs)
        results = test_vectorstore.similarity_search("文書", k=3)
        assert len(results) == 3

    def test_metadata_preserved(self, test_vectorstore):
        """メタデータ（source, chunk_index）が保存・取得できることを確認"""
        docs = [
            Document(
                page_content="メタデータテスト",
                metadata={"source": "faq.csv:r1", "chunk_index": 2},
            ),
        ]
        test_vectorstore.add_documents(docs)
        results = test_vectorstore.similarity_search("メタデータ", k=1)
        assert results[0].metadata["source"] == "faq.csv:r1"
        assert results[0].metadata["chunk_index"] == 2

    def test_empty_collection_returns_empty(self, test_vectorstore):
        """空コレクションの検索は空リストを返すことを確認"""
        results = test_vectorstore.similarity_search("anything", k=5)
        assert results == []

    def test_delete_by_ids(self, test_vectorstore):
        """ドキュメントをIDで削除できることを確認"""
        docs = [
            Document(page_content="削除テスト", metadata={"source": "del.pdf:p1", "chunk_index": 0}),
        ]
        ids = test_vectorstore.add_documents(docs)
        test_vectorstore.delete(ids)
        results = test_vectorstore.similarity_search("削除テスト", k=1)
        assert len(results) == 0
