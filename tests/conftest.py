from unittest.mock import MagicMock, patch
import numpy as np
import pytest


@pytest.fixture
def mock_db_connection():
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn, cur


@pytest.fixture
def fake_embeddings():
    return np.random.rand(3, 384).astype(np.float32)


@pytest.fixture
def mock_llm_response():
    return {"choices": [{"text": "テスト回答です。"}]}


@pytest.fixture
def mock_vectorstore():
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock()
    return vs


@pytest.fixture(autouse=True)
def reset_container():
    import app.container
    app.container._container = None
    yield
    app.container._container = None


@pytest.fixture
def mock_documents():
    from langchain_core.documents import Document
    return [
        Document(page_content="doc1 content", metadata={"source": "file.pdf:p1", "chunk_index": 0}),
        Document(page_content="doc2 content", metadata={"source": "data.csv:r1", "chunk_index": 0}),
    ]


# --- 統合テスト用 fixture ---


def _check_db_connection():
    """PostgreSQL に接続できるか確認し、不可なら pytest.skip する。"""
    from sqlalchemy import create_engine, text
    from app.config import CONNECTION_STRING

    engine = create_engine(CONNECTION_STRING)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL is not available")
    finally:
        engine.dispose()


@pytest.fixture
def test_embeddings():
    """統合テスト用のダミー埋め込み（FakeEmbeddings 384次元）"""
    from langchain_core.embeddings import FakeEmbeddings
    return FakeEmbeddings(size=384)


@pytest.fixture
def test_vectorstore(test_embeddings):
    """統合テスト用の実PGVector（test_documents コレクション）。テスト後にコレクション削除。"""
    _check_db_connection()

    from langchain_postgres import PGVector
    from app.config import CONNECTION_STRING

    vs = PGVector(
        embeddings=test_embeddings,
        collection_name="test_documents",
        connection=CONNECTION_STRING,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    yield vs
    vs.delete_collection()


# --- heavy統合テスト用 fixture（実Embeddings） ---


@pytest.fixture
def real_vectorstore():
    """実HuggingFaceEmbeddings + 実PGVectorを使うheavy統合テスト用。テスト後にコレクション削除。"""
    _check_db_connection()

    import uuid
    from langchain_postgres import PGVector
    from app.embeddings import create_embeddings
    from app.config import CONNECTION_STRING

    collection_name = f"test_heavy_{uuid.uuid4().hex[:8]}"
    embeddings = create_embeddings()

    vs = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=CONNECTION_STRING,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    yield vs
    try:
        vs.delete_collection()
    except Exception:
        pass
