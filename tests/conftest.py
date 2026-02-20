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


@pytest.fixture
def mock_documents():
    from langchain_core.documents import Document
    return [
        Document(page_content="doc1 content", metadata={"source": "file.pdf:p1", "chunk_index": 0}),
        Document(page_content="doc2 content", metadata={"source": "data.csv:r1", "chunk_index": 0}),
    ]
