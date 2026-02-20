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
