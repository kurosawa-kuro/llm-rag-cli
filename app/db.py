from langchain_postgres import PGVector
from app.config import CONNECTION_STRING, COLLECTION_NAME
from app.embeddings import get_embeddings

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PGVector(
            embeddings=get_embeddings(),
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )
    return _vectorstore


def init_db():
    get_vectorstore()
