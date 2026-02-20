from langchain_postgres import PGVector
from app.config import CONNECTION_STRING, COLLECTION_NAME


def create_vectorstore(embeddings):
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
