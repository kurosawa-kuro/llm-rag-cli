from app.config import EMBED_MODEL


def create_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def embed(texts):
    return create_embeddings().embed_documents(texts)
