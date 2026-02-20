from app.config import EMBED_MODEL

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def embed(texts):
    return get_embeddings().embed_documents(texts)
