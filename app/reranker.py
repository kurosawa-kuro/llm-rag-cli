from app.config import RERANKER_MODEL

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query, docs, top_k=3):
    if not docs:
        return []
    pairs = [[query, doc["content"]] for doc in docs]
    scores = get_model().predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]
