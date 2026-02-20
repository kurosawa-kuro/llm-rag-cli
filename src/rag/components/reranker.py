from app.config import RERANKER_MODEL, RERANK_TOP_K


def create_reranker():
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    return CrossEncoderReranker(model=model, top_n=RERANK_TOP_K)


def get_compression_retriever(base_retriever):
    from langchain.retrievers import ContextualCompressionRetriever
    return ContextualCompressionRetriever(
        base_compressor=create_reranker(),
        base_retriever=base_retriever,
    )


def rerank(query, docs, top_k=3):
    if not docs:
        return []
    from langchain_core.documents import Document
    lc_docs = [Document(page_content=d["content"], metadata={"source": d.get("source", "")}) for d in docs]
    compressor = create_reranker()
    reranked = compressor.compress_documents(lc_docs, query)
    return [{"content": doc.page_content, "source": doc.metadata.get("source", "")} for doc in reranked[:top_k]]
