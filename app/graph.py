from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from app.db import get_vectorstore
from app.llm import get_llm
from app.reranker import get_reranker
from app.config import SEARCH_K, RERANK_TOP_K


class RAGState(TypedDict):
    query: str
    documents: List[Document]
    reranked_documents: List[Document]
    contexts: List[str]
    prompt: str
    answer: str
    sources: List[str]


def retrieve(state: RAGState) -> dict:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
    docs = retriever.invoke(state["query"])
    return {"documents": docs}


def rerank_node(state: RAGState) -> dict:
    reranker = get_reranker()
    reranked = reranker.compress_documents(state["documents"], state["query"])
    return {"reranked_documents": list(reranked[:RERANK_TOP_K])}


def generate_node(state: RAGState) -> dict:
    contexts = [doc.page_content for doc in state["reranked_documents"]]
    sources = [doc.metadata.get("source", "") for doc in state["reranked_documents"]]
    prompt = f"以下の情報を基に回答してください:\n\n{contexts}\n\n質問:{state['query']}\n回答:"
    llm = get_llm()
    answer = llm.invoke(prompt)
    return {
        "contexts": contexts,
        "prompt": prompt,
        "answer": answer,
        "sources": sources,
    }


def build_rag_graph():
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph
