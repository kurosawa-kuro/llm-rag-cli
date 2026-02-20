from dataclasses import dataclass, field
from typing import List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END


@dataclass
class RAGState:
    query: str = ""
    reranked_documents: List[Document] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    prompt: str = ""
    answer: str = ""
    sources: List[str] = field(default_factory=list)


def create_retrieve(container):
    def retrieve(state: RAGState) -> dict:
        docs = container.retrieval_strategy.retrieve(state.query)
        return {"reranked_documents": docs}
    return retrieve


def create_generate(container):
    def generate_node(state: RAGState) -> dict:
        contexts = [doc.page_content for doc in state.reranked_documents]
        sources = list(dict.fromkeys(
            doc.metadata.get("source", "") for doc in state.reranked_documents
        ))
        if not contexts:
            return {
                "contexts": [],
                "prompt": "",
                "answer": "該当する情報が見つかりませんでした。",
                "sources": [],
            }
        prompt = container.prompt_builder(state.query, contexts)
        answer = container.llm.invoke(prompt)
        return {
            "contexts": contexts,
            "prompt": prompt,
            "answer": answer,
            "sources": sources,
        }
    return generate_node


def build_rag_graph(*, container=None):
    if container is None:
        from rag.core.container import get_container
        container = get_container()

    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", create_retrieve(container))
    workflow.add_node("generate", create_generate(container))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


_graph = None


def get_graph(*, container=None):
    global _graph
    if container is not None:
        return build_rag_graph(container=container)
    if _graph is None:
        _graph = build_rag_graph()
    return _graph
