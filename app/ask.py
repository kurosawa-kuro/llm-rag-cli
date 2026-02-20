import sys
from app.db import get_vectorstore
from app.llm import generate
from app.reranker import get_compression_retriever
from app.config import SEARCH_K


def search(query):
    vectorstore = get_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
    retriever = get_compression_retriever(base_retriever)
    docs = retriever.invoke(query)
    return [{"content": doc.page_content, "source": doc.metadata.get("source", "")} for doc in docs]


def main():
    query = sys.argv[1]
    from app.graph import get_graph
    graph = get_graph()
    result = graph.invoke({"query": query})

    print("\n=== Answer ===\n")
    print(result["answer"])
    print("\n=== Sources ===\n")
    for source in result.get("sources", []):
        print(f"- {source}")


if __name__ == "__main__":
    main()
