import sys
from app.graph import get_graph
from app.container import get_container


def main():
    query = sys.argv[1]
    graph = get_graph(container=get_container())
    result = graph.invoke({"query": query})

    print("\n=== Answer ===\n")
    print(result["answer"])
    print("\n=== Sources ===\n")
    for source in result.get("sources", []):
        print(f"- {source}")


if __name__ == "__main__":
    main()
