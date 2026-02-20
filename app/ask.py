import sys


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
