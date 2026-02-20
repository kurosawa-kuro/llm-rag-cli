import sys
from app.db import get_conn
from app.embeddings import embed
from app.llm import generate
from app.reranker import rerank
from app.config import SEARCH_K, RERANK_TOP_K


def search(query):
    vec = embed([query])[0].tolist()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content, source FROM documents ORDER BY embedding <-> %s LIMIT %s;",
                (vec, SEARCH_K)
            )
            candidates = [{"content": row[0], "source": row[1]} for row in cur.fetchall()]
    return rerank(query, candidates, RERANK_TOP_K)


def main():
    query = sys.argv[1]
    results = search(query)
    contexts = [r["content"] for r in results]
    prompt = f"以下の情報を基に回答してください:\n\n{contexts}\n\n質問:{query}\n回答:"
    answer = generate(prompt)
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Sources ===\n")
    for r in results:
        print(f"- {r['source']}")


if __name__ == "__main__":
    main()
