import sys
from app.db import get_conn
from app.embeddings import embed
from app.llm import generate


def search(query, k=3):
    vec = embed([query])[0].tolist()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM documents ORDER BY embedding <-> %s LIMIT %s;",
                (vec, k)
            )
            return [row[0] for row in cur.fetchall()]


def main():
    query = sys.argv[1]
    contexts = search(query)
    prompt = f"以下の情報を基に回答してください:\n\n{contexts}\n\n質問:{query}\n回答:"
    answer = generate(prompt)
    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
