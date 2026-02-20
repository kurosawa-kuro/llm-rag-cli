import json
from app.db import get_vectorstore
from app.reranker import get_compression_retriever
from app.llm import generate
from app.prompting import build_prompt
from app.metrics import retrieval_at_k, faithfulness, exact_match, measure_latency
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_K, RERANK_TOP_K


def _search(query):
    vectorstore = get_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
    retriever = get_compression_retriever(base_retriever)
    docs = retriever.invoke(query)
    return [{"content": doc.page_content, "source": doc.metadata.get("source", "")} for doc in docs]


def load_questions(path="data/eval_questions.json"):
    with open(path) as f:
        return json.load(f)


def evaluate_single(query, expected_source, expected_keywords, search_fn, generate_fn):
    def _run():
        results = search_fn(query)
        contexts = [r["content"] for r in results]
        prompt = build_prompt(query, contexts)
        answer = generate_fn(prompt)
        return results, answer

    (results, answer), latency = measure_latency(_run)

    return {
        "query": query,
        "retrieval_hit": retrieval_at_k(results, expected_source),
        "faithfulness": faithfulness(answer, expected_keywords),
        "exact_match": exact_match(answer, expected_keywords),
        "latency": latency,
        "answer": answer,
    }


def run_evaluation(questions, search_fn, generate_fn):
    results = []
    for q in questions:
        result = evaluate_single(
            query=q["query"],
            expected_source=q["expected_source"],
            expected_keywords=q["expected_keywords"],
            search_fn=search_fn,
            generate_fn=generate_fn,
        )
        results.append(result)
    return results


def print_report(results, config):
    total = len(results)
    retrieval_pct = sum(1 for r in results if r["retrieval_hit"]) / total * 100
    faithfulness_pct = sum(r["faithfulness"] for r in results) / total * 100
    exact_match_pct = sum(1 for r in results if r["exact_match"]) / total * 100
    avg_latency = sum(r["latency"] for r in results) / total

    print("\n=== Evaluation Report ===\n")
    print(f"Chunk={config['CHUNK_SIZE']} overlap={config['CHUNK_OVERLAP']}")
    print(f"Top-k={config['SEARCH_K']}")
    print(f"Re-rank={'ON' if config['RERANK_TOP_K'] > 0 else 'OFF'} (top_k={config['RERANK_TOP_K']})")
    print()
    print(f"Retrieval@k: {retrieval_pct:.1f}%")
    print(f"Faithfulness: {faithfulness_pct:.1f}%")
    print(f"Exact Match: {exact_match_pct:.1f}%")
    print(f"Latency: {avg_latency:.1f}s")
    print(f"\nQuestions evaluated: {total}")


def main():
    questions = load_questions()
    config = {
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "SEARCH_K": SEARCH_K,
        "RERANK_TOP_K": RERANK_TOP_K,
    }
    results = run_evaluation(questions, _search, generate)
    print_report(results, config)


if __name__ == "__main__":
    main()
