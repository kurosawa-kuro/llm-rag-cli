import json
import sys
from rag.pipeline.graph import get_graph
from rag.core.container import get_container
from rag.evaluation.metrics import (
    retrieval_at_k, faithfulness, exact_match, measure_latency,
    context_relevance, retrieval_mrr,
)
from rag.core.config import CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_K, RERANK_TOP_K, SCORE_THRESHOLD


def load_questions(path="data/eval_questions.json"):
    with open(path) as f:
        return json.load(f)


def evaluate_single(query, expected_source, expected_keywords, graph):
    def _run():
        return graph.invoke({"query": query})

    result, latency = measure_latency(_run)

    sources = result.get("sources", [])
    answer = result.get("answer", "")

    return {
        "query": query,
        "retrieval_hit": retrieval_at_k(sources, expected_source),
        "faithfulness": faithfulness(answer, expected_keywords),
        "exact_match": exact_match(answer, expected_keywords),
        "latency": latency,
        "answer": answer,
    }


def run_evaluation(questions, graph):
    results = []
    for q in questions:
        result = evaluate_single(
            query=q["query"],
            expected_source=q["expected_source"],
            expected_keywords=q["expected_keywords"],
            graph=graph,
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


# --- Retrieval-only evaluation ---


def evaluate_single_retrieval(query, expected_source, expected_keywords, container):
    def _run():
        return container.retrieval_strategy.retrieve(query)

    docs, latency = measure_latency(_run)
    sources = [doc.metadata.get("source", "") for doc in docs]

    return {
        "query": query,
        "retrieval_hit": retrieval_at_k(sources, expected_source),
        "context_relevance": context_relevance(docs, expected_keywords),
        "mrr": retrieval_mrr(docs, expected_source),
        "retrieved_count": len(docs),
        "latency": latency,
    }


def run_retrieval_evaluation(questions, container):
    results = []
    for q in questions:
        result = evaluate_single_retrieval(
            query=q["query"],
            expected_source=q["expected_source"],
            expected_keywords=q["expected_keywords"],
            container=container,
        )
        results.append(result)
    return results


def print_retrieval_report(results, config):
    total = len(results)
    retrieval_pct = sum(1 for r in results if r["retrieval_hit"]) / total * 100
    relevance_pct = sum(r["context_relevance"] for r in results) / total * 100
    avg_mrr = sum(r["mrr"] for r in results) / total
    avg_count = sum(r["retrieved_count"] for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total

    print("\n=== Retrieval Evaluation Report ===\n")
    print(f"Chunk={config['CHUNK_SIZE']} overlap={config['CHUNK_OVERLAP']}")
    print(f"Top-k={config['SEARCH_K']} score_threshold={config['SCORE_THRESHOLD']}")
    print(f"Re-rank={'ON' if config['RERANK_TOP_K'] > 0 else 'OFF'} (top_k={config['RERANK_TOP_K']})")
    print()
    print(f"Retrieval@k: {retrieval_pct:.1f}%")
    print(f"Context Relevance: {relevance_pct:.1f}%")
    print(f"MRR: {avg_mrr:.3f}")
    print(f"Avg Retrieved: {avg_count:.1f} docs")
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
    graph = get_graph(container=get_container())
    results = run_evaluation(questions, graph)
    print_report(results, config)


def main_retrieval():
    questions = load_questions()
    config = {
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "SEARCH_K": SEARCH_K,
        "RERANK_TOP_K": RERANK_TOP_K,
        "SCORE_THRESHOLD": SCORE_THRESHOLD,
    }
    container = get_container()
    results = run_retrieval_evaluation(questions, container)
    print_retrieval_report(results, config)


if __name__ == "__main__":
    if "--retrieval-only" in sys.argv:
        main_retrieval()
    else:
        main()
