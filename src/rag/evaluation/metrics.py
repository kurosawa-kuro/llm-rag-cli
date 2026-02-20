import time


def retrieval_at_k(sources, expected_source):
    return any(s == expected_source for s in sources)


def faithfulness(answer, expected_keywords):
    if not expected_keywords:
        return 1.0
    found = sum(1 for kw in expected_keywords if kw in answer)
    return found / len(expected_keywords)


def exact_match(answer, expected_keywords):
    if not expected_keywords:
        return True
    return all(kw in answer for kw in expected_keywords)


def measure_latency(func):
    start = time.time()
    result = func()
    elapsed = time.time() - start
    return result, elapsed
