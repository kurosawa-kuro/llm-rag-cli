def build_prompt(query: str, contexts: list[str]) -> str:
    return f"以下の情報を基に回答してください:\n\n{contexts}\n\n質問:{query}\n回答:"
