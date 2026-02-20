def build_prompt(query: str, contexts: list[str]) -> str:
    context_text = "\n".join(contexts)
    return (
        "以下の情報のみを基に、質問に簡潔に回答してください。"
        "追加の質問や回答は生成しないでください。\n\n"
        f"情報:\n{context_text}\n\n"
        f"質問:{query}\n回答:"
    )
