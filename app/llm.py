from app.config import LLM_MODEL_PATH

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_community.llms import LlamaCpp
        _llm = LlamaCpp(
            model_path=LLM_MODEL_PATH,
            n_ctx=2048,
            max_tokens=300,
            verbose=False,
        )
    return _llm


def generate(prompt):
    return get_llm().invoke(prompt)
