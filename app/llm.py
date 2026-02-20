from app.config import LLM_MODEL_PATH, LLM_N_CTX, LLM_MAX_TOKENS

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_community.llms import LlamaCpp
        _llm = LlamaCpp(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_N_CTX,
            max_tokens=LLM_MAX_TOKENS,
            verbose=False,
        )
    return _llm


def generate(prompt):
    return get_llm().invoke(prompt)
