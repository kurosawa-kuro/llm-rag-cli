from app.config import LLM_MODEL_PATH, LLM_N_CTX, LLM_MAX_TOKENS


def create_llm():
    from langchain_community.llms import LlamaCpp
    return LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_N_CTX,
        max_tokens=LLM_MAX_TOKENS,
        verbose=False,
    )


def generate(prompt):
    return create_llm().invoke(prompt)
