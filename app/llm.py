from app.config import LLM_MODEL_PATH

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=2048)
    return _llm


def generate(prompt):
    output = get_llm()(prompt, max_tokens=300)
    return output["choices"][0]["text"]
