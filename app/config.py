import os


def get_db_config():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "rag"),
        "password": os.getenv("DB_PASSWORD", "rag"),
        "dbname": os.getenv("DB_NAME", "rag"),
    }


DB_CONFIG = get_db_config()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL_PATH = "./models/llama-2-7b.Q4_K_M.gguf"
