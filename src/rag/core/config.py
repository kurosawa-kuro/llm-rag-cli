import os
from pathlib import Path

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_settings():
    settings_path = _PROJECT_ROOT / "env" / "config" / "setting.yaml"
    with open(settings_path) as f:
        return yaml.safe_load(f)


_settings = _load_settings()


def get_db_config():
    db = _settings["db"]
    return {
        "host": os.getenv("DB_HOST", db["host"]),
        "user": os.getenv("DB_USER", db["user"]),
        "password": os.getenv("DB_PASSWORD", db["password"]),
        "dbname": os.getenv("DB_NAME", db["name"]),
    }


EMBED_MODEL = _settings["models"]["embed_model"]

LLM_MODEL_PATH = _settings["models"]["llm_model_path"]

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", _settings["chunking"]["chunk_size"]))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", _settings["chunking"]["chunk_overlap"]))

RERANKER_MODEL = _settings["models"]["reranker_model"]

SEARCH_K = int(os.getenv("SEARCH_K", _settings["search"]["search_k"]))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", _settings["search"]["rerank_top_k"]))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", _settings["search"]["score_threshold"]))

LLM_N_CTX = _settings["llm"]["n_ctx"]
LLM_MAX_TOKENS = _settings["llm"]["max_tokens"]


def get_connection_string():
    c = get_db_config()
    port = _settings["db"]["port"]
    return f"postgresql+psycopg://{c['user']}:{c['password']}@{c['host']}:{port}/{c['dbname']}"


CONNECTION_STRING = get_connection_string()
COLLECTION_NAME = _settings["collection_name"]
