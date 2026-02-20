from unittest.mock import patch


class TestGetDbConfig:
    def test_default_values(self):
        with patch.dict("os.environ", {}, clear=True):
            from app.config import get_db_config

            config = get_db_config()
            assert config["host"] == "localhost"
            assert config["user"] == "rag"
            assert config["password"] == "rag"
            assert config["dbname"] == "rag"

    def test_env_override(self):
        env = {
            "DB_HOST": "myhost",
            "DB_USER": "myuser",
            "DB_PASSWORD": "mypass",
            "DB_NAME": "mydb",
        }
        with patch.dict("os.environ", env, clear=True):
            from app.config import get_db_config

            config = get_db_config()
            assert config["host"] == "myhost"
            assert config["user"] == "myuser"
            assert config["password"] == "mypass"
            assert config["dbname"] == "mydb"


class TestConstants:
    def test_embed_model(self):
        from app.config import EMBED_MODEL

        assert EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_llm_model_path(self):
        from app.config import LLM_MODEL_PATH

        assert LLM_MODEL_PATH == "./models/llama-2-7b.Q4_K_M.gguf"

    def test_chunk_size_default(self):
        from app.config import CHUNK_SIZE

        assert CHUNK_SIZE == 500

    def test_chunk_overlap_default(self):
        from app.config import CHUNK_OVERLAP

        assert CHUNK_OVERLAP == 100

    def test_reranker_model(self):
        from app.config import RERANKER_MODEL

        assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_search_k_default(self):
        from app.config import SEARCH_K

        assert SEARCH_K == 10

    def test_rerank_top_k_default(self):
        from app.config import RERANK_TOP_K

        assert RERANK_TOP_K == 3


class TestChunkConfigEnvOverride:
    def test_chunk_size_env_override(self):
        import importlib
        import app.config as config_mod
        with patch.dict("os.environ", {"CHUNK_SIZE": "1000"}):
            importlib.reload(config_mod)
            assert config_mod.CHUNK_SIZE == 1000
        # restore
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)

    def test_chunk_overlap_env_override(self):
        import importlib
        import app.config as config_mod
        with patch.dict("os.environ", {"CHUNK_OVERLAP": "200"}):
            importlib.reload(config_mod)
            assert config_mod.CHUNK_OVERLAP == 200
        # restore
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)


class TestRerankerConfigEnvOverride:
    def test_search_k_env_override(self):
        import importlib
        import app.config as config_mod
        with patch.dict("os.environ", {"SEARCH_K": "20"}):
            importlib.reload(config_mod)
            assert config_mod.SEARCH_K == 20
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)

    def test_rerank_top_k_env_override(self):
        import importlib
        import app.config as config_mod
        with patch.dict("os.environ", {"RERANK_TOP_K": "5"}):
            importlib.reload(config_mod)
            assert config_mod.RERANK_TOP_K == 5
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)
