from unittest.mock import patch
import pytest


class TestLoadSettings:
    def test_loads_yaml_file(self):
        from rag.core.config import _settings

        assert isinstance(_settings, dict)
        assert "db" in _settings
        assert "models" in _settings
        assert "llm" in _settings
        assert "chunking" in _settings
        assert "search" in _settings
        assert "collection_name" in _settings


class TestGetDbConfig:
    def test_default_values(self):
        with patch.dict("os.environ", {}, clear=True):
            from rag.core.config import get_db_config

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
            from rag.core.config import get_db_config

            config = get_db_config()
            assert config["host"] == "myhost"
            assert config["user"] == "myuser"
            assert config["password"] == "mypass"
            assert config["dbname"] == "mydb"


class TestConstants:
    def test_embed_model(self):
        from rag.core.config import EMBED_MODEL

        assert EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_llm_model_path(self):
        from rag.core.config import LLM_MODEL_PATH

        assert LLM_MODEL_PATH == "./models/llama-2-7b.Q4_K_M.gguf"

    def test_chunk_size_default(self):
        from rag.core.config import CHUNK_SIZE

        assert CHUNK_SIZE == 500

    def test_chunk_overlap_default(self):
        from rag.core.config import CHUNK_OVERLAP

        assert CHUNK_OVERLAP == 100

    def test_reranker_model(self):
        from rag.core.config import RERANKER_MODEL

        assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_search_k_default(self):
        from rag.core.config import SEARCH_K

        assert SEARCH_K == 10

    def test_rerank_top_k_default(self):
        from rag.core.config import RERANK_TOP_K

        assert RERANK_TOP_K == 3

    def test_llm_n_ctx(self):
        from rag.core.config import LLM_N_CTX

        assert LLM_N_CTX == 2048

    def test_llm_max_tokens(self):
        from rag.core.config import LLM_MAX_TOKENS

        assert LLM_MAX_TOKENS == 300

    def test_collection_name(self):
        from rag.core.config import COLLECTION_NAME

        assert COLLECTION_NAME == "documents"

    def test_connection_string_format(self):
        from rag.core.config import CONNECTION_STRING

        assert CONNECTION_STRING.startswith("postgresql+psycopg://")

    def test_connection_string_contains_default_values(self):
        from rag.core.config import CONNECTION_STRING

        assert "rag:rag@localhost" in CONNECTION_STRING


class TestConnectionString:
    def test_env_override(self):
        env = {"DB_HOST": "myhost", "DB_USER": "myuser", "DB_PASSWORD": "mypass", "DB_NAME": "mydb"}
        with patch.dict("os.environ", env, clear=True):
            from rag.core.config import get_connection_string

            cs = get_connection_string()
            assert "myuser:mypass@myhost" in cs
            assert "mydb" in cs

    def test_format(self):
        from rag.core.config import get_connection_string

        cs = get_connection_string()
        assert cs.startswith("postgresql+psycopg://")
        assert ":5432/" in cs


class TestChunkConfigEnvOverride:
    def test_chunk_size_env_override(self):
        import importlib
        import rag.core.config as config_mod
        with patch.dict("os.environ", {"CHUNK_SIZE": "1000"}):
            importlib.reload(config_mod)
            assert config_mod.CHUNK_SIZE == 1000
        # restore
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)

    def test_chunk_overlap_env_override(self):
        import importlib
        import rag.core.config as config_mod
        with patch.dict("os.environ", {"CHUNK_OVERLAP": "200"}):
            importlib.reload(config_mod)
            assert config_mod.CHUNK_OVERLAP == 200
        # restore
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)


class TestRerankerConfigEnvOverride:
    def test_search_k_env_override(self):
        import importlib
        import rag.core.config as config_mod
        with patch.dict("os.environ", {"SEARCH_K": "20"}):
            importlib.reload(config_mod)
            assert config_mod.SEARCH_K == 20
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)

    def test_rerank_top_k_env_override(self):
        import importlib
        import rag.core.config as config_mod
        with patch.dict("os.environ", {"RERANK_TOP_K": "5"}):
            importlib.reload(config_mod)
            assert config_mod.RERANK_TOP_K == 5
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)


class TestConfigValidation:
    def test_non_numeric_chunk_size_raises_value_error(self):
        import importlib
        import rag.core.config as config_mod
        with patch.dict("os.environ", {"CHUNK_SIZE": "abc"}):
            with pytest.raises(ValueError):
                importlib.reload(config_mod)
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)

    def test_non_numeric_search_k_raises_value_error(self):
        import importlib
        import rag.core.config as config_mod
        with patch.dict("os.environ", {"SEARCH_K": "not_a_number"}):
            with pytest.raises(ValueError):
                importlib.reload(config_mod)
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(config_mod)

    def test_empty_db_host_uses_empty_string(self):
        with patch.dict("os.environ", {"DB_HOST": ""}):
            from rag.core.config import get_db_config
            config = get_db_config()
            assert config["host"] == ""

    def test_connection_string_with_special_chars_in_password(self):
        env = {"DB_PASSWORD": "p@ss:word/123"}
        with patch.dict("os.environ", env):
            from rag.core.config import get_connection_string
            cs = get_connection_string()
            assert "p@ss:word/123" in cs
