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
