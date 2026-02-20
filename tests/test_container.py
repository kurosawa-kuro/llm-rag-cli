from unittest.mock import MagicMock, patch
import pytest


class TestRagSettings:
    def test_default_values(self):
        from app.container import RagSettings

        settings = RagSettings()
        assert settings.search_k == 10
        assert settings.rerank_top_k == 3

    def test_custom_values(self):
        from app.container import RagSettings

        settings = RagSettings(search_k=20, rerank_top_k=5)
        assert settings.search_k == 20
        assert settings.rerank_top_k == 5


class TestAppContainer:
    def test_default_settings(self):
        from app.container import AppContainer, RagSettings

        container = AppContainer()
        assert isinstance(container.settings, RagSettings)
        assert container.settings.search_k == 10

    def test_custom_settings(self):
        from app.container import AppContainer, RagSettings

        settings = RagSettings(search_k=15, rerank_top_k=2)
        container = AppContainer(settings=settings)
        assert container.settings.search_k == 15
        assert container.settings.rerank_top_k == 2

    def test_injected_vectorstore(self):
        from app.container import AppContainer

        mock_vs = MagicMock()
        container = AppContainer(vectorstore=mock_vs)
        assert container.vectorstore is mock_vs

    def test_injected_reranker(self):
        from app.container import AppContainer

        mock_reranker = MagicMock()
        container = AppContainer(reranker=mock_reranker)
        assert container.reranker is mock_reranker

    def test_injected_llm(self):
        from app.container import AppContainer

        mock_llm = MagicMock()
        container = AppContainer(llm=mock_llm)
        assert container.llm is mock_llm

    def test_injected_prompt_builder(self):
        from app.container import AppContainer

        builder = lambda q, c: "mock"
        container = AppContainer(prompt_builder=builder)
        assert container.prompt_builder is builder

    def test_prompt_builder_default_is_callable(self):
        from app.container import AppContainer

        container = AppContainer()
        assert callable(container.prompt_builder)

    @patch("app.db.get_vectorstore")
    def test_vectorstore_lazy_loads(self, mock_get_vs):
        from app.container import AppContainer

        mock_get_vs.return_value = MagicMock()
        container = AppContainer()
        vs = container.vectorstore
        mock_get_vs.assert_called_once()
        assert vs is mock_get_vs.return_value

    @patch("app.reranker.get_reranker")
    def test_reranker_lazy_loads(self, mock_get_reranker):
        from app.container import AppContainer

        mock_get_reranker.return_value = MagicMock()
        container = AppContainer()
        r = container.reranker
        mock_get_reranker.assert_called_once()
        assert r is mock_get_reranker.return_value

    @patch("app.llm.get_llm")
    def test_llm_lazy_loads(self, mock_get_llm):
        from app.container import AppContainer

        mock_get_llm.return_value = MagicMock()
        container = AppContainer()
        llm = container.llm
        mock_get_llm.assert_called_once()
        assert llm is mock_get_llm.return_value

    @patch("app.db.get_vectorstore")
    def test_vectorstore_cached_after_first_access(self, mock_get_vs):
        from app.container import AppContainer

        mock_get_vs.return_value = MagicMock()
        container = AppContainer()
        vs1 = container.vectorstore
        vs2 = container.vectorstore
        assert vs1 is vs2
        mock_get_vs.assert_called_once()


class TestGetContainer:
    def test_returns_container(self):
        from app.container import get_container, AppContainer

        c = get_container()
        assert isinstance(c, AppContainer)

    def test_singleton_returns_same_instance(self):
        from app.container import get_container

        c1 = get_container()
        c2 = get_container()
        assert c1 is c2
