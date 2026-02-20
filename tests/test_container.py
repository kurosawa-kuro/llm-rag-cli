from unittest.mock import MagicMock, patch
import pytest


class TestRagSettings:
    def test_default_values(self):
        from rag.core.container import RagSettings

        settings = RagSettings()
        assert settings.search_k == 10
        assert settings.rerank_top_k == 3

    def test_custom_values(self):
        from rag.core.container import RagSettings

        settings = RagSettings(search_k=20, rerank_top_k=5)
        assert settings.search_k == 20
        assert settings.rerank_top_k == 5


class TestAppContainer:
    def test_default_settings(self):
        from rag.core.container import AppContainer, RagSettings

        container = AppContainer()
        assert isinstance(container.settings, RagSettings)
        assert container.settings.search_k == 10

    def test_custom_settings(self):
        from rag.core.container import AppContainer, RagSettings

        settings = RagSettings(search_k=15, rerank_top_k=2)
        container = AppContainer(settings=settings)
        assert container.settings.search_k == 15
        assert container.settings.rerank_top_k == 2

    def test_injected_embeddings(self):
        from rag.core.container import AppContainer

        mock_emb = MagicMock()
        container = AppContainer(embeddings=mock_emb)
        assert container.embeddings is mock_emb

    def test_injected_vectorstore(self):
        from rag.core.container import AppContainer

        mock_vs = MagicMock()
        container = AppContainer(vectorstore=mock_vs)
        assert container.vectorstore is mock_vs

    def test_injected_reranker(self):
        from rag.core.container import AppContainer

        mock_reranker = MagicMock()
        container = AppContainer(reranker=mock_reranker)
        assert container.reranker is mock_reranker

    def test_injected_llm(self):
        from rag.core.container import AppContainer

        mock_llm = MagicMock()
        container = AppContainer(llm=mock_llm)
        assert container.llm is mock_llm

    def test_injected_prompt_builder(self):
        from rag.core.container import AppContainer

        builder = lambda q, c: "mock"
        container = AppContainer(prompt_builder=builder)
        assert container.prompt_builder is builder

    def test_prompt_builder_default_is_callable(self):
        from rag.core.container import AppContainer

        container = AppContainer()
        assert callable(container.prompt_builder)

    @patch("rag.components.embeddings.create_embeddings")
    @patch("rag.infra.db.create_vectorstore")
    def test_vectorstore_lazy_loads(self, mock_create_vs, mock_create_emb):
        from rag.core.container import AppContainer

        mock_create_emb.return_value = MagicMock()
        mock_create_vs.return_value = MagicMock()
        container = AppContainer()
        vs = container.vectorstore
        mock_create_vs.assert_called_once_with(mock_create_emb.return_value)
        assert vs is mock_create_vs.return_value

    @patch("rag.components.reranker.create_reranker")
    def test_reranker_lazy_loads(self, mock_create_reranker):
        from rag.core.container import AppContainer

        mock_create_reranker.return_value = MagicMock()
        container = AppContainer()
        r = container.reranker
        mock_create_reranker.assert_called_once()
        assert r is mock_create_reranker.return_value

    @patch("rag.components.llm.create_llm")
    def test_llm_lazy_loads(self, mock_create_llm):
        from rag.core.container import AppContainer

        mock_create_llm.return_value = MagicMock()
        container = AppContainer()
        llm = container.llm
        mock_create_llm.assert_called_once()
        assert llm is mock_create_llm.return_value

    @patch("rag.components.embeddings.create_embeddings")
    def test_embeddings_lazy_loads(self, mock_create_emb):
        from rag.core.container import AppContainer

        mock_create_emb.return_value = MagicMock()
        container = AppContainer()
        emb = container.embeddings
        mock_create_emb.assert_called_once()
        assert emb is mock_create_emb.return_value

    @patch("rag.components.embeddings.create_embeddings")
    @patch("rag.infra.db.create_vectorstore")
    def test_vectorstore_cached_after_first_access(self, mock_create_vs, mock_create_emb):
        from rag.core.container import AppContainer

        mock_create_emb.return_value = MagicMock()
        mock_create_vs.return_value = MagicMock()
        container = AppContainer()
        vs1 = container.vectorstore
        vs2 = container.vectorstore
        assert vs1 is vs2
        mock_create_vs.assert_called_once()

    def test_injected_retrieval_strategy(self):
        from rag.core.container import AppContainer

        mock_strategy = MagicMock()
        container = AppContainer(retrieval_strategy=mock_strategy)
        assert container.retrieval_strategy is mock_strategy

    def test_retrieval_strategy_lazy_loads(self):
        from rag.core.container import AppContainer
        from rag.pipeline.retrieval import TwoStageRetrieval

        mock_vs = MagicMock()
        mock_reranker = MagicMock()
        container = AppContainer(vectorstore=mock_vs, reranker=mock_reranker)
        strategy = container.retrieval_strategy
        assert isinstance(strategy, TwoStageRetrieval)
        assert strategy.vectorstore is mock_vs
        assert strategy.reranker is mock_reranker
        assert strategy.search_k == 10
        assert strategy.rerank_top_k == 3

    def test_retrieval_strategy_cached_after_first_access(self):
        from rag.core.container import AppContainer

        mock_vs = MagicMock()
        mock_reranker = MagicMock()
        container = AppContainer(vectorstore=mock_vs, reranker=mock_reranker)
        s1 = container.retrieval_strategy
        s2 = container.retrieval_strategy
        assert s1 is s2

    def test_rag_settings_is_frozen(self):
        from rag.core.container import RagSettings

        settings = RagSettings()
        with pytest.raises(AttributeError):
            settings.search_k = 99


class TestGetContainer:
    def test_returns_container(self):
        from rag.core.container import get_container, AppContainer

        c = get_container()
        assert isinstance(c, AppContainer)

    def test_singleton_returns_same_instance(self):
        from rag.core.container import get_container

        c1 = get_container()
        c2 = get_container()
        assert c1 is c2
