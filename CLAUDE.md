# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI-based RAG (Retrieval-Augmented Generation) system that runs entirely on CPU with no external API dependencies. Ingests PDFs and CSVs, stores embeddings in PostgreSQL with pgvector via LangChain/LangGraph, and answers queries using a local Llama-2 model. Uses DI Container (`AppContainer`) pattern with Protocol-based interfaces for type-safe dependency injection.

## Project Structure

```
src/
├── rag/                     # RAGコアロジック
│   ├── pipeline/            # パイプライン構成
│   │   ├── graph.py         # LangGraph 2-node StateGraph
│   │   └── retrieval.py     # TwoStageRetrieval strategy
│   ├── components/          # モデルコンポーネント群
│   │   ├── embeddings.py    # HuggingFaceEmbeddings factory
│   │   ├── reranker.py      # CrossEncoderReranker factory
│   │   ├── llm.py           # LlamaCpp factory
│   │   └── prompting.py     # Japanese prompt builder
│   ├── data/                # データ処理
│   │   ├── chunking.py      # Text splitting (fixed-size + structure-based)
│   │   └── ingest.py        # PDF/CSV ingestion pipeline
│   ├── evaluation/          # 評価
│   │   ├── metrics.py       # retrieval@k, faithfulness, exact_match, latency
│   │   └── evaluate.py      # Evaluation pipeline runner
│   ├── core/                # DI・Interface・設定
│   │   ├── interfaces.py    # Protocol definitions
│   │   ├── container.py     # AppContainer DI + get_container()
│   │   └── config.py        # Settings from setting.yaml + env vars
│   └── infra/               # 外部依存
│       └── db.py            # PGVector vectorstore factory
└── cli/                     # エントリポイント
    └── ask.py               # Question-answering CLI
```

Import prefix: `from rag.xxx import` (PYTHONPATH=src).

## Build and Run Commands

All commands run inside Docker containers via Make targets:

```bash
make build              # build Docker image
make up                 # start containers (PostgreSQL + app)
make down               # stop containers
make shell              # enter app container shell
make test               # run all tests (pytest -v)
make test-unit          # run unit tests only (no DB/model needed)
make test-integration   # run DB integration tests (PostgreSQL needed)
make test-heavy         # run real embeddings tests (PostgreSQL + model DL needed)
make lint               # syntax check all src/ modules via py_compile (15 modules)
make ingest             # ingest PDFs/CSVs into vector DB
make ask Q="質問文"      # query the RAG system
make evaluate           # run evaluation pipeline
make evaluate-retrieval # run retrieval-only evaluation (no LLM needed)
```

To run a single test file or test class inside the container:
```bash
docker compose exec app python -m pytest tests/test_chunking.py -v
docker compose exec app python -m pytest tests/test_evaluate.py::TestEvaluateSingle -v
```

## Architecture

Built on LangChain + LangGraph with DI Container (`AppContainer`) and a two-node StateGraph pipeline.

```
data/{pdf,csv}/  →  ingest.py  →  chunking  →  container.vectorstore (PGVector)
                                                        ↓
                    ask.py → graph.py (LangGraph StateGraph, 2-node)
                               ├── retrieve       → TwoStageRetrieval (vector search + rerank)
                               └── generate       → LlamaCpp (Japanese prompt)
```

### Pipeline: Two-stage retrieval via TwoStageRetrieval

1. **Ingest** (`rag.data.ingest`): Clears existing collection via `delete_collection()` + vectorstore recreation. PDFs use `split_by_structure` (paragraph-aware), CSVs use 1-row-1-document (category columns excluded). Chunks become `langchain_core.documents.Document` with source metadata (`file:p1` for PDF pages, `file:r1` for CSV rows) and are stored via `container.vectorstore.add_documents()`.
2. **Query** (`cli.ask` → `rag.pipeline.graph`): LangGraph StateGraph runs retrieve → generate (2-node). The `retrieve` node calls `container.retrieval_strategy.retrieve()` which encapsulates `similarity_search_with_score` (k=20) + score threshold filtering (0.5) + Cross-Encoder reranking (top_k=5) in `TwoStageRetrieval`. Generate node returns early with "該当する情報が見つかりませんでした。" when no contexts found.

### Key integration points

- **`rag.core.interfaces`** — Protocol definitions (`VectorStoreProtocol`, `RerankerProtocol`, `LLMProtocol`, `RetrievalStrategyProtocol`, `PromptBuilder`) for type-safe DI
- **`rag.core.container`** — `AppContainer` DI Container with lazy properties (embeddings, vectorstore, reranker, llm, prompt_builder, retrieval_strategy). `get_container()` for singleton. Test-time mock injection via constructor args
- **`rag.pipeline.retrieval`** — `TwoStageRetrieval` frozen dataclass encapsulating `similarity_search_with_score` → score threshold filtering → Cross-Encoder reranking
- **`rag.infra.db`** — `create_vectorstore(embeddings)` factory for `langchain_postgres.PGVector` (connection string: `postgresql+psycopg://`)
- **`rag.components.embeddings`** — `create_embeddings()` factory for `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`, 384-dim)
- **`rag.components.llm`** — `create_llm()` factory for `LlamaCpp` (Llama-2-7B Q4_K_M, n_ctx=2048, max_tokens=300, stop=["質問:", "\n\n"])
- **`rag.components.reranker`** — `CrossEncoderReranker` class implementing `RerankerProtocol`. `create_reranker()` factory for `HuggingFaceCrossEncoder` + `CrossEncoderReranker`
- **`rag.pipeline.graph`** — `RAGState` dataclass, two-node `StateGraph` (retrieve → generate → END). `create_retrieve(container)` and `create_generate(container)` factory functions. Singleton via `get_graph()`
- **`rag.core.config`** — Loads settings from `env/config/setting.yaml` via `_load_settings()`. Env vars override YAML defaults. Exports `CONNECTION_STRING`, `COLLECTION_NAME`, and all parameter constants
- **`rag.components.prompting`** — `build_prompt(query, contexts)` for Japanese prompt template
- **`rag.evaluation.metrics`** — `retrieval_at_k`, `faithfulness`, `exact_match`, `measure_latency`, `context_relevance`, `retrieval_mrr`

Infrastructure modules (`db.py`, `embeddings.py`, `llm.py`, `reranker.py`) provide stateless factory functions (`create_*`). `AppContainer` manages lifecycle via lazy properties.

## Testing Patterns

- Tests use `pytest` + `pytest-mock` with `unittest.mock` (patch, MagicMock)
- Heavy dependencies (LangChain models, PGVector, LlamaCpp) are always mocked in unit tests — no DB or model files needed
- DI Container: tests inject mocks via `AppContainer` constructor args. `conftest.py` has `reset_container` autouse fixture that resets `rag.core.container._container = None`
- Test markers: `@pytest.mark.integration` for DB tests, `@pytest.mark.heavy` for real embeddings tests
- Shared fixtures in `tests/conftest.py`: `reset_container`, `test_embeddings`, `test_vectorstore`, `real_vectorstore`
- pytest config: `pythonpath = src` (imports as `from rag.module import ...`)
- 250 tests across 17 files (240 unit + 7 integration + 3 heavy)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DB_HOST | localhost | PostgreSQL host (`db` inside Docker) |
| DB_USER | rag | Database user |
| DB_PASSWORD | rag | Database password |
| DB_NAME | rag | Database name |
| CHUNK_SIZE | 350 | Text chunk size in characters |
| CHUNK_OVERLAP | 80 | Overlap between chunks |
| SEARCH_K | 20 | Number of candidates from vector search |
| RERANK_TOP_K | 3 | Number of results after reranking |
| SCORE_THRESHOLD | 0.5 | Score threshold for vector search (cosine distance) |

Defaults are defined in `env/config/setting.yaml`. Environment variables take precedence when set.

## Language

Design documents (`docs/設計書.md`), README, and LLM prompts are in Japanese. The RAG prompt template in `rag.components.prompting` and `rag.evaluation.evaluate` uses Japanese instructions.
