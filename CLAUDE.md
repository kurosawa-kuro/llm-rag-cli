# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI-based RAG (Retrieval-Augmented Generation) system that runs entirely on CPU with no external API dependencies. Ingests PDFs and CSVs, stores embeddings in PostgreSQL with pgvector via LangChain/LangGraph, and answers queries using a local Llama-2 model.

## Build and Run Commands

All commands run inside Docker containers via Make targets:

```bash
make build              # build Docker image
make up                 # start containers (PostgreSQL + app)
make down               # stop containers
make shell              # enter app container shell
make test               # run all tests (pytest -v)
make lint               # syntax check all app/ modules via py_compile
make ingest             # ingest PDFs/CSVs into vector DB
make ask Q="質問文"      # query the RAG system
make evaluate           # run evaluation pipeline (python -m app.evaluate)
```

To run a single test file or test class inside the container:
```bash
docker compose exec app python -m pytest tests/test_chunking.py -v
docker compose exec app python -m pytest tests/test_evaluate.py::TestEvaluateSingle -v
```

## Architecture

Built on LangChain + LangGraph with a three-node StateGraph pipeline.

```
data/{pdf,csv}/  →  ingest.py  →  chunking  →  PGVector (langchain-postgres)
                                                        ↓
                    ask.py → graph.py (LangGraph StateGraph)
                               ├── retrieve       → vectorstore retriever (k=10)
                               ├── rerank_node    → CrossEncoderReranker (top_k=3)
                               └── generate_node  → LlamaCpp (Japanese prompt)
```

### Pipeline: Two-stage retrieval

1. **Ingest** (`ingest.py`): PDFs use `split_by_structure` (paragraph-aware), CSVs use `RecursiveCharacterTextSplitter`. Chunks become `langchain_core.documents.Document` with source metadata (`file:p1` for PDF pages, `file:r1` for CSV rows) and are stored via `PGVector.add_documents()`.
2. **Query** (`ask.py` → `graph.py`): LangGraph StateGraph runs retrieve → rerank → generate. The `ask.py` `main()` invokes the graph; `search()` uses `ContextualCompressionRetriever` for the standalone search path.

### Key integration points

- **db.py** — `PGVector` vectorstore via `langchain-postgres` (connection string: `postgresql+psycopg://`), singleton pattern
- **embeddings.py** — `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`, 384-dim) via `langchain-huggingface`, singleton
- **llm.py** — `LlamaCpp` (Llama-2-7B Q4_K_M, n_ctx=2048, max_tokens=300) via `langchain-community`, singleton
- **reranker.py** — `HuggingFaceCrossEncoder` + `CrossEncoderReranker` via `langchain-community`, singleton. Also provides `get_compression_retriever()` for `ContextualCompressionRetriever`
- **graph.py** — `RAGState` TypedDict, three-node `StateGraph` (retrieve → rerank → generate → END), singleton via `get_graph()`
- **config.py** — env vars + constants including `CONNECTION_STRING` and `COLLECTION_NAME`
- **metrics.py** — `retrieval_at_k`, `faithfulness`, `exact_match`, `measure_latency`

All model/DB modules use the `_instance = None` global singleton pattern with a `get_*()` lazy-loader.

## Testing Patterns

- Tests use `pytest` + `pytest-mock` with `unittest.mock` (patch, MagicMock)
- Heavy dependencies (LangChain models, PGVector, LlamaCpp) are always mocked — tests never need the DB or model files
- Singleton modules (`embeddings`, `llm`, `reranker`, `db`, `graph`) use `_var = None` pattern; tests reset these via fixtures (e.g., `app.graph._graph = None`)
- Shared fixtures in `tests/conftest.py`: `mock_db_connection`, `fake_embeddings`, `mock_llm_response`, `mock_vectorstore`, `mock_documents`
- pytest config: `pythonpath = .` (imports as `from app.module import ...`)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DB_HOST | localhost | PostgreSQL host (`db` inside Docker) |
| DB_USER | rag | Database user |
| DB_PASSWORD | rag | Database password |
| DB_NAME | rag | Database name |
| CHUNK_SIZE | 500 | Text chunk size in characters |
| CHUNK_OVERLAP | 100 | Overlap between chunks |
| SEARCH_K | 10 | Number of candidates from vector search |
| RERANK_TOP_K | 3 | Number of results after reranking |

## Language

Design documents (`docs/設計書.md`), README, and LLM prompts are in Japanese. The RAG prompt template in `graph.py` and `evaluate.py` uses Japanese instructions.
