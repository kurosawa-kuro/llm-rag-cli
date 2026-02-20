# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI-based RAG (Retrieval-Augmented Generation) system that runs entirely on CPU with no external API dependencies. Ingests PDFs and CSVs, stores embeddings in PostgreSQL with pgvector, and answers queries using a local Llama-2 model via llama-cpp-python.

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

```
data/{pdf,csv}/  →  ingest.py  →  chunking  →  embed  →  PostgreSQL+pgvector
                                                                ↓
                    ask.py  ←──  reranker  ←──  semantic search (k=10)
                      ↓
                    llm.py  →  answer via Llama-2 (CPU, GGUF)
```

### Pipeline: Two-stage retrieval

1. **Ingest** (`ingest.py`): PDFs use `split_by_structure` (paragraph-aware), CSVs use `split_text` (fixed-size). Chunks are embedded and stored with source metadata (`file:p1` for PDF pages, `file:r1` for CSV rows).
2. **Query** (`ask.py`): Embeds query → pgvector cosine distance search (top SEARCH_K=10) → cross-encoder reranking (top RERANK_TOP_K=3) → LLM generation with Japanese prompt template.

### Modules under `app/`

- **config.py** — all settings via env vars + constants (DB_CONFIG, model paths, chunk/search params)
- **db.py** — PostgreSQL connection + pgvector schema (`documents` table, VECTOR(384))
- **embeddings.py** — sentence-transformers wrapper (`all-MiniLM-L6-v2`, 384-dim), lazy-loaded singleton
- **llm.py** — llama-cpp-python wrapper (Llama-2-7B Q4_K_M, 2048 ctx), lazy-loaded singleton
- **reranker.py** — cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`), lazy-loaded singleton
- **chunking.py** — `split_text` (fixed-size with overlap) and `split_by_structure` (paragraph-aware)
- **ingest.py** — PDF/CSV loader → chunk → embed → store
- **ask.py** — CLI entry: query → vector search → rerank → LLM generate
- **metrics.py** — `retrieval_at_k`, `faithfulness`, `measure_latency`
- **evaluate.py** — evaluation runner using `data/eval_questions.json`

## Testing Patterns

- Tests use `pytest` + `pytest-mock` with `unittest.mock` (patch, MagicMock)
- Heavy dependencies (sentence-transformers, llama-cpp, psycopg2) are always mocked — tests never need the DB or model files
- Singleton models in embeddings/llm/reranker use `_model = None` pattern; tests reset these via fixtures
- Shared fixtures in `tests/conftest.py`: `mock_db_connection`, `fake_embeddings`, `mock_llm_response`
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

Design documents (`docs/設計書.md`), README, and LLM prompts are in Japanese. The RAG prompt template in `ask.py` uses Japanese instructions.
