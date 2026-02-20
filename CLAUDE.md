# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI-based RAG (Retrieval-Augmented Generation) system that runs entirely on CPU with no external API dependencies. Ingests PDFs and CSVs, stores embeddings in PostgreSQL with pgvector, and answers queries using a local Llama-2 model via llama-cpp-python.

**Status:** Design phase. The design spec is in `docs/設計書.md` (Japanese). No source code has been implemented yet.

## Architecture

```
data/{pdf,csv}/  →  ingest.py  →  PostgreSQL+pgvector
                                        ↓
                    ask.py  ←──  semantic search (k=3)
                      ↓
                    llm.py  →  answer via Llama-2 (CPU, GGUF)
```

Six planned Python modules under `app/`:
- **config.py** — env-based config (DB creds, model paths)
- **db.py** — PostgreSQL connection + pgvector schema (VECTOR(384))
- **embeddings.py** — sentence-transformers (`all-MiniLM-L6-v2`, 384-dim)
- **llm.py** — llama-cpp-python wrapper (Llama-2-7B Q4_K_M, 2048 ctx)
- **ingest.py** — PDF/CSV loader → embed → store in DB
- **ask.py** — CLI entry point: query → vector search → LLM generation

## Development Environment

Runs in Docker (Python 3.11-slim + PostgreSQL 16 with pgvector).

```bash
docker compose up -d                # start DB + app containers
docker compose exec app bash        # enter app container
python app/ingest.py                # ingest documents
python app/ask.py "your question"   # query the system
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DB_HOST | localhost | PostgreSQL host (use `db` inside Docker) |
| DB_USER | rag | Database user |
| DB_PASSWORD | rag | Database password |
| DB_NAME | rag | Database name |

### LLM Model Setup

Place the GGUF model file at `./models/llama-2-7b.Q4_K_M.gguf`. Q4_K_M quantization is recommended for CPU inference.

## Key Dependencies

psycopg2-binary, sentence-transformers, pypdf, pandas, numpy, tqdm, llama-cpp-python

## Language

The design document and default prompts are in Japanese. The RAG prompt template in `ask.py` uses Japanese instructions.
