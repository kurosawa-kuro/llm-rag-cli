PYTHON ?= python3
DOCKER_RUN = docker compose exec app
export PYTHONPATH := src

.PHONY: up down build shell test test-unit test-integration test-heavy ingest ask lint evaluate

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

shell:
	docker compose exec app bash

test:
	$(PYTHON) -m pytest tests/ -v

test-unit:
	$(PYTHON) -m pytest tests/ -v -m "not integration and not heavy"

test-integration:
	$(PYTHON) -m pytest tests/ -v -m "integration and not heavy"

test-heavy:
	$(PYTHON) -m pytest tests/ -v -m heavy

ingest:
	$(PYTHON) -m rag.data.ingest

ask:
	$(PYTHON) -m cli.ask "$(Q)"

lint:
	$(PYTHON) -m py_compile src/rag/core/config.py src/rag/infra/db.py src/rag/components/embeddings.py src/rag/components/llm.py src/rag/components/reranker.py src/rag/data/ingest.py src/cli/ask.py src/rag/data/chunking.py src/rag/evaluation/metrics.py src/rag/evaluation/evaluate.py src/rag/pipeline/graph.py src/rag/components/prompting.py src/rag/core/container.py src/rag/core/interfaces.py src/rag/pipeline/retrieval.py

evaluate:
	$(PYTHON) -m rag.evaluation.evaluate
