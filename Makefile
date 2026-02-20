PYTHON ?= python3
DOCKER_RUN = docker compose exec app

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
	$(PYTHON) -m app.ingest

ask:
	$(PYTHON) -m app.ask "$(Q)"

lint:
	$(PYTHON) -m py_compile app/config.py app/db.py app/embeddings.py app/llm.py app/reranker.py app/ingest.py app/ask.py app/chunking.py app/metrics.py app/evaluate.py app/graph.py app/prompting.py app/container.py app/interfaces.py app/retrieval.py

evaluate:
	$(PYTHON) -m app.evaluate
