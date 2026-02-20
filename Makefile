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
	docker compose exec app python -m pytest tests/ -v

test-unit:
	docker compose exec app python -m pytest tests/ -v -m "not integration and not heavy"

test-integration:
	docker compose exec app python -m pytest tests/ -v -m "integration and not heavy"

test-heavy:
	docker compose exec app python -m pytest tests/ -v -m heavy

ingest:
	docker compose exec app python -m app.ingest

ask:
	docker compose exec app python -m app.ask "$(Q)"

lint:
	docker compose exec app python -m py_compile app/config.py app/db.py app/embeddings.py app/llm.py app/reranker.py app/ingest.py app/ask.py app/chunking.py app/metrics.py app/evaluate.py app/graph.py app/prompting.py app/container.py

evaluate:
	docker compose exec app python -m app.evaluate
