.PHONY: up down build shell test ingest ask lint evaluate

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

ingest:
	docker compose exec app python app/ingest.py

ask:
	docker compose exec app python app/ask.py "$(Q)"

lint:
	docker compose exec app python -m py_compile app/config.py app/db.py app/embeddings.py app/llm.py app/reranker.py app/ingest.py app/ask.py app/chunking.py app/metrics.py app/evaluate.py

evaluate:
	docker compose exec app python -m app.evaluate
