# Convenience shortcuts (optional - every command also works without make).
#   make up        start Qdrant
#   make install   install Python requirements
#   make check     verify .env / Qdrant / LLM / embeddings
#   make chat      start the CLI chat
#   make api       start the FastAPI server
#   make bench     run the benchmark
#   make seed      insert 1,100 synthetic memories
#   make test      run the unit tests
#   make down      stop Docker services

PY ?= python

up:
	docker compose up -d qdrant

down:
	docker compose --profile app --profile local down

install:
	$(PY) -m pip install -r requirements.txt

check:
	$(PY) scripts/check_setup.py

chat:
	$(PY) scripts/chat.py

api:
	$(PY) -m uvicorn memory_agent.api:app --reload --app-dir src --port 8000

bench:
	$(PY) scripts/benchmark.py

seed:
	$(PY) scripts/seed_memories.py

test:
	$(PY) -m pytest -v

integration-test:
	RUN_INTEGRATION=1 $(PY) -m pytest tests/test_integration_qdrant.py -v

docker-app:
	docker compose --profile app up --build

.PHONY: up down install check chat api bench seed test integration-test docker-app
