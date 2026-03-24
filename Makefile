PYTHON ?= /usr/local/opt/python@3.11/bin/python3.11

.PHONY: install test lint serve ingest evaluate-fast evaluate-full benchmark docker

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	ruff check agent_bench/ tests/
	ruff format --check agent_bench/ tests/
	mypy agent_bench/ --ignore-missing-imports

serve:
	$(PYTHON) -m uvicorn agent_bench.serving.app:create_app --factory --reload --port 8000

ingest:
	$(PYTHON) scripts/ingest.py --config configs/tasks/tech_docs.yaml

evaluate-fast:
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml --mode deterministic

evaluate-full:
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml --mode full

benchmark:
	$(PYTHON) scripts/benchmark.py --output docs/benchmark_report.md

docker:
	docker-compose -f docker/docker-compose.yaml up --build
