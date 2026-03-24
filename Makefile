.PHONY: install test lint serve ingest evaluate-fast evaluate-full benchmark docker

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check agent_bench/ tests/
	ruff format --check agent_bench/ tests/
	mypy agent_bench/ --ignore-missing-imports

serve:
	uvicorn agent_bench.serving.app:create_app --factory --reload --port 8000

ingest:
	python3 scripts/ingest.py --config configs/tasks/tech_docs.yaml

evaluate-fast:
	python3 scripts/evaluate.py --config configs/default.yaml --mode deterministic

evaluate-full:
	python3 scripts/evaluate.py --config configs/default.yaml --mode full

benchmark:
	python3 scripts/benchmark.py --output docs/benchmark_report.md

docker:
	docker-compose -f docker/docker-compose.yaml up --build
