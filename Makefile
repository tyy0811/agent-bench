PYTHON ?= /usr/local/opt/python@3.11/bin/python3.11

.PHONY: install test lint serve ingest evaluate-fast evaluate-full benchmark evaluate-langchain docker modal-deploy modal-stop vllm-up benchmark-all k8s-dev k8s-prod tf-plan tf-validate

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

evaluate-langchain:
	$(PYTHON) scripts/run_langchain_eval.py --provider openai

docker:
	docker-compose -f docker/docker-compose.yaml up --build

## --- Infrastructure ---

modal-deploy:  ## Deploy vLLM on Modal (prints endpoint URL)
	modal deploy modal/serve_vllm.py

modal-stop:  ## Stop Modal deployment
	modal app stop agent-bench-vllm

vllm-up:  ## Start local vLLM via Docker Compose (requires NVIDIA GPU)
	docker compose -f docker/docker-compose.vllm.yml up --build

benchmark-all:  ## Run provider comparison (requires Modal deployment + API keys)
	$(PYTHON) modal/run_benchmark.py --base-url $(MODAL_VLLM_URL)

k8s-dev:  ## Deploy to minikube (dev values)
	helm install agent-bench k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-dev.yaml

k8s-prod:  ## Deploy via Helm (prod values)
	helm install agent-bench k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-prod.yaml

tf-plan:  ## Run terraform plan (no apply)
	cd terraform && terraform plan

tf-validate:  ## Validate terraform syntax
	cd terraform && terraform validate
