PYTHON ?= /usr/local/opt/python@3.11/bin/python3.11

.PHONY: install test lint serve ingest ingest-k8s evaluate-fast evaluate-full benchmark evaluate-langchain calibrate evaluate-judges stats-table epochs epochs-dry-run epochs-dry-run-k8s evaluate-stats plots canary-report docker modal-deploy modal-stop vllm-up benchmark-all k8s-dev k8s-prod tf-plan tf-validate

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	ruff check agent_bench/ stats/ stats_adapters/ tests/ scripts/run_epochs.py scripts/run_canary_eval.py scripts/make_plots.py
	ruff format --check stats/ stats_adapters/ tests/stats/ scripts/run_epochs.py scripts/run_canary_eval.py scripts/make_plots.py
	mypy agent_bench/ stats/ stats_adapters/ scripts/run_epochs.py scripts/run_canary_eval.py scripts/make_plots.py --ignore-missing-imports

serve:
	$(PYTHON) -m uvicorn agent_bench.serving.app:create_app --factory --reload --port 8000

ingest:
	$(PYTHON) scripts/ingest.py --config configs/tasks/tech_docs.yaml

ingest-k8s:  ## Ingest Kubernetes docs into .cache/store_k8s
	$(PYTHON) scripts/ingest.py --doc-dir data/k8s_docs --store-path .cache/store_k8s

evaluate-fast:
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml --mode deterministic

evaluate-full:
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml --mode full

benchmark:
	$(PYTHON) scripts/benchmark.py --output docs/benchmark_report.md

evaluate-langchain:
	$(PYTHON) scripts/run_langchain_eval.py --provider openai

calibrate:  ## Run full calibration pipeline (system outputs → all rows → strict κ table). Costs ~$2 in API calls.
	$(PYTHON) scripts/run_calibration.py generate-outputs
	@for cfg in configs/calibration/rows/*.yaml; do \
		echo "==> running judges for $$cfg"; \
		$(PYTHON) scripts/run_calibration.py run-judges --row-config=$$cfg || exit 1; \
	done
	$(PYTHON) scripts/run_calibration.py build-table --strict

evaluate-judges:  ## Re-run all rows + build-table against existing system_outputs (no regeneration). Costs ~$1.
	@for cfg in configs/calibration/rows/*.yaml; do \
		echo "==> running judges for $$cfg"; \
		$(PYTHON) scripts/run_calibration.py run-judges --row-config=$$cfg || exit 1; \
	done
	$(PYTHON) scripts/run_calibration.py build-table --strict

stats-table:  ## Convert legacy results JSON to validated long CSV (free, offline)
	$(PYTHON) -m stats_adapters.from_results_json --legacy \
		--input results/fastapi_postedit.json \
		--golden agent_bench/evaluation/datasets/tech_docs_golden.json \
		--config-id custom-openai-legacy \
		--out-dir results/long

epochs:  ## PAID, HUMAN-RUN: repeat eval k times per config. Usage: make epochs K=5 CONFIGS=custom-openai,custom-anthropic CONFIRM_PAID=1
	@test "$(CONFIRM_PAID)" = "1" || (echo "Refusing: paid target. Set CONFIRM_PAID=1 to run. Costs real API money." && exit 1)
	$(PYTHON) scripts/run_epochs.py --k $(K) --configs $(CONFIGS) --allow-paid

epochs-dry-run:  ## Free pre-spend gate: validate the WP5 configs (corpora, golden, store, keys) with no API calls
	$(PYTHON) scripts/run_epochs.py --k 1 --dry-run \
		--configs custom-openai,custom-anthropic,langchain-openai,langchain-anthropic

epochs-dry-run-k8s:  ## Free pre-spend gate for the UNMEASURED k8s configs (corpus/golden/store/keys); no API calls
	$(PYTHON) scripts/run_epochs.py --k 1 --dry-run \
		--configs custom-openai-k8s,custom-anthropic-k8s

evaluate-stats:  ## Regenerate docs/_generated/stats_report.md from results/long (free, offline)
	$(PYTHON) -m stats.report --tables results/long --out docs/_generated/stats_report.md

plots:  ## Regenerate README figures from the stats report (needs `pip install -e .[plots]`)
	$(PYTHON) scripts/make_plots.py generate

canary-report:  ## Regenerate docs/_generated/canary_detection.md from the committed canary fixtures (free, offline; verdicts are simulated)
	$(PYTHON) scripts/run_canary_eval.py build-report \
		--canaries tests/stats/fixtures/canary/canaries.json \
		--predictions tests/stats/fixtures/canary/predictions.json

docker:
	docker-compose -f docker/docker-compose.yaml up --build

## --- Infrastructure ---

modal-deploy:  ## Deploy vLLM on Modal (prints endpoint URL)
	@command -v modal >/dev/null 2>&1 || { echo "Error: modal CLI not found. Run: pip install -e '.[modal]' && modal setup"; exit 1; }
	modal deploy modal/serve_vllm.py

modal-stop:  ## Stop Modal deployment
	@command -v modal >/dev/null 2>&1 || { echo "Error: modal CLI not found. Run: pip install -e '.[modal]' && modal setup"; exit 1; }
	modal app stop agent-bench-vllm

vllm-up:  ## Start local vLLM via Docker Compose (requires NVIDIA GPU)
	docker compose -f docker/docker-compose.vllm.yml up --build

benchmark-all:  ## Run provider comparison (requires Modal deployment + API keys)
	$(PYTHON) modal/run_benchmark.py --base-url $(MODAL_VLLM_URL)

k8s-dev:  ## Deploy to minikube (dev values, set MODAL_VLLM_URL first)
	@test -n "$(MODAL_VLLM_URL)" || (echo "Error: MODAL_VLLM_URL is not set" && exit 1)
	helm install agent-bench k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-dev.yaml \
		--set provider.selfhosted.modalEndpoint=$(MODAL_VLLM_URL)

k8s-prod:  ## Deploy via Helm (prod values, set MODAL_VLLM_URL first)
	@test -n "$(MODAL_VLLM_URL)" || (echo "Error: MODAL_VLLM_URL is not set" && exit 1)
	helm install agent-bench k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-prod.yaml \
		--set provider.selfhosted.modalEndpoint=$(MODAL_VLLM_URL)

tf-plan:  ## Run terraform plan (no apply)
	cd terraform && terraform plan

tf-validate:  ## Validate terraform syntax
	cd terraform && terraform validate
