# agent-bench — Infrastructure Sprint Design

**Goal:** Add Kubernetes orchestration, Terraform IaC, and self-hosted LLM serving (vLLM) to agent-bench, closing the three most visible infra gaps identified in job postings. GPU inference runs on Modal; K8s handles the API layer.

**Estimated effort:** 7-9 working days
**Branch:** `feat/infra-sprint`

---

## Current State

```
agent_bench/
  core/       # Provider abstraction (OpenAI, Anthropic, MockProvider)
  agents/     # Orchestrator (tool-use loop, max 3 iterations)
  tools/      # Registry, search_documents, calculator
  rag/        # Chunker, embedder, FAISS+BM25 store, retriever
  evaluation/ # Harness, metrics, golden dataset (27 questions)
  serving/    # FastAPI app, routes, schemas, middleware
docker/
  docker-compose.yaml   # Single-service compose (app only)
configs/
  # YAML-based config (provider, retrieval strategy, model)
```

Key architectural facts:

- **Provider abstraction already exists.** `core/provider.py` defines `LLMProvider` ABC with `complete()`, `stream_complete()`, `format_tools()`. OpenAI and Anthropic are fully implemented. Adding `SelfHostedProvider` is a clean extension.
- **Docker already works.** `docker/docker-compose.yaml` builds and runs the app with pre-baked models and FAISS store. K8s manifests can mirror this.
- **`/metrics` endpoint exists.** JSON-format metrics (request count, latency p50/p95, cost). Not Prometheus format — a Prometheus exporter adapter would be needed for custom-metrics HPA.
- **`/health` endpoint exists.** Reports store stats, provider status, uptime. Maps directly to K8s liveness/readiness probes.
- **172 tests, CI via GitHub Actions.** New infra code must not break existing CI.
- **Config system uses static YAML + Pydantic.** No env var interpolation in YAML. Providers read env vars directly in `__init__` (e.g., `OPENAI_API_KEY`). The `SelfHostedProvider` will follow this same pattern for `MODAL_VLLM_URL`.

---

## Work Package 1: Self-Hosted LLM Provider via vLLM + Modal (3-5 days)

### Why this is highest priority

Job postings explicitly list "self-hosted LLM serving (vLLM, llama.cpp, TGI)" as a requirement. The current repo only demonstrates API-based providers. This is the single highest-signal addition.

### 1.1 — Implement `SelfHostedProvider` (1 day)

**File:** `agent_bench/core/providers/selfhosted.py`

```python
class SelfHostedProvider(LLMProvider):
    """Provider targeting a vLLM/TGI-compatible OpenAI-format endpoint.

    Works with any backend exposing OpenAI-compatible /v1/chat/completions:
      - Local vLLM via Docker Compose (docker/docker-compose.vllm.yml)
      - Modal serverless vLLM (modal/serve_vllm.py)
      - TGI, llama.cpp server, Ollama, etc.

    The provider is endpoint-agnostic by design. It targets the HTTP contract,
    not the serving infrastructure.
    """

    def __init__(self, config: SelfHostedConfig):
        self.base_url = config.base_url or os.environ.get("MODAL_VLLM_URL", "")
        self.model_name = config.model_name
        self.timeout = config.timeout_seconds
        self.api_key = config.api_key or os.environ.get("MODAL_AUTH_TOKEN", "")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )

    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResponse:
        # POST /v1/chat/completions with OpenAI-compatible schema
        # Key differences from OpenAI provider:
        #   - API key optional (local) or Modal token (serverless)
        #   - Tool/function calling support depends on model + vLLM version
        #   - Token counting uses local tokenizer, not tiktoken
        ...

    async def stream_complete(
        self,
        messages: list[dict],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        # SSE streaming from /v1/chat/completions with stream=true
        ...

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        # OpenAI-compatible tool format (same as OpenAI provider)
        ...

    async def health_check(self) -> ProviderHealth:
        # GET /health or /v1/models to verify endpoint is responsive
        ...
```

**Design decisions (for DECISIONS.md):**

- **Why OpenAI-compatible endpoint, not raw vLLM API:** vLLM, TGI, and llama.cpp all support the OpenAI chat completions format. Targeting this format means the provider works with any of them. This is a deliberate generalization.
- **Why `httpx.AsyncClient`, not `openai.AsyncOpenAI`:** Avoids tight coupling to the OpenAI SDK. The HTTP contract is simple. Using httpx makes the dependency explicit and testable.
- **Why endpoint-agnostic design:** The same `SelfHostedProvider` targets both local Docker Compose vLLM and Modal serverless vLLM. The difference is just a URL and an optional auth token. This mirrors real production architectures where inference backends are swappable behind a load balancer.
- **Why env var fallback in `__init__`, not YAML interpolation:** Follows the same pattern as `OpenAIProvider` reading `OPENAI_API_KEY`. Simpler, more consistent, no config loader changes needed.
- **Tool calling detection via startup smoke test:** Not all self-hosted models support tool/function calling. On provider init, send one tool-calling request and check if the response contains `tool_calls`. Cache the result as `self.supports_tool_calling: bool`. If false, fall back to prompt-based tool selection (inject tool descriptions into the system prompt and parse the model's text output). Document as a known limitation — unreliable tool calling on a self-hosted model is a legitimate benchmark finding, not a failure.

**Config extensions in `configs/`:**

```yaml
# configs/selfhosted_local.yaml
provider:
  default: selfhosted
  selfhosted:
    base_url: "http://localhost:8000/v1"
    model_name: mistralai/Mistral-7B-Instruct-v0.3
    timeout_seconds: 120
```

```yaml
# configs/selfhosted_modal.yaml
provider:
  default: selfhosted
  selfhosted:
    base_url: ""                  # Falls back to MODAL_VLLM_URL env var
    model_name: mistralai/Mistral-7B-Instruct-v0.3
    api_key: ""                   # Falls back to MODAL_AUTH_TOKEN env var
    timeout_seconds: 120
```

**Tests:** `tests/test_selfhosted_provider.py` — 8-10 unit tests using `httpx.MockTransport`. Test: completion parsing, health check, timeout handling, tool call detection, auth header injection, env var fallback. Mirror existing OpenAI provider test structure.

### 1.2 — Modal vLLM Deployment (1 day)

**Directory:** `modal/`

```
modal/
  serve_vllm.py           # Modal app: vLLM serving as web endpoint
  run_benchmark.py         # Run 27-question eval against Modal endpoint
  common.py                # Shared config (model name, GPU type, image def)
```

**`modal/serve_vllm.py`:**

```python
"""Deploy vLLM on Modal as an OpenAI-compatible endpoint.

Usage:
    modal deploy modal/serve_vllm.py     # Deploy (stays running, prints URL)
    modal serve modal/serve_vllm.py      # Dev mode (auto-redeploys)
"""

import modal

MODELS_DIR = "/models"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("agent-bench-vllm")
model_volume = modal.Volume.from_name("vllm-model-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=modal.gpu.A10G(),
    container_idle_timeout=300,
    timeout=600,
    volumes={MODELS_DIR: model_volume},
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def serve():
    """Serve vLLM as an ASGI app with OpenAI-compatible endpoints."""
    # Implementation note: check Modal's current vLLM example at implementation time.
    # The vLLM + Modal integration pattern may use @modal.cls instead of @modal.asgi_app
    # depending on vLLM version. Key contract: expose /v1/chat/completions and /health.
    ...
```

**`modal/run_benchmark.py`:**

```python
"""Run the 27-question benchmark against a Modal-hosted vLLM endpoint.

Usage:
    modal deploy modal/serve_vllm.py     # First deploy
    python modal/run_benchmark.py --base-url https://...modal.run
"""

# Calls scripts/evaluate.py --config for each provider config.
# Produces docs/provider_comparison.md with real measured data.
```

**`modal/common.py`:**

```python
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
GPU_TYPE = "a10g"
VLLM_MAX_MODEL_LEN = 4096
VLLM_DTYPE = "half"
VLLM_GPU_MEMORY_UTILIZATION = 0.85
MODAL_A10G_COST_PER_SEC = 0.000361  # ~$1.30/hr
```

### 1.3 — Docker Compose vLLM (0.5 day)

**File:** `docker/docker-compose.vllm.yml`

Demonstrates the persistent-GPU alternative to Modal. Both target the same `SelfHostedProvider` via the same OpenAI-compatible endpoint.

- **Modal** = serverless GPU, pay-per-second, cold starts
- **Docker Compose** = persistent GPU, fixed cost, no cold starts, requires NVIDIA runtime

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    command:
      - --model=mistralai/Mistral-7B-Instruct-v0.3
      - --max-model-len=4096
      - --dtype=half
      - --gpu-memory-utilization=0.85
      - --host=0.0.0.0
      - --port=8000
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - vllm-cache:/root/.cache/huggingface
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s

  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    environment:
      - AGENT_BENCH_CONFIG=configs/selfhosted_local.yaml
    depends_on:
      vllm:
        condition: service_healthy
    ports:
      - "8080:8000"

volumes:
  vllm-cache:
```

### 1.4 — Benchmark: API vs Self-Hosted (1 day)

Run the 27-question evaluation harness against all provider configurations using `scripts/evaluate.py --config`:

| Config | Provider | Model | P@5 | R@5 | Citation Acc | Latency p50 | Cost/query | Infra |
|--------|----------|-------|-----|-----|--------------|-------------|------------|-------|
| OpenAI | API | gpt-4o-mini | 0.70 | 0.83 | 1.00 | 4,690 ms | $0.0004 | None |
| Anthropic | API | claude-haiku | TBD | TBD | TBD | TBD | TBD | None |
| Self-hosted | vLLM (Modal) | Mistral-7B | TBD | TBD | TBD | TBD | TBD | A10G |

Additional Modal-specific metrics:

| Config | Cold start | Warm latency p50 | GPU util % | VRAM used (GB) |
|--------|-----------|-------------------|------------|----------------|
| Self-hosted (Modal) | ~60-90s | TBD | TBD | TBD |

**Output:** `docs/provider_comparison.md` covering:
1. Retrieval quality: does the smaller self-hosted model hurt P@5/R@5?
2. Citation accuracy: does Mistral-7B hallucinate citations?
3. Tool calling: does Mistral-7B reliably use search_documents and calculator?
4. Cost analysis: API cost/query vs Modal GPU-second cost/query
5. Latency breakdown: cold start vs warm, first-token vs total
6. Operational complexity: managed API vs self-hosted

---

## Work Package 2: Kubernetes Helm Chart (2 days)

### 2.1 — Helm Chart (1.5 days)

**Directory:** `k8s/helm/agent-bench/`

```
k8s/helm/agent-bench/
  Chart.yaml
  values.yaml
  values-dev.yaml
  values-prod.yaml
  templates/
    deployment.yaml
    service.yaml
    hpa.yaml
    configmap.yaml
    secret.yaml
    _helpers.tpl
```

No `vllm-deployment.yaml` in K8s. GPU inference is handled by Modal (external to the cluster). The K8s cluster runs only the API pods, which call the Modal vLLM endpoint via HTTPS. This separates the stateless CPU-bound API layer (K8s, horizontal scaling) from the GPU-bound inference layer (Modal, serverless elasticity).

**`values.yaml`:**

```yaml
replicaCount: 2
image:
  repository: agent-bench
  tag: latest

provider:
  type: selfhosted
  selfhosted:
    model: mistralai/Mistral-7B-Instruct-v0.3
    modalEndpoint: ""
    modalAuthToken: ""

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 8
  targetCPUUtilization: 70
```

**Key template details (`templates/deployment.yaml`):**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "agent-bench.fullname" . }}
  labels:
    {{- include "agent-bench.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "agent-bench.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "agent-bench.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: api
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: {{ include "agent-bench.fullname" . }}-config
            - secretRef:
                name: {{ include "agent-bench.fullname" . }}-secrets
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: 2000m
              memory: 4Gi
```

**HPA (`templates/hpa.yaml`):** CPU utilization is the simplest autoscaling signal that works without custom metrics infrastructure. A production improvement would use the Prometheus adapter to scale on p95 latency from the `/metrics` endpoint (requires adding a Prometheus exporter adapter to bridge JSON metrics to Prometheus format). Documented as a follow-up, not implemented.

**Environment overrides via `values-dev.yaml` / `values-prod.yaml`:**

- `values-dev.yaml`: 1 replica, autoscaling disabled
- `values-prod.yaml`: 3 replicas, autoscaling enabled (2-8 pods, 70% CPU target)

### 2.2 — Local Testing with minikube (0.5 day)

**File:** `docs/k8s-local-setup.md`

```bash
minikube start --cpus=4 --memory=8192
eval $(minikube docker-env)
docker build -t agent-bench:latest -f docker/Dockerfile .

# Deploy (dev)
helm install agent-bench k8s/helm/agent-bench/ \
  -f k8s/helm/agent-bench/values-dev.yaml \
  --set provider.selfhosted.modalEndpoint=$MODAL_VLLM_URL

# Deploy (prod)
helm install agent-bench k8s/helm/agent-bench/ \
  -f k8s/helm/agent-bench/values-prod.yaml \
  --set provider.selfhosted.modalEndpoint=$MODAL_VLLM_URL

# Verify
kubectl get pods
kubectl port-forward svc/agent-bench-api 8080:8000
curl http://localhost:8080/health
```

---

## Work Package 3: Terraform IaC (1 day)

### 3.1 — GCP Configuration (CPU-only cluster)

**Directory:** `terraform/`

```
terraform/
  main.tf
  variables.tf
  outputs.tf
  terraform.tfvars.example
  modules/
    gke/
      main.tf
      variables.tf
      outputs.tf
    networking/
      main.tf
      variables.tf
```

**`main.tf`:**

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

module "networking" {
  source       = "./modules/networking"
  project_id   = var.project_id
  region       = var.region
  cluster_name = var.cluster_name
}

module "gke" {
  source       = "./modules/gke"
  project_id   = var.project_id
  region       = var.region
  cluster_name = var.cluster_name
  network      = module.networking.network_name
  subnetwork   = module.networking.subnetwork_name
  cpu_node_count    = 2
  cpu_machine_type  = "e2-standard-4"
}
```

### 3.2 — Validation

Run `terraform validate` and `terraform plan` (no apply). Include plan output summary in README to prove structural coherence without cloud spend.

---

## Architecture Diagram

```
+---------------------------------------------------------+
|  Terraform (GCP)                                        |
|  +---------------------------------------------------+  |
|  |  GKE Cluster (CPU only)                           |  |
|  |  +-------------------+                            |  |
|  |  |  API Pods (x2+)   |---- HTTPS ------+         |  |
|  |  |  - FastAPI        |                  |         |  |
|  |  |  - FAISS index    |                  |         |  |
|  |  |  - BM25 index     |                  |         |  |
|  |  +--------+----------+                  |         |  |
|  |           | HPA (CPU %)                 |         |  |
|  |  +--------+----------+                  |         |  |
|  |  |  Service (LB)     |                  |         |  |
|  |  +--------+----------+                  |         |  |
|  +-----------+------------------------------+--------+  |
+--------------+------------------------------+----------+
               |                              |
          Client / curl                +------+-------------+
                                       |  Modal (external)  |
                                       |  +--------------+  |
                                       |  | vLLM (A10G)  |  |
                                       |  | Mistral-7B   |  |
                                       |  | /v1/chat/... |  |
                                       |  +--------------+  |
                                       +--------------------+
```

**Why this split:** The API layer is CPU-bound and benefits from horizontal scaling via K8s HPA. The LLM inference layer is GPU-bound and benefits from serverless elasticity (Modal scales to zero when idle). Co-locating both in K8s would require GPU node pools with idle cost, node autoscaler latency, and NVIDIA device plugin management. This mirrors production patterns where API/orchestration runs on K8s while inference hits dedicated GPU platforms.

---

## DECISIONS.md Additions

1. **Why vLLM over TGI/llama.cpp:** Widest model support, best throughput (PagedAttention), native OpenAI-compatible server.
2. **Why Modal for GPU inference:** Serverless GPU eliminates idle cost. A10G at ~$1.30/hr, ~$0.50 per full benchmark run. Docker Compose path retained for local GPUs.
3. **Why split topology (K8s API + Modal GPU):** See architecture rationale. GPU nodes in GKE documented as valid production alternative for sustained utilization.
4. **Why Helm only, not Kustomize + Helm:** Showing two K8s deployment methods for the same app adds complexity without demonstrating distinct skills. Helm with `values-dev.yaml` / `values-prod.yaml` covers environment-specific configuration cleanly. Saves half a day of implementation.
5. **Why GCP over AWS:** GKE's simpler setup, per-second billing. Terraform modules structured so EKS swap is a module replacement.
6. **Why CPU-based HPA, not custom metrics:** Works without Prometheus adapter. Custom-metrics HPA via /metrics documented as follow-up.
7. **Why env var fallback in SelfHostedProvider:** Follows existing pattern (OpenAIProvider reads OPENAI_API_KEY). No config loader changes needed.
8. **Why startup smoke test for tool-call detection:** Checking `/v1/models` metadata for tool-calling support is unreliable — model metadata doesn't consistently report this capability. Instead, send one tool-calling request at provider init and check if the response contains `tool_calls`. Cache as `self.supports_tool_calling`. This is a runtime capability check, not a guess from metadata.

---

## CI Impact

- No CI changes for K8s/Terraform (declarative files). Optional: add `helm lint`, `helm template`, and `terraform validate` CI steps.
- SelfHostedProvider tests use `httpx.MockTransport` — no GPU/vLLM/Modal in CI.
- Modal deployments are manual. Benchmark run once, results committed.

**New Makefile targets:**

```makefile
modal-deploy:       ## Deploy vLLM on Modal
	modal deploy modal/serve_vllm.py

modal-stop:         ## Stop Modal deployment
	modal app stop agent-bench-vllm

vllm-up:            ## Start local vLLM via Docker Compose (requires NVIDIA GPU)
	docker compose -f docker/docker-compose.vllm.yml up --build

benchmark-all:      ## Run provider comparison (requires Modal + API keys)
	python modal/run_benchmark.py --base-url $(MODAL_VLLM_URL)

k8s-dev:            ## Deploy to minikube (dev values)
	helm install agent-bench k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-dev.yaml

k8s-prod:           ## Deploy via Helm (prod values)
	helm install agent-bench k8s/helm/agent-bench/ -f k8s/helm/agent-bench/values-prod.yaml

tf-plan:            ## Run terraform plan (no apply)
	cd terraform && terraform plan

tf-validate:        ## Validate terraform syntax
	cd terraform && terraform validate
```

---

## Final Project Structure

```
agent_bench/
  core/
    providers/
      openai.py              # Existing
      anthropic.py           # Existing (fully implemented)
      selfhosted.py          # NEW
      mock.py                # Existing
  agents/                    # Unchanged
  tools/                     # Unchanged
  rag/                       # Unchanged
  evaluation/                # Unchanged
  serving/                   # Unchanged
modal/                       # NEW
  serve_vllm.py
  run_benchmark.py
  common.py
docker/
  docker-compose.yaml        # Existing
  docker-compose.vllm.yml    # NEW
k8s/                         # NEW
  helm/agent-bench/
    Chart.yaml
    values.yaml
    values-dev.yaml
    values-prod.yaml
    templates/
terraform/                   # NEW
  main.tf
  variables.tf
  outputs.tf
  terraform.tfvars.example
  modules/
    gke/
    networking/
configs/
  openai.yaml                # Existing
  anthropic.yaml             # Existing
  selfhosted_local.yaml      # NEW
  selfhosted_modal.yaml      # NEW
docs/
  benchmark_report.md        # Existing
  provider_comparison.md     # NEW
  k8s-local-setup.md         # NEW
tests/
  test_selfhosted_provider.py  # NEW (8-10 mock tests)
```

---

## Commit Strategy

| # | Content | Tests | GPU? |
|---|---------|-------|------|
| 1 | `SelfHostedProvider` + configs + mock tests | 8-10 new | No |
| 2 | `modal/serve_vllm.py` + `modal/common.py` | Manual deploy | Yes |
| 3 | `docker/docker-compose.vllm.yml` | Smoke test | No |
| 4 | `modal/run_benchmark.py` + `docs/provider_comparison.md` | Benchmark results | Yes |
| 5 | Helm chart (templates, values-dev, values-prod) | `helm template` | No |
| 6 | Terraform modules | `terraform validate` | No |
| 7 | README + DECISIONS.md + architecture diagram | - | No |

---

## Risks

- **Modal cold starts:** ~60-90s for model loading. `container_idle_timeout=300` keeps warm for 5 min. Only first benchmark request hits cold start.
- **Modal costs:** ~$0.50 per full benchmark run. Running all 3 providers costs ~$1.50 total.
- **vLLM tool calling:** Mistral-7B-Instruct support varies by vLLM version. Unreliable tool calling is a legitimate benchmark finding, not a failure. Provider falls back to prompt-based tool selection.
- **vLLM-Modal integration pattern:** The `@modal.asgi_app()` sketch may need adaptation. Check Modal's current vLLM example at implementation time. Key contract: expose `/v1/chat/completions` and `/health`.
- **Model selection:** Mistral-7B-Instruct-v0.3 chosen for A10G fit, instruction following, vLLM support. Architecture is model-agnostic; swap to newer model if better supported at implementation time.
