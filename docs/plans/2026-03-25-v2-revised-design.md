# agent-bench V2 — Revised Design (Corrected)

> **Context:** RAG agent evaluation benchmark targeting AI/ML engineering roles.
> **Constraint:** CPU-only (Intel i7, 16GB RAM). No discrete GPU.
> **Revision:** Cross-reviewed plan with 4 original corrections + 7 diagnostic fixes applied.

---

## Corrections Applied

**Original (codebase validation):**
1. **Refusal gate location** — `SearchTool.execute()`, not orchestrator. Scores are dropped at search.py:86-91; gate must fire before that.
2. **RRF score range** — Empirical sweep only, no prose claims about score ranges. Document actual distribution during tuning.
3. **RerankerConfig** — Add `top_k: int` field so reranker output count is independent of `retrieval.top_k`.
4. **Retry exceptions** — Reuse existing `ProviderRateLimitError` (already handled in middleware.py as 503). No new exception classes.

**Diagnostic (design review):**
5. **Retry wrapping order** — Catch `openai.RateLimitError` inside the raw API call, BEFORE it gets translated to `ProviderRateLimitError`. Otherwise retry logic is dead code.
6. **Refusal-reranker interaction** — Refusal gate fires on RRF `max_score` BEFORE reranking. If max_score >= threshold, the full RRF candidate set passes to the reranker. The gate is a go/no-go decision, not a per-chunk filter.
7. **Rate limiter memory** — Document unbounded IP growth as a known limitation. Acceptable for demo; production would use Redis.
8. **Fly.io RAM** — Start at 1GB, not 512MB. Two transformer models + FAISS + runtime easily exceeds 512MB.
9. **Dockerfile cross-encoder download** — Spell out the exact `RUN` command.
10. **Integration test** — Add test for refusal + reranker combined (out-of-scope query with reranker enabled still refuses).
11. **CI pip caching** — Add `actions/cache@v4` for pip dependencies.

---

## V1 Baseline

| Metric | V1 Value | Known Weakness |
|--------|----------|----------------|
| Retrieval P@5 | 0.70 | BM25 noise, no reranking |
| Retrieval R@5 | 0.83 | Good |
| Citation accuracy | 1.00 | Perfect |
| Grounded refusal | 0/5 | **Biggest gap** — LLM never refuses |
| Calculator accuracy | 2/3 | LLM skips tool use sometimes |
| Latency p50 | 4,690 ms | Acceptable for gpt-4o-mini |
| Cost per query | $0.0004 | Excellent |
| Tests | 97 | All deterministic |

---

## Feature Overview

| # | Feature | Evenings | Skill Signal | Tier |
|---|---------|----------|-------------|------|
| 1 | Grounded refusal | 1 | Trust & safety, hallucination prevention | **Core** |
| 2 | Cross-encoder reranking | 1 | Retrieval quality, precision engineering | **Core** |
| 3 | GitHub Actions CI | 0.5 | CI/CD, production hygiene | **Core** |
| 4 | Retry logic + rate limiting | 1 | Resilience, production hardening | **Core** |
| 5 | Fly.io deploy | 1 | Cloud deployment, live demo URL | **Core** |
| 6 | Streaming responses | 1 | Async Python, SSE, real-time UX | **Optional** |
| 7 | SQLite conversation sessions | 1 | State management, memory, persistence | **Optional** |
| B | Anthropic provider | 1 | Multi-provider abstraction | **Backlog** |

**Core: 4.5 evenings. Optional: 2 evenings. Backlog: 1 evening.**

---

## Feature 1 — Grounded Refusal (Evening 1, ~2-3 hours)

### Problem

The system retrieves tangentially related content for out-of-scope questions and
synthesizes an answer instead of refusing. Grounded refusal rate is 0/5.

### Where the gate goes (Correction #1)

The refusal gate belongs in `SearchTool.execute()` — NOT in the orchestrator.

**Why:** `SearchTool.execute()` (search.py:86-91) currently drops all scores
before returning results to the orchestrator. The orchestrator never sees scores.
The gate must fire while scores are still available.

### Interaction with reranking (Correction #6)

When both Feature 1 and Feature 2 are active, the refusal gate fires on RRF
`max_score` BEFORE reranking. The gate is a go/no-go decision, not a per-chunk
filter: if max_score >= threshold, the full RRF candidate set passes to the
reranker. This keeps the two features independent — the sweep calibration stays
valid regardless of whether reranking is enabled.

### Implementation

```
Files to modify:
  agent_bench/tools/search.py    — add max_score check before returning results
  agent_bench/core/config.py     — add refusal_threshold to RAGConfig
  configs/default.yaml           — set threshold value
  tests/test_agent.py            — add refusal tests (in-scope + out-of-scope)
  tests/test_tools.py            — add threshold unit tests

Steps:
  1. search.py — in SearchTool.execute(), after getting results from retriever:
     - Compute max_score = max(r.score for r in results) if results else 0.0
     - Log max_score via structlog for every query
     - If max_score < config.rag.refusal_threshold AND threshold > 0:
       → Return ToolOutput(
           success=True,
           result="No relevant documents found for this query.",
           metadata={"sources": [], "max_score": max_score, "refused": True}
         )
     - Otherwise: proceed with existing logic, but include max_score in metadata

  2. config.py — add to RAGConfig:
     refusal_threshold: float = 0.0  # 0.0 = disabled (V1 behavior preserved)

  3. configs/default.yaml:
     rag:
       refusal_threshold: 0.02  # tuned empirically via sweep

  4. Threshold tuning (Correction #2 — empirical only):
     - Run evaluate-fast with threshold=0.0 (current behavior, 0/5 refusal)
     - Sweep: 0.01, 0.015, 0.02, 0.025, 0.03
     - Pick value that maximizes refusal on out-of-scope questions
       WITHOUT breaking in-scope retrieval (no regression on P@5, R@5)
     - Log the actual RRF score distribution across all eval queries
     - Document chosen threshold + observed score distribution in DECISIONS.md
     - If no single threshold works: percentile-based fallback

  5. Tests:
     - test_refusal_out_of_scope: query about cooking → system refuses
     - test_no_refusal_in_scope: query about FastAPI auth → system answers
     - test_refusal_metadata: refused response includes max_score + refused=True
     - test_threshold_zero_disables: threshold=0.0 → never refuses (V1 behavior)
     - test_threshold_configurable: changing config changes behavior
```

### Definition of done

- Grounded refusal >= 3/5 (up from 0/5)
- No regression on in-scope P@5 (still >= 0.70) and R@5 (still >= 0.83)
- Benchmark report updated with before/after comparison
- DECISIONS.md entry with observed score distribution
- New tests pass

---

## Feature 2 — Cross-Encoder Reranking (Evening 2, ~3-4 hours)

### Problem

P@5 is 0.70. BM25 returns noisy results that dilute precision. The reranker is
feature-flagged in config but not implemented.

### Implementation

```
Files to create:
  agent_bench/rag/reranker.py

Files to modify:
  agent_bench/rag/retriever.py    — call reranker if config.rag.reranker.enabled
  agent_bench/core/config.py      — extend RerankerConfig with model + top_k
  configs/default.yaml             — set reranker.enabled: true
  docker/Dockerfile                — pre-download cross-encoder model
  tests/test_rag.py               — add reranker unit tests (mock the model)

Steps:
  1. reranker.py:
     - CrossEncoderReranker class
     - Lazy-load CrossEncoder (same pattern as embedder)
     - rerank(query, chunks, top_k) -> list[Chunk]
     - Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, CPU)

  2. config.py (Correction #3 — add top_k):
     class RerankerConfig(BaseModel):
         enabled: bool = True
         model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
         top_k: int = 5  # independent of retrieval.top_k

  3. retriever.py — after RRF fusion:
     - Pass all RRF-fused candidates to the reranker; let reranker.top_k
       handle output truncation
     - If reranker disabled: return retrieval.top_k from RRF directly

  4. Dockerfile (Correction #9 — explicit download command):
     Add build-time layer:
       RUN python -c "from sentence_transformers import CrossEncoder; \
           CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

  5. Tests (mock the cross-encoder — don't download model in CI):
     - test_reranker_reorders: mock scores → verify reordering
     - test_reranker_top_k: mock 20 inputs → verify 5 outputs
     - test_reranker_disabled: config.enabled=False → RRF order preserved
     - test_reranker_empty_input: empty list → empty list
     - test_refusal_with_reranker_enabled: out-of-scope + reranker on →
       still refuses (integration test for Feature 1 + 2 combined)
```

### Definition of done

- P@5 improves (target: >= 0.80)
- Reranker togglable via config (enabled/disabled)
- Benchmark report has before/after comparison table
- No regression on R@5 or citation accuracy
- DECISIONS.md entry: "Why reranking improves precision"
- Tests pass with mocked model

---

## Feature 3 — GitHub Actions CI (Evening 3 first half, ~1 hour)

### Problem

No automated testing on push. Highest signal-per-minute feature in the plan.

### Implementation (Correction #11 — pip caching)

```
File to create:
  .github/workflows/ci.yml

File to modify:
  README.md — add CI badge

ci.yml:
  name: CI
  on:
    push:
      branches: [main]
    pull_request:
      branches: [main]

  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4

        - uses: actions/setup-python@v5
          with:
            python-version: "3.11"

        - uses: actions/cache@v4
          with:
            path: ~/.cache/pip
            key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
            restore-keys: ${{ runner.os }}-pip-

        - run: pip install -e ".[dev]"
        - run: ruff check agent_bench/ tests/
        - run: mypy agent_bench/ --ignore-missing-imports
        - run: pytest tests/ -v --tb=short

    docker:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: docker build -f docker/Dockerfile -t agent-bench:ci .
        - run: |
            docker run --rm agent-bench:ci python -c \
              "from agent_bench import __version__; print(__version__)"
```

### Definition of done

- Green badge on GitHub repo
- Push to main triggers: lint → type check → 97+ tests → Docker build
- Badge visible in README

---

## Feature 4 — Retry Logic + Rate Limiting (Evening 3-4, ~3 hours)

### Problem

No protection against OpenAI 429 rate limit errors. No defense against
consumer abuse of the API.

### Part A: Provider Retry (~1.5 hours)

**Critical fix (Correction #5):** The retry must catch `openai.RateLimitError`
INSIDE the raw API call, BEFORE the existing error translation maps it to
`ProviderRateLimitError`. Otherwise the retry logic is dead code — every 429
immediately becomes a 503.

```
Files to modify:
  agent_bench/core/provider.py    — add retry loop inside OpenAIProvider
  agent_bench/core/config.py      — add RetryConfig
  tests/test_provider.py          — test retry behavior

Implementation:
  1. OpenAIProvider — restructure the try/except:

     Current flow:
       try:
           response = await client.chat.completions.create(...)
       except openai.RateLimitError:
           raise ProviderRateLimitError(...)  # immediate 503

     New flow:
       for attempt in range(max_retries + 1):
           try:
               response = await client.chat.completions.create(...)
               break  # success
           except openai.RateLimitError as e:
               if attempt == max_retries:
                   raise ProviderRateLimitError(...)  # exhausted → 503
               wait = min(base_delay * 2 ** attempt, max_delay)
               log.warning("provider_retry", attempt=attempt + 1,
                           wait_seconds=wait)
               await asyncio.sleep(wait)

     The retry wraps the raw openai call. ProviderRateLimitError is only
     raised after all retries are exhausted. Other exceptions (APITimeoutError,
     BadRequestError) still fail immediately via the existing except clauses.

  2. config.py:
     class RetryConfig(BaseModel):
         max_retries: int = 3
         base_delay: float = 1.0
         max_delay: float = 8.0

  3. Tests:
     - test_retry_on_rate_limit: mock openai.RateLimitError twice then
       success → returns answer (must mock at openai level, not
       ProviderRateLimitError level)
     - test_retry_exhausted: mock 4 failures → raises ProviderRateLimitError
     - test_no_retry_on_other_errors: mock BadRequestError → raises immediately
     - test_retry_backoff_timing: verify delays (mock asyncio.sleep)
```

### Part B: API Rate Limiting (~1.5 hours)

**Known limitation (Correction #7):** The in-memory sliding window dict grows
without bound across distinct IPs. Acceptable for a demo deployment with
auto-stop (memory resets on stop). Document in DECISIONS.md. Production would
use Redis.

```
Files to modify:
  agent_bench/serving/middleware.py    — add RateLimitMiddleware
  agent_bench/serving/app.py          — register middleware
  agent_bench/core/config.py          — add rate_limit_rpm to ServingConfig
  tests/test_serving.py               — test rate limit response

Implementation:
  1. RateLimitMiddleware:
     - In-memory sliding window, per-IP
     - Default: 10 requests/minute
     - /health and /metrics exempt
     - 429 response with Retry-After header

  2. Tests:
     - test_rate_limit_allows_normal_traffic: 5 requests → all 200
     - test_rate_limit_blocks_excess: 11 requests → 11th gets 429
     - test_rate_limit_retry_after_header: 429 has Retry-After
     - test_rate_limit_per_ip: two IPs each get full quota
     - test_health_exempt: /health never rate limited
```

### Definition of done

- OpenAI 429 → automatic retry with exponential backoff
- All retries exhausted → ProviderRateLimitError (503 via existing middleware)
- /ask rate limited at configurable RPM
- 429 response includes Retry-After header
- /health and /metrics exempt
- Both behaviors logged via structlog
- Tests pass with mocked providers and mocked time

### DECISIONS.md entries

```
## Provider retry with exponential backoff

OpenAI returns 429 (rate limit) errors under load. Without retry logic, a
single 429 causes a user-visible failure. We add exponential backoff:
attempt after 1s, 2s, 4s. After 3 retries, raise ProviderRateLimitError so
the middleware returns a clear 503.

The retry wraps the raw openai.RateLimitError — it must fire BEFORE the
error gets translated to ProviderRateLimitError, otherwise retry logic is
dead code. Other errors (400, 401, 500) fail immediately.

## API rate limiting

In-memory sliding window limiter: 10 requests/minute per IP. Sufficient for
a demo deployment; a production system would use Redis.

Known limitation: the per-IP dict grows without bound across distinct IPs.
Acceptable for Fly.io with auto-stop (memory resets). If running continuously
under bot traffic, add a periodic sweep or switch to TTL-based structure.
```

---

## Feature 5 — Fly.io Deployment (Evening 5, ~2-3 hours)

### Problem

No live demo URL.

### Implementation (Correction #8 — 1GB RAM)

```
Files to create:
  fly.toml

Files to modify:
  docker/Dockerfile     — ensure data/ and models included, add startup warmup
  README.md             — add live demo link + curl examples

fly.toml:
  app = "agent-bench"
  primary_region = "fra"

  [build]
    dockerfile = "docker/Dockerfile"

  [http_service]
    internal_port = 8000
    force_https = true
    auto_stop_machines = "stop"
    auto_start_machines = true
    min_machines_running = 0

  [env]
    AGENT_BENCH_ENV = "production"
    PYTHONUNBUFFERED = "1"

  [[vm]]
    size = "shared-cpu-1x"
    memory = "1024mb"  # Correction #8: 512MB is insufficient for
                       # embedder (~100MB) + reranker (~80MB) + FAISS
                       # + Python runtime. 1GB is still free tier.

Steps:
  1. fly launch --name agent-bench --region fra --no-deploy
  2. fly secrets set OPENAI_API_KEY=sk-...
  3. Startup warmup handler to eager-load embedding model + reranker
  4. fly deploy
  5. Verify: /health, /ask with in-scope + out-of-scope queries
  6. README: live demo link, curl examples, cold start note

Cost: ~$0/month (free tier + auto-stop), ~$0.04/month at 100 queries.
```

### Definition of done

- https://agent-bench.fly.dev/health returns 200
- /ask returns answers, grounded refusal works, rate limiter active
- README has live demo link with curl examples
- Cold start < 15s, warm requests match local latency (+ ~50ms network)

---

## Optional Features (after core milestone)

### Feature 6 — Streaming Responses (Evening 6, ~4 hours)

- Add `stream_complete()` to LLMProvider interface
- Stream only the final synthesis (tool calls are fast, ~100ms)
- SSE via `POST /ask/stream`, additive — `/ask` unchanged
- MockProvider yields 3 deterministic chunks for testing

### Feature 7 — SQLite Conversation Sessions (Evening 7, ~3 hours)

- `ConversationStore` backed by SQLite
- `session_id` parameter on `/ask` (None = stateless V1 behavior)
- Load history, prepend to messages, store question + answer
- Tests: append/retrieve, max_turns, session isolation, stateless fallback

### Backlog B — Anthropic Provider (only if asked)

- Implement `AnthropicProvider` (currently stub raising NotImplementedError)
- Key API differences: system parameter, input_schema, tool_result blocks
- Same test suite as OpenAI, config swap via one YAML field

---

## Implementation Order

```
Evening 1:   Feature 1 (Grounded refusal)         → commit, push
Evening 2:   Feature 2 (Reranking)                 → commit, push, update benchmark
Evening 3:   Feature 3 (CI) + Feature 4 (start)    → CI green, start retry logic
Evening 4:   Feature 4 (finish rate limiting)       → commit, push
Evening 5:   Feature 5 (Fly.io deploy)             → deploy, verify, update README
— MILESTONE: Core V2 shipped. Update README with V2 benchmark table. —
Evening 6:   Feature 6 (Streaming)                 → optional
Evening 7:   Feature 7 (SQLite sessions)           → optional
```

After Evening 5: stop building and apply unless you have spare evenings.

---

## V2 Benchmark Table (update after all features ship)

| Metric | V1 | V2 | Delta |
|--------|----|----|-------|
| P@5 | 0.70 | X.XX | +X.XX |
| R@5 | 0.83 | X.XX | +/-X.XX |
| Citation accuracy | 1.00 | X.XX | +/-X.XX |
| Grounded refusal | 0/5 | X/5 | +X |
| Calculator accuracy | 2/3 | X/3 | +/-X |
| Latency p50 | 4,690ms | X,XXXms | +/-Xms |
| Cost per query | $0.0004 | $X.XXXX | +/-$X.XXXX |
| Tests | 97 | XXX | +XX |
| Live demo URL | n/a | yes | New |
| CI/CD | n/a | yes | New |
| Provider retry | n/a | yes | New |
| Rate limiting | n/a | yes | New |
