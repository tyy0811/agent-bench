# agent-bench V2 — Implementation Plan (Validated)

> **Rule: Do NOT start V2 until demandops-lite is shipped AND you've applied to 15+ jobs.**
> Each phase is independent. Ship one, commit, move on. Stop anytime.

---

## Current V1 Baseline

| Metric | V1 Value | Known weakness |
|--------|----------|---------------|
| Retrieval P@5 | 0.70 | BM25 noise, no reranking |
| Retrieval R@5 | 0.83 | Good |
| Citation accuracy | 1.00 | Perfect |
| Grounded refusal | 0/5 | **Biggest gap** — LLM never refuses |
| Calculator accuracy | 2/3 | LLM skips tool use sometimes |
| Latency p50 | 4,690 ms | Acceptable for gpt-4o-mini |
| Cost per query | $0.0004 | Excellent |
| Tests | 97 | All deterministic |

---

## Codebase Validation Notes (2026-03-24)

Validated against actual codebase. Key findings:

1. **RRF scores are unbounded** (0-2 range, formula `1/(k+rank)` with k=60). Not normalized 0-1. Threshold tuning must be empirical.
2. **SearchResult.score is dropped** in SearchTool.execute() — scores never reach orchestrator. Adding `max_score` to metadata is the critical fix.
3. **RerankerConfig stub exists** (`enabled: false` only). Must extend with model, top_k fields.
4. **sentence-transformers already includes CrossEncoder** — no new deps needed.
5. **Dockerfile already copies data/** — plan's "gotcha" is already handled.
6. **AnthropicProvider is a stub** raising NotImplementedError — full implementation needed for Phase 5.

---

## V2 Phases

### Phase 1 — Retrieval Quality (2 evenings)

#### 1A. Grounded Refusal Fix (Evening 1, ~2-3 hours)

**The problem:** The system retrieves tangentially related content for out-of-scope questions and synthesizes an answer instead of refusing. Grounded refusal rate is 0/5.

**The fix:** Add a relevance score threshold in SearchTool. If no retrieved chunk scores above the threshold, return "No relevant documents found" — the LLM then refuses via system prompt.

**Design decision: Refusal gate in SearchTool, not Orchestrator.**
SearchTool already handles empty results at lines 67-72. The refusal gate is a smarter version of the same logic. The orchestrator stays unchanged.

Flow:
1. Retriever returns `list[SearchResult]` with `.score` fields
2. SearchTool computes `max_score = max(r.score for r in results)`
3. If `max_score < config.rag.refusal_threshold` → return existing "No relevant documents found" with empty sources
4. LLM sees "No relevant documents found" → system prompt triggers refusal
5. Orchestrator doesn't change at all

```
Files to modify:
  agent_bench/rag/retriever.py    — no change needed (already returns scores)
  agent_bench/tools/search.py     — add max_score check + pass scores in metadata
  agent_bench/core/config.py      — add refusal_threshold to RAGConfig
  configs/default.yaml             — set threshold value
  tests/test_agent.py             — add refusal test

Implementation:
  1. In SearchTool.execute(), after getting results from retriever:
     max_score = max(r.score for r in results) if results else 0.0
  2. If max_score < config threshold, return:
     ToolOutput(success=True, result="No relevant documents found.",
                metadata={"sources": [], "max_score": max_score})
  3. Otherwise, include max_score in metadata alongside existing fields
  4. Config: add refusal_threshold to RAGConfig (default: 0.0 = disabled)

Tuning strategy:
  - Run evaluate-fast with threshold=0.0 (current behavior, 0/5 refusal)
  - Try threshold=0.01, 0.015, 0.02, 0.025, 0.03
  - Pick the value that maximizes refusal on out-of-scope questions
    without breaking in-scope retrieval
  - RRF scores are unbounded (0-2 range) — don't assume 0-1 normalization

Definition of done:
  - Grounded refusal >= 3/5 (up from 0/5)
  - No regression on in-scope P@5 and R@5
  - Benchmark report updated with before/after comparison
  - DECISIONS.md updated: "Why a relevance threshold for refusal"
```

#### 1B. Cross-Encoder Reranking (Evening 2, ~3-4 hours)

**The problem:** P@5 is 0.70. BM25 returns noisy results that dilute precision. The reranker is feature-flagged but not implemented.

**The fix:** Add `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking after RRF fusion.

```
Files to create:
  agent_bench/rag/reranker.py

Files to modify:
  agent_bench/rag/retriever.py    — call reranker if config.rag.reranker.enabled
  agent_bench/core/config.py      — add model field to RerankerConfig
  configs/default.yaml             — set reranker.enabled: true, model name
  tests/test_rag.py               — add reranker tests (mock the model)

Implementation:
  1. reranker.py:
     - Load CrossEncoder lazily (same pattern as embedder)
     - rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]
     - Uses cross_encoder.predict([(query, chunk.content) for chunk in chunks])
     - Sort by cross-encoder score descending, return top_k
     - CrossEncoder is already in sentence-transformers — no new dep
  2. retriever.py:
     - After RRF fusion returns candidates_per_system * 2 results
     - If reranker enabled: pass top 20 to reranker, return top 5
     - If disabled: return top 5 from RRF directly (current behavior)
  3. Tests: mock the CrossEncoder model (return deterministic scores)
  4. Dockerfile: add pre-download of cross-encoder model at build time

Benchmark comparison table to add:
  | Config | P@5 | R@5 | Latency p50 |
  |--------|-----|-----|-------------|
  | V1 (RRF only) | 0.70 | 0.83 | 4,690 ms |
  | V2 (RRF + reranker) | X.XX | X.XX | X,XXX ms |

Note: The reranker model is ~80MB and runs on CPU. Expect ~100ms
extra latency per query.

Definition of done:
  - P@5 improves (target: >= 0.80)
  - Reranker is togglable via config (enabled/disabled)
  - Benchmark report has before/after comparison table
  - DECISIONS.md updated: "Why reranking improves precision"
  - No regression on R@5 or citation accuracy
```

**Phase 1 README update:** After both features ship, update the benchmark table with V2 numbers and add a "V1 -> V2 Improvements" section showing the deltas.

---

### Phase 2 — Production Hardening (2 evenings)

#### 2A. Caching (Evening 3, ~2 hours)

**The problem:** Identical queries re-embed and re-retrieve every time.

```
Files to create:
  agent_bench/rag/cache.py

Files to modify:
  agent_bench/rag/retriever.py    — check cache before retrieval
  agent_bench/core/config.py      — add cache config (enabled, max_size)
  configs/default.yaml
  tests/test_rag.py               — cache hit/miss tests

Implementation:
  1. cache.py:
     - In-memory LRU cache keyed by (query_text, top_k, strategy)
     - max_size: 100 queries (configurable)
     - No TTL (static corpus doesn't change)
  2. retriever.py:
     - Before embedding + search: check cache
     - On hit: return cached results, log "cache_hit" via structlog
     - On miss: run full pipeline, store result, log "cache_miss"
  3. /metrics: add cache_hits_total and cache_misses_total counters

Definition of done:
  - Second identical query returns in <10ms
  - Cache hit/miss logged in structlog
  - Cache stats in /metrics
  - Test: two identical queries, second is a cache hit
```

#### 2B. Rate Limiting + Retry Logic (Evening 3, ~2 hours)

**The problem:** No protection against OpenAI 429s or consumer abuse.

```
Files to modify:
  agent_bench/core/provider.py    — add retry logic to OpenAIProvider
  agent_bench/serving/middleware.py — add rate limiter
  agent_bench/core/config.py      — add rate_limit and retry config
  tests/test_provider.py          — test retry behavior
  tests/test_serving.py           — test rate limit response

Implementation:
  1. Provider retry (in OpenAIProvider.complete):
     - Catch openai.RateLimitError (429)
     - Exponential backoff: wait 1s, 2s, 4s (max 3 retries)
     - If all retries fail, raise ProviderTimeoutError
     - Log each retry with structlog
  2. API rate limiter (in middleware.py):
     - In-memory token bucket or sliding window
     - Default: 10 requests/minute per IP (configurable)
     - On limit: return 429 with Retry-After header

Definition of done:
  - OpenAI 429 -> automatic retry with backoff (test with mock)
  - /ask rate limited at configurable threshold
  - 429 response includes Retry-After header
```

---

### Phase 3 — Retrieval Intelligence (1 evening)

#### 3A. Query Transformation (Evening 4, ~3-4 hours)

**The problem:** Hard questions get poor retrieval because the raw query doesn't match chunk vocabulary.

```
Files to create:
  agent_bench/rag/query_transform.py

Files to modify:
  agent_bench/rag/retriever.py    — call transformer before search
  agent_bench/core/config.py      — add query_transform config
  configs/default.yaml
  tests/test_rag.py               — transformation tests

Implementation:
  1. query_transform.py:
     Two strategies (configurable):
     a) LLM rewrite (default): gpt-4o-mini rewrites query for retrieval
     b) Multi-query expansion: generate 2-3 variants, merge results
  2. retriever.py: if enabled, transform before search
  3. Track original_query and transformed_query in response metadata

Definition of done:
  - Hard-question P@5 improves
  - Transformation is configurable (on/off)
  - Original + transformed query visible in response metadata
```

---

### Phase 4 — Cloud + Streaming (2 evenings)

#### 4A. Cloud Deployment to Fly.io (Evening 5, ~2-3 hours)

```
Steps:
  1. fly launch --name agent-bench --region fra
  2. fly secrets set OPENAI_API_KEY=sk-...
  3. Create fly.toml with Dockerfile build
  4. fly deploy
  5. Update README with live demo link

Definition of done:
  - https://agent-bench.fly.dev/health returns 200
  - https://agent-bench.fly.dev/ask accepts POST requests
  - README has live demo link
```

#### 4B. Streaming Responses (Evening 6, ~4-5 hours)

```
Files to create:
  agent_bench/serving/stream.py

Files to modify:
  agent_bench/core/provider.py    — add stream_complete() to LLMProvider
  agent_bench/agents/orchestrator.py — add run_stream() method
  agent_bench/serving/routes.py   — add /ask/stream endpoint
  agent_bench/serving/schemas.py  — add StreamEvent model
  tests/test_serving.py           — streaming test

Implementation:
  1. Provider: stream_complete() yields chunks from OpenAI streaming API
  2. Orchestrator: run_stream() streams only the FINAL answer (tool calls are not streamed)
  3. Route: POST /ask/stream returns SSE
  4. /ask (non-streaming) stays unchanged — /ask/stream is additive

Definition of done:
  - POST /ask/stream returns SSE with progressive chunks
  - Final event includes sources and metadata
  - Non-streaming /ask still works identically
```

---

### Phase 5 — Provider Comparison (1 evening, only if asked)

#### 5A. Anthropic Provider (Evening 7, ~4-5 hours)

```
Files to modify:
  agent_bench/core/provider.py    — implement AnthropicProvider

Key differences from OpenAI:
  - System message: system= parameter, not in messages list
  - Tool definition: "input_schema" not "parameters"
  - Tool result: content block with type="tool_result"
  - Stop reason: stop_reason == "tool_use"

Definition of done:
  - AnthropicProvider passes the same test suite as OpenAI
  - Benchmark report has provider comparison table
  - Config swap: change one YAML field to switch providers
```

---

## Phase Summary

| Phase | Features | Evenings | When |
|-------|----------|----------|------|
| **1** | Grounded refusal + reranking | 2 | First, if any V2 |
| **2** | Caching + rate limiting + retry | 2 | After Phase 1 |
| **3** | Query transformation | 1 | After Phase 2 |
| **4** | Cloud deploy + streaming | 2 | After Phase 2 |
| **5** | Anthropic provider | 1 | Only if asked |

**Total: 8 evenings. Phase 1 alone (2 evenings) fixes the two biggest benchmark weaknesses.**
