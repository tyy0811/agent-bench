# Design Decisions

## Why build from primitives, not LangChain?

I wanted to demonstrate I understand tool dispatch, memory management,
and retrieval orchestration at the implementation level. My provider
abstraction is ~150 lines. When reranking or a second provider is needed,
I know exactly where it plugs in — because I built every layer.

## Why one provider in V1?

The interface supports multiple providers. V1 shipped OpenAI + Mock to
prove the abstraction. V2 added Anthropic (claude-haiku-4-5), confirming
that switching providers is a one-line config change. The orchestrator
and tools are completely unchanged between providers.

## Why one domain (technical docs)?

Clean text produces clean evaluation. Research papers introduce PDF
parsing noise (tables, figures, formulas) that degrades eval quality
without adding signal. The framework handles any text corpus — the
domain is a config swap.

## Why Reciprocal Rank Fusion, not score normalization?

BM25 scores and cosine similarities live on different scales.
Normalizing across scales is brittle: min-max normalization is
sensitive to outliers, z-score requires distribution assumptions.
RRF fuses by rank position `1/(k + rank)`, which is robust,
parameter-light (only k=60), and well-studied. Trade-off: loses
magnitude information, but for top-5 retrieval this rarely matters.

## Why ~16 curated docs, not a large corpus?

Small corpus produces predictable retrieval, reproducible benchmarks,
and easy debugging. Golden dataset questions map to specific source
files. A reviewer can verify any result by reading the source. The
framework scales to larger corpora — the choice is about evaluation
quality, not capability.

## Why no reranker in V1?

Feature-flagged in config (`rag.reranker.enabled: false`). V1
benchmarks without reranking establish an honest baseline. V2 adds
cross-encoder reranking and shows the delta.

## Why no delete in the vector store?

FAISS flat index doesn't support efficient deletion. For a small
corpus that changes rarely, rebuild-on-ingest is simpler and
eliminates consistency bugs.

## Why async internals, sync user behavior?

FastAPI and the OpenAI SDK are async-native. Using async for I/O
avoids blocking the event loop. V2 added SSE streaming (`/ask/stream`)
for the final synthesis step — tool calls remain non-streamed since
they complete in ~100ms.

## Why SQLite-backed conversation sessions

V1 was stateless by design — no conversation_id, no cross-request
memory. V2 adds optional SQLite-backed sessions: pass `session_id`
on `/ask` to persist and load conversation history. When omitted,
behavior is identical to V1 (stateless). See the dedicated
DECISIONS.md entry under "Why SQLite for conversation persistence"
for the full rationale.

## Why negative evaluation cases?

A system that always answers sounds confident but may hallucinate.
5 out-of-scope questions test whether the system refuses gracefully
when the corpus doesn't contain the answer. Grounded refusal
requires both refusal language AND zero cited sources — an answer
that says "not found" but still cites docs is not a valid refusal.

## Why deterministic eval + optional LLM judge?

CI needs free, deterministic tests. Retrieval P@5, keyword hit
rate, citation accuracy, and grounded refusal rate run without
API keys. LLM-judged faithfulness and correctness are manual
enrichment steps, run locally, with results committed to the
benchmark report.

## Why structlog, not LangSmith/Langfuse?

Third-party observability contradicts the "built from primitives"
narrative. structlog provides JSON-structured logs, `/metrics`
exposes latency + cost. If a team uses LangSmith, adding it is
a one-day task.

## Why explicit citation format [source: filename.md]?

The system prompt mandates inline citations so the evaluation
harness can parse them with `\[source:\s*(.+?)\]` and check
against the structured sources list. This catches hallucinated
citations and measures citation accuracy as a metric.

## Why per-request retrieval settings via kwargs, not singleton mutation?

The orchestrator is a singleton shared across concurrent requests.
Storing `top_k` / `strategy` on `self` causes cross-request state
bleed. Instead, these are passed as local variables through the
tool execution kwargs — no shared state is mutated.

## Why a relevance threshold for grounded refusal

V1 never refuses — it always retrieves tangentially related content and
synthesizes an answer. This is a trust failure: users cannot distinguish
"the system found relevant information" from "the system fabricated from
vaguely related chunks." Grounded refusal rate was 0/5.

We add a refusal gate in `SearchTool.execute()` based on the maximum RRF
score across retrieved chunks. If no chunk scores above the threshold, the
tool returns "No relevant documents found" — the LLM then refuses via the
system prompt rather than fabricating from irrelevant content.

**Gate location:** The gate fires in `SearchTool.execute()`, not the
orchestrator. `SearchTool` is where retrieval scores are still available —
they are dropped before results reach the orchestrator. This also keeps
the orchestrator unchanged.

**Threshold value:** `rag.refusal_threshold: 0.02` is a provisional default
pending an empirical sweep across the evaluation set. The sweep will test
values 0.01–0.03 and select the value that maximizes refusal on out-of-scope
queries without degrading in-scope P@5 and R@5. The actual RRF score
distribution will be documented here after tuning.

**Interaction with reranking:** The refusal gate fires on RRF scores BEFORE
reranking. It is a go/no-go decision, not a per-chunk filter. If the gate
passes, the full candidate set proceeds to the reranker. This keeps the
threshold calibration independent of whether reranking is enabled.

**Default disabled:** `refusal_threshold: 0.0` preserves V1 behavior exactly.
The feature is opt-in until the threshold is tuned.

**Alternative considered:** LLM-based relevance judgment ("is this content
relevant to the query?"). Rejected because it adds latency, cost, and a
second point of failure. The score-based approach is deterministic, fast,
and debuggable.

## Why cross-encoder reranking improves precision

BM25 retrieves lexically similar but semantically irrelevant chunks.
RRF fusion mitigates this partially, but noisy BM25 results still
dilute the top-5 set. P@5 was 0.70 in V1.

A cross-encoder (`ms-marco-MiniLM-L-6-v2`, ~80MB) scores each
(query, chunk) pair jointly, capturing semantic relevance that
bi-encoder similarity misses. The tradeoff is ~100–200ms extra latency
per query — acceptable given our 4.7s baseline is dominated by LLM
generation, not retrieval.

The reranker is enabled by default. Setting `rag.reranker.enabled: false`
restores V1 behavior exactly. `reranker.top_k` is independent of
`retrieval.top_k`, so the reranker's output count can be tuned without
affecting the RRF candidate pool.

The retriever passes all RRF-fused candidates to the reranker rather
than a computed subset. The reranker's `top_k` handles truncation.
This is simpler and more robust than computing an input size from
per-system candidate counts.

## Why provider retry with exponential backoff

OpenAI returns 429 (rate limit) errors under load. Without retry logic,
a single 429 causes a user-visible failure. We add exponential backoff:
attempt after 1s, 2s, 4s. After 3 retries, raise `ProviderRateLimitError`
so the middleware returns a clear 503.

The retry wraps the raw `openai.RateLimitError` — it must fire BEFORE
the error gets translated to `ProviderRateLimitError`, otherwise retry
logic is dead code. Other errors (400, 401, timeout) fail immediately.

## Why in-memory API rate limiting

A public-facing API needs abuse protection. We use a simple in-memory
sliding window limiter: 10 requests/minute per IP. Sufficient for a
demo deployment; a production system would use Redis.

Known limitation: the per-IP dict grows without bound across distinct
IPs. Acceptable for Fly.io with auto-stop (memory resets). If running
continuously under bot traffic, add a periodic sweep or switch to a
TTL-based structure.

Design choices:
- `/health` and `/metrics` exempt: monitoring should never be rate-limited.
- `Retry-After` header: follows HTTP 429 spec, lets clients back off.

## Why SQLite for conversation persistence

Three options considered:
1. In-memory dict: Lost on restart.
2. SQLite: Zero-dependency, file-based, survives restarts.
3. Redis/PostgreSQL: Adds infrastructure complexity.

SQLite is right for this scale. `session_id` is optional — when omitted,
the system behaves identically to V1 (stateless). This preserves backward
compatibility and keeps benchmark evaluation deterministic.

The route handler manages session state (load history, store Q+A), not
the orchestrator. The orchestrator accepts an optional `history` parameter
but has no knowledge of persistence. This keeps the agent loop testable
without a database.

Note: On HF Spaces, SQLite is ephemeral (no persistent storage on free
tier). For the demo this is acceptable — sessions last until the container
sleeps. Production would use a volume or managed database.

## Why a second provider (Anthropic)

The provider abstraction existed since V1 but only had OpenAI + Mock.
Adding Anthropic proves the abstraction works across fundamentally
different APIs:

- System message: `system=` parameter, not in the messages list
- Tool definitions: `input_schema` instead of `parameters`
- Tool results: `tool_result` content blocks in user messages
- Tool calls: `tool_use` content blocks, not a separate field
- Stop reason: `tool_use` vs `stop`

The implementation is a config swap — `provider.default: anthropic` in
YAML switches the entire system to Claude. The orchestrator, tools,
evaluation harness, and serving layer are completely unchanged.

Same retry/timeout handling as OpenAI. Both providers are tested with
mocked HTTP responses — no API keys needed in CI.

## Why ranked_sources separate from deduplicated sources?

The deduplicated `sources` list in `AgentResponse` is for the API
response. The `ranked_sources` list preserves rank order with
duplicates for evaluation metrics. P@5 and R@5 need the raw
retrieval ranking, not the post-processed answer metadata.
