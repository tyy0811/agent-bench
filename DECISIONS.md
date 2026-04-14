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

## Why vLLM over TGI / llama.cpp

vLLM has the widest model support, best throughput via PagedAttention, and a native
OpenAI-compatible server (`/v1/chat/completions`). TGI is a valid alternative; llama.cpp
targets different use cases (edge/CPU inference). This is a deliberate choice, not
ignorance of alternatives.

## Why Modal for GPU inference

Serverless GPU eliminates idle cost and GPU node management. A10G at ~$1.30/hr costs
~$0.50 per full 27-question benchmark run. The Docker Compose path (`docker-compose.vllm.yml`)
is retained for users who have local GPUs or prefer persistent serving.

## Why split topology (K8s API + Modal GPU)

The API layer (retrieval, orchestration, tool routing) is CPU-bound and benefits from
horizontal scaling via K8s HPA. The LLM inference layer is GPU-bound and benefits from
serverless elasticity — Modal scales to zero when idle, scales up on demand with no node
provisioning. Co-locating both in K8s would require GPU node pools with idle cost,
node autoscaler latency, and NVIDIA device plugin management. This mirrors a common
production pattern.

## Why Helm only, not Kustomize + Helm

Showing two K8s deployment methods for the same app adds complexity without demonstrating
distinct skills. Helm with `values-dev.yaml` / `values-prod.yaml` covers
environment-specific configuration cleanly.

## Why CPU-based HPA, not custom metrics

CPU utilization works without a Prometheus adapter or custom metrics server. A production
improvement would use the Prometheus adapter to scale on p95 latency from the `/metrics`
endpoint — this requires bridging the JSON metrics to Prometheus exposition format.
Documented as a follow-up.

## Why env var fallback in SelfHostedProvider

Follows the same pattern as OpenAIProvider reading `OPENAI_API_KEY`. The YAML config
provides defaults; env vars override at runtime. No config loader changes needed.

## Why lazy tool-call detection, not metadata check

Checking `/v1/models` metadata for tool-calling support is unreliable — model metadata
doesn't consistently report this capability. Instead, the provider sends one tool-calling
request on first `complete()` call with tools and checks if the response contains
`tool_calls`. The result is cached as `self._supports_tool_calling`. Transient failures
(timeout, 5xx) return `None` and retry on the next call rather than permanently
downgrading to prompt-based fallback.

## Why two-tier injection detection, not three

The original design included a middle tier (embedding similarity against known injection examples). Dropped because the existing embedding model (all-MiniLM-L6-v2) is a general-purpose sentence encoder, not specialized for adversarial detection. Cosine similarity can't distinguish semantic similarity from intent similarity — "how do I ignore a field in Pydantic?" clusters near "ignore previous instructions" in that embedding space. The threshold between "ambiguous" and "suspicious" is an untunable hyperparameter with no ground truth.

Two tiers are cleaner: heuristic regex is deterministic (matches or doesn't), DeBERTa classifier is probabilistic (confidence score). No ambiguous handoff between two probabilistic layers. Deployments without GPU get heuristic-only — documented, not hidden.

## Why regex + optional spaCy for PII, not a cloud API

Three reasons: cost (cloud PII APIs charge per call), latency (adds network round-trip to every retrieved chunk), and data residency (PII leaves the system boundary). Regex covers the PII types with actual legal/compliance risk: SSNs, credit cards, emails, phone numbers, IP addresses.

spaCy NER (PERSON, ORG) is optional because false-positive rates on technical text are unacceptable without domain tuning. "FastAPI" triggers ORG, "Jordan" triggers PERSON. The optional import pattern (`try: import spacy`) degrades gracefully with a logged warning — no crash if someone sets `use_ner: true` without installing spaCy.

## Why append-only JSONL for audit, not SQLite

One codepath, one format, no config branching. JSONL is append-only by nature — no schema migrations, no transactions, no connection pooling. Log rotation handles size. `jq` provides immediate queryability without building a custom API.

The original design included an optional SQLite backend and a query endpoint (`GET /admin/audit`). Both were dropped: SQLite adds a second storage codepath with no consumer, and the query endpoint would require API key authentication — an inconsistency when `/ask` itself has no auth.

JSONL imports trivially into SQLite/DuckDB if structured queries are needed later. No bridges burned.

## Why HMAC-SHA256 IP hashing in audit logs

HMAC-SHA256 with a server secret hashes client IPs before logging. Plain SHA-256 was considered but rejected: the IPv4 address space (~4.3 billion) is small enough that unsalted hashes are reversible by offline enumeration. HMAC-SHA256 with a secret key makes precomputation infeasible without the key. The key is sourced from an explicit parameter, `AUDIT_HMAC_KEY` env var, or (with a logged warning) a random per-process fallback.

## Why three output validators, not four

The original design included a "length/format sanity check" (reject suspiciously short responses or raw JSON in natural-language context). Dropped because the calculator tool returns short numeric answers and the tech docs domain legitimately contains code blocks and JSON examples. Every false positive erodes trust in the validation layer. The three remaining checks — PII leakage, URL hallucination, blocklist — are deterministic with clear pass/fail semantics.

## Why buffer-then-validate for streaming output

The `/ask/stream` endpoint buffers all events from the orchestrator before sending to the client, then validates the assembled answer. This means the client waits for the full answer before receiving any content chunks. The orchestrator emits the final synthesis as a single chunk (tool-use iterations are not streamed), so the buffering adds no perceptible latency. The alternative — streaming chunks immediately and appending a safety marker — leaks unsafe content to any client that stops reading after the `done` event.

## Why no authentication on API endpoints

The HF Spaces demo is public by design — the `curl` examples in the README work without credentials, which is the point. Adding API key authentication would gate access but break the zero-friction demo experience that makes the project evaluable.

The security pipeline protects *content* (injection detection, PII redaction, output validation), not *access*. This is a deliberate scope boundary: application-layer guardrails ensure the system behaves safely regardless of who calls it, rather than assuming trusted callers. Rate limiting (10 RPM per IP) provides basic abuse protection.

A production deployment would add authentication (API keys or OAuth) at the infrastructure layer — reverse proxy, API gateway, or middleware. The security pipeline's `getattr(..., None)` pattern means auth can be layered on without modifying the existing security components.

## Why monitor mode for output validation, not gating?

Output validation runs post-stream as a monitoring layer. The answer
streams to the client, then validation runs and emits its verdict. Gating
(buffer-then-validate) would add 4-5 seconds of dead air while the full
answer generates — unacceptable streaming UX for a documentation Q&A bot.
Trade-off: a hallucinated URL or PII fragment could reach the client
before validation catches it. For this use case (FastAPI docs, no real
PII in corpus), the risk is near-zero. The dashboard labels this
"monitored" (not "gated") to be explicit about the posture.

## Why additive SSE stage events?

The enhanced `/ask/stream` adds `meta` and `stage` event types alongside
the existing `sources`, `chunk`, and `done` events. Existing consumers
that only handle the three legacy types are unaffected — they simply
ignore events with unknown types. This avoids versioning the endpoint
or breaking the non-streaming `/ask` contract. The `meta` event fires
first (before any stages) so the frontend can display provider/model
info immediately.

## Why vanilla JS for the frontend, not Alpine or React?

The showcase dashboard has ~5 pieces of reactive state (pipeline stages,
retrieval results, security badges, stats, chat messages). The SSE
handler is inherently imperative: receive event, querySelector the
target node, update classList and textContent. Wrapping this in a
reactive framework adds a dependency, interview questions about
"why is there a framework for 5 state variables", and indirection
that fights the imperative SSE pattern. One `state` object + a few
`render()` functions handles it in ~150 lines.

## Phase 1 SSE gate closure — two baselines on record, not one

The Phase 1 acceptance gate for the SSE backend work (meta event,
stage events, iteration-aware metadata threading, route-level
injection/output-validation events) requires re-running
`make evaluate-fast` and confirming numbers match pre-change state
on the pinned `gpt-4o-mini-2024-07-18` snapshot. The re-run was
honored literally rather than substituted with a git-diff
argument, even though the SSE commits did not touch
`scripts/evaluate.py`'s legacy code path. Two reasons: the
re-commitment discipline that kept Fix 1 and Fix 2 honest applies
equally here, and the legacy path and the `--corpus fastapi` path
produce materially different baselines that cannot substitute for
each other.

**Two distinct baselines now exist at the pinned snapshot, and
both are on record** — one per prompt path:

| Baseline file | Invocation | Prompt source | In-scope P@5 | In-scope R@5 | Citation | Mean calls |
|---|---|---|---|---|---|---|
| `results/fastapi_preedit.json` @ `bd2b913` | `--corpus fastapi` | `format_system_prompt("FastAPI")` | 0.718 | 0.833 | 1.000 | 1.14 |
| `results/fastapi_legacy_baseline_pinned.json` @ this commit | `make evaluate-fast` (no `--corpus`) | `tech_docs.yaml` `task.system_prompt` | 0.655 | 0.849 | 1.000 | 1.45 |

Citation accuracy holds at 1.000 on both paths, both in-scope and
out-of-scope. The retrieval metric deltas (P@5 −0.063, R@5 +0.016,
KHR +0.045) and behavioral delta (mean tool calls +0.318 in-scope,
+1.00 out-of-scope) trace to the prompt-path divergence
(`scripts/evaluate.py:67` reads `task.system_prompt` in the legacy
branch vs. `format_system_prompt(label)` in the `--corpus` branch),
not to any change in retrieval, reranking, or refusal-gate code.
This divergence is the same one the "evaluation-layer multi-corpus
support lagged the serving-layer refactor" entry documents; the
narrowed serving-migration deferral tracks its eventual migration.

**Why both baselines are retained.** When the serving-migration
deferral lands and `scripts/evaluate.py`'s legacy branch is removed
(everything routes through `--corpus fastapi`), the regression gate
is "post-migration `make evaluate-fast` output matches pre-migration
`--corpus fastapi` output within pre-committed tolerances." That
gate requires the `--corpus fastapi` baseline as the comparison
reference AND the legacy baseline as evidence of the pre-migration
state that is being retired. Retaining both makes the migration
auditable and bounds its regression budget; retaining only one
would force the post-migration run to compare against a baseline
from a different prompt path, guaranteeing the gate fires on
prompt divergence rather than on any actual regression.

**Gate verdict: passed.** No regression vs pre-SSE legacy path
expectations (citation 1.000 holds, refusal gate fires on the same
5 out-of-scope questions, retrieval numbers in sane in-scope
ranges). Phase 1 SSE backend work is closed from the backend side;
the frontend's consumption of iteration-aware stage events is
orthogonal and owned by Week 1 step 7 (showcase UI).

## Why per-corpus refusal thresholds?

FastAPI and Kubernetes have different corpus characteristics. FastAPI
has 16 short, well-structured docs with sparse cross-references —
relevance tends to concentrate in 1-2 chunks per query. Kubernetes
has 30-40 docs with heavy cross-referencing between concepts (Pod →
Deployment → Service → Ingress), which spreads relevance across more
chunks. A single global refusal threshold would either refuse too
aggressively on K8s (no single chunk dominates, so the top score
looks "low") or not aggressively enough on FastAPI (where a
moderate-scoring chunk might be the only hit and should still refuse).

`CorpusConfig` carries `refusal_threshold` as a per-corpus field.
Each threshold gets tuned against its own golden dataset — there
is no "fair" shared threshold because BEIR showed these are not
comparable across corpora. Placeholder values ship in default.yaml
and are replaced by tuned values during the per-corpus evaluation
sweep.

## Why corpus and provider toggles compose — corpus_map[corpus][provider]

The simpler design would have been `corpus_map[corpus]` returning a
single orchestrator. It ships in 10 fewer lines. It also silently
breaks the provider toggle in multi-corpus mode: the orchestrator
inside each corpus cell holds one fixed provider, and clicking
"Anthropic" in the dashboard keeps running on OpenAI.

This project's hero-tile metric is the provider comparison (`1.00 API /
0.14 7B self-hosted`). Breaking the mechanism that demonstrates that
metric — on a portfolio demo where a reviewer will open DevTools and
notice — would erode the honest-evaluation brand the whole repo is
built around. The nested `corpus_map[corpus][provider]` structure
keeps both toggles functional. Store, retriever, and search tool are
shared across providers within a corpus (the expensive objects are
held once per corpus); only the orchestrator varies per provider
since it holds the LLM client. Per-corpus × per-provider memory
overhead is an orchestrator struct, not a FAISS index.

RSS is logged per corpus, not per corpus × provider, because the
store is what drives memory. The provider multiplier is negligible
compared to a hybrid index + embedder.

## Why one parameterized system prompt, not per-corpus templates

The template is `"You are a technical documentation assistant for
{corpus_label}..."`. The only corpus-specific element is the label;
prompt content is identical across corpora: same citation format,
same refusal language, same grounding instructions. Having two
separate prompt files would invite drift — someone tweaks the FastAPI
prompt for a specific failure mode and forgets to update the K8s
version, and the demo silently answers differently on the two toggles.

The parameterization is enforced by two tests: (a)
`format_system_prompt("")` raises `ValueError` so an unresolved
`{corpus_label}` can never reach the LLM, and (b) a spy on
`orchestrator.run_stream` asserts FastAPI and K8s requests receive
different prompts with the correct label substituted.

The wording deliberately differs from the typical "don't hallucinate"
RAG template:

- **"refuse the question explicitly"** matches our refusal-gate
  mechanism. "Say so politely" is soft language that models interpret
  as "hedge and answer anyway".
- **"do not infer, do not extrapolate, do not draw on general
  knowledge"** is the three-verb prohibition. "Do not fabricate" is
  empirically easier to slip past because models distinguish
  fabrication (making things up) from extrapolation (drawing
  conclusions from adjacent but non-authoritative context).

## Why Kubernetes curation targets recruiter-likely questions, not coverage

The K8s corpus targets ~30-40 pages curated around concepts a
technical reviewer would naturally type (Pod, Deployment, Service,
Ingress, ConfigMap, RBAC) plus cross-referencing overview pages that
stress the reranker. Cluster administration deep-dives, tutorials,
and kubectl reference are explicitly excluded — they add noise without
adding reviewer value and hurt retrieval precision when adjacent
content is thin on concept definitions.

`data/k8s_docs/SOURCES.md` is a version-controlled curation artifact.
Each ingested URL has a one-line rationale, a date pulled, and a
license note. This makes the corpus reproducible and documents the
curation reasoning for any reviewer who looks closely.

Trade-off: the corpus is not comprehensive K8s knowledge. A question
about etcd raft internals will be correctly refused. This is not a
bug — the refusal is part of the demo story, and "the system knows
what it doesn't know" is a feature of the grounded-refusal mechanism.

## Why no cross-corpus score comparison (inspired by BEIR)

Inspired by BEIR's heterogeneous-benchmark framing (Thakur et al.,
NeurIPS 2021), which spans 18 datasets across 9 task types, absolute
retrieval scores are not treated as comparable across FastAPI and
K8s corpora — score distributions depend on chunk length, vocabulary
overlap, and corpus density, none of which are held constant across
domains. Only rank-ordering of system configurations within a single
corpus is meaningful. Concrete consequences for this repo:

- Per-corpus evaluation results are reported separately, never
  aggregated into a single "combined" number.
- The hero-tile citation accuracy (`1.00 API / 0.14 7B self-hosted`)
  stays FastAPI-specific. It is not restated as a cross-corpus average.
- `make evaluate-fast` accepts a `--corpus` flag but has no "combined"
  mode. Anyone who wants a cross-corpus number has to run twice and
  acknowledge the incomparability in prose.
- The landing page "Key Findings" cards avoid sentences that compare
  FastAPI and K8s numbers directly.

The multi-corpus demo is a **surface feature for interactive
exploration**, not a rebenchmark. The benchmark section of the README
remains FastAPI-only and cites 27 questions on 16 docs with specific
chunker settings.

## K8s golden dataset uses CRAG's 8-type taxonomy as the schema

The K8s golden dataset uses CRAG's 8-type taxonomy (Yang et al.,
NeurIPS 2024) **as the schema** for `question_type`, not as a
requirement to cover all 8 types. CRAG's taxonomy: `simple`,
`simple_w_condition`, `set`, `comparison`, `aggregation`,
`multi_hop`, `post_processing_heavy`, `false_premise`. Temporal
dynamism is a separate orthogonal property captured as
`time_sensitive: bool` on the question schema — it is not a CRAG
category.

Target distribution across the 25-question K8s golden set:

- `simple` (5–6): baseline retrieval
- `simple_w_condition` (3–4): nuanced understanding under conditions
- `comparison` (3–4): retrieval across concept pages, reranker stress
- `multi_hop` (5–6): synthesis across 2–4 docs, reranker stress
- `false_premise` (3–4): grounded refusal mechanism
- `set` / `aggregation` / `post_processing_heavy` (0–3): included
  only where corpus content naturally supports

`time_sensitive: bool` flags 2–3 questions targeting version-bounded
content (feature state, deprecations, API version migration).

`false_premise` questions come in two flavors (see separate
"False-premise questions come in two flavors" entry): pure refusal
(flavor A) and documented negative (flavor B). The K8s set includes
at least one of each. Flavor A tests the path where retrieval
correctly returns nothing useful; flavor B tests the path where the
corpus contains an explicit negative answer and the agent must
surface it with citation rather than confabulating a positive.

Rationale for using CRAG as schema (not coverage requirement):
`false_premise` and `time_sensitive` stress grounded refusal and
reduce test-set contamination risk; `multi_hop` and `comparison`
stress the reranker because relevance spreads across multiple
chunks. The distribution was chosen to exercise the parts of the
pipeline the benchmark story claims — not to mimic a general-purpose
QA benchmark.

The golden dataset JSON schema (v2, backward-compatible with the
FastAPI flat list) includes:

- `source_chunk_ids: list[str]` for multi-hop partial credit
  (answer must cite at least one of the expected chunks)
- `source_snippets: list[str]` for human-readable context during
  review
- `question_type: str` (CRAG taxonomy value)
- `is_multi_hop: bool` for filtered reporting
- Dataset-level header with `corpus`, `version`, `snapshot_date`,
  and pinned `chunker` parameters so the dataset is reproducible
  against a specific K8s docs snapshot

See `docs/plans/2026-04-12-multi-corpus-refactor-design.md` for the
full schema and rationale.

## EU AI Act corpus deferred to v1.2

EU AI Act compliance mapping is deferred to v1.2. Rationale: v1
ships two corpora (FastAPI, K8s) to demonstrate the multi-corpus
architecture; EU AI Act as a third corpus would add ingestion and
golden-set work without exercising architecturally new surface.
Scoped as the first v1.2 addition after v1 launch.

## Cold-start contingency: measure first, lazy-load if needed

Loading two corpora at startup costs memory and cold-start time. On
HF Spaces (target deployment), the realistic ceiling is 8-10 GB
resident RAM and ~60 seconds cold-start before the demo feels broken.

**Policy:**

1. Measure HF Spaces cold-start on Day 1 of deployment.
2. If cold-start < 60 s: plan validated, no changes.
3. If cold-start > 60 s: implement a lazy-load path (FastAPI eager,
   K8s lazy on first K8s request). Scoped ~2 hours implementation.

This contingency is **not** pre-built. Pre-building a lazy-load path
that may never ship creates dead code that rots, and the test surface
for "lazy loading plus corpus routing plus provider switching" is
non-trivial. The RSS logging in `app.py` (Task 2) emits the exact
numbers needed to make the decision; the decision is documented here
so future-me remembers the threshold and doesn't optimize prematurely
on a hunch.

## False-premise questions come in two flavors

When authoring golden-dataset questions whose premise is wrong, the
question can point at one of two genuinely different failure modes.
Both are valid; they test different pipeline paths and should be
labeled distinctly so the evaluator routes correctly.

**Flavor A — pure refusal.** The premise is not addressed anywhere in
the corpus. Example: "How do I configure Claude API rate limits in
Kubernetes?" K8s has no such concept. Schema: `category: "out_of_scope"`,
`expected_sources: []`, `source_snippets: []`. The evaluator's
`grounded_refusal` metric expects the answer to contain a refusal
phrase ("does not contain", "no information") AND cite zero sources.
Tests the pipeline path where retrieval correctly returns nothing
useful and the agent correctly declines.

**Flavor B — documented negative.** The corpus contains an explicit
negative answer. Example: "How do I configure NetworkPolicy to enforce
mTLS?" The K8s NetworkPolicy docs have a "What you can't do with
network policies" section that explicitly says "Anything TLS related
(use a service mesh or ingress controller for this)". Schema:
`category: "retrieval"`, `question_type: "false_premise"`,
`expected_sources: [<the negative-answer page>]`, `source_snippets:
[<the verbatim negative statement>]`. The evaluator expects the agent
to retrieve the page, find the negative statement, and answer
negatively with a citation. Tests the stricter path where the corpus
genuinely contains the answer and the agent must not hallucinate a
contradictory capability.

**Why both matter for the honest-evaluation brand.** Grounded refusal
is not "refuse when retrieval is weak." It is "answer exactly what the
source says, including when the source says no." Flavor A tests the
first half (refuse when there is nothing to ground on); flavor B tests
the second half (report the documented negative instead of
confabulating a positive). The K8s golden dataset includes at least
one of each. The first K8s pilot (`k8s_pilot_005`, NetworkPolicy
mTLS) is flavor B. Flavor A is reserved for questions targeting
features that genuinely do not exist in the K8s corpus; at least one
such question is required in the full 25-question set.

## Pilot_005 refusal-gate + agent-behavior measurement

The first K8s pilot run surfaced two distinct flavor-B failure modes
on `k8s_pilot_005` (NetworkPolicy mTLS). Both are empirical, both
have specific numbers, and both are logged in
`results/k8s_pilot_threshold_0.02.json` and
`results/k8s_pilot_threshold_0.015.json`.

**Failure mode 1 — threshold calibration (at 0.02).** The
`SearchTool.execute()` refusal gate fired with `max_score=0.01639` —
exactly `1/(60+1)`, the rank-1 RRF score from a single fusion system.
BM25 hit "NetworkPolicy" at rank 1; the dense encoder contributed
nothing, because "Anything TLS related (use a service mesh or ingress
controller for this)" is a single negative sentence, not a conceptual
topic the page is semantically "about." Hybrid fusion inherited only
the BM25 rank-1 score. At threshold 0.02 (the FastAPI working value),
the gate refused before the agent saw any chunks. Retrieval P@5 and
R@5 both 0.00; answer is a generic refusal.

**Failure mode 2 — agent behavior on documented negative (at 0.015).**
With the threshold dropped just below the measured max score
(`0.015 < 0.01639`), retrieval is perfect: P@5 1.00, R@5 1.00, all
five top chunks from `k8s_network_policies.md`. But the agent still
produces a flavor-A-style refusal: *"The Kubernetes documentation
does not provide specific instructions on configuring a NetworkPolicy
to enforce mutual TLS..."* The "Anything TLS related" sentence is in
the retrieved chunks — the agent simply treats the absence of
positive instructions as grounds for refusal, rather than reading the
explicit negative sentence and citing it as the answer. KHR 0.67: the
`service mesh` and `ingress controller` keywords (the documented
alternatives the page points to) are missing from the answer.

**Implication.** The flavor-B mechanism requires more than threshold
tuning. Fixing the gate is necessary but not sufficient. The system
prompt needs a flavor-B clause (e.g., *"if the documentation
explicitly says a feature does not exist or is not supported, report
that with citation — do not treat it as unanswerable"*), **or** the
K8s golden dataset's flavor-B questions must use phrasing the
current prompt can route correctly. The 0.30 placeholder value from
the design doc was based on "prefer conservative" intuition without
empirical grounding — the measured working range for K8s pilot
retrieval is lower by more than an order of magnitude than that
intuition, and even at the working threshold the prompt layer is the
blocker.

**What this measurement is.** A pilot smoke-test result, not a
benchmark claim. Aggregates at 0.02: P@5 0.63, R@5 0.83, KHR 0.69.
Aggregates at 0.015: P@5 0.80, R@5 1.00, KHR 0.75. Five of six pilots
produce substantively correct answers on K8s content under the
working threshold — evidence the retrieval stack generalizes to K8s.
The pilot's job was schema validation + calibration evidence, not
launch metrics. Launch metrics come from the 25-question K8s golden
set with tuned threshold and (likely) a revised system prompt,
sequenced after this pilot.

## Evaluation-layer multi-corpus support lagged the serving-layer refactor

The Tasks 1–8 multi-corpus refactor wired corpora through
`app.state.corpus_map` and the `/ask` serving route. `scripts/evaluate.py`
was not touched and remained single-corpus — it read
`config.rag.store_path` and `config.evaluation.golden_dataset`
directly, with no awareness of the `corpora` dict. This was an
accurate scoping of the refactor (serving-layer, not eval-layer) but
the gap was not surfaced in the original task list.

The K8s pilot commit adds `--corpus <name>` to `scripts/evaluate.py`,
routing through `config.corpora[name]` for `store_path`,
`refusal_threshold`, and a new optional `golden_dataset` field on
`CorpusConfig`. Without `--corpus`, the legacy single-store path is
preserved for backward compatibility with `make evaluate-fast` and
any existing invocations.

`CorpusConfig.golden_dataset` is `str | None = None` — optional
rather than required — because two legitimate states exist: corpus
has a golden dataset (FastAPI, K8s post-authoring), and corpus has no
golden dataset yet (any corpus during bring-up). The CLI errors
cleanly with *"corpus '<name>' has no golden_dataset configured"*
when the field is None, rather than requiring all corpora to ship
with datasets.

## Deferred: path-preserving ingestion

`scripts/ingest.py` uses `doc_path.glob("*.md")` (non-recursive) and
stores the bare filename as the chunk's `source` field. This forces
a flat-namespace convention: FastAPI ships as `fastapi_*.md`, K8s
ships as `k8s_*.md`, and golden dataset `expected_sources` are
filename stems. The path-preserving alternative (recursive `rglob`
plus relative-path source IDs, e.g., `concepts/workloads/pods`) was
evaluated during the K8s pilot planning and explicitly deferred. The
root-cause refactor would have required FastAPI re-ingestion and a
rewrite of the FastAPI golden dataset's `expected_sources` — trading
certain regression risk on a green baseline (288 tests, citation
accuracy 1.00 on API providers) for speculative legibility benefit
on K8s authoring.

The `source_pages` field on `GoldenQuestion` preserves the
human-readable path anchor separately from the machine identifier,
so the deferral does not lose information. Authors see both
`expected_sources: ["k8s_pods.md"]` (what the evaluator matches on)
and `source_pages: ["concepts/workloads/pods"]` (where the content
came from on kubernetes.io) in the same question record.

**Pattern marker, not a promise.** This is the second visa-timeline
deferral of a root-cause refactor in favor of a minimal-blast-radius
fix; the first was the Mar 25 → Apr 12 P@5 slide bisection. Both
deferrals were deliberate, not forgetting. Not scheduled until
post-launch; marker only. Post-launch scope: modify `ingest.py` to
`rglob` + relative-path source IDs, re-ingest FastAPI, rewrite both
golden datasets' `expected_sources` to path-style. Estimated 3h.

## K8s refusal_threshold empirical calibration — 0.02 → 0.015

**Change.** `configs/default.yaml`, `corpora.k8s.refusal_threshold`:
`0.02` → `0.015`. Single-line config change, pilot-corpus only.
FastAPI threshold unchanged.

**Empirical evidence.** Diagnostic instrumentation of `k8s_pilot_005`
(*"How do I configure a Kubernetes NetworkPolicy to enforce mutual
TLS (mTLS) between Pods in the same namespace?"*) captured the
retrieval gate firing at `max_score = 0.01639344262295082` — exactly
`1 / (60 + 1)`, the algebraic floor for a single rank-1 BM25 hit
under RRF with `rrf_k = 60`, dense contribution zero. At
`refusal_threshold = 0.02`, pilot_005 tripped the gate and short-
circuited before retrieval chunks reached the agent. At
`refusal_threshold = 0.015` (one tick below the measured floor), the
gate releases and retrieval proceeds. The 0.015 value is not a
tuning guess — it is the nearest round-number floor below the
observed gate-fire value for the single worst pilot in the set.

**Validation.** `results/k8s_preedit.json` captures the full 6-pilot
run at 0.015. Aggregate: P@5 0.80, R@5 1.00, KHR 0.78, mean
`tool_calls_made` 1.167. All six questions receive retrieval; no
gate-fire short-circuits. pilot_005 still refuses as a separate
downstream issue (see next entry when the counterfactual-query fix
lands); that is not a threshold problem.

**Scope of this commit.** K8s only. FastAPI `refusal_threshold`
(0.02) is not affected and FastAPI baseline is not re-measured.
Launch-intent `0.30` placeholder for K8s remains as a comment
marker; the full threshold sweep against the 25-question golden set
replaces 0.015 with a properly-tuned value in a later commit. 0.015
is the pilot-floor safety value, not the production-target value.

**Why this is a separate commit from the prompt revision.** The
threshold calibration is empirically grounded on its own — it
removes the 0.01639 gate-fire blocker, which is the precondition for
any downstream evaluation of pilot_005's actual agent behavior. The
prompt revision addresses a *different* failure mode surfaced once
the gate releases (agent search strategy is monotone positive-
framing). Two independent changes must not entangle in one commit;
if the prompt revision fails its regression gate and is reverted,
the threshold calibration should stand on its own empirical merit.
Feedback memory `feedback_fix_before_sweep.md` applies recursively:
fix measurement-affecting bugs at every layer before combining
fixes into single experiments.

## Prep for counterfactual-query prompt regression — pin, wire, tolerances

**Three sub-changes bundled as one prep commit, each small and in
service of making the downstream regression measurement valid.**

**1. OpenAI model pin.** `agent_bench/core/provider.py:208` changes
`self.model = "gpt-4o-mini"` → `self.model = "gpt-4o-mini-2024-07-18"`.
The unpinned alias is a known drift vector — the Mar 25 → Apr 12 P@5
slide bisection is an already-open parallel track item traceable to
silent alias migration. A regression run that uses the alias across
pre-edit and post-edit phases conflates prompt-clause effect with
model drift, even within a single session if the alias happens to
roll between runs. Pinning the dated snapshot removes the variable.
Pricing dict in `configs/default.yaml` gets a matching
`gpt-4o-mini-2024-07-18` entry so the cost-lookup at
`provider.py:209` still resolves. Tests that pin the model string
live in mock response payloads (not outgoing assertions) and the
langchain baseline (separate code path) — neither affected.

**2. FastAPI multi-corpus eval wiring.** `configs/default.yaml`
adds `corpora.fastapi.golden_dataset: agent_bench/evaluation/datasets/tech_docs_golden.json`.
The production serving path at `routes.py:105-120 _resolve_system_prompt`
already routes `/ask` and `/ask/stream` through `format_system_prompt(label)`
from `core/prompts.py` — the `app.state.system_prompt` legacy fallback
(serving/app.py:276) is effectively dead code given the shipped multi-corpus
config. The **only** remaining caller of `task.system_prompt` is the
`scripts/evaluate.py` legacy branch used by `make evaluate-fast`. Adding
the missing `golden_dataset` field makes `--corpus fastapi` work so the
regression gate can measure the actual production prompt path, not the
legacy eval-scaffolding prompt. Purely additive; zero blast radius on
serving (serving doesn't read `golden_dataset`).

**3. Pre-committed four-metric tolerances.** Written down now, before
the post-edit runs, so the pass/fail call on the counterfactual-query
prompt clause is not a judgment under confirmation-bias pressure.
Applied identically to FastAPI and K8s:

| Metric | Pass criterion |
|---|---|
| P@5 | post-edit ≥ pre-edit − 0.02 |
| R@5 | post-edit ≥ pre-edit − 0.02 |
| Citation accuracy | post-edit ≥ pre-edit (**hard gate** — any drop blocks commit) |
| Mean `tool_calls_made` | post-edit ≤ pre-edit + 0.30 |
| Individual question cap | no question that used fewer than `max_iterations=3` iterations pre-edit may hit the cap post-edit |

**pilot_005 strict flip criterion (K8s-only):**
- `keyword_hit_rate ≥ 0.60` against golden keywords `["not", "does not", "NetworkPolicy", "service mesh", "TLS", "ingress controller"]`
- Answer cites `k8s_network_policies.md`
- Answer contains "service mesh" OR "ingress controller" (the concrete documented-negative evidence the pre-edit refusal lacked)
- Answer does NOT begin with refusal phrasing ("The ... documentation does not provide", "I cannot answer")

**Baseline reference:** K8s pre-edit numbers from `results/k8s_preedit.json`
at commit `b97f00f` — P@5 0.80, R@5 1.00, citation 1.00 (all 6),
mean tool_calls 1.167. FastAPI pre-edit reference established by
`results/fastapi_preedit.json` in the next step of this session,
same pinned ID, same refusal threshold (0.02).

**Rationale for bundling.** All three sub-changes answer "what must
be true before the regression measurement is valid" — drift control,
evaluation path, decision criteria. Splitting into three commits
would add noise without adding signal. None of them change the
prompt template itself; the prompt edit is the NEXT commit and is
the sole experimental variable the regression measures.

## Fix 1 (prompt-level counterfactual clause) attempted and reverted

**Outcome.** K8s regression clean on every metric (P@5, R@5, KHR,
citation, mean tool_calls all within tolerance or unchanged); K8s
pilot_005 flipped from refusal to documented-negative-with-citation
as designed (KHR 0.67 → 1.00, answer contains both "service mesh"
and "ingress controller", cites `k8s_network_policies.md`).
**FastAPI regression failed** on the iteration-inflation tolerance:
mean `tool_calls_made` 1.111 → 1.556 (delta +0.444, gate +0.30),
and two retrieval questions (q024, q025) were pushed from 1 pre-edit
tool call to 3 post-edit tool calls (hitting `max_iterations=3`
cap), violating the pre-committed "no new cap-hits from sub-cap
baseline" criterion.

**Correctness metrics on FastAPI all held.** Citation accuracy
stayed at 1.000 / 1.000 across all 27 questions. P@5 delta −0.007,
R@5 delta 0.000, KHR delta +0.006. The failure is purely process
inflation, not output regression. q024 and q025 produce identical
P@5/R@5/KHR/citation numbers pre and post despite the cap-hit — the
orchestrator's "max iterations hit → one final complete() without
tools" path happened to keep answers correct, but that is
observation, not structural protection.

**Failure mode.** The clause's trigger condition — *"your first
search returned documentation about the subject of the question
without addressing the specific capability or feature the user is
asking about"* — relies on subjective LLM judgment about whether
retrieved content "addresses" a capability. The judgment is fuzzy
on compound multi-topic questions where the first search returns
partial-topic coverage. q024 asks about "Docker + Gunicorn workers
+ health checks + Pydantic Settings"; first search returns Docker
content, LLM reads "documentation about the subject without
addressing the specific capability," fires the follow-up with
negative framing, gets nothing useful, does a third normal search
to cover the remaining topics, hits the cap. Same pattern on q025.
Over-firing on this class of question is an inherent fragility of
prompt-level LLM-judged triggers; a wording refinement might
narrow the misfire rate but cannot eliminate it as long as the
judgment itself is fuzzy.

**q023 vs q024/q025 asymmetry is a useful signal for Fix 2.** q023
is a pre-existing 3-tool-call compound question ("custom error
handling + CORS middleware + structured testing with dependency
overrides"). Under the prompt clause, **q023 was unchanged** — the
clause did not fire on it — while q024 and q025, structurally
similar compound questions, were pushed into 3-tool-call cap-hit.
The difference is not in question structure but in how the LLM
interpreted the first-search return for each. That asymmetry is
the precise reason a deterministic trigger is the right next step:
any Fix 2 / Fix 3 candidate should be unit-testable against
`(pilot_005, q023, q024, q025)` — the right fix must fire on
pilot_005 and behave predictably on all three compound questions
(either fire on all of them or none of them, but not pick them
selectively by LLM whim).

**Gate discipline honored.** The pre-committed FastAPI tolerances
fired for exactly the reason the pre-commitment was designed:
catching process-metric regressions before they ship. Tolerance-
relaxation post-hoc would burn the session's strongest discipline
artifact (pre-committed-tolerances + honored-gate) for marginal
ship-this-approach EV. The narrow pilot_005 finding does not
evaporate with the revert — chunk 63 (`d0806d5da91d6026`) is real,
the negative-framing retrieval is reproducible, and Fix 2 will
surface the documented negative the same way via a deterministic
path.

**Fix 2 deferred to a later session.** Deterministic query
expansion at the `SearchTool` layer: when a `search_documents`
call returns no chunk containing a direct answer string, issue a
second internal search with negative-framing keywords and merge
results before returning to the orchestrator. Offline-testable,
corpus-agnostic, no LLM judgment required, no iteration-budget
impact (the double-search happens inside a single tool call, not
across iterations). Unit-testable against the
`(pilot_005, q023, q024, q025)` asymmetry as an acceptance fixture.

**Evidence retained.** Four result JSONs in `results/` document the
regression measurement at the pinned `gpt-4o-mini-2024-07-18`
snapshot in this session:
- `fastapi_preedit.json` — 27 questions, HEAD prompt, 0.02 threshold
- `fastapi_postedit.json` — 27 questions, clause prompt, 0.02 threshold (**gate-failing run**)
- `k8s_preedit_pinned.json` — 6 pilots, HEAD prompt, 0.015 threshold
- `k8s_postedit.json` — 6 pilots, clause prompt, 0.015 threshold (**gate-passing run, pilot_005 strict flip confirmed**)

The previously-committed `results/k8s_preedit.json` (from `b97f00f`)
is also a valid K8s-pinned measurement at the session-equivalent
snapshot and remains the canonical threshold-commit evidence.

**Held DECISIONS.md drafts stay held.** The counterfactual-query
finding draft (to be updated when Fix 2 lands) and the threshold-
calibration entry already committed at `b97f00f` are both correct
in scope. The narrowed serving-migration deferral entry (tied to
any external reference to the counterfactual-query fix) also stays
deferred until Fix 2 lands, since the production/eval-harness
prompt divergence is unchanged by this revert.

## Fix 2 pre-committed regression gate — SearchTool deterministic query expansion

**Pre-committed BEFORE post-edit runs** (same discipline pattern
that caught Fix 1's iteration inflation cleanly).

**Mechanism under test.** `agent_bench/tools/search.py`
`SearchTool.execute` gains a deterministic two-query retrieval
path. When the primary retrieval passes the refusal gate, a
secondary retrieval is issued against an expanded query
(`original_query + " not supported limitations cannot"`), and the
final context returned to the LLM is `primary_top_3 ++
secondary_top_5` deduplicated by `chunk.id`. Both retrievals run
inside a single `SearchTool.execute` call — from the LLM's
perspective, the tool schema, name, parameters, and return shape
are unchanged, and the iteration budget is untouched.

**Why this is architecturally different from Fix 1.** Fix 1 placed
a behavioral clause in the system prompt that told the agent to
issue follow-up searches itself. The trigger was an LLM judgment
("did the first search return content addressing the specific
capability?") and the follow-up was a separate tool call, so it
counted against `max_iterations`. Over-firing on compound questions
inflated iteration counts and pushed q024/q025 to the cap. Fix 2
replaces this with a deterministic trigger (primary passes gate),
a fixed expansion suffix, and a merge that happens entirely inside
one tool call. No LLM judgment; no iteration change; corpus-
agnostic.

**Suffix choice.** `" not supported limitations cannot"`. Keyword-
dense, ungrammatical on purpose — the suffix exists to shift BM25
and embedding mass toward "what you cannot do" / "limitations"
sections, not to read well. The ungrammatical form is also a self-
documenting signal in retrieval logs: anyone reading a query trace
sees the suffix and immediately knows it is a synthetic expansion,
not user input. A one-line comment in `search.py` preserves the
rationale for future readers.

**Merge choice.** `primary_top_3 + secondary_top_5` deduped by
`chunk.id`, producing 5–8 unique chunks per call. Rationale: top-5
primary would make the expansion redundant on high-overlap queries
(defeating the mechanism), while primary-top-3 guarantees the
expansion always contributes to the final context window. Probe
data (`/tmp/probe_fix2_v2.py`, throwaway) confirms this merge
strategy surfaces pilot_005's target chunk
(`d0806d5da91d6026`, chunk_index 63, "Anything TLS related ... use
a service mesh or ingress controller for this") at position 6–8 in
the merged list.

**Opt-in flag, defaulting ON.** `SearchTool` accepts
`negative_framing_expansion: bool = True`. Default is the shipping
configuration because the regression gate must measure the shipping
behavior, not the no-op path. A `False` default would mean the gate
validates an unused parameter, and a subsequent commit flipping the
default would have no regression evidence. Kill switch is preserved
via explicit `False` at construction if a future regression
requires an A/B comparison.

**Baseline reuse.** The Fix 1 session's pre-edit JSONs
(`results/fastapi_preedit.json`, `results/k8s_preedit_pinned.json`,
both committed at `bd2b913`) were measured under the currently-
committed state of the repo: pinned `gpt-4o-mini-2024-07-18`, K8s
threshold 0.015, FastAPI threshold 0.02, HEAD `prompts.py` with no
clause, HEAD `search.py` with no expansion. The working tree
verification confirms this state is unchanged. These JSONs are
therefore reused as the Fix 2 pre-edit baseline and do not need to
be re-measured. Only post-edit runs are required for the Fix 2
regression (~$0.02 saved).

**Pre-committed tolerances.**

| Metric | Pass criterion |
|---|---|
| P@5 | post-edit ≥ pre-edit − 0.02 |
| R@5 | post-edit ≥ pre-edit − 0.02 |
| Citation accuracy | post-edit ≥ pre-edit (**hard gate** — any drop blocks commit) |
| Mean `tool_calls_made` | post-edit ≤ pre-edit + **0.05** (design-correctness gate — see note) |
| Individual cap-hit | no question that used fewer than `max_iterations=3` iterations pre-edit may hit the cap post-edit |

**Note on the tool_calls gate.** ≤ +0.05 is a *design-correctness*
gate, not a *performance* gate. Fix 2's invariant is that both
retrievals happen inside one `SearchTool.execute` call, so the
LLM's iteration count is unchanged by construction. Any non-trivial
movement in `mean tool_calls_made` indicates the design invariant
is broken — e.g., expansion accidentally exposed as a separate
tool, or the LLM observing two-call behavior and adapting its
strategy. The gate fires on design violation, not on performance
regression. The 0.05 absolute threshold absorbs legitimate run-to-
run variance from non-determinism in the LLM even at temperature
0, without absorbing real iteration-count movement.

**pilot_005 strict flip criterion (K8s-only, unchanged from Fix 1
gate):**
- `keyword_hit_rate ≥ 0.60` against golden keywords `["not", "does not", "NetworkPolicy", "service mesh", "TLS", "ingress controller"]`
- Answer cites `k8s_network_policies.md`
- Answer contains "service mesh" OR "ingress controller"
- Answer does NOT begin with refusal phrasing

**Baseline reference for the gate.**

| Corpus | Pre-edit source | P@5 | R@5 | Citation | Mean tool_calls |
|---|---|---|---|---|---|
| FastAPI (27) | `results/fastapi_preedit.json` @ `bd2b913` | 0.585 | 0.679 | 1.000 | 1.111 |
| K8s (6 pilots) | `results/k8s_preedit_pinned.json` @ `bd2b913` | 0.800 | 1.000 | 1.000 | 1.167 |

**Post-edit filenames (to be produced).**
- `results/fastapi_postedit_fix2.json`
- `results/k8s_postedit_fix2.json`

**If the gate passes:** commit Fix 2 with `search.py` change, unit
tests (including the tool-spec snapshot test), the two post-edit
result JSONs, and this DECISIONS.md entry extended with the
regression outcome.

**If the gate fires:** revert, document the failure mode, surface
the specific criterion that fired. No tolerance relaxation — same
discipline pattern as Fix 1 revert.

## Fix 2 outcome — mechanism works, response-style criterion fired, reverted

**Regression runs produced.** Two post-edit runs on K8s (FastAPI not
run — K8s findings gated the decision before API spend on the
broader set):

| Run | Merge rule | File | Purpose |
|---|---|---|---|
| Fix 2 v1 | `primary[:3] + secondary[:5]` | `results/k8s_postedit_fix2.json` | Initial implementation |
| Fix 2 v2 | `primary[:5] + secondary[:5]` | `results/k8s_postedit_fix2_merge_v2.json` | Path A refinement after v1 failed P@5 on a metric-definition mismatch |

**v1 findings.** Aggregate: P@5 0.800 → 0.767 (Δ −0.033, **FAILED**
the P@5 ≥ −0.02 tolerance). The failure traced to a merge-rule /
metric-semantics interaction: `retrieval_precision_at_k` computes
precision on `retrieved_sources[:5]`, and with `primary[:3] +
secondary[:5]` the first 5 entries were `primary_top_3 +
secondary_top_2`. For pilot_005, `secondary[1]` was
`k8s_pods.md` (chunk_index 40, surfaced because the reranker
matched its "localhost communication" content against the expanded
query). That single off-source chunk in position 5 dropped P@5
from 1.00 to 0.80 for pilot_005 and similarly for pilot_006.
Iteration invariant held (tool_calls 1.167 → 1.167). Citation
accuracy held (1.000 → 1.000). Target chunk
(`d0806d5da91d6026`, "Anything TLS related") reached the LLM
context for pilot_005 at merged position 7.

**Path A refinement (merge v2).** Change `primary[:3] +
secondary[:5]` → `primary[:5] + secondary[:5]`. Rationale:
primary_top_5 is preserved in positions 1–5 by construction, so
P@5 computed on `ranked_sources[:5]` is unchanged from the
no-expansion baseline. Expansion chunks land in positions 6–10.
Target chunk still reaches LLM context (position 9 for pilot_005).
This is an **implementation refinement, not a tolerance
relaxation** — the pre-committed gate thresholds stand; only the
merge rule was adjusted to respect the metric's window semantics.

**v2 findings — perfect metric preservation, but strict-flip fails on response style.**

Aggregate:

| Metric | Pre-edit | Fix 2 v2 | Delta |
|---|---|---|---|
| P@5 | 0.800 | 0.800 | **0.000** |
| R@5 | 1.000 | 1.000 | 0.000 |
| KHR | 0.806 | 0.806 | 0.000 |
| Citation accuracy | 1.000 | 1.000 | 0.000 |
| Mean `tool_calls_made` | 1.167 | 1.167 | **0.000** |

Every aggregate metric **literally unchanged**. Per-question
deltas: zero on every metric, every question. The design
invariant (iteration budget unchanged, tool schema unchanged,
refusal gate behavior unchanged) holds perfectly.

**But pilot_005 strict flip fails on the refusal-phrasing criterion.**
Post-edit answer:

> *"The Kubernetes documentation does not provide specific
> instructions on configuring a NetworkPolicy to enforce mutual TLS
> (mTLS) between Pods in the same namespace. For mTLS, it is
> generally recommended to use a service mesh or other proxy
> solutions, as NetworkPolicy alone does not handle TLS
> configurations directly [source: k8s_network_policies.md]."*

The answer substantively contains the documented negative with
citation. But it opens with *"The Kubernetes documentation does
not provide specific instructions..."* — the exact refusal-
phrasing opener the strict-flip criterion was pre-committed to
reject. The criterion exists because the brand is honest
evaluation: an answer that opens apologizing that the
documentation "does not provide specific instructions" reads, to
a technical reviewer, like the system failed to find the answer
and is papering over the gap, even though the facts and citation
are present. The criterion fired as designed.

**Compare to Fix 1 post-edit answer (from `bd2b913` evidence):**

> *"Kubernetes NetworkPolicy does not support enforcing mutual TLS
> (mTLS) directly. The documentation states that anything TLS
> related should be handled using a service mesh or ingress
> controller, rather than through NetworkPolicy [source: k8s_network_policies.md]."*

Fix 1's answer asserts a fact about **NetworkPolicy** ("does not
support"); Fix 2's answer asserts a fact about **the documentation**
("does not provide instructions"). The first forecloses the
capability; the second leaves open whether the capability exists
somewhere the system didn't see. That distinction is load-bearing
for any grounded-refusal narrative, and it separates a system that
handles documented negatives crisply from one that hedges around
them.

**Diagnosis.** Fix 2's mechanism successfully gets the target chunk
into the LLM's context window — the retrieval side of the problem
is solved. What Fix 2 **cannot provide** is explicit guidance on
how to phrase the documented negative once the chunk is present.
Fix 1's prompt clause was doing that guidance work; removing the
clause and relying on the LLM's unaided response style produces a
hedging answer because the LLM, seeing both NetworkPolicy-spec
content and a TLS limitation bullet, defaults to contextual
hedging rather than crisp assertion.

**Fix 2 is therefore not an alternative to Fix 1's prompt clause
— it is a prerequisite.** Fix 2 guarantees the chunk reaches
context; a future "Fix 2 + targeted prompt clause" stack could
resolve both the retrieval gap and the response-style gap without
Fix 1's over-firing problem, because the clause would no longer
need to direct the agent to do a follow-up search (Fix 2 handled
that). The over-firing on compound questions that broke Fix 1 was
caused by the agent deciding to do extra search iterations under
LLM judgment; if the expansion already happened deterministically
inside the first tool call, the clause has less work to do and
may not trigger the second-LLM-call pattern at all. **Speculative
and not for this session.** Future work item.

**Gate verdict: failed on pilot_005 strict flip criterion.**
Reverting, same Fix-1 pattern.

**What this commit contains.**
- `agent_bench/tools/search.py` **reverted** to HEAD (no Fix 2
  code changes)
- `tests/test_tools.py` retains the `MockChunk.id` hygiene fix
  (the real `Chunk` class has `id`; mock should match the real API
  for future test authors)
- `tests/test_tools.py` adds `TestSearchToolSpecSnapshot`: a
  general-purpose guard that freezes `SearchTool`'s LLM-facing
  contract (name, description, parameters). The lesson from Fix 2
  is that any future refactor exposing internal SearchTool state
  to the LLM would break iteration-budget invariants — the
  snapshot test catches that at test time, independent of whether
  Fix 2 lands.
- Two regression evidence JSONs: `results/k8s_postedit_fix2.json`
  (v1, the P@5 failure) and `results/k8s_postedit_fix2_merge_v2.json`
  (v2, the strict-flip failure). Retained as the measurement
  trail behind the revert decision.
- This DECISIONS.md entry (pre-committed gate + outcome + revert
  narrative).

**What this commit does NOT contain.** No changes to
`agent_bench/tools/search.py`, `agent_bench/core/prompts.py`, or
`configs/default.yaml`. Both Fix 1 (prompt clause) and Fix 2
(SearchTool expansion) have been attempted and reverted this
session. Three commits of progress nonetheless: `b97f00f`
(threshold calibration, empirical), `77017db` (prep bundle: model
pin + fastapi wire + Fix 1 pre-committed tolerances), `bd2b913`
(Fix 1 revert narrative). The threshold calibration and model pin
are real, shipped, measurement-grounded infrastructure changes.
The two fix attempts are documented learning that shapes the
future direction.

## `grounded_refusal` metric reads answer text, not retrieved sources — 2026-04-14

**Context.** Week 1 step 5 authoring (25-question K8s golden set). Two
flavor-A out-of-scope questions (`k8s_004` Jaeger sidecar, `k8s_024`
Envoy xDS ADS) surfaced a pre-existing bug in the
`grounded_refusal` metric during the functional check.

**Bug 1 — wrong signal.** The metric's docstring said it checks
whether the answer correctly refuses AND cites no sources, but the
implementation was checking `len(response_sources) == 0` where
`response_sources` is the *retrieved*-sources list. Real agents
retrieve candidates on any non-trivial OOS query (the grounded-refusal
gate at tool level only catches the thinnest queries), inspect the
candidates, find nothing relevant, and refuse *in the answer text*
without citing anything. Checking retrieval emptiness flagged those
correct refusals as failures. Fix: inspect the answer text for
`[source: X.md]` citations via regex; drop the `response_sources`
parameter from the signature entirely.

This was a silent false negative on all 5 fastapi out-of-scope
questions (`q008`–`q010`, `q026`–`q027`) which all correctly refuse
but were being marked `grounded_refusal=False`. Aggregate
`refusal_rate` in `report.py` shifts by the resulting 5-question
delta; any historical comparison to pre-fix fastapi numbers needs
to acknowledge this.

**Bug 2 — metric coverage gap surfaced during 25-question authoring.**
`grounded_refusal_rate` recognized "does not contain information"
phrasing (in `refusal_phrases` list) but missed "not in the
{corpus_label} documentation" phrasing — the exact shape taught by
the system prompt at `core/prompts.py:17-18`. The LLM produced the
canonical form on some questions and the phrase-list form on others;
the metric inflation/deflation was non-deterministic. Fix: narrow
regex `\bnot in the\b[^.]{0,60}\bdocumentation\b` added alongside
phrase-list matching.

**Rejected alternative.** Substring `"not in the"` would produce
false positives on valid-answer phrasing — "the rate limit is not in
the same scope as the request timeout", "the flag is not in the 1.28
release; it landed in 1.29", "this value is not in the default
range" — all of which are legitimate retrieval answers with
conditional or scope-limiting language, not refusals. Honest
evaluation cannot afford a metric that silently counts these as
grounded refusals.

**Tests.** Two unit tests pin both directions:
`test_canonical_refusal_phrasing_recognized` covers the positive
case ("The answer is not in the Kubernetes documentation"), and
`test_not_in_the_is_not_substring_refusal` covers the negative case
("The rate limit is not in the same scope as the request timeout").
The negative test is the load-bearing one — without it, a future
refactor could silently widen the matcher back to substring and pass
all existing tests. The negative test pins design intent.

**Scope bound.** This is a metric correctness fix, not a threshold
change. The 0.015 refusal-gate threshold (calibrated in `b97f00f`
against the 6-question pilot) is unchanged by this commit. Whether
the corrected metric shifts the optimal threshold against the full
25-question set is a question for the threshold-sweep session, not
this authoring session.

## Parallel tracks / deferred items — 2026-04-14

Tracked list of work items that are deferred to parallel sessions.
Each item has a reason for deferral and a rough scope boundary so
the session that picks it up has the context to pre-commit tolerances
and decision criteria before measuring.

1. **`routes.py:552` audit-logger semantics unification.** The
   serving layer's audit record field still uses the pre-fix
   `grounded_refusal = not bool(sources)` expression, which disagrees
   with the evaluation metric's answer-text-based definition. Not
   surfaced to the dashboard (audit log only), but external reviewers
   who reference audit records for runtime verification would see a
   different definition than the benchmark claims. Fix: call
   `grounded_refusal(answer, category)` from `metrics.py` directly.
   When this lands, the "grounded_refusal metric" DECISIONS.md entry
   above should get a one-line addendum noting the unification.

2. **Full 25Q threshold sweep → production-target `refusal_threshold`
   for K8s.** The 25Q set exists, the metric is correct. Sweep
   against the full set, compare to pilot-floor 0.015, pick the
   production-target value, update `configs/default.yaml` placeholder
   comment. Pre-commit before measuring: sweep range, decision
   criteria, tolerances. Do not entangle with flavor-B response-style
   work below — those are independent axes.

3. **Flavor-B response-style class (pilot_005 + k8s_022).** Two
   independent reproductions of "LLM refuses when documented negative
   is in retrieved context". Retrieval is healthy on both; the gap
   is prompting. Future session: Fix 2 (counterfactual-query
   expansion in `SearchTool`) + targeted prompt clause stacked —
   previously speculative in the Fix 2 revert entry, now addresses
   a documented reproducible class. Two reproductions, not one-off.

4. **Serving-migration deferral.** Tied to external references to
   the counterfactual-query fix. Unchanged from prior sessions.

5. **`agent-bench` → `refusal-bench` rename — CLOSED 2026-04-14.**
   Decision: keep `agent-bench`, reframe via tagline. The original
   concern was name collision with AgentBench (Liu et al., ICLR
   2024, ~1000 citations). Due-diligence at launch time: the name
   is `agent-bench` (hyphenated) vs. `AgentBench` (camelcase),
   which are distinct identifiers across GitHub, arXiv, and PyPI.
   The two projects target different audiences (LLM-as-agent
   capability vs. RAG+refusal benchmark) and any reviewer reaching
   the repo via LinkedIn or CV sees the scope in the README within
   seconds. Rename cost is substantial (~350 internal references
   across ~60 files, two external account renames, one HF Space
   URL break with no redirect) for a naming-precision benefit that
   isn't supported by the actual scope — the benchmark measures
   retrieval, grounding, multi-hop, citation accuracy, and refusal
   as seven axes, not refusal alone. Tagline reframe captures the
   honest-evaluation positioning without the rename cost:
   > "A RAG benchmark built from primitives, with honest
   >  evaluation of retrieval, refusal, and grounded citation."
   HF Space `Nomearod/agentbench` renamed to `Nomearod/agent-bench`
   for consistency with GitHub repo name at the same time (absorbs
   the HF URL break before the first LinkedIn post links out).

6. **OpenAI snapshot drift bisection.** Mar 25 → Apr 12 P@5 slide;
   the model pin at `77017db` (`gpt-4o-mini-2024-07-18`) removed
   the ongoing drift risk, so any future measurement is apples-to-
   apples. The original bisection is still unresolved but cheap at
   this point — tractable whenever there is session capacity, low
   urgency because the pin protects forward runs.

## K8s refusal_threshold sweep against 25-question golden — 2026-04-14

**Override notice.** This sweep ran in the same session as the
25-question authoring + grounded_refusal metric fix (`526be18`),
after I explicitly flagged that the parallel-tracks guidance from
earlier in the session recommended waiting for a fresh session with
pre-commitment discipline. The user issued an explicit override:
"proceed on best-judgment sweep range and criteria" — logged here
for audit trail. The pre-commitment frame below was drafted BEFORE
running any sweep value, not after. The decision criteria were
locked before the first data point was observed, not retrofitted.

**Sweep grid.** 4 threshold values: `0.010`, `0.015` (already
measured in `.cache/eval_k8s_full25_postfix.json`, the post-metric-
fix run from `526be18`), `0.020`, `0.025`.
- `0.010`: one tick below current calibration; sanity-check floor.
- `0.015`: current calibration (pilot-floor, one tick below
  pilot_005's 0.01639 max_score).
- `0.020`: matches legacy FastAPI threshold and the original
  provisional K8s default before the `b97f00f` calibration.
- `0.025`: one tick above legacy; exploration of whether aggressive
  OOS short-circuiting is worth the correctness risk.

**Decision criteria (pre-committed).**
1. **OOS refusal must hold.** Both `k8s_004` (Jaeger) and `k8s_024`
   (Envoy xDS) must retain `grounded_refusal=True` at the chosen
   threshold — whether the gate fires at the tool level or the
   LLM refuses after inspecting context doesn't matter, only that
   the metric reports True.
2. **Retrieval recall must not degrade.** Each retrieval-category
   question's R@5 at the chosen threshold must be ≥ its R@5 at
   `0.015` (the post-fix-25Q baseline) with a noise tolerance of at
   most ONE question dropping by at most 0.20. Two or more drops,
   or any drop > 0.20, disqualifies the value.
3. **Citation accuracy must hold.** All questions' citation_accuracy
   must be ≥ 0.95 at the chosen threshold. One question at 0.80 is
   noise-tolerated; two or more is a hard stop.
4. **k8s_022 (flavor-B) retrieval must remain at R@5=1.0.** The
   gap is prompting-side, not retrieval-side; any threshold that
   breaks the already-working retrieval on flavor-B questions is
   a regression.
5. **Pick the highest threshold that satisfies 1–4.** Rationale:
   a higher threshold short-circuits more OOS queries at the tool
   level, saving a retrieval round trip and an LLM call — this is
   a real latency and token-cost win when the correctness is held.
6. **Tie-break.** If multiple values all satisfy 1–4, prefer the
   value closest to a clean round number (0.020 over 0.018) for
   documentation clarity.
7. **Floor.** If no threshold > 0.015 satisfies 1–4, keep 0.015.
   No threshold < 0.015 will be chosen regardless — sub-0.015 is
   strictly less protective than the pilot-floor.

**Scope bound.** K8s only; FastAPI's `refusal_threshold: 0.02` is
unchanged. The flavor-B response-style gap (parallel track #3) is
NOT a sweep variable — changing the threshold does not fix LLM
phrasing; that's the Fix 2 + prompt guidance stacked experiment
the parallel-tracks list already defers.

**Measured results.** All four runs use the post-metric-fix pipeline
(grounded_refusal metric from `526be18`), deterministic mode,
`gpt-4o-mini-2024-07-18`, same retriever config.

| threshold | avg R@5 | OOS refusal | gate fired on                     | broken retrieval       |
|-----------|---------|-------------|-----------------------------------|------------------------|
| 0.010     | 0.957   | 2/2         | —                                 | —                      |
| 0.015     | 0.957   | 2/2         | —                                 | —                      |
| 0.020     | 0.870   | 2/2         | k8s_006, k8s_007, k8s_024         | k8s_006, k8s_007 (R@5=0.00) |
| 0.025     | 0.913   | 2/2         | k8s_004, k8s_007, k8s_024         | k8s_007 (R@5=0.00)     |

**Structural finding: LLM query variance makes max_scores non-deterministic.**
At 0.020, `k8s_006` (ConfigMap, simple) gate-fired → empty retrieval →
R@5=0.00. At 0.025, `k8s_006` did NOT gate-fire → 5 sources → R@5=1.00.
A higher threshold producing fewer gate-fires is physically impossible
if retrieval is deterministic — the SearchTool receives different
queries across runs because the orchestrator issues LLM-generated
queries, and the same question can produce different top-k max_scores
run-to-run. `k8s_006`'s max_score for the query the LLM chose lives
somewhere around the 0.018–0.025 boundary; which side of any given
threshold it lands on depends on which query the LLM wrote.

This means **any threshold above 0.015 is structurally fragile**, not
merely "failed on this run." Even if a run at 0.018 passed, a future
run could gate-fire on `k8s_006` or `k8s_007` because the query is
non-reproducible. The production threshold needs to sit below all
legitimate simple-question max_scores with enough margin to absorb
LLM query variance.

**Decision: keep `refusal_threshold: 0.015`.**

- `0.010`: meets all criteria, identical measured metrics to `0.015`
  (avg R@5=0.957, OOS refusal 2/2, no citation fails). Not chosen:
  lowering strictly weakens the gate's ability to catch low-
  confidence retrievals without improving any measured metric.
- `0.015`: chosen. Meets all criteria and is the highest value that
  does not degrade retrieval — which is the definition of the
  correct refusal-gate threshold. Preserving the gate's signal is
  the gate's purpose; `0.015` gives maximum gate strength without
  cost, `0.010` gives the same measurable behavior with less gate
  signal, so `0.015` dominates.
- `0.020`: breaks TWO retrieval questions (`k8s_006`, `k8s_007`);
  disqualified per criterion 2.
- `0.025`: breaks ONE retrieval question in this run (`k8s_007`)
  but the non-determinism finding means a future run could break
  more. Even ignoring non-determinism, still disqualified by the
  citation-accuracy-equivalent drop on `k8s_007`.

**Corpus characteristic finding.** The 0.020 default inherited from
FastAPI breaks on K8s because K8s retrieval score distributions are
lower for "easy" questions. `k8s_006` ("What is a ConfigMap?") and
`k8s_007` ("What does a Kubernetes Job do?") are both `type: simple`
with clean single-source expected answers — exactly the cases where
BM25+embedding scores should be highest. They land at max_scores in
the ~0.018 range, below the FastAPI-calibrated 0.020 default. This
is **not an authoring bug** — both questions retrieve their
`expected_sources` correctly when the gate doesn't fire. It's a
corpus characteristic: K8s documentation has more topic-overlap
across pages than FastAPI, diluting top-k concentration.

The 25-question set exposed this because the 6-question pilot had
no simple questions with low max_scores — the pilot was drawn from
retrieval-stressful areas (comparison, multi-hop, flavor-B). The
25-question authoring deliberately added simple questions to hit
the CRAG distribution target (6 simple, 5–6 target), and those
simple questions revealed the corpus-characteristic floor.

**Config change.** `configs/default.yaml` `corpora.k8s.refusal_threshold`
comment updated to reference this sweep. Value unchanged at `0.015`.

**Not in scope.** (a) Adding retry-with-query-variance to the
SearchTool to reduce max_score variance — separate session, affects
other corpora. (b) Tuning FastAPI's threshold against its golden
set — the FastAPI default was empirically fine on its own 30Q set
and is not a documented regression. (c) Fixing the `k8s_015`
R@5=0.50 value observed across all threshold runs — pre-existing
authoring state from `526be18`, tracked separately if it becomes
a concern on future runs.

**Narrative summary.** Session hypothesis: pilot_005 is a
counterfactual-query-expansion problem. Session evidence: the
hypothesis is correct on retrieval — the target chunk is reachable
via negative-framing queries and Fix 2 surfaces it deterministically
with zero iteration-budget impact. Session evidence also shows the
hypothesis is **incomplete** — retrieval-only fixes cannot close
the response-style gap, because the LLM under unaided prompting
hedges when a documented negative is surrounded by unrelated
topical content. A future session exploring **Fix 2 + targeted
prompt guidance stacked** is the natural next experiment; this
session's pilot-first discipline has been preserved against two
distinct pre-committed gates, both firing for the reasons they
were designed to catch.
