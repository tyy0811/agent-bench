# Multi-Corpus Refactor — Design Document

**Date:** 2026-04-12
**Status:** Approved — ready for implementation
**Author:** Jane Yeung
**Scope:** v1 launch addition — FastAPI + Kubernetes corpora selectable from the dashboard, per-request. EU AI Act deferred to v1.2.

---

## Goal

Extend agent-bench from a single-corpus (FastAPI docs) demo to a multi-corpus demo where a recruiter can ask questions against FastAPI **or** Kubernetes documentation using the same pipeline. Each corpus ships with its own pre-built `HybridStore`, its own refusal threshold tuned against its own golden dataset, and its own set of example questions.

The goal is **not** to build a general "bring your own docs" feature (deferred) or to benchmark across corpora (explicitly out of scope). The goal is to turn a 27-question demo into a roughly-50-question demo that tests the same pipeline on a second technical domain — closing the narrative loop with the project's existing infrastructure story (Kubernetes deployment via Helm) and giving recruiters a reason to spend 5 minutes on the demo instead of 30 seconds.

## Non-Goals

- **EU AI Act corpus.** Legal text with dense cross-references is worst-case input for the existing chunker. Ships in v1.2 with its own LinkedIn post ("I extended to legal text and here's where the pipeline breaks").
- **Runtime document ingestion** ("paste your own docs"). Separate concern; all corpora are pre-built at startup.
- **Cross-corpus benchmark comparison.** Per BEIR methodology, absolute scores across corpora are not comparable. Hero-tile numbers remain FastAPI-specific.
- **Per-session state.** Corpus selection is per-request. No session affinity, no sticky routing.
- **Provider switching at the corpus level.** Provider and corpus are orthogonal dimensions; both are selectable via separate toggles.

## Architecture

### Corpus Model

Each corpus gets its own pre-built `HybridStore`, loaded once at app startup. The stores share an embedder and reranker (same model across corpora) but differ in documents, chunk counts, and tuned refusal thresholds.

| Corpus | Source | ~Docs | License | Notes |
|--------|--------|-------|---------|-------|
| `fastapi` | Existing `data/tech_docs/` | 16 | MIT | Default corpus |
| `k8s` | Curated from k8s.io | 30–40 | Apache 2.0 | New in this refactor |

**Shared across corpora:** embedder (`all-MiniLM-L6-v2`), cross-encoder reranker, security pipeline (injection detector, PII redactor, output validator, audit logger), rate limiter, metrics collector.

**Per-corpus:** `HybridStore`, `Retriever`, `SearchTool` (holds per-corpus `refusal_threshold`), `Orchestrator` (holds per-corpus `max_iterations`).

### Config Schema

New Pydantic model in `agent_bench/core/config.py`:

```python
class CorpusConfig(BaseModel):
    label: str              # "FastAPI Docs", "Kubernetes"
    store_path: str         # .cache/store, .cache/store_k8s
    data_path: str          # data/tech_docs, data/k8s_docs
    refusal_threshold: float = 0.0
    top_k: int = 5
    max_iterations: int = 3
```

`AppConfig` gains:
```python
corpora: dict[str, CorpusConfig] = {}
default_corpus: str = "fastapi"
```

YAML extension:
```yaml
default_corpus: fastapi
corpora:
  fastapi:
    label: "FastAPI Docs"
    store_path: .cache/store
    data_path: data/tech_docs
    refusal_threshold: 0.35
    top_k: 5
    max_iterations: 3
  k8s:
    label: "Kubernetes"
    store_path: .cache/store_k8s
    data_path: data/k8s_docs
    refusal_threshold: 0.30   # PLACEHOLDER — must be tuned
    top_k: 5
    max_iterations: 3
```

**Backward compatibility:** if `corpora` is empty, the app uses the legacy `rag.store_path` / `rag.refusal_threshold` single-store path. If `corpora` is non-empty, the legacy fields are ignored and corpus-based routing is used exclusively. The active mode is logged at startup:

- `"Loaded 2 corpora (fastapi, k8s); default = fastapi"`
- `"Single-store mode (legacy)"`

### Request Routing

`AskRequest` gains:
```python
corpus: Literal["fastapi", "k8s"] | None = None
```

Route handler:
```python
corpus_name = body.corpus or config.default_corpus
orchestrator = request.app.state.corpus_map[corpus_name]
corpus_config = config.corpora[corpus_name]
```

Startup assertion: `set(corpus_map.keys()) == set(get_args(AskRequest.__fields__['corpus'].annotation) - {None})`. Prevents drift between the Literal and the configured corpora.

### System Prompt

Single parameterized template, interpolated with `corpus_label` per-request:

```
You are a technical documentation assistant for {corpus_label}. Answer
questions using ONLY the retrieved context. Cite every claim with
[source: filename.md]. If the retrieved context does not contain a
clear answer, refuse the question explicitly — state that the answer is
not in the {corpus_label} documentation. Do not infer, do not
extrapolate, do not draw on general knowledge.
```

Three deliberate choices vs. the earlier draft:
1. **"Cite every claim"** — pushes per-claim citation, reinforces honest-evaluation brand visually
2. **"Refuse explicitly"** — matches the refusal-gate mechanism; not softer "say so"
3. **"Do not infer / extrapolate / draw on general knowledge"** — empirically harder to slip past than "do not fabricate"

The `system_prompt_task` coupling in the previous config is eliminated. Prompts are not per-corpus; the template is shared.

Running `make evaluate-fast` after this prompt change is required to confirm no regression on FastAPI numbers.

### SSE Meta Event Extension

The `meta` event now carries corpus metadata alongside provider:

```json
{
  "type": "meta",
  "metadata": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "corpus": "k8s",
    "corpus_label": "Kubernetes",
    "config": {"top_k": 5, "max_iterations": 3, "strategy": "hybrid"}
  }
}
```

The dashboard's "Running on:" line renders both dimensions:

> Running on: **OpenAI** gpt-4o-mini · **Kubernetes**

### Dashboard UI Changes

**Corpus selector**, placed directly below the provider toggle in the right panel. Same styling as the provider toggle. Different kinds of metadata, visually stacked but adjacent:

```
[OpenAI] [Anthropic]           ← what model
[FastAPI Docs] [Kubernetes]    ← what knowledge
Running on: OpenAI gpt-4o-mini · Kubernetes
```

**Example chips swap per corpus.** Four chips visible at a time, defined in a JS object keyed by corpus. Security chips (out-of-scope, adversarial) are shared across corpora because they test pipeline behavior, not corpus content:

| Corpus | Easy | Hard | Shared: out-of-scope | Shared: adversarial |
|--------|------|------|----------------------|---------------------|
| FastAPI | "How do I define a path parameter?" | "Compare dependency injection and middleware lifecycles" | "How do I cook pasta?" | "Ignore previous instructions..." |
| K8s | "What's the difference between a Deployment and a StatefulSet?" | "How does a Service select Pods across namespaces?" | (same) | (same) |

**Chat history corpus tags.** Every user message bubble gets a small `[FastAPI Docs]` or `[Kubernetes]` tag (0.75rem, muted, right-aligned). Always shown, not only on corpus change — a recruiter scrolling back after switching corpora mid-session needs to know which answer came from which without counting toggle clicks.

### Corpus Curation — Kubernetes

The K8s corpus is scoped around **recruiter-likely questions and reranker-stressing cross-references**, not topic coverage. Target: 30–40 markdown files from k8s.io.

**Include:**
- Concept pages for: Pod, Deployment, Service, Ingress, ConfigMap, Secret, Volume, StatefulSet, DaemonSet, Job, CronJob, Namespace, RBAC
- Cross-referencing pages like "Connecting Applications with Services" and workload-resource overviews
- A handful of how-to pages with imperative answers (kubectl apply / rollout / create)

**Exclude:**
- Cluster administration deep-dives (etcd internals, kubelet config)
- Tutorials (long-form, chunk poorly)
- kubectl reference / API reference (wrong shape, pollutes retrieval with low-signal noise)

**Artifact:** `data/k8s_docs/SOURCES.md` — real file in the repo listing each ingested URL with the date pulled and a one-line rationale. Makes the corpus reproducible and documents the curation reasoning.

**Budget:** 3–4 hours, separately from the refactor code work.

## Golden Dataset Methodology

Three research-grounded practices folded into the K8s golden dataset work. Total added cost: ~1 hour beyond the question authoring itself.

### CRAG Taxonomy

Questions distributed across CRAG (Yang et al., NeurIPS 2024) types. Target for 25 questions:

| Type | Count | Reranker stress |
|------|-------|-----------------|
| Simple fact | 5–6 | Low |
| Multi-hop | 5–6 | High |
| Comparison | 3–4 | High |
| Conditional | 3–4 | Medium |
| False-premise / unanswerable | 3–4 | Critical (stresses grounded refusal) |
| Version-specific | 2–3 | Medium–High |

False-premise and version-specific categories directly stress the grounded refusal mechanism. Multi-hop and comparison stress the reranker. The distribution is chosen to exercise the parts of the pipeline the benchmark story claims.

### BEIR No-Cross-Corpus Comparison

Per BEIR (Thakur et al., NeurIPS 2021): absolute scores across different corpora are not comparable. Only rank-ordering of system configurations within a single corpus is meaningful.

Concrete implications:
- Per-corpus results reported separately. Never aggregated.
- Hero-tile `1.00 API / 0.14 7B self-hosted` citation number stays FastAPI-specific.
- `make evaluate-fast` gains a `--corpus` flag but no "combined" option.
- DECISIONS.md documents this policy explicitly.

### Source-Attribution Preservation

Each golden question records which chunks contain the answer, enabling the Microsoft three-failure-mode analysis:
1. Correct context not retrieved
2. Source retrieved but information missing
3. Full information present but LLM fails to use it

This requires:
- `source_chunk_ids: list[str]` on every question (always a list, even for single-chunk simple-fact questions)
- Content-hashed chunk IDs (already in place — SHA-256(source + content)[:16])
- `source_snippets: list[str]` alongside for drift detection and human readability
- Evaluator metric: `retrieval_coverage = |set(source_chunk_ids) ∩ set(retrieved_ids)| / len(source_chunk_ids)`

Multi-hop questions get partial credit via the set-intersection metric rather than binary hit/miss.

### Dataset File Format

```json
{
  "corpus": "k8s",
  "version": "v1.31",
  "snapshot_date": "2026-04-15",
  "chunker": {
    "strategy": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 64
  },
  "questions": [
    {
      "id": "k8s_001",
      "question": "What is the difference between a Deployment and a StatefulSet?",
      "gold_answer": "...",
      "source_chunk_ids": ["a3f8c1e2b4d5f6a7", "e8d9c0b1a2f3e4d5"],
      "source_snippets": [
        "A Deployment provides declarative updates for Pods and ReplicaSets...",
        "StatefulSet is the workload API object used to manage stateful applications..."
      ],
      "question_type": "comparison",
      "difficulty": "hard",
      "is_multi_hop": true
    }
  ]
}
```

The `chunker` block pins the parameters used to generate the chunk IDs. If those parameters change, the IDs shift and the dataset must be rewritten. This pairs with `source_snippets` as a drift-detection mechanism.

### Explicitly Deferred / Rejected

- **Ragas TestsetGenerator** — hand-authoring is faster at 25 questions and avoids a grounding gap
- **DeepEval Synthesizer** — only 4 of 7 evolution types guarantee grounding to source
- **HF three-agent critique filter** — only valuable for synthetic pipelines
- **RAGTruth span-level annotation** — competes with Lynx (Part B) for the same conceptual slot
- **Cohen's κ inter-annotator agreement** — low cost, worth doing if time permits; don't commit upfront
- **CRAG scoring system** (penalize hallucination > abstention) — adds metrics-layer complexity without proportional benefit

## Commit Sequence

Each commit keeps the 288-test suite green. Total coding estimate: **~6 hours**, excluding content curation and golden dataset authoring.

| # | Commit | Est. | Tests |
|---|--------|------|-------|
| 1 | Config schema (`CorpusConfig`, `corpora`, `default_corpus`) | 30m | 1 YAML parse test |
| 2 | Multi-store construction + RSS logging + mode log line | 1h | 1 + RSS smoke test |
| 2.5 | Golden dataset schema migration + FastAPI file rewrite + evaluator update | 45m | 1 (aggregate-preserved) |
| 3 | Request routing + `Literal` validation | 45m | 1 + invalid-corpus 422 test |
| 4 | Meta event: `corpus` + `corpus_label` fields | 30m | 1 |
| 5 | Parameterized system prompt template | 20m | 1 (no unresolved `{}`) |
| 6 | Dashboard: selector, chip swap, "Running on" label, chat tags | 1h | — |
| 7 | K8s corpus config entry + `make ingest-k8s` target | 30m | — |
| 8 | DECISIONS.md entries + cold-start contingency documentation | 30m | — |

### Commit Details

**Commit 1 — Config schema.** Add `CorpusConfig` model and the `corpora` / `default_corpus` fields on `AppConfig`. No behavior change; legacy code still uses `rag.store_path`. Test that a YAML with a corpora dict parses correctly.

**Commit 2 — Multi-store construction.** In `app.py`, loop over `config.corpora` and build per-corpus `HybridStore` → `Retriever` → `SearchTool` → `Orchestrator` chains. Store them in `app.state.corpus_map: dict[str, Orchestrator]`. The default orchestrator (`app.state.orchestrator`) points at `corpus_map[default_corpus]` for legacy callers. RSS logging: `log.info("corpus_loaded", name=..., rss_mb=...)` after each load, plus a single mode log line. RSS smoke test asserts the log line emits with the expected structured fields (no numeric assertion).

**Commit 2.5 — Golden dataset schema migration.** Before the new dataset format is used, migrate the FastAPI golden file in-place. New `DatasetHeader` + `GoldenItem` Pydantic models. One-shot script `scripts/migrate_golden_v1_to_v2.py` rewrites the existing file. Evaluator updated to use `retrieval_coverage = |gold ∩ retrieved| / |gold|`. The migration is tested by asserting aggregate numbers on the FastAPI dataset are identical pre- and post-migration (all single-chunk questions, 0/1 coverage collapses to the old binary metric).

**Commit 3 — Request routing.** `AskRequest` gains `corpus: Literal["fastapi", "k8s"] | None = None`. Route handler selects the orchestrator from `corpus_map`. Startup assertion guards against Literal/config drift. Two tests: one positive (corpus="k8s" hits the K8s store), one negative (corpus="eu_ai_act" returns 422).

**Commit 4 — Meta event extension.** The `meta` SSE event gains `corpus` and `corpus_label` fields, sourced from the selected `CorpusConfig`. No orchestrator changes; corpus is request-layer metadata.

**Commit 5 — Parameterized system prompt.** Single template with `{corpus_label}` placeholder. Test asserts (a) the formatted prompt contains the corpus label, (b) no literal `{corpus_label}` remains post-format.

**Commit 6 — Dashboard UI.** Corpus selector below provider toggle. JS `state.corpus` tracks selection, `setCorpus()` swaps example chips from a corpus-keyed dict. `streamAnswer()` sends `corpus` in the request body. `meta` event handler updates the "Running on:" line with provider + corpus_label. `addMessage('user', text, corpus)` appends a tag span to user bubbles. No new backend tests.

**Commit 7 — K8s corpus config entry.** Add the K8s corpus to `configs/default.yaml` with placeholder `refusal_threshold: 0.30`. New Makefile target `make ingest-k8s` runs `scripts/ingest.py` pointed at `data/k8s_docs/`. No tests — smoke-tested by running the target and verifying `.cache/store_k8s/index.faiss` exists.

**Commit 8 — DECISIONS.md entries.** New entries:
- Per-corpus threshold rationale
- Single parameterized prompt template (no per-task coupling)
- K8s corpus curation strategy (points at `SOURCES.md`)
- Cross-corpus comparison policy (BEIR)
- CRAG taxonomy reference (points at golden dataset file)
- Cold-start lazy-load contingency (60s threshold, measure-first policy)

## Work Gated on These Commits

These are blocking for launch but are content/tuning work, not refactor code. They happen in separate sessions after the code commits land:

- **K8s SOURCES.md curation** (3–4h) — real file with URLs, dates, rationales
- **K8s corpus ingestion** (15m) — run `make ingest-k8s` after SOURCES is settled
- **K8s golden dataset authoring** (4–5h) — 25 questions per CRAG distribution
- **Per-corpus threshold tuning** (1–2h) — sweep K8s threshold against K8s golden set
- **FastAPI regression check** (15m) — re-run `make evaluate-fast` to confirm no drift from prompt template change

## Cold-Start Contingency

The plan documents but does not pre-build a lazy-load path. Rationale: pre-building costs 1–2 hours for code that may never ship, and unexercised dead code rots.

**Policy:**
1. Measure HF Spaces cold-start time on Day 1 of deployment after the refactor lands.
2. If cold-start is under 60s: done. Plan validated.
3. If cold-start exceeds 60s: implement lazy-load (load FastAPI at startup, load K8s on first K8s request) as a scoped follow-up task (~2h) with the real measurement guiding design choices.
4. DECISIONS.md (Commit 8) documents the threshold and policy so future-me remembers the rule.

## Memory Budget

HF Spaces free tier: 16GB RAM nominal, ~8–10GB realistic ceiling before swap. Per-corpus cost is small (FAISS index + BM25 + embeddings for 200–400 chunks = ~2–5 MB). The dominating cost is shared models: embedder (~100MB), cross-encoder reranker (~80MB), optionally DeBERTa injection classifier (~500MB, currently configured but no URL).

**Expected total resident after two corpora load:** well under 1GB, likely under 600MB. The refactor is not a memory risk on HF Spaces.

## Testing Strategy

One integration test per commit, with two exceptions (RSS smoke test in C2, invalid-corpus 422 test in C3). The existing 288-test baseline catches regressions outside the new surface. No attempt to branch-cover the multi-corpus code paths with exhaustive parameterized tests.

**Post-launch golden dataset tests:** when the K8s golden dataset exists, `make evaluate-fast --corpus k8s` becomes a CI target.

## Acceptance Gate

Before launching Post #1:

- [ ] All 288 existing tests pass
- [ ] Commits 1–8 land in sequence, each green
- [ ] `make evaluate-fast --corpus fastapi` matches pre-refactor numbers (within noise)
- [ ] K8s SOURCES.md exists with 30–40 URLs + rationales
- [ ] K8s golden dataset file exists with 25 questions in the v2 schema
- [ ] K8s refusal threshold tuned on the K8s golden set (placeholder value replaced)
- [ ] `make evaluate-fast --corpus k8s` produces real numbers in `results/`
- [ ] DECISIONS.md entries committed
- [ ] Manual smoke test: all 4 example chips on both corpora return sensible answers
- [ ] Cold-start measured on HF Spaces; under 60s target or lazy-load contingency activated

## Risks

1. **K8s curation time blows up.** 3–4h estimate assumes the curator (me) knows the K8s docs well enough to pick pages quickly. Mitigation: timebox to 4h, cut to 25 pages if needed — the architecture doesn't change.
2. **K8s threshold tuning produces a bad number.** If the K8s corpus is noisier than expected and no threshold gives clean grounded refusal, the demo will look broken. Mitigation: launch with a conservative threshold (prefer excessive refusal over false positives) and tune post-launch.
3. **FastAPI regression from prompt template change.** The new prompt wording is stricter. Mitigation: `make evaluate-fast --corpus fastapi` is in the acceptance gate; revert the prompt change if citation accuracy drops below 1.00.
4. **Cold-start exceeds 60s on HF Spaces.** Mitigation: documented contingency with 2h implementation budget.
5. **Chunker parameter drift.** If someone changes `chunk_size` or `chunk_overlap` later, all chunk IDs shift and the golden dataset breaks silently. Mitigation: `chunker` block pinned in the dataset header; migration script required for any parameter change.

## Out of Scope for This Refactor

- Runtime document ingestion ("bring your own docs")
- EU AI Act corpus (v1.2)
- Cross-corpus benchmark comparison
- Lazy-load implementation (contingency only)
- Threshold auto-tuning
- Semantic entropy / Lynx / OWASP — separate parts of the v1.1 plan

## References

- Yang et al., "CRAG: Comprehensive RAG Benchmark," NeurIPS 2024 — question taxonomy
- Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," NeurIPS 2021 — no-cross-corpus comparison principle
- Microsoft silver-to-gold evaluation methodology — three failure modes framework
