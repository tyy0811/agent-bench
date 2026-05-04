# Judge Layer v1 — Design Document

**Date:** 2026-05-04
**Status:** Approved — ready for implementation
**Author:** Jane Yeung
**Scope:** v1 of a discrete-scale, per-dimension LLM-judge layer with a κ-validated 2-judge jury and a 30-item hand-labeled calibration set. Supersedes the existing continuous-scale `answer_faithfulness` / `answer_correctness` judges. Mistral self-hosted 3rd judge, Langfuse self-host, dual-pass intra-rater calibration, and DSPy/GEPA prompt optimization are explicitly v1.1+.

---

## Goal

Replace the existing single-call, continuous-score, no-abstain LLM-judge implementation in `agent_bench/evaluation/metrics.py` with a per-dimension judge layer that supports anchored discrete rubrics, abstain, evidence quotes, judge identity, rubric versioning, and variance-controlled aggregation (rubric permutation, jury). Validate the new layer against a 30-item hand-labeled calibration set with Cohen's κ and bootstrap CIs. Produce a κ ablation table that quantifies the contribution of each variance control (anchored rubric, abstain option, rubric permutation, 2-judge jury) on top of the single-judge baseline.

The deliverable is the merged PR. The interpretive artifact is `judge-design.md` (a separate writeup file, not this design doc) which presents the κ table, the methodology, and the closing position on when *not* to use LLM-judge — drafted in the third day of the v1 scope window, sourced from the calibration runs produced by this design.

## Non-Goals

- **3rd-judge Mistral self-hosted via Modal.** Modal serving substrate exists from PR #8; deferring the third judge to v1.1 keeps the v1 jury at 2 members and the inference cost at the API-only floor.
- **Multi-seed self-consistency** (T=0 ensemble across seeds). Variance control via rubric permutation only in v1.
- **DSPy / GEPA / MIPROv2 prompt optimization.** Rubrics are hand-authored with anchored examples; automated optimization is v1.1+.
- **Length-bias study, bypass tests, full pass^k sweep.** Out-of-scope for v1.
- **Langfuse self-host integration.** Position paragraph in writeup §10 instead.
- **Dual-pass intra-rater calibration.** v1 cites the UK AISI bio/chem ceiling (κ ~0.8) as the literature reference; v1.1 may add intra-rater κ as an empirical ceiling.
- **Synthetic-anchor calibration set** (frontier-model-as-anchor). Methodologically delicate; v1.1+ if pursued.
- **Backward-compatible Optional fields on `EvalResult`.** Hard cut: `EvalResult.faithfulness` and `EvalResult.correctness` are removed. Existing run artifacts in `results/*.json` will not deserialize against the new schema; this is acceptable because those artifacts are documentation-of-history (read by humans), not inputs to live code, and none of the README's published numbers depend on the removed fields.

## Architecture

### Three-layer evaluation hierarchy

| Layer | What | Where | Cost | When |
|---|---|---|---|---|
| **L1 — Deterministic** | retrieval P@k/R@k, KHR, source_presence, grounded_refusal, citation_accuracy, calculator_used | `agent_bench/evaluation/metrics.py` (existing, untouched) | $0, CI-safe | every harness run |
| **L2 — LLM-judge** | per-dimension judges (groundedness, relevance, completeness; +citation_faithfulness opt-in), 2-judge jury, variance-controlled | `agent_bench/evaluation/{judges,rubrics,variance}/` (new) | ~$0.001–0.005/query | optional (`evaluation.judge_provider` set + `evaluation.judge_dimensions` non-empty) |
| **L3 — Human** | calibration set hand-labels (30 items × 3 dimensions) | `measurements/2026-05-04-judge-calibration-labels.jsonl` (new, hand-authored) | manual, one-time | once; locked |

L3 wraps L2 via the κ table; L1 wraps L2 by handling the cases regex can see (citation accuracy is the canonical example — v1 keeps the existing deterministic check; the writeup's §6 argues this is the right cut even after L2 exists).

### Module layout

Four new sibling subpackages under `agent_bench/evaluation/`. Sibling siblings — not nested under a single `judging/` parent — because the file tree should make the L1/L2/L3 hierarchy legible and `calibration/` is L3 evaluation infrastructure that *uses* `judges/`, not a sub-concern of judging.

```
agent_bench/evaluation/
  harness.py             # MIGRATED — drop inline _judge_call; plug in jury
  metrics.py             # KEEP deterministic; DELETE answer_faithfulness/answer_correctness/_judge_call/_FAITHFULNESS_PROMPT/_CORRECTNESS_PROMPT
  report.py              # existing
  datasets/
    tech_docs_golden.json    # existing — 8 items get source_snippets added (calibration subset only)
    k8s_golden.json          # existing
    k8s_golden_pilot.json    # existing
    calibration_v1.json      # NEW — 30 stratified item IDs, version field, system_config_git_sha
  judges/                # NEW
    __init__.py
    base.py              # Judge ABC, ScoreResult, Rubric loader, MockJudge, abstain-reason constants
    groundedness.py
    relevance.py
    completeness.py
    citation_faithfulness.py    # opt-in v1; default-on v1.1
  rubrics/               # NEW (markdown)
    groundedness.md
    relevance.md
    completeness.md
    citation_faithfulness.md
  variance/              # NEW
    __init__.py
    rubric_permute.py    # wraps Judge; permutes rubric levels; aggregates
    jury.py              # multi-judge aggregation: mean | kappa_weighted; quorum
  calibration/           # NEW
    __init__.py
    metrics.py           # cohen_kappa (linear/quadratic), gwets_ac2, bootstrap_ci — hand-rolled
    report.py            # markdown table generator → docs/_generated/kappa_table.md

tests/evaluation/        # NEW directory (precedent: tests/test_langchain_baseline/)
  __init__.py
  test_judges.py
  test_rubric_loading.py
  test_calibration_metrics.py
  test_jury_aggregation.py
  test_calibration_report.py
  test_harness_migration.py
  test_mockjudge_coverage.py
```

### Supersession of existing judges (dedicated subsection)

The new `Judge` ABC fully supersedes `answer_faithfulness`, `answer_correctness`, and `_judge_call` in `agent_bench/evaluation/metrics.py:167-208`. The old code is **deleted** (no deprecation cycle). The supersession changes six axes:

| Axis | Old (`_judge_call`) | New (`Judge` ABC) |
|---|---|---|
| **Scale** | continuous 0.0–1.0, no anchors | discrete (binary or 3-point) with rubric-anchored examples per level |
| **Reasoning placement in JSON** | `{"score": …, "reasoning": …}` — score first | `{reasoning, evidence_quotes, score}` — score conditions on reasoning |
| **Granularity** | combined "faithfulness" / "correctness" | per-dimension (groundedness / relevance / completeness; citation_faithfulness opt-in) |
| **Versioning** | none — judge_id, rubric, prompt all unrecorded | `judge_id`, `rubric_version` (SHA-256 of rubric file content), `prompt_seed`, `system_output_hash` traceable in every `ScoreResult` |
| **Variance control** | single call only | composable wrappers (`rubric_permute`, `jury`) |
| **Failure mode** | bare `except Exception` returns `None`; harness silently drops | intentional: `"Unknown"` abstain on rubric/model noise (with structured-prefix reason); raise on caller bugs (see Error Handling) |

**Config knob preservation.** `evaluation.judge_provider` YAML field stays (5 configs reference it; `core/config.py:89`). New judges accept `judge_provider: LLMProvider` matching the existing harness signature pattern. Zero user-facing config migration. New `evaluation.judge_dimensions: list[str]` field (default `["groundedness", "relevance", "completeness"]`); `citation_faithfulness` is opt-in v1, default-on v1.1, decoupling the citation deterministic-vs-LLM head-to-head from the harness migration.

**Coupled artifact updates** (in scope of the judge PR):
- `docs/DESIGN.md:346-356, 395` — rewrite §"LLM-judge metrics (costs money, manual)" to point at this design doc and `judge-design.md` (the writeup).
- `DECISIONS.md` — append one supersession entry. Entry references file paths explicitly: `measurements/2026-05-04-judge-calibration-labels.jsonl`, the relevant `results/calibration_v1_judge_*.json` files, and the κ table file path. References by file path, not abstract claim — the supersession is defended by the calibration data, not by description.
- `measurements/README.md` — append one row pointing at the new calibration-labels file (otherwise it orphans next to the cold-start logs).
- `README.md` — add a "Targets that cost money" subheading (separate concern; see the README cost-disclosure obligation under Testing).

### Dependency direction

Judge → Rubric (filesystem markdown loader) → existing `LLMProvider` ABC at `agent_bench/core/provider.py`. **No new external runtime dependencies.** Cohen's κ, Gwet's AC2, and bootstrap CI are hand-rolled (rationale in `calibration/metrics.py` under Components). scikit-learn is *not* added to the project; sklearn appears only in dev tooling under `scripts/_dev/` (see the sklearn fixture pattern under Testing).

## Components

### Rubric (the spec object)

```python
class Rubric(BaseModel):
    dimension: Literal["groundedness", "relevance", "completeness", "citation_faithfulness"]
    scale: Literal["binary", "three_point"]
    reference_based: bool
    abstain_allowed: bool
    levels: list[RubricLevel]   # parsed from markdown sections
    body_markdown: str           # full file contents

    @property
    def source_hash(self) -> str:
        # SHA-256 of body_markdown — immutable per file content, independent of git
        ...

    def render_prompt(self, *, level_permutation_seed: int = 0) -> str:
        # if seed > 0, permute self.levels deterministically using PRNG(seed)
        ...
```

**Two-hash provenance.** `source_hash` (SHA-256 of canonical body) is immutable per rubric file; `prompt_seed` (per-call int, 0 = no permutation) is recorded on the call. κ aggregation groups by `source_hash`; ScoreResults with the same `source_hash` and different `prompt_seed` are agreement-eligible against the same label. Both fields appear in every `ScoreResult` so records are self-contained.

Loader reads markdown with YAML frontmatter (matching repo convention). Anchored examples are parsed by section header pattern (`## Score 0`, `## Score 1`, …) so level-permutation rewrites the prompt by reordering sections.

**Construction validates aggressively** (see Rubric construction validation under Error Handling): scale ∈ {binary, three_point}, levels arity matches scale, every level has at least one anchored example with thinking-trace explanation, frontmatter has all required fields. ValidationError raises with file path + field path. Failing at rubric construction (Day 1) is much cheaper than failing on first `judge.score` call (Day 2 with API budget already spent).

### ScoreResult (per-call record)

```python
class ScoreResult(BaseModel):
    # Reasoning-first ordering — matters for Pydantic field order
    # AND for the JSON schema sent to the model
    reasoning: str
    evidence_quotes: list[str] = Field(default_factory=list)
    score: int | Literal["Unknown"]

    # Provenance (self-contained — no run-metadata cross-reference needed)
    judge_id: str              # f"{model_id}_{dimension}", e.g. "claude-haiku-4-5_groundedness"
    rubric_version: str        # = Rubric.source_hash
    prompt_seed: int = 0
    system_output_hash: str    # SHA-256 of canonical (item.id, output.answer, sorted(output.sources))

    # Operations
    cost_usd: float
    latency_ms: float

    @property
    def abstained(self) -> bool:
        return self.score == "Unknown"
```

`score` is `int | Literal["Unknown"]` (not `int | None`) so abstain is structurally distinct from "we don't have a value yet" — the silent-`None` failure mode that the old `_judge_call` exhibited becomes impossible.

`system_output_hash` is the cross-run-aggregation guard: scores are agreement-eligible iff `(item.id, dimension, system_output_hash)` match. Any mismatch between labels and predictions raises in the calibration report (see Calibration report failure modes under Error Handling).

### Judge ABC + concrete judges

```python
class Judge(ABC):
    def __init__(self, judge_provider: LLMProvider, rubric: Rubric, model_id: str):
        self.judge_provider = judge_provider
        self.rubric = rubric
        self.model_id = model_id
        self.judge_id = f"{model_id}_{rubric.dimension}"

    @abstractmethod
    async def score(
        self,
        item: GoldenQuestion,
        output: AgentResponse,
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult: ...
```

Concrete judges (`GroundednessJudge`, `RelevanceJudge`, `CompletenessJudge`, `CitationFaithfulnessJudge`) are thin per-dimension classes (~30 lines each), no shared base method. Factoring the prompt-assembly into a base method is rejected: at 3–4 judges of 30 lines each, each is more readable in full than as a delta against a base, and a shared base creates a future trap where dimension-specific logic creeps into the base via `if self.dimension == ...` branches.

**Per-judge input expectations** (matters for the FastAPI snippet-authoring scope):

| Judge | Reads from `item` | Reads from `output` |
|---|---|---|
| `GroundednessJudge` | `source_snippets` (the 8 FastAPI calibration items get hand-snippeted; see FastAPI snippet authoring under Calibration Methodology) | `answer` |
| `RelevanceJudge` | `question` only | `answer` |
| `CompletenessJudge` | `reference_answer` | `answer` |
| `CitationFaithfulnessJudge` | `source_chunk_ids` + retrieved-chunk text | `answer` (parsed for claims + citations) |

`CitationFaithfulnessJudge` returns one aggregate `ScoreResult` per item (preserving ABC polymorphism), with per-pair (claim, citation) detail in `evidence_quotes`. Aggregation rule for binary: **all-or-nothing** — any unfaithful citation → score=0. The rule is documented explicitly in `rubrics/citation_faithfulness.md`.

### MockJudge

Same shape as `Judge`; constructor takes `verdicts: dict[str, ScoreResult]` keyed by `item.id`. Returns the pre-baked verdict on `score()`, no API call. **Raises `LookupError` on missing keys** — never returns a default — so test fixtures are self-checking. A separate fixture-validation test (`test_mockjudge_coverage.py`) walks `item.id` across all goldens and asserts every MockJudge instance has coverage for items its tests reference. Two-layer defense against the rename-breaking-tests failure mode. Mirrors the `MockProvider` pattern at `agent_bench/core/provider.py:118`.

### rubric_permute (variance wrapper)

```python
def rubric_permute(judge: Judge, n: int = 2, seeds: list[int] | None = None) -> PermutedJudge: ...
```

`PermutedJudge.score(item, output)` runs `judge.score(item, output, prompt_seed=s)` for each `s` in `seeds` (default `[1, 2]`), aggregates:
- Binary: majority (n=2 → tie-break to lower score, more conservative)
- Three-point: mean, rounded to nearest level **with ties broken downward** (e.g., 1.5 → 1, 0.5 → 0); same conservative principle as the binary tie-break
- **Any abstain → "Unknown"** (any sample, not all): the whole point of rubric permutation is to surface whether judge behavior depends on prompt structure; averaging an abstain away with a confident sample defeats the technique. At N=2, "all abstain" essentially never fires, making it a silent aggressive default. "Any abstain → Unknown" is the conservative choice that preserves the variance signal.

Returns one `ScoreResult` with `judge_id = f"{judge.judge_id}_perm{n}"`, `prompt_seed=0` on the aggregate. Per-permutation results are written to a sidecar JSONL (same pattern as the jury subsection below) for traceability.

### jury (multi-judge aggregator)

```python
def jury(
    judges: list[Judge],
    aggregation: Literal["mean", "kappa_weighted"],
    weights: dict[str, float] | None = None,         # required if kappa_weighted
    quorum: int | None = None,                       # default: len(judges) — strict
    sidecar_path: str | None = None,                 # default: results/calibration_v1_judge_{aggregation}_members.jsonl
) -> Jury: ...
```

`Jury.score(item, output)` runs `asyncio.gather(*[j.score(item, output) for j in judges], return_exceptions=False)` with try/except at the jury level (so non-retryable exceptions cancel sibling tasks immediately — failing fast on caller bugs). Per-member ScoreResults always written to sidecar (successes and failure-as-abstains alike). Aggregate behavior:

1. Count `successful_members = sum(1 for r in member_results if not r.abstained)`.
2. If `successful_members < quorum`: aggregate = `ScoreResult(score="Unknown", reasoning=f"jury_below_quorum: {successful_members}/{len(judges)} members succeeded; required {quorum}", ...)`.
3. Else: aggregate using `aggregation` strategy over the successful members' scores. **Discretization rule (same as `rubric_permute`):** binary scores threshold at 0.5 with ties → 0; three-point scores round to nearest with ties → lower level. Discretization happens at the aggregation step, before the κ join — Cohen's κ requires both inputs discrete.

**Strict quorum default for v1.** `quorum=N` (= `len(judges)`) at v1's 2-judge jury means any member abstain → jury abstain. Tolerant defaults at N=2 are silent single-judge in jury clothing. The parameter exists in v1 so v1.1's 3-judge jury can shift to `quorum=2` (majority) without rearchitecting failure semantics.

`kappa_weighted` requires explicit `weights` injection — computed offline once on the calibration set, *not* at jury construction (would be circular).

### calibration/metrics.py (hand-rolled κ + bootstrap)

```python
def cohen_kappa(
    y1: list[int | str], y2: list[int | str],
    weights: Literal[None, "linear", "quadratic"] = None,
) -> float

def gwets_ac2(
    y1: list[int | str], y2: list[int | str],
    weights: Literal[None, "linear", "quadratic"] = None,
) -> float

def bootstrap_ci(
    y1: list, y2: list, metric_fn: Callable[[list, list], float],
    n_iter: int = 1000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float, float]   # (point_estimate, ci_lo, ci_hi)
```

**Hand-rolled, not sklearn.** Adding scikit-learn for one function (and transitively numpy + scipy + threadpoolctl + joblib) contradicts agent-bench's "built from primitives" identity. The hand-roll also serves the writeup: `(P_o − P_e) / (1 − P_e)` with explicit `P_e` computation demonstrates formula understanding in a way that an `sklearn.metrics.cohen_kappa_score` import does not. Fixture-tested against sklearn run *outside* the project venv (see the sklearn fixture pattern under Testing).

**Abstain handling in κ.** Excluded pairwise — if either side abstains on item *i*, item *i* drops from that κ calculation. Standard treatment (Tu et al. 2024, *Beyond Correlation*); abstain as "I don't know" is neither agreement nor disagreement. Abstain count per dimension is reported separately by the calibration report (see `calibration/report.py` below).

**Gwet's AC2 deferred from headline numbers.** AC2 is implemented in v1 but the published numbers in the v1 writeup come from κ only; AC2 fixture-test rigor (sympy-derived intermediate steps, not arithmetic-derived) is v1.1 work. Hand-computed AC2 fixtures in v1 cover three inspection-verifiable cases (perfect agreement, perfect disagreement, mid-range).

### calibration/report.py

One function: `generate_kappa_table(predictions_glob, labels_path, output_path, *, strict: bool = False)` → writes `docs/_generated/kappa_table.md`. Idempotent. Joins predictions ⋈ labels on `(item_id, dimension, system_output_hash)`; raises on hash mismatch (collect-all, error includes first-item expected/actual hashes plus full mismatched-id list). Computes per-config, per-dimension κ + bootstrap CI + abstain rate; flags rows where abstain rate **strictly greater than** 20% with a footnote (`"κ computed on N=X of 30 items; high abstain rate (Y% — breakdown: Z% schema parse, W% genuine abstain) suggests rubric ambiguity"`).

**Two modes for missing predictions/labels:**
- Default: WARN-and-exclude (Day-2 development loop — partial coverage is real interim state).
- `--strict`: RAISE on any missing prediction/label (final-artifact path; `make calibrate` invokes this; the writeup is by-construction produced from `--strict` output).

The κ table is copy-pasted into the writeup at draft time, not include-by-reference — the writeup is a frozen v1 artifact and copy-paste lets the writeup add inline annotations to specific cells.

## Data Flow

### Production harness run (existing, migrated)

```
golden file → load_golden_dataset() → list[GoldenQuestion]
  → for each item, parallel:
      orchestrator.run() → AgentResponse
      compute L1 metrics (existing — untouched)
      if judge_provider is not None and item.category != "out_of_scope":
          system_output_hash = hash(item.id, response.answer, sorted(response.sources))
          for each Judge in evaluation.judge_dimensions:
              ScoreResult = await judge.score(item, response)
          attach to EvalResult.judge_scores: dict[str, ScoreResult]
  → write results/{run_label}.json
```

**Migration delta** at `agent_bench/evaluation/harness.py:153-166`:
- DELETE inline import of `answer_faithfulness, answer_correctness`
- DELETE `result.faithfulness = ...` and `result.correctness = ...` assignments
- ADD: load configured judges from `evaluation.judge_dimensions` config; build with existing `judge_provider`
- ADD: `result.judge_scores: dict[str, ScoreResult]` field on `EvalResult`
- KEEP: `if judge_provider is not None and q.category != "out_of_scope"` gate (out-of-scope items still bypass L2; refusal is deterministic)
- KEEP: `evaluation.judge_provider` YAML field (5 configs reference it)

### Calibration run (new)

```
calibration_v1.json (30 IDs + version + system_config_git_sha)
  → filter k8s_golden.json + tech_docs_golden.json → 30 GoldenQuestions

Step A (once, frozen): generate system outputs
  → orchestrator.run() with frozen config for each item
  → write results/calibration_v1_system_outputs.json
     (each record includes system_output_hash, item_id, answer, sources, source_chunks, citations)

Step B (manual): hand-label
  → labeling notebook reads system_outputs file, injects system_output_hash automatically
  → for each (item, dimension), human authors score + notes
  → append to measurements/2026-05-04-judge-calibration-labels.jsonl
     {item_id, dimension, score | "Unknown", abstained, notes, label_timestamp, system_output_hash}

Step C (per ablation row): score with judges
  → load row config from configs/calibration/rows/{label}.yaml
  → load system_outputs file (frozen)
  → for each item, judge.score(item, output) per row's judge configuration
  → write results/calibration_v1_judge_{row_label}.json
     and (jury rows) results/calibration_v1_judge_jury_{aggregation}_members.jsonl

Step D (κ table):
  → calibration/report.generate_kappa_table(strict=True for final artifact)
  → join predictions ⋈ labels on (item_id, system_output_hash); raise on mismatch
  → exclude pairs where either side abstains
  → cohen_kappa + bootstrap_ci + abstain_rate per (config, dimension)
  → write docs/_generated/kappa_table.md
```

**Hash propagation through labels** is intentional: labels carry `system_output_hash` because they are tied to specific outputs. If `system_outputs` are ever regenerated (config change, retry), labels become stale and the κ join raises loudly. This eliminates the cross-run aggregation bug class.

### Concurrency

- **Within an item, across judges (jury):** `asyncio.gather` over `judges`; existing provider rate-limit/retry kicks in.
- **Across items in a calibration row:** `asyncio.gather` with semaphore, default concurrency=5, configurable via CLI flag with config-field fallback. **Resolved value logged at run start** so artifacts capture which concurrency was used.
- **Across rows of the ablation:** rows run sequentially. Each row writes its predictions file before the next starts — partial progress survives interruption.

### New scripts and Makefile targets

```
scripts/
  evaluate.py             # existing — full-corpus harness runs
  run_calibration.py      # NEW — orchestrates Steps A, C, D
                          #   subcommands: generate-outputs | run-judges --row-config=<path> | build-table [--strict]
                          # Step B (labeling) is manual — done in a notebook
configs/calibration/rows/  # NEW — one YAML per ablation row (config-file-per-row)
  baseline.yaml
  baseline_no_cot.yaml
  baseline_no_anchors.yaml
  baseline_no_abstain.yaml
  permute.yaml
  jury_kappa_weighted.yaml

Makefile:
  calibrate             # runs full pipeline: generate-outputs → run-judges (all rows) → build-table --strict
  evaluate-judges       # runs run-judges + build-table against existing system_outputs (no regeneration)
```

Row configs are independently versioned reproducible artifacts in the PR. `run-judges` is a generic runner taking `--row-config=<path>`; the script does not own the row inventory. Discovering a bug in row 4 means fixing row 4's config and rerunning rows 4-6 without touching 1-3.

### Failure modes eliminated by this design

| Bug class | Eliminated by |
|---|---|
| Cross-run aggregation (run-A outputs scored against run-B labels) | `system_output_hash` join with raise-on-mismatch |
| Stale labels after system re-run | Same |
| MockJudge silently passing tests with renamed item IDs | `LookupError` on missing keys + fixture-validation test |
| Single-call judge bias hidden | Rubric permutation surfaces it via abstain propagation |
| Per-judge κ unrecoverable from jury aggregate | Sidecar JSONL with deterministic path |
| Partial progress lost on Step C interruption | One predictions file per row, written sequentially |
| Schema parse failures silently dropped (old `_judge_call` `None`) | Discrete abstain-with-prefix; abstain rate flagged at >20% |
| Final writeup citing N=28 while prose claims N=30 | `--strict` mode for final-artifact build; default warns |

## Error Handling

### Failure taxonomy at L2

| Category | Source | Where caught | Decision |
|---|---|---|---|
| Provider retryable (rate limit, timeout, network) | Infra | Existing `LLMProvider` retry/backoff | Bubbles up only on retry exhaustion |
| Provider exhausted (retries exhausted) | Infra | `Judge.score` | Abstain with `ABSTAIN_REASON_PROVIDER_EXHAUSTED` |
| Provider non-retryable (401, 400) | Caller misconfig | `Judge.score`; jury cancels siblings | **Raise** — bug, not noise |
| Schema parse error | Model glitch or broken prompt | `Judge.score` | Abstain after one strict-reprompt retry; `ABSTAIN_REASON_SCHEMA_PARSE` |
| Score out of range | Model glitch | `Judge.score` | Abstain after one strict-reprompt retry; `ABSTAIN_REASON_OUT_OF_RANGE` |
| Genuine model abstain (rubric allows) | Model judgment | `Judge.score` | Abstain with empty-prefix sentinel (`ABSTAIN_REASON_GENUINE` = `""`) |
| Hash mismatch on κ join | Stale labels | `calibration/report.py` | Raise after collect-all; first-item expected/actual hashes in message |

### The abstain-vs-raise discipline

**One retry with strict reprompt** on schema parse / score out of range. Original prompt's formatting instructions are augmented at the end with a recency-positioned reminder: `STRICT FORMATTING NOTE: respond ONLY with a JSON object matching the schema; reasoning first, then evidence_quotes, then score`. If second attempt also fails, abstain with structured-prefix reason. **Exactly one retry** — zero retries throws away signal that recovers cheaply; N>1 retries silently mask systematic schema breaks.

**Failure-reason prefixes as constants** in `judges/base.py`:

```python
ABSTAIN_REASON_PROVIDER_EXHAUSTED = "judge_call_failed_after_retry: "
ABSTAIN_REASON_SCHEMA_PARSE       = "schema_parse_failed_after_retry: "
ABSTAIN_REASON_OUT_OF_RANGE       = "score_out_of_range_after_retry: "
ABSTAIN_REASON_GENUINE            = ""   # empty-prefix sentinel for rubric-allowed abstain
```

Calibration report imports + pattern-matches against typed constants for the four-way abstain-cause breakdown in the >20% threshold flag.

### First-attempt-failure log schema (fires on success-after-retry too)

WARN-level structured log line, fixed key set, no schema drift. Uses `structlog` matching repo precedent at `agent_bench/evaluation/metrics.py:14` (`logger = structlog.get_logger()`):

```python
logger.warning(
    "judge_first_attempt_failure",
    judge_id=self.judge_id,
    item_id=item.id,
    provider=type(self.judge_provider).__name__,
    failure_cause=ABSTAIN_REASON_SCHEMA_PARSE,  # one of the four constants
    attempt_index=1,
)
```

Fires on first-attempt failure regardless of whether the second attempt succeeds. The "first failed, second succeeded" branch is the most analytically interesting case — it tells you the reprompt is doing work rather than just consuming budget. Without this log, that branch is invisible.

### Jury partial-failure (quorum)

Per the jury subsection above: strict quorum default; per-member ScoreResults always written to sidecar; aggregate is `score="Unknown"` with `jury_below_quorum` reason if `successful_members < quorum`. Provider non-retryable in any member → jury raises immediately, cancels sibling `gather` tasks (the `return_exceptions=False` + try/except pattern; *not* `return_exceptions=True` + inspection — the two look identical to a careless reader but only the former cancels siblings).

### Permutation wrapper failure

Per the `rubric_permute` subsection above: any-permutation abstain → aggregate abstain. Per-permutation results written to sidecar.

### Rubric construction validation

`Rubric.from_markdown_file()` validates aggressively: scale ∈ {binary, three_point}, levels arity matches scale, every level has at least one anchored example with thinking-trace explanation, frontmatter has all required fields. ValidationError raises with file path + field path. Validation discipline is named explicitly in the spec because the alternative ("validate lazily on first score call") is the kind of thing that creeps in if not specified — and a malformed-rubric error on Day 2 after API budget has been spent is materially worse than a malformed-rubric error on Day 1.

### Calibration report failure modes

| Condition | Default behavior | `--strict` behavior |
|---|---|---|
| Hash mismatch | Raise after collect-all (first item expected/actual + full id list) — **applies to both modes; never warn** | Same |
| Missing prediction (label exists, no prediction for `(item_id, dim)`) | WARN; exclude from κ; coverage row in footer | RAISE |
| Missing label (prediction exists, no label) | WARN; exclude; coverage row in footer | RAISE |
| κ undefined (insufficient variance after exclusion, or N<3 agreement-eligible) | Render `"—"` with footnote — **applies to both modes** | Same |
| Abstain rate > 20% (strictly greater) | Render κ + footnote with cause breakdown — **applies to both modes** | Same |

## Testing

### File layout

Six new files under `tests/evaluation/` matching the new module subpackages. Existing `tests/test_evaluation.py` stays at top level (precedent: `tests/test_langchain_baseline/`); the existing file's faithfulness/correctness assertions are dropped, but the file is not renamed (preserves git blame).

### sklearn fixture pattern (κ parity tests)

Four-part discipline:

1. **Generation script** at `scripts/_dev/generate_kappa_fixtures.py` — committed; `_dev` prefix marks as not-runtime. Imports sklearn; documented to run from a venv outside the project. **Action item:** verify `_dev/*` is excluded from ruff/mypy via `pyproject.toml` (currently no `extend-exclude` set; add as part of this PR).
2. **Inline constants** in `test_calibration_metrics.py` — `SKLEARN_KAPPA_FIXTURES: dict[str, float]` and `SKLEARN_KAPPA_INPUTS: dict[str, dict]`. Locality preserved, type-checked.
3. **Version-pinned comment header** — `# Fixtures generated against scikit-learn==1.5.2 cohen_kappa_score on 2026-05-04` with regeneration instructions. Drift detection if sklearn behavior changes in a future version.
4. **Load-bearing comment** — `# DO NOT add scikit-learn to the project's dependencies — these constants are the contract.` Prevents the well-meaning future contributor from "fixing" tests by importing sklearn at runtime.

**Cross-check CI test:** the generation script writes its inputs to a JSON sidecar under `tests/evaluation/fixtures/sklearn_kappa_inputs.json`; a CI test asserts `SKLEARN_KAPPA_INPUTS` matches that JSON. Catches the "updated CASES list, forgot to regenerate" failure mode at CI time. Five lines of test code.

**No sklearn parity for AC2 in v1.** sklearn doesn't have AC2; pulling `irrCAC` reintroduces the dependency problem one level over. Three hand-computed AC2 cases (perfect agreement, perfect disagreement, mid-range) where the formula reduces to inspection-verifiable values. v1.1 may add sympy-derived AC2 fixtures (script under `scripts/_dev/generate_ac2_fixtures.py` with sympy as dev-only dep, sympy intermediate steps printed for audit). v1.1 spec line: *"AC2 hand-computed fixtures are sympy-derived not arithmetic-derived; verification requires reading the sympy intermediate output, not just inspecting the test."*

### Test inventory (~30 tests total)

| File | Tests | Notes |
|---|---|---|
| `test_judges.py` | ~7 | ABC contract, MockJudge round-trip + LookupError, ScoreResult validation, abstain-with-prefix (parameterized over 3 causes), raise on non-retryable, first-attempt-failure log fires |
| `test_rubric_loading.py` | ~6 | Construction validation (parameterized over 4 invalid cases), source_hash determinism, source_hash changes with content, permutation seed reproducibility, permutation changes prompt |
| `test_calibration_metrics.py` | ~7 | 3 hand-computed κ cases + 3 sklearn-fixture parity + 1 bootstrap-CI seed reproducibility |
| `test_jury_aggregation.py` | ~5 | mean, kappa_weighted, strict-quorum-abstain, sidecar capture, cancel-on-non-retryable |
| `test_calibration_report.py` | ~6 | hash-mismatch with first-item detail, --strict raise, default WARN, undefined-κ dash, abstain-flag boundary 6/30 (does not fire) and 7/30 (fires), abstain breakdown by cause |
| `test_harness_migration.py` | ~3 | judge_scores populated when configured, out_of_scope skipped, judge_provider config preserved |
| `test_mockjudge_coverage.py` | ~1 | item.id walk across all goldens |
| **Total** | **~35** | |

The original "~15–20" estimate was made before the Error Handling section was designed. Designing error handling and not expanding the test count is the inconsistency: the abstain-cause logic is the highest-stakes-when-silently-wrong piece of the project (wrong abstain semantics → quietly wrong κ in the published report). If Day 3 budget runs short, the cuttable margin is `test_harness_migration.py` (integration-y, failures show up loudly); the metric-correctness and judge-failure-handling tests do not get cut.

### Discipline conventions

- Mocked providers everywhere. Zero network calls in CI. `MockProvider` for the underlying LLM; `MockJudge` for tests that need pre-baked verdicts.
- `pytest-asyncio` (`asyncio_mode = "auto"` already set) for async tests.
- Hand-computed κ cases include worked-out arithmetic in a comment block so a reader can verify the formula without running the test.
- Larger reusable fixtures live under `tests/evaluation/fixtures/`; one-off small fixtures stay inline.

### CI scope

- All ~35 new tests run in `make test` in the existing GitHub Actions workflow. No new workflow files.
- `make lint` covers new modules (ruff + mypy).
- `make calibrate` and `make evaluate-judges` are **not** run in CI — they require API keys and burn budget. Manual invocation only.
- **GitHub Actions config** explicitly omits provider keys via an empty `env:` block, preventing the "PR worked in upstream because secret was injected; fails in contributor's fork because no secret" failure mode.

### README cost-disclosure obligation (separate from spec)

`README.md` gets a "Targets that cost money" subheading with a four-column table (target, requires API key, approximate cost, what it produces). Not part of the spec body — a doc obligation owed to anyone running `make help` who shouldn't have to read the spec to know that `make calibrate` costs ~$2.

## Calibration Methodology

### Stratified sampling (30 items)

Stratification across the actual 52 golden items (FastAPI 27 + K8s 25):

FastAPI uses `category` as the stratification axis (the only typing in `tech_docs_golden.json`); K8s uses `question_type` (the CRAG 8-type taxonomy in `k8s_golden.json`). The 2 K8s items with `category: out_of_scope` are subsumed within their question_type stratum (most are within `false_premise`); they are not a separate K8s stratum.

| Stratum | Available | Sampled |
|---|---|---|
| FastAPI retrieval | 19 | 5 |
| FastAPI calculation | 3 | 1 |
| FastAPI out-of-scope | 5 | 2 |
| K8s simple | 6 | 4 |
| K8s simple_w_condition | 4 | 3 |
| K8s comparison | 4 | 3 |
| K8s multi_hop | 6 | 4 |
| K8s false_premise | 4 | 3 |
| K8s set | 1 | 1 |
| **Subtotal stratified** | **52** | **26** |
| Spare slots (filled from highest-variance R@5 strata) | — | 4 |
| **Total** | — | **30** |

The K8s `time_sensitive=True` flag is an overlay attribute, not an exclusive stratum — 2 K8s items carry the flag and are sampled incidentally based on the question_type they belong to. The flag does not constrain sampling.

**OOS items in calibration.** The 2 FastAPI items with `category: out_of_scope` (and however many of the sampled K8s false_premise items also carry `category: out_of_scope` — at most 2, since K8s has 2 OOS items total) follow the production harness gate: L2 judges are **skipped** for items where `category == "out_of_scope"` (the existing gate at `harness.py:153`). OOS items are still in the calibration set so that L1's `grounded_refusal` is exercised on the same items that produced labels. The κ-eligible item count per dimension is therefore at most 28 (30 minus the 2 FastAPI OOS) and possibly 26 (if both K8s OOS items get sampled into the K8s false_premise stratum); the writeup's κ table reports the actual N per row. This is the right cut because OOS handling is L1's job (deterministic refusal check) — judging "groundedness of a refusal" is methodologically incoherent (nothing to ground against).

IDs locked in `agent_bench/evaluation/datasets/calibration_v1.json` with `version: "v1"` field and `system_config_git_sha: <commit>` (the git SHA of the commit producing `system_outputs_v1.json` — name carries the limitation; v1.1 may add `system_config_resolved_hash` for stricter reproducibility).

### FastAPI snippet authoring (calibration set only)

The 8 FastAPI items in the calibration set get hand-snippeted before labeling begins. Snippets are **verbatim spans** from `data/tech_docs/`, not paraphrases — same convention as the existing K8s `source_snippets`. **Scope discipline:** only the 8 calibration items, not the full 27-item FastAPI golden. The remaining 19 FastAPI items can be backfilled in v1.1.

If a verbatim span supporting the gold answer cannot be found, the gold answer is itself underspecified and the item is removed from the calibration set (replaced from the spare-slot stratum).

Slots into Day 1 between sampling and labeling; ~30 min of additional work; Day 1 budget shifts from 8h to 8.5h.

### Hand-labeling rules

- Score by the rubric, not by intuition. If the rubric and intuition disagree, fix the rubric *after* the labeling pass — do not change the labels mid-pass.
- Genuine uncertainty → `abstained: true` with note. Abstains are signal.
- Track time per item; >2 minutes → rubric ambiguity, note it.
- **No AI assistance on label values.** AI may help with the labeling notebook, JSONL formatting, schema validation. Label values are hand-authored.

### Opus stress-test (rubric ambiguity assist)

After hand-labeling, Claude Opus labels the same 30 items × 3 dimensions blind to the human labels. Disagreements are flagged as `rubric_ambiguous` for v1.1 rubric revision. **Labels are not changed.** The Opus output is a rubric-quality signal, not a ground-truth substitute. ~20 minutes of work; methodological texture for the writeup's calibration section.

## Implementation Sequencing Notes

### Rubric authoring order

Write the **groundedness rubric first**, alone. Dry-fit it against 3–4 calibration items to test operationalizability before authoring the other two. *Then* write relevance and completeness using whatever pattern worked for groundedness. This converts rubric authoring from "three parallel risky tasks" into "one risky task plus two near-mechanical replications," compressing realistic time variance and reducing spillover risk. The dry-fit step is what makes the tactic load-bearing: if groundedness turns out to be ill-shaped, you know after one rubric, not after three.

### Contingency cuts (priority order)

If scope pressure forces cuts:

1. Drop the citation deterministic-vs-LLM head-to-head section of the writeup (this section was already a stretch goal).
2. Drop the per-judge individual κ table — keep only the variance ablation.
3. Reduce the variance ablation to 4 rows (baseline → CoT → rubric+abstain → 2-judge jury), skipping rubric-permute.
4. Reduce calibration set to 20 items if labeling has slipped — cite literature ceiling more heavily.

**Do not cut:** the writeup itself, the κ numbers, the rubric files, the closing position-statement paragraph (when NOT to use LLM-judge). Those are non-negotiable.

## Acceptance Gates

Two gates with different scopes. The code PR is reviewable and mergeable independently of the writeup; coupling them creates an artificial blocker.

### PR-open gate (required to merge `feat/judge-layer-v1`)

- All ~35 new tests pass; full `make test` suite green; `make lint` clean.
- `make calibrate --strict` runs end-to-end from a clean checkout (with API keys) and produces `docs/_generated/kappa_table.md`.
- `agent_bench/evaluation/metrics.py` no longer contains `answer_faithfulness`, `answer_correctness`, `_judge_call`, `_FAITHFULNESS_PROMPT`, or `_CORRECTNESS_PROMPT`.
- `agent_bench/evaluation/harness.py` no longer imports the deleted functions; new judges populate `EvalResult.judge_scores`.
- `evaluation.judge_provider` YAML field still functions (regression test).
- DECISIONS.md has the supersession entry referencing file paths explicitly.
- `docs/DESIGN.md` §"LLM-judge metrics" is rewritten to point at this design doc and `judge-design.md`.
- `measurements/README.md` has the new row.
- `README.md` has the "Targets that cost money" subheading.
- `pyproject.toml` excludes `scripts/_dev/*` from ruff/mypy if not already excluded.
- GitHub Actions workflow has an explicit empty `env:` block on the test job (verified to be documentation of existing behavior, not a behavior change — current workflow has no `env:` block and tests already run without provider keys via MockProvider).

### v1-completion gate (lags PR merge by 1–2 days)

The writeup is interview material, not a PR-merge dependency. It is produced from the merged PR's calibration runs and is committed separately.

- `judge-design.md` (the writeup, separate file at `docs/judge-design.md`) is drafted with the κ ablation table copy-pasted in from `docs/_generated/kappa_table.md`.
- DECISIONS supersession entry's file-path references resolve (the calibration-labels JSONL and the relevant `results/calibration_v1_judge_*.json` files exist on `main` post-merge).

## Out of Scope (v1.1+)

- 3rd judge (Mistral self-hosted via Modal) and quorum=2 default for the 3-judge jury.
- Multi-seed self-consistency (T=0 ensemble) on top of rubric permutation.
- DSPy / GEPA / MIPROv2 prompt optimization for rubric refinement.
- Length-bias study, bypass tests, full pass^k sweep.
- Langfuse self-host integration (judge call traces, cost dashboards).
- Dual-pass intra-rater calibration (4–6 day calendar gap; replaces literature ceiling with measured intra-rater κ in the writeup).
- Synthetic-anchor calibration set scaling (frontier-model-as-anchor on 200 items).
- AC2 sympy-derived parity tests (sympy as dev-only dep; intermediate steps printed for audit).
- Backfill `source_snippets` for the remaining 19 FastAPI golden items (only the 8 calibration items get snippets in v1).
- `system_config_resolved_hash` (canonical serialization of resolved config) added alongside `system_config_git_sha` for stricter reproducibility across noise commits.
- Citation faithfulness default-on (currently opt-in v1; `judge_dimensions` default extends to include it in v1.1).

## Risks

| Risk | Mitigation |
|---|---|
| Day 1 rubric authoring overflows 2.5h budget | The rubric-authoring sequencing tactic (Implementation Sequencing Notes) compresses variance; if all three rubrics need full 2.5h each, fall back to the Contingency cuts subsection |
| Bootstrap CI half-width >0.15 at N=30 (κ values not defensibly distinct between rows) | Note in writeup; reduces strength of comparative claims but doesn't invalidate the table |
| Jury κ worse than the better individual judge (kappa-weighting wrong, or worse judge drags mean) | Sanity-check before final table; possible switch to trimmed mean; sidecar JSONL preserves per-judge data either way |
| Schema parse failures spike >20% on one dimension (rubric-prompt mismatch) | Abstain-rate flag surfaces in the report; fix prompt or rubric, rerun affected row only (config-file-per-row makes this cheap) |
| Hand-labeling time exceeds 2h budget | Reduce to 20-item subset (contingency cut #4); cite literature ceiling more heavily in writeup |
| Branch state at start (in-flight `docs/readme-test-count` README diff) | Land that 4-line PR first (~5 min — README test-count only; the previously-pending Option A DECISIONS entries and the warmup-penalty addendum already landed via commit `6409a40` on 2026-04-22, so they are not on the docs-PR critical path); branch `feat/judge-layer-v1` off updated main |

---

**End of design document.** Implementation plan to follow in `docs/plans/2026-05-04-judge-layer-v1-implementation.md` (produced via the `writing-plans` skill).
