# agent-bench v3.1 Statistics Layer: Design Document

**Date:** 2026-06-11
**Status:** Approved pending one verification (SCHEMA_VERSION, section 3.1)
**Author:** Jane Yeung
**Scope:** Statistics and reporting layer over the existing evaluation harness. Architecture, work-package structure, guardrails, and budget are fixed by the v3.1 implementation plan (Claude Code edition, v0.2) and are not restated here. This document records what that plan left open: the resolved [DECIDE] items, the pre-registered analysis rules, the schema restatement and its conventions, the named hook points, and the report-generator requirements. The companion implementation plan follows as `docs/plans/2026-06-11-stats-layer-v3.1-implementation.md`.

**Relation to the engine plan:** this work implements Phase 1 of the eval-statistics-engine plan. That plan lives outside this repository by design; it carries scope and positioning that does not belong in a public portfolio repo. Nothing in this repository requires it to resolve: the schema is restated verbatim in section 3.2 and pinned by `SCHEMA_VERSION` in `stats/schema.py`.

---

## Goal

Turn agent-bench's point-estimate benchmark tables into measurements with defensible uncertainty: 95 percent intervals, clustered standard errors with an honest few-cluster policy, paired tests, a pre-registered TOST equivalence claim for the framework comparison, a power statement, an exact zero-failure bound on the citation-accuracy claim, and (droppable) judge-agreement CIs and pass^k. Every analysis rule that touches WP5 data is frozen in this document before any WP5 data exists.

## Non-Goals

Per the v3.1 plan section 1: no new eval questions, no metric redefinitions, no judge changes, no new providers, no dashboard, no service. The 528 existing tests pass untouched. Wild cluster bootstrap, canary injection (WP8), and judge unfolding (WP9) are v3.2 backlog (section 9).

## 1. Decisions closed by this design

| Item | Resolution | Confirmed |
|---|---|---|
| Session stance | Tighten and finalize plan v0.2; architecture unchanged | Jane Yeung, 2026-06-11 |
| TOST equivalence margin | 0.10 absolute on P@5 and R@5 | Jane Yeung, 2026-06-11 |
| Optional schema fields | `latency_ms`, `cost_usd`, `refused` adopted; `judge_id`, `judge_version`, `trajectory_len` defined but not adopted | Jane Yeung, 2026-06-11 |
| Raw epoch output storage | Committed to the repo | Jane Yeung, 2026-06-11 |
| WP7 scope | Agreement CIs attempted in v3.1; pass^k deferred first if time presses | Jane Yeung, 2026-06-11 |
| Long-table storage format | CSV | plan default, unchanged |
| Spec location | `docs/plans/` design + implementation pairs (repo convention) | Jane Yeung, 2026-06-11 |
| Few-cluster policy | Threshold rule (section 2.2) | Jane Yeung, 2026-06-11 |
| Rule-of-three unit | Question-level headline (section 2.3) | Jane Yeung, 2026-06-11 |
| TOST framing | Verdict plus equivalence CI (section 2.4) | Jane Yeung, 2026-06-11 |

Still open, not blocking v3.1: WP8 canary set size (v3.2; canaries are authored by Jane Yeung, never machine-invented).

## 2. Pre-registration

Frozen 2026-06-11. No WP5 epoch data existed when these rules were fixed; the WP5 campaign has not run. WP0's `docs/stats-design.md` restates this section alongside the discovered-format documentation, so the freeze travels with the engineering doc.

### 2.1 Cluster definitions

- FastAPI corpus: `cluster_id` = first-listed `expected_sources` entry. 4 of 27 questions are multi-source (q022 to q025) and take their first-listed file. The 5 out-of-scope questions have empty `expected_sources` in the dataset and take the literal `cluster_id` value `out_of_scope`. Result: 13 file clusters plus 1 out-of-scope cluster; clustered metrics span only the 13 file clusters, because retrieval and citation metrics are defined only for in-scope questions.
- K8s corpus: `cluster_id` = `question_type`. 6 types with sizes {simple 6, multi_hop 6, comparison 4, false_premise 4, simple_w_condition 4, set 1}. The singleton cluster is kept, never merged.
- `cluster_id` is populated on every row, with no validator exceptions. The analysis layer simply never uses it for metrics that are not clustered (refusal correctness, calculator correctness).
- Order stability: first-listed order is dataset content. Reordering `expected_sources` is by definition a `dataset_version` change, and every row carries `dataset_version`, so cluster assignments are reproducible per version.
- These definitions reflect the substantive dependence structure (shared source material for FastAPI; shared question construction for K8s) and are reconfirmed and frozen pre-data.

### 2.2 Primary-interval threshold rule (few clusters)

- Clustered SE is the primary interval for a corpus when n_clusters >= 10; otherwise the question-level (naive) interval is primary and the clustered SE is reported as a sensitivity check.
- Applied to current data: FastAPI (13 clusters on clustered metrics) is clustered-primary; K8s (6 clusters) is naive-primary.
- Justification: cluster bootstrap is unreliable at single-digit cluster counts (Cameron, Gelbach, Miller few-cluster literature). The threshold is fixed here, pre-data, so the primary-interval choice never depends on observed results.
- Divergence flag (deterministic): on a naive-primary corpus, if the clustered SE exceeds the naive SE by more than 1.5x on a headline metric, the report prints a correlation-sensitivity caution on that corpus's headline (section 7).
- Wild cluster bootstrap is v3.2 backlog (section 9).

### 2.3 Zero-failure bound on citation accuracy (rule of three)

- Trial = question. A trial succeeds only if zero hallucinated citations occurred across all k epochs and all citations in that question's answers.
- Inclusion rule: a question enters n if at least one epoch produced at least one citation. The report states the exclusion count and how many included questions cited in all epochs versus only some.
- Headline: exact Clopper-Pearson zero-failure 95 percent upper bound, with the 3/n approximation alongside. Granularity: per configuration per corpus, matching the README claim.
- Citation level: descriptive counts only (zero hallucinations across N citations in M answers), no bound.
- Methods appendix carries the bounding argument: collapsing epochs bounds the any-of-k-epochs failure rate, which also bounds the per-answer rate, so the collapse strengthens rather than wastes the data.
- Graceful degradation: the general Clopper-Pearson interval is always computed; rule-of-three phrasing appears only when observed failures equal zero (section 7).

### 2.4 TOST equivalence (custom vs LangChain)

- Margin: 0.10 absolute. Alpha: 0.05 per one-sided test (90 percent CI). Metrics carrying the equivalence claim: P@5 and R@5 only. Multiplicity: no adjustment across the two metrics; stated openly in the methods appendix.
- Paired unit: per-question score = mean over epochs; n = 22 in-scope FastAPI pairs. Out-of-scope questions never enter retrieval-metric comparisons.
- Symmetric outcome handling (section 7): pass reads "equivalent within plus or minus 0.10; the data support equivalence down to plus or minus X"; fail reads "equivalence not established at plus or minus 0.10; the data support only plus or minus X". X is the larger absolute endpoint of the 90 percent CI in both branches.

### 2.5 Paired-bootstrap resampling unit

- FastAPI (clustered-primary): the paired bootstrap resamples clusters of paired differences. K8s (naive-primary): resamples questions, with the correlation caveat.
- Budget escape hatch, recorded pre-data: if dual-mode resampling complicates `stats/paired.py` beyond the WP3 budget, downgrade to question-level resampling everywhere with a documented caveat. Whichever branch is taken is recorded in `docs/stats-design.md`.

## 3. Schema restatement and conventions

### 3.1 SCHEMA_VERSION and provenance

- The restatement in section 3.2 is a verbatim copy of section 5.1 text pasted into the design session by Jane Yeung on 2026-06-11. The paste attributed it to `IMPLEMENTATION_PLAN_eval-statistics-engine.md` v0.2, lines 71 to 88.
- Evidence status: the v0.2 attribution comes from the paste's own header line. The agent session had no access to the file, so the attribution is relayed, not independently verified.
- Verification protocol: Jane Yeung opens the local `IMPLEMENTATION_PLAN_eval-statistics-engine.md` (the artifact of record), reads its status header and changelog, and diffs its section 5.1 against section 3.2 below character for character. The version carried by the matching local file is the value pinned here and in `stats/schema.py`.
- Verification status: verified 2026-06-13 (Jane Yeung). The local `IMPLEMENTATION_PLAN_eval-statistics-engine.md` section 5.1 was diffed against section 3.2 above and is byte-identical: 906 bytes each, no non-ASCII characters, no non-breaking spaces, no trailing whitespace. The file is internally inconsistent on version: its status header reads v0.1, while its changelog's latest entry is v0.2 whose scope is section 8 only, so section 5.1 is unchanged from v0.1. `SCHEMA_VERSION` in `stats/schema.py` is pinned to `eval-statistics-engine/v0.2#5.1`, taking the changelog as the authoritative version record; the value is reached by changelog reading plus the clean diff, not relayed from the paste attribution. Correction to the evidence-status line above: the v0.2 designation comes from the changelog, not the status header (which reads v0.1). The stale header is corrected to v0.2 in Jane's local artifact, which is never committed to agent-bench.
- Drift policy: when the evidence-engine repository eventually owns the schema, agent-bench pins the version it implements via `SCHEMA_VERSION` and updates deliberately, never implicitly.

### 3.2 Verbatim restatement (engine plan section 5.1)

The block below is the restatement of record. It is a character-for-character copy of the pasted source text and must not be edited except through the verification protocol above.

```markdown
### 5.1 Results table (long format, one row per run-question-metric)

Core fields (required):

| Field | Type | Notes |
|---|---|---|
| `run_id` | string (ULID) | One execution of the eval harness |
| `timestamp` | ISO 8601 | |
| `config_id` | string | Provider, framework, prompt version, decoding params, as one versioned identifier |
| `code_version` | string | Git SHA of the system under test |
| `dataset_version` | string | Golden-set version |
| `question_id` | string | Stable across runs |
| `cluster_id` | string | Grouping unit for clustered SEs (source file, CRAG type) |
| `epoch` | int | 1..k for repeated trials of the same question |
| `metric` | string | e.g. `p_at_5`, `citation_acc`, `groundedness` |
| `score` | float | |

Optional extensions: `judge_id`, `judge_version`, `latency_ms`, `cost_usd`, `trajectory_len`, `refused` (bool). **[DECIDE]** exact optional set at Phase 0 review.
```

### 3.3 Optional-field adoption status (annotation alongside the verbatim text, not part of it)

All six optional extensions remain schema-defined in `stats/schema.py`; adoption status is marked alongside, never by omission. The [DECIDE] in the verbatim text is resolved by this table.

| Optional field | Status at v3.1 |
|---|---|
| `latency_ms` | Adopted. Mapped from `EvalResult.latency_ms` |
| `cost_usd` | Adopted. Mapped from `EvalResult.tokens_used.estimated_cost_usd`; provider-estimated, not billed spend |
| `refused` | Adopted. Observational flag; see section 3.4 |
| `judge_id` | Defined by the schema, not adopted at v3.1; adding later is non-breaking |
| `judge_version` | Defined by the schema, not adopted at v3.1; adding later is non-breaking |
| `trajectory_len` | Defined by the schema, not adopted at v3.1; adding later is non-breaking |

### 3.4 Conventions (carried in the `stats/schema.py` docstring)

- `score` is float. Boolean-valued metrics encode 0.0/1.0 (refusal correctness, calculator correctness), the same convention as `citation_acc`.
- `refused` is observational: it records that the system emitted a refusal, regardless of question scope. Refusal correctness is a downstream metric computed by joining `refused` against the golden set's scope label; it lives in `metric` rows, never in the column. In-scope false refusals therefore stay visible.
- `cost_usd` provenance: provider-estimated, not ground-truth spend.
- Missing `refused`: an empty CSV cell encodes missing; `validate_table()` accepts true, false, or empty for adopted optional boolean columns; values load as a nullable boolean and are never coerced. `refused` is derivable only from non-empty stored answer text; an empty or absent answer yields missing, not false.
- Metric rows exist only where the metric is defined: retrieval and citation metrics for in-scope questions; refusal correctness for out-of-scope questions; calculator correctness where `requires_calculator`. Vacuous values (`grounded_refusal` True on in-scope rows, `citation_accuracy` 1.0 on citation-free answers) never enter the table as data.

## 4. Adapter and table semantics (`stats_adapters/from_results_json.py`)

- `refused` is populated for every row by calling `is_refusal(answer)` (hook point 5.1) on the stored answer text. One derivation path serves legacy and new runs.
- Legacy rows (existing `results/*.json`):
  - `run_id` = `legacy-` followed by the first 12 hex digits of the source file's content hash. Not a ULID by construction.
  - `validate_table()` enforces ULID format exactly for rows whose `run_id` lacks the `legacy-` prefix. The prefix is the in-row marking mechanism and survives table concatenation.
  - Legacy tables are written under `results/long/legacy/`, separate from WP5 epoch tables; legacy rows never mix silently with epoch data.
  - `config_id` comes from a filename mapping table recorded in `docs/stats-design.md` at WP1; `code_version` and `dataset_version` are `unknown` unless recoverable from git history; `epoch` = 1; `timestamp` = git commit date of the source results file.
- `stats_adapters/` is the only code allowed to import agent-bench modules (boundary layer; guardrail 1).

## 5. Hook points (guardrail 5; named here and in the PR description)

1. `agent_bench/evaluation/metrics.py`: pure behavior-preserving extraction of the refusal-shape detection (currently inline in `grounded_refusal`, lines 78 to 98) into `is_refusal(answer) -> bool`; `grounded_refusal` delegates to it. Rationale: `grounded_refusal` returns a vacuous True for in-scope questions and cannot serve as the observational flag. The 528 existing tests are untouched and must pass; new tests for `is_refusal()` are added under the new test tree.
2. `Makefile`: `lint` extends to `stats/`, `stats_adapters/`, and `scripts/run_epochs.py` for ruff and mypy; new targets `stats-table`, `epochs` (documented PAID, HUMAN-RUN), and `evaluate-stats`.
3. `pyproject.toml`: scipy and pandas land in a new optional extra `stats`; the `dev` extra includes `stats` so CI and `make install` get them; main dependencies stay lean.
4. No other edits to existing modules. `stats/` and `stats_adapters/` remain unpackaged (setuptools `include` unchanged); they are in-repo tooling executed from the repo root, documented as such.

## 6. Epoch runner mechanics (WP2)

`scripts/run_epochs.py` drives both entry points, `scripts/evaluate.py` and `scripts/run_langchain_eval.py`, through their existing `--output` flags. Provenance fields (`run_id` ULID, `epoch`, `config_id`, `code_version` git SHA, `dataset_version`) are injected by post-processing the JSON each entry point writes to `--output`, or an adjacent sidecar file if post-processing proves cleaner; no harness-internal edits are planned. MockProvider supports the full path without API keys, and CI exercises a k=2 mock epoch run end to end.

## 7. Report generator requirements (WP4)

The three degradation branches are report logic with fixture-backed golden-file tests, never prose intentions:

1. Divergence flag: on a naive-primary corpus, a clustered SE exceeding the naive SE by more than 1.5x on a headline metric prints the correlation-sensitivity caution on that corpus's headline.
2. Zero-failure phrasing: the general Clopper-Pearson interval is always computed; rule-of-three phrasing appears only when observed failures equal zero. A nonzero failure degrades the wording without a hand edit.
3. TOST verdict: pass and fail wordings per section 2.4, with X the larger absolute endpoint of the 90 percent CI in both branches.

Always printed: n_clusters and the design effect (clustered variance over naive variance) for both corpora. The methods appendix carries estimators, seeds, replicate counts, the multiplicity stance, and the any-of-k bounding argument. The variance-decomposition section carries the one-line error-budget preview: the reported interval is the statistical term; template sensitivity and judge bias are named systematic terms, v3.2 scope.

Byte-stability: no wall-clock timestamps anywhere in the report body; the report embeds input-table content hashes and seeds instead. The golden-file test on the fixture table is byte-stable under pinned seeds. Every number in the report body is formatted explicitly (fixed precision, explicit float-to-string); the renderer never relies on dataframe repr or library default formatting, so golden files survive pandas and numpy version bumps (added 2026-06-11 after the dev environment resolved pandas 3.0.2). The fixture set includes a nonzero-failure variant, a failed-equivalence variant, and a divergent-SE variant, so all three degradation branches run in CI.

## 8. Import isolation and testing rules

- Import isolation is enforced two ways: a test installs a meta-path blocker that raises on any `agent_bench` import while importing every `stats` submodule, and a source scan asserts the string `agent_bench` appears nowhere under `stats/`.
- All randomness is seeded; tests pin seeds; a test that can flake is a bug (guardrail 3).
- Numerical tests assert against independently computed reference values (statsmodels or R), with provenance comments stating tool and version (guardrail 4).
- Edge cases required by the actual data: zero failures, all-ties pairs, single cluster (the K8s `set` type is a real singleton), missing `refused`.
- Format enforcement is two tier by decision (Jane Yeung, 2026-06-11): `ruff check` and mypy run repo-wide, while `ruff format --check` covers only the new packages (`stats/`, `stats_adapters/`, `tests/stats/`). The pre-existing files have never been format-clean under any ruff version (probed 0.6.9 through 0.15.10) and are never reformatted in passing; a repo-wide reformat is a standalone chore PR after v3.1 ships, decided then. CI runs `make lint` so CI and local enforcement cannot drift.

## 9. Backlog notes (v3.2, recorded for adjacency, not v3.1 scope)

- Wild cluster bootstrap: reference implementations exist (fwildclusterboot, boottest, wildboottest), but 6 clusters including a singleton limits p-value resolution under Rademacher weights. Revisit only with a changed cluster structure or more questions.
- WP8 canary injection and WP9 judge unfolding: per the v3.1 plan section 6, unchanged by this design.
- `judge_id`, `judge_version`, `trajectory_len` adoption: when judge metrics enter the long table.

## 10. Documents this design produces or touches

- This file: the decision record for v3.1.
- `docs/stats-design.md` (WP0): discovered result-format documentation, the schema mapping, the filename-to-config_id table, and a restatement of section 2 so the pre-registration travels with the engineering doc.
- `docs/plans/2026-06-11-stats-layer-v3.1-implementation.md` (next step): the executable work-package plan, derived from the v3.1 Claude Code plan v0.2 plus this design.
- `README.md` (WP6) and `docs/_generated/stats_report.md` (WP4): per the plan's exact edit list, using the section 2.4 outcome wordings.

---

*Changelog*

- v1.0 (2026-06-11): initial version, consolidating the brainstorm session decisions and amendments 1 to 7.
