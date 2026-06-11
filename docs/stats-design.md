# agent-bench v3.1 statistics layer: engineering design notes

Companion to `docs/plans/2026-06-11-stats-layer-v3.1-design.md` (the decision record) and `docs/plans/2026-06-11-stats-layer-v3.1-implementation.md` (the work-package plan). This document records what WP0 discovered by reading the code, the mapping from harness output to the long-format table, a restatement of the pre-registration so the freeze travels with the engineering doc, and the operational notes later WPs need.

## Discovered result formats

`scripts/evaluate.py` writes a JSON list of `EvalResult` objects (`results_data = [r.model_dump() for r in results]`) to `--output`, default `.cache/eval_results.json`. Its CLI also takes `--config` (YAML path), `--corpus` (a key of `config.corpora`, e.g. `fastapi` or `k8s`; omitted means legacy `rag.store_path` plus `evaluation.golden_dataset`), and `--mode` (`deterministic` or `full`).

`EvalResult` is defined at `agent_bench/evaluation/harness.py:71` with fields: `question_id`, `question`, `category`, `difficulty`, `retrieval_precision`, `retrieval_recall`, `keyword_hit_rate`, `has_source_citation`, `grounded_refusal`, `citation_accuracy`, `calculator_used_correctly`, `tool_calls_made`, `latency_ms`, `tokens_used` (`TokenUsage` at `agent_bench/core/types.py:36`: `input_tokens`, `output_tokens`, `estimated_cost_usd`), `answer`, `retrieved_sources`, `judge_scores` (empty dict when no judge provider is configured).

The LangChain baseline is written by `scripts/run_langchain_eval.py` with flags `--provider` (`openai` or `anthropic`), `--config`, `--output` (default `.cache/langchain_eval_results.json`), and `--max-questions`.

Provenance of the three identity fields the adapter needs:

- `question_id` comes from `EvalResult.question_id`, which carries the golden question's `id`.
- The source file for FastAPI cluster assignment comes from the golden question's `expected_sources` (first-listed entry; `GoldenQuestion` at `agent_bench/evaluation/harness.py`).
- The CRAG question type for K8s cluster assignment comes from the golden question's `question_type` field (multi-corpus schema v2; K8s only, empty string on the FastAPI corpus).

`agent_bench/evaluation/harness.py:load_golden_dataset` accepts both golden formats: the legacy flat list (FastAPI, `tech_docs_golden.json`) and the nested header form `{"corpus": ..., "version": ..., "questions": [...]}` (K8s, `k8s_golden.json`).

Vacuous-value facts (why two `EvalResult` fields cannot be ingested as data unconditionally):

- `grounded_refusal` (`agent_bench/evaluation/metrics.py:60`) returns True for every row whose category is not `out_of_scope` ("not applicable"). It is meaningful only on out-of-scope questions.
- `citation_accuracy` (`agent_bench/evaluation/metrics.py:103`) returns 1.0 when the answer contains no `[source: ...]` citations ("no citations to check"). It is meaningful only on answers that actually cite.

## Mapping to the long format

Pinned metric names for the `metric` column: `p_at_5`, `r_at_5`, `khr`, `citation_acc`, `refusal_correct`, `calculator_correct`.

Metric emission rules (design spec section 3.4): metric rows exist only where the metric is defined; vacuous values never enter the table as data.

| Condition (from the golden question) | Metric rows emitted | Source `EvalResult` field |
|---|---|---|
| In-scope (category is not `out_of_scope`) | `p_at_5` | `retrieval_precision` |
| In-scope | `r_at_5` | `retrieval_recall` |
| In-scope | `khr` | `keyword_hit_rate` |
| In-scope and the answer contains at least one `[source: ...]` citation | `citation_acc` | `citation_accuracy` |
| `category == "out_of_scope"` | `refusal_correct` (0.0 or 1.0) | `grounded_refusal` |
| `requires_calculator` | `calculator_correct` (0.0 or 1.0) | `calculator_used_correctly` |

Adopted optional columns, populated on every emitted row: `latency_ms` from `EvalResult.latency_ms`; `cost_usd` from `tokens_used.estimated_cost_usd` (provider-estimated, not billed); `refused` from `is_refusal(answer)` on non-empty stored answer text, missing when the answer is empty or absent.

Cluster assignment: FastAPI rows take the first-listed `expected_sources` entry, out-of-scope rows take the literal `out_of_scope`, K8s rows take `question_type` (pre-registration 2.1 below).

## Pre-registration restatement

The following restates design spec section 2 verbatim so the freeze travels with this document.

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

## Filename-to-config_id table

| Results file | `config_id` |
|---|---|

Populated at WP1 when the adapter lands; legacy inputs require explicit `--config-id`, so this table is documentation of the choices made, not a runtime input.

## Blockers

Two findings this session, both resolved by decision (Jane Yeung, 2026-06-11):

1. `make lint` had never been green on main. The Makefile's `lint` target included `ruff format --check agent_bench/ tests/`, which fails on 53 pre-existing files (a roughly 2,100 line mechanical diff), and the failure is independent of ruff version: probes of 0.6.9, 0.8.6, 0.9.10, 0.11.13, 0.12.12, 0.14.13, and 0.15.10 all flag the same files. The gap went unnoticed because CI enforced a subset of the Makefile target (`ruff check` plus `mypy agent_bench/` only, no format check); that Makefile-versus-CI enforcement mismatch predates v3.1. Resolution: format enforcement is two tier. `ruff check` and mypy run repo-wide; `ruff format --check` covers only the new packages (`stats/`, `stats_adapters/`, `tests/stats/`), which are format-clean from birth. CI now runs `make lint` so CI and local enforcement cannot drift. The pre-existing files are never reformatted in passing; a repo-wide reformat is explicitly deferred to a standalone chore PR after v3.1 ships, decided then, and is not sanctioned now. See design spec section 8.
2. The `stats` extra resolved pandas 3.0.2 under the original `pandas>=2.2.0` pin, so dev and CI could silently straddle a major version. Resolution: the pin is tightened to `pandas>=3.0,<4`, and the design spec (section 7) now requires `stats/report.py` to format every number explicitly (fixed precision, explicit float-to-string, no reliance on dataframe repr or library default formatting) so WP4 golden files survive pandas and numpy version bumps.

## How to run the harness offline

MockProvider selection: the provider factory (`agent_bench/core/provider.py`) dispatches on `config.provider.default`; the value `mock` returns `MockProvider()`, which needs no API key. Setting `provider.default: mock` in a config YAML passed via `--config` runs the full evaluation path keyless. MockProvider is deterministic (fixed `provider="mock"`, `model="mock-1"` responses).

Exact `configs/default.yaml` corpora shape (read this session), for WP2's mini-corpus work:

```yaml
default_corpus: fastapi

corpora:
  fastapi:
    label: "FastAPI Docs"
    store_path: .cache/store
    data_path: data/tech_docs
    refusal_threshold: 0.02
    top_k: 5
    max_iterations: 3
    golden_dataset: agent_bench/evaluation/datasets/tech_docs_golden.json
  k8s:
    label: "Kubernetes"
    store_path: .cache/store_k8s
    data_path: data/k8s_docs
    refusal_threshold: 0.015
    top_k: 5
    max_iterations: 3
    golden_dataset: agent_bench/evaluation/datasets/k8s_golden.json
    available: true
```

`default_corpus` must be a key of `corpora` (enforced by an AppConfig validator), and when `corpora` is non-empty the legacy `rag.refusal_threshold` is ignored in favor of the per-corpus values. A WP2 mini-corpus config therefore needs: `provider.default: mock`, a `corpora` entry whose `data_path` points at the fixture docs, whose `store_path` points at a throwaway cache path, and whose `golden_dataset` points at the mini golden JSON.
