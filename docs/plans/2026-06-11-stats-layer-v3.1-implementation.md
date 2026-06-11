# agent-bench v3.1 Statistics Layer: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn agent-bench's point-estimate benchmark tables into measurements: a validated long-format results table, an epoch runner, a pure statistics package (intervals, clustered SEs, paired tests, TOST, power, variance decomposition), a one-command report, README integration, and (droppable) judge-agreement CIs and pass^k.

**Architecture:** Two new top-level packages: `stats/` (pure: stdlib + numpy + scipy + pandas only, never imports agent-bench) and `stats_adapters/` (the only boundary allowed to import agent-bench). A thin epoch wrapper at `scripts/run_epochs.py` drives the two existing entry points through their `--output` flags. Long tables are CSV under `results/long/`; the report regenerates with one command into `docs/_generated/stats_report.md`.

**Tech Stack:** Python 3.11, numpy, scipy, pandas (new optional extra `stats`, included by `dev`), pytest, ruff, mypy. statsmodels is used only to generate independent reference vectors, from a throwaway venv outside the project, never as a dependency.

**Authority:** This plan operationalizes the v3.1 plan (Claude Code edition, v0.2) plus the approved design spec at `docs/plans/2026-06-11-stats-layer-v3.1-design.md` (commit `41389b5`). Where this plan and the spec disagree, the spec wins; report the conflict instead of improvising.

**Author:** Jane Yeung

---

## Session protocol (applies to every work package)

1. One work package (WP) per Claude Code session. Do not start a WP before the previous WP's exit gate passed and its PR merged.
2. Branch per WP: `feat/stats-wp<N>-<slug>`. PR per WP into `main`. Squash merge per repo habit.
3. Hard rules from the v3.1 plan section 2 apply to every task: never run paid targets (`evaluate-full`, `benchmark`, `evaluate-langchain`, `calibrate`, `evaluate-judges`) or anything needing real API keys; all randomness seeded; never modify existing tests or existing eval logic outside the hook points named in this plan; ruff and mypy clean before a WP closes; no em or en dashes in any doc or report text.
4. End every session by answering explicitly: "what previously agreed scope does this WP's output violate?" If the answer is not "none", stop and report.
5. Verification before completion: run the exit-gate commands and paste their output into the PR description. Claims without output are not verification.

**Input gate (WP0, blocking):** `SCHEMA_VERSION` in `stats/schema.py` is written from the engine-plan version Jane verified against her local artifact (spec section 3.1). If that verified value has not been provided in the session, STOP at Task 0.4 step 3 and ask for it. Do not write a value from the paste attribution.

---

## File structure

### New code files

| File | Responsibility |
|---|---|
| `stats/__init__.py` | Package marker; exports `SCHEMA_VERSION` |
| `stats/schema.py` | Schema constants, `SchemaError`, `validate_table()` |
| `stats/intervals.py` | `wilson`, `clopper_pearson`, `zero_failure_upper`, `rule_of_three` |
| `stats/cluster.py` | `cluster_bootstrap`, threshold and divergence constants, `design_effect` |
| `stats/paired.py` | `paired_bootstrap`, `mcnemar_exact` |
| `stats/equivalence.py` | `tost_paired`, `TostResult` |
| `stats/power.py` | `simulated_power`, `mde`, `mde_normal_approx` |
| `stats/variance.py` | `decompose`, `VarianceDecomposition` |
| `stats/report.py` | `render_report` plus argparse main; all degradation branches |
| `stats/agreement.py` | WP7: `cohen_kappa`, `gwet_ac1`, `bootstrap_agreement_ci` |
| `stats/reliability.py` | WP7: `pass_k_table` |
| `stats_adapters/__init__.py` | Package marker |
| `stats_adapters/from_results_json.py` | results JSON to validated long CSV; legacy provenance; argparse main |
| `scripts/run_epochs.py` | Epoch wrapper: ULID, envelope injection, config registry, subprocess driver |
| `scripts/check_readme_stats.py` | WP6: README versus generated report consistency check |

### New test files (all under `tests/stats/`, fixture-driven, keyless)

| File | Covers |
|---|---|
| `tests/stats/test_schema.py` | `validate_table` accept and reject paths |
| `tests/stats/test_is_refusal.py` | the extracted `is_refusal` hook |
| `tests/stats/test_adapter.py` | row emission rules, legacy provenance, round-trip |
| `tests/stats/test_run_epochs.py` | ULID, envelope, registry; mock end-to-end |
| `tests/stats/test_intervals.py` | reference vectors |
| `tests/stats/test_cluster.py` | reference vectors, seeds, edge cases |
| `tests/stats/test_paired.py` | reference vectors, ties, seeds |
| `tests/stats/test_equivalence.py` | TOST verdicts, support margin, cross-check |
| `tests/stats/test_power.py` | simulation versus normal approximation |
| `tests/stats/test_variance.py` | hand-worked decomposition |
| `tests/stats/test_isolation.py` | meta-path blocker plus source scan |
| `tests/stats/test_report.py` | golden files, all degradation branches |
| `tests/stats/test_agreement.py` | WP7 kappa and AC1 vectors |
| `tests/stats/test_reliability.py` | WP7 pass^k |
| `tests/stats/fixtures/` | mini goldens, mini results, mini corpus docs, long-table CSVs, golden reports |

### Modified files (sanctioned hook points only, spec section 5)

| File | Edit |
|---|---|
| `agent_bench/evaluation/metrics.py` | extract `is_refusal(answer)`; `grounded_refusal` delegates (WP1) |
| `Makefile` | lint paths; `stats-table`, `epochs`, `evaluate-stats` targets (WP0, WP1, WP2, WP4) |
| `pyproject.toml` | `stats` optional extra; `dev` includes it (WP0) |
| `README.md` | WP6 exact edit list only |
| `CLAUDE.md` | created once in WP0 from the v3.1 plan section 3 seed |

Pinned metric names for the `metric` column: `p_at_5`, `r_at_5`, `khr`, `citation_acc`, `refusal_correct`, `calculator_correct`.
Pinned constants: `PRIMARY_THRESHOLD = 10` and `DIVERGENCE_RATIO = 1.5` (cluster.py), `DEFAULT_MARGIN = 0.10` and `DEFAULT_ALPHA = 0.05` (equivalence.py), `DEFAULT_SEED = 20260611`, `DEFAULT_N_BOOT = 10_000` (each consuming module).

---

## WP0: Schema freeze and plumbing (1 session)

Branch: `feat/stats-wp0-schema`

### Task 0.1: CLAUDE.md seed and branch

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Create the branch**

```bash
git checkout -b feat/stats-wp0-schema
```

- [ ] **Step 2: Create CLAUDE.md with the v3.1 seed**

The repo has no CLAUDE.md. Create it with exactly this content (from the v3.1 plan section 3):

```markdown
## v3.1 statistics layer rules
- stats/ is a pure package: stdlib + numpy + scipy + pandas only. Never import agent-bench modules inside stats/. Adapters live in stats_adapters/ and are the only boundary.
- Never run make targets that cost money (evaluate-full, calibrate, evaluate-judges, evaluate-langchain) or anything needing real API keys. Use fixtures and MockProvider.
- All randomness is seeded and tests pin seeds.
- Numerical tests assert against independently computed reference values with provenance comments (statsmodels or R, tool and version stated).
- Do not modify existing tests or existing eval logic; add alongside.
- ruff and mypy clean before finishing. No em or en dashes in docs. Reports regenerate with one command.
- End every session by answering: what previously agreed scope does this change violate?
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "stats(wp0): add v3.1 statistics layer CLAUDE.md seed"
```

### Task 0.2: pyproject `stats` extra and package skeletons

**Files:**
- Modify: `pyproject.toml` (optional-dependencies block, lines 27 to 41)
- Create: `stats/__init__.py`, `stats_adapters/__init__.py`, `tests/stats/__init__.py`, `tests/stats/fixtures/.gitkeep`

- [ ] **Step 1: Add the `stats` extra and include it in `dev`**

In `pyproject.toml`, change the `[project.optional-dependencies]` block to:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
    "respx>=0.21.0",
    "types-PyYAML",
    "agent-bench[stats]",
]
stats = [
    "scipy>=1.11.0",
    "pandas>=2.2.0",
]
```

- [ ] **Step 2: Create the package skeletons**

`stats/__init__.py`:

```python
"""Pure statistics package for the v3.1 layer.

Depends only on the standard library, numpy, scipy, and pandas.
Never imports agent-bench modules (guardrail 1); tests/stats/test_isolation.py
enforces this with a meta-path blocker and a source scan.
"""

from stats.schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
```

`stats_adapters/__init__.py`:

```python
"""Boundary layer between agent-bench result formats and the stats package.

This is the only package allowed to import agent_bench (guardrail 1).
"""
```

`tests/stats/__init__.py`: empty file. `tests/stats/fixtures/.gitkeep`: empty file.

Note: `stats/__init__.py` imports from `stats.schema`, which Task 0.4 creates. Order within this WP: finish Task 0.4 before running the import; the intermediate state is fine because nothing imports `stats` yet.

- [ ] **Step 3: Reinstall and verify the extra resolves**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pip install -e ".[dev]"
/usr/local/opt/python@3.11/bin/python3.11 -c "import scipy, pandas; print(scipy.__version__, pandas.__version__)"
```

Expected: both versions print without error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml stats/ stats_adapters/ tests/stats/
git commit -m "stats(wp0): add stats extra and package skeletons"
```

### Task 0.3: Makefile lint hook

**Files:**
- Modify: `Makefile` (the `lint` target, lines 11 to 14)

- [ ] **Step 1: Extend lint to the new packages**

Replace the `lint` target with (recipe lines are tab-indented):

```makefile
lint:
	ruff check agent_bench/ stats/ stats_adapters/ tests/
	ruff format --check agent_bench/ stats/ stats_adapters/ tests/
	mypy agent_bench/ stats/ stats_adapters/ --ignore-missing-imports
```

`scripts/run_epochs.py` is added to these lines in WP2 when the file exists (ruff errors on missing paths).

- [ ] **Step 2: Run and verify clean**

```bash
make lint
```

Expected: exits 0. If `ruff format --check` flags the new files, run `ruff format stats/ stats_adapters/ tests/stats/` and re-check.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "stats(wp0): extend lint to stats packages"
```

### Task 0.4: `stats/schema.py` constants

**Files:**
- Create: `stats/schema.py`
- Test: `tests/stats/test_schema.py`

- [ ] **Step 1: Write the failing test for constants**

```python
"""Tests for stats.schema. Fixture-driven, no API keys, no agent-bench imports."""

import re

import pandas as pd
import pytest

from stats import schema


def test_required_fields_match_engine_plan_5_1():
    assert schema.REQUIRED_FIELDS == (
        "run_id",
        "timestamp",
        "config_id",
        "code_version",
        "dataset_version",
        "question_id",
        "cluster_id",
        "epoch",
        "metric",
        "score",
    )


def test_optional_fields_all_six_defined_three_adopted():
    assert schema.OPTIONAL_FIELDS == (
        "judge_id",
        "judge_version",
        "latency_ms",
        "cost_usd",
        "trajectory_len",
        "refused",
    )
    assert schema.ADOPTED_OPTIONAL_FIELDS == ("latency_ms", "cost_usd", "refused")
    assert set(schema.ADOPTED_OPTIONAL_FIELDS) <= set(schema.OPTIONAL_FIELDS)


def test_schema_version_is_pinned():
    assert re.fullmatch(r"eval-statistics-engine/v\d+\.\d+#5\.1", schema.SCHEMA_VERSION)
```

- [ ] **Step 2: Run to verify failure**

Run: `/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError` (schema module not yet written).

- [ ] **Step 3: INPUT GATE, then write the module header and constants**

STOP if Jane has not yet supplied the verified engine-plan version from her local diff (spec section 3.1 verification protocol). Ask: "What version does your local engine plan carry for the section 5.1 text? I need it for SCHEMA_VERSION." Substitute the verified value below (shown here as v0.2, the paste attribution, which must be confirmed or corrected).

```python
"""Results-table schema for the v3.1 statistics layer.

Restated from the eval-statistics-engine plan section 5.1, verbatim source
recorded in docs/plans/2026-06-11-stats-layer-v3.1-design.md section 3.2.
Self-contained: this module never imports agent-bench (guardrail 1).

Conventions (design spec section 3.4):
- score is float. Boolean-valued metrics encode 0.0/1.0 (refusal correctness,
  calculator correctness), the same convention as citation_acc.
- refused is observational: it records that the system emitted a refusal,
  regardless of question scope. Refusal correctness is a downstream metric
  computed by joining refused against the golden set's scope label; it lives
  in metric rows, never in this column.
- cost_usd is provider-estimated (tokens_used.estimated_cost_usd), not billed.
- Missing refused: empty CSV cell; loads as nullable boolean; never coerced.
- Legacy rows: run_id carries the legacy- prefix and is exempt from the ULID
  format check; everything else validates identically.
"""

import re

import pandas as pd

SCHEMA_VERSION = "eval-statistics-engine/v0.2#5.1"

REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "config_id",
    "code_version",
    "dataset_version",
    "question_id",
    "cluster_id",
    "epoch",
    "metric",
    "score",
)

OPTIONAL_FIELDS: tuple[str, ...] = (
    "judge_id",
    "judge_version",
    "latency_ms",
    "cost_usd",
    "trajectory_len",
    "refused",
)

ADOPTED_OPTIONAL_FIELDS: tuple[str, ...] = ("latency_ms", "cost_usd", "refused")

ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
LEGACY_RUN_ID_RE = re.compile(r"^legacy-[0-9a-f]{12}$")
ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


class SchemaError(ValueError):
    """Raised by validate_table with every violation listed, not just the first."""
```

- [ ] **Step 4: Run constants tests, verify pass**

Run: `/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_schema.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add stats/schema.py tests/stats/test_schema.py
git commit -m "stats(wp0): schema constants with verified SCHEMA_VERSION"
```

### Task 0.5: `validate_table()`

**Files:**
- Modify: `stats/schema.py`
- Test: `tests/stats/test_schema.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/stats/test_schema.py`:

```python
def _valid_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["01HZXJ5M8N9PQRSTVWXYZ01234", "legacy-0a1b2c3d4e5f"],
            "timestamp": ["2026-06-15T10:00:00+00:00", "2026-05-06T12:00:00+00:00"],
            "config_id": ["custom-openai+ab12cd34", "custom-openai-legacy"],
            "code_version": ["41389b5", "unknown"],
            "dataset_version": ["sha-deadbeef", "unknown"],
            "question_id": ["q001", "q001"],
            "cluster_id": ["fastapi_path_params.md", "fastapi_path_params.md"],
            "epoch": [1, 1],
            "metric": ["p_at_5", "p_at_5"],
            "score": [0.4, 0.4],
            "latency_ms": [10965.5, 9871.2],
            "cost_usd": [0.000296, 0.000301],
            "refused": pd.array([False, pd.NA], dtype="boolean"),
        }
    )


def test_valid_table_passes():
    schema.validate_table(_valid_table())  # must not raise


def test_missing_required_column_rejected():
    df = _valid_table().drop(columns=["cluster_id"])
    with pytest.raises(schema.SchemaError, match="missing required column: cluster_id"):
        schema.validate_table(df)


def test_unknown_column_rejected():
    df = _valid_table().assign(surprise=1)
    with pytest.raises(schema.SchemaError, match="unknown column: surprise"):
        schema.validate_table(df)


def test_bad_run_id_rejected():
    df = _valid_table()
    df.loc[0, "run_id"] = "not-a-ulid"
    with pytest.raises(schema.SchemaError, match="run_id"):
        schema.validate_table(df)


def test_legacy_run_id_accepted():
    df = _valid_table()
    df["run_id"] = ["legacy-0a1b2c3d4e5f", "legacy-9f8e7d6c5b4a"]
    schema.validate_table(df)  # must not raise


def test_empty_cluster_id_rejected():
    df = _valid_table()
    df.loc[0, "cluster_id"] = ""
    with pytest.raises(schema.SchemaError, match="cluster_id"):
        schema.validate_table(df)


def test_null_score_rejected():
    df = _valid_table()
    df.loc[0, "score"] = None
    with pytest.raises(schema.SchemaError, match="score"):
        schema.validate_table(df)


def test_epoch_below_one_rejected():
    df = _valid_table()
    df.loc[0, "epoch"] = 0
    with pytest.raises(schema.SchemaError, match="epoch"):
        schema.validate_table(df)


def test_bad_timestamp_rejected():
    df = _valid_table()
    df.loc[0, "timestamp"] = "yesterday"
    with pytest.raises(schema.SchemaError, match="timestamp"):
        schema.validate_table(df)


def test_violations_aggregate():
    df = _valid_table().drop(columns=["metric"]).assign(surprise=1)
    with pytest.raises(schema.SchemaError) as exc:
        schema.validate_table(df)
    msg = str(exc.value)
    assert "missing required column: metric" in msg
    assert "unknown column: surprise" in msg
```

- [ ] **Step 2: Run to verify failure**

Run: `/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_schema.py -v`
Expected: new tests FAIL with `AttributeError: ... no attribute 'validate_table'`.

- [ ] **Step 3: Implement `validate_table`**

Append to `stats/schema.py`:

```python
def validate_table(df: pd.DataFrame) -> None:
    """Validate a long-format results table. Collects all violations.

    Raises SchemaError with one line per violation. Never mutates df.
    """
    errors: list[str] = []

    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            errors.append(f"missing required column: {col}")
    allowed = set(REQUIRED_FIELDS) | set(ADOPTED_OPTIONAL_FIELDS)
    for col in df.columns:
        if col not in allowed:
            errors.append(f"unknown column: {col}")

    if not errors:
        bad_run = df.loc[
            ~df["run_id"].astype(str).str.match(ULID_RE)
            & ~df["run_id"].astype(str).str.match(LEGACY_RUN_ID_RE)
        ]
        if len(bad_run):
            errors.append(f"run_id not ULID or legacy-prefixed on rows {list(bad_run.index)[:5]}")

        bad_ts = df.loc[~df["timestamp"].astype(str).str.match(ISO_TS_RE)]
        if len(bad_ts):
            errors.append(f"timestamp not ISO 8601 on rows {list(bad_ts.index)[:5]}")

        for col in ("config_id", "code_version", "dataset_version", "question_id", "cluster_id", "metric"):
            empty = df.loc[df[col].astype(str).str.len() == 0]
            if len(empty):
                errors.append(f"{col} empty on rows {list(empty.index)[:5]}")

        epochs = pd.to_numeric(df["epoch"], errors="coerce")
        if epochs.isna().any() or (epochs < 1).any() or (epochs != epochs.astype("Int64")).any():
            errors.append("epoch must be an integer >= 1")

        scores = pd.to_numeric(df["score"], errors="coerce")
        if scores.isna().any():
            errors.append(f"score null or non-numeric on rows {list(df.index[scores.isna()])[:5]}")

        if "refused" in df.columns:
            try:
                df["refused"].astype("boolean")
            except (TypeError, ValueError):
                errors.append("refused must be true, false, or missing")

    if errors:
        raise SchemaError("schema violations:\n" + "\n".join(f"  - {e}" for e in errors))
```

- [ ] **Step 4: Run, verify all pass, lint**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_schema.py -v && make lint
```

Expected: all PASS, lint exits 0.

- [ ] **Step 5: Commit**

```bash
git add stats/schema.py tests/stats/test_schema.py
git commit -m "stats(wp0): validate_table with aggregated violations"
```

### Task 0.6: `docs/stats-design.md`

**Files:**
- Create: `docs/stats-design.md`

- [ ] **Step 1: Write the document**

Sections, in order. Content for (a) through (c) is already established; write it out, do not leave headers empty. No em or en dashes.

(a) **Discovered result formats.** `EvalResult` JSON list written by `scripts/evaluate.py` (default `.cache/eval_results.json`, `--output` supported) with fields: `question_id`, `question`, `category`, `difficulty`, `retrieval_precision`, `retrieval_recall`, `keyword_hit_rate`, `has_source_citation`, `grounded_refusal`, `citation_accuracy`, `calculator_used_correctly`, `tool_calls_made`, `latency_ms`, `tokens_used{input_tokens, output_tokens, estimated_cost_usd}`, `answer`, `retrieved_sources`, `judge_scores`. LangChain baseline written by `scripts/run_langchain_eval.py` (`--provider`, `--config`, `--output`, `--max-questions`). Record where `question_id` (`EvalResult.question_id`), source file (`golden.expected_sources`), and CRAG type (`golden.question_type`, K8s only) come from. Record the vacuous-value facts: `grounded_refusal` is True for all in-scope rows; `citation_accuracy` is 1.0 for citation-free answers.
(b) **Mapping to the long format.** The metric emission table from spec section 3.4 plus the pinned metric names from this plan's file-structure section.
(c) **Pre-registration restatement.** Copy spec section 2 (2.1 through 2.5) verbatim so the freeze travels with this doc.
(d) **Filename-to-config_id table.** Header plus the sentence "Populated at WP1 when the adapter lands; legacy inputs require explicit `--config-id`, so this table is documentation of the choices made, not a runtime input."
(e) **Blockers.** State "none" or list what was found this session.
(f) **How to run the harness offline.** Record the MockProvider selection mechanism (`provider.default: mock` in a config YAML) and the exact `configs/default.yaml` corpora shape discovered by reading it this session, for WP2's mini-corpus work.

- [ ] **Step 2: Dash and placeholder scan**

```bash
python3 - <<'EOF'
import re
text = open('docs/stats-design.md', encoding='utf-8').read()
assert not re.search("[\\u2012\\u2013\\u2014\\u2015\\u2212]", text), "banned dash found"
assert not re.search(r'\b(TBD|TODO|XXX|FIXME)\b', text), "placeholder found"
print("clean")
EOF
```

Expected: `clean`.

- [ ] **Step 3: Commit, run exit gate, open PR**

```bash
git add docs/stats-design.md
git commit -m "stats(wp0): stats-design doc with pre-registration restatement"
make test && make lint
```

Expected: 528 existing tests plus the new `tests/stats/test_schema.py` tests all pass; lint clean.

Open the PR titled `stats(wp0): schema freeze and plumbing`. PR description: name the two sanctioned file edits (pyproject, Makefile), paste the test and lint output, answer the scope question.

**WP0 exit gate:** `make test` and `make lint` green; `docs/stats-design.md` reviewed by Jane before WP1 starts; SCHEMA_VERSION carries the verified value.

---

## WP1: Results-table adapter (1 session)

Branch: `feat/stats-wp1-adapter`

### Task 1.1: `is_refusal` extraction (sanctioned hook, spec section 5 item 1)

**Files:**
- Modify: `agent_bench/evaluation/metrics.py:60-100`
- Test: `tests/stats/test_is_refusal.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the is_refusal hook extracted from grounded_refusal.

The extraction is behavior-preserving: the 528 pre-existing tests must pass
untouched. These tests cover the new public function only.
"""

from agent_bench.evaluation.metrics import grounded_refusal, is_refusal


def test_phrase_refusal_detected():
    assert is_refusal("The documentation does not contain information about this.")


def test_canonical_refusal_detected():
    assert is_refusal("That topic is not in the FastAPI documentation provided.")


def test_plain_answer_not_refusal():
    assert not is_refusal("Path parameters are declared with curly braces.")


def test_not_in_the_without_documentation_anchor_not_refusal():
    assert not is_refusal("The value is not in the default range for this field.")


def test_empty_answer_not_refusal():
    assert not is_refusal("")


def test_grounded_refusal_still_vacuous_true_for_in_scope():
    assert grounded_refusal("any answer at all", "retrieval") is True


def test_grounded_refusal_delegates_for_out_of_scope():
    refusing = "No relevant information was found."
    assert grounded_refusal(refusing, "out_of_scope") is True
    assert grounded_refusal(refusing + " [source: a.md]", "out_of_scope") is False
```

- [ ] **Step 2: Run to verify failure**

Run: `/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_is_refusal.py -v`
Expected: FAIL with `ImportError: cannot import name 'is_refusal'`.

- [ ] **Step 3: Extract the function**

In `agent_bench/evaluation/metrics.py`, add `is_refusal` directly above `grounded_refusal`, moving lines 78 to 98 (the phrase list, the canonical regex, and the combination) into it unchanged:

```python
def is_refusal(answer: str) -> bool:
    """Does the answer text take the refusal action? Observational only.

    Extracted unchanged from grounded_refusal so the stats adapter can record
    refusal as an action flag on every question regardless of scope (v3.1
    design spec section 3.4). Correctness of a refusal is a separate, scope-
    conditioned judgment and does not belong here.
    """
    refusal_phrases = [
        "does not contain",
        "no information",
        "not contain",
        "not available",
        "not found",
        "cannot find",
        "no relevant",
        "outside the scope",
    ]
    answer_lower = answer.lower()
    has_phrase_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
    # Canonical shape taught by the system prompt at core/prompts.py:17-18:
    # "not in the {corpus_label} documentation". Narrow regex anchors on
    # "documentation" within 60 chars so plain "not in the" fragments from
    # retrieval answers ("not in the same scope", "not in the default range")
    # do not count as refusals.
    has_canonical_refusal = bool(
        re.search(r"\bnot in the\b[^.]{0,60}\bdocumentation\b", answer, re.IGNORECASE)
    )
    return has_phrase_refusal or has_canonical_refusal
```

Then replace the body of `grounded_refusal` between the `category` guard and the citation check so it reads:

```python
    if category != "out_of_scope":
        return True  # not applicable
    has_refusal = is_refusal(answer)
    cites_in_answer = re.findall(r"\[source:\s*[^\]]+\]", answer, re.IGNORECASE)
    return has_refusal and len(cites_in_answer) == 0
```

The docstring of `grounded_refusal` stays untouched.

- [ ] **Step 4: Run new tests AND the full suite**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_is_refusal.py -v
make test
```

Expected: new tests PASS; the full suite passes with the pre-existing tests unmodified (`git diff --stat tests/` shows only `tests/stats/` additions).

- [ ] **Step 5: Commit**

```bash
git add agent_bench/evaluation/metrics.py tests/stats/test_is_refusal.py
git commit -m "stats(wp1): extract is_refusal hook from grounded_refusal (behavior-preserving)"
```

### Task 1.2: Mini goldens and mini results fixtures

**Files:**
- Create: `tests/stats/fixtures/golden_mini_fastapi.json`
- Create: `tests/stats/fixtures/golden_mini_k8s.json`
- Create: `tests/stats/fixtures/results_mini.json`

- [ ] **Step 1: Write `golden_mini_fastapi.json`** (flat array, real golden field shape)

```json
[
  {
    "id": "mini_q1",
    "question": "How do you paginate results?",
    "expected_answer_keywords": ["limit", "offset"],
    "expected_sources": ["mini_pagination.md", "mini_auth.md"],
    "category": "retrieval",
    "difficulty": "easy",
    "requires_calculator": false,
    "reference_answer": "Use limit and offset query parameters."
  },
  {
    "id": "mini_q2",
    "question": "What is the capital of France?",
    "expected_answer_keywords": [],
    "expected_sources": [],
    "category": "out_of_scope",
    "difficulty": "easy",
    "requires_calculator": false,
    "reference_answer": ""
  },
  {
    "id": "mini_q3",
    "question": "If each page holds 25 items, how many pages for 120 items?",
    "expected_answer_keywords": ["5"],
    "expected_sources": ["mini_pagination.md"],
    "category": "calculation",
    "difficulty": "easy",
    "requires_calculator": true,
    "reference_answer": "5 pages."
  }
]
```

- [ ] **Step 2: Write `golden_mini_k8s.json`** (nested v2 shape with `question_type`)

```json
{
  "corpus": "k8s",
  "version": "v1.31",
  "snapshot_date": "2026-06-11",
  "questions": [
    {
      "id": "mini_k1",
      "question": "What does a liveness probe do?",
      "expected_answer_keywords": ["restart"],
      "expected_sources": ["mini_probes.md"],
      "category": "retrieval",
      "difficulty": "easy",
      "requires_calculator": false,
      "reference_answer": "It restarts unhealthy containers.",
      "question_type": "simple",
      "is_multi_hop": false,
      "time_sensitive": false
    },
    {
      "id": "mini_k2",
      "question": "Compare liveness and readiness probes.",
      "expected_answer_keywords": ["traffic", "restart"],
      "expected_sources": ["mini_probes.md"],
      "category": "retrieval",
      "difficulty": "medium",
      "requires_calculator": false,
      "reference_answer": "Liveness restarts; readiness gates traffic.",
      "question_type": "comparison",
      "is_multi_hop": false,
      "time_sensitive": false
    }
  ]
}
```

- [ ] **Step 3: Write `results_mini.json`** (anonymized, exact `EvalResult` shape, answers exercising every adapter rule: in-scope with citation, out-of-scope refusal, calculator)

```json
[
  {
    "question_id": "mini_q1",
    "question": "How do you paginate results?",
    "category": "retrieval",
    "difficulty": "easy",
    "retrieval_precision": 0.4,
    "retrieval_recall": 1.0,
    "keyword_hit_rate": 1.0,
    "has_source_citation": true,
    "grounded_refusal": true,
    "citation_accuracy": 1.0,
    "calculator_used_correctly": true,
    "tool_calls_made": 1,
    "latency_ms": 1200.5,
    "tokens_used": {"input_tokens": 900, "output_tokens": 120, "estimated_cost_usd": 0.0002},
    "answer": "Use limit and offset query parameters [source: mini_pagination.md].",
    "retrieved_sources": ["mini_pagination.md", "mini_auth.md"],
    "judge_scores": {}
  },
  {
    "question_id": "mini_q2",
    "question": "What is the capital of France?",
    "category": "out_of_scope",
    "difficulty": "easy",
    "retrieval_precision": 0.0,
    "retrieval_recall": 0.0,
    "keyword_hit_rate": 0.0,
    "has_source_citation": false,
    "grounded_refusal": true,
    "citation_accuracy": 1.0,
    "calculator_used_correctly": true,
    "tool_calls_made": 1,
    "latency_ms": 800.0,
    "tokens_used": {"input_tokens": 700, "output_tokens": 40, "estimated_cost_usd": 0.0001},
    "answer": "The provided documentation does not contain information about this topic.",
    "retrieved_sources": [],
    "judge_scores": {}
  },
  {
    "question_id": "mini_q3",
    "question": "If each page holds 25 items, how many pages for 120 items?",
    "category": "calculation",
    "difficulty": "easy",
    "retrieval_precision": 0.2,
    "retrieval_recall": 1.0,
    "keyword_hit_rate": 1.0,
    "has_source_citation": true,
    "grounded_refusal": true,
    "citation_accuracy": 1.0,
    "calculator_used_correctly": true,
    "tool_calls_made": 2,
    "latency_ms": 1500.0,
    "tokens_used": {"input_tokens": 1100, "output_tokens": 90, "estimated_cost_usd": 0.00025},
    "answer": "120 items at 25 per page needs 5 pages [source: mini_pagination.md].",
    "retrieved_sources": ["mini_pagination.md"],
    "judge_scores": {}
  }
]
```

- [ ] **Step 4: Commit**

```bash
git add tests/stats/fixtures/
git commit -m "stats(wp1): mini golden and results fixtures"
```

### Task 1.3: Adapter core `rows_from_result`

**Files:**
- Create: `stats_adapters/from_results_json.py`
- Test: `tests/stats/test_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Adapter tests: emission rules, cluster assignment, refused derivation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats import schema
from stats_adapters import from_results_json as adapter

FIXTURES = Path(__file__).parent / "fixtures"


def _golden_fastapi() -> adapter.GoldenIndex:
    return adapter.load_golden(FIXTURES / "golden_mini_fastapi.json")


def _results() -> list[dict]:
    return json.loads((FIXTURES / "results_mini.json").read_text())


def _meta() -> adapter.RowMeta:
    return adapter.RowMeta(
        run_id="01HZXJ5M8N9PQRSTVWXYZ01234",
        timestamp="2026-06-15T10:00:00+00:00",
        config_id="custom-mock+00000000",
        code_version="41389b5",
        dataset_version="sha-deadbeef",
        epoch=1,
    )


def test_in_scope_emits_retrieval_metrics_only():
    rows = adapter.rows_from_result(_results()[0], _meta(), _golden_fastapi())
    metrics = {r["metric"] for r in rows}
    assert metrics == {"p_at_5", "r_at_5", "khr", "citation_acc"}


def test_out_of_scope_emits_refusal_correct_only():
    rows = adapter.rows_from_result(_results()[1], _meta(), _golden_fastapi())
    metrics = {r["metric"] for r in rows}
    assert metrics == {"refusal_correct"}
    assert all(r["score"] in (0.0, 1.0) for r in rows)


def test_calculation_emits_calculator_correct_plus_retrieval():
    rows = adapter.rows_from_result(_results()[2], _meta(), _golden_fastapi())
    metrics = {r["metric"] for r in rows}
    assert metrics == {"p_at_5", "r_at_5", "khr", "citation_acc", "calculator_correct"}


def test_citation_free_in_scope_answer_omits_citation_acc():
    rec = dict(_results()[0], answer="Use limit and offset query parameters.")
    rows = adapter.rows_from_result(rec, _meta(), _golden_fastapi())
    assert {r["metric"] for r in rows} == {"p_at_5", "r_at_5", "khr"}


def test_cluster_id_first_listed_source_for_in_scope():
    rows = adapter.rows_from_result(_results()[0], _meta(), _golden_fastapi())
    assert {r["cluster_id"] for r in rows} == {"mini_pagination.md"}


def test_cluster_id_literal_for_out_of_scope():
    rows = adapter.rows_from_result(_results()[1], _meta(), _golden_fastapi())
    assert {r["cluster_id"] for r in rows} == {"out_of_scope"}


def test_cluster_id_question_type_for_k8s():
    golden = adapter.load_golden(FIXTURES / "golden_mini_k8s.json")
    rec = dict(_results()[0], question_id="mini_k1")
    rows = adapter.rows_from_result(rec, _meta(), golden)
    assert {r["cluster_id"] for r in rows} == {"simple"}


def test_refused_observational_on_every_row():
    refusing = adapter.rows_from_result(_results()[1], _meta(), _golden_fastapi())
    answering = adapter.rows_from_result(_results()[0], _meta(), _golden_fastapi())
    assert all(r["refused"] is True for r in refusing)
    assert all(r["refused"] is False for r in answering)


def test_refused_missing_when_answer_empty():
    rec = dict(_results()[0], answer="")
    rows = adapter.rows_from_result(rec, _meta(), _golden_fastapi())
    assert all(r["refused"] is None for r in rows)


def test_latency_and_cost_mapped():
    rows = adapter.rows_from_result(_results()[0], _meta(), _golden_fastapi())
    assert all(r["latency_ms"] == 1200.5 for r in rows)
    assert all(r["cost_usd"] == 0.0002 for r in rows)
```

- [ ] **Step 2: Run to verify failure**

Run: `/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_adapter.py -v`
Expected: FAIL with `ImportError` (module functions not yet written).

- [ ] **Step 3: Implement the core**

`stats_adapters/from_results_json.py`:

```python
"""Convert agent-bench results JSON into validated long-format CSV tables.

Boundary layer: the only package allowed to import agent_bench (guardrail 1).
Emission rules, cluster assignment, and legacy provenance per
docs/plans/2026-06-11-stats-layer-v3.1-design.md sections 2.1, 3.4, and 4.
"""

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agent_bench.evaluation.metrics import is_refusal
from stats import schema

# Mirrors the pattern inside agent_bench.evaluation.metrics.citation_accuracy.
# Used only to decide whether an answer exercised the citation metric at all:
# citation_acc 1.0 on a citation-free answer is vacuous and never enters the
# table as data (design spec section 3.4).
CITATION_RE = re.compile(r"\[source:\s*(.+?)\]")


@dataclass(frozen=True)
class RowMeta:
    run_id: str
    timestamp: str
    config_id: str
    code_version: str
    dataset_version: str
    epoch: int


@dataclass(frozen=True)
class GoldenQuestion:
    category: str
    requires_calculator: bool
    cluster_id: str


@dataclass(frozen=True)
class GoldenIndex:
    questions: dict[str, GoldenQuestion]


def load_golden(path: Path) -> GoldenIndex:
    raw = json.loads(path.read_text())
    is_nested = isinstance(raw, dict)
    questions = raw["questions"] if is_nested else raw
    index: dict[str, GoldenQuestion] = {}
    for q in questions:
        if is_nested:
            cluster = q["question_type"]
        elif q["expected_sources"]:
            cluster = q["expected_sources"][0]
        else:
            cluster = "out_of_scope"
        index[q["id"]] = GoldenQuestion(
            category=q["category"],
            requires_calculator=q["requires_calculator"],
            cluster_id=cluster,
        )
    return GoldenIndex(questions=index)


def _metric_values(rec: dict, golden_q: GoldenQuestion) -> dict[str, float]:
    if golden_q.category == "out_of_scope":
        return {"refusal_correct": 1.0 if rec["grounded_refusal"] else 0.0}
    values = {
        "p_at_5": float(rec["retrieval_precision"]),
        "r_at_5": float(rec["retrieval_recall"]),
        "khr": float(rec["keyword_hit_rate"]),
    }
    # citation_acc only when the answer actually cited something: vacuous 1.0
    # on citation-free answers never enters the table (spec section 3.4). The
    # rule-of-three inclusion rule (spec section 2.3) then falls out of the
    # table itself: a question enters n if any epoch emitted a citation_acc row.
    if CITATION_RE.search(rec.get("answer", "")):
        values["citation_acc"] = float(rec["citation_accuracy"])
    if golden_q.requires_calculator:
        values["calculator_correct"] = 1.0 if rec["calculator_used_correctly"] else 0.0
    return values


def rows_from_result(rec: dict, meta: RowMeta, golden: GoldenIndex) -> list[dict]:
    golden_q = golden.questions[rec["question_id"]]
    answer = rec.get("answer", "")
    refused: bool | None = is_refusal(answer) if answer else None
    rows = []
    for metric, score in _metric_values(rec, golden_q).items():
        rows.append(
            {
                "run_id": meta.run_id,
                "timestamp": meta.timestamp,
                "config_id": meta.config_id,
                "code_version": meta.code_version,
                "dataset_version": meta.dataset_version,
                "question_id": rec["question_id"],
                "cluster_id": golden_q.cluster_id,
                "epoch": meta.epoch,
                "metric": metric,
                "score": score,
                "latency_ms": float(rec["latency_ms"]),
                "cost_usd": float(rec["tokens_used"]["estimated_cost_usd"]),
                "refused": refused,
            }
        )
    return rows
```

- [ ] **Step 4: Run, verify pass**

Run: `/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_adapter.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add stats_adapters/from_results_json.py tests/stats/test_adapter.py
git commit -m "stats(wp1): adapter row emission with cluster and refused rules"
```

### Task 1.4: Legacy provenance, file conversion, CLI, `make stats-table`

**Files:**
- Modify: `stats_adapters/from_results_json.py`
- Modify: `Makefile`
- Test: `tests/stats/test_adapter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/stats/test_adapter.py`:

```python
def test_legacy_run_id_and_provenance(tmp_path):
    src = FIXTURES / "results_mini.json"
    df = adapter.convert_legacy_file(
        src,
        golden_path=FIXTURES / "golden_mini_fastapi.json",
        config_id="custom-openai-legacy",
        out_dir=tmp_path,
    )
    schema.validate_table(df)
    assert df["run_id"].str.match(schema.LEGACY_RUN_ID_RE).all()
    assert (df["epoch"] == 1).all()
    assert (df["code_version"] == "unknown").all()
    expected_hash = adapter.content_hash(src)
    assert (df["run_id"] == f"legacy-{expected_hash}").all()


def test_legacy_csv_written_under_legacy_dir(tmp_path):
    adapter.convert_legacy_file(
        FIXTURES / "results_mini.json",
        golden_path=FIXTURES / "golden_mini_fastapi.json",
        config_id="custom-openai-legacy",
        out_dir=tmp_path,
    )
    out = tmp_path / "legacy" / "results_mini.csv"
    assert out.exists()
    schema.validate_table(pd.read_csv(out, dtype={"refused": "boolean"}))


def test_legacy_round_trip_deterministic(tmp_path):
    kwargs = dict(
        golden_path=FIXTURES / "golden_mini_fastapi.json",
        config_id="custom-openai-legacy",
    )
    a = adapter.convert_legacy_file(FIXTURES / "results_mini.json", out_dir=tmp_path / "a", **kwargs)
    b = adapter.convert_legacy_file(FIXTURES / "results_mini.json", out_dir=tmp_path / "b", **kwargs)
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run to verify failure**

Run: `/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_adapter.py -v -k legacy`
Expected: FAIL with `AttributeError` on the new functions.

- [ ] **Step 3: Implement legacy conversion and CLI**

Append to `stats_adapters/from_results_json.py`:

```python
def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _git_commit_date(path: Path) -> str:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    iso = proc.stdout.strip()
    # Untracked files (tmp fixtures in tests) have no commit date; epoch zero
    # keeps the row valid and visibly artificial.
    return iso if iso else "1970-01-01T00:00:00+00:00"


def convert_legacy_file(
    src: Path,
    golden_path: Path,
    config_id: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Convert one pre-v3.1 results file. Spec section 4 provenance synthesis."""
    golden = load_golden(golden_path)
    meta = RowMeta(
        run_id=f"legacy-{content_hash(src)}",
        timestamp=_git_commit_date(src),
        config_id=config_id,
        code_version="unknown",
        dataset_version="unknown",
        epoch=1,
    )
    records = json.loads(src.read_text())
    rows = [row for rec in records for row in rows_from_result(rec, meta, golden)]
    df = pd.DataFrame(rows)
    df["refused"] = df["refused"].astype("boolean")
    schema.validate_table(df)
    legacy_dir = out_dir / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(legacy_dir / f"{src.stem}.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="results JSON path, or envelope dir in WP2")
    parser.add_argument("--golden", required=True, help="golden dataset JSON path")
    parser.add_argument("--config-id", default=None, help="explicit config_id; required with --legacy")
    parser.add_argument("--out-dir", default="results/long")
    parser.add_argument("--legacy", action="store_true", help="pre-v3.1 file: synthesize provenance")
    args = parser.parse_args()
    if not args.legacy:
        raise SystemExit("epoch-envelope inputs arrive in WP2; use --legacy for existing files")
    if args.config_id is None:
        parser.error("--config-id is required with --legacy; no guessing (spec section 4)")
    df = convert_legacy_file(
        Path(args.input), Path(args.golden), args.config_id, Path(args.out_dir)
    )
    print(f"wrote {len(df)} rows for {args.input}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the Makefile target** (tab-indented recipe; demonstrates on a real results file with an explicit config id)

```makefile
stats-table:  ## Convert legacy results JSON to validated long CSV (free, offline)
	$(PYTHON) -m stats_adapters.from_results_json --legacy \
		--input results/fastapi_postedit.json \
		--golden agent_bench/evaluation/datasets/tech_docs_golden.json \
		--config-id custom-openai-legacy \
		--out-dir results/long
```

Record `fastapi_postedit.json -> custom-openai-legacy` in the `docs/stats-design.md` filename-to-config_id table in the same commit. If the provider attribution of a results file is unclear from its name and git history, ask Jane; never guess.

- [ ] **Step 5: Run everything, verify, commit, open PR**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/ -v
make stats-table && make test && make lint
git add stats_adapters/from_results_json.py tests/stats/test_adapter.py Makefile docs/stats-design.md
git commit -m "stats(wp1): legacy conversion, CLI, and stats-table target"
```

Expected: tests pass; `make stats-table` prints a row count and writes `results/long/legacy/fastapi_postedit.csv`; suite and lint green.

**WP1 exit gate:** fixture round-trips to a validated table; `make test` and `make lint` green; PR description names the `metrics.py` hook edit; scope question answered.

---

## WP2: Epoch runner (1 session)

Branch: `feat/stats-wp2-epochs`

### Task 2.1: ULID helper and envelope writer

**Files:**
- Create: `scripts/run_epochs.py`
- Test: `tests/stats/test_run_epochs.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Epoch runner tests. Everything here runs keyless via MockProvider."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

import run_epochs  # noqa: E402

ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def test_ulid_shape_and_uniqueness():
    ids = {run_epochs.new_ulid() for _ in range(200)}
    assert len(ids) == 200
    assert all(ULID_RE.fullmatch(u) for u in ids)


def test_envelope_wraps_results(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([{"question_id": "mini_q1"}]))
    out = run_epochs.write_envelope(
        raw_output=raw,
        dest_dir=tmp_path / "epochs",
        run_id="01HZXJ5M8N9PQRSTVWXYZ01234",
        config_id="custom-mock+00000000",
        epoch=2,
        code_version="41389b5",
        dataset_version="sha-deadbeef",
        timestamp="2026-06-15T10:00:00+00:00",
    )
    env = json.loads(out.read_text())
    assert env["epoch"] == 2
    assert env["run_id"] == "01HZXJ5M8N9PQRSTVWXYZ01234"
    assert env["results"][0]["question_id"] == "mini_q1"
```

- [ ] **Step 2: Run to verify failure**

Run: `/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_run_epochs.py -v`
Expected: FAIL with `ModuleNotFoundError: run_epochs`.

- [ ] **Step 3: Implement the helpers**

`scripts/run_epochs.py` (first half):

```python
"""Repeat harness runs k times per configuration with injected provenance.

PAID when run against API configs; the Makefile target requires CONFIRM_PAID=1.
The mock config path is free and exercised in CI. Provenance is injected by
post-processing each entry point's --output JSON into an envelope file
(design spec section 6); harness internals are never edited.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    ts = int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp() * 1000)
    time_part = ""
    for _ in range(10):
        time_part = CROCKFORD[ts % 32] + time_part
        ts //= 32
    rand = int.from_bytes(os.urandom(10), "big")
    rand_part = ""
    for _ in range(16):
        rand_part = CROCKFORD[rand % 32] + rand_part
        rand //= 32
    return time_part + rand_part


def write_envelope(
    raw_output: Path,
    dest_dir: Path,
    run_id: str,
    config_id: str,
    epoch: int,
    code_version: str,
    dataset_version: str,
    timestamp: str,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config_id": config_id,
        "code_version": code_version,
        "dataset_version": dataset_version,
        "epoch": epoch,
        "results": json.loads(raw_output.read_text()),
    }
    out = dest_dir / f"{config_id.split('+')[0]}_e{epoch}.json"
    out.write_text(json.dumps(envelope, indent=1))
    return out
```

- [ ] **Step 4: Run, verify pass, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_run_epochs.py -v
git add scripts/run_epochs.py tests/stats/test_run_epochs.py
git commit -m "stats(wp2): ulid and envelope helpers"
```

### Task 2.2: Config registry and subprocess driver

**Files:**
- Modify: `scripts/run_epochs.py`
- Modify: `Makefile` (lint paths from Task 0.3 gain `scripts/run_epochs.py`; new `epochs` target)

- [ ] **Step 1: Verify config file names before pinning the registry**

```bash
ls configs/*.yaml
```

Adjust the `config` paths in the registry below to the actual file names for the OpenAI and Anthropic custom configs (expected `configs/default.yaml` and `configs/anthropic.yaml`; if they differ, use what `ls` shows and say so in the PR description).

- [ ] **Step 2: Implement the registry and driver**

Append to `scripts/run_epochs.py`:

```python
def _config_hash(path: Path | None) -> str:
    if path is None:
        return "00000000"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def _dataset_version(golden_path: Path) -> str:
    return "sha-" + hashlib.sha256(golden_path.read_bytes()).hexdigest()[:8]


def _code_version() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return sha + ("-dirty" if dirty else "")


# name -> (entry, config yaml or None, provider flag or None, corpus, golden path)
REGISTRY: dict[str, dict] = {
    "custom-openai": {
        "entry": "custom",
        "config": Path("configs/default.yaml"),
        "corpus": "fastapi",
        "golden": Path("agent_bench/evaluation/datasets/tech_docs_golden.json"),
    },
    "custom-anthropic": {
        "entry": "custom",
        "config": Path("configs/anthropic.yaml"),
        "corpus": "fastapi",
        "golden": Path("agent_bench/evaluation/datasets/tech_docs_golden.json"),
    },
    "langchain-openai": {
        "entry": "langchain",
        "provider": "openai",
        "golden": Path("agent_bench/evaluation/datasets/tech_docs_golden.json"),
    },
    "langchain-anthropic": {
        "entry": "langchain",
        "provider": "anthropic",
        "golden": Path("agent_bench/evaluation/datasets/tech_docs_golden.json"),
    },
}


def _entry_cmd(spec: dict, raw_out: Path, mock_config: Path | None) -> list[str]:
    config = mock_config or spec.get("config")
    if spec["entry"] == "custom":
        cmd = [sys.executable, "scripts/evaluate.py", "--mode", "deterministic", "--output", str(raw_out)]
        if config:
            cmd += ["--config", str(config)]
        if spec.get("corpus"):
            cmd += ["--corpus", spec["corpus"]]
        return cmd
    cmd = [sys.executable, "scripts/run_langchain_eval.py", "--provider", spec["provider"], "--output", str(raw_out)]
    if config:
        cmd += ["--config", str(config)]
    return cmd


def run_config_epochs(
    name: str, k: int, dest_root: Path, mock_config: Path | None = None, golden_override: Path | None = None
) -> list[Path]:
    spec = REGISTRY[name]
    golden = golden_override or spec["golden"]
    config_id = f"{name}+{_config_hash(mock_config or spec.get('config'))}"
    run_id = new_ulid()
    written = []
    for epoch in range(1, k + 1):
        raw_out = dest_root / "raw" / f"{name}_e{epoch}.json"
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(_entry_cmd(spec, raw_out, mock_config), check=True)
        written.append(
            write_envelope(
                raw_output=raw_out,
                dest_dir=dest_root / run_id,
                run_id=run_id,
                config_id=config_id,
                epoch=epoch,
                code_version=_code_version(),
                dataset_version=_dataset_version(golden),
                timestamp=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            )
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--configs", required=True, help="comma-separated registry names")
    parser.add_argument("--dest", default="results/epochs")
    parser.add_argument("--mock-config", default=None, help="config YAML forcing provider mock (free)")
    parser.add_argument("--golden", default=None, help="override golden path (tests only)")
    args = parser.parse_args()
    for name in args.configs.split(","):
        files = run_config_epochs(
            name,
            args.k,
            Path(args.dest),
            mock_config=Path(args.mock_config) if args.mock_config else None,
            golden_override=Path(args.golden) if args.golden else None,
        )
        print(f"{name}: wrote {len(files)} epoch envelopes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Makefile: lint paths and the PAID target** (tab-indented recipes)

Extend the three `lint` lines from Task 0.3 with `scripts/run_epochs.py`, and add:

```makefile
epochs:  ## PAID, HUMAN-RUN: repeat eval k times per config. Usage: make epochs K=5 CONFIGS=custom-openai,custom-anthropic CONFIRM_PAID=1
	@test "$(CONFIRM_PAID)" = "1" || (echo "Refusing: paid target. Set CONFIRM_PAID=1 to run. Costs real API money." && exit 1)
	$(PYTHON) scripts/run_epochs.py --k $(K) --configs $(CONFIGS)
```

- [ ] **Step 4: Lint and commit**

```bash
make lint
git add scripts/run_epochs.py Makefile
git commit -m "stats(wp2): config registry, subprocess driver, guarded epochs target"
```

### Task 2.3: Mock end-to-end in CI (mini corpus, k=2, adapter envelope mode)

**Files:**
- Create: `tests/stats/fixtures/mini_corpus/mini_pagination.md`, `tests/stats/fixtures/mini_corpus/mini_auth.md`
- Create: `tests/stats/fixtures/mock_config.yaml`
- Modify: `stats_adapters/from_results_json.py` (envelope input mode)
- Test: `tests/stats/test_run_epochs.py`, `tests/stats/test_adapter.py`

- [ ] **Step 1: Write the mini corpus docs**

`mini_pagination.md`:

```markdown
# Pagination

Endpoints accept limit and offset query parameters. The limit caps the page
size; the offset skips rows. The default limit is 25 items per page.
```

`mini_auth.md`:

```markdown
# Authentication

Requests carry a bearer token in the Authorization header. Tokens expire
after 60 minutes and are refreshed at the token endpoint.
```

- [ ] **Step 2: Build the mock config**

Read `configs/default.yaml` and copy its structure, overriding three things: `provider.default: mock`, the store path to `.cache/store_mini` (built in step 3), and a corpora entry named `mini` whose golden path is `tests/stats/fixtures/golden_mini_fastapi.json`. The exact key names follow the corpora shape recorded in `docs/stats-design.md` section (f) during WP0; mirror them exactly. Save as `tests/stats/fixtures/mock_config.yaml`. Verify it loads:

```bash
/usr/local/opt/python@3.11/bin/python3.11 scripts/evaluate.py --config tests/stats/fixtures/mock_config.yaml --corpus mini --output /tmp/mini_check.json
python3 -c "import json; d=json.load(open('/tmp/mini_check.json')); print(len(d), d[0]['question_id'])"
```

Expected: `3 mini_q1`. If `evaluate.py` requires the store to exist first, run the ingest from step 3 before this check and note the ordering in the PR.

- [ ] **Step 3: Add a session-scoped ingest fixture and the e2e test**

Append to `tests/stats/test_run_epochs.py`:

```python
import subprocess

import pandas as pd
import pytest

from stats import schema
from stats_adapters import from_results_json as adapter

FIXTURES = Path(__file__).parent / "fixtures"
PYTHON = sys.executable


@pytest.fixture(scope="session")
def mini_store():
    subprocess.run(
        [
            PYTHON,
            "scripts/ingest.py",
            "--doc-dir",
            str(FIXTURES / "mini_corpus"),
            "--store-path",
            ".cache/store_mini",
        ],
        check=True,
    )


def test_mock_epoch_run_end_to_end(tmp_path, mini_store):
    run_epochs.REGISTRY["custom-mock"] = {
        "entry": "custom",
        "config": FIXTURES / "mock_config.yaml",
        "corpus": "mini",
        "golden": FIXTURES / "golden_mini_fastapi.json",
    }
    files = run_epochs.run_config_epochs(
        "custom-mock", k=2, dest_root=tmp_path, mock_config=FIXTURES / "mock_config.yaml"
    )
    assert len(files) == 2
    df = adapter.convert_envelopes(
        [f for f in files], golden_path=FIXTURES / "golden_mini_fastapi.json", out_dir=tmp_path / "long"
    )
    schema.validate_table(df)
    assert set(df["epoch"]) == {1, 2}
    assert df["run_id"].nunique() == 1
    # Structural assertions only: whether MockProvider's canned answer contains
    # [source: ...] citations decides citation_acc emission, so no hard row count.
    per_q = df.groupby("question_id")["metric"].agg(set)
    assert {"p_at_5", "r_at_5", "khr"} <= per_q["mini_q1"]
    assert per_q["mini_q2"] == {"refusal_correct"}
    assert "calculator_correct" in per_q["mini_q3"]
    assert len(df) == 2 * df[df["epoch"] == 1].shape[0]  # epochs emit identical shapes
```

- [ ] **Step 4: Implement `convert_envelopes`**

Append to `stats_adapters/from_results_json.py`:

```python
def convert_envelopes(paths: list[Path], golden_path: Path, out_dir: Path) -> pd.DataFrame:
    """Convert WP2 epoch envelopes into one validated long table (CSV per run_id)."""
    golden = load_golden(golden_path)
    frames = []
    for path in paths:
        env = json.loads(path.read_text())
        meta = RowMeta(
            run_id=env["run_id"],
            timestamp=env["timestamp"],
            config_id=env["config_id"],
            code_version=env["code_version"],
            dataset_version=env["dataset_version"],
            epoch=env["epoch"],
        )
        frames.append(
            pd.DataFrame(
                [row for rec in env["results"] for row in rows_from_result(rec, meta, golden)]
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["refused"] = df["refused"].astype("boolean")
    schema.validate_table(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{df['run_id'].iloc[0]}.csv", index=False)
    return df
```

Also extend `main()`: replace the `--legacy` SystemExit branch with envelope support (`--input` may be a directory of envelopes; without `--legacy`, glob `*.json` inside it and call `convert_envelopes`).

- [ ] **Step 5: Run the full suite and lint; verify the e2e runs keyless**

```bash
env -i HOME="$HOME" PATH="$PATH" /usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/ -v
make test && make lint
```

Expected: all pass with an empty environment (proves no key dependence).

- [ ] **Step 6: Commit and open PR**

```bash
git add tests/stats/ stats_adapters/from_results_json.py scripts/run_epochs.py
git commit -m "stats(wp2): mock epoch e2e and envelope adapter mode"
```

**WP2 exit gate:** mock k=2 epoch run produces a valid long table end to end in CI; tests and lint green; PR names the Makefile edit; scope question answered.

---

## WP3: Statistics core (1 to 2 sessions)

Branch: `feat/stats-wp3-core`

Reference-vector protocol for every numerical test in this WP (guardrail 4): generate values with statsmodels in a throwaway venv, never add statsmodels as a dependency, and paste the printed values into the test with a provenance comment naming tool and version. Setup once per session:

```bash
python3 -m venv /tmp/refvec && /tmp/refvec/bin/pip -q install "statsmodels==0.14.*" numpy
```

### Task 3.1: `stats/intervals.py`

**Files:**
- Create: `stats/intervals.py`
- Test: `tests/stats/test_intervals.py`

- [ ] **Step 1: Generate reference values**

```bash
/tmp/refvec/bin/python - <<'EOF'
import statsmodels
from statsmodels.stats.proportion import proportion_confint
print("statsmodels", statsmodels.__version__)
print("wilson 17/22:", proportion_confint(17, 22, alpha=0.05, method="wilson"))
print("cp 17/22:", proportion_confint(17, 22, alpha=0.05, method="beta"))
print("cp 0/22 upper:", proportion_confint(0, 22, alpha=0.05, method="beta"))
print("cp 0/110 upper:", proportion_confint(0, 110, alpha=0.05, method="beta"))
EOF
```

- [ ] **Step 2: Write the failing tests**

`tests/stats/test_intervals.py`, inserting the printed values where marked PASTE, with the version in the comment:

```python
"""Reference vectors generated with statsmodels <PASTE version>,
proportion_confint, commands recorded in the implementation plan Task 3.1.
Do not regenerate silently; provenance comments must name tool and version."""

import pytest

from stats import intervals

TOL = 1e-9


def test_wilson_against_statsmodels():
    lo, hi = intervals.wilson(17, 22, confidence=0.95)
    assert lo == pytest.approx(PASTE_WILSON_LO, abs=1e-6)
    assert hi == pytest.approx(PASTE_WILSON_HI, abs=1e-6)


def test_clopper_pearson_against_statsmodels():
    lo, hi = intervals.clopper_pearson(17, 22, confidence=0.95)
    assert lo == pytest.approx(PASTE_CP_LO, abs=1e-6)
    assert hi == pytest.approx(PASTE_CP_HI, abs=1e-6)


def test_zero_failure_upper_matches_cp_at_zero():
    assert intervals.zero_failure_upper(22) == pytest.approx(PASTE_CP0_22_HI, abs=1e-6)
    assert intervals.zero_failure_upper(110) == pytest.approx(PASTE_CP0_110_HI, abs=1e-6)


def test_rule_of_three_is_3_over_n():
    assert intervals.rule_of_three(22) == pytest.approx(3 / 22)
    assert intervals.rule_of_three(110) == pytest.approx(3 / 110)


def test_zero_failure_closed_form():
    # Exact CP upper bound at k=0 is 1 - alpha**(1/n); independent identity check.
    assert intervals.zero_failure_upper(27) == pytest.approx(1 - 0.05 ** (1 / 27), abs=1e-12)


def test_degenerate_inputs_rejected():
    with pytest.raises(ValueError):
        intervals.wilson(5, 0)
    with pytest.raises(ValueError):
        intervals.wilson(6, 5)
```

Replace each `PASTE_*` name with the literal float printed in step 1 (e.g. `0.5635...`); the names must not survive into the committed test.

- [ ] **Step 3: Run to verify failure**

Run: `/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_intervals.py -v`
Expected: FAIL with `ModuleNotFoundError: stats.intervals`.

- [ ] **Step 4: Implement**

```python
"""Proportion intervals: Wilson, Clopper-Pearson, zero-failure bounds.

Pure module: stdlib + scipy only (guardrail 1).
"""

import math

from scipy.stats import beta, norm


def _check(successes: int, n: int) -> None:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside [0, {n}]")


def wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    _check(successes, n)
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    _check(successes, n)
    alpha = 1 - confidence
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lo, hi


def zero_failure_upper(n: int, confidence: float = 0.95) -> float:
    """Exact Clopper-Pearson one-sided upper bound at zero observed failures."""
    _check(0, n)
    return 1 - (1 - confidence) ** (1 / n)


def rule_of_three(n: int) -> float:
    _check(0, n)
    return 3 / n
```

Note: `proportion_confint(0, n, method="beta")` reports the two-sided interval; if the pasted upper differs from `1 - alpha**(1/n)` because of the two-sided alpha split, assert against `1 - (alpha/2)**(1/n)` for that statsmodels comparison and keep `zero_failure_upper` one-sided as specified, documenting the difference in the test comment. The design spec headline uses the one-sided zero-failure bound.

- [ ] **Step 5: Run, verify pass, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_intervals.py -v && make lint
git add stats/intervals.py tests/stats/test_intervals.py
git commit -m "stats(wp3): intervals with statsmodels reference vectors"
```

### Task 3.2: `stats/cluster.py`

**Files:**
- Create: `stats/cluster.py`
- Test: `tests/stats/test_cluster.py`

- [ ] **Step 1: Generate the independent reference** (cluster-robust SE of the mean via intercept-only OLS)

```bash
/tmp/refvec/bin/python - <<'EOF'
import numpy as np, statsmodels, statsmodels.api as sm
print("statsmodels", statsmodels.__version__)
rng = np.random.default_rng(7)
clusters = np.repeat(np.arange(8), 5)
y = rng.normal(0.7, 0.1, 40) + np.repeat(rng.normal(0, 0.15, 8), 5)
res = sm.OLS(y, np.ones_like(y)).fit(cov_type="cluster", cov_kwds={"groups": clusters})
print("mean:", float(res.params[0]))
print("crse:", float(res.bse[0]))
print("y:", y.tolist())
EOF
```

- [ ] **Step 2: Write the failing tests**

```python
"""Cluster bootstrap SE tests.

Reference: cluster-robust SE from statsmodels <PASTE version> intercept-only
OLS (cov_type=cluster), data generated with numpy seed 7 as recorded in the
implementation plan Task 3.2. Bootstrap and analytic CRSE are different
estimators; tolerance is 25 percent relative, which a correct implementation
meets easily on this fixture while transposed-cluster bugs fail it.
"""

import numpy as np
import pytest

from stats import cluster

Y = np.array(PASTE_Y_LIST)
CLUSTERS = np.repeat(np.arange(8), 5)
REF_MEAN = PASTE_MEAN
REF_CRSE = PASTE_CRSE


def test_mean_matches_reference():
    res = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    assert res.mean == pytest.approx(REF_MEAN, abs=1e-9)


def test_clustered_se_near_analytic_crse():
    res = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    assert res.clustered_se == pytest.approx(REF_CRSE, rel=0.25)
    assert res.n_clusters == 8


def test_seed_reproducibility():
    a = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    b = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    c = cluster.cluster_bootstrap(Y, CLUSTERS, seed=1)
    assert a.clustered_se == b.clustered_se
    assert a.clustered_se != c.clustered_se


def test_design_effect_above_one_with_cluster_correlation():
    res = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    assert res.design_effect > 1.0


def test_single_cluster_se_is_degenerate_zero():
    res = cluster.cluster_bootstrap(np.array([0.5, 0.6, 0.7]), np.array([1, 1, 1]), seed=20260611)
    assert res.n_clusters == 1
    assert res.clustered_se == 0.0  # resampling one cluster reproduces the sample


def test_primary_rule_threshold():
    assert cluster.primary_is_clustered(13)
    assert cluster.primary_is_clustered(10)
    assert not cluster.primary_is_clustered(6)
```

- [ ] **Step 3: Run to verify failure, then implement**

```python
"""Cluster bootstrap standard errors and the pre-registered primary rule.

Pre-registration (design spec section 2.2, frozen 2026-06-11): clustered SE
is primary when n_clusters >= PRIMARY_THRESHOLD, else question-level is
primary and clustered is sensitivity. DIVERGENCE_RATIO drives the report's
correlation-sensitivity caution on naive-primary corpora.
"""

from dataclasses import dataclass

import numpy as np

PRIMARY_THRESHOLD = 10
DIVERGENCE_RATIO = 1.5
DEFAULT_SEED = 20260611
DEFAULT_N_BOOT = 10_000


@dataclass(frozen=True)
class ClusterSE:
    mean: float
    naive_se: float
    clustered_se: float
    n_clusters: int
    design_effect: float


def primary_is_clustered(n_clusters: int) -> bool:
    return n_clusters >= PRIMARY_THRESHOLD


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> ClusterSE:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.shape != clusters.shape:
        raise ValueError("values and clusters must align")
    labels = np.unique(clusters)
    groups = [values[clusters == c] for c in labels]
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[j] for j in picks])
        boot_means[i] = sample.mean()
    mean = float(values.mean())
    naive_se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    clustered_se = float(boot_means.std(ddof=1))
    deff = float((clustered_se / naive_se) ** 2) if naive_se > 0 else float("nan")
    return ClusterSE(mean, naive_se, clustered_se, len(labels), deff)
```

- [ ] **Step 4: Run, verify pass, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_cluster.py -v && make lint
git add stats/cluster.py tests/stats/test_cluster.py
git commit -m "stats(wp3): cluster bootstrap with CRSE reference and primary rule"
```

### Task 3.3: `stats/paired.py`

**Files:**
- Create: `stats/paired.py`
- Test: `tests/stats/test_paired.py`

- [ ] **Step 1: Generate references**

```bash
/tmp/refvec/bin/python - <<'EOF'
import statsmodels, numpy as np
from statsmodels.stats.contingency_tables import mcnemar
print("statsmodels", statsmodels.__version__)
print("mcnemar b=2 c=9:", mcnemar([[0, 2], [9, 0]], exact=True).pvalue)
print("mcnemar b=0 c=0:", "define p=1.0, no discordant pairs")
rng = np.random.default_rng(11)
diffs = rng.normal(0.02, 0.08, 22).tolist()
print("diffs:", diffs)
print("mean:", float(np.mean(diffs)))
EOF
```

- [ ] **Step 2: Write the failing tests**

```python
"""Paired bootstrap and exact McNemar.

McNemar reference: statsmodels <PASTE version> mcnemar(exact=True) on b=2,
c=9 (command in implementation plan Task 3.3). Bootstrap CI checked for
seed-stability, coverage direction, and the all-ties edge case.
"""

import numpy as np
import pytest

from stats import paired

DIFFS = np.array(PASTE_DIFFS_LIST)


def test_mcnemar_exact_matches_statsmodels():
    assert paired.mcnemar_exact(b=2, c=9) == pytest.approx(PASTE_MCNEMAR_P, abs=1e-12)


def test_mcnemar_no_discordant_pairs_p_one():
    assert paired.mcnemar_exact(b=0, c=0) == 1.0


def test_paired_bootstrap_seeded_and_centered():
    res = paired.paired_bootstrap(DIFFS, confidence=0.90, seed=20260611)
    again = paired.paired_bootstrap(DIFFS, confidence=0.90, seed=20260611)
    assert (res.ci_low, res.ci_high) == (again.ci_low, again.ci_high)
    assert res.ci_low < res.mean_diff < res.ci_high
    assert res.mean_diff == pytest.approx(float(DIFFS.mean()), abs=1e-12)


def test_all_ties_collapse_to_point():
    res = paired.paired_bootstrap(np.zeros(22), confidence=0.90, seed=20260611)
    assert res.ci_low == res.ci_high == res.mean_diff == 0.0


def test_cluster_mode_resamples_clusters():
    clusters = np.repeat(np.arange(11), 2)
    res = paired.paired_bootstrap(DIFFS, clusters=clusters, confidence=0.90, seed=20260611)
    free = paired.paired_bootstrap(DIFFS, confidence=0.90, seed=20260611)
    assert res.n_units == 11
    assert free.n_units == 22
    assert (res.ci_low, res.ci_high) != (free.ci_low, free.ci_high)
```

- [ ] **Step 3: Implement**

```python
"""Paired bootstrap CIs on per-question differences; exact McNemar.

Resampling unit per design spec section 2.5: clusters of paired differences
on a clustered-primary corpus, questions otherwise.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

DEFAULT_SEED = 20260611
DEFAULT_N_BOOT = 10_000


@dataclass(frozen=True)
class PairedResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    n_units: int


def paired_bootstrap(
    diffs: np.ndarray,
    clusters: np.ndarray | None = None,
    confidence: float = 0.90,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> PairedResult:
    diffs = np.asarray(diffs, dtype=float)
    if clusters is None:
        groups = [np.array([d]) for d in diffs]
    else:
        clusters = np.asarray(clusters)
        groups = [diffs[clusters == c] for c in np.unique(clusters)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        boot[i] = np.concatenate([groups[j] for j in picks]).mean()
    alpha = 1 - confidence
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return PairedResult(float(diffs.mean()), float(lo), float(hi), len(groups))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * binom.cdf(k, n, 0.5)
    return float(min(p, 1.0))
```

- [ ] **Step 4: Run, verify pass, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_paired.py -v && make lint
git add stats/paired.py tests/stats/test_paired.py
git commit -m "stats(wp3): paired bootstrap and exact mcnemar"
```

### Task 3.4: `stats/equivalence.py`

**Files:**
- Create: `stats/equivalence.py`
- Test: `tests/stats/test_equivalence.py`

- [ ] **Step 1: Generate the cross-check reference**

```bash
/tmp/refvec/bin/python - <<'EOF'
import statsmodels, numpy as np
from statsmodels.stats.weightstats import ttost_paired
print("statsmodels", statsmodels.__version__)
rng = np.random.default_rng(11)
a = rng.normal(0.72, 0.1, 50)
b = a + rng.normal(0.01, 0.05, 50)
p, t1, t2 = ttost_paired(b, a, -0.10, 0.10)
print("tost p:", p)
print("a:", a.tolist())
print("b:", b.tolist())
EOF
```

- [ ] **Step 2: Write the failing tests**

```python
"""TOST equivalence per design spec section 2.4.

Cross-check: statsmodels <PASTE version> ttost_paired on the seed-11 fixture
(command in implementation plan Task 3.4) must agree with the bootstrap TOST
verdict at margin 0.10; the two methods differ numerically, so the assertion
is verdict agreement plus CI sanity, not equality.
"""

import numpy as np
import pytest

from stats import equivalence

A = np.array(PASTE_A_LIST)
B = np.array(PASTE_B_LIST)
TOST_P = PASTE_TOST_P


def test_default_margin_and_alpha_are_preregistered():
    assert equivalence.DEFAULT_MARGIN == 0.10
    assert equivalence.DEFAULT_ALPHA == 0.05


def test_equivalent_fixture_passes_and_agrees_with_statsmodels():
    res = equivalence.tost_paired(B - A, seed=20260611)
    assert res.equivalent is (TOST_P < 0.05)
    assert res.equivalent is True
    assert res.support_margin == max(abs(res.ci_low), abs(res.ci_high))
    assert res.support_margin < 0.10


def test_large_shift_fails_equivalence():
    res = equivalence.tost_paired((B - A) + 0.15, seed=20260611)
    assert res.equivalent is False
    assert res.support_margin > 0.10


def test_degenerate_constant_diffs():
    res = equivalence.tost_paired(np.full(22, 0.02), seed=20260611)
    assert res.equivalent is True
    assert res.ci_low == res.ci_high == pytest.approx(0.02)
```

- [ ] **Step 3: Implement**

```python
"""TOST equivalence on paired differences via the bootstrap 90 percent CI.

Pre-registered (design spec section 2.4, frozen 2026-06-11): margin 0.10
absolute on P@5 and R@5, alpha 0.05 per one-sided test, no multiplicity
adjustment across the two metrics, paired score = per-question mean over
epochs. support_margin is the smallest margin at which equivalence would
pass: the larger absolute endpoint of the 90 percent CI, reported in both
the pass and fail branches (README wording in stats/report.py).
"""

from dataclasses import dataclass

import numpy as np

from stats.paired import DEFAULT_N_BOOT, DEFAULT_SEED, paired_bootstrap

DEFAULT_MARGIN = 0.10
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class TostResult:
    equivalent: bool
    ci_low: float
    ci_high: float
    margin: float
    support_margin: float
    n_units: int


def tost_paired(
    diffs: np.ndarray,
    margin: float = DEFAULT_MARGIN,
    alpha: float = DEFAULT_ALPHA,
    clusters: np.ndarray | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> TostResult:
    res = paired_bootstrap(
        np.asarray(diffs, dtype=float),
        clusters=clusters,
        confidence=1 - 2 * alpha,
        n_boot=n_boot,
        seed=seed,
    )
    equivalent = -margin < res.ci_low and res.ci_high < margin
    support = max(abs(res.ci_low), abs(res.ci_high))
    return TostResult(equivalent, res.ci_low, res.ci_high, margin, support, res.n_units)
```

- [ ] **Step 4: Run, verify pass, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_equivalence.py -v && make lint
git add stats/equivalence.py tests/stats/test_equivalence.py
git commit -m "stats(wp3): bootstrap TOST with statsmodels verdict cross-check"
```

### Task 3.5: `stats/power.py`

**Files:**
- Create: `stats/power.py`
- Test: `tests/stats/test_power.py`

- [ ] **Step 1: Write the failing tests** (the normal approximation is the independent reference here, per the v3.1 plan's own cross-check mandate)

```python
"""Power and MDE: seeded simulation cross-checked against the closed-form
normal approximation MDE = (z_{1-alpha/2} + z_{power}) * sd / sqrt(n).
Tolerance 15 percent relative: the two methods differ by design, but an
implementation bug (wrong test, wrong sd, wrong n) lands far outside it.
"""

import numpy as np
import pytest

from stats import power


def _diffs() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(0.0, 0.08, 22)


def test_power_increases_with_effect():
    d = _diffs()
    lo = power.simulated_power(d, delta=0.01, seed=20260611)
    hi = power.simulated_power(d, delta=0.10, seed=20260611)
    assert hi > lo
    assert 0.0 <= lo <= hi <= 1.0


def test_mde_simulation_near_normal_approx():
    d = _diffs()
    sim = power.mde(d, target_power=0.80, seed=20260611)
    approx = power.mde_normal_approx(d, target_power=0.80)
    assert sim == pytest.approx(approx, rel=0.15)


def test_seeded_reproducibility():
    d = _diffs()
    assert power.mde(d, seed=20260611) == power.mde(d, seed=20260611)
```

- [ ] **Step 2: Implement**

```python
"""Simulation-based power and MDE from measured per-question differences.

The simulation resamples the observed differences (preserving their measured
variance), shifts by delta, and tests at alpha with a paired t-test; the
closed-form normal approximation is the cross-check, per the v3.1 plan WP3.
"""

import numpy as np
from scipy.stats import norm, ttest_1samp

DEFAULT_SEED = 20260611
DEFAULT_N_SIM = 2_000


def simulated_power(
    diffs: np.ndarray,
    delta: float,
    alpha: float = 0.05,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
) -> float:
    centered = np.asarray(diffs, dtype=float)
    centered = centered - centered.mean()
    rng = np.random.default_rng(seed)
    n = len(centered)
    hits = 0
    for _ in range(n_sim):
        sample = rng.choice(centered, size=n, replace=True) + delta
        if ttest_1samp(sample, 0.0).pvalue < alpha:
            hits += 1
    return hits / n_sim


def mde(
    diffs: np.ndarray,
    target_power: float = 0.80,
    alpha: float = 0.05,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
    tol: float = 1e-3,
) -> float:
    lo, hi = 0.0, float(np.std(diffs, ddof=1)) * 4 + 1e-6
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if simulated_power(diffs, mid, alpha=alpha, n_sim=n_sim, seed=seed) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


def mde_normal_approx(diffs: np.ndarray, target_power: float = 0.80, alpha: float = 0.05) -> float:
    d = np.asarray(diffs, dtype=float)
    sd = float(d.std(ddof=1))
    z = norm.ppf(1 - alpha / 2) + norm.ppf(target_power)
    return float(z * sd / np.sqrt(len(d)))
```

- [ ] **Step 3: Run, verify pass, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_power.py -v && make lint
git add stats/power.py tests/stats/test_power.py
git commit -m "stats(wp3): simulated power and MDE with normal cross-check"
```

### Task 3.6: `stats/variance.py`

**Files:**
- Create: `stats/variance.py`
- Test: `tests/stats/test_variance.py`

- [ ] **Step 1: Write the failing test with a hand-worked reference**

```python
"""Variance decomposition reference: hand-computed one-way random-effects
ANOVA on 3 questions x 2 epochs, scores [[0,1],[1,1],[0,0]].
Arithmetic: question means 0.5, 1.0, 0.0; grand mean 0.5.
SSB = 2*((0)^2 + (0.5)^2 + (0.5)^2) = 1.0, df=2, MSB = 0.5.
SSW = (0.25 + 0.25) + 0 + 0 = 0.5, df = 3, MSW = 1/6.
between = (MSB - MSW) / k = (0.5 - 1/6) / 2 = 1/6; within = MSW = 1/6;
ICC = (1/6) / (1/6 + 1/6) = 0.5. Provenance: hand computation above.
"""

import pandas as pd
import pytest

from stats import variance


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "epoch": [1, 2, 1, 2, 1, 2],
            "score": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        }
    )


def test_hand_worked_decomposition():
    res = variance.decompose(_table(), value_col="score", question_col="question_id")
    assert res.within_question == pytest.approx(1 / 6)
    assert res.between_question == pytest.approx(1 / 6)
    assert res.icc == pytest.approx(0.5)


def test_zero_within_when_epochs_identical():
    df = _table()
    df["score"] = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    res = variance.decompose(df, value_col="score", question_col="question_id")
    assert res.within_question == 0.0
    assert res.icc == 1.0


def test_negative_between_clamped_to_zero():
    df = _table()
    df["score"] = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]  # all variance within
    res = variance.decompose(df, value_col="score", question_col="question_id")
    assert res.between_question == 0.0
```

- [ ] **Step 2: Implement**

```python
"""Between-question versus within-question variance from epoch data.

Balanced one-way random-effects ANOVA estimators; between-question variance
clamped at zero. Requires equal epochs per question (the epoch runner
produces balanced data; the adapter validates this upstream).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VarianceDecomposition:
    between_question: float
    within_question: float
    icc: float
    n_questions: int
    epochs_per_question: int


def decompose(df: pd.DataFrame, value_col: str, question_col: str) -> VarianceDecomposition:
    counts = df.groupby(question_col)[value_col].count()
    if counts.nunique() != 1:
        raise ValueError("unbalanced epochs per question; decomposition assumes balance")
    k = int(counts.iloc[0])
    if k < 2:
        raise ValueError("need at least 2 epochs per question to decompose variance")
    means = df.groupby(question_col)[value_col].mean()
    grand = df[value_col].mean()
    q = len(means)
    msb = k * float(((means - grand) ** 2).sum()) / (q - 1)
    ssw = float(((df[value_col] - df[question_col].map(means)) ** 2).sum())
    msw = ssw / (q * (k - 1))
    between = max((msb - msw) / k, 0.0)
    total = between + msw
    icc = between / total if total > 0 else 0.0
    return VarianceDecomposition(between, msw, icc, q, k)
```

- [ ] **Step 3: Run, verify pass, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_variance.py -v && make lint
git add stats/variance.py tests/stats/test_variance.py
git commit -m "stats(wp3): variance decomposition with hand-worked reference"
```

### Task 3.7: Import isolation

**Files:**
- Test: `tests/stats/test_isolation.py`

- [ ] **Step 1: Write the tests** (these should pass immediately if guardrail 1 held; a failure is a real defect)

```python
"""Guardrail 1 enforcement: stats/ never imports agent-bench.

Two mechanisms (design spec section 8): a meta-path blocker that fails any
agent_bench import while importing every stats submodule, and a source scan.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

import stats


class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "agent_bench" or fullname.startswith("agent_bench."):
            raise ImportError("stats/ must not import agent_bench (guardrail 1)")
        return None


def test_every_stats_submodule_imports_with_agent_bench_blocked():
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        for mod in pkgutil.walk_packages(stats.__path__, prefix="stats."):
            module = importlib.import_module(mod.name)
            importlib.reload(module)
    finally:
        sys.meta_path.remove(blocker)


def test_source_scan_finds_no_agent_bench_string():
    root = Path(stats.__file__).parent
    offenders = [
        p for p in root.rglob("*.py") if "agent_bench" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
```

- [ ] **Step 2: Run, verify pass, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_isolation.py -v
git add tests/stats/test_isolation.py
git commit -m "stats(wp3): import isolation blocker and source scan"
```

- [ ] **Step 3: Full exit gate and PR**

```bash
make test && make lint
```

**WP3 exit gate:** all reference-vector tests pass with provenance comments naming tool and version; seeds pinned; isolation tests pass; suite and lint green; scope question answered.

---

## WP4: Report generator (1 session)

Branch: `feat/stats-wp4-report`

### Task 4.1: Fixture long tables (base plus three degradation variants)

**Files:**
- Create: `tests/stats/fixtures/make_fixture_tables.py`
- Create (generated, committed): `tests/stats/fixtures/long_base.csv`, `long_nonzero_failure.csv`, `long_failed_equivalence.csv`, `long_divergent_se.csv`

- [ ] **Step 1: Write the deterministic generator**

```python
"""Generate the four fixture long tables for report tests. Deterministic:
seed 20260611, no wall clock (timestamps are literals). Rerunning must
reproduce the committed CSVs byte for byte; tests/stats/test_report.py
asserts that, so fixture drift is loud."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
SEED = 20260611
N_Q = 12
EPOCHS = 2
CONFIGS = ("custom-mock+00000000", "langchain-mock+00000000")
TS = "2026-06-15T10:00:00+00:00"


def _base() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for cfg_i, cfg in enumerate(CONFIGS):
        run = f"01HZXJ5M8N9PQRSTVWXYZ0{cfg_i}234"
        for q in range(N_Q):
            cluster = f"file_{q % 4}.md"  # 4 clusters: naive-primary branch exercised
            base_p5 = 0.7 - 0.02 * cfg_i + 0.05 * (q % 3)
            for epoch in range(1, EPOCHS + 1):
                noise = rng.normal(0, 0.02)
                for metric, score in (
                    ("p_at_5", min(max(base_p5 + noise, 0.0), 1.0)),
                    ("r_at_5", min(max(base_p5 + 0.1 + noise, 0.0), 1.0)),
                    ("citation_acc", 1.0),
                ):
                    rows.append(
                        dict(
                            run_id=run,
                            timestamp=TS,
                            config_id=cfg,
                            code_version="fixture0",
                            dataset_version="sha-fixture0",
                            question_id=f"fq{q:03d}",
                            cluster_id=cluster,
                            epoch=epoch,
                            metric=metric,
                            score=round(float(score), 6),
                            latency_ms=1000.0,
                            cost_usd=0.0002,
                            refused=False,
                        )
                    )
    return pd.DataFrame(rows)


def main() -> None:
    base = _base()
    base.to_csv(OUT / "long_base.csv", index=False)

    nonzero = base.copy()
    first_cit = nonzero.index[(nonzero["metric"] == "citation_acc")][0]
    nonzero.loc[first_cit, "score"] = 0.5
    nonzero.to_csv(OUT / "long_nonzero_failure.csv", index=False)

    failed_eq = base.copy()
    custom = (failed_eq["config_id"] == CONFIGS[0]) & failed_eq["metric"].isin(["p_at_5", "r_at_5"])
    failed_eq.loc[custom, "score"] = (failed_eq.loc[custom, "score"] + 0.2).clip(upper=1.0)
    failed_eq.to_csv(OUT / "long_failed_equivalence.csv", index=False)

    divergent = base.copy()
    bump = divergent["cluster_id"].map({"file_0.md": 0.15, "file_1.md": -0.15}).fillna(0.0)
    mask = divergent["metric"] == "p_at_5"
    divergent.loc[mask, "score"] = (divergent.loc[mask, "score"] + bump[mask]).clip(0.0, 1.0).round(6)
    divergent.to_csv(OUT / "long_divergent_se.csv", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate, eyeball, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 tests/stats/fixtures/make_fixture_tables.py
head -3 tests/stats/fixtures/long_base.csv
git add tests/stats/fixtures/
git commit -m "stats(wp4): deterministic fixture long tables"
```

### Task 4.2: `stats/report.py` with the three degradation branches

**Files:**
- Create: `stats/report.py`
- Test: `tests/stats/test_report.py`
- Modify: `Makefile` (`evaluate-stats` target)

- [ ] **Step 1: Write the failing behavior tests** (golden files arrive in Task 4.3; these pin the branch logic)

```python
"""Report generator tests: degradation branches, byte-stability, fixture drift."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from stats import report

FIXTURES = Path(__file__).parent / "fixtures"


def _render(name: str) -> str:
    df = pd.read_csv(FIXTURES / name, dtype={"refused": "boolean"})
    return report.render_report({"mini": df}, seed=20260611)


def test_fixture_tables_have_not_drifted(tmp_path):
    src = FIXTURES / "make_fixture_tables.py"
    subprocess.run([sys.executable, str(src)], check=True, cwd=FIXTURES)
    out = subprocess.run(["git", "diff", "--stat", str(FIXTURES)], capture_output=True, text=True)
    assert "csv" not in out.stdout, "regenerating fixtures changed committed CSVs"


def test_zero_failure_branch_uses_rule_of_three_phrasing():
    text = _render("long_base.csv")
    assert "rule of three" in text.lower()
    assert "Clopper-Pearson" in text


def test_nonzero_failure_branch_drops_rule_of_three_phrasing():
    text = _render("long_nonzero_failure.csv")
    assert "rule of three" not in text.lower()
    assert "Clopper-Pearson" in text


def test_equivalence_pass_and_fail_wordings():
    passing = _render("long_base.csv")
    failing = _render("long_failed_equivalence.csv")
    assert "equivalent within plus or minus 0.10" in passing
    assert "equivalence not established at plus or minus 0.10" in failing
    assert "the data support" in passing and "the data support" in failing


def test_divergence_caution_fires_only_on_divergent_fixture():
    calm = _render("long_base.csv")
    loud = _render("long_divergent_se.csv")
    assert "correlation-sensitivity caution" not in calm
    assert "correlation-sensitivity caution" in loud


def test_always_prints_n_clusters_and_design_effect():
    text = _render("long_base.csv")
    assert "n_clusters" in text
    assert "design effect" in text


def test_byte_stability_double_render():
    # No wall-clock anywhere in the renderer: two renders of the same input
    # are byte-identical, and the golden tests in Task 4.3 pin the bytes
    # across processes and machines. Dates in the output may come only from
    # the input table or from pre-registration literals in fixed prose.
    assert _render("long_base.csv") == _render("long_base.csv")
```

- [ ] **Step 2: Implement the renderer**

`stats/report.py`:

```python
"""Render docs/_generated/stats_report.md from long-format tables.

Pure function of its inputs: no wall clock anywhere (byte-stable golden
tests); identity comes from input-table content hashes and seeds. The three
degradation branches (design spec section 7) are code here, not prose:
divergence caution, zero-failure phrasing, TOST pass and fail wordings.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from stats.cluster import DIVERGENCE_RATIO, cluster_bootstrap, primary_is_clustered
from stats.equivalence import tost_paired
from stats.intervals import clopper_pearson, rule_of_three, zero_failure_upper
from stats.power import mde, mde_normal_approx
from stats.variance import decompose

HEADLINE_METRICS = ("p_at_5", "r_at_5")
DEFAULT_SEED = 20260611


def _question_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df["metric"] == metric]
    return sub.groupby(["config_id", "question_id", "cluster_id"], as_index=False)["score"].mean()


def _headline_section(df: pd.DataFrame, corpus: str, seed: int) -> list[str]:
    lines = [f"## Headline intervals: {corpus}", ""]
    lines.append("| config | metric | mean | 95 percent interval (primary) | naive SE | clustered SE | n_clusters | design effect |")
    lines.append("|---|---|---|---|---|---|---|---|")
    cautions = []
    for config in sorted(df["config_id"].unique()):
        for metric in HEADLINE_METRICS:
            qm = _question_means(df[df["config_id"] == config], metric)
            if qm.empty:
                continue
            res = cluster_bootstrap(qm["score"].to_numpy(), qm["cluster_id"].to_numpy(), seed=seed)
            primary_se = res.clustered_se if primary_is_clustered(res.n_clusters) else res.naive_se
            label = "clustered" if primary_is_clustered(res.n_clusters) else "question-level"
            lo, hi = res.mean - 1.96 * primary_se, res.mean + 1.96 * primary_se
            lines.append(
                f"| {config} | {metric} | {res.mean:.3f} | [{lo:.3f}, {hi:.3f}] ({label}) "
                f"| {res.naive_se:.4f} | {res.clustered_se:.4f} | n_clusters={res.n_clusters} "
                f"| design effect={res.design_effect:.2f} |"
            )
            if not primary_is_clustered(res.n_clusters) and res.clustered_se > DIVERGENCE_RATIO * res.naive_se:
                cautions.append(
                    f"correlation-sensitivity caution: {corpus} {config} {metric}: clustered SE "
                    f"{res.clustered_se:.4f} exceeds {DIVERGENCE_RATIO}x the question-level SE "
                    f"{res.naive_se:.4f}; the question-level headline interval likely understates uncertainty."
                )
    lines.append("")
    lines.extend(cautions)
    return lines


def _citation_section(df: pd.DataFrame, corpus: str) -> list[str]:
    lines = [f"## Citation accuracy zero-failure bound: {corpus}", ""]
    for config in sorted(df["config_id"].unique()):
        sub = df[df["config_id"] == config]
        # Inclusion rule, spec section 2.3: a question enters n if any epoch
        # emitted a citation_acc row (the adapter omits vacuous citation-free
        # rows, so the table itself encodes the rule). Exclusions counted
        # against the in-scope universe, which always emits p_at_5.
        in_scope = sub.loc[sub["metric"] == "p_at_5", "question_id"].nunique()
        cit = sub[sub["metric"] == "citation_acc"]
        if cit.empty:
            lines.append(f"- {config}: no answers contained citations; bound not computable.")
            continue
        k = int(sub.groupby("question_id")["epoch"].nunique().max())
        per_q_min = cit.groupby("question_id")["score"].min()
        n = len(per_q_min)
        excluded = in_scope - n
        all_epochs = int((cit.groupby("question_id")["epoch"].nunique() == k).sum())
        failures = int((per_q_min < 1.0).sum())
        bookkeeping = (
            f"included n={n}, excluded {excluded} citation-free questions; "
            f"cited in all epochs: {all_epochs}, in some epochs: {n - all_epochs}"
        )
        if failures == 0:
            lines.append(
                f"- {config}: 0 failures in {n} included questions ({bookkeeping}). "
                f"Exact Clopper-Pearson 95 percent upper bound on the per-question failure rate: "
                f"{zero_failure_upper(n):.3f} (rule of three approximation 3/n = {rule_of_three(n):.3f})."
            )
        else:
            lo, hi = clopper_pearson(failures, n)
            lines.append(
                f"- {config}: {failures} of {n} included questions showed a citation failure "
                f"({bookkeeping}). Clopper-Pearson 95 percent interval on the failure rate: "
                f"[{lo:.3f}, {hi:.3f}]."
            )
    lines.append("")
    return lines


def _equivalence_section(df: pd.DataFrame, corpus: str, seed: int) -> list[str]:
    configs = sorted(df["config_id"].unique())
    customs = [c for c in configs if c.startswith("custom")]
    langchains = [c for c in configs if c.startswith("langchain")]
    lines = [f"## Framework equivalence (TOST): {corpus}", ""]
    for custom in customs:
        for lc in langchains:
            for metric in HEADLINE_METRICS:
                a = _question_means(df[df["config_id"] == custom], metric).set_index("question_id")
                b = _question_means(df[df["config_id"] == lc], metric).set_index("question_id")
                shared = a.index.intersection(b.index)
                if shared.empty:
                    continue
                diffs = (a.loc[shared, "score"] - b.loc[shared, "score"]).to_numpy()
                clusters = a.loc[shared, "cluster_id"].to_numpy()
                use_clusters = primary_is_clustered(len(np.unique(clusters)))
                res = tost_paired(diffs, clusters=clusters if use_clusters else None, seed=seed)
                if res.equivalent:
                    verdict = (
                        f"equivalent within plus or minus {res.margin:.2f}; "
                        f"the data support equivalence down to plus or minus {res.support_margin:.3f}"
                    )
                else:
                    verdict = (
                        f"equivalence not established at plus or minus {res.margin:.2f}; "
                        f"the data support only plus or minus {res.support_margin:.3f}"
                    )
                lines.append(
                    f"- {custom} vs {lc}, {metric}: mean diff {diffs.mean():+.3f}, "
                    f"90 percent CI [{res.ci_low:+.3f}, {res.ci_high:+.3f}], n={res.n_units}: {verdict}."
                )
    lines.append("")
    return lines


def _variance_power_section(df: pd.DataFrame, corpus: str, seed: int) -> list[str]:
    lines = [f"## Variance decomposition and power: {corpus}", ""]
    p5 = df[df["metric"] == "p_at_5"]
    if p5.groupby("question_id")["epoch"].nunique().min() >= 2:
        dec = decompose(
            p5.groupby(["question_id", "epoch"], as_index=False)["score"].mean(),
            value_col="score",
            question_col="question_id",
        )
        lines.append(
            f"- p_at_5 variance: between-question {dec.between_question:.5f}, "
            f"within-question {dec.within_question:.5f}, ICC {dec.icc:.2f} "
            f"({dec.n_questions} questions x {dec.epochs_per_question} epochs)."
        )
        lines.append(
            "- Error budget preview: the interval above is the statistical term only; "
            "template sensitivity and judge bias are systematic terms, scoped for v3.2."
        )
    configs = sorted(df["config_id"].unique())
    if len(configs) >= 2:
        a = _question_means(df[df["config_id"] == configs[0]], "p_at_5").set_index("question_id")
        b = _question_means(df[df["config_id"] == configs[1]], "p_at_5").set_index("question_id")
        shared = a.index.intersection(b.index)
        diffs = (a.loc[shared, "score"] - b.loc[shared, "score"]).to_numpy()
        lines.append(
            f"- Minimum detectable p_at_5 difference at 80 percent power: "
            f"{mde(diffs, seed=seed):.3f} (normal approximation {mde_normal_approx(diffs):.3f})."
        )
    lines.append("")
    return lines


def _methods_appendix(tables: dict[str, pd.DataFrame], seed: int) -> list[str]:
    lines = ["## Methods appendix", ""]
    lines.append(
        "- Estimators: cluster bootstrap over cluster_id (10000 replicates); paired bootstrap "
        "on per-question epoch-mean differences; TOST at margin 0.10 absolute, alpha 0.05 per "
        "one-sided test, no multiplicity adjustment across P@5 and R@5 (pre-registered, design "
        "spec section 2, frozen 2026-06-11 before any WP5 data existed)."
    )
    lines.append(
        "- Zero-failure bounding: a question succeeds only if zero hallucinated citations "
        "occurred across all epochs and all citations; collapsing epochs bounds the any-of-k "
        "failure rate, which also bounds the per-answer rate."
    )
    lines.append(f"- Seed: {seed}. No wall-clock values appear in this report.")
    for name, df in sorted(tables.items()):
        digest = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:12]
        lines.append(f"- Input table {name}: {len(df)} rows, content hash {digest}.")
    lines.append("")
    return lines


def render_report(tables: dict[str, pd.DataFrame], seed: int = DEFAULT_SEED) -> str:
    lines = ["# Statistics report", ""]
    for corpus, df in sorted(tables.items()):
        lines.extend(_headline_section(df, corpus, seed))
        lines.extend(_citation_section(df, corpus))
        lines.extend(_equivalence_section(df, corpus, seed))
        lines.extend(_variance_power_section(df, corpus, seed))
    lines.extend(_methods_appendix(tables, seed))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default="results/long", help="directory of long CSVs")
    parser.add_argument("--out", default="docs/_generated/stats_report.md")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    tables = {
        p.stem: pd.read_csv(p, dtype={"refused": "boolean"})
        for p in sorted(Path(args.tables).rglob("*.csv"))
    }
    if not tables:
        raise SystemExit(f"no CSV tables under {args.tables}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(tables, seed=args.seed))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Makefile target** (tab-indented)

```makefile
evaluate-stats:  ## Regenerate docs/_generated/stats_report.md from results/long (free, offline)
	$(PYTHON) -m stats.report --tables results/long --out docs/_generated/stats_report.md
```

- [ ] **Step 4: Run, verify pass, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_report.py -v && make lint
git add stats/report.py tests/stats/test_report.py Makefile
git commit -m "stats(wp4): report renderer with three degradation branches"
```

### Task 4.3: Golden files

**Files:**
- Create (generated, eyeballed, committed): `tests/stats/fixtures/golden_report_base.md`, `golden_report_nonzero_failure.md`, `golden_report_failed_equivalence.md`, `golden_report_divergent_se.md`
- Test: `tests/stats/test_report.py`

- [ ] **Step 1: Generate the four reports, read each top to bottom**

```bash
/usr/local/opt/python@3.11/bin/python3.11 - <<'EOF'
from pathlib import Path
import pandas as pd
from stats import report

fix = Path("tests/stats/fixtures")
for name in ("base", "nonzero_failure", "failed_equivalence", "divergent_se"):
    df = pd.read_csv(fix / f"long_{name}.csv", dtype={"refused": "boolean"})
    (fix / f"golden_report_{name}.md").write_text(report.render_report({"mini": df}, seed=20260611))
    print("wrote", name)
EOF
```

Eyeball each golden file before committing: intervals plausible, no NaN, no banned dashes, each branch's wording present in its variant. This eyeball is the human gate that makes golden tests meaningful; do not skip it.

- [ ] **Step 2: Add the golden test**

Append to `tests/stats/test_report.py`:

```python
import pytest


@pytest.mark.parametrize(
    "name", ["base", "nonzero_failure", "failed_equivalence", "divergent_se"]
)
def test_golden_reports_byte_stable(name):
    expected = (FIXTURES / f"golden_report_{name}.md").read_text()
    assert _render(f"long_{name}.csv") == expected
```

- [ ] **Step 3: Run everything, commit, PR**

```bash
make test && make lint
git add tests/stats/fixtures/golden_report_*.md tests/stats/test_report.py
git commit -m "stats(wp4): golden report files for all degradation branches"
```

**WP4 exit gate:** `make evaluate-stats` regenerates the report from `results/long/` with one command; golden tests byte-stable; suite and lint green; scope question answered.

---

## WP5: Paid measurement campaign (HUMAN-RUN, not a Claude Code session)

Jane runs, after confirming WP0 through WP4 merged and `.env` keys present:

```bash
make epochs K=5 CONFIGS=custom-openai,custom-anthropic,langchain-openai,langchain-anthropic CONFIRM_PAID=1
python -m stats_adapters.from_results_json --input results/epochs/<run_id> \
    --golden agent_bench/evaluation/datasets/tech_docs_golden.json --out-dir results/long
make evaluate-stats
```

Expected scale: 27 questions x 5 epochs x 4 configs = 540 question-runs, cost well under 5 USD. Commit the raw envelopes under `results/epochs/` and the long tables under `results/long/` (storage decision: committed, per the design spec section 1).

**WP5 exit gate:** `make stats-table` equivalents and `make evaluate-stats` complete on real data; report eyeballed (intervals plausible, no NaN); raw epoch outputs committed so every later analysis is rerunnable without new spend.

---

## WP6: README integration (1 session, requires WP5 data)

Branch: `feat/stats-wp6-readme`

### Task 6.1: Consistency checker

**Files:**
- Create: `scripts/check_readme_stats.py`
- Test: extend `tests/stats/test_report.py`

- [ ] **Step 1: Write the checker**

```python
"""Verify README statistics match docs/_generated/stats_report.md.

Protocol: every number the README quotes from the report is wrapped in an
HTML marker comment: <!-- stats:KEY -->value<!-- /stats -->. The same KEY
must appear in the report as a line containing "KEY = value". Drift in
either direction fails CI loudly.
"""

import re
import sys
from pathlib import Path

MARKER = re.compile(r"<!-- stats:([a-z0-9_]+) -->([^<]+)<!-- /stats -->")


def main() -> int:
    readme = Path("README.md").read_text()
    report = Path("docs/_generated/stats_report.md").read_text()
    failures = []
    pairs = MARKER.findall(readme)
    if not pairs:
        failures.append("README contains no stats markers; WP6 edits not applied")
    for key, value in pairs:
        needle = f"{key} = {value.strip()}"
        if needle not in report:
            failures.append(f"README claims {key} = {value.strip()} but report does not state it")
    for line in failures:
        print(f"FAIL: {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

Add to `stats/report.py`'s methods appendix a `README values` block emitting one `KEY = value` line per README-quoted number (keys: `p_at_5_custom_openai_ci`, `tost_verdict_p_at_5`, `citation_zero_failure_bound_n`, `mde_p_at_5_80`, one per quoted figure; extend the list as the WP6 edits are made, same commit).

- [ ] **Step 2: README edit list** (apply exactly; every number via marker comments)

1. Both benchmark tables gain a plus or minus 95 percent interval column from the report's headline section.
2. Winner bolding replaced by the stated rule: bold only differences significant under the paired test; if none survive, bold nothing and say so under the table.
3. The framework-comparison key insight restated as the TOST claim, using the report's exact pass or fail wording (section 2.4 of the design spec).
4. The rule-of-three sentence lands next to the citation-accuracy claim, with n from the report, never hardcoded prose.
5. New short paragraph "What this benchmark can detect": MDE at 80 percent power, from the report.
6. Link `docs/_generated/stats_report.md`.
7. Methodology Notes extended: fold the existing refusal-gate variance finding into the formal variance-decomposition result.

- [ ] **Step 3: Verify and commit**

```bash
make evaluate-stats && /usr/local/opt/python@3.11/bin/python3.11 scripts/check_readme_stats.py
make test && make lint
git add README.md scripts/check_readme_stats.py stats/report.py
git commit -m "stats(wp6): README intervals, TOST claim, and consistency checker"
```

**WP6 exit gate:** checker exits 0; README numbers match the generated report exactly; Jane reviews the diff before merge; scope question answered.

---

## WP7 (droppable, in priority order): agreement CIs, then pass^k

Branch: `feat/stats-wp7-agreement`

### Task 7.1: `stats/agreement.py`

**Files:**
- Create: `stats/agreement.py`
- Test: `tests/stats/test_agreement.py`

- [ ] **Step 1: Write the failing tests with the hand-worked reference**

```python
"""Kappa and AC1 references: hand-computed 2x2 example, n=50,
a=20 (yes/yes), b=5 (yes/no), c=10 (no/yes), d=15 (no/no).
po = 35/50 = 0.7. Rater1 yes = 0.5, rater2 yes = 0.6.
kappa: pe = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = (0.7-0.5)/0.5 = 0.4.
AC1: pgamma = (0.5+0.6)/2 = 0.55; e = 2*0.55*0.45 = 0.495;
AC1 = (0.7-0.495)/(1-0.495) = 0.405940594...
Provenance: hand computation above; cross-checkable against the R irrCAC
package (gwet.ac1.raw) if ever in doubt."""

import numpy as np
import pytest

from stats import agreement

A = np.array([1] * 20 + [1] * 5 + [0] * 10 + [0] * 15)
B = np.array([1] * 20 + [0] * 5 + [1] * 10 + [0] * 15)


def test_cohen_kappa_hand_worked():
    assert agreement.cohen_kappa(A, B) == pytest.approx(0.4, abs=1e-12)


def test_gwet_ac1_hand_worked():
    assert agreement.gwet_ac1(A, B) == pytest.approx(0.205 / 0.505, abs=1e-9)


def test_bootstrap_ci_contains_point_and_is_seeded():
    lo1, hi1 = agreement.bootstrap_agreement_ci(agreement.cohen_kappa, A, B, seed=20260611)
    lo2, hi2 = agreement.bootstrap_agreement_ci(agreement.cohen_kappa, A, B, seed=20260611)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= 0.4 <= hi1


def test_degenerate_single_category_kappa_is_nan_ac1_defined():
    ones = np.ones(30)
    assert np.isnan(agreement.cohen_kappa(ones, ones))
    assert agreement.gwet_ac1(ones, ones) == pytest.approx(1.0)
```

- [ ] **Step 2: Implement**

```python
"""Chance-corrected agreement with bootstrap CIs over the item set.

kappa degenerates when marginals are extreme (returns nan when pe = 1);
AC1 stays defined, which is why both are reported (see the calibration
work and docs/judge-design.md). Pure module: numpy only.
"""

from collections.abc import Callable

import numpy as np

DEFAULT_SEED = 20260611
DEFAULT_N_BOOT = 10_000


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = float((a == b).mean())
    pgamma = np.array([((a == c).mean() + (b == c).mean()) / 2 for c in cats])
    q = len(cats)
    if q == 1:
        return 1.0
    pe = float((pgamma * (1 - pgamma)).sum() / (q - 1))
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def bootstrap_agreement_ci(
    stat: Callable[[np.ndarray, np.ndarray], float],
    a: np.ndarray,
    b: np.ndarray,
    confidence: float = 0.95,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = stat(a[idx], b[idx])
        if not np.isnan(v):
            vals.append(v)
    alpha = 1 - confidence
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
```

- [ ] **Step 3: Wire into the judge-design doc.** Compute CIs for the 30-item label set per dimension by joining `measurements/2026-05-04-judge-calibration-labels.jsonl` against the jury outputs in `results/calibration_v1_judge_jury_kappa_weighted_v1_1.json` (read both via `stats_adapters`, never inside `stats/`). Add the per-dimension kappa and AC1 intervals to `docs/judge-design.md` as a new subsection titled "Agreement uncertainty", stating the wide-interval honesty rule: the completeness kappa of 0.416 gets an interval, not an excuse.

- [ ] **Step 4: Run, verify, commit**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m pytest tests/stats/test_agreement.py -v && make test && make lint
git add stats/agreement.py tests/stats/test_agreement.py docs/judge-design.md
git commit -m "stats(wp7): agreement CIs for kappa and AC1"
```

### Task 7.2: `stats/reliability.py` (pass^k)

**Files:**
- Create: `stats/reliability.py`
- Test: `tests/stats/test_reliability.py`

- [ ] **Step 1: Write the failing tests**

```python
"""pass^k for refusal behavior: fraction of questions passing in all k epochs,
with a Wilson interval over questions. Hand reference: 3 questions over 2
epochs with refusal_correct (1,1), (1,0), (1,1): pass^2 = 2/3."""

import pandas as pd
import pytest

from stats import reliability
from stats.intervals import wilson


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "epoch": [1, 2, 1, 2, 1, 2],
            "metric": ["refusal_correct"] * 6,
            "score": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )


def test_pass_k_hand_worked():
    res = reliability.pass_k(_table(), metric="refusal_correct")
    assert res.k == 2
    assert res.n_questions == 3
    assert res.rate == pytest.approx(2 / 3)
    assert (res.ci_low, res.ci_high) == pytest.approx(wilson(2, 3))


def test_requires_balanced_epochs():
    df = _table().drop(index=5)
    with pytest.raises(ValueError):
        reliability.pass_k(df, metric="refusal_correct")
```

- [ ] **Step 2: Implement**

```python
"""pass^k from epoch data: a question passes only if every epoch passes."""

from dataclasses import dataclass

import pandas as pd

from stats.intervals import wilson


@dataclass(frozen=True)
class PassK:
    k: int
    n_questions: int
    rate: float
    ci_low: float
    ci_high: float


def pass_k(df: pd.DataFrame, metric: str) -> PassK:
    sub = df[df["metric"] == metric]
    counts = sub.groupby("question_id")["epoch"].nunique()
    if counts.nunique() != 1:
        raise ValueError("unbalanced epochs per question")
    k = int(counts.iloc[0])
    all_pass = sub.groupby("question_id")["score"].min() >= 1.0
    n = len(all_pass)
    passed = int(all_pass.sum())
    lo, hi = wilson(passed, n)
    return PassK(k, n, passed / n, lo, hi)
```

- [ ] **Step 3: Add one pass^k table to the report** (new short section in `stats/report.py` after the variance section, rendered only when `refusal_correct` rows exist; regenerate goldens via the Task 4.3 procedure in the same commit, re-eyeballing the diff). Run, verify, commit:

```bash
make test && make lint
git add stats/reliability.py tests/stats/test_reliability.py stats/report.py tests/stats/fixtures/golden_report_*.md
git commit -m "stats(wp7): pass^k table for refusal behavior"
```

**WP7 exit gate (per item):** tests and lint green; report section regenerates; scope question answered. If time presses, ship Task 7.1 and defer Task 7.2, in that order.

---

## Definition of done (v3.1, mirrors the v0.2 plan section 7)

- [ ] `stats/` pure package with the import-isolation tests passing.
- [ ] Adapter, epoch runner, and report generator fixture-tested and CI-green without API keys.
- [ ] WP5 campaign data committed and rerunnable; report regenerates from it with one command.
- [ ] README updated per the WP6 edit list; `scripts/check_readme_stats.py` exits 0.
- [ ] 528 pre-existing tests untouched and passing; all new tests under `tests/stats/`.
- [ ] No em or en dashes introduced in docs; ruff and mypy clean; `SCHEMA_VERSION` carries Jane's verified value.

*Changelog*

- v1.0 (2026-06-11): initial conversion from the v3.1 plan v0.2 plus the design spec at `docs/plans/2026-06-11-stats-layer-v3.1-design.md`.
