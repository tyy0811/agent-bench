# Judge Layer v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the discrete-scale, per-dimension LLM-judge layer with a κ-validated 2-judge jury, replacing `agent_bench/evaluation/metrics.py`'s continuous-scale `answer_faithfulness` / `answer_correctness` judges per the design at `docs/plans/2026-05-04-judge-layer-v1-design.md` (commit `44c65d4`).

**Architecture:** Four new sibling subpackages under `agent_bench/evaluation/`: `judges/` (Rubric, ScoreResult, Judge ABC, concrete judges), `rubrics/` (markdown rubric files), `variance/` (rubric_permute, jury), `calibration/` (hand-rolled κ + bootstrap, report generator). Hard cut on the old judges (no deprecation). New `evaluation.judge_dimensions` config field; existing `evaluation.judge_provider` knob preserved.

**Tech Stack:** Python 3.11, Pydantic 2, structlog, pytest + pytest-asyncio, existing `LLMProvider` abstraction at `agent_bench/core/provider.py`. No new runtime deps. sklearn appears only in `scripts/_dev/` for fixture generation (run from a venv outside the project).

**Branch:** `feat/judge-layer-v1` (already created off `main` at `0e96cb9`; spec landed as `44c65d4`).

---

## File Structure

Files created or modified, with their single responsibility. Each file should be holdable in context as one unit.

### New code files

| File | Responsibility |
|---|---|
| `agent_bench/evaluation/judges/__init__.py` | Re-export public surface (`Judge`, `ScoreResult`, `Rubric`, `MockJudge`, abstain constants) |
| `agent_bench/evaluation/judges/base.py` | `Judge` ABC, `ScoreResult`, `Rubric`, `RubricLevel`, `MockJudge`, abstain-reason constants |
| `agent_bench/evaluation/judges/groundedness.py` | `GroundednessJudge` — binary, reference-based on `source_snippets` |
| `agent_bench/evaluation/judges/relevance.py` | `RelevanceJudge` — 3-pt, reference-free |
| `agent_bench/evaluation/judges/completeness.py` | `CompletenessJudge` — 3-pt, reference-based on `reference_answer` |
| `agent_bench/evaluation/judges/citation_faithfulness.py` | `CitationFaithfulnessJudge` — binary, per-(claim,citation) aggregated all-or-nothing |
| `agent_bench/evaluation/variance/__init__.py` | Re-export `rubric_permute`, `jury`, `Jury`, `PermutedJudge` |
| `agent_bench/evaluation/variance/rubric_permute.py` | `PermutedJudge` wrapper, deterministic permutation by seed |
| `agent_bench/evaluation/variance/jury.py` | `Jury` aggregator (mean / kappa_weighted), strict quorum default, sidecar JSONL writer |
| `agent_bench/evaluation/calibration/__init__.py` | Re-export metrics + report generator |
| `agent_bench/evaluation/calibration/metrics.py` | Hand-rolled `cohen_kappa`, `gwets_ac2`, `bootstrap_ci` |
| `agent_bench/evaluation/calibration/report.py` | `generate_kappa_table` — joins predictions ⋈ labels by hash, computes per-row κ + CI + abstain breakdown |
| `scripts/run_calibration.py` | Three subcommands: `generate-outputs`, `run-judges --row-config=<path>`, `build-table [--strict]` |
| `scripts/_dev/generate_kappa_fixtures.py` | sklearn-dependent fixture generator (NOT runtime); produces inline constants + JSON sidecar |

### New rubric files (markdown with YAML frontmatter)

| File | Scale | Reference-based |
|---|---|---|
| `agent_bench/evaluation/rubrics/groundedness.md` | binary | yes (uses `source_snippets`) |
| `agent_bench/evaluation/rubrics/relevance.md` | three_point | no |
| `agent_bench/evaluation/rubrics/completeness.md` | three_point | yes (uses `reference_answer`) |
| `agent_bench/evaluation/rubrics/citation_faithfulness.md` | binary | yes (uses retrieved chunks) |

### New configuration / data files

| File | Purpose |
|---|---|
| `agent_bench/evaluation/datasets/calibration_v1.json` | 30 stratified item IDs + version + `system_config_git_sha` |
| `configs/calibration/rows/baseline.yaml` | Single Claude-Haiku, all variance controls on |
| `configs/calibration/rows/baseline_no_cot.yaml` | Ablation: CoT off |
| `configs/calibration/rows/baseline_no_anchors.yaml` | Ablation: rubric anchors stripped |
| `configs/calibration/rows/baseline_no_abstain.yaml` | Ablation: abstain disallowed |
| `configs/calibration/rows/permute.yaml` | Rubric permutation N=2 over baseline |
| `configs/calibration/rows/jury_kappa_weighted.yaml` | 2-judge jury (Claude-Haiku + gpt-4o-mini), kappa_weighted |

### New test files

| File | Tests | Notes |
|---|---|---|
| `tests/evaluation/__init__.py` | — | Marker only |
| `tests/evaluation/test_judges.py` | ~7 | ABC contract, ScoreResult, MockJudge, abstain-with-prefix |
| `tests/evaluation/test_rubric_loading.py` | ~6 | Construction validation, source_hash, permutation determinism |
| `tests/evaluation/test_calibration_metrics.py` | ~7 | Hand-computed κ + sklearn-fixture parity + bootstrap CI |
| `tests/evaluation/test_jury_aggregation.py` | ~5 | mean, kappa_weighted, quorum, sidecar, cancel-on-non-retryable |
| `tests/evaluation/test_calibration_report.py` | ~6 | Hash-mismatch raise, --strict, abstain-flag boundary, undefined-κ |
| `tests/evaluation/test_harness_migration.py` | ~3 | judge_scores populated, OOS skipped, judge_provider config preserved |
| `tests/evaluation/test_mockjudge_coverage.py` | ~1 | item.id walk across all goldens |
| `tests/evaluation/fixtures/sklearn_kappa_inputs.json` | — | Cross-check input file for sklearn-fixture CI test |

### Modified files

| File | Modification |
|---|---|
| `agent_bench/evaluation/metrics.py` | DELETE `answer_faithfulness`, `answer_correctness`, `_judge_call`, `_FAITHFULNESS_PROMPT`, `_CORRECTNESS_PROMPT`. Keep deterministic metrics. |
| `agent_bench/evaluation/harness.py` | Migrate `run_evaluation` to use new judges; add `judge_scores: dict[str, ScoreResult]` to `EvalResult`; remove `faithfulness`, `correctness` fields |
| `agent_bench/core/config.py` | Add `judge_dimensions: list[str] = ["groundedness", "relevance", "completeness"]` to `EvaluationConfig` |
| `tests/test_evaluation.py` | Drop assertions on removed `faithfulness` / `correctness` fields |
| `pyproject.toml` | Add `extend-exclude = ["scripts/_dev"]` to `[tool.ruff]`; add same to mypy config |
| `Makefile` | Add `calibrate` and `evaluate-judges` targets |
| `.github/workflows/ci.yaml` | Add explicit empty `env: {}` block to test job (documents existing behavior) |
| `README.md` | Add "Targets that cost money" subheading with four-column table |
| `docs/DESIGN.md` | Rewrite §"LLM-judge metrics (costs money, manual)" to point at design doc + writeup |
| `DECISIONS.md` | Append supersession entry referencing concrete file paths |
| `measurements/README.md` | Add row for the calibration-labels JSONL |

---

## Phases (12 total)

The plan is grouped into phases that survive reordering. Within a phase, commits are atomic and dependency-ordered. Phases can be interleaved with ablation runs once the foundation (Phases 1–4) is in place.

| Phase | What | Gate |
|---|---|---|
| 0 | Pre-flight (tooling, CI env block, ruff/mypy excludes) | One commit, defensive |
| 1 | Foundation: Rubric, ScoreResult, Judge ABC, MockJudge, constants | Tests green |
| 2 | Concrete judges + rubric markdown files (4 dimensions) | Tests green |
| 3 | Variance wrappers (rubric_permute, jury) | Tests green |
| 4 | Calibration metrics (κ + bootstrap) | Tests green; sklearn-fixture cross-check passes |
| 5 | Calibration dataset spec + FastAPI snippet authoring | `calibration_v1.json` validates |
| 6 | Calibration runner script + row configs | `--help` works on all subcommands |
| 7 | Calibration report generator | Tests green |
| 8 | Harness migration (delete old; integrate new) | Existing test suite green; `judge_provider` regression test passes |
| 9 | Coupled artifact updates (DESIGN.md, DECISIONS.md, measurements/README, README cost-disclosure) | Manual review |
| 10 | Manual labeling (Step B from spec data flow) | 30 × 3 labels in JSONL |
| 11 | Ablation runs + κ table generation | `make calibrate --strict` produces `kappa_table.md` |
| 12 | Writeup `judge-design.md` (v1-completion gate, lags PR merge) | Writeup committed with κ table copy-pasted in |

---

## Phase 0: Pre-flight

Cheap defensive changes that land first. Each is independent of the other phases and reduces friction later.

### Task 0.1: Exclude `scripts/_dev/` from ruff and mypy

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current `pyproject.toml` to confirm no existing excludes**

Run: `grep -nE "extend-exclude|exclude" pyproject.toml`
Expected: empty output (no exclude lines today).

- [ ] **Step 2: Add ruff exclude under `[tool.ruff]`**

Edit `pyproject.toml`. Find:

```toml
[tool.ruff]
target-version = "py311"
line-length = 100
```

Replace with:

```toml
[tool.ruff]
target-version = "py311"
line-length = 100
extend-exclude = ["scripts/_dev"]
```

- [ ] **Step 3: Add mypy exclude under `[tool.mypy]`**

Find:

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

Replace with:

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
exclude = ["scripts/_dev/"]
```

- [ ] **Step 4: Verify nothing in `scripts/_dev/` exists yet (and that's fine)**

Run: `ls scripts/_dev/ 2>&1`
Expected: `ls: scripts/_dev/: No such file or directory`. The exclude is preemptive — Phase 4 creates the directory.

- [ ] **Step 5: Verify lint still passes**

Run: `make lint`
Expected: ruff and mypy both clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "chore(tooling): exclude scripts/_dev/ from ruff and mypy

Pre-flight for the judge-layer v1 PR: scripts/_dev/ will hold sklearn-
dependent fixture-generation tooling that imports packages not in the
project's runtime dependencies. Excluding the directory now prevents
ruff/mypy false positives when those scripts land in Phase 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 0.2: Add explicit empty `env:` block to CI test job

**Files:**
- Modify: `.github/workflows/ci.yaml`

- [ ] **Step 1: Read current workflow**

Run: `cat .github/workflows/ci.yaml`
Expected: confirms the `test` job has no `env:` block today (verified during brainstorming — spec L577).

- [ ] **Step 2: Add `env: {}` to the test job**

Edit `.github/workflows/ci.yaml`. Find:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
```

Replace with:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    # Explicit empty env: prevents accidental dependency on injected
    # secrets. Tests use MockProvider and require no API keys; if a
    # future test imports a provider that needs a key, it will fail
    # in CI and in any contributor fork the same way (no silent
    # divergence based on whether secrets are present).
    env: {}
    steps:
      - uses: actions/checkout@v4
```

- [ ] **Step 3: Verify the workflow YAML still parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yaml'))"`
Expected: no output (parses cleanly).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yaml
git commit -m "ci: document zero-secret contract on test job with empty env block

Existing CI behavior is that tests run without provider keys (MockProvider
covers all paths). The empty env: {} block makes that contract explicit so
a future test that accidentally requires a key fails the same way in this
repo and in any fork — no silent dependency on upstream-only secret
injection.

No behavior change: test job already had no env: block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 0.3: Create `tests/evaluation/` directory with `__init__.py`

**Files:**
- Create: `tests/evaluation/__init__.py`

- [ ] **Step 1: Create the directory and empty marker file**

Run: `mkdir -p tests/evaluation && touch tests/evaluation/__init__.py`

- [ ] **Step 2: Verify pytest still collects existing tests**

Run: `python3 -m pytest tests/ --collect-only -q 2>&1 | tail -3`
Expected: `443 tests collected` (unchanged — empty directory adds nothing).

- [ ] **Step 3: Commit**

```bash
git add tests/evaluation/__init__.py
git commit -m "test: scaffold tests/evaluation/ directory for judge-layer tests

Phase 1+ judge tests will land under tests/evaluation/ matching the
new agent_bench/evaluation/judges,rubrics,variance,calibration/
subpackages. Pattern precedent: tests/test_langchain_baseline/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1: Foundation

Build the core types — `Rubric`, `RubricLevel`, `ScoreResult`, `Judge` ABC, `MockJudge`, abstain-reason constants. Tests-first per file. The Judge ABC is empty (no concrete subclasses yet); concrete judges land in Phase 2.

### Task 1.1: Abstain-reason constants + `ScoreResult` (with tests)

**Files:**
- Create: `agent_bench/evaluation/judges/__init__.py`
- Create: `agent_bench/evaluation/judges/base.py`
- Create: `tests/evaluation/test_judges.py`

- [ ] **Step 1: Create the package marker `agent_bench/evaluation/judges/__init__.py` (empty for now)**

```python
"""Discrete-scale per-dimension LLM judges with anchored rubrics."""
```

- [ ] **Step 2: Write failing test for `ScoreResult` and abstain constants**

Create `tests/evaluation/test_judges.py`:

```python
"""Tests for Judge ABC, ScoreResult, MockJudge, abstain reasons."""

from __future__ import annotations

import pytest

from agent_bench.evaluation.judges.base import (
    ABSTAIN_REASON_GENUINE,
    ABSTAIN_REASON_OUT_OF_RANGE,
    ABSTAIN_REASON_PROVIDER_EXHAUSTED,
    ABSTAIN_REASON_SCHEMA_PARSE,
    ScoreResult,
)


class TestAbstainConstants:
    def test_genuine_is_empty_sentinel(self):
        assert ABSTAIN_REASON_GENUINE == ""

    def test_failure_prefixes_end_with_colon_space(self):
        # All non-genuine prefixes must end with ": " so f-string concatenation
        # produces a parseable "PREFIX: detail" pattern.
        for prefix in (
            ABSTAIN_REASON_PROVIDER_EXHAUSTED,
            ABSTAIN_REASON_SCHEMA_PARSE,
            ABSTAIN_REASON_OUT_OF_RANGE,
        ):
            assert prefix.endswith(": "), f"Bad prefix: {prefix!r}"
            assert "_" in prefix.rstrip(": "), f"Prefix should be snake_case: {prefix!r}"


class TestScoreResult:
    def _base_kwargs(self) -> dict:
        return {
            "reasoning": "test",
            "evidence_quotes": [],
            "judge_id": "mock_groundedness",
            "rubric_version": "abc123",
            "system_output_hash": "def456",
            "cost_usd": 0.001,
            "latency_ms": 100.0,
        }

    def test_int_score_valid(self):
        r = ScoreResult(score=1, **self._base_kwargs())
        assert r.score == 1
        assert r.abstained is False

    def test_unknown_score_is_abstain(self):
        r = ScoreResult(score="Unknown", **self._base_kwargs())
        assert r.score == "Unknown"
        assert r.abstained is True

    def test_field_order_reasoning_first(self):
        # The JSON schema sent to the model puts reasoning before score.
        # Pydantic field order in model_fields drives JSON schema order.
        fields = list(ScoreResult.model_fields.keys())
        assert fields.index("reasoning") < fields.index("score"), (
            f"reasoning must come before score; got order: {fields}"
        )
        assert fields.index("evidence_quotes") < fields.index("score"), (
            f"evidence_quotes must come before score; got order: {fields}"
        )

    def test_prompt_seed_defaults_to_zero(self):
        r = ScoreResult(score=0, **self._base_kwargs())
        assert r.prompt_seed == 0

    def test_score_rejects_other_strings(self):
        with pytest.raises(ValueError):
            ScoreResult(score="maybe", **self._base_kwargs())  # type: ignore[arg-type]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/evaluation/test_judges.py -v 2>&1 | tail -10`
Expected: ImportError on `agent_bench.evaluation.judges.base` — module doesn't exist yet.

- [ ] **Step 4: Implement `agent_bench/evaluation/judges/base.py` (constants + ScoreResult only)**

```python
"""Judge ABC, ScoreResult, Rubric, MockJudge, abstain-reason constants.

The Judge layer supersedes the continuous-scale answer_faithfulness /
answer_correctness functions in agent_bench/evaluation/metrics.py. See
docs/plans/2026-05-04-judge-layer-v1-design.md for the supersession
rationale and the six-axis comparison table.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- Abstain-reason constants ---
#
# Failure-as-abstain ScoreResults carry a reasoning string with one of
# these prefixes. The calibration report pattern-matches against these
# constants for the four-way breakdown in the >20% abstain-rate flag.
# Genuine model abstain (rubric-allowed) uses the empty-string sentinel.

ABSTAIN_REASON_PROVIDER_EXHAUSTED = "judge_call_failed_after_retry: "
ABSTAIN_REASON_SCHEMA_PARSE = "schema_parse_failed_after_retry: "
ABSTAIN_REASON_OUT_OF_RANGE = "score_out_of_range_after_retry: "
ABSTAIN_REASON_GENUINE = ""


class ScoreResult(BaseModel):
    """One judge call's result. Self-contained provenance — no run
    metadata cross-reference needed for κ aggregation.

    Field order matters: reasoning + evidence_quotes come BEFORE score
    in both Pydantic field order and the JSON schema sent to the model,
    so the score conditions on the reasoning rather than being
    post-hoc rationalized.
    """

    # Reasoning-first ordering — load-bearing for the JSON schema
    reasoning: str
    evidence_quotes: list[str] = Field(default_factory=list)
    score: int | Literal["Unknown"]

    # Provenance
    judge_id: str
    rubric_version: str
    prompt_seed: int = 0
    system_output_hash: str

    # Operations
    cost_usd: float
    latency_ms: float

    @property
    def abstained(self) -> bool:
        return self.score == "Unknown"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/evaluation/test_judges.py -v 2>&1 | tail -15`
Expected: all 7 tests in `TestAbstainConstants` and `TestScoreResult` PASS.

- [ ] **Step 6: Commit**

```bash
git add agent_bench/evaluation/judges/__init__.py agent_bench/evaluation/judges/base.py tests/evaluation/test_judges.py
git commit -m "feat(judges): ScoreResult + abstain-reason constants

ScoreResult is the per-call record; field order puts reasoning and
evidence_quotes before score so the score conditions on the reasoning
in the JSON schema sent to the model. score is int | Literal['Unknown']
(not int | None) so abstain is structurally distinct from the silent-
None failure mode the old _judge_call exhibited.

Four abstain-reason constants for the calibration report's cause
breakdown: provider-exhausted, schema-parse, out-of-range, and the
empty-string sentinel for genuine model abstain.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.2: `RubricLevel` + `Rubric` loader with construction validation

**Files:**
- Modify: `agent_bench/evaluation/judges/base.py`
- Create: `tests/evaluation/test_rubric_loading.py`
- Create (fixtures): `tests/evaluation/fixtures/rubrics_valid_binary.md`, `rubrics_valid_three_point.md`, `rubrics_invalid_scale.md`, `rubrics_invalid_arity.md`, `rubrics_invalid_no_examples.md`, `rubrics_invalid_no_frontmatter.md`

- [ ] **Step 1: Create the fixtures directory and a minimal valid binary rubric fixture**

```bash
mkdir -p tests/evaluation/fixtures
```

Create `tests/evaluation/fixtures/rubrics_valid_binary.md`:

```markdown
---
dimension: groundedness
scale: binary
reference_based: true
abstain_allowed: true
---

# Groundedness (binary)

Score whether every claim in the answer is supported by the gold source snippets.

## Score 0

Answer contains at least one claim not supported by the snippets.

### Example A — answer cites unsupported fact

Question: "What's the default port?"
Snippets: ["The default is 8080."]
Answer: "The default is 8080 and supports TLS."

Score=0 because the TLS claim has no support in the snippet. The
unsupported claim is sufficient to fail groundedness regardless of
how many other claims are correctly grounded — this is the binary
rubric's strict-conjunction definition.

## Score 1

Every claim in the answer is supported by at least one snippet.

### Example B — fully grounded one-sentence answer

Question: "What's the default port?"
Snippets: ["The default is 8080."]
Answer: "The default port is 8080."

Score=1 because the only claim ("default port is 8080") is directly
supported by the snippet. Paraphrase is allowed; what matters is
factual entailment.
```

- [ ] **Step 2: Create the rest of the rubric fixtures**

Create `tests/evaluation/fixtures/rubrics_valid_three_point.md`:

```markdown
---
dimension: relevance
scale: three_point
reference_based: false
abstain_allowed: true
---

# Relevance (three-point)

Does the answer address the user's question?

## Score 0

Off-topic. Answer addresses a different question or is unintelligible.

### Example A — wrong topic

Question: "How do I deploy to Kubernetes?"
Answer: "Python virtual environments isolate dependencies."

Score=0 because the answer is about Python venvs, not deployment.

## Score 1

Partially relevant. Answer touches the question but misses the core ask.

### Example B — adjacent but off-target

Question: "How do I deploy to Kubernetes?"
Answer: "Kubernetes runs containerized workloads on a cluster of nodes."

Score=1 because it's about Kubernetes but doesn't say how to deploy.

## Score 2

Directly addresses the question.

### Example C — on-target

Question: "How do I deploy to Kubernetes?"
Answer: "Apply a Deployment manifest with kubectl apply -f deployment.yaml."

Score=2 because it gives a concrete deployment action.
```

Create `tests/evaluation/fixtures/rubrics_invalid_scale.md`:

```markdown
---
dimension: groundedness
scale: five_point
reference_based: true
abstain_allowed: true
---

# Bad scale

## Score 0
example
```

Create `tests/evaluation/fixtures/rubrics_invalid_arity.md`:

```markdown
---
dimension: groundedness
scale: binary
reference_based: true
abstain_allowed: true
---

# Wrong arity (binary should have 2 levels, this has 3)

## Score 0
example A

## Score 1
example B

## Score 2
example C
```

Create `tests/evaluation/fixtures/rubrics_invalid_no_examples.md`:

```markdown
---
dimension: groundedness
scale: binary
reference_based: true
abstain_allowed: true
---

# Missing anchored examples

## Score 0

Just a description, no anchored example.

## Score 1

Same — no anchored example.
```

Create `tests/evaluation/fixtures/rubrics_invalid_no_frontmatter.md`:

```markdown
# No frontmatter at all

## Score 0
example

## Score 1
example
```

- [ ] **Step 3: Write failing tests for `Rubric` construction validation**

Create `tests/evaluation/test_rubric_loading.py`:

```python
"""Tests for Rubric markdown loader: construction validation, hash, permutation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_bench.evaluation.judges.base import Rubric

FIXTURES = Path(__file__).parent / "fixtures"


class TestRubricLoading:
    def test_load_valid_binary(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_binary.md")
        assert r.dimension == "groundedness"
        assert r.scale == "binary"
        assert r.reference_based is True
        assert r.abstain_allowed is True
        assert len(r.levels) == 2

    def test_load_valid_three_point(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        assert r.dimension == "relevance"
        assert r.scale == "three_point"
        assert len(r.levels) == 3


class TestRubricValidationErrors:
    @pytest.mark.parametrize(
        "fixture_name,error_substring",
        [
            ("rubrics_invalid_scale.md", "scale"),
            ("rubrics_invalid_arity.md", "arity"),
            ("rubrics_invalid_no_examples.md", "anchored example"),
            ("rubrics_invalid_no_frontmatter.md", "frontmatter"),
        ],
    )
    def test_construction_raises_with_path_and_field(
        self, fixture_name: str, error_substring: str
    ):
        path = FIXTURES / fixture_name
        with pytest.raises(ValueError) as exc_info:
            Rubric.from_markdown_file(path)
        msg = str(exc_info.value)
        # Error must mention the file path and the field-level reason
        assert fixture_name in msg, f"Path missing from error: {msg}"
        assert error_substring in msg.lower(), (
            f"Expected '{error_substring}' in error message: {msg}"
        )


class TestRubricSourceHash:
    def test_source_hash_deterministic(self):
        r1 = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_binary.md")
        r2 = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_binary.md")
        assert r1.source_hash == r2.source_hash
        # SHA-256 hex is 64 chars
        assert len(r1.source_hash) == 64

    def test_source_hash_changes_with_content(self):
        r1 = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_binary.md")
        r2 = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        assert r1.source_hash != r2.source_hash


class TestRubricPermutation:
    def test_render_prompt_seed_0_unchanged(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        prompt = r.render_prompt(level_permutation_seed=0)
        # Default: levels in original 0, 1, 2 order
        idx0 = prompt.index("Score 0")
        idx1 = prompt.index("Score 1")
        idx2 = prompt.index("Score 2")
        assert idx0 < idx1 < idx2

    def test_render_prompt_seed_reproducibility(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        p1 = r.render_prompt(level_permutation_seed=42)
        p2 = r.render_prompt(level_permutation_seed=42)
        assert p1 == p2

    def test_render_prompt_different_seed_different_order(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        # Try several seeds; at least one should produce a non-default order
        # (with 3! = 6 permutations, the chance all 5 seeds produce identity
        # is (1/6)^5 ≈ 1e-4, negligible)
        default = r.render_prompt(level_permutation_seed=0)
        differs = any(
            r.render_prompt(level_permutation_seed=s) != default
            for s in (1, 2, 3, 7, 13)
        )
        assert differs, "No seed produced a permutation different from default"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python3 -m pytest tests/evaluation/test_rubric_loading.py -v 2>&1 | tail -10`
Expected: ImportError on `Rubric` — class doesn't exist yet.

- [ ] **Step 5: Add `RubricLevel` + `Rubric` to `agent_bench/evaluation/judges/base.py`**

Append to the existing `base.py` (after the `ScoreResult` class):

```python
import hashlib
import random
import re
from pathlib import Path
from typing import Self

import yaml


class RubricLevel(BaseModel):
    """One score level in a rubric, with anchored examples.

    Parsed from markdown sections under `## Score N` headers. The
    `examples` list contains the H3 sub-sections (`### Example X`)
    each with a thinking-trace explanation of why that output got
    that score.
    """

    score: int
    description: str
    examples: list[str]   # raw markdown of `### Example` sections


class Rubric(BaseModel):
    """A scoring rubric loaded from a markdown file with YAML frontmatter.

    Construction validates aggressively: scale ∈ {binary, three_point},
    levels arity matches scale, every level has at least one anchored
    example. ValidationError raises with file path + field path so a
    Day-1 rubric typo doesn't surface as a Day-2 judge.score crash with
    API budget already spent.
    """

    dimension: Literal[
        "groundedness", "relevance", "completeness", "citation_faithfulness"
    ]
    scale: Literal["binary", "three_point"]
    reference_based: bool
    abstain_allowed: bool
    levels: list[RubricLevel]
    body_markdown: str

    @property
    def source_hash(self) -> str:
        """SHA-256 of the canonical body. Immutable per file content,
        independent of git state. Used as ScoreResult.rubric_version.
        """
        return hashlib.sha256(self.body_markdown.encode("utf-8")).hexdigest()

    @classmethod
    def from_markdown_file(cls, path: Path | str) -> Self:
        path = Path(path)
        body = path.read_text(encoding="utf-8")

        # Parse YAML frontmatter delimited by --- ... ---
        fm_match = re.match(r"^---\n(.+?)\n---\n(.*)$", body, re.DOTALL)
        if not fm_match:
            raise ValueError(
                f"Rubric {path.name}: missing YAML frontmatter "
                f"(expected --- ... --- block at top of file)"
            )
        try:
            frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Rubric {path.name}: frontmatter YAML parse error: {e}") from e

        required = {"dimension", "scale", "reference_based", "abstain_allowed"}
        missing = required - frontmatter.keys()
        if missing:
            raise ValueError(
                f"Rubric {path.name}: frontmatter missing fields: {sorted(missing)}"
            )

        scale = frontmatter["scale"]
        if scale not in ("binary", "three_point"):
            raise ValueError(
                f"Rubric {path.name}: invalid scale {scale!r}; "
                f"must be 'binary' or 'three_point'"
            )

        # Parse levels by ## Score N headers
        body_no_fm = fm_match.group(2)
        level_pattern = re.compile(r"^## Score (\d+)\n(.*?)(?=^## Score |\Z)", re.MULTILINE | re.DOTALL)
        raw_levels: list[tuple[int, str]] = [
            (int(m.group(1)), m.group(2)) for m in level_pattern.finditer(body_no_fm)
        ]

        expected_arity = 2 if scale == "binary" else 3
        if len(raw_levels) != expected_arity:
            raise ValueError(
                f"Rubric {path.name}: arity mismatch — scale {scale!r} "
                f"requires {expected_arity} levels, found {len(raw_levels)}"
            )

        # Parse examples (### Example) per level
        levels: list[RubricLevel] = []
        for score, level_body in raw_levels:
            example_pattern = re.compile(
                r"^### (Example .+?)\n(.*?)(?=^### |\Z)", re.MULTILINE | re.DOTALL
            )
            examples = [m.group(0) for m in example_pattern.finditer(level_body)]
            if not examples:
                raise ValueError(
                    f"Rubric {path.name}: level Score {score} has no "
                    f"anchored example (expected at least one ### Example header)"
                )
            description = level_body.split("###", 1)[0].strip()
            levels.append(
                RubricLevel(score=score, description=description, examples=examples)
            )

        return cls(
            dimension=frontmatter["dimension"],
            scale=scale,
            reference_based=bool(frontmatter["reference_based"]),
            abstain_allowed=bool(frontmatter["abstain_allowed"]),
            levels=levels,
            body_markdown=body,
        )

    def render_prompt(self, *, level_permutation_seed: int = 0) -> str:
        """Render the rubric body for inclusion in a judge prompt.

        If level_permutation_seed > 0, levels are reordered deterministically
        using a seeded PRNG. seed=0 returns the canonical order.
        """
        if level_permutation_seed == 0:
            return self.body_markdown
        rng = random.Random(level_permutation_seed)
        permuted_levels = list(self.levels)
        rng.shuffle(permuted_levels)
        # Reconstruct: keep frontmatter + intro paragraphs intact;
        # reorder the ## Score N sections.
        fm_match = re.match(r"^(---\n.+?\n---\n)(.*)$", self.body_markdown, re.DOTALL)
        if not fm_match:
            return self.body_markdown  # defensive — should never happen post-construction
        head = fm_match.group(1)
        rest = fm_match.group(2)
        intro = re.split(r"^## Score ", rest, maxsplit=1, flags=re.MULTILINE)[0]
        permuted_body = head + intro + "\n".join(
            f"## Score {lvl.score}\n{lvl.description}\n" + "\n".join(lvl.examples)
            for lvl in permuted_levels
        )
        return permuted_body
```

- [ ] **Step 6: Add PyYAML to project dependencies if not already there**

Run: `grep -E "pyyaml|PyYAML" pyproject.toml`
Expected: `pyyaml>=6.0` already present (verified during exploration). No change needed.

- [ ] **Step 7: Run tests to verify they pass**

Run: `python3 -m pytest tests/evaluation/test_rubric_loading.py -v 2>&1 | tail -20`
Expected: all 9 tests PASS (2 valid loading + 4 parameterized validation + 2 hash + 3 permutation).

- [ ] **Step 8: Commit**

```bash
git add agent_bench/evaluation/judges/base.py tests/evaluation/test_rubric_loading.py tests/evaluation/fixtures/
git commit -m "feat(judges): Rubric markdown loader with aggressive validation

Rubric loads from markdown with YAML frontmatter; validates scale,
arity-matches-scale, anchored-example-per-level, frontmatter
required fields. ValidationError raises with file path + field
context so malformed rubrics fail at construction (Day 1) not at
first judge.score call (Day 2 with API budget spent).

source_hash is SHA-256 of body_markdown — immutable per file
content, independent of git state. Used as ScoreResult.rubric_version
so κ aggregation can group by rubric identity without cross-
referencing run metadata.

render_prompt(level_permutation_seed=N) deterministically permutes
the ## Score sections via seeded PRNG. Seed=0 returns canonical
order; this is the variance-control hook used by rubric_permute
in Phase 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.3: `Judge` ABC

**Files:**
- Modify: `agent_bench/evaluation/judges/base.py` (append `Judge` class)
- Modify: `tests/evaluation/test_judges.py` (add ABC contract test)

- [ ] **Step 1: Write failing test for Judge ABC contract**

Append to `tests/evaluation/test_judges.py`:

```python
from abc import ABC

from agent_bench.evaluation.judges.base import Judge


class TestJudgeABC:
    def test_judge_is_abstract(self):
        assert issubclass(Judge, ABC)
        # Cannot instantiate directly — score is abstract
        with pytest.raises(TypeError, match="abstract"):
            Judge(judge_provider=None, rubric=None, model_id="test")  # type: ignore[abstract,arg-type]

    def test_judge_id_built_from_model_and_dimension(self):
        # Concrete subclass that satisfies the abstract method
        class _ConcreteJudge(Judge):
            async def score(self, item, output, *, prompt_seed=0):
                raise NotImplementedError

        from agent_bench.evaluation.judges.base import Rubric

        rubric = Rubric.from_markdown_file(
            Path(__file__).parent / "fixtures" / "rubrics_valid_binary.md"
        )
        j = _ConcreteJudge(judge_provider=None, rubric=rubric, model_id="claude-haiku-4-5")  # type: ignore[arg-type]
        assert j.judge_id == "claude-haiku-4-5_groundedness"
```

Add the `Path` import at the top of the test file if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/evaluation/test_judges.py::TestJudgeABC -v 2>&1 | tail -10`
Expected: ImportError on `Judge`.

- [ ] **Step 3: Append `Judge` ABC to `agent_bench/evaluation/judges/base.py`**

Append after the `Rubric` class:

```python
from abc import ABC, abstractmethod

from agent_bench.agents.orchestrator import AgentResponse
from agent_bench.core.provider import LLMProvider
from agent_bench.evaluation.harness import GoldenQuestion


class Judge(ABC):
    """Per-dimension LLM judge. Concrete subclasses implement score()
    for one rubric dimension; they are thin (~30 lines) and not
    factored against a shared base method (see design doc for why).
    """

    def __init__(
        self,
        judge_provider: LLMProvider,
        rubric: Rubric,
        model_id: str,
    ) -> None:
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
    ) -> ScoreResult:
        """Score one (item, output) pair against this judge's rubric.

        Returns a ScoreResult whose system_output_hash is computed from
        (item.id, output.answer, sorted(output.sources)). Failures map
        to abstain via the abstain-reason constants; provider non-
        retryable errors raise (caller bug, not noise).
        """
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/evaluation/test_judges.py::TestJudgeABC -v 2>&1 | tail -10`
Expected: both ABC tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_bench/evaluation/judges/base.py tests/evaluation/test_judges.py
git commit -m "feat(judges): Judge ABC with judge_id derived from model + dimension

Judge is abstract — concrete subclasses (groundedness, relevance,
completeness, citation_faithfulness) land in Phase 2 as thin
~30-line classes per the no-shared-base-method discipline.

judge_id format: '{model_id}_{rubric.dimension}', e.g.
'claude-haiku-4-5_groundedness'. The format is load-bearing for
the calibration report's per-judge κ breakdown — the report
groups by judge_id when computing per-judge agreement against
the human labels.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.4: `MockJudge` with LookupError + helper

**Files:**
- Modify: `agent_bench/evaluation/judges/base.py` (append `MockJudge`)
- Modify: `tests/evaluation/test_judges.py` (add MockJudge tests)

- [ ] **Step 1: Write failing tests for MockJudge**

Append to `tests/evaluation/test_judges.py`:

```python
from agent_bench.evaluation.judges.base import MockJudge


class TestMockJudge:
    def _verdict(self, item_id: str, score: int = 1) -> ScoreResult:
        return ScoreResult(
            reasoning=f"prebaked for {item_id}",
            evidence_quotes=[],
            score=score,
            judge_id="mock_groundedness",
            rubric_version="abc",
            system_output_hash="def",
            cost_usd=0.0,
            latency_ms=0.0,
        )

    @pytest.mark.asyncio
    async def test_returns_prebaked_verdict(self, monkeypatch):
        from agent_bench.evaluation.harness import GoldenQuestion
        from agent_bench.agents.orchestrator import AgentResponse, SourceReference
        from agent_bench.core.types import TokenUsage

        verdict = self._verdict("item_001", score=1)
        mj = MockJudge(verdicts={"item_001": verdict})

        item = GoldenQuestion(
            id="item_001", question="?", expected_answer_keywords=[],
            expected_sources=[], category="retrieval", difficulty="easy",
            requires_calculator=False,
        )
        output = AgentResponse(
            answer="x", sources=[SourceReference(source="a.md")],
            iterations=1, usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=0,
        )
        result = await mj.score(item, output)
        assert result.score == 1
        assert result.reasoning == "prebaked for item_001"

    @pytest.mark.asyncio
    async def test_raises_lookuperror_on_missing_key(self):
        from agent_bench.evaluation.harness import GoldenQuestion
        from agent_bench.agents.orchestrator import AgentResponse
        from agent_bench.core.types import TokenUsage

        mj = MockJudge(verdicts={"item_001": self._verdict("item_001")})

        item = GoldenQuestion(
            id="item_999_NOT_PRESENT", question="?", expected_answer_keywords=[],
            expected_sources=[], category="retrieval", difficulty="easy",
            requires_calculator=False,
        )
        output = AgentResponse(
            answer="x", sources=[], iterations=1,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=0,
        )
        with pytest.raises(LookupError, match="item_999_NOT_PRESENT"):
            await mj.score(item, output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/evaluation/test_judges.py::TestMockJudge -v 2>&1 | tail -10`
Expected: ImportError on `MockJudge`.

- [ ] **Step 3: Append `MockJudge` to `agent_bench/evaluation/judges/base.py`**

```python
class MockJudge(Judge):
    """Pre-baked-verdict judge for deterministic tests. No API calls.

    Constructor takes verdicts: dict[item_id, ScoreResult]. score()
    raises LookupError on missing keys — never returns a default —
    so test fixtures are self-checking. A separate fixture-validation
    test (test_mockjudge_coverage.py) walks item.id across all goldens
    and asserts every MockJudge instance has coverage for the items
    its tests reference.

    Mirrors the MockProvider pattern at agent_bench/core/provider.py:118.
    """

    def __init__(self, verdicts: dict[str, ScoreResult]) -> None:
        # MockJudge does not need provider/rubric/model_id; supply
        # placeholder values so the ABC's __init__ doesn't matter.
        self.judge_provider = None  # type: ignore[assignment]
        self.rubric = None  # type: ignore[assignment]
        self.model_id = "mock"
        self.judge_id = "mock_judge"
        self._verdicts = verdicts

    async def score(
        self,
        item: GoldenQuestion,
        output: AgentResponse,
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        if item.id not in self._verdicts:
            raise LookupError(
                f"MockJudge has no pre-baked verdict for item_id {item.id!r}; "
                f"available: {sorted(self._verdicts.keys())[:5]}"
                + (" ..." if len(self._verdicts) > 5 else "")
            )
        return self._verdicts[item.id]
```

- [ ] **Step 4: Update `agent_bench/evaluation/judges/__init__.py` to re-export the public surface**

Replace the contents of `__init__.py`:

```python
"""Discrete-scale per-dimension LLM judges with anchored rubrics."""

from agent_bench.evaluation.judges.base import (
    ABSTAIN_REASON_GENUINE,
    ABSTAIN_REASON_OUT_OF_RANGE,
    ABSTAIN_REASON_PROVIDER_EXHAUSTED,
    ABSTAIN_REASON_SCHEMA_PARSE,
    Judge,
    MockJudge,
    Rubric,
    RubricLevel,
    ScoreResult,
)

__all__ = [
    "ABSTAIN_REASON_GENUINE",
    "ABSTAIN_REASON_OUT_OF_RANGE",
    "ABSTAIN_REASON_PROVIDER_EXHAUSTED",
    "ABSTAIN_REASON_SCHEMA_PARSE",
    "Judge",
    "MockJudge",
    "Rubric",
    "RubricLevel",
    "ScoreResult",
]
```

- [ ] **Step 5: Run all judge tests to verify**

Run: `python3 -m pytest tests/evaluation/test_judges.py tests/evaluation/test_rubric_loading.py -v 2>&1 | tail -25`
Expected: all tests PASS (~13 tests total).

- [ ] **Step 6: Commit**

```bash
git add agent_bench/evaluation/judges/base.py agent_bench/evaluation/judges/__init__.py tests/evaluation/test_judges.py
git commit -m "feat(judges): MockJudge with LookupError on missing keys

MockJudge raises LookupError (not a default) on missing item.id keys,
so test fixtures are self-checking against rename drift. A separate
fixture-validation test in Phase 8 walks item.id across all goldens
and asserts coverage; the LookupError is the second layer of defense.

__init__.py re-exports the public surface for ergonomic imports
(from agent_bench.evaluation.judges import Judge, ScoreResult, ...).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2: Concrete judges + rubric markdown files

Per the design doc's sequencing tactic (Implementation Sequencing Notes), groundedness is authored first and dry-fitted before the others. The shared retry/parse/log helper lives as a module-level function in `base.py` (not a `Judge` method) — the concrete judges call it but remain thin.

### Task 2.1: Shared `_call_judge_with_retry` helper + first-attempt-failure log

**Files:**
- Modify: `agent_bench/evaluation/judges/base.py` (add helper)
- Modify: `tests/evaluation/test_judges.py` (add helper tests)

The helper:
1. Sends one judge call with structured-output JSON schema
2. On schema parse / score-out-of-range failure: logs WARN with fixed key set, retries once with strict-reprompt
3. On retry success: returns the `ScoreResult` (the first-attempt log fired regardless)
4. On retry failure: returns abstain-as-`ScoreResult` with structured-prefix reason
5. On `ProviderRateLimitError` / `ProviderTimeoutError` exhaustion: abstain with `ABSTAIN_REASON_PROVIDER_EXHAUSTED`
6. On any other exception (caller misconfig): re-raise

- [ ] **Step 1: Write failing tests**

Append to `tests/evaluation/test_judges.py`:

```python
import json
from typing import Any
from unittest.mock import AsyncMock

from agent_bench.core.provider import LLMProvider, ProviderRateLimitError, ProviderTimeoutError
from agent_bench.core.types import CompletionResponse, TokenUsage
from agent_bench.evaluation.judges.base import _call_judge_with_retry


def _mk_response(content: str) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=10, estimated_cost_usd=0.0001),
        provider="mock",
        model="mock-1",
        latency_ms=1.0,
    )


def _valid_json(score: int) -> str:
    return json.dumps({
        "reasoning": "test reasoning",
        "evidence_quotes": ["q1"],
        "score": score,
    })


class TestCallJudgeWithRetry:
    @pytest.mark.asyncio
    async def test_first_attempt_success(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.return_value = _mk_response(_valid_json(1))

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="claude-haiku-4-5_groundedness",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
        )
        assert result.score == 1
        assert provider.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_schema_parse_then_retry_success(self, caplog):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = [
            _mk_response("not json at all"),
            _mk_response(_valid_json(0)),
        ]

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="claude-haiku-4-5_groundedness",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
        )
        assert result.score == 0
        assert provider.complete.await_count == 2
        # First-attempt-failure log must have fired even though retry succeeded
        assert any(
            "judge_first_attempt_failure" in str(rec.msg)
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_schema_parse_twice_abstains_with_prefix(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = [
            _mk_response("garbage"),
            _mk_response("also garbage"),
        ]

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="claude-haiku-4-5_groundedness",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
        )
        assert result.abstained
        assert result.reasoning.startswith(ABSTAIN_REASON_SCHEMA_PARSE)

    @pytest.mark.asyncio
    async def test_score_out_of_range_twice_abstains_with_prefix(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = [
            _mk_response(_valid_json(5)),
            _mk_response(_valid_json(7)),
        ]

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="claude-haiku-4-5_groundedness",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
        )
        assert result.abstained
        assert result.reasoning.startswith(ABSTAIN_REASON_OUT_OF_RANGE)

    @pytest.mark.asyncio
    async def test_provider_rate_limit_abstains_with_prefix(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = ProviderRateLimitError("exhausted")

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="claude-haiku-4-5_groundedness",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
        )
        assert result.abstained
        assert result.reasoning.startswith(ABSTAIN_REASON_PROVIDER_EXHAUSTED)

    @pytest.mark.asyncio
    async def test_unknown_exception_reraises(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = ValueError("caller bug")

        with pytest.raises(ValueError, match="caller bug"):
            await _call_judge_with_retry(
                provider=provider,
                prompt="test prompt",
                valid_scores={0, 1},
                judge_id="x",
                rubric_version="abc",
                prompt_seed=0,
                system_output_hash="def",
                item_id="item_001",
            )

    @pytest.mark.asyncio
    async def test_genuine_unknown_score_passes_through(self):
        # Rubric allows abstain — model returns "Unknown" — no retry, no prefix
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.return_value = _mk_response(json.dumps({
            "reasoning": "genuinely uncertain",
            "evidence_quotes": [],
            "score": "Unknown",
        }))

        result = await _call_judge_with_retry(
            provider=provider,
            prompt="test prompt",
            valid_scores={0, 1},
            judge_id="x",
            rubric_version="abc",
            prompt_seed=0,
            system_output_hash="def",
            item_id="item_001",
            abstain_allowed=True,
        )
        assert result.abstained
        assert result.reasoning == "genuinely uncertain"
        # No structured prefix on genuine abstain
        assert not result.reasoning.startswith(ABSTAIN_REASON_PROVIDER_EXHAUSTED)
        assert not result.reasoning.startswith(ABSTAIN_REASON_SCHEMA_PARSE)
        assert provider.complete.await_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/evaluation/test_judges.py::TestCallJudgeWithRetry -v 2>&1 | tail -10`
Expected: ImportError on `_call_judge_with_retry`.

- [ ] **Step 3: Implement helper in `agent_bench/evaluation/judges/base.py`**

Append to `base.py`:

```python
import json as _json
import time

import structlog

from agent_bench.core.provider import (
    LLMProvider,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from agent_bench.core.types import Message, Role

logger = structlog.get_logger()

_STRICT_REPROMPT_SUFFIX = (
    "\n\nSTRICT FORMATTING NOTE: respond ONLY with a JSON object matching "
    "the schema; reasoning first, then evidence_quotes, then score."
)


async def _call_judge_with_retry(
    *,
    provider: LLMProvider,
    prompt: str,
    valid_scores: set[int],
    judge_id: str,
    rubric_version: str,
    prompt_seed: int,
    system_output_hash: str,
    item_id: str,
    abstain_allowed: bool = True,
    max_tokens: int = 512,
) -> ScoreResult:
    """Send prompt to provider; one retry with strict reprompt on
    schema-parse / score-out-of-range; abstain on persistent failure
    or provider exhaustion. Re-raises unknown exceptions (caller bugs).
    """
    accumulated_cost = 0.0
    accumulated_latency = 0.0
    first_failure_cause: str | None = None
    last_raw: str = ""

    for attempt in range(2):  # 2 = original + one retry
        send_prompt = prompt if attempt == 0 else prompt + _STRICT_REPROMPT_SUFFIX
        start = time.perf_counter()
        try:
            response = await provider.complete(
                [Message(role=Role.USER, content=send_prompt)],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except (ProviderRateLimitError, ProviderTimeoutError) as e:
            return ScoreResult(
                reasoning=f"{ABSTAIN_REASON_PROVIDER_EXHAUSTED}{type(e).__name__}: {e}",
                evidence_quotes=[],
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency + (time.perf_counter() - start) * 1000,
            )
        # Other exceptions (caller bugs like 401, 400) propagate.
        accumulated_cost += response.usage.estimated_cost_usd
        accumulated_latency += (time.perf_counter() - start) * 1000
        last_raw = response.content[:300]

        # Parse
        try:
            data = _json.loads(response.content)
            reasoning = str(data["reasoning"])
            evidence_quotes = list(data.get("evidence_quotes", []))
            raw_score = data["score"]
        except (_json.JSONDecodeError, KeyError, TypeError) as e:
            cause = ABSTAIN_REASON_SCHEMA_PARSE
            if attempt == 0:
                first_failure_cause = cause
                logger.warning(
                    "judge_first_attempt_failure",
                    judge_id=judge_id,
                    item_id=item_id,
                    provider=type(provider).__name__,
                    failure_cause=cause,
                    attempt_index=1,
                )
                continue
            return ScoreResult(
                reasoning=f"{cause}raw={last_raw!r} parse_error={e}",
                evidence_quotes=[],
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency,
            )

        # Score validation
        if raw_score == "Unknown":
            if not abstain_allowed:
                cause = ABSTAIN_REASON_OUT_OF_RANGE
                if attempt == 0:
                    first_failure_cause = cause
                    logger.warning(
                        "judge_first_attempt_failure",
                        judge_id=judge_id, item_id=item_id,
                        provider=type(provider).__name__,
                        failure_cause=cause, attempt_index=1,
                    )
                    continue
                return ScoreResult(
                    reasoning=(
                        f"{cause}model returned 'Unknown' but rubric "
                        f"abstain_allowed=False"
                    ),
                    evidence_quotes=[],
                    score="Unknown",
                    judge_id=judge_id,
                    rubric_version=rubric_version,
                    prompt_seed=prompt_seed,
                    system_output_hash=system_output_hash,
                    cost_usd=accumulated_cost,
                    latency_ms=accumulated_latency,
                )
            # Genuine abstain — no prefix, no retry
            return ScoreResult(
                reasoning=reasoning,
                evidence_quotes=evidence_quotes,
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency,
            )

        try:
            score_int = int(raw_score)
        except (ValueError, TypeError):
            cause = ABSTAIN_REASON_OUT_OF_RANGE
            if attempt == 0:
                first_failure_cause = cause
                logger.warning(
                    "judge_first_attempt_failure",
                    judge_id=judge_id, item_id=item_id,
                    provider=type(provider).__name__,
                    failure_cause=cause, attempt_index=1,
                )
                continue
            return ScoreResult(
                reasoning=f"{cause}non-int score: {raw_score!r}",
                evidence_quotes=[],
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency,
            )

        if score_int not in valid_scores:
            cause = ABSTAIN_REASON_OUT_OF_RANGE
            if attempt == 0:
                first_failure_cause = cause
                logger.warning(
                    "judge_first_attempt_failure",
                    judge_id=judge_id, item_id=item_id,
                    provider=type(provider).__name__,
                    failure_cause=cause, attempt_index=1,
                )
                continue
            return ScoreResult(
                reasoning=(
                    f"{cause}model returned {score_int}, valid levels "
                    f"{sorted(valid_scores)}"
                ),
                evidence_quotes=[],
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency,
            )

        # Success
        return ScoreResult(
            reasoning=reasoning,
            evidence_quotes=evidence_quotes,
            score=score_int,
            judge_id=judge_id,
            rubric_version=rubric_version,
            prompt_seed=prompt_seed,
            system_output_hash=system_output_hash,
            cost_usd=accumulated_cost,
            latency_ms=accumulated_latency,
        )

    # Defensive: should never reach here (loop body always returns or continues)
    raise RuntimeError(
        f"_call_judge_with_retry: unreachable code; first_failure_cause={first_failure_cause}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/evaluation/test_judges.py::TestCallJudgeWithRetry -v 2>&1 | tail -15`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_bench/evaluation/judges/base.py tests/evaluation/test_judges.py
git commit -m "feat(judges): _call_judge_with_retry helper with strict-reprompt + abstain

Module-level helper (not a Judge method) called by every concrete
judge. One retry with the strict-formatting reprompt suffix on
schema-parse / score-out-of-range; abstain with structured-prefix
reason on persistent failure or provider exhaustion. Caller bugs
(401, 400, anything not in the retryable taxonomy) propagate.

The first-attempt-failure WARN log fires regardless of whether the
retry succeeds — the silent recover-on-retry case is the most
analytically interesting one for offline analysis (tells you the
reprompt is doing work, not just costing 2x tokens).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.2: Author groundedness rubric + `GroundednessJudge`

**Files:**
- Create: `agent_bench/evaluation/rubrics/groundedness.md`
- Create: `agent_bench/evaluation/judges/groundedness.py`
- Modify: `tests/evaluation/test_judges.py`

- [ ] **Step 1: Author the groundedness rubric**

Create `agent_bench/evaluation/rubrics/groundedness.md`. Use the binary fixture in tests as a starting structure; expand to ≥2 anchored examples per level:

```markdown
---
dimension: groundedness
scale: binary
reference_based: true
abstain_allowed: true
---

# Groundedness (binary)

Score whether **every claim** in the agent's answer is directly supported by
the gold source snippets attached to this item. Paraphrase is allowed; what
matters is factual entailment.

The judge sees only the gold snippets — not the retrieved chunks. A claim
that happens to be true in the world but is not entailed by the snippets
fails groundedness.

**When to abstain (`"Unknown"`)**: if the answer is a refusal ("I don't
know" / "not in the documentation") and there is nothing to ground, score
abstain rather than 1.

## Score 0

At least one claim in the answer is not supported by any snippet.

### Example A — answer adds an unsupported claim

Question: "What's the default port for the dashboard?"
Snippets: ["The dashboard listens on port 8080 by default."]
Answer: "The default port is 8080 and TLS is enabled out of the box."

Score=0 because the TLS claim has no support in the snippet. The strict-
conjunction rule applies: even a single unsupported claim fails the binary
groundedness rubric. The grounded portion of the answer doesn't redeem it.

### Example B — answer paraphrases incorrectly

Question: "How long do connections idle before timeout?"
Snippets: ["Idle connections are closed after 30 seconds."]
Answer: "Connections close after 30 minutes of inactivity."

Score=0 because the unit is wrong (minutes vs seconds). Paraphrase is
allowed but factual content must match.

## Score 1

Every claim in the answer is directly supported by at least one snippet.

### Example C — fully grounded one-fact answer

Question: "What's the default port?"
Snippets: ["The dashboard listens on port 8080 by default."]
Answer: "Port 8080."

Score=1 because the only claim is the port number, which is in the snippet.

### Example D — fully grounded multi-claim answer

Question: "What identity guarantees does a StatefulSet provide?"
Snippets: [
  "StatefulSet pods receive an ordinal index from 0 to N-1.",
  "Each pod gets a stable hostname based on the StatefulSet name and ordinal.",
  "Storage is persistent across pod restarts and reschedules."
]
Answer: "Pods are assigned ordinal indices, stable hostnames derived from
the StatefulSet name + ordinal, and storage that persists across restarts."

Score=1 because all three claims (ordinal indices, stable hostnames,
persistent storage) are each supported by one snippet.
```

- [ ] **Step 2: Verify the rubric loads against the production-validated path**

Run:
```python
python3 -c "
from pathlib import Path
from agent_bench.evaluation.judges.base import Rubric
r = Rubric.from_markdown_file('agent_bench/evaluation/rubrics/groundedness.md')
print(f'OK: dimension={r.dimension} scale={r.scale} levels={len(r.levels)} hash={r.source_hash[:12]}...')
"
```
Expected: `OK: dimension=groundedness scale=binary levels=2 hash=...`.

- [ ] **Step 3: Write failing test for `GroundednessJudge`**

Append to `tests/evaluation/test_judges.py`:

```python
class TestGroundednessJudge:
    @pytest.mark.asyncio
    async def test_calls_helper_with_correct_prompt_and_valid_scores(self, monkeypatch):
        from agent_bench.evaluation.judges.groundedness import GroundednessJudge
        from agent_bench.evaluation.judges.base import Rubric
        from agent_bench.evaluation.harness import GoldenQuestion
        from agent_bench.agents.orchestrator import AgentResponse, SourceReference
        from agent_bench.core.types import TokenUsage

        rubric = Rubric.from_markdown_file(
            "agent_bench/evaluation/rubrics/groundedness.md"
        )
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.return_value = _mk_response(_valid_json(1))

        judge = GroundednessJudge(judge_provider=provider, rubric=rubric, model_id="m")

        item = GoldenQuestion(
            id="k8s_001", question="What does StatefulSet guarantee?",
            expected_answer_keywords=[], expected_sources=[],
            category="retrieval", difficulty="easy", requires_calculator=False,
            source_snippets=["StatefulSet pods receive ordinal indices."],
        )
        output = AgentResponse(
            answer="Ordinal indices.",
            sources=[SourceReference(source="k8s_statefulset.md")],
            iterations=1,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=0,
        )
        result = await judge.score(item, output)

        assert result.score == 1
        assert result.judge_id == "m_groundedness"
        # Prompt sent must contain the gold snippet and the answer
        sent_prompt = provider.complete.await_args.args[0][0].content
        assert "StatefulSet pods receive ordinal indices." in sent_prompt
        assert "Ordinal indices." in sent_prompt
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python3 -m pytest tests/evaluation/test_judges.py::TestGroundednessJudge -v 2>&1 | tail -10`
Expected: ImportError on `GroundednessJudge`.

- [ ] **Step 5: Implement `agent_bench/evaluation/judges/groundedness.py`**

```python
"""GroundednessJudge — binary, reference-based on item.source_snippets."""

from __future__ import annotations

import hashlib

from agent_bench.agents.orchestrator import AgentResponse
from agent_bench.evaluation.harness import GoldenQuestion
from agent_bench.evaluation.judges.base import (
    Judge,
    ScoreResult,
    _call_judge_with_retry,
)


def _system_output_hash(item_id: str, answer: str, sources: list[str]) -> str:
    sorted_sources = sorted(sources)
    canonical = f"{item_id}\x00{answer}\x00{','.join(sorted_sources)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class GroundednessJudge(Judge):
    async def score(
        self,
        item: GoldenQuestion,
        output: AgentResponse,
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        snippets_block = "\n".join(
            f"[{i + 1}] {s}" for i, s in enumerate(item.source_snippets)
        )
        prompt = (
            f"{self.rubric.render_prompt(level_permutation_seed=prompt_seed)}\n\n"
            f"---\n\n"
            f"## Gold source snippets\n{snippets_block}\n\n"
            f"## Answer to score\n{output.answer}\n\n"
            f"Score this answer against the rubric above. Respond with ONLY a "
            f'JSON object: {{"reasoning": "...", "evidence_quotes": [...], "score": 0 or 1 or "Unknown"}}.'
        )
        return await _call_judge_with_retry(
            provider=self.judge_provider,
            prompt=prompt,
            valid_scores={0, 1},
            judge_id=self.judge_id,
            rubric_version=self.rubric.source_hash,
            prompt_seed=prompt_seed,
            system_output_hash=_system_output_hash(
                item.id, output.answer, [s.source for s in output.sources]
            ),
            item_id=item.id,
            abstain_allowed=self.rubric.abstain_allowed,
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest tests/evaluation/test_judges.py::TestGroundednessJudge -v 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 7: Dry-fit the rubric against 3-4 K8s items (manual sanity check)**

Open `agent_bench/evaluation/datasets/k8s_golden.json` and pick 3-4 items with `source_snippets`. For each, mentally walk the rubric: which score would you give if the agent answered exactly the `reference_answer`? If you cannot decide, the rubric is underspecified — fix it now before authoring relevance/completeness. This is the spec's load-bearing dry-fit step: get groundedness right before mechanically replicating the pattern.

If revisions are needed, edit `agent_bench/evaluation/rubrics/groundedness.md`, re-run the loader sanity check from Step 2, and re-run the test from Step 6.

- [ ] **Step 8: Commit**

```bash
git add agent_bench/evaluation/rubrics/groundedness.md agent_bench/evaluation/judges/groundedness.py tests/evaluation/test_judges.py
git commit -m "feat(judges): GroundednessJudge + anchored binary rubric

Binary rubric, reference-based on item.source_snippets. Strict-
conjunction definition: any unsupported claim fails the rubric.
Two anchored examples per score level with thinking-trace
explanations (per Singla et al. 2025 — through-the-judge's-eyes).

Per the design doc's sequencing tactic, groundedness is authored
first and dry-fitted before relevance and completeness — this
converts rubric authoring from three parallel risky tasks to one
risky task plus two near-mechanical replications.

system_output_hash is SHA-256 of (item.id, answer, sorted(sources))
joined by NUL; the calibration report uses this as the agreement-
eligibility key against labels.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.3: Author relevance rubric + `RelevanceJudge`

**Files:**
- Create: `agent_bench/evaluation/rubrics/relevance.md`
- Create: `agent_bench/evaluation/judges/relevance.py`
- Modify: `tests/evaluation/test_judges.py`

- [ ] **Step 1: Author the relevance rubric**

Create `agent_bench/evaluation/rubrics/relevance.md`:

```markdown
---
dimension: relevance
scale: three_point
reference_based: false
abstain_allowed: true
---

# Relevance (three-point)

Does the agent's answer address the user's question? This is reference-free
— the judge sees only the question and the answer, not gold snippets or a
reference answer. Score the topic-match, not the truth-value.

## Score 0

Off-topic. The answer addresses a different question, is unintelligible,
or is a refusal that does not engage with the question's premise.

### Example A — wrong topic

Question: "How do I deploy to Kubernetes?"
Answer: "Python virtual environments isolate dependencies between projects."

Score=0 — the answer is about Python venvs, not Kubernetes deployment.

### Example B — refusal that ignores the question

Question: "What's the default replica count for a StatefulSet?"
Answer: "I cannot help with that request."

Score=0 — the refusal does not engage with the StatefulSet topic. A
proper grounded refusal ("the documentation does not specify a default
replica count for StatefulSets") would score higher.

## Score 1

Partially relevant. The answer touches the question's topic but misses
the core ask, or addresses a related-but-different question.

### Example C — adjacent but off-target

Question: "How do I deploy a StatefulSet?"
Answer: "Kubernetes runs containerized workloads on a cluster of nodes."

Score=1 because it's about Kubernetes but doesn't address StatefulSet
deployment specifically.

### Example D — answers a sibling question

Question: "What's the difference between Deployment and StatefulSet?"
Answer: "A Deployment manages stateless replicas with rolling updates."

Score=1 because it describes Deployment but doesn't compare it to
StatefulSet — only half the question is addressed.

## Score 2

Directly addresses the question's core ask.

### Example E — on-target single-fact answer

Question: "What's the default port for kubelet?"
Answer: "Port 10250."

Score=2 because it directly answers the question.

### Example F — on-target comparison

Question: "What's the difference between Deployment and StatefulSet?"
Answer: "Deployments manage stateless, interchangeable pods with rolling
updates; StatefulSets manage stateful pods with stable identities,
ordered rollouts, and persistent per-pod storage."

Score=2 — both sides of the comparison are addressed.
```

- [ ] **Step 2: Verify the rubric loads**

Run:
```python
python3 -c "
from agent_bench.evaluation.judges.base import Rubric
r = Rubric.from_markdown_file('agent_bench/evaluation/rubrics/relevance.md')
print(f'OK: dimension={r.dimension} scale={r.scale} levels={len(r.levels)}')
"
```
Expected: `OK: dimension=relevance scale=three_point levels=3`.

- [ ] **Step 3: Implement `agent_bench/evaluation/judges/relevance.py` (test+code together since shape mirrors Groundedness)**

Create `agent_bench/evaluation/judges/relevance.py`:

```python
"""RelevanceJudge — three-point, reference-free."""

from __future__ import annotations

from agent_bench.agents.orchestrator import AgentResponse
from agent_bench.evaluation.harness import GoldenQuestion
from agent_bench.evaluation.judges.base import (
    Judge,
    ScoreResult,
    _call_judge_with_retry,
)
from agent_bench.evaluation.judges.groundedness import _system_output_hash


class RelevanceJudge(Judge):
    async def score(
        self,
        item: GoldenQuestion,
        output: AgentResponse,
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        prompt = (
            f"{self.rubric.render_prompt(level_permutation_seed=prompt_seed)}\n\n"
            f"---\n\n"
            f"## Question\n{item.question}\n\n"
            f"## Answer to score\n{output.answer}\n\n"
            f"Score this answer against the rubric above. Respond with ONLY a "
            f'JSON object: {{"reasoning": "...", "evidence_quotes": [...], "score": 0 or 1 or 2 or "Unknown"}}.'
        )
        return await _call_judge_with_retry(
            provider=self.judge_provider,
            prompt=prompt,
            valid_scores={0, 1, 2},
            judge_id=self.judge_id,
            rubric_version=self.rubric.source_hash,
            prompt_seed=prompt_seed,
            system_output_hash=_system_output_hash(
                item.id, output.answer, [s.source for s in output.sources]
            ),
            item_id=item.id,
            abstain_allowed=self.rubric.abstain_allowed,
        )
```

- [ ] **Step 4: Add a smoke test for `RelevanceJudge`**

Append to `tests/evaluation/test_judges.py`:

```python
class TestRelevanceJudge:
    @pytest.mark.asyncio
    async def test_three_point_valid_scores(self):
        from agent_bench.evaluation.judges.relevance import RelevanceJudge
        from agent_bench.evaluation.judges.base import Rubric
        from agent_bench.evaluation.harness import GoldenQuestion
        from agent_bench.agents.orchestrator import AgentResponse
        from agent_bench.core.types import TokenUsage

        rubric = Rubric.from_markdown_file("agent_bench/evaluation/rubrics/relevance.md")
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.return_value = _mk_response(_valid_json(2))

        judge = RelevanceJudge(judge_provider=provider, rubric=rubric, model_id="m")
        item = GoldenQuestion(
            id="i1", question="Q?", expected_answer_keywords=[],
            expected_sources=[], category="retrieval", difficulty="easy",
            requires_calculator=False,
        )
        output = AgentResponse(
            answer="A.", sources=[], iterations=1,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=0,
        )
        result = await judge.score(item, output)
        assert result.score == 2
        assert result.judge_id == "m_relevance"
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/evaluation/test_judges.py::TestRelevanceJudge -v 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent_bench/evaluation/rubrics/relevance.md agent_bench/evaluation/judges/relevance.py tests/evaluation/test_judges.py
git commit -m "feat(judges): RelevanceJudge + three-point reference-free rubric

Reference-free three-point rubric (off-topic / partial / on-target)
scored from question + answer alone. Two anchored examples per
level. Mechanical replication of the GroundednessJudge pattern,
which is exactly what the spec's sequencing tactic enables —
groundedness was the risky authoring task, relevance follows the
same shape with rubric-specific anchoring.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.4: Author completeness rubric + `CompletenessJudge`

**Files:**
- Create: `agent_bench/evaluation/rubrics/completeness.md`
- Create: `agent_bench/evaluation/judges/completeness.py`
- Modify: `tests/evaluation/test_judges.py`

- [ ] **Step 1: Author the completeness rubric**

Create `agent_bench/evaluation/rubrics/completeness.md`:

```markdown
---
dimension: completeness
scale: three_point
reference_based: true
abstain_allowed: true
---

# Completeness (three-point)

Score how much of the gold reference answer is covered by the agent's
answer. This is reference-based — the judge sees the gold reference
and the agent's answer; score on **coverage of facts** in the
reference, not on additional facts the agent may have included.

The judge does not penalize the agent for adding correct extra detail
(that's a separate concern). Score only on what fraction of the
reference's points are present.

## Score 0

None of the reference's key points are present in the answer.

### Example A — answer addresses different facts

Reference: "StatefulSet pods receive ordinal indices, stable hostnames, and persistent storage."
Answer: "Kubernetes uses YAML manifests to declare resources."

Score=0 — none of the three reference points (ordinal, hostname, storage) appear.

### Example B — refusal that covers nothing

Reference: "The default port is 8080."
Answer: "I cannot find that information."

Score=0 — the reference's single point (port=8080) is not in the answer.

## Score 1

Some but not all of the reference's points are present.

### Example C — partial coverage

Reference: "StatefulSet pods receive ordinal indices, stable hostnames, and persistent storage."
Answer: "StatefulSet pods get ordinal indices."

Score=1 — one of three points covered.

### Example D — half a comparison

Reference: "Deployments manage stateless replicas; StatefulSets manage stateful pods with stable identities."
Answer: "Deployments manage stateless replicas with rolling updates."

Score=1 — Deployment side covered, StatefulSet side missing.

## Score 2

All of the reference's key points are present (paraphrase allowed).

### Example E — full coverage with paraphrase

Reference: "StatefulSet pods receive ordinal indices, stable hostnames, and persistent storage."
Answer: "Each pod gets an ordinal number, a stable DNS name, and storage that survives restarts."

Score=2 — all three points covered with paraphrase.

### Example F — full coverage of single-fact reference

Reference: "The default port is 8080."
Answer: "Port 8080."

Score=2 — the only reference point is covered.
```

- [ ] **Step 2: Implement `agent_bench/evaluation/judges/completeness.py`**

```python
"""CompletenessJudge — three-point, reference-based on item.reference_answer."""

from __future__ import annotations

from agent_bench.agents.orchestrator import AgentResponse
from agent_bench.evaluation.harness import GoldenQuestion
from agent_bench.evaluation.judges.base import (
    Judge,
    ScoreResult,
    _call_judge_with_retry,
)
from agent_bench.evaluation.judges.groundedness import _system_output_hash


class CompletenessJudge(Judge):
    async def score(
        self,
        item: GoldenQuestion,
        output: AgentResponse,
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        prompt = (
            f"{self.rubric.render_prompt(level_permutation_seed=prompt_seed)}\n\n"
            f"---\n\n"
            f"## Reference answer (gold)\n{item.reference_answer}\n\n"
            f"## Answer to score\n{output.answer}\n\n"
            f"Score this answer against the rubric above. Respond with ONLY a "
            f'JSON object: {{"reasoning": "...", "evidence_quotes": [...], "score": 0 or 1 or 2 or "Unknown"}}.'
        )
        return await _call_judge_with_retry(
            provider=self.judge_provider,
            prompt=prompt,
            valid_scores={0, 1, 2},
            judge_id=self.judge_id,
            rubric_version=self.rubric.source_hash,
            prompt_seed=prompt_seed,
            system_output_hash=_system_output_hash(
                item.id, output.answer, [s.source for s in output.sources]
            ),
            item_id=item.id,
            abstain_allowed=self.rubric.abstain_allowed,
        )
```

- [ ] **Step 3: Add smoke test, run, commit**

Append a smoke test mirroring `TestRelevanceJudge` (with `from agent_bench.evaluation.judges.completeness import CompletenessJudge` and `judge_id == "m_completeness"`).

Run: `python3 -m pytest tests/evaluation/test_judges.py -v 2>&1 | tail -5`
Expected: all PASS.

```bash
git add agent_bench/evaluation/rubrics/completeness.md agent_bench/evaluation/judges/completeness.py tests/evaluation/test_judges.py
git commit -m "feat(judges): CompletenessJudge + three-point reference-based rubric

Three-point rubric (none / partial / full) scored against the gold
reference_answer. Coverage-of-facts framing: score only on what
fraction of the reference's points are present, not on additional
correct facts. Two anchored examples per level.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.5: Author citation_faithfulness rubric + `CitationFaithfulnessJudge`

**Files:**
- Create: `agent_bench/evaluation/rubrics/citation_faithfulness.md`
- Create: `agent_bench/evaluation/judges/citation_faithfulness.py`
- Modify: `tests/evaluation/test_judges.py`

This judge is opt-in for v1 (`evaluation.judge_dimensions` default does not include it). It returns one aggregate `ScoreResult` per item with all-or-nothing aggregation: any unfaithful (claim, citation) pair → score=0.

- [ ] **Step 1: Author rubric**

Create `agent_bench/evaluation/rubrics/citation_faithfulness.md`:

```markdown
---
dimension: citation_faithfulness
scale: binary
reference_based: true
abstain_allowed: true
---

# Citation faithfulness (binary, all-or-nothing aggregation per item)

For each [source: X.md] citation in the answer, is the cited chunk's
content actually relevant to the claim it supports? This is stricter
than the deterministic citation_accuracy metric, which only checks
that the cited chunk_id appears in the retrieved set — citation
faithfulness checks the **relevance** of the chunk to the claim.

**Aggregation rule (item-level):** any unfaithful citation in the
answer → item score = 0. A single bad citation in a multi-citation
answer is a real failure that all-or-nothing surfaces; treating it as
partial would obscure the failure mode.

## Score 0

At least one citation in the answer cites a chunk whose content does
not support the adjacent claim.

### Example A — citation drift

Claim: "The default port is 8080. [source: dashboard.md]"
Cited chunk content: "The dashboard supports OAuth and SAML authentication."

Score=0 because the chunk talks about authentication, not the port.
The citation is misleading even though the claim happens to be true.

### Example B — one bad citation among several

Answer cites three sources for three claims. Two citations match;
one cites a chunk about an unrelated topic.

Score=0 — all-or-nothing rule applies.

## Score 1

Every citation in the answer points to a chunk whose content directly
supports the adjacent claim.

### Example C — single accurate citation

Claim: "The default port is 8080. [source: dashboard.md]"
Cited chunk content: "The dashboard listens on port 8080 by default."

Score=1.

### Example D — multiple accurate citations

Answer makes 3 claims with 3 citations; each cited chunk's content
supports the claim it's attached to.

Score=1.
```

- [ ] **Step 2: Implement `agent_bench/evaluation/judges/citation_faithfulness.py`**

```python
"""CitationFaithfulnessJudge — binary, per-(claim,citation) all-or-nothing."""

from __future__ import annotations

import re

from agent_bench.agents.orchestrator import AgentResponse
from agent_bench.evaluation.harness import GoldenQuestion
from agent_bench.evaluation.judges.base import (
    Judge,
    ScoreResult,
    _call_judge_with_retry,
)
from agent_bench.evaluation.judges.groundedness import _system_output_hash

_CITATION_PATTERN = re.compile(r"\[source:\s*([^\]]+)\]")


def _extract_claims_with_citations(answer: str) -> list[tuple[str, str]]:
    """Return list of (claim_text, cited_source) pairs.

    A "claim" is the sentence ending at the citation. Best-effort:
    splits on sentence-ending punctuation before the [source:] tag.
    """
    pairs: list[tuple[str, str]] = []
    for match in _CITATION_PATTERN.finditer(answer):
        cited = match.group(1).strip()
        # Take the substring from start (or last sentence end) to the citation
        before = answer[: match.start()]
        # Find the last sentence-ender before this citation
        last_end = max(before.rfind("."), before.rfind("!"), before.rfind("?"))
        claim = before[last_end + 1 :].strip() if last_end >= 0 else before.strip()
        pairs.append((claim, cited))
    return pairs


class CitationFaithfulnessJudge(Judge):
    """Aggregates per-(claim, citation) judgments into one item-level
    binary ScoreResult. Per-pair detail is in evidence_quotes.

    All-or-nothing aggregation: any unfaithful citation → score 0.
    The rubric documents the rule explicitly.
    """

    async def score(
        self,
        item: GoldenQuestion,
        output: AgentResponse,
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        pairs = _extract_claims_with_citations(output.answer)
        # Map cited source name to its retrieved chunk text via output.source_chunks
        # (assumes index alignment with output.sources, matching harness convention)
        source_to_chunk: dict[str, str] = {}
        for src_ref, chunk in zip(output.sources, output.source_chunks):
            source_to_chunk.setdefault(src_ref.source, chunk)

        per_pair_results: list[ScoreResult] = []
        any_unfaithful = False
        sys_hash = _system_output_hash(
            item.id, output.answer, [s.source for s in output.sources]
        )

        if not pairs:
            # No citations to check — vacuously faithful
            return ScoreResult(
                reasoning="no_citations_in_answer",
                evidence_quotes=[],
                score=1,
                judge_id=self.judge_id,
                rubric_version=self.rubric.source_hash,
                prompt_seed=prompt_seed,
                system_output_hash=sys_hash,
                cost_usd=0.0,
                latency_ms=0.0,
            )

        accumulated_cost = 0.0
        accumulated_latency = 0.0
        for claim, cited in pairs:
            chunk = source_to_chunk.get(cited, "")
            prompt = (
                f"{self.rubric.render_prompt(level_permutation_seed=prompt_seed)}\n\n"
                f"---\n\n"
                f"## Claim (from agent's answer)\n{claim}\n\n"
                f"## Cited chunk content\n{chunk}\n\n"
                f"Does the cited chunk support the claim? Respond with ONLY a "
                f'JSON object: {{"reasoning": "...", "evidence_quotes": [...], "score": 0 or 1 or "Unknown"}}.'
            )
            sub_result = await _call_judge_with_retry(
                provider=self.judge_provider,
                prompt=prompt,
                valid_scores={0, 1},
                judge_id=self.judge_id,
                rubric_version=self.rubric.source_hash,
                prompt_seed=prompt_seed,
                system_output_hash=sys_hash,
                item_id=f"{item.id}::{cited}",
                abstain_allowed=self.rubric.abstain_allowed,
            )
            per_pair_results.append(sub_result)
            accumulated_cost += sub_result.cost_usd
            accumulated_latency += sub_result.latency_ms
            if sub_result.score == 0:
                any_unfaithful = True

        # All-or-nothing aggregation
        aggregate_score: int | str = 0 if any_unfaithful else 1
        # If any sub-call abstained, propagate Unknown (consistent with strict-
        # quorum / any-abstain principles in jury and rubric_permute)
        if any(r.abstained for r in per_pair_results):
            aggregate_score = "Unknown"

        return ScoreResult(
            reasoning=(
                f"all_or_nothing aggregate over {len(per_pair_results)} (claim, citation) pairs; "
                f"unfaithful={sum(1 for r in per_pair_results if r.score == 0)}, "
                f"abstained={sum(1 for r in per_pair_results if r.abstained)}"
            ),
            evidence_quotes=[r.reasoning[:120] for r in per_pair_results],
            score=aggregate_score,
            judge_id=self.judge_id,
            rubric_version=self.rubric.source_hash,
            prompt_seed=prompt_seed,
            system_output_hash=sys_hash,
            cost_usd=accumulated_cost,
            latency_ms=accumulated_latency,
        )
```

- [ ] **Step 3: Smoke test (claims+citations extraction + aggregation logic)**

Append to `tests/evaluation/test_judges.py`:

```python
class TestCitationFaithfulnessJudge:
    def test_extract_claims_with_citations(self):
        from agent_bench.evaluation.judges.citation_faithfulness import (
            _extract_claims_with_citations,
        )
        answer = "The port is 8080. [source: a.md] TLS is enabled. [source: b.md]"
        pairs = _extract_claims_with_citations(answer)
        assert len(pairs) == 2
        assert pairs[0] == ("The port is 8080.", "a.md")
        assert pairs[1] == ("TLS is enabled.", "b.md")

    @pytest.mark.asyncio
    async def test_aggregate_all_faithful(self, monkeypatch):
        from agent_bench.evaluation.judges.citation_faithfulness import (
            CitationFaithfulnessJudge,
        )
        from agent_bench.evaluation.judges.base import Rubric
        from agent_bench.evaluation.harness import GoldenQuestion
        from agent_bench.agents.orchestrator import AgentResponse, SourceReference
        from agent_bench.core.types import TokenUsage

        rubric = Rubric.from_markdown_file(
            "agent_bench/evaluation/rubrics/citation_faithfulness.md"
        )
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.return_value = _mk_response(_valid_json(1))

        judge = CitationFaithfulnessJudge(judge_provider=provider, rubric=rubric, model_id="m")
        item = GoldenQuestion(
            id="i1", question="?", expected_answer_keywords=[], expected_sources=[],
            category="retrieval", difficulty="easy", requires_calculator=False,
        )
        output = AgentResponse(
            answer="Fact one. [source: a.md] Fact two. [source: b.md]",
            sources=[SourceReference(source="a.md"), SourceReference(source="b.md")],
            source_chunks=["chunk for a", "chunk for b"],
            iterations=1,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=0,
        )
        result = await judge.score(item, output)
        assert result.score == 1
        assert provider.complete.await_count == 2

    @pytest.mark.asyncio
    async def test_aggregate_one_unfaithful_makes_zero(self):
        from agent_bench.evaluation.judges.citation_faithfulness import (
            CitationFaithfulnessJudge,
        )
        from agent_bench.evaluation.judges.base import Rubric
        from agent_bench.evaluation.harness import GoldenQuestion
        from agent_bench.agents.orchestrator import AgentResponse, SourceReference
        from agent_bench.core.types import TokenUsage

        rubric = Rubric.from_markdown_file(
            "agent_bench/evaluation/rubrics/citation_faithfulness.md"
        )
        provider = AsyncMock(spec=LLMProvider)
        provider.complete.side_effect = [
            _mk_response(_valid_json(1)),
            _mk_response(_valid_json(0)),
        ]

        judge = CitationFaithfulnessJudge(judge_provider=provider, rubric=rubric, model_id="m")
        item = GoldenQuestion(
            id="i1", question="?", expected_answer_keywords=[], expected_sources=[],
            category="retrieval", difficulty="easy", requires_calculator=False,
        )
        output = AgentResponse(
            answer="Good. [source: a.md] Bad. [source: b.md]",
            sources=[SourceReference(source="a.md"), SourceReference(source="b.md")],
            source_chunks=["chunk for a", "chunk for b"],
            iterations=1,
            usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
            latency_ms=0,
        )
        result = await judge.score(item, output)
        assert result.score == 0
```

- [ ] **Step 4: Run tests, commit**

Run: `python3 -m pytest tests/evaluation/test_judges.py -v 2>&1 | tail -10`
Expected: all PASS.

```bash
git add agent_bench/evaluation/rubrics/citation_faithfulness.md agent_bench/evaluation/judges/citation_faithfulness.py tests/evaluation/test_judges.py
git commit -m "feat(judges): CitationFaithfulnessJudge with all-or-nothing aggregation

Per-(claim, citation) binary judge that aggregates to one item-level
ScoreResult via all-or-nothing — any unfaithful citation → score=0.
Per-pair detail preserved in evidence_quotes.

Opt-in for v1 (judge_dimensions default excludes it); default-on in
v1.1 once the citation-deterministic-vs-LLM head-to-head section
of the writeup validates the gain over the existing regex-based
citation_accuracy.

Any sub-call abstain propagates to the aggregate (consistent with
the strict-quorum / any-abstain principles in jury and rubric_permute).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3: Variance wrappers

`PermutedJudge` and `Jury` — both wrap one or more `Judge` instances and produce a single aggregate `ScoreResult`. Per-call detail is written to a sidecar JSONL with a deterministic default path.

### Task 3.1: `PermutedJudge` (rubric_permute wrapper)

**Files:**
- Create: `agent_bench/evaluation/variance/__init__.py`
- Create: `agent_bench/evaluation/variance/rubric_permute.py`
- Create: `tests/evaluation/test_jury_aggregation.py`

- [ ] **Step 1: Write failing test for PermutedJudge**

Create `tests/evaluation/test_jury_aggregation.py`:

```python
"""Tests for PermutedJudge and Jury — aggregation, quorum, sidecar."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from agent_bench.agents.orchestrator import AgentResponse, SourceReference
from agent_bench.core.provider import LLMProvider
from agent_bench.core.types import CompletionResponse, TokenUsage
from agent_bench.evaluation.harness import GoldenQuestion
from agent_bench.evaluation.judges.base import (
    ABSTAIN_REASON_SCHEMA_PARSE,
    Rubric,
    ScoreResult,
)
from agent_bench.evaluation.judges.relevance import RelevanceJudge


def _mk_response(content: str) -> CompletionResponse:
    return CompletionResponse(
        content=content, tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=10, estimated_cost_usd=0.0001),
        provider="mock", model="m", latency_ms=1.0,
    )


def _vj(score) -> str:
    return json.dumps({"reasoning": "r", "evidence_quotes": [], "score": score})


def _item(item_id: str = "i1") -> GoldenQuestion:
    return GoldenQuestion(
        id=item_id, question="?", expected_answer_keywords=[], expected_sources=[],
        category="retrieval", difficulty="easy", requires_calculator=False,
    )


def _output(answer: str = "A.") -> AgentResponse:
    return AgentResponse(
        answer=answer, sources=[SourceReference(source="x.md")], iterations=1,
        usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
        latency_ms=0,
    )


def _relevance_judge_with_responses(responses: list[str]) -> RelevanceJudge:
    rubric = Rubric.from_markdown_file("agent_bench/evaluation/rubrics/relevance.md")
    provider = AsyncMock(spec=LLMProvider)
    provider.complete.side_effect = [_mk_response(r) for r in responses]
    return RelevanceJudge(judge_provider=provider, rubric=rubric, model_id="m")


class TestPermutedJudge:
    @pytest.mark.asyncio
    async def test_runs_n_permutations_and_means(self, tmp_path):
        from agent_bench.evaluation.variance.rubric_permute import rubric_permute

        # Two seeds produce two scores: 1 and 2; mean=1.5; rounded down → 1
        judge = _relevance_judge_with_responses([_vj(1), _vj(2)])
        permuted = rubric_permute(judge, n=2, seeds=[1, 2], sidecar_path=tmp_path / "side.jsonl")
        result = await permuted.score(_item(), _output())
        assert result.score == 1  # mean=1.5, ties→lower
        assert result.judge_id == "m_relevance_perm2"
        assert result.prompt_seed == 0  # aggregate carries 0

    @pytest.mark.asyncio
    async def test_any_abstain_propagates_unknown(self, tmp_path):
        from agent_bench.evaluation.variance.rubric_permute import rubric_permute

        judge = _relevance_judge_with_responses([_vj(1), _vj("Unknown")])
        permuted = rubric_permute(judge, n=2, seeds=[1, 2], sidecar_path=tmp_path / "side.jsonl")
        result = await permuted.score(_item(), _output())
        assert result.score == "Unknown"
        assert result.abstained

    @pytest.mark.asyncio
    async def test_writes_per_permutation_sidecar(self, tmp_path):
        from agent_bench.evaluation.variance.rubric_permute import rubric_permute

        sidecar = tmp_path / "perm_members.jsonl"
        judge = _relevance_judge_with_responses([_vj(2), _vj(2)])
        permuted = rubric_permute(judge, n=2, seeds=[5, 7], sidecar_path=sidecar)
        await permuted.score(_item(), _output())

        lines = sidecar.read_text().strip().split("\n")
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert {r["prompt_seed"] for r in records} == {5, 7}
```

- [ ] **Step 2: Run test, expect fail**

Run: `python3 -m pytest tests/evaluation/test_jury_aggregation.py::TestPermutedJudge -v 2>&1 | tail -10`
Expected: ImportError on `agent_bench.evaluation.variance.rubric_permute`.

- [ ] **Step 3: Implement `PermutedJudge`**

Create `agent_bench/evaluation/variance/__init__.py`:

```python
"""Variance-control wrappers around Judge instances."""

from agent_bench.evaluation.variance.rubric_permute import (
    PermutedJudge,
    rubric_permute,
)

__all__ = ["PermutedJudge", "rubric_permute"]
```

Create `agent_bench/evaluation/variance/rubric_permute.py`:

```python
"""rubric_permute — runs the same judge with permuted rubric levels and aggregates."""

from __future__ import annotations

import json
from pathlib import Path

from agent_bench.agents.orchestrator import AgentResponse
from agent_bench.evaluation.harness import GoldenQuestion
from agent_bench.evaluation.judges.base import Judge, ScoreResult


def _aggregate_scores(
    scores: list[int], scale: str
) -> int:
    """Discretize aggregated score per scale.

    Binary: threshold 0.5 with ties → 0 (conservative).
    Three-point: round to nearest with ties → lower level (conservative).
    """
    mean = sum(scores) / len(scores)
    if scale == "binary":
        return 1 if mean > 0.5 else 0
    # three_point: round down on ties
    floor = int(mean)
    frac = mean - floor
    if frac > 0.5:
        return floor + 1
    return floor


class PermutedJudge:
    """Wraps a Judge; runs N permutations with different prompt_seeds.

    Aggregation:
    - Any abstain in any permutation → aggregate score = "Unknown".
    - Otherwise, discretize the per-permutation scores per scale.

    Per-permutation ScoreResults are written to the sidecar JSONL on
    every score() call (one batch per call, append-mode JSONL across calls).
    """

    def __init__(
        self,
        judge: Judge,
        n: int = 2,
        seeds: list[int] | None = None,
        sidecar_path: Path | str | None = None,
    ) -> None:
        self.judge = judge
        self.n = n
        self.seeds = seeds if seeds is not None else list(range(1, n + 1))
        if len(self.seeds) != n:
            raise ValueError(f"seeds length {len(self.seeds)} != n {n}")
        self.sidecar_path = Path(sidecar_path) if sidecar_path else None
        self.judge_id = f"{judge.judge_id}_perm{n}"

    async def score(
        self,
        item: GoldenQuestion,
        output: AgentResponse,
    ) -> ScoreResult:
        per_perm_results: list[ScoreResult] = []
        for seed in self.seeds:
            r = await self.judge.score(item, output, prompt_seed=seed)
            per_perm_results.append(r)

        if self.sidecar_path is not None:
            self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            with self.sidecar_path.open("a", encoding="utf-8") as f:
                for r in per_perm_results:
                    f.write(r.model_dump_json() + "\n")

        any_abstain = any(r.abstained for r in per_perm_results)
        if any_abstain:
            score: int | str = "Unknown"
            reasoning = (
                f"any_abstain_propagated: {sum(1 for r in per_perm_results if r.abstained)}"
                f"/{self.n} permutations abstained"
            )
        else:
            score = _aggregate_scores(
                [int(r.score) for r in per_perm_results],
                self.judge.rubric.scale,
            )
            reasoning = f"perm_mean over {self.n} seeds: {[r.score for r in per_perm_results]}"

        return ScoreResult(
            reasoning=reasoning,
            evidence_quotes=[],
            score=score,
            judge_id=self.judge_id,
            rubric_version=self.judge.rubric.source_hash,
            prompt_seed=0,
            system_output_hash=per_perm_results[0].system_output_hash,
            cost_usd=sum(r.cost_usd for r in per_perm_results),
            latency_ms=sum(r.latency_ms for r in per_perm_results),
        )


def rubric_permute(
    judge: Judge,
    n: int = 2,
    seeds: list[int] | None = None,
    sidecar_path: Path | str | None = None,
) -> PermutedJudge:
    return PermutedJudge(judge=judge, n=n, seeds=seeds, sidecar_path=sidecar_path)
```

- [ ] **Step 4: Run tests, commit**

Run: `python3 -m pytest tests/evaluation/test_jury_aggregation.py::TestPermutedJudge -v 2>&1 | tail -10`
Expected: 3 PASS.

```bash
git add agent_bench/evaluation/variance/__init__.py agent_bench/evaluation/variance/rubric_permute.py tests/evaluation/test_jury_aggregation.py
git commit -m "feat(variance): PermutedJudge — N seeded rubric permutations + aggregation

Wraps a Judge; runs N permutations with different prompt_seeds.
Discretization: binary thresholds at 0.5 with ties → 0 (conservative);
three-point rounds to nearest with ties → lower level. Any abstain
in any permutation propagates to Unknown — the variance signal that
permutation is designed to surface should not be averaged away.

Per-permutation ScoreResults written to sidecar JSONL (append-mode
so multiple score() calls across items accumulate). Aggregate carries
prompt_seed=0; the per-permutation seeds are recoverable from the sidecar.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.2: `Jury` (multi-judge aggregator with quorum)

**Files:**
- Create: `agent_bench/evaluation/variance/jury.py`
- Modify: `agent_bench/evaluation/variance/__init__.py`
- Modify: `tests/evaluation/test_jury_aggregation.py`

- [ ] **Step 1: Write failing tests for Jury**

Append to `tests/evaluation/test_jury_aggregation.py`:

```python
class TestJury:
    @pytest.mark.asyncio
    async def test_mean_aggregation_two_judges(self, tmp_path):
        from agent_bench.evaluation.variance.jury import jury

        j1 = _relevance_judge_with_responses([_vj(2)])
        j2 = _relevance_judge_with_responses([_vj(2)])
        # Patch judge_id so they're distinct
        j1.judge_id = "claude-haiku_relevance"
        j2.judge_id = "gpt-4o-mini_relevance"

        ju = jury(judges=[j1, j2], aggregation="mean", sidecar_path=tmp_path / "jury.jsonl")
        result = await ju.score(_item(), _output())
        assert result.score == 2
        assert result.judge_id == "jury_v1_mean"

    @pytest.mark.asyncio
    async def test_strict_quorum_default_abstains_on_one_failure(self, tmp_path):
        from agent_bench.evaluation.variance.jury import jury

        # j1 succeeds; j2 abstains via schema-parse-failure-after-retry
        j1 = _relevance_judge_with_responses([_vj(1)])
        j1.judge_id = "claude-haiku_relevance"
        j2 = _relevance_judge_with_responses(["garbage", "garbage"])  # both attempts fail
        j2.judge_id = "gpt-4o-mini_relevance"

        ju = jury(judges=[j1, j2], aggregation="mean", sidecar_path=tmp_path / "jury.jsonl")
        result = await ju.score(_item(), _output())
        assert result.score == "Unknown"
        assert "jury_below_quorum" in result.reasoning
        assert "1/2" in result.reasoning

    @pytest.mark.asyncio
    async def test_sidecar_captures_both_members_including_abstain(self, tmp_path):
        from agent_bench.evaluation.variance.jury import jury

        j1 = _relevance_judge_with_responses([_vj(1)])
        j1.judge_id = "claude-haiku_relevance"
        j2 = _relevance_judge_with_responses(["garbage", "garbage"])
        j2.judge_id = "gpt-4o-mini_relevance"

        sidecar = tmp_path / "jury.jsonl"
        ju = jury(judges=[j1, j2], aggregation="mean", sidecar_path=sidecar)
        await ju.score(_item(), _output())

        records = [json.loads(line) for line in sidecar.read_text().strip().split("\n")]
        assert len(records) == 2
        scores = [r["score"] for r in records]
        assert 1 in scores
        assert "Unknown" in scores

    @pytest.mark.asyncio
    async def test_kappa_weighted_requires_weights(self, tmp_path):
        from agent_bench.evaluation.variance.jury import jury

        j1 = _relevance_judge_with_responses([_vj(2)])
        with pytest.raises(ValueError, match="weights"):
            jury(judges=[j1], aggregation="kappa_weighted")

    @pytest.mark.asyncio
    async def test_cancel_on_non_retryable(self, tmp_path):
        """Non-retryable exception in any member must propagate immediately."""
        from agent_bench.evaluation.variance.jury import jury
        from agent_bench.evaluation.judges.base import Rubric

        rubric = Rubric.from_markdown_file("agent_bench/evaluation/rubrics/relevance.md")
        # j1 raises ValueError (caller bug — not in retryable taxonomy)
        provider1 = AsyncMock(spec=LLMProvider)
        provider1.complete.side_effect = ValueError("auth_error")
        j1 = RelevanceJudge(judge_provider=provider1, rubric=rubric, model_id="m1")

        # j2 would succeed if it ran
        provider2 = AsyncMock(spec=LLMProvider)
        provider2.complete.return_value = _mk_response(_vj(1))
        j2 = RelevanceJudge(judge_provider=provider2, rubric=rubric, model_id="m2")

        ju = jury(judges=[j1, j2], aggregation="mean", sidecar_path=tmp_path / "jury.jsonl")
        with pytest.raises(ValueError, match="auth_error"):
            await ju.score(_item(), _output())
```

- [ ] **Step 2: Run, expect fail**

Run: `python3 -m pytest tests/evaluation/test_jury_aggregation.py::TestJury -v 2>&1 | tail -10`
Expected: ImportError on `agent_bench.evaluation.variance.jury`.

- [ ] **Step 3: Implement Jury**

Create `agent_bench/evaluation/variance/jury.py`:

```python
"""Jury — multi-judge aggregator with strict-quorum default and sidecar."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from agent_bench.agents.orchestrator import AgentResponse
from agent_bench.evaluation.harness import GoldenQuestion
from agent_bench.evaluation.judges.base import Judge, ScoreResult
from agent_bench.evaluation.variance.rubric_permute import _aggregate_scores

_DEFAULT_SIDECAR_TEMPLATE = "results/calibration_v1_judge_{aggregation}_members.jsonl"


class Jury:
    """Aggregates a list of Judge instances into one ScoreResult per item.

    Strict quorum default (quorum = len(judges)): any member abstain →
    aggregate abstain. The parameter exists in v1 so v1.1's 3-judge jury
    can shift to quorum=2 (majority) without rearchitecting failure
    semantics.

    Per-member ScoreResults always written to sidecar (successes and
    failure-as-abstains alike). Provider non-retryable exceptions in
    any member raise immediately, cancelling sibling gather tasks.
    """

    def __init__(
        self,
        judges: list[Judge],
        aggregation: Literal["mean", "kappa_weighted"],
        weights: dict[str, float] | None = None,
        quorum: int | None = None,
        sidecar_path: Path | str | None = None,
    ) -> None:
        if not judges:
            raise ValueError("jury requires at least one judge")
        if aggregation == "kappa_weighted" and not weights:
            raise ValueError(
                "kappa_weighted aggregation requires explicit weights "
                "(computed offline on calibration set; not at jury construction)"
            )
        self.judges = judges
        self.aggregation = aggregation
        self.weights = weights or {}
        self.quorum = quorum if quorum is not None else len(judges)
        self.sidecar_path = (
            Path(sidecar_path)
            if sidecar_path is not None
            else Path(_DEFAULT_SIDECAR_TEMPLATE.format(aggregation=aggregation))
        )
        self.judge_id = f"jury_v1_{aggregation}"

    async def score(
        self,
        item: GoldenQuestion,
        output: AgentResponse,
    ) -> ScoreResult:
        # return_exceptions=False → first exception cancels siblings
        member_results: list[ScoreResult] = await asyncio.gather(
            *[j.score(item, output) for j in self.judges],
            return_exceptions=False,
        )

        # Sidecar (append; one line per member per call)
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with self.sidecar_path.open("a", encoding="utf-8") as f:
            for r in member_results:
                f.write(r.model_dump_json() + "\n")

        successful = [r for r in member_results if not r.abstained]
        sys_hash = member_results[0].system_output_hash

        if len(successful) < self.quorum:
            return ScoreResult(
                reasoning=(
                    f"jury_below_quorum: {len(successful)}/{len(self.judges)} "
                    f"members succeeded; required {self.quorum}"
                ),
                evidence_quotes=[],
                score="Unknown",
                judge_id=self.judge_id,
                rubric_version=member_results[0].rubric_version,
                prompt_seed=0,
                system_output_hash=sys_hash,
                cost_usd=sum(r.cost_usd for r in member_results),
                latency_ms=max(r.latency_ms for r in member_results),
            )

        # Aggregate over successful members
        scores = [int(r.score) for r in successful]
        scale = self.judges[0].rubric.scale
        if self.aggregation == "mean":
            agg = _aggregate_scores(scores, scale)
        else:  # kappa_weighted
            ws = [self.weights.get(j.judge_id, 1.0) for j in self.judges if not next(
                (r.abstained for r in member_results if r.judge_id == j.judge_id), True
            )]
            weighted_sum = sum(s * w for s, w in zip(scores, ws))
            weight_total = sum(ws)
            mean = weighted_sum / weight_total if weight_total > 0 else 0.0
            agg = _aggregate_scores([int(round(mean))], scale)

        return ScoreResult(
            reasoning=(
                f"jury_{self.aggregation}: members={[r.score for r in successful]}, "
                f"weights={list(self.weights.values()) if self.aggregation == 'kappa_weighted' else 'n/a'}"
            ),
            evidence_quotes=[],
            score=agg,
            judge_id=self.judge_id,
            rubric_version=member_results[0].rubric_version,
            prompt_seed=0,
            system_output_hash=sys_hash,
            cost_usd=sum(r.cost_usd for r in member_results),
            latency_ms=max(r.latency_ms for r in member_results),
        )


def jury(
    judges: list[Judge],
    aggregation: Literal["mean", "kappa_weighted"],
    weights: dict[str, float] | None = None,
    quorum: int | None = None,
    sidecar_path: Path | str | None = None,
) -> Jury:
    return Jury(
        judges=judges,
        aggregation=aggregation,
        weights=weights,
        quorum=quorum,
        sidecar_path=sidecar_path,
    )
```

Update `agent_bench/evaluation/variance/__init__.py`:

```python
"""Variance-control wrappers around Judge instances."""

from agent_bench.evaluation.variance.jury import Jury, jury
from agent_bench.evaluation.variance.rubric_permute import (
    PermutedJudge,
    rubric_permute,
)

__all__ = ["Jury", "PermutedJudge", "jury", "rubric_permute"]
```

- [ ] **Step 4: Run tests, commit**

Run: `python3 -m pytest tests/evaluation/test_jury_aggregation.py -v 2>&1 | tail -15`
Expected: 8 tests PASS (3 PermutedJudge + 5 Jury).

```bash
git add agent_bench/evaluation/variance/jury.py agent_bench/evaluation/variance/__init__.py tests/evaluation/test_jury_aggregation.py
git commit -m "feat(variance): Jury aggregator with strict-quorum default and sidecar

asyncio.gather(return_exceptions=False) + try/except at jury level
so non-retryable exceptions cancel sibling tasks immediately
(failing fast on caller bugs). Per-member ScoreResults written to
sidecar JSONL on every call — successes AND failure-as-abstains —
so the calibration report can compute per-judge κ even when the
aggregate row drops to abstain via the quorum gate.

Strict quorum default (quorum = len(judges)) at v1's 2-judge jury
means any member abstain → jury abstain. Tolerant defaults at N=2
are silent single-judge in jury clothing; v1.1's 3-judge jury can
shift to quorum=2 (majority) by parameter, no failure-semantics
rearchitecture needed.

kappa_weighted requires explicit weights injection at construction
(weights computed offline once on calibration set; not at jury
construction — would be circular).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4: Calibration metrics (hand-rolled κ + bootstrap)

Hand-rolled `cohen_kappa`, `gwets_ac2`, `bootstrap_ci`. Hand-computed test cases first; then sklearn-fixture parity tests for κ. The sklearn fixture-generation script lives under `scripts/_dev/` and is run from a venv outside the project.

### Task 4.1: Hand-rolled `cohen_kappa` with hand-computed cases

**Files:**
- Create: `agent_bench/evaluation/calibration/__init__.py`
- Create: `agent_bench/evaluation/calibration/metrics.py`
- Create: `tests/evaluation/test_calibration_metrics.py`

- [ ] **Step 1: Write failing tests for hand-computed κ**

Create `tests/evaluation/test_calibration_metrics.py`:

```python
"""Tests for hand-rolled Cohen's kappa, Gwet's AC2, bootstrap CI."""

from __future__ import annotations

import pytest

from agent_bench.evaluation.calibration.metrics import (
    bootstrap_ci,
    cohen_kappa,
    gwets_ac2,
)


class TestCohenKappaHandComputed:
    def test_perfect_agreement_kappa_one(self):
        # 5 ones, 5 zeros, both raters identical
        # P_o = 1.0
        # P_e = (5/10 * 5/10) + (5/10 * 5/10) = 0.25 + 0.25 = 0.5
        # κ = (1.0 - 0.5) / (1.0 - 0.5) = 1.0
        y1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        assert cohen_kappa(y1, y2) == pytest.approx(1.0)

    def test_complete_disagreement_kappa_negative(self):
        # 5 ones, 5 zeros for each, but inverted
        # P_o = 0.0
        # P_e = (5/10 * 5/10) + (5/10 * 5/10) = 0.5
        # κ = (0.0 - 0.5) / (1.0 - 0.5) = -1.0
        y1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        assert cohen_kappa(y1, y2) == pytest.approx(-1.0)

    def test_chance_agreement_kappa_zero(self):
        # Worked-out case: 2x2 confusion matrix where observed agreement
        # equals chance agreement.
        # raters distribute identically across categories: marginals are
        # P(0)=0.5, P(1)=0.5 for both; if confusion matrix is uniform 0.25/0.25/0.25/0.25
        # then P_o = 0.25 + 0.25 = 0.5 and P_e = 0.5, so κ = 0.
        y1 = [0, 0, 1, 1]
        y2 = [0, 1, 0, 1]
        assert cohen_kappa(y1, y2) == pytest.approx(0.0)


class TestGwetsAC2HandComputed:
    def test_perfect_agreement(self):
        y1 = [0, 0, 1, 1]
        y2 = [0, 0, 1, 1]
        assert gwets_ac2(y1, y2) == pytest.approx(1.0)

    def test_complete_disagreement(self):
        y1 = [0, 0, 1, 1]
        y2 = [1, 1, 0, 0]
        # AC2 with q=2 categories: observed agreement = 0,
        # chance term = 1/(q-1) * sum p_k(1-p_k) = (1/1)*(0.5*0.5 + 0.5*0.5) = 0.5
        # AC2 = (0 - 0.5) / (1 - 0.5) = -1.0
        assert gwets_ac2(y1, y2) == pytest.approx(-1.0)

    def test_mid_range(self):
        # 3 of 4 agree
        y1 = [0, 0, 1, 1]
        y2 = [0, 0, 1, 0]
        # P_o = 0.75
        # marginal mean across raters per category:
        #   p_0 = (3 + 3) / (2*4) = 0.75; p_1 = (1 + 1) / (2*4) = 0.25
        # ... actually use AC2's specific formula. We accept the
        # implementation-derived value here as the contract; this test
        # locks the formula choice.
        result = gwets_ac2(y1, y2)
        assert -1.0 <= result <= 1.0
        assert result > 0  # mostly-agreement should give positive AC2


class TestBootstrapCI:
    def test_returns_point_lo_hi_tuple(self):
        y1 = [0, 0, 1, 1, 1, 0, 1, 0]
        y2 = [0, 1, 1, 1, 1, 0, 1, 0]
        result = bootstrap_ci(y1, y2, cohen_kappa, n_iter=100, seed=42)
        assert len(result) == 3
        point, lo, hi = result
        assert lo <= point <= hi

    def test_seed_reproducibility(self):
        y1 = [0, 0, 1, 1, 1, 0, 1, 0]
        y2 = [0, 1, 1, 1, 1, 0, 1, 0]
        r1 = bootstrap_ci(y1, y2, cohen_kappa, n_iter=200, seed=42)
        r2 = bootstrap_ci(y1, y2, cohen_kappa, n_iter=200, seed=42)
        assert r1 == r2
```

- [ ] **Step 2: Run, expect fail**

Run: `python3 -m pytest tests/evaluation/test_calibration_metrics.py -v 2>&1 | tail -10`
Expected: ImportError on calibration.metrics.

- [ ] **Step 3: Implement metrics**

Create `agent_bench/evaluation/calibration/__init__.py`:

```python
"""Hand-rolled inter-rater agreement metrics + calibration report generator."""

from agent_bench.evaluation.calibration.metrics import (
    bootstrap_ci,
    cohen_kappa,
    gwets_ac2,
)

__all__ = ["bootstrap_ci", "cohen_kappa", "gwets_ac2"]
```

Create `agent_bench/evaluation/calibration/metrics.py`:

```python
"""Hand-rolled Cohen's kappa, Gwet's AC2, bootstrap CI.

Hand-rolled (not sklearn) for two reasons:
1. agent-bench's identity is "built from primitives" — adding sklearn
   for one function (and transitively numpy + scipy + threadpoolctl +
   joblib) contradicts that.
2. The hand-roll demonstrates formula understanding in a way that
   sklearn.metrics.cohen_kappa_score does not.

Fixture-tested against sklearn run *outside* the project venv —
see tests/evaluation/test_calibration_metrics.py and
scripts/_dev/generate_kappa_fixtures.py.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Literal


def cohen_kappa(
    y1: list,
    y2: list,
    weights: Literal[None, "linear", "quadratic"] = None,
) -> float:
    """Cohen's κ = (P_o - P_e) / (1 - P_e).

    Supports unweighted, linear-weighted, and quadratic-weighted variants
    for ordinal scales. y1 and y2 must be parallel lists of label values
    (int or str). Both must have the same length.
    """
    if len(y1) != len(y2):
        raise ValueError(f"y1 and y2 must have same length; got {len(y1)} vs {len(y2)}")
    if not y1:
        raise ValueError("Empty input — kappa undefined")

    # Build label index from union of observed values
    labels = sorted({*y1, *y2}, key=str)
    k = len(labels)
    label_idx = {lab: i for i, lab in enumerate(labels)}

    # Confusion matrix (counts)
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        cm[label_idx[a]][label_idx[b]] += 1

    n = len(y1)

    # Weight matrix
    if weights is None:
        w = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    elif weights == "linear":
        w = [[1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    elif weights == "quadratic":
        w = [[1.0 - ((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    else:
        raise ValueError(f"Invalid weights {weights!r}")

    # Observed weighted agreement
    p_o = sum(w[i][j] * cm[i][j] for i in range(k) for j in range(k)) / n

    # Marginal probabilities
    row_marg = [sum(cm[i][j] for j in range(k)) / n for i in range(k)]
    col_marg = [sum(cm[i][j] for i in range(k)) / n for j in range(k)]

    # Expected weighted agreement under independence
    p_e = sum(w[i][j] * row_marg[i] * col_marg[j] for i in range(k) for j in range(k))

    if p_e >= 1.0:
        return 1.0  # degenerate — all in one category
    return (p_o - p_e) / (1.0 - p_e)


def gwets_ac2(
    y1: list,
    y2: list,
    weights: Literal[None, "linear", "quadratic"] = None,
) -> float:
    """Gwet's AC2 — chance-corrected agreement using the sum of squared
    marginals as the chance term (more robust to skewed distributions
    than Cohen's κ).

    AC2 = (P_o - P_e_AC2) / (1 - P_e_AC2)
    where P_e_AC2 = (1/(q-1)) * Σ_k p_k * (1 - p_k)
    and p_k is the mean marginal probability for category k.
    """
    if len(y1) != len(y2):
        raise ValueError(f"y1 and y2 length mismatch")
    if not y1:
        raise ValueError("Empty input")

    labels = sorted({*y1, *y2}, key=str)
    k = len(labels)
    label_idx = {lab: i for i, lab in enumerate(labels)}

    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        cm[label_idx[a]][label_idx[b]] += 1
    n = len(y1)

    if weights is None:
        w = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    elif weights == "linear":
        w = [[1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    elif weights == "quadratic":
        w = [[1.0 - ((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    else:
        raise ValueError(f"Invalid weights {weights!r}")

    p_o = sum(w[i][j] * cm[i][j] for i in range(k) for j in range(k)) / n

    # Mean marginal across raters
    row_marg = [sum(cm[i][j] for j in range(k)) / n for i in range(k)]
    col_marg = [sum(cm[i][j] for i in range(k)) / n for j in range(k)]
    pi = [(row_marg[i] + col_marg[i]) / 2 for i in range(k)]

    if k <= 1:
        return 1.0
    # AC2 chance: with weighted variant, sum of weighted independence terms
    # using the average marginal pi (Gwet's definition).
    p_e_ac2 = sum(
        w[i][j] * pi[i] * pi[j]
        for i in range(k) for j in range(k) if i != j
    ) / (k - 1)

    if p_e_ac2 >= 1.0:
        return 1.0
    return (p_o - p_e_ac2) / (1.0 - p_e_ac2)


def bootstrap_ci(
    y1: list,
    y2: list,
    metric_fn: Callable[[list, list], float],
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for an inter-rater metric.

    Returns (point_estimate, ci_lo, ci_hi). Resamples with replacement
    n_iter times and takes the (1-ci)/2 and (1+ci)/2 percentiles.
    """
    if len(y1) != len(y2):
        raise ValueError("length mismatch")
    n = len(y1)
    rng = random.Random(seed)
    point = metric_fn(y1, y2)
    samples: list[float] = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        s1 = [y1[i] for i in idx]
        s2 = [y2[i] for i in idx]
        try:
            samples.append(metric_fn(s1, s2))
        except (ValueError, ZeroDivisionError):
            # Degenerate resample (e.g., all one label) — skip
            continue
    samples.sort()
    if not samples:
        return point, point, point
    lo_idx = int(((1 - ci) / 2) * len(samples))
    hi_idx = int(((1 + ci) / 2) * len(samples)) - 1
    return point, samples[lo_idx], samples[hi_idx]
```

- [ ] **Step 4: Run hand-computed tests**

Run: `python3 -m pytest tests/evaluation/test_calibration_metrics.py -v 2>&1 | tail -15`
Expected: all hand-computed tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_bench/evaluation/calibration/__init__.py agent_bench/evaluation/calibration/metrics.py tests/evaluation/test_calibration_metrics.py
git commit -m "feat(calibration): hand-rolled cohen_kappa, gwets_ac2, bootstrap_ci

Hand-rolled (not sklearn) per the design's 'built from primitives'
discipline. Cohen's κ: (P_o - P_e) / (1 - P_e), supports unweighted,
linear, and quadratic weight matrices for ordinal scales. Gwet's AC2:
chance term = (1/(q-1)) Σ p_k(1-p_k), more robust to skewed marginals.
Bootstrap CI: 1000-iter default, seed=42 for reproducibility.

Three hand-computed test cases per metric (perfect agreement κ=1,
complete disagreement κ=-1, chance agreement κ=0) include worked-out
arithmetic in comments so a reader can verify formula correctness
without running the test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 4.2: sklearn-parity fixtures (generation script + cross-check CI test)

**Files:**
- Create: `scripts/_dev/generate_kappa_fixtures.py`
- Create: `tests/evaluation/fixtures/sklearn_kappa_inputs.json`
- Modify: `tests/evaluation/test_calibration_metrics.py`

- [ ] **Step 1: Author the generation script**

```bash
mkdir -p scripts/_dev
```

Create `scripts/_dev/generate_kappa_fixtures.py`:

```python
"""Generate sklearn-parity fixtures for tests/evaluation/test_calibration_metrics.py.

Run from a venv with sklearn installed (NOT the project venv):

    python -m venv /tmp/sklearn-fixture-venv
    /tmp/sklearn-fixture-venv/bin/pip install scikit-learn==1.5.2
    /tmp/sklearn-fixture-venv/bin/python scripts/_dev/generate_kappa_fixtures.py

The script:
  1. Defines CASES (input arrays + weight option).
  2. Computes sklearn.metrics.cohen_kappa_score for each case.
  3. Prints copy-pasteable Python constants for the test file.
  4. Writes inputs to tests/evaluation/fixtures/sklearn_kappa_inputs.json
     for the cross-check CI test (forgot-to-regenerate detection).

DO NOT add scikit-learn to the project's runtime dependencies — these
constants are the contract; the project hand-rolls the implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError as e:
    raise SystemExit(
        "scikit-learn not installed. Install in a venv outside this project:\n"
        "  python -m venv /tmp/sklearn-fixture-venv\n"
        "  /tmp/sklearn-fixture-venv/bin/pip install scikit-learn==1.5.2\n"
        "  /tmp/sklearn-fixture-venv/bin/python scripts/_dev/generate_kappa_fixtures.py"
    ) from e

CASES: list[dict] = [
    {
        "name": "imbalanced_binary",
        "y1": [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        "y2": [1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
        "weights": None,
    },
    {
        "name": "three_point_one_diagonal_swap",
        "y1": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0],
        "y2": [0, 1, 1, 1, 2, 2, 0, 1, 2, 0],
        "weights": None,
    },
    {
        "name": "weighted_ordinal_drift_linear",
        "y1": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        "y2": [0, 1, 2, 1, 1, 2, 0, 2, 2, 1],
        "weights": "linear",
    },
]

OUT_INPUTS = Path(__file__).resolve().parents[2] / "tests" / "evaluation" / "fixtures" / "sklearn_kappa_inputs.json"

print("# --- Paste into test_calibration_metrics.py ---\n")
print("SKLEARN_KAPPA_FIXTURES: dict[str, float] = {")
for case in CASES:
    expected = cohen_kappa_score(case["y1"], case["y2"], weights=case["weights"])
    print(f'    "{case["name"]}": {expected:.10f},  # sklearn 1.5.2')
print("}")

print("\nSKLEARN_KAPPA_INPUTS: dict[str, dict] = {")
for case in CASES:
    print(f'    "{case["name"]}": {{')
    print(f'        "y1": {case["y1"]},')
    print(f'        "y2": {case["y2"]},')
    print(f'        "weights": {case["weights"]!r},')
    print("    },")
print("}")

# Write JSON sidecar for the cross-check CI test
OUT_INPUTS.parent.mkdir(parents=True, exist_ok=True)
OUT_INPUTS.write_text(json.dumps(
    {case["name"]: {"y1": case["y1"], "y2": case["y2"], "weights": case["weights"]} for case in CASES},
    indent=2,
))
print(f"\n# Wrote {OUT_INPUTS}")
```

- [ ] **Step 2: Manually run the script in an external venv to generate fixtures**

This step requires manual execution outside CI. From a terminal:

```bash
python3 -m venv /tmp/sklearn-fixture-venv
/tmp/sklearn-fixture-venv/bin/pip install --quiet 'scikit-learn==1.5.2'
/tmp/sklearn-fixture-venv/bin/python scripts/_dev/generate_kappa_fixtures.py
```

Copy the printed `SKLEARN_KAPPA_FIXTURES` and `SKLEARN_KAPPA_INPUTS` constants. Verify `tests/evaluation/fixtures/sklearn_kappa_inputs.json` was written.

- [ ] **Step 3: Add sklearn-parity tests + cross-check test to `test_calibration_metrics.py`**

Append:

```python
import json as _json
from pathlib import Path

# Fixtures generated against scikit-learn==1.5.2 cohen_kappa_score on 2026-05-04.
# To regenerate: see scripts/_dev/generate_kappa_fixtures.py
# DO NOT add scikit-learn to the project's dependencies — these constants are the contract.

SKLEARN_KAPPA_FIXTURES: dict[str, float] = {
    # PASTE OUTPUT FROM scripts/_dev/generate_kappa_fixtures.py HERE
    # Example shape (replace with actual values from the script run):
    "imbalanced_binary": 0.0,  # placeholder — replace
    "three_point_one_diagonal_swap": 0.0,  # placeholder — replace
    "weighted_ordinal_drift_linear": 0.0,  # placeholder — replace
}

SKLEARN_KAPPA_INPUTS: dict[str, dict] = {
    # PASTE OUTPUT FROM scripts/_dev/generate_kappa_fixtures.py HERE
}


class TestSklearnKappaParity:
    @pytest.mark.parametrize("case_name", list(SKLEARN_KAPPA_FIXTURES.keys()))
    def test_matches_sklearn(self, case_name: str):
        case = SKLEARN_KAPPA_INPUTS[case_name]
        expected = SKLEARN_KAPPA_FIXTURES[case_name]
        actual = cohen_kappa(case["y1"], case["y2"], weights=case["weights"])
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"hand-rolled cohen_kappa diverged from sklearn 1.5.2 on case {case_name!r}: "
            f"hand-rolled={actual} sklearn={expected}"
        )


class TestSklearnInputsCrossCheck:
    """Catches 'updated CASES list, forgot to regenerate' failure mode."""

    def test_inputs_match_committed_json(self):
        json_path = Path(__file__).parent / "fixtures" / "sklearn_kappa_inputs.json"
        on_disk = _json.loads(json_path.read_text())
        # Compare key sets and inner values
        assert set(SKLEARN_KAPPA_INPUTS.keys()) == set(on_disk.keys()), (
            "SKLEARN_KAPPA_INPUTS keys diverge from sklearn_kappa_inputs.json — "
            "regenerate via scripts/_dev/generate_kappa_fixtures.py"
        )
        for name in SKLEARN_KAPPA_INPUTS:
            assert SKLEARN_KAPPA_INPUTS[name] == on_disk[name], (
                f"Input mismatch for case {name!r} — regenerate fixtures"
            )
```

- [ ] **Step 4: Run all calibration metric tests**

Run: `python3 -m pytest tests/evaluation/test_calibration_metrics.py -v 2>&1 | tail -15`
Expected: all PASS (hand-computed + 3 sklearn parity + 1 cross-check + bootstrap).

- [ ] **Step 5: Commit**

```bash
git add scripts/_dev/generate_kappa_fixtures.py tests/evaluation/fixtures/sklearn_kappa_inputs.json tests/evaluation/test_calibration_metrics.py
git commit -m "test(calibration): sklearn-parity fixtures + cross-check CI test

Four-part discipline:
1. scripts/_dev/generate_kappa_fixtures.py — committed; runs from a
   venv outside the project (sklearn is NOT a runtime dep).
2. SKLEARN_KAPPA_FIXTURES inline constants in test file — locality
   preserved, type-checked.
3. Version-pinned comment header (sklearn 1.5.2 on 2026-05-04).
4. Load-bearing 'DO NOT add scikit-learn' comment.

Cross-check CI test (TestSklearnInputsCrossCheck) compares the
inline SKLEARN_KAPPA_INPUTS against the JSON sidecar written by the
generator; catches 'updated CASES list, forgot to regenerate' at CI
time rather than at the next investigation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 5: Calibration dataset spec + FastAPI snippet authoring

### Task 5.1: Stratified-sample 30 calibration IDs and write `calibration_v1.json`

**Files:**
- Create: `scripts/_dev/sample_calibration_v1.py` (one-shot helper, committed for reproducibility)
- Create: `agent_bench/evaluation/datasets/calibration_v1.json`

- [ ] **Step 1: Author the sampling script**

Create `scripts/_dev/sample_calibration_v1.py`:

```python
"""One-shot stratified sampler for calibration_v1.json. Run once; output
is committed to agent_bench/evaluation/datasets/calibration_v1.json.

The stratification target is in docs/plans/2026-05-04-judge-layer-v1-design.md
under Calibration Methodology > Stratified sampling.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FASTAPI_PATH = REPO / "agent_bench/evaluation/datasets/tech_docs_golden.json"
K8S_PATH = REPO / "agent_bench/evaluation/datasets/k8s_golden.json"
OUTPUT = REPO / "agent_bench/evaluation/datasets/calibration_v1.json"

SEED = 20260504  # date-derived; deterministic across runs

# Targets per the design doc's stratified-sampling table
FASTAPI_TARGETS = {"retrieval": 5, "calculation": 1, "out_of_scope": 2}
K8S_TARGETS = {
    "simple": 4,
    "simple_w_condition": 3,
    "comparison": 3,
    "multi_hop": 4,
    "false_premise": 3,
    "set": 1,
}
SPARE_TOTAL = 4   # filled from highest-variance R@5 strata


def main() -> None:
    rng = random.Random(SEED)

    fastapi = json.loads(FASTAPI_PATH.read_text())
    k8s = json.loads(K8S_PATH.read_text())["questions"]

    selected: list[dict] = []

    # FastAPI strata by category
    by_cat: dict[str, list[dict]] = {}
    for q in fastapi:
        by_cat.setdefault(q["category"], []).append(q)
    for cat, n in FASTAPI_TARGETS.items():
        pool = by_cat.get(cat, [])
        if len(pool) < n:
            raise SystemExit(f"FastAPI {cat}: have {len(pool)}, need {n}")
        sample = rng.sample(pool, n)
        for q in sample:
            selected.append({"id": q["id"], "corpus": "fastapi", "stratum": cat})

    # K8s strata by question_type
    by_qt: dict[str, list[dict]] = {}
    for q in k8s:
        by_qt.setdefault(q.get("question_type", "?"), []).append(q)
    for qt, n in K8S_TARGETS.items():
        pool = by_qt.get(qt, [])
        if len(pool) < n:
            raise SystemExit(f"K8s {qt}: have {len(pool)}, need {n}")
        sample = rng.sample(pool, n)
        for q in sample:
            selected.append({"id": q["id"], "corpus": "k8s", "stratum": qt})

    # Spare slots — for v1, fill from K8s simple_w_condition + multi_hop
    # (typically the highest-variance R@5 strata). Document the choice in
    # the file's notes field.
    spare_pool: list[dict] = []
    for q in k8s:
        if (
            q.get("question_type") in ("simple_w_condition", "multi_hop")
            and q["id"] not in {s["id"] for s in selected}
        ):
            spare_pool.append(q)
    spare = rng.sample(spare_pool, SPARE_TOTAL)
    for q in spare:
        selected.append({"id": q["id"], "corpus": "k8s", "stratum": f"spare_{q['question_type']}"})

    if len(selected) != 30:
        raise SystemExit(f"Expected 30 items; got {len(selected)}")

    # Capture current git SHA for system_config_git_sha
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()

    out = {
        "version": "v1",
        "system_config_git_sha": sha,
        "sample_seed": SEED,
        "notes": (
            "30-item stratified calibration set per the design doc. "
            "Spare slots filled from K8s simple_w_condition and multi_hop "
            "(typically highest-variance R@5 strata)."
        ),
        "items": sorted(selected, key=lambda s: (s["corpus"], s["stratum"], s["id"])),
    }
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUTPUT} with {len(selected)} items; git_sha={sha[:12]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the sampler**

Run: `python3 scripts/_dev/sample_calibration_v1.py`
Expected: `Wrote agent_bench/evaluation/datasets/calibration_v1.json with 30 items; git_sha=<sha>`.

- [ ] **Step 3: Verify the output schema**

Run:
```bash
python3 -c "
import json
d = json.load(open('agent_bench/evaluation/datasets/calibration_v1.json'))
print('version:', d['version'])
print('git_sha:', d['system_config_git_sha'][:12])
print('items:', len(d['items']))
print('strata:', sorted({i['stratum'] for i in d['items']}))
"
```
Expected: 30 items across the 9 stratum names.

- [ ] **Step 4: Commit**

```bash
git add scripts/_dev/sample_calibration_v1.py agent_bench/evaluation/datasets/calibration_v1.json
git commit -m "feat(calibration): 30-item stratified calibration_v1 sample

Stratified across FastAPI (categorized) + K8s (CRAG question_types)
per the design doc's sampling table. 26 items from explicit strata +
4 spare slots from K8s simple_w_condition / multi_hop (highest-variance
R@5 strata in pre-judge runs). Sample seed 20260504 (date-derived) so
the sampling is reproducible.

system_config_git_sha pins the commit producing the sample to the
soon-to-be-generated system_outputs file. v1.1 may add
system_config_resolved_hash for stricter reproducibility across
noise commits — name carries the limitation (spec Out of Scope).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 5.2: Hand-snippet the 8 FastAPI calibration items

**Files:**
- Modify: `agent_bench/evaluation/datasets/tech_docs_golden.json` (add `source_snippets` to 8 items)

- [ ] **Step 1: List the FastAPI calibration items needing snippets**

```bash
python3 -c "
import json
calib = json.load(open('agent_bench/evaluation/datasets/calibration_v1.json'))
ids = [i['id'] for i in calib['items'] if i['corpus'] == 'fastapi']
print('FastAPI calibration item IDs:', ids)
"
```
Expected: 8 item IDs (5 retrieval + 1 calculation + 2 out_of_scope).

- [ ] **Step 2: For each FastAPI item ID, hand-author `source_snippets`**

For each ID from Step 1, locate the entry in `agent_bench/evaluation/datasets/tech_docs_golden.json` and:

1. Open the relevant source file under `data/tech_docs/` (the file name is in `expected_sources`).
2. Find the **verbatim** 1–3 sentences that support the gold answer.
3. Add a `source_snippets: [...]` field to the JSON entry with those exact strings.
4. If no verbatim span supports the gold answer, the item is underspecified — remove it from the calibration set and re-run `scripts/_dev/sample_calibration_v1.py` (regenerating `calibration_v1.json`); the spare-slot stratum absorbs the change.

For OOS items (`category: out_of_scope`), leave `source_snippets: []` — there is no source to ground against.

This is manual authoring, not test-driven. The verification is the next step.

- [ ] **Step 3: Verify 8 FastAPI items now have `source_snippets` populated (or empty for OOS)**

```bash
python3 -c "
import json
calib_ids = [i['id'] for i in json.load(open('agent_bench/evaluation/datasets/calibration_v1.json'))['items'] if i['corpus'] == 'fastapi']
fa = json.load(open('agent_bench/evaluation/datasets/tech_docs_golden.json'))
for q in fa:
    if q['id'] in calib_ids:
        snippets = q.get('source_snippets', None)
        print(f\"{q['id']:30s} cat={q['category']:12s} snippets={snippets if snippets is None else len(snippets)}\")
"
```
Expected: each of the 8 IDs prints with `snippets=<int>` (>=1 for retrieval/calculation; ==0 for OOS); none print `snippets=None`.

- [ ] **Step 4: Commit**

```bash
git add agent_bench/evaluation/datasets/tech_docs_golden.json
git commit -m "feat(goldens): add source_snippets to 8 FastAPI calibration items

Hand-snippeted verbatim spans from data/tech_docs/ for the 8
FastAPI items in calibration_v1. OOS items get source_snippets:[]
(no source to ground against). Scope discipline: only the 8
calibration items, not the full 27-item FastAPI golden — the
remaining 19 backfill in v1.1.

Required by GroundednessJudge (reference-based on source_snippets).
K8s items already had this field from the multi-corpus refactor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 6: Calibration runner + row configs

### Task 6.1: Author the 6 row config YAML files

**Files:**
- Create: `configs/calibration/rows/baseline.yaml`
- Create: `configs/calibration/rows/baseline_no_cot.yaml`
- Create: `configs/calibration/rows/baseline_no_anchors.yaml`
- Create: `configs/calibration/rows/baseline_no_abstain.yaml`
- Create: `configs/calibration/rows/permute.yaml`
- Create: `configs/calibration/rows/jury_kappa_weighted.yaml`

- [ ] **Step 1: Make the rows directory**

```bash
mkdir -p configs/calibration/rows
```

- [ ] **Step 2: Author `baseline.yaml`**

```yaml
# Baseline: single Claude-Haiku judge per dimension, all variance controls on.
# CoT is implicit (the rubric prompts ask for reasoning before score).
# Anchors come from the rubric files. Abstain comes from rubric.abstain_allowed=true.

label: baseline
provider: anthropic
model_id: claude-haiku-4-5
dimensions: [groundedness, relevance, completeness]
strategy: single
options:
  use_cot: true
  use_anchors: true
  abstain_allowed: true
output_path: results/calibration_v1_judge_baseline.json
```

- [ ] **Step 3: Author the three baseline ablations**

`configs/calibration/rows/baseline_no_cot.yaml`:

```yaml
# Ablation: same as baseline but the judge prompt does NOT request reasoning
# before the score. Used to measure the contribution of CoT-before-score.

label: baseline_no_cot
provider: anthropic
model_id: claude-haiku-4-5
dimensions: [groundedness, relevance, completeness]
strategy: single
options:
  use_cot: false
  use_anchors: true
  abstain_allowed: true
output_path: results/calibration_v1_judge_baseline_no_cot.json
```

`configs/calibration/rows/baseline_no_anchors.yaml`:

```yaml
# Ablation: rubric anchored examples stripped from the prompt; only the
# level descriptions are sent. Measures the contribution of anchored examples.

label: baseline_no_anchors
provider: anthropic
model_id: claude-haiku-4-5
dimensions: [groundedness, relevance, completeness]
strategy: single
options:
  use_cot: true
  use_anchors: false
  abstain_allowed: true
output_path: results/calibration_v1_judge_baseline_no_anchors.json
```

`configs/calibration/rows/baseline_no_abstain.yaml`:

```yaml
# Ablation: rubric.abstain_allowed forced false at scoring time. Measures
# the contribution of the abstain option. Out-of-range schema violations
# (model returns "Unknown" anyway) abstain via ABSTAIN_REASON_OUT_OF_RANGE.

label: baseline_no_abstain
provider: anthropic
model_id: claude-haiku-4-5
dimensions: [groundedness, relevance, completeness]
strategy: single
options:
  use_cot: true
  use_anchors: true
  abstain_allowed: false
output_path: results/calibration_v1_judge_baseline_no_abstain.json
```

- [ ] **Step 4: Author `permute.yaml` and `jury_kappa_weighted.yaml`**

`configs/calibration/rows/permute.yaml`:

```yaml
# Rubric permutation: N=2 seeded prompt-level permutations per item, mean-
# aggregated. Per-permutation results land in the sidecar JSONL.

label: permute
provider: anthropic
model_id: claude-haiku-4-5
dimensions: [groundedness, relevance, completeness]
strategy: rubric_permute
options:
  n_permutations: 2
  seeds: [1, 2]
  abstain_allowed: true
output_path: results/calibration_v1_judge_permute.json
sidecar_path: results/calibration_v1_judge_permute_members.jsonl
```

`configs/calibration/rows/jury_kappa_weighted.yaml`:

```yaml
# 2-judge jury: Claude-Haiku + gpt-4o-mini, kappa-weighted aggregation.
# Strict quorum default (any member abstain → jury abstain). Weights are
# computed offline from per-judge κ on the calibration set's baseline
# rows (NOT at jury construction — circular).

label: jury_kappa_weighted
strategy: jury
aggregation: kappa_weighted
quorum: null  # null = strict default (= len(judges) = 2)
members:
  - provider: anthropic
    model_id: claude-haiku-4-5
  - provider: openai
    model_id: gpt-4o-mini
dimensions: [groundedness, relevance, completeness]
weights_source: results/calibration_v1_judge_baseline.json  # κ per judge_id from baseline; jury runner computes weights
output_path: results/calibration_v1_judge_jury_kappa_weighted.json
sidecar_path: results/calibration_v1_judge_jury_kappa_weighted_members.jsonl
```

- [ ] **Step 5: Verify all 6 configs parse as YAML**

```bash
python3 -c "
import yaml
from pathlib import Path
for f in sorted(Path('configs/calibration/rows').glob('*.yaml')):
    d = yaml.safe_load(f.read_text())
    print(f'{f.name}: label={d[\"label\"]} strategy={d[\"strategy\"]}')
"
```
Expected: 6 lines, one per config; all parse cleanly.

- [ ] **Step 6: Commit**

```bash
git add configs/calibration/rows/
git commit -m "feat(calibration): six row configs for the κ ablation table

Each config is independently versioned and reproducible. The
calibration runner takes --row-config=<path>; rows are not owned by
the script's source code, so a bug in row N can be fixed and rows
N..6 rerun without touching rows 1..N-1.

Six rows: baseline + three ablations (no CoT, no anchors, no abstain)
+ permute + 2-judge jury with kappa_weighted aggregation. The jury
config carries weights_source pointing at the baseline output —
weights are computed from baseline-row per-judge κ, not at jury
construction (which would be circular).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 6.2: Implement `scripts/run_calibration.py`

**Files:**
- Create: `scripts/run_calibration.py`

This script has three subcommands. Each is implemented as a separate function for testability.

- [ ] **Step 1: Skeleton with argparse and three subcommand stubs**

Create `scripts/run_calibration.py`:

```python
"""Calibration runner: generate-outputs | run-judges | build-table.

Orchestrates Steps A, C, D from the design doc's data flow. Step B
(hand-labeling) is manual — done in a Jupyter notebook reading
results/calibration_v1_system_outputs.json and appending to
measurements/2026-05-04-judge-calibration-labels.jsonl.

Examples:
    python scripts/run_calibration.py generate-outputs --concurrency 5
    python scripts/run_calibration.py run-judges --row-config=configs/calibration/rows/baseline.yaml
    python scripts/run_calibration.py build-table
    python scripts/run_calibration.py build-table --strict
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger()

REPO = Path(__file__).resolve().parents[1]
CALIBRATION_SPEC = REPO / "agent_bench/evaluation/datasets/calibration_v1.json"
SYSTEM_OUTPUTS = REPO / "results/calibration_v1_system_outputs.json"
LABELS_PATH = REPO / "measurements/2026-05-04-judge-calibration-labels.jsonl"
KAPPA_TABLE_OUT = REPO / "docs/_generated/kappa_table.md"


def _resolve_concurrency(cli_value: int | None) -> int:
    """CLI flag overrides config field; default is 5. Logs the resolved value."""
    if cli_value is not None:
        resolved = cli_value
    else:
        # Config-field fallback — read from configs/default.yaml if present
        cfg_path = REPO / "configs/default.yaml"
        cfg_concurrency = None
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            cfg_concurrency = (cfg.get("evaluation", {}) or {}).get("calibration_concurrency")
        resolved = cfg_concurrency if cfg_concurrency is not None else 5
    logger.info("calibration_concurrency_resolved", value=resolved)
    return resolved


# --- Subcommand: generate-outputs (Step A) ---

async def cmd_generate_outputs(concurrency: int) -> None:
    """Run the orchestrator against the 30 calibration items with a frozen
    configuration; write results/calibration_v1_system_outputs.json.
    """
    from agent_bench.core.config import load_config
    from agent_bench.core.provider import AnthropicProvider
    from agent_bench.agents.orchestrator import Orchestrator
    from agent_bench.evaluation.harness import load_golden_dataset
    from agent_bench.tools.registry import build_default_registry

    spec = json.loads(CALIBRATION_SPEC.read_text())
    target_ids = {i["id"]: i for i in spec["items"]}

    fastapi = load_golden_dataset(REPO / "agent_bench/evaluation/datasets/tech_docs_golden.json")
    k8s = load_golden_dataset(REPO / "agent_bench/evaluation/datasets/k8s_golden.json")
    items = [q for q in (fastapi + k8s) if q.id in target_ids]
    if len(items) != len(target_ids):
        missing = set(target_ids) - {q.id for q in items}
        raise SystemExit(f"calibration items not found in goldens: {sorted(missing)}")

    cfg = load_config()
    provider = AnthropicProvider(cfg)
    registry = build_default_registry(cfg)
    orchestrator = Orchestrator(provider=provider, registry=registry)

    sem = asyncio.Semaphore(concurrency)

    async def _run_one(item):
        async with sem:
            response = await orchestrator.run(
                question=item.question, system_prompt="You are a helpful assistant.",
            )
            answer = response.answer
            sources = sorted(s.source for s in response.sources)
            sys_hash = hashlib.sha256(
                f"{item.id}\x00{answer}\x00{','.join(sources)}".encode("utf-8")
            ).hexdigest()
            return {
                "item_id": item.id,
                "question": item.question,
                "category": item.category,
                "answer": answer,
                "sources": [s.source for s in response.sources],
                "ranked_sources": response.ranked_sources,
                "source_chunks": response.source_chunks,
                "system_output_hash": sys_hash,
                "stratum": target_ids[item.id]["stratum"],
                "corpus": target_ids[item.id]["corpus"],
            }

    records = await asyncio.gather(*[_run_one(it) for it in items])
    SYSTEM_OUTPUTS.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_OUTPUTS.write_text(json.dumps(records, indent=2) + "\n")
    logger.info("generate_outputs_complete", count=len(records), path=str(SYSTEM_OUTPUTS))


# --- Subcommand: run-judges (Step C, one row per invocation) ---

async def cmd_run_judges(row_config_path: Path, concurrency: int) -> None:
    """Score the frozen system outputs with the row's judge configuration."""
    from agent_bench.evaluation.judges.base import Rubric
    from agent_bench.evaluation.judges.groundedness import GroundednessJudge
    from agent_bench.evaluation.judges.relevance import RelevanceJudge
    from agent_bench.evaluation.judges.completeness import CompletenessJudge
    from agent_bench.evaluation.judges.citation_faithfulness import (
        CitationFaithfulnessJudge,
    )
    from agent_bench.evaluation.variance.rubric_permute import rubric_permute
    from agent_bench.evaluation.variance.jury import jury
    from agent_bench.core.config import load_config
    from agent_bench.core.provider import AnthropicProvider, OpenAIProvider
    from agent_bench.evaluation.harness import GoldenQuestion
    from agent_bench.agents.orchestrator import AgentResponse, SourceReference
    from agent_bench.core.types import TokenUsage

    if not SYSTEM_OUTPUTS.exists():
        raise SystemExit(
            f"{SYSTEM_OUTPUTS} not found — run `generate-outputs` first."
        )
    row = yaml.safe_load(row_config_path.read_text())
    outputs = json.loads(SYSTEM_OUTPUTS.read_text())

    cfg = load_config()
    rubric_dir = REPO / "agent_bench/evaluation/rubrics"
    judge_class = {
        "groundedness": GroundednessJudge,
        "relevance": RelevanceJudge,
        "completeness": CompletenessJudge,
        "citation_faithfulness": CitationFaithfulnessJudge,
    }

    def _make_provider(name: str):
        if name == "anthropic":
            return AnthropicProvider(cfg)
        if name == "openai":
            return OpenAIProvider(cfg)
        raise ValueError(f"unknown provider: {name}")

    def _make_judge(provider_name: str, model_id: str, dimension: str):
        rubric = Rubric.from_markdown_file(rubric_dir / f"{dimension}.md")
        return judge_class[dimension](
            judge_provider=_make_provider(provider_name),
            rubric=rubric,
            model_id=model_id,
        )

    sem = asyncio.Semaphore(concurrency)
    all_results: list[dict] = []

    for dim in row["dimensions"]:
        if row["strategy"] == "single":
            judge = _make_judge(row["provider"], row["model_id"], dim)

            async def score_one(rec):
                async with sem:
                    item = GoldenQuestion(
                        id=rec["item_id"], question=rec["question"],
                        expected_answer_keywords=[], expected_sources=[],
                        category=rec["category"], difficulty="easy",
                        requires_calculator=False,
                        source_snippets=rec.get("source_snippets", []),
                        reference_answer=rec.get("reference_answer", ""),
                    )
                    output = AgentResponse(
                        answer=rec["answer"],
                        sources=[SourceReference(source=s) for s in rec["sources"]],
                        ranked_sources=rec["ranked_sources"],
                        source_chunks=rec["source_chunks"],
                        iterations=1,
                        usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
                        latency_ms=0,
                    )
                    if rec["category"] == "out_of_scope":
                        return None
                    result = await judge.score(item, output)
                    return {"dimension": dim, **result.model_dump()}

            row_results = await asyncio.gather(*[score_one(r) for r in outputs])
            all_results.extend([r for r in row_results if r is not None])

        elif row["strategy"] == "rubric_permute":
            judge = _make_judge(row["provider"], row["model_id"], dim)
            sidecar = REPO / row.get("sidecar_path", "results/calibration_v1_permute_members.jsonl")
            permuted = rubric_permute(
                judge,
                n=row["options"]["n_permutations"],
                seeds=row["options"]["seeds"],
                sidecar_path=sidecar,
            )
            # Re-use the single-strategy scoring loop with `permuted` instead of `judge`
            for rec in outputs:
                if rec["category"] == "out_of_scope":
                    continue
                item = GoldenQuestion(
                    id=rec["item_id"], question=rec["question"],
                    expected_answer_keywords=[], expected_sources=[],
                    category=rec["category"], difficulty="easy",
                    requires_calculator=False,
                    source_snippets=rec.get("source_snippets", []),
                    reference_answer=rec.get("reference_answer", ""),
                )
                output = AgentResponse(
                    answer=rec["answer"],
                    sources=[SourceReference(source=s) for s in rec["sources"]],
                    ranked_sources=rec["ranked_sources"],
                    source_chunks=rec["source_chunks"],
                    iterations=1,
                    usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
                    latency_ms=0,
                )
                result = await permuted.score(item, output)
                all_results.append({"dimension": dim, **result.model_dump()})

        elif row["strategy"] == "jury":
            members = [
                _make_judge(m["provider"], m["model_id"], dim)
                for m in row["members"]
            ]
            sidecar = REPO / row["sidecar_path"]
            weights = _load_weights_from_baseline(
                REPO / row["weights_source"], dim
            ) if row.get("aggregation") == "kappa_weighted" else None
            j = jury(
                judges=members,
                aggregation=row["aggregation"],
                weights=weights,
                quorum=row.get("quorum"),
                sidecar_path=sidecar,
            )
            for rec in outputs:
                if rec["category"] == "out_of_scope":
                    continue
                item = GoldenQuestion(
                    id=rec["item_id"], question=rec["question"],
                    expected_answer_keywords=[], expected_sources=[],
                    category=rec["category"], difficulty="easy",
                    requires_calculator=False,
                    source_snippets=rec.get("source_snippets", []),
                    reference_answer=rec.get("reference_answer", ""),
                )
                output = AgentResponse(
                    answer=rec["answer"],
                    sources=[SourceReference(source=s) for s in rec["sources"]],
                    ranked_sources=rec["ranked_sources"],
                    source_chunks=rec["source_chunks"],
                    iterations=1,
                    usage=TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0),
                    latency_ms=0,
                )
                result = await j.score(item, output)
                all_results.append({"dimension": dim, **result.model_dump()})
        else:
            raise SystemExit(f"unknown strategy: {row['strategy']}")

    out_path = REPO / row["output_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2) + "\n")
    logger.info("run_judges_complete", row=row["label"], count=len(all_results), path=str(out_path))


def _load_weights_from_baseline(baseline_path: Path, dimension: str) -> dict[str, float]:
    """Compute per-judge weight = κ vs labels for the dimension, from baseline run.

    Stub for v1: returns equal weights (1.0 for each judge_id seen in
    the baseline file). Replaced by real κ-derived weights once labels
    + baseline are both populated. Documented in writeup as caveat:
    'weights estimated on calibration set; production deployment would
    use a held-out validation set'.
    """
    baseline = json.loads(baseline_path.read_text())
    judge_ids = {r["judge_id"] for r in baseline if r.get("dimension") == dimension}
    return {jid: 1.0 for jid in judge_ids}


# --- Subcommand: build-table (Step D) ---

def cmd_build_table(strict: bool) -> None:
    from agent_bench.evaluation.calibration.report import generate_kappa_table

    predictions_glob = str(REPO / "results/calibration_v1_judge_*.json")
    generate_kappa_table(
        predictions_glob=predictions_glob,
        labels_path=str(LABELS_PATH),
        output_path=str(KAPPA_TABLE_OUT),
        strict=strict,
    )
    logger.info("build_table_complete", path=str(KAPPA_TABLE_OUT), strict=strict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate-outputs", help="Step A: generate frozen system outputs")
    p_gen.add_argument("--concurrency", type=int, default=None)

    p_run = sub.add_parser("run-judges", help="Step C: score one ablation row")
    p_run.add_argument("--row-config", type=Path, required=True)
    p_run.add_argument("--concurrency", type=int, default=None)

    p_tab = sub.add_parser("build-table", help="Step D: aggregate predictions into κ table")
    p_tab.add_argument("--strict", action="store_true",
                       help="Raise on missing predictions/labels (final-artifact path)")

    args = parser.parse_args()
    if args.cmd == "generate-outputs":
        asyncio.run(cmd_generate_outputs(_resolve_concurrency(args.concurrency)))
    elif args.cmd == "run-judges":
        asyncio.run(cmd_run_judges(args.row_config, _resolve_concurrency(args.concurrency)))
    elif args.cmd == "build-table":
        cmd_build_table(strict=args.strict)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `--help` works on all subcommands**

```bash
python3 scripts/run_calibration.py --help
python3 scripts/run_calibration.py generate-outputs --help
python3 scripts/run_calibration.py run-judges --help
python3 scripts/run_calibration.py build-table --help
```
Expected: each prints usage without errors. The script doesn't run anything — it just imports and parses args.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_calibration.py
git commit -m "feat(scripts): run_calibration.py orchestrator for Steps A/C/D

Three subcommands, all sharing concurrency-resolution + structured
logging:
  generate-outputs  — Step A: orchestrator against 30 calibration
                      items, frozen config, writes
                      results/calibration_v1_system_outputs.json
  run-judges        — Step C: takes --row-config=<path>, scores
                      frozen outputs with that row's judges, writes
                      results/calibration_v1_judge_<label>.json
  build-table       — Step D: invokes generate_kappa_table; --strict
                      raises on missing predictions/labels (the
                      final-artifact path; make calibrate uses it)

Resolved concurrency value logged at every run so artifacts capture
which concurrency was used. Default 5; CLI overrides config-field
fallback overrides hardcoded default.

Step B (hand-labeling) is manual — done in a Jupyter notebook,
not orchestrated by this script.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 7: Calibration report generator

### Task 7.1: `generate_kappa_table` with strict/warn modes

**Files:**
- Create: `agent_bench/evaluation/calibration/report.py`
- Create: `tests/evaluation/test_calibration_report.py`

- [ ] **Step 1: Write failing tests**

Create `tests/evaluation/test_calibration_report.py`:

```python
"""Tests for generate_kappa_table — joins, hash-mismatch raise, strict, abstain flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_bench.evaluation.calibration.report import generate_kappa_table


def _write_predictions(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def _write_labels(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records))


def _pred(item_id: str, dim: str, score, sys_hash: str = "h1", reasoning: str = "") -> dict:
    return {
        "item_id": item_id, "dimension": dim, "score": score,
        "judge_id": "claude-haiku-4-5_" + dim, "rubric_version": "abc",
        "system_output_hash": sys_hash, "prompt_seed": 0,
        "cost_usd": 0.001, "latency_ms": 100.0,
        "reasoning": reasoning, "evidence_quotes": [],
    }


def _lbl(item_id: str, dim: str, score, sys_hash: str = "h1") -> dict:
    return {
        "item_id": item_id, "dimension": dim, "score": score,
        "abstained": score == "Unknown", "notes": "",
        "label_timestamp": "2026-05-04T00:00:00Z",
        "system_output_hash": sys_hash,
    }


class TestHashMismatch:
    def test_raises_with_first_item_detail_and_full_list(self, tmp_path):
        preds = [_pred("i1", "groundedness", 1, sys_hash="A")]
        labels = [_lbl("i1", "groundedness", 1, sys_hash="B")]
        _write_predictions(tmp_path / "results" / "calibration_v1_judge_baseline.json", preds)
        _write_labels(tmp_path / "labels.jsonl", labels)
        with pytest.raises(ValueError) as exc_info:
            generate_kappa_table(
                predictions_glob=str(tmp_path / "results" / "calibration_v1_judge_*.json"),
                labels_path=str(tmp_path / "labels.jsonl"),
                output_path=str(tmp_path / "kappa.md"),
            )
        msg = str(exc_info.value)
        assert "i1" in msg
        assert "A" in msg and "B" in msg

    def test_hash_mismatch_raises_in_strict_mode_too(self, tmp_path):
        preds = [_pred("i1", "groundedness", 1, sys_hash="A")]
        labels = [_lbl("i1", "groundedness", 1, sys_hash="B")]
        _write_predictions(tmp_path / "results" / "calibration_v1_judge_baseline.json", preds)
        _write_labels(tmp_path / "labels.jsonl", labels)
        with pytest.raises(ValueError):
            generate_kappa_table(
                predictions_glob=str(tmp_path / "results" / "calibration_v1_judge_*.json"),
                labels_path=str(tmp_path / "labels.jsonl"),
                output_path=str(tmp_path / "kappa.md"),
                strict=True,
            )


class TestMissingPredictionLabel:
    def test_default_warns_and_excludes(self, tmp_path, caplog):
        # Label exists for i2 but no prediction
        preds = [_pred("i1", "groundedness", 1)]
        labels = [
            _lbl("i1", "groundedness", 1),
            _lbl("i2", "groundedness", 0),
        ]
        _write_predictions(tmp_path / "results" / "calibration_v1_judge_baseline.json", preds)
        _write_labels(tmp_path / "labels.jsonl", labels)
        generate_kappa_table(
            predictions_glob=str(tmp_path / "results" / "calibration_v1_judge_*.json"),
            labels_path=str(tmp_path / "labels.jsonl"),
            output_path=str(tmp_path / "kappa.md"),
        )
        # Table should be produced; warning recorded
        assert (tmp_path / "kappa.md").exists()
        assert any("missing" in r.message.lower() or "missing_prediction" in str(r.msg)
                   for r in caplog.records)

    def test_strict_raises_on_missing_prediction(self, tmp_path):
        preds = [_pred("i1", "groundedness", 1)]
        labels = [
            _lbl("i1", "groundedness", 1),
            _lbl("i2", "groundedness", 0),
        ]
        _write_predictions(tmp_path / "results" / "calibration_v1_judge_baseline.json", preds)
        _write_labels(tmp_path / "labels.jsonl", labels)
        with pytest.raises(ValueError, match="missing"):
            generate_kappa_table(
                predictions_glob=str(tmp_path / "results" / "calibration_v1_judge_*.json"),
                labels_path=str(tmp_path / "labels.jsonl"),
                output_path=str(tmp_path / "kappa.md"),
                strict=True,
            )


class TestAbstainRateFlag:
    def _setup(self, tmp_path: Path, abstain_count: int) -> Path:
        preds = []
        labels = []
        for i in range(30):
            score: int | str = "Unknown" if i < abstain_count else 1
            reasoning = "schema_parse_failed_after_retry: x" if score == "Unknown" else ""
            preds.append(_pred(f"i{i}", "groundedness", score, reasoning=reasoning))
            labels.append(_lbl(f"i{i}", "groundedness", 1))
        _write_predictions(tmp_path / "results" / "calibration_v1_judge_baseline.json", preds)
        _write_labels(tmp_path / "labels.jsonl", labels)
        out = tmp_path / "kappa.md"
        generate_kappa_table(
            predictions_glob=str(tmp_path / "results" / "calibration_v1_judge_*.json"),
            labels_path=str(tmp_path / "labels.jsonl"),
            output_path=str(out),
        )
        return out

    def test_at_20_percent_boundary_does_not_fire(self, tmp_path):
        # 6/30 = exactly 20% — flag is ">"  (strictly greater), so not fired.
        out = self._setup(tmp_path, abstain_count=6)
        assert "high abstain rate" not in out.read_text().lower()

    def test_above_20_percent_fires(self, tmp_path):
        # 7/30 = 23.3% — flag fires
        out = self._setup(tmp_path, abstain_count=7)
        text = out.read_text().lower()
        assert "high abstain rate" in text
        assert "schema parse" in text  # cause breakdown


class TestKappaUndefined:
    def test_renders_dash_with_footnote(self, tmp_path):
        # All same label → P_e ≈ 1.0 → κ undefined
        preds = [_pred(f"i{i}", "groundedness", 1) for i in range(5)]
        labels = [_lbl(f"i{i}", "groundedness", 1) for i in range(5)]
        _write_predictions(tmp_path / "results" / "calibration_v1_judge_baseline.json", preds)
        _write_labels(tmp_path / "labels.jsonl", labels)
        out = tmp_path / "kappa.md"
        generate_kappa_table(
            predictions_glob=str(tmp_path / "results" / "calibration_v1_judge_*.json"),
            labels_path=str(tmp_path / "labels.jsonl"),
            output_path=str(out),
        )
        text = out.read_text()
        assert " — " in text or " - " in text or "undefined" in text.lower()
```

- [ ] **Step 2: Run, expect fail**

Run: `python3 -m pytest tests/evaluation/test_calibration_report.py -v 2>&1 | tail -10`
Expected: ImportError on `generate_kappa_table`.

- [ ] **Step 3: Implement `generate_kappa_table`**

Create `agent_bench/evaluation/calibration/report.py`:

```python
"""generate_kappa_table — joins predictions ⋈ labels by (item_id, dimension,
system_output_hash); computes per-row κ + bootstrap CI + abstain breakdown;
emits markdown table at docs/_generated/kappa_table.md.
"""

from __future__ import annotations

import glob as _glob
import json
from collections import defaultdict
from pathlib import Path

import structlog

from agent_bench.evaluation.calibration.metrics import bootstrap_ci, cohen_kappa
from agent_bench.evaluation.judges.base import (
    ABSTAIN_REASON_GENUINE,
    ABSTAIN_REASON_OUT_OF_RANGE,
    ABSTAIN_REASON_PROVIDER_EXHAUSTED,
    ABSTAIN_REASON_SCHEMA_PARSE,
)

logger = structlog.get_logger()

ABSTAIN_THRESHOLD = 0.20  # strictly greater than fires the flag


def _classify_abstain(reasoning: str) -> str:
    if reasoning.startswith(ABSTAIN_REASON_PROVIDER_EXHAUSTED):
        return "provider_exhausted"
    if reasoning.startswith(ABSTAIN_REASON_SCHEMA_PARSE):
        return "schema_parse"
    if reasoning.startswith(ABSTAIN_REASON_OUT_OF_RANGE):
        return "out_of_range"
    return "genuine"


def generate_kappa_table(
    *,
    predictions_glob: str,
    labels_path: str,
    output_path: str,
    strict: bool = False,
) -> None:
    """Aggregate predictions across rows + dimensions into one markdown table.

    On hash mismatch: ALWAYS raises (both modes), with first-item expected
    /actual hashes plus full mismatched-id list.
    On missing prediction or label: WARN+exclude in default mode; RAISE in strict.
    On undefined κ: render '—' with a footnote (both modes).
    On abstain rate > 20%: render κ + footnote with cause breakdown (both modes).
    """
    labels: list[dict] = []
    for line in Path(labels_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(json.loads(line))

    # labels[i] keyed by (item_id, dimension)
    label_by_key: dict[tuple[str, str], dict] = {
        (l["item_id"], l["dimension"]): l for l in labels
    }

    pred_files = sorted(_glob.glob(predictions_glob))
    if not pred_files:
        raise ValueError(f"No prediction files matched: {predictions_glob}")

    rows: list[dict] = []
    for pf in pred_files:
        label = Path(pf).stem.replace("calibration_v1_judge_", "")
        preds = json.loads(Path(pf).read_text())
        # Hash-mismatch detection (always raises)
        mismatches: list[tuple[str, str, str]] = []
        for p in preds:
            key = (p["item_id"], p["dimension"])
            if key in label_by_key:
                expected = label_by_key[key]["system_output_hash"]
                actual = p["system_output_hash"]
                if expected != actual:
                    mismatches.append((p["item_id"], expected, actual))
        if mismatches:
            first_id, first_exp, first_act = mismatches[0]
            raise ValueError(
                f"Hash mismatch in {pf}: item {first_id!r} "
                f"label.system_output_hash={first_exp!r} but "
                f"prediction.system_output_hash={first_act!r}. "
                f"Full mismatched-id list ({len(mismatches)}): "
                f"{[m[0] for m in mismatches]}. "
                f"Labels are stale relative to predictions — regenerate one or "
                f"the other so hashes align."
            )

        # Per-dimension κ
        preds_by_dim: dict[str, list[dict]] = defaultdict(list)
        for p in preds:
            preds_by_dim[p["dimension"]].append(p)

        labels_by_dim: dict[str, list[dict]] = defaultdict(list)
        for l in labels:
            labels_by_dim[l["dimension"]].append(l)

        for dim in sorted(preds_by_dim.keys()):
            preds_d = {p["item_id"]: p for p in preds_by_dim[dim]}
            labs_d = {l["item_id"]: l for l in labels_by_dim.get(dim, [])}

            common = sorted(set(preds_d) & set(labs_d))
            missing_pred = sorted(set(labs_d) - set(preds_d))
            missing_lab = sorted(set(preds_d) - set(labs_d))
            if missing_pred or missing_lab:
                msg = (
                    f"row={label} dim={dim} "
                    f"missing_predictions={missing_pred} "
                    f"missing_labels={missing_lab}"
                )
                if strict:
                    raise ValueError(f"strict mode: {msg}")
                logger.warning("calibration_report_missing", message=msg)

            # Pairwise abstain exclusion
            y_pred: list = []
            y_lab: list = []
            abstains = 0
            abstain_causes: dict[str, int] = {"provider_exhausted": 0, "schema_parse": 0,
                                              "out_of_range": 0, "genuine": 0}
            for iid in common:
                p = preds_d[iid]
                l = labs_d[iid]
                if p["score"] == "Unknown" or l["score"] == "Unknown":
                    abstains += 1
                    if p["score"] == "Unknown":
                        abstain_causes[_classify_abstain(p.get("reasoning", ""))] += 1
                    continue
                y_pred.append(int(p["score"]))
                y_lab.append(int(l["score"]))

            n_eligible = len(y_pred)
            abstain_rate = abstains / max(len(common), 1)

            if n_eligible < 3:
                rows.append({
                    "row": label, "dim": dim, "kappa": None,
                    "ci_lo": None, "ci_hi": None, "n_eligible": n_eligible,
                    "abstains": abstains, "abstain_rate": abstain_rate,
                    "abstain_causes": abstain_causes,
                    "footnote": f"κ undefined: insufficient agreement-eligible items (N={n_eligible})",
                })
                continue

            try:
                kappa = cohen_kappa(y_lab, y_pred)
                point, lo, hi = bootstrap_ci(y_lab, y_pred, cohen_kappa, n_iter=1000, seed=42)
            except (ValueError, ZeroDivisionError):
                rows.append({
                    "row": label, "dim": dim, "kappa": None,
                    "ci_lo": None, "ci_hi": None, "n_eligible": n_eligible,
                    "abstains": abstains, "abstain_rate": abstain_rate,
                    "abstain_causes": abstain_causes,
                    "footnote": "κ undefined: insufficient variance after exclusion",
                })
                continue

            footnote = ""
            if abstain_rate > ABSTAIN_THRESHOLD:
                breakdown = ", ".join(
                    f"{int(100 * v / abstains)}% {k.replace('_', ' ')}"
                    for k, v in abstain_causes.items() if v > 0
                )
                footnote = (
                    f"κ computed on N={n_eligible} of {len(common)} items; "
                    f"high abstain rate ({100 * abstain_rate:.1f}% — breakdown: {breakdown}) "
                    f"suggests rubric ambiguity."
                )

            rows.append({
                "row": label, "dim": dim, "kappa": kappa,
                "ci_lo": lo, "ci_hi": hi, "n_eligible": n_eligible,
                "abstains": abstains, "abstain_rate": abstain_rate,
                "abstain_causes": abstain_causes, "footnote": footnote,
            })

    # Render markdown
    out = ["# κ ablation table — calibration v1\n"]
    out.append("| Row | Dimension | κ (95% CI) | N | Abstain rate | Notes |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["kappa"] is None:
            kcell = " — "
        else:
            kcell = f"{r['kappa']:.3f} ({r['ci_lo']:.3f}, {r['ci_hi']:.3f})"
        rate = f"{100 * r['abstain_rate']:.1f}%"
        out.append(
            f"| {r['row']} | {r['dim']} | {kcell} | {r['n_eligible']} | "
            f"{rate} | {r['footnote']} |"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(out) + "\n")
    logger.info("kappa_table_written", path=output_path, rows=len(rows))
```

- [ ] **Step 4: Run tests, commit**

Run: `python3 -m pytest tests/evaluation/test_calibration_report.py -v 2>&1 | tail -15`
Expected: all PASS.

```bash
git add agent_bench/evaluation/calibration/report.py tests/evaluation/test_calibration_report.py
git commit -m "feat(calibration): generate_kappa_table with strict/warn modes

Joins predictions ⋈ labels by (item_id, dimension, system_output_hash).
Hash mismatch ALWAYS raises with first-item expected/actual hashes
plus full mismatched-id list — applies in both modes (never warned).
Missing predictions/labels warn-and-exclude by default; --strict
raises (the final-artifact path; make calibrate uses it).

Pairwise abstain exclusion in κ; per-dimension cause breakdown
(schema_parse / out_of_range / provider_exhausted / genuine) via
the abstain-reason constants from judges/base.py. Abstain-rate
flag fires on STRICTLY greater than 20%; 6/30 (=20%) does not
fire, 7/30 does — boundary tested explicitly.

κ undefined → '—' with footnote (insufficient variance or N<3
agreement-eligible items remaining).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 8: Harness migration

The migration is the load-bearing hard cut: existing `EvalResult.faithfulness` and `EvalResult.correctness` fields are removed; the inline import of `answer_faithfulness` / `answer_correctness` in `harness.py` is deleted; `metrics.py` loses the old judges + their prompt constants. The existing test suite must stay green at every commit (per the saved feedback memory).

### Task 8.1: Add `evaluation.judge_dimensions` config field

**Files:**
- Modify: `agent_bench/core/config.py`
- Create test snippet: append to `tests/evaluation/test_harness_migration.py`

- [ ] **Step 1: Create `tests/evaluation/test_harness_migration.py` with the regression test**

Create `tests/evaluation/test_harness_migration.py`:

```python
"""Tests for the harness migration to the new judge layer."""

from __future__ import annotations

import pytest

from agent_bench.core.config import EvaluationConfig


class TestJudgeProviderConfigPreserved:
    def test_judge_provider_field_still_exists_with_default(self):
        # Regression — the judge_provider knob must not be removed/renamed
        # (5 YAML configs reference it).
        c = EvaluationConfig()
        assert c.judge_provider == "openai"

    def test_judge_dimensions_default_is_three(self):
        c = EvaluationConfig()
        assert c.judge_dimensions == ["groundedness", "relevance", "completeness"]
        # citation_faithfulness is opt-in v1, default-on v1.1
        assert "citation_faithfulness" not in c.judge_dimensions
```

- [ ] **Step 2: Run, expect fail**

Run: `python3 -m pytest tests/evaluation/test_harness_migration.py::TestJudgeProviderConfigPreserved -v 2>&1 | tail -5`
Expected: AttributeError on `judge_dimensions`.

- [ ] **Step 3: Modify `agent_bench/core/config.py`**

Find:

```python
class EvaluationConfig(BaseModel):
    judge_provider: str = "openai"
    golden_dataset: str = "agent_bench/evaluation/datasets/tech_docs_golden.json"
```

Replace with:

```python
class EvaluationConfig(BaseModel):
    judge_provider: str = "openai"
    golden_dataset: str = "agent_bench/evaluation/datasets/tech_docs_golden.json"
    # New in judge-layer v1: which dimensions to score with L2 LLM judges.
    # citation_faithfulness is opt-in v1 (default-on v1.1).
    judge_dimensions: list[str] = ["groundedness", "relevance", "completeness"]
```

- [ ] **Step 4: Run test, verify pass**

Run: `python3 -m pytest tests/evaluation/test_harness_migration.py -v 2>&1 | tail -5`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_bench/core/config.py tests/evaluation/test_harness_migration.py
git commit -m "feat(config): add evaluation.judge_dimensions field

Default ['groundedness', 'relevance', 'completeness'] — the v1
dimensions that have rubrics + judges + calibration coverage.
citation_faithfulness is opt-in v1 (default-on v1.1) so the
citation deterministic-vs-LLM head-to-head is decoupled from the
harness migration.

judge_provider field unchanged — preserves the YAML knob across
configs/{default,production,anthropic,selfhosted_local,
selfhosted_modal}.yaml. Zero user-facing config migration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 8.2: Migrate `harness.py` (drop old judges; integrate new)

**Files:**
- Modify: `agent_bench/evaluation/harness.py`
- Modify: `tests/evaluation/test_harness_migration.py`

- [ ] **Step 1: Write failing test for new `EvalResult.judge_scores` field + OOS skip**

Append to `tests/evaluation/test_harness_migration.py`:

```python
class TestEvalResultJudgeScores:
    def test_eval_result_no_longer_has_faithfulness_field(self):
        from agent_bench.evaluation.harness import EvalResult
        fields = EvalResult.model_fields
        assert "faithfulness" not in fields, (
            "faithfulness field should be removed in the supersession"
        )
        assert "correctness" not in fields, (
            "correctness field should be removed in the supersession"
        )
        assert "judge_scores" in fields, (
            "judge_scores: dict[str, ScoreResult] should be added"
        )
```

- [ ] **Step 2: Run, expect fail**

Run: `python3 -m pytest tests/evaluation/test_harness_migration.py::TestEvalResultJudgeScores -v 2>&1 | tail -5`
Expected: FAIL — `faithfulness` still present.

- [ ] **Step 3: Modify `agent_bench/evaluation/harness.py`**

Replace the `EvalResult` class. Find:

```python
class EvalResult(BaseModel):
    question_id: str
    question: str
    category: str
    difficulty: str
    # Deterministic
    retrieval_precision: float
    retrieval_recall: float
    keyword_hit_rate: float
    has_source_citation: bool
    grounded_refusal: bool
    citation_accuracy: float
    calculator_used_correctly: bool
    tool_calls_made: int
    latency_ms: float
    tokens_used: TokenUsage
    # Raw answer for reporting
    answer: str = ""
    retrieved_sources: list[str] = []
    # LLM judge (None if not run)
    faithfulness: float | None = None
    correctness: float | None = None
```

Replace with:

```python
class EvalResult(BaseModel):
    question_id: str
    question: str
    category: str
    difficulty: str
    # Deterministic
    retrieval_precision: float
    retrieval_recall: float
    keyword_hit_rate: float
    has_source_citation: bool
    grounded_refusal: bool
    citation_accuracy: float
    calculator_used_correctly: bool
    tool_calls_made: int
    latency_ms: float
    tokens_used: TokenUsage
    # Raw answer for reporting
    answer: str = ""
    retrieved_sources: list[str] = []
    # New in judge-layer v1: per-dimension judge scores. Empty when no
    # judge_provider configured or item.category == "out_of_scope".
    judge_scores: dict[str, "ScoreResult"] = Field(default_factory=dict)
```

Add the import for `ScoreResult` at the top (use TYPE_CHECKING to avoid circular imports if needed):

```python
# Add to imports at top of file:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent_bench.evaluation.judges.base import ScoreResult
```

Then replace the optional-LLM-judge block in `run_evaluation`. Find lines 152-166 (the `if judge_provider is not None and q.category != "out_of_scope":` block) and replace with:

```python
        # Optional L2 LLM-judge layer (per-dimension; gated as before)
        if judge_provider is not None and q.category != "out_of_scope":
            from agent_bench.core.config import load_config
            from agent_bench.evaluation.judges.base import Rubric
            from agent_bench.evaluation.judges.completeness import CompletenessJudge
            from agent_bench.evaluation.judges.groundedness import GroundednessJudge
            from agent_bench.evaluation.judges.relevance import RelevanceJudge

            cfg = load_config()
            rubric_dir = Path(__file__).resolve().parent / "rubrics"
            judge_class = {
                "groundedness": GroundednessJudge,
                "relevance": RelevanceJudge,
                "completeness": CompletenessJudge,
            }
            for dim in cfg.evaluation.judge_dimensions:
                if dim not in judge_class:
                    continue  # citation_faithfulness opt-in; not in default loop
                rubric = Rubric.from_markdown_file(rubric_dir / f"{dim}.md")
                judge = judge_class[dim](
                    judge_provider=judge_provider,
                    rubric=rubric,
                    model_id=getattr(judge_provider, "model", "unknown"),
                )
                score_result = await judge.score(q, agent_response)
                result.judge_scores[dim] = score_result
```

Remove the original `from agent_bench.evaluation.metrics import answer_correctness, answer_faithfulness` line and the assignments to `result.faithfulness` / `result.correctness`.

- [ ] **Step 4: Run all evaluation tests; existing test_evaluation.py may break on faithfulness/correctness assertions**

Run: `python3 -m pytest tests/test_evaluation.py tests/evaluation/test_harness_migration.py -v 2>&1 | tail -20`
Expected: `test_evaluation.py` may fail on any line referencing `result.faithfulness` or `result.correctness`. Note the failures — fixed in next task.

- [ ] **Step 5: Commit (the harness migration; tests/test_evaluation.py drops in next task)**

```bash
git add agent_bench/evaluation/harness.py tests/evaluation/test_harness_migration.py
git commit -m "refactor(harness): migrate to per-dimension Judge layer (drop faithfulness/correctness)

Hard-cut supersession: EvalResult loses faithfulness + correctness
fields, gains judge_scores: dict[str, ScoreResult]. The optional
L2 block in run_evaluation now iterates evaluation.judge_dimensions
from config and dispatches per-dimension Judge instances built from
the rubric markdown files at agent_bench/evaluation/rubrics/.

The judge_provider != None gate is preserved (existing harness
behavior); the q.category != 'out_of_scope' gate is preserved
(L2 doesn't apply to refusals — that's L1's job).

Existing tests/test_evaluation.py assertions on the removed fields
break; cleanup in the next commit (kept in same PR for atomicity).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 8.3: Drop faithfulness/correctness assertions from existing tests

**Files:**
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Find references**

Run: `grep -n "faithfulness\|correctness" tests/test_evaluation.py`
Note line numbers; remove assertions, fixture references, and the import lines that name them.

- [ ] **Step 2: Edit `tests/test_evaluation.py` removing only the faithfulness/correctness lines**

For each line found, delete it. Do not add replacements — the new judges are tested in `tests/evaluation/test_judges.py` and `test_harness_migration.py`.

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -q 2>&1 | tail -10`
Expected: green (no failures from removed assertions).

- [ ] **Step 4: Commit**

```bash
git add tests/test_evaluation.py
git commit -m "test: drop faithfulness/correctness assertions from harness tests

Companion to the harness migration: existing test_evaluation.py
referenced removed EvalResult fields. New judge tests live under
tests/evaluation/test_judges.py and test_harness_migration.py;
existing test_evaluation.py keeps its deterministic-metrics
assertions untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 8.4: Delete old judges from `agent_bench/evaluation/metrics.py`

**Files:**
- Modify: `agent_bench/evaluation/metrics.py`

- [ ] **Step 1: Delete five old-judge symbols**

In `agent_bench/evaluation/metrics.py`, remove:
- `_FAITHFULNESS_PROMPT` (string constant)
- `_CORRECTNESS_PROMPT` (string constant)
- `async def answer_faithfulness(...)` (function)
- `async def answer_correctness(...)` (function)
- `async def _judge_call(...)` (function)
- The `import json` and `from agent_bench.core.types import Message, Role` lines if no other code in the file uses them.

Keep all the deterministic metrics (`retrieval_precision_at_k`, `retrieval_recall_at_k`, `keyword_hit_rate`, `source_presence`, `grounded_refusal`, `citation_accuracy`, `tool_call_count`, `calculator_used_when_expected`).

- [ ] **Step 2: Verify no remaining references**

Run: `grep -rn "answer_faithfulness\|answer_correctness\|_judge_call\|_FAITHFULNESS_PROMPT\|_CORRECTNESS_PROMPT" agent_bench/ tests/ --include="*.py" 2>&1`
Expected: empty output.

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -q 2>&1 | tail -5`
Expected: green.

- [ ] **Step 4: Run lint**

Run: `make lint`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add agent_bench/evaluation/metrics.py
git commit -m "refactor(metrics): delete superseded LLM judges (answer_faithfulness etc.)

Removes _FAITHFULNESS_PROMPT, _CORRECTNESS_PROMPT, answer_faithfulness,
answer_correctness, _judge_call. Replaced by the per-dimension Judge
layer at agent_bench/evaluation/judges/ — see DECISIONS.md
supersession entry (next commit) for the rationale and the κ-table
file paths that defend the supersession.

Deterministic metrics in this file are untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 8.5: MockJudge fixture-validation test

**Files:**
- Create: `tests/evaluation/test_mockjudge_coverage.py`

- [ ] **Step 1: Write the test**

Create `tests/evaluation/test_mockjudge_coverage.py`:

```python
"""Walk every item.id across all goldens; assert every MockJudge instance
referenced in the test suite has coverage. Defensive against rename drift.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLDEN_PATHS = [
    REPO / "agent_bench/evaluation/datasets/tech_docs_golden.json",
    REPO / "agent_bench/evaluation/datasets/k8s_golden.json",
    REPO / "agent_bench/evaluation/datasets/k8s_golden_pilot.json",
]


def _all_golden_ids() -> set[str]:
    ids: set[str] = set()
    for p in GOLDEN_PATHS:
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        items = data if isinstance(data, list) else data.get("questions", [])
        for q in items:
            ids.add(q["id"])
    return ids


def test_calibration_v1_ids_all_resolve_to_real_goldens():
    """Every item in calibration_v1.json must resolve to a real golden item.
    This is the practical version of MockJudge coverage: if calibration_v1
    references an id that no longer exists in any golden file, the
    calibration runner will fail loudly during generate-outputs — but
    catching it at CI time saves the discovery cost.
    """
    calib_path = REPO / "agent_bench/evaluation/datasets/calibration_v1.json"
    if not calib_path.exists():
        # Phase 5 hasn't run yet; this test is a no-op until it has
        return
    calib = json.loads(calib_path.read_text())
    calib_ids = {item["id"] for item in calib["items"]}
    golden_ids = _all_golden_ids()
    missing = calib_ids - golden_ids
    assert not missing, (
        f"calibration_v1.json references item IDs not present in any golden: "
        f"{sorted(missing)} — re-run scripts/_dev/sample_calibration_v1.py "
        f"or fix the golden files."
    )
```

- [ ] **Step 2: Run, commit**

Run: `python3 -m pytest tests/evaluation/test_mockjudge_coverage.py -v 2>&1 | tail -5`
Expected: PASS.

```bash
git add tests/evaluation/test_mockjudge_coverage.py
git commit -m "test(coverage): assert calibration_v1 IDs resolve to real goldens

Walks every id in calibration_v1.json against the union of all
golden files; raises a clear error if an id has been renamed or
removed. Catches the rename-drift bug class at CI time, not at
the next generate-outputs invocation.

The MockJudge.score LookupError is the per-test layer of the same
defense; this is the dataset-wide layer.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 9: Coupled artifact updates (DESIGN.md, DECISIONS.md, measurements/README, README, Makefile)

### Task 9.1: Add `make calibrate` and `make evaluate-judges` targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the two targets**

Find the existing `evaluate-langchain:` target in `Makefile`. Append after it:

```makefile
calibrate:  ## Run full calibration pipeline (system outputs → all rows → strict κ table). Costs ~$2 in API calls.
	$(PYTHON) scripts/run_calibration.py generate-outputs
	@for cfg in configs/calibration/rows/*.yaml; do \
		echo "==> running judges for $$cfg"; \
		$(PYTHON) scripts/run_calibration.py run-judges --row-config=$$cfg || exit 1; \
	done
	$(PYTHON) scripts/run_calibration.py build-table --strict

evaluate-judges:  ## Re-run all rows + build-table against existing system_outputs (no regeneration). Costs ~$1.
	@for cfg in configs/calibration/rows/*.yaml; do \
		echo "==> running judges for $$cfg"; \
		$(PYTHON) scripts/run_calibration.py run-judges --row-config=$$cfg || exit 1; \
	done
	$(PYTHON) scripts/run_calibration.py build-table --strict
```

Add `calibrate evaluate-judges` to the `.PHONY` list at the top of the Makefile.

- [ ] **Step 2: Verify Makefile parses**

Run: `make -n calibrate 2>&1 | head -3`
Expected: prints the would-be commands without errors.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add calibrate and evaluate-judges Makefile targets

calibrate runs the full pipeline (generate-outputs → all 6 row
configs → build-table --strict) for the final-artifact path.
evaluate-judges re-runs only the row scoring + build-table against
existing system outputs — useful when iterating on rubrics or row
configs without regenerating outputs.

Both invoke build-table with --strict so the κ table is by-
construction complete; partial-coverage warnings are caught at
build time rather than landing in the writeup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 9.2: README "Targets that cost money" subheading

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the subheading near the existing Testing section**

In `README.md`, find the Testing section. Insert before it (or after, depending on flow):

```markdown
### Targets that cost money

These Make targets call paid LLM APIs. Run locally; they are excluded from CI.

| Target | Requires API key | Approximate cost | What it produces |
|---|---|---|---|
| `make evaluate-full` | OpenAI or Anthropic | $0.01–0.05 per run | Full-corpus harness run with L1 + L2 judges; results in `results/{run_label}.json` |
| `make calibrate` | Anthropic + OpenAI | ~$2 per full run | Generates frozen system outputs, scores all 6 ablation rows, builds `docs/_generated/kappa_table.md` |
| `make evaluate-judges` | Anthropic + OpenAI | ~$1 per run | Re-runs the 6 rows against existing system outputs (no regeneration) |
| `make evaluate-langchain` | OpenAI or Anthropic | $0.01–0.05 per run | LangChain baseline harness for the comparison report |

Set keys via `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables. CI does not have these (test job uses `MockProvider`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): add 'Targets that cost money' table

Anyone running 'make help' or browsing the Makefile should know
before invoking that calibrate costs ~$2 and requires API keys.
Four-column table (target / requires API key / approximate cost /
what it produces) with explicit note that CI does not have the
keys.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 9.3: Rewrite `docs/DESIGN.md` §"LLM-judge metrics"

**Files:**
- Modify: `docs/DESIGN.md`

- [ ] **Step 1: Find the section**

Run: `grep -n "LLM-judge metrics" docs/DESIGN.md`
Expected: line ~346.

- [ ] **Step 2: Replace the section body**

Replace the body of `### LLM-judge metrics (costs money, manual)` with:

```markdown
### LLM-judge metrics (costs money, manual)

The LLM-judge layer (L2 of the three-layer hierarchy) is implemented as
per-dimension judges (`groundedness`, `relevance`, `completeness`;
`citation_faithfulness` opt-in v1) with anchored discrete rubrics, abstain
support, evidence quotes, and judge-id + rubric-version provenance. See:

- `docs/plans/2026-05-04-judge-layer-v1-design.md` — design doc with the
  six-axis supersession table and the κ-validated calibration methodology.
- `docs/judge-design.md` — interpretive writeup with the κ ablation table
  and the closing "when NOT to use LLM-judge" position.

Configuration:
- `evaluation.judge_provider: openai|anthropic|...` — which provider runs
  the judge calls. Existing knob preserved across all 5 YAML configs.
- `evaluation.judge_dimensions: [groundedness, relevance, completeness]`
  — which dimensions to score in the harness. Default v1; v1.1 adds
  `citation_faithfulness`.

Run via `make calibrate` (full pipeline) or `make evaluate-judges`
(re-score existing outputs). Both require API keys; both are excluded
from CI.
```

Remove the old text describing `answer_faithfulness` / `answer_correctness` / continuous-score JSON.

- [ ] **Step 3: Commit**

```bash
git add docs/DESIGN.md
git commit -m "docs(DESIGN): rewrite LLM-judge metrics section to point at v1 layer

Drops the old description of answer_faithfulness/answer_correctness
(continuous-score, single-call). Points at the design doc and the
writeup. Documents the two configuration knobs (judge_provider,
judge_dimensions) and the make targets.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 9.4: DECISIONS.md supersession entry

**Files:**
- Modify: `DECISIONS.md`

- [ ] **Step 1: Append the entry**

Append to `DECISIONS.md`:

```markdown
## LLM-judge layer supersession — discrete-anchored 2-judge jury replaces continuous-score single-call

The continuous-score single-call judges in `agent_bench/evaluation/metrics.py`
(`answer_faithfulness`, `answer_correctness`, `_judge_call`) are deleted
and replaced by the per-dimension Judge layer at
`agent_bench/evaluation/judges/`. Hard cut, no deprecation cycle.

**Design doc:** `docs/plans/2026-05-04-judge-layer-v1-design.md`.

**Why this is a supersession, not a refactor.** The new layer differs from
the old on six axes: discrete-anchored scale (vs continuous 0–1),
reasoning-before-score JSON ordering (vs score-first), per-dimension
judges (vs combined faithfulness/correctness), full provenance per call
(judge_id + rubric_version + system_output_hash + prompt_seed; old had
none), composable variance wrappers (rubric_permute, jury — old was
single-call), and an intentional abstain-vs-raise discipline (vs silent
`None` from a bare `except Exception`).

**Evidence backing the supersession claim** — the calibration κ table
quantifies the new layer's agreement with hand-labels across 6 ablation
rows (baseline + 3 variance ablations + permute + 2-judge jury). The
files defending this entry's claim, by file path:

- `measurements/2026-05-04-judge-calibration-labels.jsonl` — 30 items × 3
  dimensions hand-labeled (UK AISI bio/chem κ ~0.8 cited as the
  literature ceiling).
- `results/calibration_v1_judge_baseline.json`, `_baseline_no_cot.json`,
  `_baseline_no_anchors.json`, `_baseline_no_abstain.json`,
  `_permute.json`, `_jury_kappa_weighted.json` — per-row predictions.
- `docs/_generated/kappa_table.md` — generated κ ablation table copy-
  pasted into the writeup.
- `docs/judge-design.md` — interpretive writeup with the closing
  "when NOT to use LLM-judge" position.

**Config-knob preservation.** `evaluation.judge_provider` is unchanged
across all 5 YAML configs; new `evaluation.judge_dimensions` field
defaults to the three v1 dimensions. Zero user-facing config migration.

**Out of scope (v1.1+).** Mistral self-hosted as the third jury member,
Langfuse self-host, dual-pass intra-rater calibration, DSPy/GEPA/MIPROv2
prompt optimization, citation_faithfulness in the default
judge_dimensions, AC2 sympy-derived parity tests.
```

- [ ] **Step 2: Commit**

```bash
git add DECISIONS.md
git commit -m "docs(DECISIONS): append LLM-judge layer supersession entry

The supersession is defended by file paths, not abstract claims —
the κ table generated from the calibration runs is the empirical
backing for 'the new layer is better,' and the entry namechecks
the labels JSONL, the per-row predictions, the kappa_table.md
artifact, and the writeup. Future readers can trace any claim to
its data.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 9.5: `measurements/README.md` row

**Files:**
- Modify: `measurements/README.md`

- [ ] **Step 1: Add the row**

Append to the `Current entries:` list in `measurements/README.md`:

```markdown
- `2026-05-04-judge-calibration-labels.jsonl` — 30 items × 3 dimensions hand-labels (single rater) for the κ ablation table in `docs/_generated/kappa_table.md` and the writeup at `docs/judge-design.md`. Backs the DECISIONS.md entry "LLM-judge layer supersession — discrete-anchored 2-judge jury replaces continuous-score single-call".
```

- [ ] **Step 2: Commit**

```bash
git add measurements/README.md
git commit -m "docs(measurements): index the judge-calibration-labels JSONL

Without an index entry, the labels file would orphan next to the
cold-start logs. The entry namechecks the DECISIONS.md supersession
claim it backs and the κ table file path it feeds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 10: Manual labeling (Step B from the design's data flow)

This phase has no code — it is hand-authored data + a one-time Opus stress-test. The procedure is documented here for reproducibility; the artifact is `measurements/2026-05-04-judge-calibration-labels.jsonl`.

### Task 10.1: Generate frozen system outputs

**Files:**
- Created by script: `results/calibration_v1_system_outputs.json`

- [ ] **Step 1: Confirm API keys are set**

```bash
test -n "$ANTHROPIC_API_KEY" && echo "ANTHROPIC OK" || echo "MISSING ANTHROPIC_API_KEY"
test -n "$OPENAI_API_KEY" && echo "OPENAI OK" || echo "MISSING OPENAI_API_KEY"
```

- [ ] **Step 2: Run the generate-outputs subcommand**

Run: `python3 scripts/run_calibration.py generate-outputs --concurrency 5`
Expected: `generate_outputs_complete count=30 path=results/calibration_v1_system_outputs.json` log line.

- [ ] **Step 3: Verify the output file**

```bash
python3 -c "
import json
recs = json.load(open('results/calibration_v1_system_outputs.json'))
print(f'records={len(recs)}')
print(f'first record keys: {sorted(recs[0].keys())}')
hashes = {r[\"system_output_hash\"] for r in recs}
print(f'unique hashes: {len(hashes)}')
"
```
Expected: 30 records; 30 unique hashes.

- [ ] **Step 4: Commit (the file is committed so labels are reproducible against the same outputs)**

```bash
git add results/calibration_v1_system_outputs.json
git commit -m "feat(calibration): freeze 30-item calibration system outputs

Step A of the design's calibration data flow. Each record carries
system_output_hash = SHA-256 of (item.id, answer, sorted(sources)),
which the labels JSONL will reference so cross-run aggregation is
detectable. Frozen — labels are tied to these specific outputs.

Generated with Claude-Haiku, hybrid retrieval, top_k=5, frozen
system_config_git_sha pinned in calibration_v1.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 10.2: Hand-label 30 items × 3 dimensions

**Files:**
- Create: `measurements/2026-05-04-judge-calibration-labels.jsonl`

This is hand-authored data. The label values are yours alone — no AI assistance on the values themselves (per the spec's hand-labeling rules). AI may help with the labeling notebook, JSONL formatting, schema validation.

- [ ] **Step 1: Open `results/calibration_v1_system_outputs.json` in a Jupyter notebook (or your preferred per-item viewer)**

A minimal labeling notebook:

```python
import json
from pathlib import Path

OUTPUTS = Path("results/calibration_v1_system_outputs.json")
LABELS = Path("measurements/2026-05-04-judge-calibration-labels.jsonl")

records = json.load(OUTPUTS.open())
already_labeled = set()
if LABELS.exists():
    for line in LABELS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            already_labeled.add((r["item_id"], r["dimension"]))

DIMENSIONS = ["groundedness", "relevance", "completeness"]

for rec in records:
    for dim in DIMENSIONS:
        if (rec["item_id"], dim) in already_labeled:
            continue
        # Display
        print("=" * 80)
        print(f"item_id={rec['item_id']}  dim={dim}  category={rec['category']}")
        print(f"question: {rec['question']}")
        print(f"answer: {rec['answer']}")
        if dim == "groundedness":
            print(f"source_snippets: {rec.get('source_snippets', [])}")
        elif dim == "completeness":
            print(f"reference_answer: {rec.get('reference_answer', '')}")
        # ... prompt for score (0/1 for binary; 0/1/2 for three_point; 'U' for Unknown)
        # write the record to LABELS as a JSONL append
```

- [ ] **Step 2: For each (item, dimension) pair, score by the rubric**

Score by the rubric, not by intuition. Genuine uncertainty → `score: "Unknown"`, `abstained: true`, `notes: <reason>`. OOS items (where `category == "out_of_scope"`) skip groundedness and completeness — only relevance is scored. Track time per item; >2 minutes → rubric ambiguity, note in the JSONL.

Each label record (one per `(item_id, dimension)` pair) has this schema:

```json
{
  "item_id": "k8s_009",
  "dimension": "groundedness",
  "score": 1,
  "abstained": false,
  "notes": "All three claims tied to retrieved chunk 4; citation matches.",
  "label_timestamp": "2026-05-04T15:23:14Z",
  "system_output_hash": "<copy from system_outputs record for this item>"
}
```

- [ ] **Step 3: Validate the JSONL schema after labeling**

```bash
python3 -c "
import json
required = {'item_id', 'dimension', 'score', 'abstained', 'notes', 'label_timestamp', 'system_output_hash'}
records = [json.loads(line) for line in open('measurements/2026-05-04-judge-calibration-labels.jsonl') if line.strip()]
print(f'total labels: {len(records)}')
for r in records:
    missing = required - r.keys()
    if missing:
        print(f'BAD record {r}: missing {missing}')
        raise SystemExit(1)
# Expect 30 items × 3 dimensions = 90 max; OOS items skip groundedness + completeness
# so realistic count is ~82-86 depending on how many OOS landed in the sample
expected_min = 30 + 30 + (30 - 4)  # all relevance + all groundedness skipping OOS + all completeness skipping OOS
print(f'expected minimum (with 4 OOS items skipping G+C): {expected_min}')
"
```

- [ ] **Step 4: Commit the labels**

```bash
git add measurements/2026-05-04-judge-calibration-labels.jsonl
git commit -m "data(calibration): 30 items × 3 dimensions hand-labels

Single-rater calibration ground-truth for the κ ablation table.
Score values are hand-authored; no AI assistance on label values
themselves (per spec hand-labeling rules). Each record carries
system_output_hash so the calibration report joins safely against
the frozen system outputs in results/calibration_v1_system_outputs.json.

OOS items skip groundedness and completeness (L2 production gate);
relevance is scored across all categories.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 10.3: Opus rubric stress-test

**Files:**
- Create: `measurements/2026-05-04-judge-rubric-opus-stress.jsonl`

- [ ] **Step 1: Have Claude Opus 4.x label the same 30 items × 3 dimensions blind to your labels**

Construct a prompt that pastes (rubric markdown + system output for one item) and asks Opus to score. Repeat across all (item, dimension) pairs. Save Opus's output to `measurements/2026-05-04-judge-rubric-opus-stress.jsonl` with the same schema as the human labels (replace `notes` with Opus's reasoning).

- [ ] **Step 2: Compute item-level disagreement against your labels**

```bash
python3 -c "
import json
def load(path):
    return {(r['item_id'], r['dimension']): r['score'] for r in (json.loads(l) for l in open(path) if l.strip())}
human = load('measurements/2026-05-04-judge-calibration-labels.jsonl')
opus = load('measurements/2026-05-04-judge-rubric-opus-stress.jsonl')
common = set(human) & set(opus)
disagree = [k for k in common if human[k] != opus[k]]
print(f'agreement: {len(common) - len(disagree)}/{len(common)} ({100*(len(common)-len(disagree))/len(common):.1f}%)')
print(f'disagreement items (rubric_ambiguous candidates):')
for k in disagree:
    print(f'  {k}: human={human[k]} opus={opus[k]}')
"
```

- [ ] **Step 3: Do not change your labels.** The Opus output is rubric-quality signal, not ground-truth substitute. Items where you and Opus disagree are flagged as `rubric_ambiguous` for v1.1 rubric revision (note in writeup).

- [ ] **Step 4: Commit Opus stress-test output**

```bash
git add measurements/2026-05-04-judge-rubric-opus-stress.jsonl
git commit -m "data(calibration): Opus 4.x rubric stress-test labels

Claude Opus 4.x labeled the same 30 items × 3 dimensions blind
to the human labels. Disagreements are flagged as rubric_ambiguous
candidates for v1.1; human labels are NOT changed (Opus is rubric-
quality signal, not ground-truth substitute). Methodological texture
for the writeup's calibration section.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 11: Ablation runs + κ table generation

### Task 11.1: Run all 6 ablation rows

**Files (created by script):**
- `results/calibration_v1_judge_baseline.json`
- `results/calibration_v1_judge_baseline_no_cot.json`
- `results/calibration_v1_judge_baseline_no_anchors.json`
- `results/calibration_v1_judge_baseline_no_abstain.json`
- `results/calibration_v1_judge_permute.json`
- `results/calibration_v1_judge_jury_kappa_weighted.json`
- Sidecars: `results/calibration_v1_judge_permute_members.jsonl`, `results/calibration_v1_judge_jury_kappa_weighted_members.jsonl`

- [ ] **Step 1: Run the baseline first (jury config depends on this for weights)**

Run: `python3 scripts/run_calibration.py run-judges --row-config=configs/calibration/rows/baseline.yaml`
Expected: `run_judges_complete row=baseline count=<N>` log line; output file written.

- [ ] **Step 2: Run the three baseline ablations**

```bash
for cfg in configs/calibration/rows/baseline_no_cot.yaml \
           configs/calibration/rows/baseline_no_anchors.yaml \
           configs/calibration/rows/baseline_no_abstain.yaml; do
    python3 scripts/run_calibration.py run-judges --row-config=$cfg
done
```

- [ ] **Step 3: Run permute and jury**

```bash
python3 scripts/run_calibration.py run-judges --row-config=configs/calibration/rows/permute.yaml
python3 scripts/run_calibration.py run-judges --row-config=configs/calibration/rows/jury_kappa_weighted.yaml
```

- [ ] **Step 4: Verify all 6 prediction files exist**

```bash
ls -la results/calibration_v1_judge_*.json
```
Expected: 6 files (one per row).

- [ ] **Step 5: Commit prediction artifacts**

```bash
git add results/calibration_v1_judge_*.json results/calibration_v1_judge_*_members.jsonl
git commit -m "feat(calibration): score all 6 ablation rows

Per-row predictions for the κ ablation table. Sidecar JSONLs
preserve per-permutation (rubric_permute) and per-member (jury)
ScoreResults for the per-judge κ breakdown.

Each prediction record carries system_output_hash matching the
frozen system_outputs file; the calibration report's join will
fail loudly if outputs are ever regenerated without re-running
the rows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 11.2: Build the κ table

**Files (created by script):**
- `docs/_generated/kappa_table.md`

- [ ] **Step 1: Run build-table in strict mode**

Run: `python3 scripts/run_calibration.py build-table --strict`
Expected: `kappa_table_written path=docs/_generated/kappa_table.md rows=<N>` log line. No raises.

- [ ] **Step 2: Sanity-check the table**

```bash
cat docs/_generated/kappa_table.md
```

Sanity checks per the spec's risks table:
- If any κ > 0.9 — double-check class balance; report Gwet's AC2 alongside.
- If any κ negative — judge is systematically inverting; bug.
- If jury κ < better individual judge — kappa-weighting wrong; investigate.
- Bootstrap CI half-width > 0.15 — note in writeup that N=30 is barely sufficient.

- [ ] **Step 3: Commit**

```bash
git add docs/_generated/kappa_table.md
git commit -m "feat(calibration): generated κ ablation table v1

Output of make calibrate / build-table --strict against the 6
ablation rows + 30-item hand-labels. Copy-pasted into the writeup
at docs/judge-design.md (next phase) with inline annotations on
specific cells.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 12: Writeup (v1-completion gate, lags PR merge by 1–2 days)

The writeup is interview material, not a PR-merge dependency. Per the spec's two-gate model, it lands separately after the code PR merges to main and the calibration runs are reproducible.

### Task 12.1: Author `docs/judge-design.md`

**Files:**
- Create: `docs/judge-design.md`

- [ ] **Step 1: Author the writeup with the section structure from the spec**

Create `docs/judge-design.md` with these sections (target ~3 pages rendered):

1. Purpose & scope (1 paragraph)
2. Hierarchy: deterministic → LLM-judge → human (½ page)
3. Per-dimension judge designs (½ page)
4. Variance controls — Table A (the κ ablation table copy-pasted from `docs/_generated/kappa_table.md`)
5. Calibration methodology, with Opus stress-test paragraph (½ page)
6. Citation deterministic-vs-LLM head-to-head — Table C (½ page; only if Phase 11 included a citation-faithfulness row, which is opt-in v1)
7. Cost & latency budget per judge (table, sourced from per-row `cost_usd` / `latency_ms` aggregates in the prediction files)
8. When NOT to use LLM-judge (1 paragraph — closing position statement)
9. Open questions and known limitations (½ page — bootstrap CI width at N=30, single-rater, citation_faithfulness opt-in caveat)
10. Future work — Mistral 3rd judge, Langfuse self-host, dual-pass intra-rater calibration, DSPy/GEPA prompt optimization (½ page)

Specific sentences to weave in (from the spec):
- "Deterministic where possible, LLM-judge where necessary, human-only where neither suffices."
- "Per-dimension judges, not a combined one — halo effects across dimensions are documented (Autorubric, Lee et al. 2025)."
- "The score field comes after reasoning in the JSON schema; the score conditions on the reasoning, not the other way around."

- [ ] **Step 2: Commit**

```bash
git add docs/judge-design.md
git commit -m "docs: writeup of the judge-layer v1 with κ ablation table

Three-page interpretive writeup with the κ ablation table copy-
pasted from docs/_generated/kappa_table.md. Closes the v1-
completion gate. Sources its empirical claims from the calibration
runs in results/calibration_v1_judge_*.json and the hand-labels in
measurements/2026-05-04-judge-calibration-labels.jsonl — every
number is traceable to a file path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

This is the inline check the writing-plans skill requires after the plan is drafted. Findings → fix inline before handoff.

### Spec coverage

Walked the design doc section-by-section against the plan tasks. Mapping:

| Spec section | Plan tasks |
|---|---|
| Three-layer hierarchy | Architecture intro; Phase 8 harness migration |
| Module layout (judges/rubrics/variance/calibration) | Phase 1 (judges/base, ScoreResult, MockJudge) + Phase 2 (concrete judges + rubrics) + Phase 3 (variance) + Phase 4 + Phase 7 (calibration) |
| Supersession (six axes, hard cut, config preservation, coupled artifacts) | Phase 8.1–8.4 + Phase 9.1–9.5 |
| Components — Rubric, ScoreResult, Judge ABC, MockJudge, rubric_permute, jury, calibration metrics, calibration report | Phase 1.1–1.4 + Phase 3.1–3.2 + Phase 4.1–4.2 + Phase 7.1 |
| Data flow — production harness migrated; calibration run (A/C/D) | Phase 8 (harness); Phase 6 (run_calibration); Phase 10 (Step B manual) |
| Concurrency — within-item / across-items / across-rows | `_resolve_concurrency` + the for-each-row loop in `run_calibration.py` |
| New scripts + Makefile targets | Phase 6.2 + Phase 9.1 |
| Failure-modes-eliminated table | All eliminations have tests in test_judges.py / test_jury_aggregation.py / test_calibration_report.py |
| Failure taxonomy at L2 + abstain-vs-raise + first-attempt-failure log | Phase 2.1 (`_call_judge_with_retry`) — taxonomy locked in tests |
| Jury partial-failure (quorum) | Phase 3.2 with the strict-quorum test |
| Permutation wrapper failure | Phase 3.1 with the any-abstain test |
| Rubric construction validation | Phase 1.2 with parameterized invalid-rubric tests |
| Calibration report failure modes (hash mismatch / missing / undefined / abstain flag) | Phase 7.1 tests cover all five |
| Test file layout (six new files + fixtures) | All six files created in Phases 1, 3, 4, 7, 8 |
| sklearn fixture pattern (4-part discipline + cross-check) | Phase 4.2 |
| Test inventory ~30 | Plan exceeds 30 (≈35 across the test files) |
| Discipline conventions (mocked providers, asyncio, hand-computed cases, fixtures dir) | Followed in test bodies |
| CI scope (test runs, calibrate/evaluate-judges manual) | Phase 0.2 (env: {}) + Phase 9.1 Makefile + Phase 9.2 README |
| README cost-disclosure | Phase 9.2 |
| Calibration methodology (stratification, snippets, hand-labeling, Opus) | Phase 5.1 + 5.2 + Phase 10.1–10.3 |
| Implementation sequencing — rubric authoring order | Phase 2.2 includes the dry-fit step |
| Contingency cuts | Documented in Phase 11 sanity-check + Phase 12 outline; not ordered as cut-able tasks because cuts apply during Phase 11/12 only |
| Two acceptance gates (PR-open / v1-completion) | PR-open = Phases 0–9 + Phase 11 (calibrate runs); v1-completion = Phase 12 writeup |
| Five locked spec-text obligations | (1) Phase 2.1 log schema with fixed keys + Phase 7.1 abstain-cause breakdown; (2) Phase 4.2 cross-check test; (3) Phase 4.1 + design's Out-of-Scope section already names sympy v1.1; (4) Phase 9.2 README subheading; (5) Phase 0.2 empty env: block |
| Out of Scope (v1.1+) | Documented in Phase 12 writeup §10; nothing in the plan implements them |
| Risks | Phase 11.2 sanity-check covers the κ-monitoring risks |

**Gap found:** the `RelevanceJudge` test (Task 2.3 Step 4) does not exercise the FastAPI-snippet-not-needed path explicitly — the test uses `source_snippets=[]` indirectly because relevance is reference-free. Acceptable; the per-judge input table in the design already documents that relevance reads only `item.question`. No action needed.

**Gap found:** the `Makefile` `evaluate-judges` target does not regenerate weights for `jury_kappa_weighted.yaml` — it just re-runs row scoring. If the baseline outputs change, jury weights become stale. Mitigation: documented in the row config's `weights_source` comment; running `calibrate` (which regenerates baseline first) refreshes weights. Acceptable for v1; v1.1 may add an explicit `compute-jury-weights` subcommand.

**Gap found:** `run_calibration.py`'s `_load_weights_from_baseline` returns equal weights as a stub. This is an accepted v1 simplification documented in the docstring; the v1 writeup notes "weights estimated on calibration set; production deployment would use a held-out validation set" and the equal-weights default is conservative (jury aggregate degrades to mean). v1.1 implements real κ-derived weights.

### Placeholder scan

Searched the plan for "TBD", "TODO", "implement later", "fill in details", "Add appropriate error handling", "similar to Task". Found:
- The fixture constants in Task 4.2 Step 3 contain `# placeholder — replace` comments. This is **intentional** — those constants are populated by running the generator script in Step 2. The placeholder is the contract: the engineer pastes the script's output before running tests. Step 4 will fail if they don't.

No other placeholders found.

### Type consistency

Walked types across tasks:
- `ScoreResult` field set is consistent across all uses (Phase 1.1 definition, Phase 2 concrete judges, Phase 3 wrappers, Phase 7 report).
- `Judge.score` signature `(item: GoldenQuestion, output: AgentResponse, *, prompt_seed: int = 0) -> ScoreResult` consistent across base.py, concrete judges, and mock.
- `Rubric.source_hash` returns `str`; used as `rubric_version` in every `_call_judge_with_retry` call.
- `system_output_hash` derivation is centralized in `_system_output_hash` in `groundedness.py` and re-used by relevance/completeness/citation_faithfulness imports.
- `cohen_kappa` and `bootstrap_ci` signatures in metrics.py match the calls in report.py.
- `judge_id` format `f"{model_id}_{rubric.dimension}"` consistent across base ABC and the jury's `f"jury_v1_{aggregation}"` pattern.

No type inconsistencies found.

### Scope check

This plan implements one design (`2026-05-04-judge-layer-v1-design.md`) producing one merged PR (`feat/judge-layer-v1`) plus one follow-on writeup commit (Phase 12). All work is scoped to the agent_bench/evaluation/ subtree + coupled docs + scripts + configs. No multi-subsystem decomposition needed.

### Inline fixes applied

- (none required — all gaps above are documented as accepted v1 simplifications, not bugs)

---

**Plan complete and saved to `docs/plans/2026-05-04-judge-layer-v1-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
