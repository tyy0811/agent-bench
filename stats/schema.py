"""Results-table schema for the v3.1 statistics layer.

Restated from the eval-statistics-engine plan section 5.1.
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

        for col in (
            "config_id",
            "code_version",
            "dataset_version",
            "question_id",
            "cluster_id",
            "metric",
        ):
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
