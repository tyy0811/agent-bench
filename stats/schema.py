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
