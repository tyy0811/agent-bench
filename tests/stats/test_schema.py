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
