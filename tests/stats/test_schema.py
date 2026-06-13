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
