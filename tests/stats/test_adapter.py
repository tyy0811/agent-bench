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
