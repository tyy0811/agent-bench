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
    a = adapter.convert_legacy_file(
        FIXTURES / "results_mini.json", out_dir=tmp_path / "a", **kwargs
    )
    b = adapter.convert_legacy_file(
        FIXTURES / "results_mini.json", out_dir=tmp_path / "b", **kwargs
    )
    pd.testing.assert_frame_equal(a, b)


# --- Robustness: clear errors instead of bare KeyError on bad input ---


def test_empty_input_raises_clear_error(tmp_path):
    src = tmp_path / "empty.json"
    src.write_text("[]")
    with pytest.raises(ValueError, match="no rows"):
        adapter.convert_legacy_file(
            src,
            golden_path=FIXTURES / "golden_mini_fastapi.json",
            config_id="custom-openai-legacy",
            out_dir=tmp_path,
        )


def test_unknown_question_id_raises_clear_error():
    rec = dict(_results()[0], question_id="does_not_exist")
    with pytest.raises(ValueError, match="not in golden"):
        adapter.rows_from_result(rec, _meta(), _golden_fastapi())


def test_record_missing_question_id_raises_clear_error():
    rec = {k: v for k, v in _results()[0].items() if k != "question_id"}
    with pytest.raises(ValueError, match="missing 'question_id'"):
        adapter.rows_from_result(rec, _meta(), _golden_fastapi())


def test_missing_metric_field_raises_contextual_error(tmp_path):
    bad = {k: v for k, v in _results()[0].items() if k != "retrieval_precision"}
    src = tmp_path / "bad.json"
    src.write_text(json.dumps([bad]))
    with pytest.raises(ValueError, match="missing field"):
        adapter.convert_legacy_file(
            src,
            golden_path=FIXTURES / "golden_mini_fastapi.json",
            config_id="custom-openai-legacy",
            out_dir=tmp_path,
        )


def test_nested_golden_without_questions_key_raises(tmp_path):
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"corpus": "k8s"}))
    with pytest.raises(ValueError, match="no 'questions' key"):
        adapter.load_golden(g)


def test_null_answer_does_not_crash():
    # A record with JSON null answer must not raise TypeError (Codex review).
    rec = dict(_results()[0], answer=None)
    rows = adapter.rows_from_result(rec, _meta(), _golden_fastapi())
    assert {r["metric"] for r in rows} == {"p_at_5", "r_at_5", "khr"}
    assert all(r["refused"] is None for r in rows)


def test_null_cluster_source_rejected(tmp_path):
    g = tmp_path / "g.json"
    g.write_text(
        json.dumps(
            {
                "corpus": "k8s",
                "questions": [
                    {
                        "id": "x",
                        "category": "retrieval",
                        "requires_calculator": False,
                        "expected_sources": ["a.md"],
                        "question_type": None,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="cluster source"):
        adapter.load_golden(g)


def test_convert_envelopes_refuses_mixed_run_ids(tmp_path):
    recs = json.loads((FIXTURES / "results_mini.json").read_text())

    def _env(path, run_id):
        path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "timestamp": "2026-06-15T10:00:00+00:00",
                    "config_id": "custom-mock+00000000",
                    "code_version": "x",
                    "dataset_version": "sha-deadbeef",
                    "epoch": 1,
                    "results": recs,
                }
            )
        )

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _env(a, "01HZXJ5M8N9PQRSTVWXYZ01234")
    _env(b, "01HZXJ5M8N9PQRSTVWXYZ09999")
    with pytest.raises(ValueError, match="run_ids"):
        adapter.convert_envelopes(
            [a, b], golden_path=FIXTURES / "golden_mini_fastapi.json", out_dir=tmp_path / "out"
        )


def test_nested_out_of_scope_clusters_by_question_type():
    # Pre-registered design (stats-design.md section 2.1): K8s clusters by
    # question_type, so an out_of_scope k8s row clusters under its type, not
    # the literal "out_of_scope". _results()[1] is the OOS refusal record.
    golden = adapter.load_golden(FIXTURES / "golden_mini_k8s.json")
    rec = dict(_results()[1], question_id="mini_k3")
    rows = adapter.rows_from_result(rec, _meta(), golden)
    assert {r["metric"] for r in rows} == {"refusal_correct"}
    assert {r["cluster_id"] for r in rows} == {"false_premise"}
