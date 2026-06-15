"""Epoch runner tests. Everything here runs keyless via MockProvider."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from stats import schema
from stats_adapters import from_results_json as adapter

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

import run_epochs  # noqa: E402

ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
FIXTURES = Path(__file__).parent / "fixtures"
PYTHON = sys.executable


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


def test_mock_epoch_run_end_to_end(tmp_path, mini_store, monkeypatch):
    monkeypatch.setitem(
        run_epochs.REGISTRY,
        "custom-mock",
        {
            "entry": "custom",
            "config": FIXTURES / "mock_config.yaml",
            "corpus": "mini",
            "golden": FIXTURES / "golden_mini_fastapi.json",
        },
    )
    files = run_epochs.run_config_epochs(
        "custom-mock", k=2, dest_root=tmp_path, mock_config=FIXTURES / "mock_config.yaml"
    )
    assert len(files) == 2
    df = adapter.convert_envelopes(
        list(files),
        golden_path=FIXTURES / "golden_mini_fastapi.json",
        out_dir=tmp_path / "long",
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


# Guardrail 2: no silent paid calls. These refuse BEFORE any subprocess runs,
# so they spend nothing and need no API key.


def test_paid_guard_refuses_langchain_without_allow_paid(tmp_path):
    with pytest.raises(SystemExit, match="paid"):
        run_epochs.run_config_epochs("langchain-openai", k=1, dest_root=tmp_path)


def test_paid_guard_refuses_custom_without_mock_config(tmp_path):
    with pytest.raises(SystemExit, match="paid"):
        run_epochs.run_config_epochs("custom-openai", k=1, dest_root=tmp_path)
