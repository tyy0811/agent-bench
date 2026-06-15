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
