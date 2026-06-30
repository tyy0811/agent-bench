"""The committed reveal_anchor.json must equal a fresh build from its three
source files. Reference = the source files (re-read every run); the JSON is the
derived artifact under test. Fails on drift either way (hand-edited JSON, or a
source change not yet regenerated). No hand-entered expected values: this
asserts derived == source, the same discipline as scripts/check_readme_stats.py.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _gen():
    # Dynamic import mirrors tests/scripts/test_run_calibration_dispatch.py and
    # keeps the sibling script out of static mypy scope.
    return importlib.import_module("scripts.build_reveal_anchor")


def test_committed_anchor_matches_sources():
    gen = _gen()
    committed = json.loads(Path(gen.ANCHOR_OUT).read_text())
    assert committed == gen.build_anchor()


def test_provenance_tiers_are_distinct():
    anchor = _gen().build_anchor()
    assert anchor["collapse"]["provenance"] == "campaign-bootstrap-ci"
    assert anchor["cost"]["provenance"] == "single-run"
    assert anchor["floor"]["provenance"] == "single-run-citation"
