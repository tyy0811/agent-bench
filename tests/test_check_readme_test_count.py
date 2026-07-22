"""The README test-count checker flags drifted claims and ignores the
V1/V2/V3 milestone table, so a hand-edited count cannot slip past CI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import check_readme_test_count as checker  # noqa: E402


def test_matching_claims_pass():
    text = "`731 tests` chip\nmake test  # 731 deterministic tests, no keys\n"
    assert checker.check(text, 731) == []


def test_drifted_claim_fails_with_both_numbers():
    failures = checker.check("`999 tests`\n", 731)
    assert len(failures) == 1
    assert "999" in failures[0] and "731" in failures[0]


def test_readme_without_claims_fails():
    assert checker.check("no counts stated here", 5) != []


def test_milestone_table_is_not_a_count_claim():
    text = "| Tests | 97 | 205 | 528 |\n\n`10 tests`\n"
    assert checker.check(text, 10) == []
