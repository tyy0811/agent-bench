"""Report generator tests: degradation branches, byte-stability, fixture drift."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from stats import report

FIXTURES = Path(__file__).parent / "fixtures"


def _render(name: str) -> str:
    df = pd.read_csv(FIXTURES / name, dtype={"refused": "boolean"})
    return report.render_report({"mini": df}, seed=20260611)


def test_fixture_tables_have_not_drifted(tmp_path):
    src = FIXTURES / "make_fixture_tables.py"
    subprocess.run([sys.executable, str(src)], check=True, cwd=FIXTURES)
    out = subprocess.run(["git", "diff", "--stat", str(FIXTURES)], capture_output=True, text=True)
    assert "csv" not in out.stdout, "regenerating fixtures changed committed CSVs"


def test_zero_failure_branch_uses_rule_of_three_phrasing():
    text = _render("long_base.csv")
    assert "rule of three" in text.lower()
    assert "Clopper-Pearson" in text


def test_nonzero_failure_branch_drops_rule_of_three_phrasing():
    text = _render("long_nonzero_failure.csv")
    assert "rule of three" not in text.lower()
    assert "Clopper-Pearson" in text


def test_equivalence_pass_and_fail_wordings():
    passing = _render("long_base.csv")
    failing = _render("long_failed_equivalence.csv")
    assert "equivalent within plus or minus 0.10" in passing
    assert "equivalence not established at plus or minus 0.10" in failing
    assert "the data support" in passing and "the data support" in failing


def test_divergence_caution_fires_only_on_divergent_fixture():
    calm = _render("long_base.csv")
    loud = _render("long_divergent_se.csv")
    assert "correlation-sensitivity caution" not in calm
    assert "correlation-sensitivity caution" in loud


def test_always_prints_n_clusters_and_design_effect():
    text = _render("long_base.csv")
    assert "n_clusters" in text
    assert "design effect" in text


def test_byte_stability_double_render():
    # No wall-clock anywhere in the renderer: two renders of the same input
    # are byte-identical, and the golden tests in Task 4.3 pin the bytes
    # across processes and machines. Dates in the output may come only from
    # the input table or from pre-registration literals in fixed prose.
    assert _render("long_base.csv") == _render("long_base.csv")
