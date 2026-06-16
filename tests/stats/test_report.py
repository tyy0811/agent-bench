"""Report generator tests: degradation branches, byte-stability, fixture drift."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

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


@pytest.mark.parametrize("name", ["base", "nonzero_failure", "failed_equivalence", "divergent_se"])
def test_golden_reports_byte_stable(name):
    expected = (FIXTURES / f"golden_report_{name}.md").read_text()
    assert _render(f"long_{name}.csv") == expected


def _split_by_config(df: pd.DataFrame, dest: Path) -> None:
    # Mirror the real pipeline shape: the adapter writes one CSV per run_id, and
    # each config is its own run_id, so a corpus directory holds one CSV per
    # config -- never a single combined table like the goldens use.
    dest.mkdir(parents=True, exist_ok=True)
    for i, (_, sub) in enumerate(df.groupby("config_id")):
        sub.to_csv(dest / f"run{i}.csv", index=False)


def test_load_tables_concatenates_per_config_csvs_so_cross_framework_sections_appear(tmp_path):
    # The committed goldens render from one combined CSV carrying both configs,
    # a shape the adapter cannot emit. This exercises the shape the WP5 pipeline
    # actually produces (results/long/<corpus>/<run_id>.csv, one per config) and
    # asserts the headline cross-framework sections are not silently empty.
    base = pd.read_csv(FIXTURES / "long_base.csv", dtype={"refused": "boolean"})
    _split_by_config(base, tmp_path / "fastapi")

    tables = report.load_tables(tmp_path)

    assert set(tables) == {"fastapi"}
    assert sorted(tables["fastapi"]["config_id"].unique()) == sorted(base["config_id"].unique())
    text = report.render_report(tables, seed=20260611)
    assert "custom-mock+00000000 vs langchain-mock+00000000" in text  # equivalence populated
    assert "Minimum detectable p_at_5 difference" in text  # MDE populated


def test_load_tables_keeps_legacy_in_its_own_corpus_bucket(tmp_path):
    # WP1 rule: legacy rows never silently mix with fresh data. Under the
    # parent-directory convention, results/long/legacy/ is just another corpus
    # bucket, kept separate by construction.
    base = pd.read_csv(FIXTURES / "long_base.csv", dtype={"refused": "boolean"})
    _split_by_config(base, tmp_path / "fastapi")
    (tmp_path / "legacy").mkdir()
    base.to_csv(tmp_path / "legacy" / "fastapi_postedit.csv", index=False)

    tables = report.load_tables(tmp_path)

    assert set(tables) == {"fastapi", "legacy"}
