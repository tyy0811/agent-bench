"""Report generator tests: degradation branches, byte-stability, fixture drift."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats import report
from stats.paired import paired_bootstrap

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


def test_load_tables_rejects_duplicate_config_across_runs(tmp_path):
    # A config under two run_ids in one corpus (orphan re-run dir) would
    # double-count its epochs; load_tables must refuse, not silently merge.
    base = pd.read_csv(FIXTURES / "long_base.csv", dtype={"refused": "boolean"})
    one = base[base["config_id"] == "custom-mock+00000000"].copy()
    corpus = tmp_path / "fastapi"
    corpus.mkdir()
    one.to_csv(corpus / "run_a.csv", index=False)
    two = one.copy()
    two["run_id"] = "01HZXJ5M8N9PQRSTVWXYZ09999"  # a different run of the same config
    two.to_csv(corpus / "run_b.csv", index=False)
    with pytest.raises(ValueError, match="multiple run_ids"):
        report.load_tables(tmp_path)


def _block_values(text: str) -> dict[str, str]:
    """Parse the README-values section into {KEY: value}."""
    block = text.split("## README values", 1)[1]
    return dict(re.findall(r"^- ([a-z0-9_]+) = (.+)$", block, flags=re.MULTILINE))


def _headline_cells(text: str) -> dict[tuple[str, str], tuple[str, str]]:
    """Parse headline table rows into {(config, metric): (mean, ci)}."""
    rows = re.findall(r"^\| (\S+) \| (p_at_5|r_at_5) \| ([\d.]+) \| (\[[^\]]+\]) \(", text, re.M)
    return {(c, m): (mean, ci) for c, m, mean, ci in rows}


def test_readme_values_block_mirrors_headline_table():
    # Consistency invariant: every headline mean/CI in the tables has a
    # byte-identical README-values anchor. README markers pin to the block, so
    # this guarantees they cannot silently disagree with the headline section.
    text = _render("long_base.csv")
    values = _block_values(text)
    cells = _headline_cells(text)
    assert cells, "no headline rows parsed"
    for (config, metric), (mean, ci) in cells.items():
        stem = f"mini_{report._key(config)}_{metric}"
        assert values[f"{stem}_mean"] == mean
        assert values[f"{stem}_ci"] == ci


def test_readme_values_block_mirrors_tost_mde_icc_and_citation():
    # The non-table anchors must also mirror their sections verbatim.
    text = _render("long_base.csv")
    values = _block_values(text)
    support = re.search(r"vs langchain-mock\+\S+, p_at_5:.*plus or minus ([\d.]+)\.", text)
    assert values["mini_custom_mock_vs_langchain_mock_p_at_5_support"] == support.group(1)
    mde = re.search(r"Minimum detectable p_at_5 difference at 80 percent power: ([\d.]+)", text)
    assert values["mini_mde_p_at_5_80"] == mde.group(1)
    icc = re.search(r"ICC ([\d.]+) ", text)
    assert values["mini_icc_p_at_5"] == icc.group(1)
    cit = re.search(r"custom-mock\+\S+: 0 failures in (\d+) included", text)
    assert values["mini_custom_mock_citation_n"] == cit.group(1)


def test_significant_pairs_flags_large_consistent_difference_not_noise():
    # failed_equivalence bumps custom by +0.2 on both metrics; the 95% paired CI
    # must exclude 0 for both -> both flagged. For base, the helper's verdict
    # must equal a direct paired_bootstrap recompute (pins the pairing and
    # clustering glue rather than a hand-guessed answer to a bootstrap).
    failed = pd.read_csv(FIXTURES / "long_failed_equivalence.csv", dtype={"refused": "boolean"})
    sig = report._significant_pairs(failed, seed=20260611)
    assert "custom_mock vs langchain_mock p_at_5" in sig
    assert "custom_mock vs langchain_mock r_at_5" in sig

    base = pd.read_csv(FIXTURES / "long_base.csv", dtype={"refused": "boolean"})
    expected = set()
    for metric in report.HEADLINE_METRICS:
        a = report._question_means(base[base["config_id"] == "custom-mock+00000000"], metric)
        b = report._question_means(base[base["config_id"] == "langchain-mock+00000000"], metric)
        a, b = a.set_index("question_id"), b.set_index("question_id")
        shared = a.index.intersection(b.index)
        diffs = (a.loc[shared, "score"] - b.loc[shared, "score"]).to_numpy()
        clusters = a.loc[shared, "cluster_id"].to_numpy()
        use = report.primary_is_clustered(len(np.unique(clusters)))
        res = paired_bootstrap(
            diffs, clusters=clusters if use else None, confidence=0.95, seed=20260611
        )
        if res.ci_low > 0.0 or res.ci_high < 0.0:
            expected.add(f"custom_mock vs langchain_mock {metric}")
    assert set(report._significant_pairs(base, seed=20260611)) == expected


def _load_checker():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "check_readme_stats.py"
    spec = importlib.util.spec_from_file_location("check_readme_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_checker_passes_when_marker_matches_report():
    checker = _load_checker()
    readme = "P@5 is <!-- stats:fastapi_custom_openai_p_at_5_mean -->0.718<!-- /stats -->."
    report = "## README values\n- fastapi_custom_openai_p_at_5_mean = 0.718\n"
    assert checker.check(readme, report) == []


def test_readme_checker_fails_on_drift_and_on_absent_markers():
    checker = _load_checker()
    no_markers = checker.check("plain prose, no markers", "- k = 0.718")
    assert len(no_markers) == 1 and "no stats markers" in no_markers[0]
    drift = checker.check("<!-- stats:k -->0.999<!-- /stats -->", "- k = 0.718")
    assert len(drift) == 1 and "report says k = 0.718" in drift[0]


def test_readme_checker_catches_prefix_drift():
    # A substring match would pass 0.71 against report 0.718; exact match must fail.
    checker = _load_checker()
    drift = checker.check("<!-- stats:k -->0.71<!-- /stats -->", "- k = 0.718")
    assert len(drift) == 1 and "report says k = 0.718" in drift[0]


def test_readme_checker_handles_value_containing_lt():
    # Non-greedy marker group must capture a value that itself contains "<".
    checker = _load_checker()
    assert checker.check("<!-- stats:k --><0.05<!-- /stats -->", "- k = <0.05") == []


def test_pass_k_section_renders_per_config_when_refusal_correct_present():
    df = pd.DataFrame(
        {
            "config_id": ["c"] * 6,
            "question_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "epoch": [1, 2, 1, 2, 1, 2],
            "metric": ["refusal_correct"] * 6,
            "score": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )
    text = "\n".join(report._pass_k_section(df, "mini"))
    assert "Refusal reliability (pass^k): mini" in text
    assert "| c | 2 | 0.833 | 0.667 |" in text  # single-run 5/6, pass^2 2/3


def test_pass_k_section_absent_without_refusal_correct():
    df = pd.DataFrame(
        {
            "config_id": ["c"],
            "question_id": ["q1"],
            "epoch": [1],
            "metric": ["p_at_5"],
            "score": [0.7],
        }
    )
    assert report._pass_k_section(df, "mini") == []


def test_pass_k_section_tolerates_unbalanced_epochs():
    # One config with a ragged refusal epoch must be skipped-and-noted, not crash
    # the whole render (graceful degradation, matching the citation section).
    df = pd.DataFrame(
        {
            "config_id": ["c", "c", "c"],
            "question_id": ["q1", "q1", "q2"],
            "epoch": [1, 2, 1],
            "metric": ["refusal_correct"] * 3,
            "score": [1.0, 1.0, 1.0],
        }
    )
    text = "\n".join(report._pass_k_section(df, "mini"))
    assert "unbalanced epochs" in text
