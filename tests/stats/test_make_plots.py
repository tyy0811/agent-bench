"""Plot provenance tests: parsing, source-hash, and the freshness check.

Keyless and matplotlib-free -- the renderer is lazy-imported inside make_plots,
so these import and run without the [plots] extra installed (mirrors how CI runs
the freshness check without matplotlib).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

import make_plots  # noqa: E402

# A minimal README-values block carrying exactly the keys the forest plot reads.
# Values are independent literals (not the live report) so the test pins the
# extraction logic, not today's campaign numbers.
SAMPLE = """## README values

- fastapi_custom_openai_p_at_5_mean = 0.718
- fastapi_custom_openai_p_at_5_ci = [0.610, 0.827]
- fastapi_custom_openai_r_at_5_mean = 0.833
- fastapi_custom_openai_r_at_5_ci = [0.715, 0.951]
- fastapi_custom_anthropic_p_at_5_mean = 0.791
- fastapi_custom_anthropic_p_at_5_ci = [0.664, 0.917]
- fastapi_custom_anthropic_r_at_5_mean = 0.841
- fastapi_custom_anthropic_r_at_5_ci = [0.710, 0.971]
- fastapi_langchain_openai_p_at_5_mean = 0.627
- fastapi_langchain_openai_p_at_5_ci = [0.484, 0.770]
- fastapi_langchain_openai_r_at_5_mean = 0.836
- fastapi_langchain_openai_r_at_5_ci = [0.702, 0.971]
- fastapi_langchain_anthropic_p_at_5_mean = 0.760
- fastapi_langchain_anthropic_p_at_5_ci = [0.645, 0.875]
- fastapi_langchain_anthropic_r_at_5_mean = 0.841
- fastapi_langchain_anthropic_r_at_5_ci = [0.710, 0.971]
- fastapi_significant_pairs_95 = custom_anthropic vs langchain_openai p_at_5
- fastapi_significant_pairs_95_count = 1
- fastapi_mde_p_at_5_80 = 0.110
- fastapi_custom_anthropic_vs_langchain_openai_p_at_5_diff = +0.164
- fastapi_custom_anthropic_vs_langchain_openai_p_at_5_tost = not established
- fastapi_custom_anthropic_vs_langchain_openai_p_at_5_ci90 = [+0.031, +0.317]
- fastapi_custom_anthropic_vs_langchain_openai_p_at_5_ci95 = [+0.016, +0.353]
- fastapi_custom_anthropic_vs_langchain_anthropic_r_at_5_diff = +0.000
- fastapi_custom_anthropic_vs_langchain_anthropic_r_at_5_tost = equivalent
- fastapi_custom_anthropic_vs_langchain_anthropic_r_at_5_ci90 = [+0.000, +0.000]
- fastapi_custom_anthropic_vs_langchain_anthropic_r_at_5_ci95 = [+0.000, +0.000]
- fastapi_between_question_var_p_at_5 = 0.06051
- fastapi_within_question_var_p_at_5 = 0.00036
- fastapi_icc_p_at_5 = 0.99
- k8s_between_question_var_p_at_5 = 0.03249
- k8s_within_question_var_p_at_5 = 0.00209
- k8s_icc_p_at_5 = 0.94
"""


def test_forest_rows_reads_eight_pinned_points():
    rows = make_plots.forest_rows(make_plots.read_values(SAMPLE), "fastapi")
    assert len(rows) == 8  # 4 configs x 2 metrics
    co_p = next(r for r in rows if r["label"] == "Custom OpenAI" and r["metric_label"] == "P@5")
    assert co_p["metric"] == "p_at_5"  # key kept for significant-pair matching
    assert co_p["mean"] == 0.718
    assert (co_p["lo"], co_p["hi"]) == (0.610, 0.827)
    assert co_p["framework"] == "custom"


def test_source_hash_is_stable_and_sensitive():
    h1 = make_plots.source_hash(make_plots.forest_source(make_plots.read_values(SAMPLE)))
    h2 = make_plots.source_hash(make_plots.forest_source(make_plots.read_values(SAMPLE)))
    assert h1 == h2  # deterministic across calls
    drifted = make_plots.read_values(SAMPLE.replace("0.718", "0.719"))
    assert make_plots.source_hash(make_plots.forest_source(drifted)) != h1  # one digit moves it


def test_paired_rows_reads_nested_cis_and_flags():
    rows = make_plots.paired_rows(make_plots.read_values(SAMPLE), "fastapi")
    sig = next(r for r in rows if r["significant"])
    assert sig["label"] == "Custom Anthropic - LC OpenAI" and sig["metric"] == "p_at_5"
    assert sig["diff"] == 0.164
    assert sig["ci90"] == (0.031, 0.317) and sig["ci95"] == (0.016, 0.353)
    assert sig["same_provider"] is False  # anthropic vs openai
    tie = next(r for r in rows if r["ci95"] == (0.0, 0.0))
    assert tie["same_provider"] is True and tie["tost"] == "equivalent"


def test_variance_rows_contrasts_within_fraction():
    rows = make_plots.variance_rows(make_plots.read_values(SAMPLE))
    fa = next(r for r in rows if r["label"] == "FastAPI")
    k8 = next(r for r in rows if r["label"] == "Kubernetes")
    assert fa["icc"] == 0.99 and k8["icc"] == 0.94
    assert fa["within_frac"] < k8["within_frac"]  # k8s hides more epoch noise
    assert abs(k8["within_frac"] - 0.00209 / (0.03249 + 0.00209)) < 1e-9


def test_mde_source_reads_floor_and_gaps():
    src = make_plots.mde_source(make_plots.read_values(SAMPLE), "fastapi")
    assert src["mde"] == 0.110
    sig = next(g for g in src["gaps"] if g["significant"])
    assert abs(sig["abs_diff"] - 0.164) < 1e-9


def test_unfolding_source_parses_judge_design_1_9():
    # Pins the plot to judge-design.md section 1.9 (not the auto report); if the
    # table is edited this fails, the intended drift signal for a hand-doc source.
    src = make_plots.unfolding_source()
    assert src["observed"] == 0.700
    assert src["dagostini"] == {"point": 0.641, "lo": 0.287, "hi": 0.959}
    mi = src["matrix_inversion"]
    assert mi["lo"] < 0.0 and mi["hi"] > 1.0  # naive estimator leaves the unit interval


def test_check_flags_missing_fresh_and_stale(tmp_path):
    vals = make_plots.read_values(SAMPLE)
    # the committed figures are PNGs (GitHub will not inline relative-path SVGs);
    # check reads them as bytes, so text stand-ins carrying the hash are enough.
    # missing: no files yet
    assert any("missing" in f for f in make_plots.check(SAMPLE, tmp_path))
    # fresh: every expected plot present with its correct 16-hex source-hash
    for name, builder in make_plots.EXPECTED_PLOTS.items():
        (tmp_path / name).write_text(f"PNG\x00source-hash:{make_plots.source_hash(builder(vals))}")
    assert make_plots.check(SAMPLE, tmp_path) == []
    # stale: one plot carries a valid-shape but wrong hash
    bad = next(iter(make_plots.EXPECTED_PLOTS))
    (tmp_path / bad).write_text("PNG\x00source-hash:deadbeefdeadbeef")
    assert any("stale" in f for f in make_plots.check(SAMPLE, tmp_path))
