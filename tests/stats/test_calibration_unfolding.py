"""The unfolding demonstration adapter reproduces the judge-design section 1.9
numbers from the committed calibration + canary files. The anchors and point
estimates are independently checkable (raw/true pass-rates verified against the
raw JSON; the matrix-inversion point is hand-worked); the bootstrap intervals are
pinned for reproducibility (seed 20260611, 10k, matching stats.agreement).
"""

from pathlib import Path

import pytest

from stats_adapters.calibration_unfolding import build_demo

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "measurements" / "2026-05-04-judge-calibration-labels.jsonl"
JURY = ROOT / "results" / "calibration_v1_judge_jury_kappa_weighted_v1_1.json"
CANARIES = ROOT / "tests" / "stats" / "fixtures" / "canary" / "canaries.json"
CANARY_PREDS = ROOT / "results" / "canary" / "predictions_real_v1.json"


def _demo(n_boot):
    return build_demo(
        labels_path=LABELS,
        judge_predictions_path=JURY,
        canaries_path=CANARIES,
        canary_predictions_path=CANARY_PREDS,
        n_boot=n_boot,
    )


def test_demo_anchors_and_point_estimates_match_raw_data():
    # Anchors independently verified against the raw JSON (canary completeness:
    # 14/20 jury pass, 14/20 true pass, identity confusion). The matrix-inversion
    # point is hand-worked: R^{-1} @ [6, 14] = [6.33, 13.67], pass = 13.67/20.
    # The off-diagonals are the transfer caveat made numeric: R is estimated
    # where the judge erred (calibration 4/26 = 0.154) and applied where it did
    # not (canary 0.000). All four are bootstrap-independent, so n_boot is tiny.
    d = _demo(n_boot=50)
    assert d.dimension == "completeness"
    assert (d.n_calibration, d.n_canary, d.n_abstain) == (26, 20, 0)
    assert d.raw_pass_rate == pytest.approx(0.70)
    assert d.true_pass_rate == pytest.approx(0.70)
    assert d.calibration_offdiag == pytest.approx(4 / 26, abs=1e-4)
    assert d.canary_offdiag == pytest.approx(0.0)
    assert d.corrected_invert == pytest.approx(0.6838, abs=1e-3)
    assert d.corrected_dagostini == pytest.approx(0.6406, abs=1e-3)
    assert d.divergence == pytest.approx(0.0432, abs=1e-3)


def test_demo_dagostini_interval_contains_truth_and_is_pinned():
    # Headline validation: the wide corrected interval stays consistent with the
    # known truth 0.70 (propagation calibrated against a target the judge already
    # scored correctly). Intervals pinned at the layer's 10k bootstrap. The
    # matrix-inversion interval leaves the simplex entirely, the precise signal
    # that the correction is unidentified at this sample size.
    d = _demo(n_boot=10_000)
    assert d.ci_dagostini[0] <= d.true_pass_rate <= d.ci_dagostini[1]
    assert d.ci_dagostini == pytest.approx((0.2871, 0.9587), abs=1e-3)
    assert d.ci_dagostini_r_only == pytest.approx((0.3887, 0.8795), abs=1e-3)
    assert d.ci_dagostini_sampling_only == pytest.approx((0.3601, 0.8988), abs=1e-3)
    assert d.ci_invert[0] < 0.0 < 1.0 < d.ci_invert[1]
    assert d.dominant_source == "comparable"


def test_demo_is_reproducible_under_fixed_seed():
    assert _demo(n_boot=200) == _demo(n_boot=200)
