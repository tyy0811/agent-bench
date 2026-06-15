"""Paired bootstrap and exact McNemar.

McNemar reference: statsmodels 0.14.6 mcnemar(exact=True) on b=2, c=9 (command
in implementation plan Task 3.3). Bootstrap CI checked for seed-stability,
coverage direction, and the all-ties edge case.
"""

import numpy as np
import pytest

from stats import paired

DIFFS = np.array(
    [
        0.022735421380254733,
        0.12877980322479693,
        0.11797768628687459,
        -0.0208245661430134,
        -0.003837560888515768,
        -0.022190735442674015,
        0.06557810860575682,
        0.015514844876350593,
        0.07975084930052352,
        -0.12778598391792878,
        0.14532390197596165,
        0.012285427187550355,
        0.07443027626193169,
        0.009074693281853782,
        -0.010327885365988262,
        0.05704881268780694,
        0.08596108220240904,
        0.003797610344523878,
        0.0077771057143842335,
        0.07485588886474064,
        -0.0496272513557737,
        -0.10115068029851164,
    ]
)


def test_mcnemar_exact_matches_statsmodels():
    assert paired.mcnemar_exact(b=2, c=9) == pytest.approx(0.0654296875, abs=1e-12)


def test_mcnemar_no_discordant_pairs_p_one():
    assert paired.mcnemar_exact(b=0, c=0) == 1.0


def test_paired_bootstrap_seeded_and_centered():
    res = paired.paired_bootstrap(DIFFS, confidence=0.90, seed=20260611)
    again = paired.paired_bootstrap(DIFFS, confidence=0.90, seed=20260611)
    assert (res.ci_low, res.ci_high) == (again.ci_low, again.ci_high)
    assert res.ci_low < res.mean_diff < res.ci_high
    assert res.mean_diff == pytest.approx(float(DIFFS.mean()), abs=1e-12)


def test_all_ties_collapse_to_point():
    res = paired.paired_bootstrap(np.zeros(22), confidence=0.90, seed=20260611)
    assert res.ci_low == res.ci_high == res.mean_diff == 0.0


def test_cluster_mode_resamples_clusters():
    clusters = np.repeat(np.arange(11), 2)
    res = paired.paired_bootstrap(DIFFS, clusters=clusters, confidence=0.90, seed=20260611)
    free = paired.paired_bootstrap(DIFFS, confidence=0.90, seed=20260611)
    assert res.n_units == 11
    assert free.n_units == 22
    assert (res.ci_low, res.ci_high) != (free.ci_low, free.ci_high)


def test_single_unit_rejected():
    with pytest.raises(ValueError, match="units"):
        paired.paired_bootstrap(np.array([0.02]), seed=20260611)


def test_non_finite_diffs_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        paired.paired_bootstrap(np.array([0.02, np.nan, 0.01]), seed=20260611)


def test_mcnemar_negative_count_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        paired.mcnemar_exact(b=-1, c=2)
