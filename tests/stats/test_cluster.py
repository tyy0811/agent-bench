"""Cluster bootstrap SE tests.

Reference: cluster-robust SE from statsmodels 0.14.6 intercept-only OLS
(cov_type=cluster), data generated with numpy seed 7 as recorded in the
implementation plan Task 3.2. Bootstrap and analytic CRSE are different
estimators; tolerance is 25 percent relative, which a correct implementation
meets easily on this fixture while transposed-cluster bugs fail it.
"""

import numpy as np
import pytest

from stats import cluster

Y = np.array(
    [
        0.7166926368231703,
        0.7464441752382691,
        0.6891558359512003,
        0.6275104376116947,
        0.6711025429702498,
        0.610402610638613,
        0.7155816263980032,
        0.8435887906937126,
        0.6603466142831262,
        0.6475197761562652,
        0.5652258310558658,
        0.551930326853352,
        0.5267830509371357,
        0.42319482156652544,
        0.5133164437910186,
        0.7809513540023799,
        0.576999579828043,
        0.6656594584525294,
        0.5212987605764667,
        0.5824672605780535,
        0.7196500094820574,
        0.8803144001537624,
        0.7770788651168602,
        0.9309499491434007,
        0.9194986219236531,
        0.44923520381773224,
        0.21625232719867632,
        0.414059008696064,
        0.4630782037406204,
        0.4792591968810584,
        0.6758938266527004,
        0.7811320755998465,
        0.7310554953975756,
        0.7480236792606798,
        0.9349972655418475,
        0.6371496363212975,
        0.7146509333599351,
        0.8063420905928046,
        0.659543060580157,
        0.7067329088960712,
    ]
)
CLUSTERS = np.repeat(np.arange(8), 5)
REF_MEAN = 0.657026717319062
REF_CRSE = 0.04978800048752877


def test_mean_matches_reference():
    res = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    assert res.mean == pytest.approx(REF_MEAN, abs=1e-9)


def test_clustered_se_near_analytic_crse():
    res = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    assert res.clustered_se == pytest.approx(REF_CRSE, rel=0.25)
    assert res.n_clusters == 8


def test_seed_reproducibility():
    a = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    b = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    c = cluster.cluster_bootstrap(Y, CLUSTERS, seed=1)
    assert a.clustered_se == b.clustered_se
    assert a.clustered_se != c.clustered_se


def test_design_effect_above_one_with_cluster_correlation():
    res = cluster.cluster_bootstrap(Y, CLUSTERS, seed=20260611)
    assert res.design_effect > 1.0


def test_single_cluster_rejected():
    # One cluster is a point mass under resampling: it would report se=0 (false
    # certainty), so the preflight rejects it instead (review hardening).
    with pytest.raises(ValueError, match="clusters"):
        cluster.cluster_bootstrap(np.array([0.5, 0.6, 0.7]), np.array([1, 1, 1]), seed=20260611)


def test_non_finite_values_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        cluster.cluster_bootstrap(
            np.array([0.5, np.nan, 0.7, 0.8]), np.array([1, 1, 2, 2]), seed=20260611
        )


def test_primary_rule_threshold():
    assert cluster.primary_is_clustered(13)
    assert cluster.primary_is_clustered(10)
    assert not cluster.primary_is_clustered(6)
