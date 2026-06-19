"""Tests for the pure judge-unfolding engine (stats/unfolding.py).

Engine correctness is pinned on synthetic, well-conditioned cases with known
answers (algebraic invariants + a hand-worked D'Agostini iteration + a
numpy.linalg cross-check), SEPARATELY from whether the correction is
informative on the real degenerate corpus. Reference values carry provenance
comments per the project numerical-reference rule.
"""

import numpy as np
import pytest

from stats.unfolding import (
    DEFAULT_N_ITER,
    _dominant_source,
    dagostini_unfold,
    invert_unfold,
    response_matrix,
    unfold_with_uncertainty,
)


def test_response_matrix_is_column_conditional_and_stochastic():
    # true class 0 has 3 samples observed as [0, 0, 1]; true class 1 has one
    # sample observed as [1]. R[j, i] = P(obs=j | true=i):
    #   column i=0 -> [2/3, 1/3];  column i=1 -> [0, 1]   (each column sums to 1)
    true = np.array([0, 0, 0, 1])
    observed = np.array([0, 0, 1, 1])
    resp = response_matrix(true, observed, [0, 1])
    expected = np.array([[2 / 3, 0.0], [1 / 3, 1.0]])
    np.testing.assert_allclose(resp, expected)
    np.testing.assert_allclose(resp.sum(axis=0), [1.0, 1.0])


def test_invert_unfold_recovers_truth_on_noiseless_fold():
    # Folding a known truth through a well-conditioned R gives a noiseless
    # observed vector; the matrix-inversion baseline must recover the truth
    # exactly (R @ true = observed  =>  invert_unfold(R, observed) = true).
    resp = np.array([[0.8, 0.1], [0.2, 0.9]])
    true_counts = np.array([30.0, 70.0])
    observed = resp @ true_counts
    np.testing.assert_allclose(invert_unfold(resp, observed), true_counts)


def test_invert_unfold_identity_returns_observed_unchanged():
    # With R = I the judge is perfect, so the corrected counts equal the observed
    # counts (no smearing to undo).
    n_obs = np.array([5.0, 0.0, 11.0])
    np.testing.assert_allclose(invert_unfold(np.eye(3), n_obs), n_obs)


def test_invert_unfold_matches_explicit_matrix_inverse():
    # Independent path cross-check: solving R @ t = n_obs must equal inv(R) @
    # n_obs on a well-conditioned 3-level matrix (columns sum to 1).
    resp = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    n_obs = np.array([40.0, 35.0, 25.0])
    np.testing.assert_allclose(invert_unfold(resp, n_obs), np.linalg.inv(resp) @ n_obs)


def test_dagostini_single_iteration_matches_hand_worked_bayes():
    # One D'Agostini iteration, efficiency=1, uniform prior, worked by hand.
    #   R[j,i] = P(obs=j | true=i) = [[0.8, 0.3], [0.2, 0.7]]
    #   n_obs = [60, 40];  prior p = [0.5, 0.5]
    #   denominators d[j] = sum_k R[j,k] p[k] = [0.8*.5+0.3*.5, 0.2*.5+0.7*.5]
    #                     = [0.55, 0.45]
    #   unfolding M[i,j] = R[j,i] p[i] / d[j]:
    #     M[0,0]=0.8*.5/0.55=8/11   M[0,1]=0.2*.5/0.45=2/9
    #     M[1,0]=0.3*.5/0.55=3/11   M[1,1]=0.7*.5/0.45=7/9
    #   n_true[i] = sum_j M[i,j] n_obs[j]:
    #     n_true[0] = (8/11)*60 + (2/9)*40 = 480/11 + 80/9 = 5200/99 = 52.5253
    #     n_true[1] = (3/11)*60 + (7/9)*40 = 180/11 + 280/9 = 4700/99 = 47.4747
    #   (totals conserved: 5200/99 + 4700/99 = 100)
    resp = np.array([[0.8, 0.3], [0.2, 0.7]])
    n_obs = np.array([60.0, 40.0])
    result = dagostini_unfold(resp, n_obs, n_iter=1, prior=np.array([0.5, 0.5]))
    np.testing.assert_allclose(result, [5200 / 99, 4700 / 99])


def test_dagostini_converges_to_matrix_inverse_when_iterated():
    # On a well-conditioned invertible R with an interior (non-negative) inverse,
    # D'Agostini iterated to convergence reproduces the matrix-inversion estimate.
    # Agreement here calibrates the engine; the DIVERGENCE between the two methods
    # at the small default n_iter on an ill-conditioned matrix is the diagnostic
    # the engine exists to surface (measured: dist-to-inverse 10.3 -> ~0 over
    # n_iter 1 -> 30 at cond 1.43).
    resp = np.array([[0.85, 0.15], [0.15, 0.85]])
    n_obs = np.array([55.0, 45.0])
    np.testing.assert_allclose(
        dagostini_unfold(resp, n_obs, n_iter=100), invert_unfold(resp, n_obs), atol=1e-3
    )


def test_default_n_iter_is_small_for_early_stopping_regularization():
    # Early stopping IS the regularization, so the default must stay small enough
    # to hold an ill-conditioned matrix near the prior instead of running it to
    # the amplified inverse. Set result-blind from the synthetic reference's
    # convergence knee; a large value here would silently defeat the regularizer.
    assert 1 <= DEFAULT_N_ITER <= 6


def test_unfold_with_uncertainty_identity_calibration_returns_observed():
    # A perfect judge: calibration pairs agree exactly -> R = I -> the corrected
    # distribution equals the observed one, both estimators agree (divergence 0),
    # and every bootstrap pass yields a per-level interval.
    true = np.array([0, 0, 0, 1, 1])
    observed = np.array([0, 0, 0, 1, 1])  # identical -> R = identity
    n_obs = np.array([30.0, 70.0])
    res = unfold_with_uncertainty(true, observed, n_obs, n_boot=200, seed=20260611)
    np.testing.assert_allclose(res.point, [0.3, 0.7], atol=1e-9)
    np.testing.assert_allclose(res.invert, [0.3, 0.7], atol=1e-9)
    np.testing.assert_allclose(res.dagostini, [0.3, 0.7], atol=1e-9)
    assert res.divergence < 1e-9
    assert res.method == "dagostini"
    assert len(res.ci_r_only) == len(res.ci_sampling_only) == len(res.ci_combined) == 2


def test_unfold_with_uncertainty_is_reproducible_under_fixed_seed():
    true = np.array([0, 0, 1, 1, 1, 2, 2])
    observed = np.array([0, 1, 1, 2, 1, 2, 2])
    n_obs = np.array([10.0, 20.0, 30.0])
    a = unfold_with_uncertainty(true, observed, n_obs, n_boot=300, seed=20260611)
    b = unfold_with_uncertainty(true, observed, n_obs, n_boot=300, seed=20260611)
    assert a.point == b.point
    assert a.ci_r_only == b.ci_r_only
    assert a.ci_combined == b.ci_combined


def test_uncertainty_decomposition_attributes_width_to_r_when_calibration_is_tiny():
    # A tiny, noisy calibration set leaves R poorly known while a large observed
    # sample makes the sampling term small, so the interval width is carried by
    # the R term. This is the precise form of "uninformative because the judge
    # model is undernailed, not because the corpus is small".
    true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    observed = np.array([0, 0, 0, 1, 1, 1, 0, 1])  # R ~ [[.75,.25],[.25,.75]]
    n_obs = np.array([500.0, 500.0])

    def width(cis):
        return float(np.mean([hi - lo for lo, hi in cis]))

    res = unfold_with_uncertainty(true, observed, n_obs, n_boot=1000, seed=20260611)
    assert width(res.ci_r_only) > width(res.ci_sampling_only)
    assert res.dominant_source == "R"


def test_both_estimators_reported_and_diverge_on_ill_conditioned_correction():
    # An ill-conditioned R with the observed far from its fixed point forces a
    # large correction: the matrix inverse amplifies (here past the simplex,
    # invert prop 1.4 / -0.4) while regularized D'Agostini stays near the
    # observed. Their divergence is the headline diagnostic.
    true = np.array([0, 0, 0, 1, 1, 1])
    observed = np.array([0, 0, 1, 1, 0, 1])  # R = [[2/3,1/3],[1/3,2/3]], cond 3
    n_obs = np.array([80.0, 20.0])
    res = unfold_with_uncertainty(true, observed, n_obs, n_boot=100, seed=20260611)
    assert res.divergence > 0.02
    assert len(res.invert) == 2 and len(res.dagostini) == 2


def test_unfold_tolerates_a_true_class_absent_from_calibration():
    # When the calibration never contains true level 2, that column is unobserved;
    # the engine fills it with the uniform-over-observed convention instead of a
    # singular matrix, so the unfold still returns a finite 3-level result.
    true = np.array([0, 0, 1, 1])  # no true == 2
    observed = np.array([0, 1, 1, 2])
    n_obs = np.array([10.0, 20.0, 5.0])
    res = unfold_with_uncertainty(true, observed, n_obs, n_boot=100, seed=1)
    assert len(res.point) == 3
    assert np.isfinite(res.divergence)


def test_dagostini_matches_pyunfold_at_convergence():
    # Confirmatory cross-check against the published pyunfold implementation
    # (verified locally against pyunfold 0.5.0: max abs diff 7.9e-11). CI never
    # installs pyunfold, so this skips there and nothing the engine needs depends
    # on it; it lives on the manual pre-merge checklist. The comparison is at
    # convergence (high max_iter), where both reach the maximum-likelihood
    # solution regardless of prior (pyunfold defaults to Jeffreys, ours to
    # uniform), so it is robust to iteration-indexing and prior conventions
    # rather than config-fragile.
    pyunfold = pytest.importorskip("pyunfold")
    resp = np.array([[0.85, 0.15], [0.15, 0.85]])
    n_obs = np.array([55.0, 45.0])
    result = pyunfold.iterative_unfold(
        data=n_obs,
        data_err=np.sqrt(n_obs),
        response=resp,
        response_err=np.zeros_like(resp),
        efficiencies=np.ones(2),
        efficiencies_err=np.zeros(2),
        ts="ks",
        ts_stopping=1e-12,
        max_iter=500,
    )
    np.testing.assert_allclose(
        dagostini_unfold(resp, n_obs, n_iter=200), result["unfolded"], atol=1e-6
    )


def test_unfold_reports_nan_invert_when_point_matrix_is_singular():
    # Both true classes have the same response (each observed half at 0, half at
    # 1), so R = [[0.5, 0.5], [0.5, 0.5]] is rank-1 singular but has no zero row.
    # The matrix inverse is unidentified and reported as NaN (and so the
    # divergence), not a silent fall-back to the observed distribution; the
    # regularized D'Agostini estimate stays finite.
    true = np.array([0, 0, 1, 1])
    observed = np.array([0, 1, 0, 1])
    n_obs = np.array([10.0, 10.0])
    res = unfold_with_uncertainty(true, observed, n_obs, n_boot=50, seed=1)
    assert np.isnan(res.invert).all()
    assert np.isnan(res.divergence)
    assert np.all(np.isfinite(res.dagostini))
    # With method="invert" the inversion estimate is unavailable end to end: both
    # the point AND the bootstrap interval are NaN, not a finite obs-based one.
    inv = unfold_with_uncertainty(true, observed, n_obs, method="invert", n_boot=50, seed=1)
    assert np.isnan(inv.point).all()
    assert all(np.isnan(lo) and np.isnan(hi) for lo, hi in inv.ci_combined)


def test_unfold_reports_nan_when_an_observed_level_is_unmodeled():
    # The calibration never produces observed level 1 (a zero response-matrix
    # row), but the target has mass there; that mass cannot be unfolded by either
    # estimator, so both are NaN rather than silently dropped and renormalized.
    true = np.array([0, 1])
    observed = np.array([0, 0])  # neither true class is ever observed as level 1
    n_obs = np.array([5.0, 5.0])  # but the target has 5 at the unmodeled level 1
    res = unfold_with_uncertainty(true, observed, n_obs, n_boot=20, seed=1)
    assert np.isnan(res.dagostini).all()
    assert np.isnan(res.invert).all()
    assert np.isnan(res.divergence)


def test_unfold_rejects_empty_observed_vector():
    with pytest.raises(ValueError):
        unfold_with_uncertainty(np.array([0, 1]), np.array([0, 1]), np.array([0.0, 0.0]))


def test_dominant_source_is_comparable_when_both_widths_are_zero():
    # A fully-deterministic case has zero width from both sources, so neither
    # dominates and the honest label is "comparable", not a false "R".
    assert _dominant_source(0.0, 0.0) == "comparable"
    assert _dominant_source(0.5, 0.0) == "R"
    assert _dominant_source(0.0, 0.5) == "sampling"
