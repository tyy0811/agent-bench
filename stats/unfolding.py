"""Judge unfolding: correct an observed (jury) rubric-level distribution for the
judge's measured confusion, with uncertainty propagated.

Pure module (numpy/scipy only; no project imports, no pandas). The estimand is the
TRUE rubric-level distribution given a noisy judge: the judge maps a true level
i to an observed level j with probability ``R[j, i]`` (the response/confusion
matrix), and unfolding inverts that map. Two estimators are exposed and BOTH are
meant to be reported, never silently chosen between:

* ``invert_unfold`` -- the maximum-likelihood / matrix-inversion baseline. Exact
  when R is invertible, but it amplifies noise as R becomes ill-conditioned.
* ``dagostini_unfold`` -- D'Agostini iterative Bayesian unfolding with a fixed,
  small iteration count. Early stopping IS the regularization: it stays near the
  prior where the data are uninformative instead of running to the noise-
  amplifying inverse. The iteration count is set result-blind from a well-
  conditioned synthetic reference, never tuned on real corrected output.

Reading the DIVERGENCE between the two is the diagnostic that a correction is
unidentified at the given sample size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Result-blind early-stopping iteration count for D'Agostini unfolding. Early
# stopping IS the regularization, so this stays small: on the well-conditioned
# synthetic reference (cond 1.43) the per-iteration step drops ~4x by iteration
# 4 (dist-to-inverse 10.3 -> 5.3 -> 2.8 -> 1.5 -> 0.8 over iters 1..5), so 4 sits
# at the knee of the curve -- enough to move off the prior, short of chasing the
# last few percent toward the noise-amplifying inverse. Confirmed benign against
# the real completeness conditioning (cond 1.97), where this same small count
# keeps the estimate near the prior rather than the amplified inverse. Fixed
# BEFORE looking at any real corrected output, the same result-blind discipline
# as the TOST margin and the canary top-anchor threshold.
DEFAULT_N_ITER = 4

# Bootstrap settings, matching the rest of the statistics layer (stats.agreement).
DEFAULT_SEED = 20260611
DEFAULT_N_BOOT = 10_000


def response_matrix(true: np.ndarray, observed: np.ndarray, levels: Sequence[int]) -> np.ndarray:
    """Column-stochastic response matrix ``R[j, i] = P(observed=levels[j] |
    true=levels[i])`` estimated from paired per-sample (true, observed) labels.

    ``true`` and ``observed`` are equal-length integer arrays of rubric levels;
    ``levels`` is the ordered level set. Each column is normalized by its true-
    class count, so a populated column is a proper conditional distribution
    summing to 1. A true class with zero samples leaves an all-zero column here;
    the degenerate-column convention (uniform over observed levels) is applied
    by the bootstrap in :func:`unfold_with_uncertainty`, not baked into the raw
    estimate.
    """
    true_arr = np.asarray(true)
    observed_arr = np.asarray(observed)
    index = {level: k for k, level in enumerate(levels)}
    size = len(levels)
    counts = np.zeros((size, size))
    for t, o in zip(true_arr, observed_arr):
        counts[index[int(o)], index[int(t)]] += 1.0
    col_totals = counts.sum(axis=0)
    resp = np.zeros((size, size))
    nonzero = col_totals > 0
    resp[:, nonzero] = counts[:, nonzero] / col_totals[nonzero]
    return resp


def invert_unfold(resp: np.ndarray, n_obs: np.ndarray) -> np.ndarray:
    """Matrix-inversion / maximum-likelihood baseline: solve ``R @ true = n_obs``
    for the true-level counts.

    Exact when R is invertible (efficiency 1, so the columns sum to 1 and counts
    are conserved). It amplifies noise as R becomes ill-conditioned -- that is
    the intended behavior, not a defect: the divergence between this estimate and
    the regularized D'Agostini estimate is the diagnostic that the correction is
    unidentified at the given sample size. An exactly singular R (a true class
    that maps nowhere distinguishable) raises ``numpy.linalg.LinAlgError``, which
    is itself the signal that the level is unobservable here.
    """
    solution: np.ndarray = np.linalg.solve(
        np.asarray(resp, dtype=float), np.asarray(n_obs, dtype=float)
    )
    return solution


def dagostini_unfold(
    resp: np.ndarray,
    n_obs: np.ndarray,
    n_iter: int,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """D'Agostini iterative Bayesian unfolding (efficiency = 1).

    Each iteration applies Bayes' theorem to turn the response matrix
    ``R[j, i] = P(obs=j | true=i)`` and the current prior over true levels into
    the unfolding matrix ``M[i, j] = P(true=i | obs=j)``, estimates the true
    counts ``n_true[i] = sum_j M[i, j] n_obs[j]``, and feeds the normalized
    estimate back as the next prior. ``n_iter`` IS the regularization: early
    stopping holds the estimate near the prior where the data are uninformative
    instead of running to the noise-amplifying matrix inverse, so it must be a
    small fixed value set result-blind from a well-conditioned synthetic
    reference, never tuned on real corrected output. ``prior`` defaults to
    uniform over the true levels.
    """
    resp_arr = np.asarray(resp, dtype=float)
    obs = np.asarray(n_obs, dtype=float)
    n_true_levels = resp_arr.shape[1]
    if prior is None:
        p = np.full(n_true_levels, 1.0 / n_true_levels)
    else:
        p = np.asarray(prior, dtype=float)
        p = p / p.sum()
    n_true = p * obs.sum()
    for _ in range(n_iter):
        denom = resp_arr @ p  # d[j] = sum_k R[j,k] p[k]
        safe = np.where(denom > 0, denom, 1.0)
        unfolding = (resp_arr.T * p[:, None]) / safe[None, :]  # M[i,j] = R[j,i] p[i] / d[j]
        unfolding = np.where(denom[None, :] > 0, unfolding, 0.0)
        n_true = unfolding @ obs  # n_true[i] = sum_j M[i,j] n_obs[j]
        total = n_true.sum()
        if total > 0:
            p = n_true / total
    return np.asarray(n_true, dtype=float)


@dataclass(frozen=True)
class UnfoldResult:
    """Corrected true-level distribution with its uncertainty decomposition.

    Every vector is a proportion over ``levels``, so the top-anchor passing rate
    is just the entry at the top level. ``invert`` and ``dagostini`` are BOTH
    reported and their ``divergence`` (max per-level gap) is the diagnostic that
    the correction is unidentified at this sample size. The three CI families
    decompose the uncertainty by source: ``ci_r_only`` resamples the calibration
    pairs (R varies, observed fixed), ``ci_sampling_only`` resamples the observed
    counts (observed varies, R fixed), ``ci_combined`` resamples both;
    ``dominant_source`` names which single source carries the width (with small
    calibration sets it is the R term, which is what makes "uninformative at
    this n" a precise, decomposed claim).
    """

    levels: tuple[int, ...]
    observed: tuple[float, ...]
    invert: tuple[float, ...]
    dagostini: tuple[float, ...]
    point: tuple[float, ...]
    method: str
    divergence: float
    ci_r_only: tuple[tuple[float, float], ...]
    ci_sampling_only: tuple[tuple[float, float], ...]
    ci_combined: tuple[tuple[float, float], ...]
    dominant_source: str
    n_boot: int
    seed: int


def _filled_response_matrix(
    true: np.ndarray, observed: np.ndarray, levels: Sequence[int]
) -> np.ndarray:
    """``response_matrix`` with the degenerate-column convention applied: a true
    class with zero samples (an empty column) is replaced by a uniform
    distribution over the OBSERVED levels -- maximally uncertain, a conservative
    widening rather than a singular matrix. This is the bootstrap-time handling
    the design calls for (a resample can draw zero of a rare true class), applied
    here so every R handed to an estimator is column-stochastic.
    """
    resp = response_matrix(true, observed, levels)
    empty = resp.sum(axis=0) == 0
    if empty.any():
        observed_levels = sorted({int(o) for o in np.asarray(observed)})
        level_index = {level: k for k, level in enumerate(levels)}
        fill = np.zeros(len(levels))
        for level in observed_levels:
            fill[level_index[level]] = 1.0 / len(observed_levels)
        resp = resp.copy()
        resp[:, empty] = fill[:, None]
    return resp


def _unfold_props(resp: np.ndarray, n_obs: np.ndarray, method: str, n_iter: int) -> np.ndarray:
    """Unfold ``n_obs`` with ``resp`` and return the corrected distribution as
    proportions (counts are conserved, so this is a rescale). A resample whose R
    is too degenerate to invert contributes the observed proportions (a no-
    information correction) rather than being dropped -- the keep-degenerate
    convention shared with the agreement bootstrap.
    """
    obs = np.asarray(n_obs, dtype=float)
    try:
        counts = (
            invert_unfold(resp, obs) if method == "invert" else dagostini_unfold(resp, obs, n_iter)
        )
    except np.linalg.LinAlgError:
        counts = obs
    total = counts.sum()
    return counts / total if total != 0 else counts


def _point_invert_props(resp: np.ndarray, n_obs: np.ndarray) -> np.ndarray:
    """Point matrix-inversion estimate as proportions, or all-NaN if R is
    singular. Unlike the bootstrap path, a singular POINT matrix is reported as
    NaN (the correction is unavailable) rather than falling back to the observed
    distribution, which would make the divergence diagnostic read as a real zero
    correction instead of an unidentified one.
    """
    obs = np.asarray(n_obs, dtype=float)
    try:
        counts = invert_unfold(resp, obs)
    except np.linalg.LinAlgError:
        return np.full(len(obs), np.nan)
    total = counts.sum()
    return counts / total if total != 0 else counts


def _per_level_ci(samples: np.ndarray, confidence: float) -> tuple[tuple[float, float], ...]:
    alpha = 1.0 - confidence
    lo = np.quantile(samples, alpha / 2, axis=0)
    hi = np.quantile(samples, 1 - alpha / 2, axis=0)
    return tuple((float(a), float(b)) for a, b in zip(lo, hi))


def _dominant_source(width_r: float, width_s: float, ratio: float = 1.25) -> str:
    """Which single resampling source carries the interval width: "comparable"
    when neither exceeds the other by more than ``ratio``, otherwise the wider.
    """
    hi, lo = max(width_r, width_s), min(width_r, width_s)
    if hi == 0 or (lo > 0 and hi / lo < ratio):
        return "comparable"
    return "R" if width_r >= width_s else "sampling"


def unfold_with_uncertainty(
    true: np.ndarray,
    observed: np.ndarray,
    n_obs: np.ndarray,
    *,
    method: str = "dagostini",
    n_iter: int = DEFAULT_N_ITER,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> UnfoldResult:
    """Unfold an observed level distribution and propagate its uncertainty.

    ``true`` and ``observed`` are paired per-sample calibration labels that build
    the response matrix R; ``n_obs`` is the observed (jury) count vector to
    correct, indexed by level 0..len(n_obs)-1. Both estimators are computed and
    their divergence reported; the bootstrap CI is on the requested ``method``.
    Three resampling passes decompose the uncertainty (R-only, sampling-only,
    combined), each drawn from an independent child stream of ``seed`` so the run
    is reproducible.
    """
    cal_true = np.asarray(true)
    cal_obs = np.asarray(observed)
    obs_counts = np.asarray(n_obs, dtype=float)
    if obs_counts.sum() <= 0:
        raise ValueError("n_obs must have a positive total to unfold")
    levels = list(range(len(obs_counts)))
    total = int(round(float(obs_counts.sum())))
    obs_props = obs_counts / obs_counts.sum()

    resp_point = _filled_response_matrix(cal_true, cal_obs, levels)
    # Observed mass in a level the calibration never produced (a zero response-
    # matrix row) cannot be attributed to any true class: D'Agostini would
    # silently drop it and renormalize, and the inverse is singular. Report both
    # estimators as unavailable (NaN) there rather than losing counts silently.
    if bool(((resp_point.sum(axis=1) == 0) & (obs_counts > 0)).any()):
        invert_props = np.full(len(levels), np.nan)
        dagostini_props = np.full(len(levels), np.nan)
    else:
        invert_props = _point_invert_props(resp_point, obs_counts)
        dagostini_props = _unfold_props(resp_point, obs_counts, "dagostini", n_iter)
    point = invert_props if method == "invert" else dagostini_props
    divergence = float(np.max(np.abs(dagostini_props - invert_props)))

    n_cal = len(cal_true)
    rng_r, rng_s, rng_c = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    r_only = np.empty((n_boot, len(levels)))
    sampling_only = np.empty((n_boot, len(levels)))
    combined = np.empty((n_boot, len(levels)))
    for b in range(n_boot):
        idx_r = rng_r.integers(0, n_cal, size=n_cal)
        resp_r = _filled_response_matrix(cal_true[idx_r], cal_obs[idx_r], levels)
        r_only[b] = _unfold_props(resp_r, obs_counts, method, n_iter)

        n_obs_s = rng_s.multinomial(total, obs_props).astype(float)
        sampling_only[b] = _unfold_props(resp_point, n_obs_s, method, n_iter)

        idx_c = rng_c.integers(0, n_cal, size=n_cal)
        resp_c = _filled_response_matrix(cal_true[idx_c], cal_obs[idx_c], levels)
        n_obs_c = rng_c.multinomial(total, obs_props).astype(float)
        combined[b] = _unfold_props(resp_c, n_obs_c, method, n_iter)

    ci_r = _per_level_ci(r_only, confidence)
    ci_s = _per_level_ci(sampling_only, confidence)
    ci_c = _per_level_ci(combined, confidence)
    width_r = float(np.mean([hi - lo for lo, hi in ci_r]))
    width_s = float(np.mean([hi - lo for lo, hi in ci_s]))
    dominant = _dominant_source(width_r, width_s)

    # When the reported point estimate is unavailable (NaN) -- a singular base for
    # the inversion method, or observed mass in an unmodeled level -- its interval
    # is unavailable too. The keep-degenerate obs-fallback in the loop is for
    # individual resamples, not the base estimate, so a finite obs-based CI next
    # to a NaN point would mislabel the uncorrected distribution as the interval.
    if bool(np.isnan(point).any()):
        nan_ci = tuple((float("nan"), float("nan")) for _ in levels)
        ci_r = ci_s = ci_c = nan_ci
        dominant = "comparable"

    return UnfoldResult(
        levels=tuple(levels),
        observed=tuple(float(x) for x in obs_props),
        invert=tuple(float(x) for x in invert_props),
        dagostini=tuple(float(x) for x in dagostini_props),
        point=tuple(float(x) for x in point),
        method=method,
        divergence=divergence,
        ci_r_only=ci_r,
        ci_sampling_only=ci_s,
        ci_combined=ci_c,
        dominant_source=dominant,
        n_boot=n_boot,
        seed=seed,
    )
