"""Kappa and AC1 references: hand-computed 2x2 example, n=50,
a=20 (yes/yes), b=5 (yes/no), c=10 (no/yes), d=15 (no/no).
po = 35/50 = 0.7. Rater1 yes = 0.5, rater2 yes = 0.6.
kappa: pe = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = (0.7-0.5)/0.5 = 0.4.
AC1: pgamma = (0.5+0.6)/2 = 0.55; e = 2*0.55*0.45 = 0.495;
AC1 = (0.7-0.495)/(1-0.495) = 0.405940594...
Provenance: hand computation above; cross-checkable against the R irrCAC
package (gwet.ac1.raw) if ever in doubt."""

import numpy as np
import pytest

from stats import agreement

A = np.array([1] * 20 + [1] * 5 + [0] * 10 + [0] * 15)
B = np.array([1] * 20 + [0] * 5 + [1] * 10 + [0] * 15)


def test_cohen_kappa_hand_worked():
    assert agreement.cohen_kappa(A, B) == pytest.approx(0.4, abs=1e-12)


def test_gwet_ac1_hand_worked():
    assert agreement.gwet_ac1(A, B) == pytest.approx(0.205 / 0.505, abs=1e-9)


def test_cohen_kappa_matches_sklearn():
    # Independent reference (guardrail 4): scikit-learn's cohen_kappa_score is a
    # separate implementation and must agree with ours on the same data.
    skm = pytest.importorskip("sklearn.metrics")
    assert agreement.cohen_kappa(A, B) == pytest.approx(skm.cohen_kappa_score(A, B), abs=1e-12)


def test_bootstrap_ci_contains_point_and_is_seeded():
    lo1, hi1 = agreement.bootstrap_agreement_ci(agreement.cohen_kappa, A, B, seed=20260611)
    lo2, hi2 = agreement.bootstrap_agreement_ci(agreement.cohen_kappa, A, B, seed=20260611)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= 0.4 <= hi1


def test_degenerate_single_category_kappa_is_nan_ac1_defined():
    ones = np.ones(30)
    assert np.isnan(agreement.cohen_kappa(ones, ones))
    assert agreement.gwet_ac1(ones, ones) == pytest.approx(1.0)
