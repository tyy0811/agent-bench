"""Tests for hand-rolled Cohen's kappa, Gwet's AC2, bootstrap CI."""

from __future__ import annotations

import json as _json
from pathlib import Path

import pytest

from agent_bench.evaluation.calibration.metrics import (
    bootstrap_ci,
    cohen_kappa,
    gwets_ac2,
)


class TestCohenKappaHandComputed:
    def test_perfect_agreement_kappa_one(self):
        # 5 ones, 5 zeros, both raters identical
        # P_o = 1.0
        # P_e = (5/10 * 5/10) + (5/10 * 5/10) = 0.5
        # κ = (1.0 - 0.5) / (1.0 - 0.5) = 1.0
        y1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        assert cohen_kappa(y1, y2) == pytest.approx(1.0)

    def test_complete_disagreement_kappa_negative(self):
        # 5 ones, 5 zeros for each, but inverted
        # P_o = 0.0; P_e = 0.5 → κ = -1.0
        y1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        assert cohen_kappa(y1, y2) == pytest.approx(-1.0)

    def test_chance_agreement_kappa_zero(self):
        # 2x2 confusion matrix where observed = chance.
        # P(0)=0.5, P(1)=0.5 for both; uniform 0.25/0.25/0.25/0.25 →
        # P_o = 0.5, P_e = 0.5, κ = 0.
        y1 = [0, 0, 1, 1]
        y2 = [0, 1, 0, 1]
        assert cohen_kappa(y1, y2) == pytest.approx(0.0)


class TestGwetsAC2HandComputed:
    def test_perfect_agreement(self):
        y1 = [0, 0, 1, 1]
        y2 = [0, 0, 1, 1]
        assert gwets_ac2(y1, y2) == pytest.approx(1.0)

    def test_complete_disagreement(self):
        y1 = [0, 0, 1, 1]
        y2 = [1, 1, 0, 0]
        # AC2 with q=2 categories: observed agreement = 0;
        # chance term = (1/1) * sum p_k(1-p_k) computed from mean marginals
        assert gwets_ac2(y1, y2) == pytest.approx(-1.0)

    def test_mid_range(self):
        y1 = [0, 0, 1, 1]
        y2 = [0, 0, 1, 0]
        # 3/4 agree → AC2 should land in (0, 1)
        result = gwets_ac2(y1, y2)
        assert -1.0 <= result <= 1.0
        assert result > 0


class TestBootstrapCI:
    def test_returns_point_lo_hi_tuple(self):
        y1 = [0, 0, 1, 1, 1, 0, 1, 0]
        y2 = [0, 1, 1, 1, 1, 0, 1, 0]
        result = bootstrap_ci(y1, y2, cohen_kappa, n_iter=100, seed=42)
        assert len(result) == 3
        point, lo, hi = result
        assert lo <= point <= hi

    def test_seed_reproducibility(self):
        y1 = [0, 0, 1, 1, 1, 0, 1, 0]
        y2 = [0, 1, 1, 1, 1, 0, 1, 0]
        r1 = bootstrap_ci(y1, y2, cohen_kappa, n_iter=200, seed=42)
        r2 = bootstrap_ci(y1, y2, cohen_kappa, n_iter=200, seed=42)
        assert r1 == r2


# --- sklearn-parity fixtures ---
#
# Generated against scikit-learn==1.5.2 cohen_kappa_score on 2026-05-04.
# To regenerate: scripts/_dev/generate_kappa_fixtures.py
# DO NOT add scikit-learn to the project's runtime dependencies — these
# constants are the contract; the project hand-rolls the implementation.

SKLEARN_KAPPA_FIXTURES: dict[str, float] = {
    # Generated against scikit-learn==1.5.2 cohen_kappa_score on 2026-05-04.
    # To regenerate: scripts/_dev/generate_kappa_fixtures.py
    "imbalanced_binary": 0.2105263158,
    "three_point_one_diagonal_swap": 0.8507462687,
    "weighted_ordinal_drift_linear": 0.6666666667,
}

SKLEARN_KAPPA_INPUTS: dict[str, dict] = {
    "imbalanced_binary": {
        "y1": [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        "y2": [1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
        "weights": None,
    },
    "three_point_one_diagonal_swap": {
        "y1": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0],
        "y2": [0, 1, 1, 1, 2, 2, 0, 1, 2, 0],
        "weights": None,
    },
    "weighted_ordinal_drift_linear": {
        "y1": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        "y2": [0, 1, 2, 1, 1, 2, 0, 2, 2, 1],
        "weights": "linear",
    },
}


class TestSklearnKappaParity:
    @pytest.mark.parametrize("case_name", list(SKLEARN_KAPPA_FIXTURES.keys()))
    def test_matches_sklearn(self, case_name: str):
        case = SKLEARN_KAPPA_INPUTS[case_name]
        expected = SKLEARN_KAPPA_FIXTURES[case_name]
        actual = cohen_kappa(case["y1"], case["y2"], weights=case["weights"])
        # Tolerance 1e-7 accommodates sklearn's printed precision of 10 decimals
        assert actual == pytest.approx(expected, abs=1e-7), (
            f"hand-rolled cohen_kappa diverged from sklearn 1.5.2 on case "
            f"{case_name!r}: hand-rolled={actual} sklearn={expected}"
        )


class TestSklearnInputsCrossCheck:
    """Catches 'updated CASES list, forgot to regenerate' failure mode."""

    def test_inputs_match_committed_json(self):
        json_path = Path(__file__).parent / "fixtures" / "sklearn_kappa_inputs.json"
        if not json_path.exists():
            pytest.skip(
                "sklearn_kappa_inputs.json not yet generated — see "
                "scripts/_dev/generate_kappa_fixtures.py"
            )
        on_disk = _json.loads(json_path.read_text())
        assert set(SKLEARN_KAPPA_INPUTS.keys()) == set(on_disk.keys()), (
            "SKLEARN_KAPPA_INPUTS keys diverge from sklearn_kappa_inputs.json — "
            "regenerate via scripts/_dev/generate_kappa_fixtures.py"
        )
        for name in SKLEARN_KAPPA_INPUTS:
            assert SKLEARN_KAPPA_INPUTS[name] == on_disk[name], (
                f"Input mismatch for case {name!r} — regenerate fixtures"
            )
