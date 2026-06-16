"""The stats.agreement module on the real calibration join must reproduce the
canonical kappa_table.md headline row (jury_kappa_weighted_v1_1). Reference
values are the committed docs/_generated/kappa_table.md point estimates and Ns
(groundedness AC1=1.000 N=26, relevance AC1=1.000 N=30, completeness kappa=0.416
N=26), produced by the independent calibration pipeline in
agent_bench/evaluation/calibration/. Matching them confirms the adapter join."""

from pathlib import Path

import pytest

from stats_adapters.calibration_agreement import agreement_intervals

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "measurements" / "2026-05-04-judge-calibration-labels.jsonl"
PREDS = ROOT / "results" / "calibration_v1_judge_jury_kappa_weighted_v1_1.json"

# (metric, n, point) per dimension, from docs/_generated/kappa_table.md row
# jury_kappa_weighted_v1_1.
EXPECTED = {
    "groundedness": ("AC1", 26, 1.000),
    "relevance": ("AC1", 30, 1.000),
    "completeness": ("kappa", 26, 0.416),
}


def test_reproduces_kappa_table_v1_1_points_and_n():
    by_dim = {r.dimension: r for r in agreement_intervals(LABELS, PREDS)}
    assert set(by_dim) == set(EXPECTED)
    for dim, (metric, n, point) in EXPECTED.items():
        r = by_dim[dim]
        assert r.metric == metric
        assert r.n == n
        assert r.point == pytest.approx(point, abs=1e-3)
        assert r.ci_low <= r.point <= r.ci_high


def test_intervals_are_seeded_and_reproducible():
    a = agreement_intervals(LABELS, PREDS, seed=20260611)
    b = agreement_intervals(LABELS, PREDS, seed=20260611)
    assert [(x.ci_low, x.ci_high) for x in a] == [(y.ci_low, y.ci_high) for y in b]
