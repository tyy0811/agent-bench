"""Reference vectors generated with statsmodels 0.14.6, proportion_confint,
commands recorded in the implementation plan Task 3.1.
Do not regenerate silently; provenance comments must name tool and version.

Zero-failure boundary: statsmodels proportion_confint(0, n, method="beta") is
the TWO-SIDED interval, whose upper is 1-(alpha/2)**(1/n) and matches
clopper_pearson(0, n)'s two-sided upper. zero_failure_upper is the ONE-SIDED
bound 1-alpha**(1/n) (design spec headline), so it is validated against its
closed form, not against the two-sided statsmodels value (plan Task 3.1 note).
"""

import pytest

from stats import intervals


def test_wilson_against_statsmodels():
    lo, hi = intervals.wilson(17, 22, confidence=0.95)
    assert lo == pytest.approx(0.5656004682477828, abs=1e-6)
    assert hi == pytest.approx(0.8987696001475148, abs=1e-6)


def test_clopper_pearson_against_statsmodels():
    lo, hi = intervals.clopper_pearson(17, 22, confidence=0.95)
    assert lo == pytest.approx(0.5462963763378781, abs=1e-6)
    assert hi == pytest.approx(0.9217937396481075, abs=1e-6)


def test_clopper_pearson_at_zero_matches_statsmodels():
    # statsmodels beta(0, n) two-sided upper == clopper_pearson two-sided upper.
    assert intervals.clopper_pearson(0, 22)[1] == pytest.approx(0.15437251281557463, abs=1e-6)
    assert intervals.clopper_pearson(0, 110)[1] == pytest.approx(0.0329791940322197, abs=1e-6)


def test_rule_of_three_is_3_over_n():
    assert intervals.rule_of_three(22) == pytest.approx(3 / 22)
    assert intervals.rule_of_three(110) == pytest.approx(3 / 110)


def test_zero_failure_upper_one_sided_closed_form():
    # One-sided exact CP upper at k=0 is 1 - alpha**(1/n) (design spec headline),
    # distinct from the two-sided statsmodels value above.
    assert intervals.zero_failure_upper(22) == pytest.approx(1 - 0.05 ** (1 / 22), abs=1e-12)
    assert intervals.zero_failure_upper(110) == pytest.approx(1 - 0.05 ** (1 / 110), abs=1e-12)
    assert intervals.zero_failure_upper(27) == pytest.approx(1 - 0.05 ** (1 / 27), abs=1e-12)


def test_degenerate_inputs_rejected():
    with pytest.raises(ValueError):
        intervals.wilson(5, 0)
    with pytest.raises(ValueError):
        intervals.wilson(6, 5)
