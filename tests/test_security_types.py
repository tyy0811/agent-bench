"""Tests for security type definitions."""

from agent_bench.security.types import OutputVerdict, SecurityVerdict


class TestSecurityVerdict:
    def test_safe_verdict(self):
        v = SecurityVerdict(safe=True, tier="heuristic", confidence=1.0)
        assert v.safe is True
        assert v.tier == "heuristic"
        assert v.confidence == 1.0
        assert v.matched_pattern is None

    def test_unsafe_verdict_with_pattern(self):
        v = SecurityVerdict(
            safe=False, tier="heuristic", confidence=1.0,
            matched_pattern="ignore_previous",
        )
        assert v.safe is False
        assert v.matched_pattern == "ignore_previous"

    def test_classifier_verdict(self):
        v = SecurityVerdict(safe=False, tier="classifier", confidence=0.92)
        assert v.tier == "classifier"
        assert v.confidence == 0.92


class TestOutputVerdict:
    def test_passed(self):
        v = OutputVerdict(passed=True, violations=[], action="pass")
        assert v.passed is True
        assert v.action == "pass"

    def test_blocked(self):
        v = OutputVerdict(
            passed=False,
            violations=["pii_leakage: EMAIL detected"],
            action="block",
        )
        assert v.passed is False
        assert len(v.violations) == 1
        assert v.action == "block"
