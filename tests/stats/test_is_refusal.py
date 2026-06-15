"""Tests for the is_refusal hook extracted from grounded_refusal.

The extraction is behavior-preserving: the 528 pre-existing tests must pass
untouched. These tests cover the new public function only.
"""

from agent_bench.evaluation.metrics import grounded_refusal, is_refusal


def test_phrase_refusal_detected():
    assert is_refusal("The documentation does not contain information about this.")


def test_canonical_refusal_detected():
    assert is_refusal("That topic is not in the FastAPI documentation provided.")


def test_plain_answer_not_refusal():
    assert not is_refusal("Path parameters are declared with curly braces.")


def test_not_in_the_without_documentation_anchor_not_refusal():
    assert not is_refusal("The value is not in the default range for this field.")


def test_empty_answer_not_refusal():
    assert not is_refusal("")


def test_grounded_refusal_still_vacuous_true_for_in_scope():
    assert grounded_refusal("any answer at all", "retrieval") is True


def test_grounded_refusal_delegates_for_out_of_scope():
    refusing = "No relevant information was found."
    assert grounded_refusal(refusing, "out_of_scope") is True
    assert grounded_refusal(refusing + " [source: a.md]", "out_of_scope") is False
