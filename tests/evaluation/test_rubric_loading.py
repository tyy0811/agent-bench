"""Tests for Rubric markdown loader: construction validation, hash, permutation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_bench.evaluation.judges.base import Rubric

FIXTURES = Path(__file__).parent / "fixtures"


class TestRubricLoading:
    def test_load_valid_binary(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_binary.md")
        assert r.dimension == "groundedness"
        assert r.scale == "binary"
        assert r.reference_based is True
        assert r.abstain_allowed is True
        assert len(r.levels) == 2

    def test_load_valid_three_point(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        assert r.dimension == "relevance"
        assert r.scale == "three_point"
        assert len(r.levels) == 3


class TestRubricValidationErrors:
    @pytest.mark.parametrize(
        "fixture_name,error_substring",
        [
            ("rubrics_invalid_scale.md", "scale"),
            ("rubrics_invalid_arity.md", "arity"),
            ("rubrics_invalid_no_examples.md", "anchored example"),
            ("rubrics_invalid_no_frontmatter.md", "frontmatter"),
        ],
    )
    def test_construction_raises_with_path_and_field(
        self, fixture_name: str, error_substring: str
    ):
        path = FIXTURES / fixture_name
        with pytest.raises(ValueError) as exc_info:
            Rubric.from_markdown_file(path)
        msg = str(exc_info.value)
        # Error must mention the file path and the field-level reason
        assert fixture_name in msg, f"Path missing from error: {msg}"
        assert error_substring in msg.lower(), (
            f"Expected '{error_substring}' in error message: {msg}"
        )


class TestRubricSourceHash:
    def test_source_hash_deterministic(self):
        r1 = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_binary.md")
        r2 = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_binary.md")
        assert r1.source_hash == r2.source_hash
        # SHA-256 hex is 64 chars
        assert len(r1.source_hash) == 64

    def test_source_hash_changes_with_content(self):
        r1 = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_binary.md")
        r2 = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        assert r1.source_hash != r2.source_hash


class TestRubricPermutation:
    def test_render_prompt_seed_0_unchanged(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        prompt = r.render_prompt(level_permutation_seed=0)
        # Default: levels in original 0, 1, 2 order
        idx0 = prompt.index("Score 0")
        idx1 = prompt.index("Score 1")
        idx2 = prompt.index("Score 2")
        assert idx0 < idx1 < idx2

    def test_render_prompt_seed_reproducibility(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        p1 = r.render_prompt(level_permutation_seed=42)
        p2 = r.render_prompt(level_permutation_seed=42)
        assert p1 == p2

    def test_render_prompt_different_seed_different_order(self):
        r = Rubric.from_markdown_file(FIXTURES / "rubrics_valid_three_point.md")
        # Try several seeds; at least one should produce a non-default order
        # (with 3! = 6 permutations, the chance all 5 seeds produce identity
        # is (1/6)^5 ≈ 1e-4, negligible)
        default = r.render_prompt(level_permutation_seed=0)
        differs = any(
            r.render_prompt(level_permutation_seed=s) != default
            for s in (1, 2, 3, 7, 13)
        )
        assert differs, "No seed produced a permutation different from default"
