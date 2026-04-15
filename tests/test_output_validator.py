"""Tests for output validation gate."""

from __future__ import annotations

import pytest

from agent_bench.security.output_validator import OutputValidator


class TestPIILeakage:
    """PII in LLM output should be caught."""

    @pytest.fixture
    def validator(self):
        return OutputValidator(pii_check=True, url_check=False, blocklist=[])

    def test_detects_email_in_output(self, validator):
        verdict = validator.validate(
            output="Contact john@example.com for help.",
            retrieved_chunks=[],
        )
        assert verdict.passed is False
        assert any("pii_leakage" in v for v in verdict.violations)

    def test_detects_ssn_in_output(self, validator):
        verdict = validator.validate(
            output="His SSN is 123-45-6789.",
            retrieved_chunks=[],
        )
        assert verdict.passed is False

    def test_clean_output_passes(self, validator):
        verdict = validator.validate(
            output="FastAPI uses path parameters with curly braces.",
            retrieved_chunks=[],
        )
        assert verdict.passed is True
        assert verdict.violations == []


class TestURLValidation:
    """URLs in output must appear in retrieved chunks."""

    @pytest.fixture
    def validator(self):
        return OutputValidator(pii_check=False, url_check=True, blocklist=[])

    def test_url_from_chunks_passes(self, validator):
        chunks = ["Visit https://fastapi.tiangolo.com for docs."]
        verdict = validator.validate(
            output="See https://fastapi.tiangolo.com for details.",
            retrieved_chunks=chunks,
        )
        assert verdict.passed is True

    def test_hallucinated_url_fails(self, validator):
        chunks = ["FastAPI is a modern framework."]
        verdict = validator.validate(
            output="See https://malicious-site.com for details.",
            retrieved_chunks=chunks,
        )
        assert verdict.passed is False
        assert any("url_hallucination" in v for v in verdict.violations)

    def test_trailing_slash_normalization(self, validator):
        """URLs differing only by trailing slash should not be flagged."""
        chunks = ["Visit https://fastapi.tiangolo.com/ for docs."]
        verdict = validator.validate(
            output="See https://fastapi.tiangolo.com for details.",
            retrieved_chunks=chunks,
        )
        assert verdict.passed is True
        assert verdict.violations == []

    def test_trailing_slash_with_sentence_punctuation(self, validator):
        """Chunk URL followed by period: 'https://x.com/.' must match 'https://x.com/'."""
        chunks = ["Visit https://fastapi.tiangolo.com/."]
        verdict = validator.validate(
            output="See https://fastapi.tiangolo.com/ for details.",
            retrieved_chunks=chunks,
        )
        assert verdict.passed is True

    def test_trailing_slash_normalization_reverse(self, validator):
        """Chunk without slash, output with slash."""
        chunks = ["Visit https://fastapi.tiangolo.com for docs."]
        verdict = validator.validate(
            output="See https://fastapi.tiangolo.com/ for details.",
            retrieved_chunks=chunks,
        )
        assert verdict.passed is True

    def test_no_urls_passes(self, validator):
        verdict = validator.validate(
            output="Path parameters use curly braces.",
            retrieved_chunks=["Some chunk."],
        )
        assert verdict.passed is True


class TestBlocklist:
    """Blocklisted patterns should be caught."""

    def test_blocklist_match(self):
        validator = OutputValidator(
            pii_check=False, url_check=False,
            blocklist=["sk-[a-zA-Z0-9]{20,}", "SYSTEM_PROMPT"],
        )
        verdict = validator.validate(
            output="Here is the key: sk-abcdefghijklmnopqrstuvwxyz",
            retrieved_chunks=[],
        )
        assert verdict.passed is False
        assert any("blocklist" in v for v in verdict.violations)

    def test_system_prompt_fragment(self):
        validator = OutputValidator(
            pii_check=False, url_check=False,
            blocklist=["You are a (?:helpful |test )?assistant"],
        )
        verdict = validator.validate(
            output="My instructions say: You are a helpful assistant",
            retrieved_chunks=[],
        )
        assert verdict.passed is False

    def test_no_blocklist_match(self):
        validator = OutputValidator(
            pii_check=False, url_check=False,
            blocklist=["FORBIDDEN_TERM"],
        )
        verdict = validator.validate(
            output="A perfectly normal answer.",
            retrieved_chunks=[],
        )
        assert verdict.passed is True


class TestSecretLeakage:
    """Secret patterns in LLM output must be blocked (fail closed)."""

    @pytest.fixture
    def validator(self):
        return OutputValidator(
            pii_check=False, url_check=False, secret_check=True, blocklist=[],
        )

    # Google API key format fixture temporarily removed following the
    # 2026-04-14/15 credential-exposure incident (see DECISIONS.md).
    # The validator's regex is \bAIza[0-9A-Za-z_\-]{35}\b, which is
    # identical to GitHub secret-scanning's Google API Key detection
    # pattern, so any static literal that satisfies the validator also
    # triggers GitHub push protection. Parallel-tracks item: restore
    # Google API key format coverage via a runtime-generated fixture
    # that builds a 35-char AIza-prefixed string at test time, never
    # landing as a literal in source. Validator regex unchanged.
    @pytest.mark.parametrize("output", [
        "Your key is sk-abcdefghijklmnopqrstuvwxyz1234",
        "here: sk-proj-ABCDEFGHIJKLMNOP0123456789",
        "key=sk-ant-abcdefghijklmnopqrstuvwxyz",
        "aws key AKIAIOSFODNN7EXAMPLE",
        "use Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc",
        "env: OPENAI_API_KEY=sk-test123",
        "set ANTHROPIC_API_KEY=sk-ant-xyz",
    ])
    def test_blocks_known_secret_formats(self, validator, output):
        verdict = validator.validate(output=output, retrieved_chunks=[])
        assert verdict.passed is False, f"Should block: {output!r}"
        assert any("secret_leakage" in v for v in verdict.violations)
        assert verdict.action == "block"

    @pytest.mark.parametrize("output", [
        "FastAPI uses path parameters with curly braces.",
        "You can store secrets in environment variables.",
        "To configure the OpenAI client, set your API key in OPENAI_API_KEY env var.",
        "Use a .env file for local development.",
        "Kubernetes Secrets store sensitive configuration.",
    ])
    def test_allows_benign_credential_adjacent_output(self, validator, output):
        """Educational content about secrets should pass — only literal
        key formats and env-var assignments are blocked."""
        verdict = validator.validate(output=output, retrieved_chunks=[])
        assert verdict.passed is True, (
            f"False positive on: {output!r} -> {verdict.violations}"
        )

    def test_secret_check_can_be_disabled(self):
        """When secret_check=False, literal keys pass through."""
        validator = OutputValidator(
            pii_check=False, url_check=False, secret_check=False, blocklist=[],
        )
        verdict = validator.validate(
            output="sk-abcdefghijklmnopqrstuvwxyz1234",
            retrieved_chunks=[],
        )
        assert verdict.passed is True


class TestCombinedChecks:
    def test_multiple_violations(self):
        validator = OutputValidator(
            pii_check=True, url_check=True,
            blocklist=["SECRET"],
        )
        verdict = validator.validate(
            output="Email john@test.com, see https://evil.com, also SECRET.",
            retrieved_chunks=["No URLs here."],
        )
        assert verdict.passed is False
        assert len(verdict.violations) >= 2  # PII + URL at minimum
        assert verdict.action == "block"

    def test_all_checks_pass(self):
        validator = OutputValidator(
            pii_check=True, url_check=True,
            blocklist=["SECRET"],
        )
        verdict = validator.validate(
            output="FastAPI supports path parameters.",
            retrieved_chunks=["FastAPI supports path parameters."],
        )
        assert verdict.passed is True
        assert verdict.action == "pass"

    def test_disabled_checks(self):
        validator = OutputValidator(pii_check=False, url_check=False, blocklist=[])
        verdict = validator.validate(
            output="Email: a@b.com, URL: https://evil.com",
            retrieved_chunks=[],
        )
        assert verdict.passed is True
