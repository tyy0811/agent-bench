"""Tests for PII redaction."""

from __future__ import annotations

import pytest

from agent_bench.security.pii_redactor import PIIRedactor


class TestRegexPatterns:
    """Test each regex pattern individually."""

    @pytest.fixture
    def redactor(self):
        return PIIRedactor(redact_patterns=["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP_ADDRESS"])

    def test_email_redaction(self, redactor):
        text = "Contact john@example.com for details."
        result = redactor.redact(text)
        assert "john@example.com" not in result.text
        assert "[EMAIL_1]" in result.text
        assert "EMAIL" in result.types_found

    def test_multiple_emails(self, redactor):
        text = "Emails: a@b.com and c@d.com"
        result = redactor.redact(text)
        assert "[EMAIL_1]" in result.text
        assert "[EMAIL_2]" in result.text
        assert result.redactions_count >= 2

    def test_phone_us(self, redactor):
        text = "Call 555-123-4567 now."
        result = redactor.redact(text)
        assert "555-123-4567" not in result.text
        assert "PHONE" in result.types_found

    def test_phone_international(self, redactor):
        text = "Call +1-555-123-4567 now."
        result = redactor.redact(text)
        assert "+1-555-123-4567" not in result.text

    def test_ssn(self, redactor):
        text = "SSN: 123-45-6789"
        result = redactor.redact(text)
        assert "123-45-6789" not in result.text
        assert "SSN" in result.types_found

    def test_credit_card(self, redactor):
        text = "Card: 4111-1111-1111-1111"
        result = redactor.redact(text)
        assert "4111-1111-1111-1111" not in result.text
        assert "CREDIT_CARD" in result.types_found

    def test_credit_card_no_dashes(self, redactor):
        text = "Card: 4111111111111111"
        result = redactor.redact(text)
        assert "4111111111111111" not in result.text

    def test_ip_address(self, redactor):
        text = "Server at 192.168.1.100 is down."
        result = redactor.redact(text)
        assert "192.168.1.100" not in result.text
        assert "IP_ADDRESS" in result.types_found

    def test_no_pii(self, redactor):
        text = "FastAPI is a modern web framework."
        result = redactor.redact(text)
        assert result.text == text
        assert result.redactions_count == 0
        assert result.types_found == []

    def test_mixed_pii(self, redactor):
        text = "Email john@test.com, SSN 123-45-6789, call 555-123-4567."
        result = redactor.redact(text)
        assert "john@test.com" not in result.text
        assert "123-45-6789" not in result.text
        assert "555-123-4567" not in result.text
        assert result.redactions_count == 3


class TestRedactionModes:
    def test_detect_only_mode(self):
        redactor = PIIRedactor(redact_patterns=["EMAIL"], mode="detect_only")
        result = redactor.redact("Email: a@b.com")
        assert result.text == "Email: a@b.com"  # unchanged
        assert result.redactions_count == 1
        assert "EMAIL" in result.types_found

    def test_passthrough_mode(self):
        redactor = PIIRedactor(redact_patterns=["EMAIL"], mode="passthrough")
        result = redactor.redact("Email: a@b.com")
        assert result.text == "Email: a@b.com"
        assert result.redactions_count == 0

    def test_redact_mode(self):
        redactor = PIIRedactor(redact_patterns=["EMAIL"], mode="redact")
        result = redactor.redact("Email: a@b.com")
        assert "a@b.com" not in result.text
        assert "[EMAIL_1]" in result.text


class TestPlaceholderConsistency:
    def test_same_entity_same_placeholder_within_request(self):
        """Same PII value gets the same placeholder in one redact() call."""
        redactor = PIIRedactor(redact_patterns=["EMAIL"])
        text = "From a@b.com to you. Reply to a@b.com"
        result = redactor.redact(text)
        # Both occurrences of a@b.com should get the same placeholder
        assert result.text.count("[EMAIL_1]") == 2

    def test_different_entities_different_placeholders(self):
        redactor = PIIRedactor(redact_patterns=["EMAIL"])
        text = "From a@b.com to c@d.com"
        result = redactor.redact(text)
        assert "[EMAIL_1]" in result.text
        assert "[EMAIL_2]" in result.text


class TestSelectivePatterns:
    def test_only_selected_patterns_run(self):
        """Only configured patterns trigger redaction."""
        redactor = PIIRedactor(redact_patterns=["EMAIL"])  # Only email
        text = "Email a@b.com, SSN 123-45-6789"
        result = redactor.redact(text)
        assert "a@b.com" not in result.text
        assert "123-45-6789" in result.text  # SSN untouched
