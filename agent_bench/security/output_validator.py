"""Post-generation output validation gate.

Four deterministic checks:
  1. PII leakage: reuses PIIRedactor to detect PII in LLM output
  2. URL validation: URLs must appear in retrieved chunks
  3. Secret leakage: deny-list of API key formats and env var literals
  4. Blocklist scan: configurable forbidden patterns
"""

from __future__ import annotations

import re

from agent_bench.security.pii_redactor import PIIRedactor
from agent_bench.security.types import OutputVerdict

# Always-on secret-leakage deny list. These fire regardless of config.
# Matches the well-known API-key prefixes and the common env var literals
# that a docs assistant should never emit.
_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai_api_key_format", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_\-]{20,}")),
    ("anthropic_api_key_format", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("google_api_key_format", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("aws_access_key_format", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token_format", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("bearer_token_header", re.compile(
        r"\b[Bb]earer\s+[A-Za-z0-9_\-\.=]{20,}",
    )),
    ("env_var_literal", re.compile(
        r"\b(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|"
        r"AWS_SECRET(?:_ACCESS_KEY)?|AWS_ACCESS_KEY(?:_ID)?|"
        r"GITHUB_TOKEN|DATABASE_URL|DB_PASSWORD)\s*=\s*\S+",
    )),
]


class OutputValidator:
    """Validate LLM output before returning to user."""

    def __init__(
        self,
        pii_check: bool = True,
        url_check: bool = True,
        secret_check: bool = True,
        blocklist: list[str] | None = None,
    ) -> None:
        self.pii_check = pii_check
        self.url_check = url_check
        self.secret_check = secret_check
        self.blocklist_patterns = [re.compile(p) for p in (blocklist or [])]
        if pii_check:
            self._pii = PIIRedactor(mode="detect_only")

    def validate(
        self,
        output: str,
        retrieved_chunks: list[str],
    ) -> OutputVerdict:
        """Run all configured checks. Returns verdict with violations."""
        violations: list[str] = []

        if self.pii_check:
            violations.extend(self._check_pii(output))

        if self.url_check:
            violations.extend(self._check_urls(output, retrieved_chunks))

        if self.secret_check:
            violations.extend(self._check_secrets(output))

        if self.blocklist_patterns:
            violations.extend(self._check_blocklist(output))

        passed = len(violations) == 0
        return OutputVerdict(
            passed=passed,
            violations=violations,
            action="pass" if passed else "block",
        )

    def _check_secrets(self, output: str) -> list[str]:
        """Fail closed on known-secret formats and env var assignments.

        These patterns never match legitimate FastAPI / Kubernetes doc
        content. Any hit is a leaked credential that must block the
        response before the client sees it.
        """
        violations = []
        for name, pattern in _SECRET_PATTERNS:
            if pattern.search(output):
                violations.append(f"secret_leakage: {name} detected in output")
        return violations

    def _check_pii(self, output: str) -> list[str]:
        result = self._pii.redact(output)
        if result.redactions_count > 0:
            types = ", ".join(result.types_found)
            return [f"pii_leakage: {types} detected in output"]
        return []

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Strip trailing punctuation then trailing slashes for comparison."""
        return url.rstrip(".,;:").rstrip("/")

    def _check_urls(self, output: str, retrieved_chunks: list[str]) -> list[str]:
        url_pattern = re.compile(r"https?://[^\s\)\"'>]+")
        output_urls = url_pattern.findall(output)
        if not output_urls:
            return []

        chunk_text = " ".join(retrieved_chunks)
        chunk_urls_normalized = {self._normalize_url(u) for u in url_pattern.findall(chunk_text)}

        hallucinated = []
        for url in output_urls:
            if self._normalize_url(url) not in chunk_urls_normalized:
                hallucinated.append(url)

        if hallucinated:
            return [f"url_hallucination: {url}" for url in set(hallucinated)]
        return []

    def _check_blocklist(self, output: str) -> list[str]:
        violations = []
        for pattern in self.blocklist_patterns:
            if pattern.search(output):
                violations.append(f"blocklist: matched pattern '{pattern.pattern}'")
        return violations
