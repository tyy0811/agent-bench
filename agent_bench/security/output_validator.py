"""Post-generation output validation gate.

Three deterministic checks:
  1. PII leakage: reuses PIIRedactor to detect PII in LLM output
  2. URL validation: URLs must appear in retrieved chunks
  3. Blocklist scan: configurable forbidden patterns
"""

from __future__ import annotations

import re

from agent_bench.security.pii_redactor import PIIRedactor
from agent_bench.security.types import OutputVerdict


class OutputValidator:
    """Validate LLM output before returning to user."""

    def __init__(
        self,
        pii_check: bool = True,
        url_check: bool = True,
        blocklist: list[str] | None = None,
    ) -> None:
        self.pii_check = pii_check
        self.url_check = url_check
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

        if self.blocklist_patterns:
            violations.extend(self._check_blocklist(output))

        passed = len(violations) == 0
        return OutputVerdict(
            passed=passed,
            violations=violations,
            action="pass" if passed else "block",
        )

    def _check_pii(self, output: str) -> list[str]:
        result = self._pii.redact(output)
        if result.redactions_count > 0:
            types = ", ".join(result.types_found)
            return [f"pii_leakage: {types} detected in output"]
        return []

    def _check_urls(self, output: str, retrieved_chunks: list[str]) -> list[str]:
        url_pattern = re.compile(r"https?://[^\s\)\"'>]+")
        output_urls = set(url_pattern.findall(output))
        if not output_urls:
            return []

        chunk_text = " ".join(retrieved_chunks)
        chunk_urls = set(url_pattern.findall(chunk_text))

        hallucinated = output_urls - chunk_urls
        if hallucinated:
            return [f"url_hallucination: {url}" for url in hallucinated]
        return []

    def _check_blocklist(self, output: str) -> list[str]:
        violations = []
        for pattern in self.blocklist_patterns:
            if pattern.search(output):
                violations.append(f"blocklist: matched pattern '{pattern.pattern}'")
        return violations
