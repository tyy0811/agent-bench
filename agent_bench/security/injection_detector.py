"""Prompt injection detection.

Two-tier detection:
  Tier 1 — Heuristic regex (local, <1ms): catches common injection patterns
  Tier 2 — DeBERTa classifier (Modal GPU): high-confidence arbiter

Deployments without GPU run heuristic-only.
"""

from __future__ import annotations

import base64
import re

import structlog

from agent_bench.security.types import SecurityVerdict

logger = structlog.get_logger()

# --- Tier 1: Heuristic patterns ---
# Each pattern is (name, compiled_regex).
# Patterns use word boundaries and case-insensitive matching.
# Ordered from most specific to least specific.

_HEURISTIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Role/identity hijacking
    ("role_switch", re.compile(
        r"\byou\s+are\s+now\b", re.IGNORECASE
    )),
    ("act_as", re.compile(
        r"\b(?:from\s+now\s+on\s+)?(?:you\s+will\s+)?act\s+(?:as\s+(?:if\s+)?)", re.IGNORECASE
    )),
    ("pretend", re.compile(
        r"\bpretend\s+you\s+are\b", re.IGNORECASE
    )),
    # Instruction override
    ("ignore_previous", re.compile(
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier|your|my)\s+(?:instructions?|context|rules|guidelines|directives)\b",
        re.IGNORECASE,
    )),
    ("disregard", re.compile(
        r"\bdisregard\s+(?:all\s+)?(?:your|previous|prior)?\s*(?:instructions?|rules|guidelines)\b",
        re.IGNORECASE,
    )),
    ("forget_instructions", re.compile(
        r"\bforget\s+(?:all\s+|everything\s+)?(?:you\s+were\s+told|previous|prior|your\s+instructions?|your\s+context)\b",
        re.IGNORECASE,
    )),
    ("do_not_follow", re.compile(
        r"\bdo\s+not\s+follow\s+(?:your\s+)?(?:original\s+)?instructions?\b",
        re.IGNORECASE,
    )),
    # System prompt extraction
    ("reveal_prompt", re.compile(
        r"\b(?:reveal|show|display|output|print|repeat|tell\s+me|give\s+me|share|leak|dump|paste|write\s+out)\s+(?:me\s+)?(?:your\s+)?(?:system\s+prompt|initial\s+instructions?|instructions?\s+verbatim|original\s+instructions?|hidden\s+prompt|internal\s+prompt)\b",
        re.IGNORECASE,
    )),
    ("what_is_prompt", re.compile(
        r"\bwhat\s+(?:is|are)\s+your\s+(?:system\s+prompt|instructions?|initial\s+prompt|hidden\s+prompt)\b",
        re.IGNORECASE,
    )),
    # Direct prompt requests (catches "give me your system prompt")
    ("give_prompt", re.compile(
        r"\b(?:give|send|copy|provide)\s+(?:me\s+)?(?:the\s+|your\s+)?(?:system\s+prompt|full\s+prompt|original\s+prompt|system\s+instructions?|internal\s+instructions?|hidden\s+instructions?)\b",
        re.IGNORECASE,
    )),
    # Prompt as a noun target (catches "I want your system prompt")
    ("want_prompt", re.compile(
        r"\b(?:i\s+want|i\s+need|hand\s+over|access)\s+(?:to\s+see\s+)?(?:your\s+)?(?:system\s+prompt|internal\s+prompt|original\s+instructions?|system\s+instructions?)\b",
        re.IGNORECASE,
    )),
    # System message injection
    ("system_prefix", re.compile(
        r"^(?:system\s*:|###\s*SYSTEM\s*###|```system)", re.IGNORECASE | re.MULTILINE
    )),
    ("system_block", re.compile(
        r"```system\b", re.IGNORECASE
    )),
    # Jailbreak keywords
    ("jailbreak", re.compile(
        r"\b(?:DAN|jailbreak|jailbroken|unrestricted\s+(?:AI|assistant|mode))\b",
        re.IGNORECASE,
    )),
    ("no_restrictions", re.compile(
        r"\b(?:no|without|remove)\s+(?:content\s+policy|safety\s+guidelines|restrictions|filters|guardrails)\b",
        re.IGNORECASE,
    )),
]


class InjectionDetector:
    """Two-tier injection detection."""

    def __init__(
        self,
        tiers: list[str] | None = None,
        classifier_url: str = "",
        enabled: bool = True,
    ) -> None:
        self.tiers = tiers or ["heuristic", "classifier"]
        self.classifier_url = classifier_url
        self.enabled = enabled

    def detect(self, text: str) -> SecurityVerdict:
        """Run detection tiers in order. Return on first match."""
        if not self.enabled or not text.strip():
            return SecurityVerdict(safe=True, tier="heuristic", confidence=1.0)

        # Tier 1: Heuristic
        if "heuristic" in self.tiers:
            verdict = self._heuristic(text)
            if not verdict.safe:
                return verdict

        # Tier 2: Classifier (async call needed — see detect_async)
        # Synchronous detect() only runs heuristic. Use detect_async() for
        # the full pipeline including the Modal classifier.

        return SecurityVerdict(safe=True, tier="heuristic", confidence=1.0)

    async def detect_async(self, text: str) -> SecurityVerdict:
        """Run all configured tiers including async classifier."""
        if not self.enabled or not text.strip():
            return SecurityVerdict(safe=True, tier="heuristic", confidence=1.0)

        # Tier 1: Heuristic
        if "heuristic" in self.tiers:
            verdict = self._heuristic(text)
            if not verdict.safe:
                return verdict

        # Tier 2: Classifier
        if "classifier" in self.tiers and self.classifier_url:
            verdict = await self._classify(text)
            if not verdict.safe:
                return verdict

        return SecurityVerdict(safe=True, tier=self.tiers[-1], confidence=1.0)

    def _heuristic(self, text: str) -> SecurityVerdict:
        """Tier 1: regex-based heuristic detection."""
        # Check base64-encoded payloads
        b64_verdict = self._check_base64(text)
        if b64_verdict is not None:
            return b64_verdict

        for name, pattern in _HEURISTIC_PATTERNS:
            if pattern.search(text):
                logger.warning("injection_detected", tier="heuristic", pattern=name)
                return SecurityVerdict(
                    safe=False,
                    tier="heuristic",
                    confidence=1.0,
                    matched_pattern=name,
                )

        return SecurityVerdict(safe=True, tier="heuristic", confidence=1.0)

    def _check_base64(self, text: str) -> SecurityVerdict | None:
        """Check for base64-encoded injection payloads."""
        b64_pattern = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
        for match in b64_pattern.finditer(text):
            try:
                decoded = base64.b64decode(match.group()).decode("utf-8", errors="ignore").lower()
                for name, pattern in _HEURISTIC_PATTERNS:
                    if pattern.search(decoded):
                        logger.warning(
                            "injection_detected",
                            tier="heuristic",
                            pattern="base64_injection",
                            decoded_match=name,
                        )
                        return SecurityVerdict(
                            safe=False,
                            tier="heuristic",
                            confidence=1.0,
                            matched_pattern="base64_injection",
                        )
            except Exception:
                continue
        return None

    async def _classify(self, text: str) -> SecurityVerdict:
        """Tier 2: DeBERTa classifier via Modal endpoint."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self.classifier_url,
                    json={"text": text},
                )
                resp.raise_for_status()
                data = resp.json()

            label = data.get("label", "SAFE")
            score = float(data.get("score", 0.0))

            is_injection = label == "INJECTION" and score > 0.5
            if is_injection:
                logger.warning("injection_detected", tier="classifier", score=score)
            return SecurityVerdict(
                safe=not is_injection,
                tier="classifier",
                confidence=score,
            )
        except Exception as exc:
            logger.error("classifier_error", error=str(exc))
            # Fail open: if classifier is unavailable, allow the request
            return SecurityVerdict(safe=True, tier="classifier", confidence=0.0)
