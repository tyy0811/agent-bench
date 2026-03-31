"""PII detection and redaction for retrieved context and generated output.

Regex-based detection for high-risk PII types (EMAIL, PHONE, SSN, CREDIT_CARD,
IP_ADDRESS). Optional spaCy NER for PERSON/ORG entities (off by default).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

# --- Regex patterns ---

_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "PHONE": re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}

# Order matters: SSN before PHONE (SSN is more specific, avoids partial matches)
_PATTERN_ORDER = ["SSN", "CREDIT_CARD", "EMAIL", "IP_ADDRESS", "PHONE"]


@dataclass
class RedactionResult:
    """Result of a redaction pass."""
    text: str
    redactions_count: int = 0
    types_found: list[str] = field(default_factory=list)


class PIIRedactor:
    """Detect and redact PII using regex patterns and optional NER."""

    def __init__(
        self,
        redact_patterns: list[str] | None = None,
        mode: str = "redact",
        use_ner: bool = False,
        ner_entities: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.active_patterns: list[tuple[str, re.Pattern]] = []

        if redact_patterns is None:
            redact_patterns = list(_PATTERNS.keys())

        for name in _PATTERN_ORDER:
            if name in redact_patterns and name in _PATTERNS:
                self.active_patterns.append((name, _PATTERNS[name]))

        # Optional NER
        self.use_ner = False
        self.ner_entities = ner_entities or ["PERSON"]
        self._nlp = None
        if use_ner:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                self.use_ner = True
            except ImportError:
                logger.warning("pii.use_ner=true but spaCy not installed, falling back to regex-only")
            except OSError:
                logger.warning("pii.use_ner=true but en_core_web_sm not found, falling back to regex-only")

    def redact(self, text: str) -> RedactionResult:
        """Detect and optionally redact PII in the given text."""
        if self.mode == "passthrough":
            return RedactionResult(text=text)

        # Collect all matches: (start, end, type, value)
        matches: list[tuple[int, int, str, str]] = []

        for name, pattern in self.active_patterns:
            for m in pattern.finditer(text):
                matches.append((m.start(), m.end(), name, m.group()))

        # Optional NER matches
        if self.use_ner and self._nlp is not None:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ in self.ner_entities:
                    matches.append((ent.start_char, ent.end_char, ent.label_, ent.text))

        if not matches:
            return RedactionResult(text=text)

        # Deduplicate overlapping spans: keep longest match
        matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
        filtered: list[tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, pii_type, value in matches:
            if start >= last_end:
                filtered.append((start, end, pii_type, value))
                last_end = end

        types_found = list(dict.fromkeys(m[2] for m in filtered))

        if self.mode == "detect_only":
            return RedactionResult(
                text=text,
                redactions_count=len(filtered),
                types_found=types_found,
            )

        # Redact mode: replace with deterministic placeholders
        # Same value -> same placeholder within one call
        placeholder_map: dict[str, str] = {}
        type_counters: dict[str, int] = {}

        result = text
        offset = 0
        for start, end, pii_type, value in filtered:
            key = f"{pii_type}:{value}"
            if key not in placeholder_map:
                type_counters[pii_type] = type_counters.get(pii_type, 0) + 1
                placeholder_map[key] = f"[{pii_type}_{type_counters[pii_type]}]"

            placeholder = placeholder_map[key]
            result = result[:start + offset] + placeholder + result[end + offset:]
            offset += len(placeholder) - (end - start)

        return RedactionResult(
            text=result,
            redactions_count=len(filtered),
            types_found=types_found,
        )
