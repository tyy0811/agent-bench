"""Security type definitions shared across security modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SecurityVerdict:
    """Result of injection detection."""
    safe: bool
    tier: str  # "heuristic" | "classifier"
    confidence: float
    matched_pattern: str | None = None


@dataclass
class OutputVerdict:
    """Result of output validation."""
    passed: bool
    violations: list[str] = field(default_factory=list)
    action: str = "pass"  # "pass" | "redact" | "block"
