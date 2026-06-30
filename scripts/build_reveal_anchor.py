"""Generate agent_bench/serving/static/reveal_anchor.json from committed sources.

The dashboard reveal renders ONLY from this JSON; no stat number lives in the
page markup. Three blocks, three sources, three provenance tiers:
  - collapse: campaign bootstrap spans  <- docs/_generated/stats_report.md
  - cost:     single-run cost row        <- results/comparison_custom_vs_langchain.md
  - floor:    single-run citation row    <- docs/provider_comparison.md

Regenerate with one command:  python scripts/build_reveal_anchor.py
tests/test_reveal_anchor.py asserts the committed JSON equals a fresh build.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATS_REPORT = ROOT / "docs" / "_generated" / "stats_report.md"
COST_SOURCE = ROOT / "results" / "comparison_custom_vs_langchain.md"
FLOOR_SOURCE = ROOT / "docs" / "provider_comparison.md"
ANCHOR_OUT = ROOT / "agent_bench" / "serving" / "static" / "reveal_anchor.json"

# Mirrors scripts/check_readme_stats.py REPORT_LINE. Kept local (not imported)
# so this script is not pulled into a sibling's mypy scope.
_REPORT_LINE = re.compile(r"^- ([a-z0-9_]+) = (.+)$", re.MULTILINE)


def _spans(report_text: str) -> dict[str, str]:
    return dict(_REPORT_LINE.findall(report_text))


def _num(raw: str) -> float:
    return float(raw.strip().replace("+", ""))


def _pair(raw: str) -> list[float]:
    inner = raw.strip().strip("[]")
    return [float(p.replace("+", "").strip()) for p in inner.split(",")]


def _collapse(g: dict[str, str]) -> dict[str, Any]:
    base = "fastapi_custom_anthropic_vs_langchain_anthropic"
    return {
        "corpus": "FastAPI",
        "metric": "precision@5",
        "model_label": "claude-haiku-4-5",
        "a": {
            "label": "Custom · Anthropic",
            "p_at_5": _num(g["fastapi_custom_anthropic_p_at_5_mean"]),
            "ci": _pair(g["fastapi_custom_anthropic_p_at_5_ci"]),
        },
        "b": {
            "label": "LangChain · Anthropic",
            "p_at_5": _num(g["fastapi_langchain_anthropic_p_at_5_mean"]),
            "ci": _pair(g["fastapi_langchain_anthropic_p_at_5_ci"]),
        },
        "paired_diff": _num(g[f"{base}_p_at_5_diff"]),
        "paired_ci90": _pair(g[f"{base}_p_at_5_ci90"]),
        "tost": g[f"{base}_p_at_5_tost"].strip(),
        "tost_support": _num(g[f"{base}_p_at_5_support"]),
        "r_at_5_diff": _num(g[f"{base}_r_at_5_diff"]),
        "source": "docs/_generated/stats_report.md",
        "provenance": "campaign-bootstrap-ci",
    }


def _cost(text: str) -> dict[str, Any]:
    # The Anthropic section's row: "| Cost per query | **$0.0007** | $0.0046 | ... |"
    section = text.split("## Anthropic", 1)[1]
    row = next(
        ln for ln in section.splitlines() if ln.strip().startswith("| Cost per query")
    )
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    a = float(cells[1].replace("*", "").replace("$", ""))
    b = float(cells[2].replace("*", "").replace("$", ""))
    return {
        "a_per_query": a,
        "b_per_query": b,
        "ratio": f"{round(b / a, 1)}x",
        "source": "results/comparison_custom_vs_langchain.md",
        "provenance": "single-run",
    }


def _floor(text: str) -> dict[str, Any]:
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    header = next(ln for ln in rows if "Citation Acc" in ln)
    cols = [c.strip() for c in header.strip().strip("|").split("|")]
    ci_idx = cols.index("Citation Acc")
    api_vals: set[str] = set()
    self_val: str | None = None
    self_model: str | None = None
    for ln in rows:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) <= ci_idx:
            continue
        provider = cells[0]
        if "(API)" in provider:
            api_vals.add(cells[ci_idx])
        elif "Self-hosted" in provider:
            self_val = cells[ci_idx]
            self_model = cells[1]
    assert len(api_vals) == 1, f"API citation accuracies disagree: {api_vals}"
    assert self_val is not None and self_model is not None, "self-hosted row not found"
    return {
        "api_citation": float(api_vals.pop()),
        "self_hosted_citation": float(self_val),
        "self_hosted_model": self_model,
        "settings": "iterations=1, top_k=3, 8K context",
        "source": "docs/provider_comparison.md",
        "provenance": "single-run-citation",
    }


def build_anchor() -> dict[str, Any]:
    return {
        "collapse": _collapse(_spans(STATS_REPORT.read_text())),
        "cost": _cost(COST_SOURCE.read_text()),
        "floor": _floor(FLOOR_SOURCE.read_text()),
    }


def main() -> int:
    ANCHOR_OUT.write_text(json.dumps(build_anchor(), indent=2) + "\n")
    print(f"wrote {ANCHOR_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
