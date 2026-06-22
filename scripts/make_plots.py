"""Generate (and freshness-check) the README's statistics plots.

A figure is a measurement too: an unpinned plot is an unpinned number. Every
plot here is generated from the SAME pinned source the README cells read --
``docs/_generated/stats_report.md`` -- and carries a ``source-hash`` of the
exact values it drew. ``check`` recomputes that hash from the current report and
fails loudly if a committed SVG drifted, the same discipline
``scripts/check_readme_stats.py`` applies to the table cells.

Usage (matplotlib only needed for ``generate``, lazy-imported there so ``check``
and the tests run without the ``[plots]`` extra)::

    python scripts/make_plots.py generate   # rebuild SVGs from the report
    python scripts/make_plots.py check       # assert committed SVGs are fresh
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "_generated" / "stats_report.md"
PLOTS_DIR = ROOT / "docs" / "_generated" / "plots"

# Same KEY = value block the README markers pin to (scripts/check_readme_stats.py).
VALUE_RE = re.compile(r"^- ([a-z0-9_]+) = (.+)$", re.MULTILINE)
# Exactly 16 hex chars: source_hash() returns sha256()[:16], and bounding the
# length stops the match from running into a PNG tEXt chunk's trailing CRC byte
# when that byte happens to be an ASCII hex digit (seen on paired_diff).
HASH_RE = re.compile(r"source-hash:\s*([0-9a-f]{16})")

# (value-key fragment, display label, framework) in table-column order.
_CONFIGS = (
    ("custom_openai", "Custom OpenAI", "custom"),
    ("custom_anthropic", "Custom Anthropic", "custom"),
    ("langchain_openai", "LangChain OpenAI", "langchain"),
    ("langchain_anthropic", "LangChain Anthropic", "langchain"),
)
_METRICS = (("p_at_5", "P@5"), ("r_at_5", "R@5"))


def read_values(report_text: str) -> dict[str, str]:
    """Parse the report's README-values block into {KEY: value}."""
    return dict(VALUE_RE.findall(report_text))


def _ci(raw: str) -> tuple[float, float]:
    lo, hi = raw.strip().strip("[]").split(",")
    return float(lo), float(hi)


def forest_rows(values: dict[str, str], corpus: str = "fastapi") -> list[dict]:
    """The eight (config x metric) headline points with their 95 percent CIs."""
    rows = []
    for key, label, framework in _CONFIGS:
        for metric, mlabel in _METRICS:
            lo, hi = _ci(values[f"{corpus}_{key}_{metric}_ci"])
            rows.append(
                {
                    "config": key,
                    "label": label,
                    "framework": framework,
                    "metric": metric,
                    "metric_label": mlabel,
                    "mean": float(values[f"{corpus}_{key}_{metric}_mean"]),
                    "lo": lo,
                    "hi": hi,
                }
            )
    return rows


def significant_points(values: dict[str, str], corpus: str = "fastapi") -> set:
    """The (config, metric) points belonging to a 95 percent significant pair."""
    raw = values.get(f"{corpus}_significant_pairs_95", "none")
    points = set()
    for clause in raw.split(";"):
        m = re.match(r"\s*(\w+) vs (\w+) (p_at_5|r_at_5)\s*$", clause)
        if m:
            a, b, metric = m.groups()
            points.update({(a, metric), (b, metric)})
    return points


def forest_source(values: dict[str, str], corpus: str = "fastapi") -> dict:
    """The exact value set the forest plot draws -- hashed for provenance."""
    return {
        "rows": forest_rows(values, corpus),
        "significant": sorted(f"{c}:{m}" for c, m in significant_points(values, corpus)),
        "mde": values.get(f"{corpus}_mde_p_at_5_80"),
    }


# (custom-key, langchain-key, display label) for the framework difference plot.
_PAIRS = (
    ("custom_openai", "langchain_openai", "Custom OpenAI - LC OpenAI"),
    ("custom_anthropic", "langchain_anthropic", "Custom Anthropic - LC Anthropic"),
    ("custom_openai", "langchain_anthropic", "Custom OpenAI - LC Anthropic"),
    ("custom_anthropic", "langchain_openai", "Custom Anthropic - LC OpenAI"),
)


def significant_pairs(values: dict[str, str], corpus: str = "fastapi") -> set:
    """The (custom, langchain, metric) tuples flagged significant at 95 percent."""
    raw = values.get(f"{corpus}_significant_pairs_95", "none")
    out = set()
    for clause in raw.split(";"):
        m = re.match(r"\s*(\w+) vs (\w+) (p_at_5|r_at_5)\s*$", clause)
        if m:
            out.add((m.group(1), m.group(2), m.group(3)))
    return out


def paired_rows(values: dict[str, str], corpus: str = "fastapi") -> list[dict]:
    """One row per framework comparison: the mean paired difference with its
    nested 90/95 percent CIs and the pinned TOST verdict. The 90 percent CI reads
    against the +/-margin band (equivalence), the 95 percent against zero."""
    sig = significant_pairs(values, corpus)
    rows = []
    for custom, lc, label in _PAIRS:
        for metric, mlabel in _METRICS:
            stem = f"{corpus}_{custom}_vs_{lc}_{metric}"
            if f"{stem}_diff" not in values:
                continue
            rows.append(
                {
                    "label": label,
                    "metric": metric,
                    "metric_label": mlabel,
                    "diff": float(values[f"{stem}_diff"]),
                    "ci90": _ci(values[f"{stem}_ci90"]),
                    "ci95": _ci(values[f"{stem}_ci95"]),
                    "tost": values.get(f"{stem}_tost", ""),
                    "significant": (custom, lc, metric) in sig,
                    "same_provider": custom.rsplit("_", 1)[1] == lc.rsplit("_", 1)[1],
                }
            )
    return rows


def paired_source(values: dict[str, str], corpus: str = "fastapi") -> dict:
    """The exact value set the paired-difference plot draws -- hashed for provenance."""
    return {"rows": paired_rows(values, corpus)}


def source_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def embedded_hash(svg_text: str) -> str | None:
    m = HASH_RE.search(svg_text)
    return m.group(1) if m else None


# name -> source-set builder. One entry per committed figure. PNG (not SVG):
# GitHub's Markdown sanitizer does not reliably render relative-path SVGs inline,
# so the figure ships as PNG and the source-hash lives in a PNG tEXt chunk.
EXPECTED_PLOTS = {
    "forest_fastapi.png": lambda v: forest_source(v, "fastapi"),
    "paired_diff_fastapi.png": lambda v: paired_source(v, "fastapi"),
}


def check(report_text: str, plots_dir: Path) -> list[str]:
    """One failure string per missing or stale plot (empty == all fresh)."""
    values = read_values(report_text)
    failures = []
    for name, builder in EXPECTED_PLOTS.items():
        svg = plots_dir / name
        if not svg.exists():
            failures.append(f"{name} missing; run `make plots`")
            continue
        want = source_hash(builder(values))
        # latin-1 maps every byte, so this never raises on a binary PNG and keeps
        # the ASCII source-hash substring intact (a PNG tEXt chunk is plain text).
        got = embedded_hash(svg.read_bytes().decode("latin-1", "ignore"))
        if got != want:
            failures.append(
                f"{name} stale: embedded source-hash {got} != report {want}; run `make plots`"
            )
    return failures


def _save_with_hash(fig, out_path: Path, h: str) -> None:
    """Write the figure and embed ``source-hash:<h>`` -- an XML comment for SVG,
    a PNG tEXt chunk for raster. ``check`` reads it back to detect drift."""
    import matplotlib.pyplot as plt

    if out_path.suffix == ".svg":
        fig.savefig(out_path, metadata={"Date": None})
        plt.close(fig)
        svg = out_path.read_text().replace("</svg>", f"<!-- source-hash: {h} -->\n</svg>", 1)
        out_path.write_text(svg)
    else:  # raster (png): the hash rides in a PNG tEXt chunk via savefig metadata
        fig.savefig(out_path, dpi=200, metadata={"Description": f"source-hash:{h}"})
        plt.close(fig)


def _render_forest(values: dict[str, str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "agent-bench"  # deterministic element ids
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = forest_rows(values)
    sig = significant_points(values)
    colors = {"custom": "#2b6cb0", "langchain": "#dd6b20"}

    # P@5 group on top, R@5 below; four configs each in table order. y descends so
    # the first config sits highest in its group, with a one-row gap between groups.
    groups = [[r for r in rows if r["metric"] == metric] for metric, _ in _METRICS]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    y = float(len(rows) + 1)
    group_tops = []
    for group in groups:
        group_tops.append(y)
        for r in group:
            highlighted = (r["config"], r["metric"]) in sig
            fw = r["framework"]
            ax.errorbar(
                r["mean"],
                y,
                xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                fmt="o",
                color=colors[fw],
                ecolor=colors[fw],
                elinewidth=2,
                capsize=4,
                markersize=10 if highlighted else 7,
                markeredgecolor="#b7791f" if highlighted else "white",
                markeredgewidth=2.2 if highlighted else 0.8,
                zorder=3 if highlighted else 2,
            )
            ax.text(r["lo"] - 0.012, y, r["label"], ha="right", va="center", fontsize=8.5)
            y -= 1
        y -= 1  # gap between metric groups

    ax.set_yticks([])
    ax.set_xlim(0.40, 1.02)
    ax.set_xlabel("score (95% CI, cluster bootstrap)")
    for (_, mlabel), top in zip(_METRICS, group_tops):
        ax.text(0.40, top + 0.5, mlabel, fontsize=11, fontweight="bold", va="bottom")
    ax.set_title("FastAPI retrieval: framework comparison (overlapping CIs)", fontsize=11)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["custom"],
            markersize=8,
            label="custom",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["langchain"],
            markersize=8,
            label="LangChain",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#cbd5e0",
            markeredgecolor="#b7791f",
            markeredgewidth=2,
            markersize=8,
            label="only significant pair (95%)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_with_hash(fig, out_path, source_hash(forest_source(values)))


def _render_paired(values: dict[str, str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "agent-bench"
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = paired_rows(values)
    margin = 0.10
    sig_color, base_color, tie_color = "#b7791f", "#2b6cb0", "#718096"

    # group by metric (P@5 top, R@5 below); same-provider before cross within each
    def order(r: dict) -> tuple:
        return (0 if r["same_provider"] else 1, r["label"])

    groups = [
        (mlabel, sorted((r for r in rows if r["metric"] == metric), key=order))
        for metric, mlabel in _METRICS
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.axvspan(-margin, margin, color="#e2e8f0", zorder=0)  # equivalence band, for the 90% bar
    ax.axvline(0.0, color="#1a202c", lw=1.3, zorder=1)  # zero rule, for the 95% caps

    y = float(sum(len(g) for _, g in groups) + len(groups))
    group_tops = []
    for mlabel, group in groups:
        group_tops.append((mlabel, y))
        for r in group:
            color = sig_color if r["significant"] else base_color
            lo95, hi95 = r["ci95"]
            lo90, hi90 = r["ci90"]
            if lo95 == hi95 == 0.0:  # identical recall: a 0-width bar would read as a render bug
                ax.plot(0.0, y, marker="D", color=tie_color, markersize=8, zorder=6)
                ax.text(0.024, y, "exact tie (Δ=0)", va="center", fontsize=7.5, color=tie_color)
            else:
                ax.plot([lo95, hi95], [y, y], color=color, lw=1.3, zorder=3)  # 95% thin
                for x in (lo95, hi95):  # 95% caps
                    ax.plot([x, x], [y - 0.14, y + 0.14], color=color, lw=1.3, zorder=3)
                ax.plot([lo90, hi90], [y, y], color=color, lw=5.5, solid_capstyle="butt", zorder=4)
                ax.plot(
                    r["diff"],
                    y,
                    marker="o",
                    color=color,
                    markersize=9 if r["significant"] else 6,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=6,
                )
            ax.text(-0.215, y, r["label"], ha="right", va="center", fontsize=8)
            y -= 1
        y -= 1

    ax.set_yticks([])
    ax.set_xlim(-0.22, 0.40)
    # Headroom above the top group so the title clears the bold "P@5" label.
    ax.set_ylim(1.2, max(t for _, t in group_tops) + 1.6)
    ax.set_xlabel("paired difference: custom − LangChain  (per-question, cluster bootstrap)")
    for mlabel, top in group_tops:
        ax.text(-0.215, top + 0.5, mlabel, fontsize=11, fontweight="bold", va="bottom")
    ax.set_title(
        "Framework difference (paired): equivalence vs the ±0.10 band, significance vs zero",
        fontsize=10.5,
        pad=12,
    )

    handles = [
        Line2D([0], [0], color=base_color, lw=5.5, label="90% CI — equivalence (vs ±0.10 band)"),
        Line2D([0], [0], color=base_color, lw=1.3, label="95% CI — significance (vs zero)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=sig_color,
            markersize=8,
            label="significant pair (95%)",
        ),
        Patch(facecolor="#e2e8f0", label="±0.10 TOST margin"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    _save_with_hash(fig, out_path, source_hash(paired_source(values)))


def generate() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    values = read_values(REPORT.read_text())
    _render_forest(values, PLOTS_DIR / "forest_fastapi.png")
    _render_paired(values, PLOTS_DIR / "paired_diff_fastapi.png")
    print(f"wrote {len(EXPECTED_PLOTS)} plot(s) to {PLOTS_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "check"), nargs="?", default="generate")
    args = parser.parse_args()
    if args.mode == "generate":
        generate()
        return 0
    failures = check(REPORT.read_text(), PLOTS_DIR)
    for line in failures:
        print(f"FAIL: {line}")
    if not failures:
        print(f"OK: all {len(EXPECTED_PLOTS)} plot(s) fresh against the report")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
