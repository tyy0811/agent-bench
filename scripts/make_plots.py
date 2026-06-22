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
HASH_RE = re.compile(r"source-hash:\s*([0-9a-f]+)")

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


def source_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def embedded_hash(svg_text: str) -> str | None:
    m = HASH_RE.search(svg_text)
    return m.group(1) if m else None


# name -> source-set builder. One entry per committed figure. PNG (not SVG):
# GitHub's Markdown sanitizer does not reliably render relative-path SVGs inline,
# so the figure ships as PNG and the source-hash lives in a PNG tEXt chunk.
EXPECTED_PLOTS = {"forest_fastapi.png": lambda v: forest_source(v, "fastapi")}


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
    h = source_hash(forest_source(values))
    if out_path.suffix == ".svg":
        fig.savefig(out_path, metadata={"Date": None})
        plt.close(fig)
        svg = out_path.read_text().replace("</svg>", f"<!-- source-hash: {h} -->\n</svg>", 1)
        out_path.write_text(svg)
    else:  # raster (png): embed the hash as a PNG tEXt chunk via savefig metadata
        fig.savefig(out_path, dpi=200, metadata={"Description": f"source-hash:{h}"})
        plt.close(fig)


def generate() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    values = read_values(REPORT.read_text())
    _render_forest(values, PLOTS_DIR / "forest_fastapi.png")
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
