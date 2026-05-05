"""generate_kappa_table — joins predictions ⋈ labels by (item_id, dimension,
system_output_hash); computes per-row κ + bootstrap CI + abstain breakdown;
emits markdown table at docs/_generated/kappa_table.md.
"""

from __future__ import annotations

import glob as _glob
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import structlog

from agent_bench.evaluation.calibration.metrics import (
    bootstrap_ci,
    cohen_kappa,
    gwets_ac2,
)
from agent_bench.evaluation.judges.base import (
    ABSTAIN_REASON_OUT_OF_RANGE,
    ABSTAIN_REASON_PROVIDER_EXHAUSTED,
    ABSTAIN_REASON_SCHEMA_PARSE,
)

logger = structlog.get_logger()

ABSTAIN_THRESHOLD = 0.20  # strictly greater than fires the flag

# Per-dimension headline metric. Cohen's κ degenerates under the prevalence
# imbalance produced by the v1.1 strict-snippet groundedness rubric (1×score=1,
# ~25×score=0) and by the inherent skew on relevance (29×score=2, 1×score=1):
# both Po and Pe approach 1.0, the formula collapses to ~0/0, and the rendered
# κ reads as 0.000 even when raw agreement is >95%. Gwet's AC1 (gwets_ac2 with
# weights=None per metrics.py) uses mean marginals and stays informative under
# imbalance. Completeness has a more balanced gold (23×2, 5×1, 2×Unknown) so
# Cohen's κ is the conventional choice there. The metric per dim is rendered
# explicitly in the footer so a writeup reader sees the methodology choice.
# Type annotation prevents a mypy 1.20.x INTERNAL ERROR triggered by the
# tuple-unpack of `_DIM_METRIC.get(dim, default)` further down. Without it
# mypy fails to infer the metric_fn callable signature consistently across
# the dict literal and the fallback default, and crashes with no real
# user-facing type error to fix.
_MetricFn = Callable[[list, list], float]
_DIM_METRIC: dict[str, tuple[str, _MetricFn]] = {
    "groundedness": ("AC1", gwets_ac2),
    "relevance": ("AC1", gwets_ac2),
    "completeness": ("κ", cohen_kappa),
}

# Filename marker for jury / permute sidecar files. Any prediction file whose
# basename contains this token is per-member detail, not aggregate predictions,
# and is excluded from the κ table. Pinned here so a future extension change
# (jsonl → json) is caught at the contract site rather than at report time.
_SIDECAR_BASENAME_MARKER = "_members."


def _classify_abstain(reasoning: str) -> str:
    if reasoning.startswith(ABSTAIN_REASON_PROVIDER_EXHAUSTED):
        return "provider_exhausted"
    if reasoning.startswith(ABSTAIN_REASON_SCHEMA_PARSE):
        return "schema_parse"
    if reasoning.startswith(ABSTAIN_REASON_OUT_OF_RANGE):
        return "out_of_range"
    return "genuine"


def generate_kappa_table(
    *,
    predictions_glob: str,
    labels_path: str,
    output_path: str,
    strict: bool = False,
) -> None:
    """Aggregate predictions across rows + dimensions into one markdown table.

    On hash mismatch: ALWAYS raises (both modes), with first-item expected
    /actual hashes plus full mismatched-id list.
    On missing prediction or label: WARN+exclude in default mode; RAISE in strict.
    On undefined κ: render '—' with a footnote (both modes).
    On abstain rate > 20%: render κ + footnote with cause breakdown (both modes).
    """
    labels: list[dict] = []
    for line in Path(labels_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(json.loads(line))

    label_by_key: dict[tuple[str, str], dict] = {
        (label_rec["item_id"], label_rec["dimension"]): label_rec
        for label_rec in labels
    }

    pred_files = sorted(_glob.glob(predictions_glob))
    if not pred_files:
        raise ValueError(f"No prediction files matched: {predictions_glob}")

    rows: list[dict] = []
    for pf in pred_files:
        # Skip sidecars (per-member detail, not aggregate predictions).
        # Match the basename marker, not a specific extension, so a future
        # jsonl → json migration of jury._DEFAULT_SIDECAR_TEMPLATE doesn't
        # silently start contaminating the κ table.
        if _SIDECAR_BASENAME_MARKER in Path(pf).name:
            continue
        row_label = (
            Path(pf).stem.replace("calibration_v1_judge_", "")
        )
        preds = json.loads(Path(pf).read_text())

        # Hash-mismatch detection (always raises)
        mismatches: list[tuple[str, str, str]] = []
        for p in preds:
            key = (p["item_id"], p["dimension"])
            if key in label_by_key:
                expected = label_by_key[key]["system_output_hash"]
                actual = p["system_output_hash"]
                if expected != actual:
                    mismatches.append((p["item_id"], expected, actual))
        if mismatches:
            first_id, first_exp, first_act = mismatches[0]
            raise ValueError(
                f"Hash mismatch in {pf}: item {first_id!r} "
                f"label.system_output_hash={first_exp!r} but "
                f"prediction.system_output_hash={first_act!r}. "
                f"Full mismatched-id list ({len(mismatches)}): "
                f"{[m[0] for m in mismatches]}. "
                f"Labels are stale relative to predictions — regenerate one or "
                f"the other so hashes align."
            )

        preds_by_dim: dict[str, list[dict]] = defaultdict(list)
        for p in preds:
            preds_by_dim[p["dimension"]].append(p)

        labels_by_dim: dict[str, list[dict]] = defaultdict(list)
        for label_rec in labels:
            labels_by_dim[label_rec["dimension"]].append(label_rec)

        for dim in sorted(preds_by_dim.keys()):
            # Resolve dimension's headline metric once per dim, instead of
            # tuple-unpacking _DIM_METRIC.get(...) at each use site below.
            # The repeated unpack pattern triggered a mypy 1.19+ INTERNAL
            # ERROR; one resolution call here is also less code.
            metric_name, metric_fn = _DIM_METRIC.get(
                dim, ("κ", cohen_kappa)
            )

            preds_d = {p["item_id"]: p for p in preds_by_dim[dim]}
            labs_d = {
                label_rec["item_id"]: label_rec
                for label_rec in labels_by_dim.get(dim, [])
            }

            common = sorted(set(preds_d) & set(labs_d))
            missing_pred = sorted(set(labs_d) - set(preds_d))
            missing_lab = sorted(set(preds_d) - set(labs_d))
            if missing_pred or missing_lab:
                msg = (
                    f"row={row_label} dim={dim} "
                    f"missing_predictions={missing_pred} "
                    f"missing_labels={missing_lab}"
                )
                if strict:
                    raise ValueError(f"strict mode: missing items: {msg}")
                logger.warning("calibration_report_missing", message=msg)

            y_pred: list = []
            y_lab: list = []
            abstains = 0
            abstain_causes: dict[str, int] = {
                "provider_exhausted": 0,
                "schema_parse": 0,
                "out_of_range": 0,
                "genuine": 0,
            }
            for iid in common:
                p = preds_d[iid]
                label_rec = labs_d[iid]
                if p["score"] == "Unknown" or label_rec["score"] == "Unknown":
                    abstains += 1
                    if p["score"] == "Unknown":
                        abstain_causes[
                            _classify_abstain(p.get("reasoning", ""))
                        ] += 1
                    continue
                y_pred.append(int(p["score"]))
                y_lab.append(int(label_rec["score"]))

            n_eligible = len(y_pred)
            abstain_rate = abstains / max(len(common), 1)

            if n_eligible < 3:
                rows.append(
                    {
                        "row": row_label,
                        "dim": dim,
                        "metric": metric_name,
                        "kappa": None,
                        "ci_lo": None,
                        "ci_hi": None,
                        "n_eligible": n_eligible,
                        "abstains": abstains,
                        "abstain_rate": abstain_rate,
                        "abstain_causes": abstain_causes,
                        "footnote": (
                            f"{metric_name} undefined: insufficient "
                            f"agreement-eligible items (N={n_eligible})"
                        ),
                    }
                )
                continue

            try:
                kappa = metric_fn(y_lab, y_pred)
                point, lo, hi = bootstrap_ci(
                    y_lab, y_pred, metric_fn, n_iter=1000, seed=42
                )
            except (ValueError, ZeroDivisionError):
                rows.append(
                    {
                        "row": row_label,
                        "dim": dim,
                        "metric": metric_name,
                        "kappa": None,
                        "ci_lo": None,
                        "ci_hi": None,
                        "n_eligible": n_eligible,
                        "abstains": abstains,
                        "abstain_rate": abstain_rate,
                        "abstain_causes": abstain_causes,
                        "footnote": (
                            f"{metric_name} undefined: insufficient "
                            f"variance after exclusion"
                        ),
                    }
                )
                continue

            # Detect degenerate κ (perfectly constant labels → P_e=1 → kappa
            # was clamped to 1.0 in metrics.py, but with no observed
            # disagreement the result is statistically meaningless)
            if len(set(y_lab)) <= 1 and len(set(y_pred)) <= 1:
                rows.append(
                    {
                        "row": row_label,
                        "dim": dim,
                        "metric": metric_name,
                        "kappa": None,
                        "ci_lo": None,
                        "ci_hi": None,
                        "n_eligible": n_eligible,
                        "abstains": abstains,
                        "abstain_rate": abstain_rate,
                        "abstain_causes": abstain_causes,
                        "footnote": (
                            f"{metric_name} undefined: all labels and "
                            f"predictions in a single category (no variance "
                            f"to measure)"
                        ),
                    }
                )
                continue

            footnote = ""
            if abstain_rate > ABSTAIN_THRESHOLD:
                breakdown = ", ".join(
                    f"{int(100 * v / abstains)}% {k.replace('_', ' ')}"
                    for k, v in abstain_causes.items()
                    if v > 0
                )
                footnote = (
                    f"{metric_name} computed on N={n_eligible} of "
                    f"{len(common)} items; high abstain rate "
                    f"({100 * abstain_rate:.1f}% — breakdown: {breakdown}) "
                    f"suggests rubric ambiguity."
                )

            rows.append(
                {
                    "row": row_label,
                    "dim": dim,
                    "metric": metric_name,
                    "kappa": kappa,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_eligible": n_eligible,
                    "abstains": abstains,
                    "abstain_rate": abstain_rate,
                    "abstain_causes": abstain_causes,
                    "footnote": footnote,
                }
            )

    out = ["# κ ablation table — calibration v1\n"]
    out.append(
        "Headline metric per dimension: " + ", ".join(
            f"**{d} → {m}**" for d, (m, _) in _DIM_METRIC.items()
        ) + ". "
        "AC1 (Gwet 2008, unweighted) is used on dimensions whose v1.1 gold "
        "is prevalence-skewed enough to make Cohen's κ degenerate "
        "(groundedness 1×`1`/29×`0`, relevance 29×`2`/1×`1`); both metrics "
        "produce ≥0.95 raw agreement on those rows but Cohen's κ collapses "
        "to ≈0 because Pe approaches 1. Completeness uses Cohen's κ — its "
        "gold (23×`2`/5×`1`) is balanced enough for κ to behave normally."
    )
    out.append("")
    out.append("| Row | Dimension | Metric | Agreement (95% CI) | N | Abstain rate | Notes |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kappa"] is None:
            kcell = " — "
        else:
            kcell = f"{r['kappa']:.3f} ({r['ci_lo']:.3f}, {r['ci_hi']:.3f})"
        rate = f"{100 * r['abstain_rate']:.1f}%"
        out.append(
            f"| {r['row']} | {r['dim']} | {r['metric']} | {kcell} | "
            f"{r['n_eligible']} | {rate} | {r['footnote']} |"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(out) + "\n")
    logger.info("kappa_table_written", path=output_path, rows=len(rows))
