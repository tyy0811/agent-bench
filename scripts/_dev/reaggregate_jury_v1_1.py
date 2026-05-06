"""Plan B (v1.1 jury rescue): re-aggregate the existing 164 member-rows
in `results/calibration_v1_judge_jury_kappa_weighted_members.jsonl` with
corrected κ-derived weights, no new API spend.

Maps the resulting jury κ on completeness to the predefined outcome
criteria committed in DECISIONS.md ("v1.1 jury rescue" entry):
  - Outcome 1: jury κ ≥ Haiku-baseline + 0.05  → A+B sufficient
  - Outcome 2: jury κ within ±0.05 of Haiku   → soft exclusion via weighting
  - Outcome 3: jury κ < Haiku-baseline - 0.05 → escalate to per-dim exclusion (C)

Run:
    python scripts/_dev/reaggregate_jury_v1_1.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIDECAR = REPO / "results/calibration_v1_judge_jury_kappa_weighted_members.jsonl"
LABELS = REPO / "measurements/2026-05-04-judge-calibration-labels.jsonl"
HAIKU_BASELINE_COMPLETENESS_KAPPA = 0.416  # from kappa_table.md

# Mirror agent_bench.evaluation.variance.jury._discretize_mean
def _discretize_mean(mean: float, scale: str) -> int:
    if scale == "binary":
        return 1 if mean > 0.5 else 0
    floor = int(mean)
    frac = mean - floor
    return floor + 1 if frac > 0.5 else floor


def _load_labels(path: Path, dimension: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("dimension") != dimension or rec.get("abstained"):
            continue
        out[rec["system_output_hash"]] = int(rec["score"])
    return out


def _load_predictions_by_judge(
    path: Path, dimension: str
) -> dict[str, dict[str, int | str]]:
    """Return {judge_id: {hash: score}} for the dimension.

    The sidecar is append-only; if there are duplicate (judge, hash)
    pairs from re-runs, the last write wins (mirrors what generate-table
    sees from the JSON output file path which is overwritten per row).
    """
    by_judge: dict[str, dict[str, int | str]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec["judge_id"].endswith(f"_{dimension}"):
            continue
        by_judge[rec["judge_id"]][rec["system_output_hash"]] = rec["score"]
    return by_judge


def _kappa(y1: list[int], y2: list[int]) -> float:
    from agent_bench.evaluation.calibration.metrics import cohen_kappa
    return cohen_kappa(y1, y2)


def _per_judge_kappa(
    by_judge: dict[str, dict[str, int | str]], labels: dict[str, int]
) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for jid, preds in by_judge.items():
        y_lab: list[int] = []
        y_pred: list[int] = []
        for h, score in preds.items():
            if score == "Unknown":
                continue
            if h not in labels:
                continue
            y_lab.append(labels[h])
            y_pred.append(int(score))
        if not y_lab:
            continue
        out[jid] = (_kappa(y_lab, y_pred), len(y_lab))
    return out


def _load_full_member_rows(path: Path, dimension: str) -> list[dict]:
    """Return the most-recent record per (judge_id, system_output_hash) for
    the dimension. The sidecar is append-only; if there are duplicates from
    re-runs, the later record wins (mirrors how the JSON output file would
    reflect the last successful run)."""
    by_key: dict[tuple[str, str], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec["judge_id"].endswith(f"_{dimension}"):
            continue
        by_key[(rec["judge_id"], rec["system_output_hash"])] = rec
    return list(by_key.values())


def _aggregate_jury(
    by_judge: dict[str, dict[str, int | str]],
    labels: dict[str, int],
    weights: dict[str, float],
    scale: str,
) -> tuple[list[int], list[int], int]:
    """Strict quorum: any member abstain on an item → jury abstain (skipped).

    Returns (y_lab, y_pred, abstained_count) where each list element is
    one item that survived strict quorum.
    """
    judge_ids = list(by_judge.keys())
    # Common item set: hashes scored by every judge (any judge abstaining
    # on an item also drops it under strict quorum).
    all_hashes = set.intersection(*[set(d.keys()) for d in by_judge.values()])
    y_lab: list[int] = []
    y_pred: list[int] = []
    abstained = 0
    for h in sorted(all_hashes):
        scores = [by_judge[jid][h] for jid in judge_ids]
        if any(s == "Unknown" for s in scores):
            abstained += 1
            continue
        if h not in labels:
            continue
        int_scores = [int(s) for s in scores]
        wts = [weights[jid] for jid in judge_ids]
        weighted_sum = sum(s * w for s, w in zip(int_scores, wts))
        weight_total = sum(wts)
        if weight_total <= 0:
            abstained += 1
            continue
        agg = _discretize_mean(weighted_sum / weight_total, scale)
        y_lab.append(labels[h])
        y_pred.append(agg)
    return y_lab, y_pred, abstained


def _hash_to_item_id_map(labels_path: Path) -> dict[str, str]:
    """Recover hash → item_id from the labels file, since the sidecar
    JSONL was written before the v1.1 item_id backfill (which only
    touched the per-row JSON output files, not the sidecar)."""
    out: dict[str, str] = {}
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["system_output_hash"]] = rec["item_id"]
    return out


def _build_v1_1_jury_predictions(
    by_judge: dict[str, dict[str, int | str]],
    member_rows: list[dict],
    weights: dict[str, float],
    scale: str,
    dimension: str,
    hash_to_item: dict[str, str],
) -> list[dict]:
    """Per-item jury verdicts for the κ-table-format output. Pulls metadata
    (rubric_version, item_id) from member rows; aggregates score/cost/latency
    via the same rules as the production Jury class."""
    judge_ids = list(by_judge.keys())
    by_judge_hash_row = {
        (r["judge_id"], r["system_output_hash"]): r for r in member_rows
    }
    common_hashes = set.intersection(*[set(d.keys()) for d in by_judge.values()])
    out: list[dict] = []
    for h in sorted(common_hashes):
        scores = [by_judge[jid][h] for jid in judge_ids]
        member_meta = [by_judge_hash_row[(jid, h)] for jid in judge_ids]
        rubric_version = member_meta[0]["rubric_version"]
        item_id = member_meta[0].get("item_id") or hash_to_item.get(h)
        if item_id is None:
            # Sidecar + labels both lack mapping for this hash — drop,
            # since κ-table can't join without item_id.
            continue
        cost = sum(r.get("cost_usd", 0.0) for r in member_meta)
        latency = max(r.get("latency_ms", 0.0) for r in member_meta)

        if any(s == "Unknown" for s in scores):
            out.append({
                "item_id": item_id,
                "dimension": dimension,
                "reasoning": (
                    f"jury_below_quorum: 1+ member abstain (members="
                    f"{[s for s in scores]})"
                ),
                "evidence_quotes": [],
                "score": "Unknown",
                "judge_id": "jury_v1_1_kappa_weighted",
                "rubric_version": rubric_version,
                "prompt_seed": 0,
                "system_output_hash": h,
                "cost_usd": cost,
                "latency_ms": latency,
            })
            continue
        int_scores = [int(s) for s in scores]
        wts = [weights[jid] for jid in judge_ids]
        weighted_sum = sum(s * w for s, w in zip(int_scores, wts))
        weight_total = sum(wts)
        weighted_mean = weighted_sum / weight_total if weight_total > 0 else 0.0
        agg = _discretize_mean(weighted_mean, scale)
        out.append({
            "item_id": item_id,
            "dimension": dimension,
            "reasoning": (
                f"jury_kappa_weighted_v1_1: members={int_scores}, weights={wts}"
            ),
            "evidence_quotes": [],
            "score": agg,
            "judge_id": "jury_v1_1_kappa_weighted",
            "rubric_version": rubric_version,
            "prompt_seed": 0,
            "system_output_hash": h,
            "cost_usd": cost,
            "latency_ms": latency,
        })
    return out


def _classify_outcome(jury_k: float, baseline_k: float) -> str:
    delta = jury_k - baseline_k
    if delta >= 0.05:
        return f"OUTCOME 1 (Δ={delta:+.3f}, ≥+0.05) — A+B sufficient; writeup as 'weights bug masked aggregation'"
    if delta > -0.05:
        return f"OUTCOME 2 (Δ={delta:+.3f}, within ±0.05) — soft exclusion via weighting"
    return f"OUTCOME 3 (Δ={delta:+.3f}, <-0.05) — escalate to per-dim exclusion (C)"


def main(write_output: bool = False) -> None:
    print("=" * 78)
    print("v1.1 jury rescue — Plan B re-aggregation")
    print("=" * 78)

    all_predictions: list[dict] = []
    for dim, scale in [
        ("completeness", "three_point"),
        ("groundedness", "binary"),
        ("relevance", "three_point"),
    ]:
        print(f"\n--- dimension: {dim} (scale={scale}) ---")
        labels = _load_labels(LABELS, dim)
        by_judge = _load_predictions_by_judge(SIDECAR, dim)
        if not by_judge:
            print(f"  no predictions for {dim} in sidecar — skipping")
            continue

        # Per-judge κ → weight (negative κ clipped to 0)
        per_judge = _per_judge_kappa(by_judge, labels)
        print(f"  Gold labels (non-abstain): {len(labels)}")
        for jid, (k, n) in sorted(per_judge.items()):
            w = max(0.0, k)
            print(f"  per-judge κ: {jid}  κ={k:+.3f}  n={n}  → weight={w:.3f}")
        weights = {jid: max(0.0, k) for jid, (k, _) in per_judge.items()}

        # Jury aggregate with corrected weights
        y_lab, y_pred, abstained = _aggregate_jury(by_judge, labels, weights, scale)
        if len(y_lab) < 2:
            print(f"  insufficient data after strict-quorum filter (n={len(y_lab)})")
            continue
        jury_k = _kappa(y_lab, y_pred)
        # Raw agreement
        raw_agree = sum(1 for a, b in zip(y_lab, y_pred) if a == b) / len(y_lab)
        print(
            f"  JURY (corrected weights): κ={jury_k:+.3f}  "
            f"raw={raw_agree:.3f}  n={len(y_lab)}  abstained={abstained}"
        )
        if dim == "completeness":
            print(f"\n  Haiku-baseline completeness κ = {HAIKU_BASELINE_COMPLETENESS_KAPPA}")
            print(f"  → {_classify_outcome(jury_k, HAIKU_BASELINE_COMPLETENESS_KAPPA)}")

        if write_output:
            member_rows = _load_full_member_rows(SIDECAR, dim)
            hash_to_item = _hash_to_item_id_map(LABELS)
            all_predictions.extend(
                _build_v1_1_jury_predictions(
                    by_judge, member_rows, weights, scale, dim, hash_to_item
                )
            )

    if write_output:
        out_path = REPO / "results/calibration_v1_judge_jury_kappa_weighted_v1_1.json"
        out_path.write_text(json.dumps(all_predictions, indent=2) + "\n")
        print(f"\nwrote {len(all_predictions)} v1.1-jury predictions to {out_path}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO))
    main(write_output="--write-output" in sys.argv)
