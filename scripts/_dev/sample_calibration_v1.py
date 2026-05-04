"""One-shot stratified sampler for calibration_v1.json. Run once; output
is committed to agent_bench/evaluation/datasets/calibration_v1.json.

The stratification target is in docs/plans/2026-05-04-judge-layer-v1-design.md
under Calibration Methodology > Stratified sampling.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FASTAPI_PATH = REPO / "agent_bench/evaluation/datasets/tech_docs_golden.json"
K8S_PATH = REPO / "agent_bench/evaluation/datasets/k8s_golden.json"
OUTPUT = REPO / "agent_bench/evaluation/datasets/calibration_v1.json"

SEED = 20260504  # date-derived; deterministic across runs

FASTAPI_TARGETS = {"retrieval": 5, "calculation": 1, "out_of_scope": 2}
K8S_TARGETS = {
    "simple": 4,
    "simple_w_condition": 3,
    "comparison": 3,
    "multi_hop": 4,
    "false_premise": 3,
    "set": 1,
}
SPARE_TOTAL = 4


def main() -> None:
    rng = random.Random(SEED)

    fastapi = json.loads(FASTAPI_PATH.read_text())
    k8s = json.loads(K8S_PATH.read_text())["questions"]

    selected: list[dict] = []

    by_cat: dict[str, list[dict]] = {}
    for q in fastapi:
        by_cat.setdefault(q["category"], []).append(q)
    for cat, n in FASTAPI_TARGETS.items():
        pool = by_cat.get(cat, [])
        if len(pool) < n:
            raise SystemExit(f"FastAPI {cat}: have {len(pool)}, need {n}")
        sample = rng.sample(pool, n)
        for q in sample:
            selected.append({"id": q["id"], "corpus": "fastapi", "stratum": cat})

    by_qt: dict[str, list[dict]] = {}
    for q in k8s:
        by_qt.setdefault(q.get("question_type", "?"), []).append(q)
    for qt, n in K8S_TARGETS.items():
        pool = by_qt.get(qt, [])
        if len(pool) < n:
            raise SystemExit(f"K8s {qt}: have {len(pool)}, need {n}")
        sample = rng.sample(pool, n)
        for q in sample:
            selected.append({"id": q["id"], "corpus": "k8s", "stratum": qt})

    # Spare slots — fill from highest-variance K8s strata. Original target
    # was simple_w_condition + multi_hop; expanded to include comparison and
    # false_premise because the K8s golden set has only 4 simple_w_condition
    # and 6 multi_hop items, of which Targets already consumed 7, leaving
    # only 3 in the original pool. Adding comparison/false_premise gives
    # enough headroom for 4 spares.
    selected_ids = {s["id"] for s in selected}
    spare_pool: list[dict] = [
        q
        for q in k8s
        if q.get("question_type")
        in ("simple_w_condition", "multi_hop", "comparison", "false_premise")
        and q["id"] not in selected_ids
    ]
    if len(spare_pool) < SPARE_TOTAL:
        raise SystemExit(
            f"Spare pool exhausted: have {len(spare_pool)}, need {SPARE_TOTAL}"
        )
    spare = rng.sample(spare_pool, SPARE_TOTAL)
    for q in spare:
        selected.append(
            {
                "id": q["id"],
                "corpus": "k8s",
                "stratum": f"spare_{q['question_type']}",
            }
        )

    if len(selected) != 30:
        raise SystemExit(f"Expected 30 items; got {len(selected)}")

    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()

    out = {
        "version": "v1",
        "system_config_git_sha": sha,
        "sample_seed": SEED,
        "notes": (
            "30-item stratified calibration set per the design doc. "
            "Spare slots filled from K8s simple_w_condition and multi_hop "
            "(typically highest-variance R@5 strata)."
        ),
        "items": sorted(selected, key=lambda s: (s["corpus"], s["stratum"], s["id"])),
    }
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUTPUT} with {len(selected)} items; git_sha={sha[:12]}")


if __name__ == "__main__":
    main()
