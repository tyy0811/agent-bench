"""Convert agent-bench results JSON into validated long-format CSV tables.

Boundary layer: the only package allowed to import agent_bench (guardrail 1).
Emission rules, cluster assignment, and legacy provenance per
docs/plans/2026-06-11-stats-layer-v3.1-design.md sections 2.1, 3.4, and 4.
"""

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agent_bench.evaluation.metrics import is_refusal
from stats import schema

# Mirrors the pattern inside agent_bench.evaluation.metrics.citation_accuracy.
# Used only to decide whether an answer exercised the citation metric at all:
# citation_acc 1.0 on a citation-free answer is vacuous and never enters the
# table as data (design spec section 3.4).
CITATION_RE = re.compile(r"\[source:\s*(.+?)\]")


@dataclass(frozen=True)
class RowMeta:
    run_id: str
    timestamp: str
    config_id: str
    code_version: str
    dataset_version: str
    epoch: int


@dataclass(frozen=True)
class GoldenQuestion:
    category: str
    requires_calculator: bool
    cluster_id: str


@dataclass(frozen=True)
class GoldenIndex:
    questions: dict[str, GoldenQuestion]


def load_golden(path: Path) -> GoldenIndex:
    raw = json.loads(path.read_text())
    is_nested = isinstance(raw, dict)
    if is_nested and "questions" not in raw:
        raise ValueError(f"nested golden {path} has no 'questions' key")
    questions = raw["questions"] if is_nested else raw
    index: dict[str, GoldenQuestion] = {}
    for q in questions:
        try:
            if is_nested:
                cluster = q["question_type"]
            elif q["expected_sources"]:
                cluster = q["expected_sources"][0]
            else:
                cluster = "out_of_scope"
            if not isinstance(cluster, str) or not cluster:
                raise ValueError(
                    f"golden {path}: question {q.get('id', '?')!r} has an empty or "
                    "null cluster source (question_type / expected_sources)"
                )
            index[q["id"]] = GoldenQuestion(
                category=q["category"],
                requires_calculator=q["requires_calculator"],
                cluster_id=cluster,
            )
        except KeyError as e:
            raise ValueError(
                f"golden {path}: question {q.get('id', '?')!r} missing field {e}"
            ) from e
    return GoldenIndex(questions=index)


def _metric_values(rec: dict, golden_q: GoldenQuestion) -> dict[str, float]:
    if golden_q.category == "out_of_scope":
        return {"refusal_correct": 1.0 if rec["grounded_refusal"] else 0.0}
    values = {
        "p_at_5": float(rec["retrieval_precision"]),
        "r_at_5": float(rec["retrieval_recall"]),
        "khr": float(rec["keyword_hit_rate"]),
    }
    # citation_acc only when the answer actually cited something: vacuous 1.0
    # on citation-free answers never enters the table (spec section 3.4). The
    # rule-of-three inclusion rule (spec section 2.3) then falls out of the
    # table itself: a question enters n if any epoch emitted a citation_acc row.
    if CITATION_RE.search(rec.get("answer") or ""):
        values["citation_acc"] = float(rec["citation_accuracy"])
    if golden_q.requires_calculator:
        values["calculator_correct"] = 1.0 if rec["calculator_used_correctly"] else 0.0
    return values


def rows_from_result(rec: dict, meta: RowMeta, golden: GoldenIndex) -> list[dict]:
    qid = rec.get("question_id")
    if qid is None:
        raise ValueError("results record missing 'question_id'")
    if qid not in golden.questions:
        raise ValueError(
            f"question_id {qid!r} not in golden index ({len(golden.questions)} "
            "questions); check that the results file and --golden correspond"
        )
    golden_q = golden.questions[qid]
    answer = rec.get("answer") or ""  # JSON null or missing -> empty (no answer)
    refused: bool | None = is_refusal(answer) if answer else None
    rows = []
    for metric, score in _metric_values(rec, golden_q).items():
        rows.append(
            {
                "run_id": meta.run_id,
                "timestamp": meta.timestamp,
                "config_id": meta.config_id,
                "code_version": meta.code_version,
                "dataset_version": meta.dataset_version,
                "question_id": qid,
                "cluster_id": golden_q.cluster_id,
                "epoch": meta.epoch,
                "metric": metric,
                "score": score,
                "latency_ms": float(rec["latency_ms"]),
                "cost_usd": float(rec["tokens_used"]["estimated_cost_usd"]),
                "refused": refused,
            }
        )
    return rows


def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _git_commit_date(path: Path) -> str:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    iso = proc.stdout.strip()
    # Untracked files (tmp fixtures in tests) have no commit date; epoch zero
    # keeps the row valid and visibly artificial.
    return iso if iso else "1970-01-01T00:00:00+00:00"


def convert_legacy_file(
    src: Path,
    golden_path: Path,
    config_id: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Convert one pre-v3.1 results file. Spec section 4 provenance synthesis."""
    golden = load_golden(golden_path)
    meta = RowMeta(
        run_id=f"legacy-{content_hash(src)}",
        timestamp=_git_commit_date(src),
        config_id=config_id,
        code_version="unknown",
        dataset_version="unknown",
        epoch=1,
    )
    records = json.loads(src.read_text())
    rows: list[dict] = []
    for i, rec in enumerate(records):
        try:
            rows.extend(rows_from_result(rec, meta, golden))
        except KeyError as e:
            raise ValueError(
                f"{src}: record {i} (question_id={rec.get('question_id', '?')!r}) missing field {e}"
            ) from e
    if not rows:
        raise ValueError(f"{src}: produced no rows (empty file or no records?)")
    df = pd.DataFrame(rows)
    df["refused"] = df["refused"].astype("boolean")
    schema.validate_table(df)
    legacy_dir = out_dir / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(legacy_dir / f"{src.stem}.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="results JSON path, or envelope dir in WP2")
    parser.add_argument("--golden", required=True, help="golden dataset JSON path")
    parser.add_argument(
        "--config-id", default=None, help="explicit config_id; required with --legacy"
    )
    parser.add_argument("--out-dir", default="results/long")
    parser.add_argument(
        "--legacy", action="store_true", help="pre-v3.1 file: synthesize provenance"
    )
    args = parser.parse_args()
    if not args.legacy:
        raise SystemExit("epoch-envelope inputs arrive in WP2; use --legacy for existing files")
    if args.config_id is None:
        parser.error("--config-id is required with --legacy; no guessing (spec section 4)")
    df = convert_legacy_file(
        Path(args.input), Path(args.golden), args.config_id, Path(args.out_dir)
    )
    print(f"wrote {len(df)} rows for {args.input}")


if __name__ == "__main__":
    main()
