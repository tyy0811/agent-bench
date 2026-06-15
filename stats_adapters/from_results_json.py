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
    questions = raw["questions"] if is_nested else raw
    index: dict[str, GoldenQuestion] = {}
    for q in questions:
        if is_nested:
            cluster = q["question_type"]
        elif q["expected_sources"]:
            cluster = q["expected_sources"][0]
        else:
            cluster = "out_of_scope"
        index[q["id"]] = GoldenQuestion(
            category=q["category"],
            requires_calculator=q["requires_calculator"],
            cluster_id=cluster,
        )
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
    if CITATION_RE.search(rec.get("answer", "")):
        values["citation_acc"] = float(rec["citation_accuracy"])
    if golden_q.requires_calculator:
        values["calculator_correct"] = 1.0 if rec["calculator_used_correctly"] else 0.0
    return values


def rows_from_result(rec: dict, meta: RowMeta, golden: GoldenIndex) -> list[dict]:
    golden_q = golden.questions[rec["question_id"]]
    answer = rec.get("answer", "")
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
                "question_id": rec["question_id"],
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
