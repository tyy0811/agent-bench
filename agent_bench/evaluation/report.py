"""Generate markdown benchmark report from evaluation results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from agent_bench.evaluation.harness import EvalResult


def generate_report(
    results: list[EvalResult],
    config_dict: dict | None = None,
    provider_name: str = "unknown",
    corpus_size: int = 0,
) -> str:
    """Generate a markdown benchmark report."""
    lines: list[str] = []
    lines.append("# Benchmark Results — Technical Documentation Q&A")
    lines.append("")
    lines.append(f"**Provider:** {provider_name} | **Corpus:** {corpus_size} markdown files")
    lines.append("")

    # --- Aggregate metrics ---
    lines.append("## Aggregate Metrics")
    lines.append("")

    positive = [r for r in results if r.category != "out_of_scope"]
    negative = [r for r in results if r.category == "out_of_scope"]
    calc_qs = [r for r in results if r.category == "calculation"]

    avg_p5 = _safe_avg([r.retrieval_precision for r in positive])
    avg_r5 = _safe_avg([r.retrieval_recall for r in positive])
    avg_khr = _safe_avg([r.keyword_hit_rate for r in positive])
    source_rate = sum(1 for r in positive if r.has_source_citation)
    avg_citation = _safe_avg([r.citation_accuracy for r in positive])
    refusal_rate = sum(1 for r in negative if r.grounded_refusal)
    calc_correct = sum(1 for r in calc_qs if r.calculator_used_correctly)
    latencies = sorted([r.latency_ms for r in results])
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    total_cost = sum(r.tokens_used.estimated_cost_usd for r in results)
    avg_cost = total_cost / max(len(results), 1)

    # Optional faithfulness
    faith_scores = [r.faithfulness for r in positive if r.faithfulness is not None]
    avg_faith = _safe_avg(faith_scores) if faith_scores else None

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Retrieval P@5 | {avg_p5:.2f} |")
    lines.append(f"| Retrieval R@5 | {avg_r5:.2f} |")
    lines.append(f"| Keyword Hit Rate | {avg_khr:.2f} |")
    lines.append(f"| Source Citation Rate | {source_rate}/{len(positive)} |")
    lines.append(f"| Citation Accuracy | {avg_citation:.2f} |")
    lines.append(f"| Grounded Refusal Rate | {refusal_rate}/{len(negative)} |")
    lines.append(f"| Calculator Accuracy | {calc_correct}/{len(calc_qs)} |")
    if avg_faith is not None:
        lines.append(f"| Answer Faithfulness (LLM) | {avg_faith:.2f} |")
    lines.append(f"| Latency p50 | {p50:,.0f} ms |")
    lines.append(f"| Latency p95 | {p95:,.0f} ms |")
    lines.append(f"| Cost per query | ${avg_cost:.4f} |")
    lines.append("")

    # --- By Category ---
    lines.append("## By Category")
    lines.append("")
    lines.append("| Category | Count | P@5 | R@5 | Keyword Hit | Refusal |")
    lines.append("|----------|-------|-----|-----|-------------|---------|")

    by_cat: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    for cat in ["retrieval", "calculation", "out_of_scope"]:
        cat_results = by_cat.get(cat, [])
        if not cat_results:
            continue
        count = len(cat_results)
        if cat == "out_of_scope":
            ref_count = sum(1 for r in cat_results if r.grounded_refusal)
            lines.append(f"| {cat} | {count} | n/a | n/a | n/a | {ref_count}/{count} |")
        else:
            cp5 = _safe_avg([r.retrieval_precision for r in cat_results])
            cr5 = _safe_avg([r.retrieval_recall for r in cat_results])
            ckhr = _safe_avg([r.keyword_hit_rate for r in cat_results])
            lines.append(f"| {cat} | {count} | {cp5:.2f} | {cr5:.2f} | {ckhr:.2f} | n/a |")
    lines.append("")

    # --- By Difficulty ---
    lines.append("## By Difficulty")
    lines.append("")
    lines.append("| Difficulty | Count | P@5 | R@5 | Keyword Hit |")
    lines.append("|-----------|-------|-----|-----|-------------|")

    by_diff: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_diff[r.difficulty].append(r)

    for diff in ["easy", "medium", "hard"]:
        diff_results = by_diff.get(diff, [])
        if not diff_results:
            continue
        pos_only = [r for r in diff_results if r.category != "out_of_scope"]
        if not pos_only:
            lines.append(f"| {diff} | {len(diff_results)} | n/a | n/a | n/a |")
            continue
        dp5 = _safe_avg([r.retrieval_precision for r in pos_only])
        dr5 = _safe_avg([r.retrieval_recall for r in pos_only])
        dkhr = _safe_avg([r.keyword_hit_rate for r in pos_only])
        lines.append(f"| {diff} | {len(diff_results)} | {dp5:.2f} | {dr5:.2f} | {dkhr:.2f} |")
    lines.append("")

    # --- Chunking strategy comparison ---
    lines.append("## Chunking Strategy Comparison")
    lines.append("")
    lines.append("| Strategy | Note |")
    lines.append("|----------|------|")
    lines.append("| Recursive (default) | Used for this benchmark run |")
    lines.append(
        "| Fixed-size | Available via `--chunk-strategy fixed` in ingest. "
        "Re-run evaluation to compare. |"
    )
    lines.append("")
    lines.append(
        "_To generate a comparison, run `make ingest` with each strategy "
        "and `make evaluate-fast` for each, then compare the results JSON files._"
    )
    lines.append("")

    # --- Failure analysis (3 worst by retrieval precision) ---
    lines.append("## Failure Analysis (3 worst queries)")
    lines.append("")

    scorable = [r for r in results if r.category != "out_of_scope"]
    worst = sorted(scorable, key=lambda r: r.retrieval_precision)[:3]
    for r in worst:
        lines.append(f'**{r.question_id}: "{r.question}"**')
        lines.append(f"- Retrieval P@5: {r.retrieval_precision:.2f}")
        lines.append(f"- Retrieval R@5: {r.retrieval_recall:.2f}")
        lines.append(f"- Keyword Hit Rate: {r.keyword_hit_rate:.2f}")
        lines.append(f"- Retrieved: {r.retrieved_sources[:3]}")
        if r.retrieval_precision == 0.0 and r.keyword_hit_rate > 0.5:
            lines.append(
                "- Root cause: MockProvider returned canned answer — "
                "retrieval worked but answer text doesn't match expected sources"
            )
        elif r.retrieval_precision == 0.0:
            lines.append(
                "- Root cause: MockProvider canned response does not target "
                "this question's expected sources"
            )
        else:
            lines.append("- Root cause: _(manual analysis needed for real provider runs)_")
        lines.append("")

    # --- Per-question detail ---
    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| ID | Cat | Diff | P@5 | R@5 | KHR | Citation | Refusal | Calc |")
    lines.append("|----|-----|------|-----|-----|-----|----------|---------|------|")

    for r in results:
        p5 = f"{r.retrieval_precision:.2f}" if r.category != "out_of_scope" else "n/a"
        r5 = f"{r.retrieval_recall:.2f}" if r.category != "out_of_scope" else "n/a"
        khr = f"{r.keyword_hit_rate:.2f}"
        cit = f"{r.citation_accuracy:.2f}" if r.category != "out_of_scope" else "n/a"
        ref = "PASS" if r.grounded_refusal else "FAIL"
        calc = "PASS" if r.calculator_used_correctly else "FAIL"
        lines.append(
            f"| {r.question_id} | {r.category} | {r.difficulty} "
            f"| {p5} | {r5} | {khr} | {cit} | {ref} | {calc} |"
        )
    lines.append("")

    # --- Config snapshot ---
    if config_dict:
        lines.append("## Configuration Snapshot")
        lines.append("")
        lines.append("```yaml")
        lines.append(yaml.dump(config_dict, default_flow_style=False).strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def save_report(report: str, path: str | Path) -> None:
    """Write report to file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report + "\n")


def _safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * p / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]
