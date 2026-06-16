"""Render docs/_generated/stats_report.md from long-format tables.

Pure function of its inputs: no wall clock anywhere (byte-stable golden
tests); identity comes from input-table content hashes and seeds. The three
degradation branches (design spec section 7) are code here, not prose:
divergence caution, zero-failure phrasing, TOST pass and fail wordings.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from stats.cluster import DIVERGENCE_RATIO, cluster_bootstrap, primary_is_clustered
from stats.equivalence import tost_paired
from stats.intervals import clopper_pearson, rule_of_three, zero_failure_upper
from stats.power import mde, mde_normal_approx
from stats.variance import decompose

HEADLINE_METRICS = ("p_at_5", "r_at_5")
DEFAULT_SEED = 20260611


def _question_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df["metric"] == metric]
    return sub.groupby(["config_id", "question_id", "cluster_id"], as_index=False).agg(
        {"score": "mean"}
    )


def _headline_section(df: pd.DataFrame, corpus: str, seed: int) -> list[str]:
    lines = [f"## Headline intervals: {corpus}", ""]
    lines.append(
        "| config | metric | mean | 95 percent interval (primary) "
        "| naive SE | clustered SE | n_clusters | design effect |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    cautions = []
    for config in sorted(df["config_id"].unique()):
        for metric in HEADLINE_METRICS:
            qm = _question_means(df[df["config_id"] == config], metric)
            if qm.empty:
                continue
            res = cluster_bootstrap(qm["score"].to_numpy(), qm["cluster_id"].to_numpy(), seed=seed)
            clustered_primary = primary_is_clustered(res.n_clusters)
            primary_se = res.clustered_se if clustered_primary else res.naive_se
            label = "clustered" if clustered_primary else "question-level"
            lo, hi = res.mean - 1.96 * primary_se, res.mean + 1.96 * primary_se
            lines.append(
                f"| {config} | {metric} | {res.mean:.3f} | [{lo:.3f}, {hi:.3f}] ({label}) "
                f"| {res.naive_se:.4f} | {res.clustered_se:.4f} | n_clusters={res.n_clusters} "
                f"| design effect={res.design_effect:.2f} |"
            )
            if not clustered_primary and res.clustered_se > DIVERGENCE_RATIO * res.naive_se:
                cautions.append(
                    f"correlation-sensitivity caution: {corpus} {config} {metric}: clustered SE "
                    f"{res.clustered_se:.4f} exceeds {DIVERGENCE_RATIO}x the question-level SE "
                    f"{res.naive_se:.4f}; the question-level headline interval "
                    f"likely understates uncertainty."
                )
    lines.append("")
    lines.extend(cautions)
    if cautions:
        lines.append("")
    return lines


def _citation_section(df: pd.DataFrame, corpus: str) -> list[str]:
    lines = [f"## Citation accuracy zero-failure bound: {corpus}", ""]
    for config in sorted(df["config_id"].unique()):
        sub = df[df["config_id"] == config]
        # Inclusion rule, spec section 2.3: a question enters n if any epoch
        # emitted a citation_acc row (the adapter omits vacuous citation-free
        # rows, so the table itself encodes the rule). Exclusions counted
        # against the in-scope universe, which always emits p_at_5.
        in_scope = sub.loc[sub["metric"] == "p_at_5", "question_id"].nunique()
        cit = sub[sub["metric"] == "citation_acc"]
        if cit.empty:
            lines.append(f"- {config}: no answers contained citations; bound not computable.")
            continue
        k = int(sub.groupby("question_id")["epoch"].nunique().max())
        per_q_min = cit.groupby("question_id")["score"].min()
        n = len(per_q_min)
        excluded = in_scope - n
        all_epochs = int((cit.groupby("question_id")["epoch"].nunique() == k).sum())
        failures = int((per_q_min < 1.0).sum())
        bookkeeping = (
            f"included n={n}, excluded {excluded} citation-free questions; "
            f"cited in all epochs: {all_epochs}, in some epochs: {n - all_epochs}"
        )
        if failures == 0:
            lines.append(
                f"- {config}: 0 failures in {n} included questions ({bookkeeping}). "
                f"Exact Clopper-Pearson 95 percent upper bound on the per-question failure rate: "
                f"{zero_failure_upper(n):.3f} (rule of three approximation "
                f"3/n = {rule_of_three(n):.3f})."
            )
        else:
            lo, hi = clopper_pearson(failures, n)
            lines.append(
                f"- {config}: {failures} of {n} included questions showed a citation failure "
                f"({bookkeeping}). Clopper-Pearson 95 percent interval on the failure rate: "
                f"[{lo:.3f}, {hi:.3f}]."
            )
    lines.append("")
    return lines


def _equivalence_section(df: pd.DataFrame, corpus: str, seed: int) -> list[str]:
    configs = sorted(df["config_id"].unique())
    customs = [c for c in configs if c.startswith("custom")]
    langchains = [c for c in configs if c.startswith("langchain")]
    lines = [f"## Framework equivalence (TOST): {corpus}", ""]
    for custom in customs:
        for lc in langchains:
            for metric in HEADLINE_METRICS:
                a = _question_means(df[df["config_id"] == custom], metric).set_index("question_id")
                b = _question_means(df[df["config_id"] == lc], metric).set_index("question_id")
                shared = a.index.intersection(b.index)
                if shared.empty:
                    continue
                diffs = (a.loc[shared, "score"] - b.loc[shared, "score"]).to_numpy()
                clusters = a.loc[shared, "cluster_id"].to_numpy()
                use_clusters = primary_is_clustered(len(np.unique(clusters)))
                res = tost_paired(diffs, clusters=clusters if use_clusters else None, seed=seed)
                if res.equivalent:
                    verdict = (
                        f"equivalent within plus or minus {res.margin:.2f}; "
                        f"the data support equivalence down to plus or minus "
                        f"{res.support_margin:.3f}"
                    )
                else:
                    verdict = (
                        f"equivalence not established at plus or minus {res.margin:.2f}; "
                        f"the data support only plus or minus {res.support_margin:.3f}"
                    )
                lines.append(
                    f"- {custom} vs {lc}, {metric}: mean diff {diffs.mean():+.3f}, "
                    f"90 percent CI [{res.ci_low:+.3f}, {res.ci_high:+.3f}], "
                    f"n={res.n_units}: {verdict}."
                )
    lines.append("")
    return lines


def _variance_power_section(df: pd.DataFrame, corpus: str, seed: int) -> list[str]:
    lines = [f"## Variance decomposition and power: {corpus}", ""]
    p5 = df[df["metric"] == "p_at_5"]
    if p5.groupby("question_id")["epoch"].nunique().min() >= 2:
        dec = decompose(
            p5.groupby(["question_id", "epoch"], as_index=False).agg({"score": "mean"}),
            value_col="score",
            question_col="question_id",
        )
        lines.append(
            f"- p_at_5 variance: between-question {dec.between_question:.5f}, "
            f"within-question {dec.within_question:.5f}, ICC {dec.icc:.2f} "
            f"({dec.n_questions} questions x {dec.epochs_per_question} epochs)."
        )
        lines.append(
            "- Error budget preview: the interval above is the statistical term only; "
            "template sensitivity and judge bias are systematic terms, scoped for v3.2."
        )
    configs = sorted(df["config_id"].unique())
    if len(configs) >= 2:
        a = _question_means(df[df["config_id"] == configs[0]], "p_at_5").set_index("question_id")
        b = _question_means(df[df["config_id"] == configs[1]], "p_at_5").set_index("question_id")
        shared = a.index.intersection(b.index)
        diffs = (a.loc[shared, "score"] - b.loc[shared, "score"]).to_numpy()
        lines.append(
            f"- Minimum detectable p_at_5 difference at 80 percent power: "
            f"{mde(diffs, seed=seed):.3f} (normal approximation {mde_normal_approx(diffs):.3f})."
        )
    lines.append("")
    return lines


def _methods_appendix(tables: dict[str, pd.DataFrame], seed: int) -> list[str]:
    lines = ["## Methods appendix", ""]
    lines.append(
        "- Estimators: cluster bootstrap over cluster_id (10000 replicates); paired bootstrap "
        "on per-question epoch-mean differences; TOST at margin 0.10 absolute, alpha 0.05 per "
        "one-sided test, no multiplicity adjustment across P@5 and R@5 (pre-registered, design "
        "spec section 2, frozen 2026-06-11 before any WP5 data existed)."
    )
    lines.append(
        "- Zero-failure bounding: a question succeeds only if zero hallucinated citations "
        "occurred across all epochs and all citations; collapsing epochs bounds the any-of-k "
        "failure rate, which also bounds the per-answer rate."
    )
    lines.append(f"- Seed: {seed}. No wall-clock values appear in this report.")
    for name, df in sorted(tables.items()):
        digest = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:12]
        lines.append(f"- Input table {name}: {len(df)} rows, content hash {digest}.")
    lines.append("")
    return lines


def render_report(tables: dict[str, pd.DataFrame], seed: int = DEFAULT_SEED) -> str:
    lines = ["# Statistics report", ""]
    for corpus, df in sorted(tables.items()):
        lines.extend(_headline_section(df, corpus, seed))
        lines.extend(_citation_section(df, corpus))
        lines.extend(_equivalence_section(df, corpus, seed))
        lines.extend(_variance_power_section(df, corpus, seed))
    lines.extend(_methods_appendix(tables, seed))
    return "\n".join(lines) + "\n"


def load_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Assemble one long table per corpus from the per-config CSVs on disk.

    WP5 directory convention: ``results/long/<corpus>/<run_id>.csv`` -- one CSV
    per run, and each config is its own run_id, so a corpus directory holds one
    CSV per config. The corpus is the CSV's parent directory name; all CSVs in
    a corpus directory are concatenated into a single table so the cross-config
    sections (framework equivalence, MDE) see every config at once. Keying by
    file stem instead would split each config into its own single-config corpus
    and those sections would render empty. Legacy rows land under ``legacy/``
    and so form their own corpus bucket -- they never silently mix with fresh
    data (WP1 rule).
    """
    groups: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(tables_dir.rglob("*.csv")):
        df = pd.read_csv(path, dtype={"refused": "boolean"})
        groups.setdefault(path.parent.name, []).append(df)
    return {
        corpus: pd.concat(frames, ignore_index=True) for corpus, frames in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default="results/long", help="directory of long CSVs")
    parser.add_argument("--out", default="docs/_generated/stats_report.md")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    tables = load_tables(Path(args.tables))
    if not tables:
        raise SystemExit(f"no CSV tables under {args.tables}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(tables, seed=args.seed))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
