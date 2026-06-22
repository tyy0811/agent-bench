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
from stats.paired import paired_bootstrap
from stats.power import mde, mde_normal_approx
from stats.reliability import pass_k
from stats.variance import decompose

HEADLINE_METRICS = ("p_at_5", "r_at_5")
DEFAULT_SEED = 20260611


def _unit_ci(mean: float, se: float) -> tuple[float, float, bool]:
    """95 percent normal-approximation interval for a proportion, clamped to
    [0, 1]. Every metric this report intervals (p_at_5, r_at_5) is a proportion
    bounded at 1, so a raw bound outside the unit interval is a ceiling/floor
    artifact, not a reachable parameter value (a recall CI upper bound of 1.015
    reads as a bug). Returns (lo, hi, censored); ``censored`` is True when a raw
    bound fell outside [0, 1] and was clamped, so callers disclose it instead of
    hiding it. A no-op when both bounds are already inside [0, 1], so corpora
    that never approach the ceiling render byte-identically.
    """
    raw_lo, raw_hi = mean - 1.96 * se, mean + 1.96 * se
    return max(0.0, raw_lo), min(1.0, raw_hi), (raw_lo < 0.0 or raw_hi > 1.0)


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
            lo, hi, censored = _unit_ci(res.mean, primary_se)
            lines.append(
                f"| {config} | {metric} | {res.mean:.3f} | [{lo:.3f}, {hi:.3f}] ({label}) "
                f"| {res.naive_se:.4f} | {res.clustered_se:.4f} | n_clusters={res.n_clusters} "
                f"| design effect={res.design_effect:.2f} |"
            )
            if censored:
                cautions.append(
                    f"ceiling-censored: {corpus} {config} {metric}: the normal-approximation "
                    "interval extends past the [0,1] proportion bound and is reported clamped "
                    "to it; the bound sits on the ceiling rather than estimating beyond it."
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


# Difference significance for the README bolding rule (design spec section 2,
# edit-2 of WP6): a pair is "significant" only if the conventional two-sided 95
# percent paired-bootstrap CI on the per-question mean difference excludes zero.
# This is a stricter, separate question from the pre-registered TOST equivalence
# (90 percent CI vs the 0.10 margin); a pair can be non-equivalent yet still not
# significant, so the README must read significance from here, not from TOST.
PAIRED_SIGNIFICANCE_CONFIDENCE = 0.95


def _key(name: str) -> str:
    """Marker-safe key fragment: drop any ``+fingerprint`` suffix, hyphens to
    underscores. ``custom-openai+470d79fa`` -> ``custom_openai`` so README
    markers stay readable and stable when only the config fingerprint changes.
    Config names are unique per corpus (the load-time dup guard enforces it)."""
    return name.split("+")[0].replace("-", "_")


def _significant_pairs(df: pd.DataFrame, seed: int) -> list[str]:
    """Custom-vs-langchain pairs whose 95 percent paired CI on P@5 or R@5
    excludes zero, using the same pairing and clustering as the TOST section."""
    configs = sorted(df["config_id"].unique())
    customs = [c for c in configs if c.startswith("custom")]
    langchains = [c for c in configs if c.startswith("langchain")]
    significant = []
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
                res = paired_bootstrap(
                    diffs,
                    clusters=clusters if use_clusters else None,
                    confidence=PAIRED_SIGNIFICANCE_CONFIDENCE,
                    seed=seed,
                )
                if res.ci_low > 0.0 or res.ci_high < 0.0:
                    significant.append(f"{_key(custom)} vs {_key(lc)} {metric}")
    return significant


def _readme_values_section(tables: dict[str, pd.DataFrame], seed: int) -> list[str]:
    """Machine-checked anchors for README.md. Every number the README quotes
    from this report is emitted here as a ``KEY = value`` line and wrapped in a
    ``<!-- stats:KEY -->value<!-- /stats -->`` marker in the README;
    scripts/check_readme_stats.py fails loudly on any drift between the two.

    Values are recomputed with the same estimators and seed as the sections
    above, so they are byte-identical to them (locked by a consistency test in
    tests/stats/test_report.py that parses both and asserts agreement)."""
    lines = ["## README values", ""]
    for corpus, df in sorted(tables.items()):
        ck = _key(corpus)
        configs = sorted(df["config_id"].unique())
        # Headline mean and primary 95 percent interval, per config and metric.
        for config in configs:
            for metric in HEADLINE_METRICS:
                qm = _question_means(df[df["config_id"] == config], metric)
                if qm.empty:
                    continue
                res = cluster_bootstrap(
                    qm["score"].to_numpy(), qm["cluster_id"].to_numpy(), seed=seed
                )
                se = res.clustered_se if primary_is_clustered(res.n_clusters) else res.naive_se
                lo, hi, _ = _unit_ci(res.mean, se)
                stem = f"{ck}_{_key(config)}_{metric}"
                lines.append(f"- {stem}_mean = {res.mean:.3f}")
                lines.append(f"- {stem}_ci = [{lo:.3f}, {hi:.3f}]")
        # Citation zero-failure bound: included n and the exact upper bound (only
        # in the zero-failure branch, which is what the README claims).
        for config in configs:
            sub = df[df["config_id"] == config]
            cit = sub[sub["metric"] == "citation_acc"]
            if cit.empty:
                continue
            per_q_min = cit.groupby("question_id")["score"].min()
            n = len(per_q_min)
            stem = f"{ck}_{_key(config)}_citation"
            lines.append(f"- {stem}_n = {n}")
            if int((per_q_min < 1.0).sum()) == 0:
                lines.append(f"- {stem}_upper = {zero_failure_upper(n):.3f}")
                lines.append(f"- {stem}_rule_of_three = {rule_of_three(n):.3f}")
        # Framework equivalence verdicts, support margins, and mean differences.
        customs = [c for c in configs if c.startswith("custom")]
        langchains = [c for c in configs if c.startswith("langchain")]
        for custom in customs:
            for lc in langchains:
                for metric in HEADLINE_METRICS:
                    a = _question_means(df[df["config_id"] == custom], metric).set_index(
                        "question_id"
                    )
                    b = _question_means(df[df["config_id"] == lc], metric).set_index("question_id")
                    shared = a.index.intersection(b.index)
                    if shared.empty:
                        continue
                    diffs = (a.loc[shared, "score"] - b.loc[shared, "score"]).to_numpy()
                    clusters = a.loc[shared, "cluster_id"].to_numpy()
                    use = primary_is_clustered(len(np.unique(clusters)))
                    teq = tost_paired(diffs, clusters=clusters if use else None, seed=seed)
                    stem = f"{ck}_{_key(custom)}_vs_{_key(lc)}_{metric}"
                    lines.append(
                        f"- {stem}_tost = {'equivalent' if teq.equivalent else 'not established'}"
                    )
                    lines.append(f"- {stem}_support = {teq.support_margin:.3f}")
                    lines.append(f"- {stem}_diff = {diffs.mean():+.3f}")
        # Pairwise significance summary (drives the README bolding rule).
        sig = _significant_pairs(df, seed)
        lines.append(f"- {ck}_significant_pairs_95 = {'; '.join(sig) if sig else 'none'}")
        lines.append(f"- {ck}_significant_pairs_95_count = {len(sig)}")
        # Variance decomposition and minimum detectable effect.
        p5 = df[df["metric"] == "p_at_5"]
        if p5.groupby("question_id")["epoch"].nunique().min() >= 2:
            dec = decompose(
                p5.groupby(["question_id", "epoch"], as_index=False).agg({"score": "mean"}),
                value_col="score",
                question_col="question_id",
            )
            lines.append(f"- {ck}_icc_p_at_5 = {dec.icc:.2f}")
            lines.append(f"- {ck}_between_question_var_p_at_5 = {dec.between_question:.5f}")
        if len(configs) >= 2:
            a = _question_means(df[df["config_id"] == configs[0]], "p_at_5").set_index(
                "question_id"
            )
            b = _question_means(df[df["config_id"] == configs[1]], "p_at_5").set_index(
                "question_id"
            )
            shared = a.index.intersection(b.index)
            diffs = (a.loc[shared, "score"] - b.loc[shared, "score"]).to_numpy()
            lines.append(f"- {ck}_mde_p_at_5_80 = {mde(diffs, seed=seed):.3f}")
            lines.append(f"- {ck}_mde_p_at_5_80_normal = {mde_normal_approx(diffs):.3f}")
    lines.append("")
    return lines


def _pass_k_section(df: pd.DataFrame, corpus: str) -> list[str]:
    # Rendered only when refusal_correct rows exist (the fixtures carry no such
    # rows, so the golden reports are unaffected). pass^k is per config: a
    # question passes only if every epoch refused correctly. Single-run accuracy
    # is shown alongside so the reader sees where epoch variance bites (pass^k <
    # single-run) and where it does not (they are equal, e.g. a config with no
    # epoch variance), rather than reading a uniform reliability claim into it.
    if df[df["metric"] == "refusal_correct"].empty:
        return []
    lines = [f"## Refusal reliability (pass^k): {corpus}", ""]
    lines.append("| config | k | single-run | pass^k | 95 percent interval | n_questions |")
    lines.append("|---|---|---|---|---|---|")
    for config in sorted(df["config_id"].unique()):
        csub = df[df["config_id"] == config]
        if csub[csub["metric"] == "refusal_correct"].empty:
            continue
        try:
            res = pass_k(csub, "refusal_correct")
        except ValueError:
            # Tolerate-and-note unbalanced refusal epochs rather than crash the
            # whole report render (matches the citation section's graceful
            # degradation, design spec section 7).
            lines.append(f"| {config} | unbalanced epochs, skipped |  |  |  |  |")
            continue
        lines.append(
            f"| {config} | {res.k} | {res.single_run_rate:.3f} | {res.rate:.3f} | "
            f"[{res.ci_low:.3f}, {res.ci_high:.3f}] | {res.n_questions} |"
        )
    lines.append("")
    return lines


def render_report(tables: dict[str, pd.DataFrame], seed: int = DEFAULT_SEED) -> str:
    lines = ["# Statistics report", ""]
    for corpus, df in sorted(tables.items()):
        lines.extend(_headline_section(df, corpus, seed))
        lines.extend(_citation_section(df, corpus))
        lines.extend(_equivalence_section(df, corpus, seed))
        lines.extend(_variance_power_section(df, corpus, seed))
        lines.extend(_pass_k_section(df, corpus))
    lines.extend(_methods_appendix(tables, seed))
    lines.extend(_readme_values_section(tables, seed))
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
    tables: dict[str, pd.DataFrame] = {}
    for corpus, frames in sorted(groups.items()):
        merged = pd.concat(frames, ignore_index=True)
        # A config must come from exactly one run per corpus. Two run_ids for the
        # same config (e.g. a re-run after a partial failure left an orphan run
        # directory) would double-count that config's epochs, so refuse rather
        # than silently average a partial and a full run (paid-path audit #9).
        runs_per_config = merged.groupby("config_id")["run_id"].nunique()
        dupes = sorted(runs_per_config[runs_per_config > 1].index)
        if dupes:
            raise ValueError(
                f"corpus {corpus!r}: config(s) {dupes} appear under multiple run_ids; "
                "remove the stale run directory before loading"
            )
        tables[corpus] = merged
    return tables


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
