"""Generate the four fixture long tables for report tests. Deterministic:
seed 20260611, no wall clock (timestamps are literals). Rerunning must
reproduce the committed CSVs byte for byte; tests/stats/test_report.py
asserts that, so fixture drift is loud."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
SEED = 20260611
N_Q = 12
EPOCHS = 2
CONFIGS = ("custom-mock+00000000", "langchain-mock+00000000")
TS = "2026-06-15T10:00:00+00:00"


def _base() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for cfg_i, cfg in enumerate(CONFIGS):
        run = f"01HZXJ5M8N9PQRSTVWXYZ0{cfg_i}234"
        for q in range(N_Q):
            cluster = f"file_{q % 4}.md"  # 4 clusters: naive-primary branch exercised
            base_p5 = 0.7 - 0.02 * cfg_i + 0.05 * (q % 3)
            for epoch in range(1, EPOCHS + 1):
                noise = rng.normal(0, 0.02)
                for metric, score in (
                    ("p_at_5", min(max(base_p5 + noise, 0.0), 1.0)),
                    ("r_at_5", min(max(base_p5 + 0.1 + noise, 0.0), 1.0)),
                    ("citation_acc", 1.0),
                ):
                    rows.append(
                        dict(
                            run_id=run,
                            timestamp=TS,
                            config_id=cfg,
                            code_version="fixture0",
                            dataset_version="sha-fixture0",
                            question_id=f"fq{q:03d}",
                            cluster_id=cluster,
                            epoch=epoch,
                            metric=metric,
                            score=round(float(score), 6),
                            latency_ms=1000.0,
                            cost_usd=0.0002,
                            refused=False,
                        )
                    )
    return pd.DataFrame(rows)


def main() -> None:
    base = _base()
    base.to_csv(OUT / "long_base.csv", index=False)

    nonzero = base.copy()
    # One citation failure per config: the nonzero-failure branch (Clopper-Pearson
    # interval, no rule-of-three phrasing) must fire for every config, else a
    # zero-failure config would still emit rule-of-three and the report would mix
    # both branches. Mutating only the first row left langchain at zero failures.
    for cfg in CONFIGS:
        cfg_cit = (nonzero["config_id"] == cfg) & (nonzero["metric"] == "citation_acc")
        nonzero.loc[nonzero.index[cfg_cit][0], "score"] = 0.5
    nonzero.to_csv(OUT / "long_nonzero_failure.csv", index=False)

    failed_eq = base.copy()
    custom = (failed_eq["config_id"] == CONFIGS[0]) & failed_eq["metric"].isin(["p_at_5", "r_at_5"])
    failed_eq.loc[custom, "score"] = (failed_eq.loc[custom, "score"] + 0.2).clip(upper=1.0)
    failed_eq.to_csv(OUT / "long_failed_equivalence.csv", index=False)

    divergent = base.copy()
    bump = divergent["cluster_id"].map({"file_0.md": 0.15, "file_1.md": -0.15}).fillna(0.0)
    mask = divergent["metric"] == "p_at_5"
    bumped = (divergent.loc[mask, "score"] + bump[mask]).clip(0.0, 1.0).round(6)
    divergent.loc[mask, "score"] = bumped
    divergent.to_csv(OUT / "long_divergent_se.csv", index=False)


if __name__ == "__main__":
    main()
