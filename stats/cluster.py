"""Cluster bootstrap standard errors and the pre-registered primary rule.

Pre-registration (design spec section 2.2, frozen 2026-06-11): clustered SE
is primary when n_clusters >= PRIMARY_THRESHOLD, else question-level is
primary and clustered is sensitivity. DIVERGENCE_RATIO drives the report's
correlation-sensitivity caution on naive-primary corpora.

Pure module: stdlib + numpy only (guardrail 1).
"""

from dataclasses import dataclass

import numpy as np

PRIMARY_THRESHOLD = 10
DIVERGENCE_RATIO = 1.5
DEFAULT_SEED = 20260611
DEFAULT_N_BOOT = 10_000


@dataclass(frozen=True)
class ClusterSE:
    mean: float
    naive_se: float
    clustered_se: float
    n_clusters: int
    design_effect: float


def primary_is_clustered(n_clusters: int) -> bool:
    return n_clusters >= PRIMARY_THRESHOLD


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> ClusterSE:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.shape != clusters.shape:
        raise ValueError("values and clusters must align")
    labels = np.unique(clusters)
    groups = [values[clusters == c] for c in labels]
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[j] for j in picks])
        boot_means[i] = sample.mean()
    mean = float(values.mean())
    naive_se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    clustered_se = float(boot_means.std(ddof=1))
    deff = float((clustered_se / naive_se) ** 2) if naive_se > 0 else float("nan")
    return ClusterSE(mean, naive_se, clustered_se, len(labels), deff)
