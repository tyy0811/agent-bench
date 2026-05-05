# κ ablation table — calibration v1

Headline metric per dimension: **groundedness → AC1**, **relevance → AC1**, **completeness → κ**. AC1 (Gwet 2008, unweighted) is used on dimensions whose v1.1 gold is prevalence-skewed enough to make Cohen's κ degenerate (groundedness 1×`1`/29×`0`, relevance 29×`2`/1×`1`); both metrics produce ≥0.95 raw agreement on those rows but Cohen's κ collapses to ≈0 because Pe approaches 1. Completeness uses Cohen's κ — its gold (23×`2`/5×`1`) is balanced enough for κ to behave normally.

| Row | Dimension | Metric | Agreement (95% CI) | N | Abstain rate | Notes |
|---|---|---|---|---|---|---|
| baseline | completeness | κ | 0.416 (-0.068, 0.866) | 26 | 0.0% |  |
| baseline | groundedness | AC1 | 1.000 (1.000, 1.000) | 26 | 0.0% |  |
| baseline | relevance | AC1 | 0.964 (0.885, 1.000) | 29 | 3.3% |  |
| baseline_no_abstain | completeness | κ | 0.416 (-0.068, 0.866) | 26 | 0.0% |  |
| baseline_no_abstain | groundedness | AC1 | 1.000 (1.000, 1.000) | 26 | 0.0% |  |
| baseline_no_abstain | relevance | AC1 | 0.963 (0.881, 1.000) | 28 | 6.7% |  |
| baseline_no_anchors | completeness | κ | 0.623 (-0.054, 1.000) | 26 | 0.0% |  |
| baseline_no_anchors | groundedness | AC1 | 0.953 (0.834, 1.000) | 24 | 7.7% |  |
| baseline_no_anchors | relevance | AC1 | 0.964 (0.885, 1.000) | 29 | 3.3% |  |
| baseline_no_cot | completeness | κ | 1.000 (1.000, 1.000) | 24 | 7.7% |  |
| baseline_no_cot | groundedness | AC1 | 0.897 (0.707, 1.000) | 23 | 11.5% |  |
| baseline_no_cot | relevance | AC1 | 0.963 (0.881, 1.000) | 28 | 6.7% |  |
| jury_kappa_weighted | completeness | κ | 0.014 (-0.077, 0.112) | 26 | 0.0% |  |
| jury_kappa_weighted | groundedness | AC1 | 1.000 (1.000, 1.000) | 26 | 0.0% |  |
| jury_kappa_weighted | relevance | AC1 | 1.000 (1.000, 1.000) | 30 | 0.0% |  |
| permute | completeness | κ | 0.506 (-0.061, 1.000) | 26 | 0.0% |  |
| permute | groundedness | AC1 | 1.000 (1.000, 1.000) | 25 | 3.8% |  |
| permute | relevance | AC1 | 0.966 (0.890, 1.000) | 30 | 0.0% |  |
