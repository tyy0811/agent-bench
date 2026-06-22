# Statistics report

## Headline intervals: mini

| config | metric | mean | 95 percent interval (primary) | naive SE | clustered SE | n_clusters | design effect |
|---|---|---|---|---|---|---|---|
| custom-mock+00000000 | p_at_5 | 0.750 | [0.719, 0.780] (question-level) | 0.0155 | 0.0016 | n_clusters=4 | design effect=0.01 |
| custom-mock+00000000 | r_at_5 | 0.850 | [0.819, 0.880] (question-level) | 0.0155 | 0.0016 | n_clusters=4 | design effect=0.01 |
| langchain-mock+00000000 | p_at_5 | 0.727 | [0.701, 0.753] (question-level) | 0.0134 | 0.0032 | n_clusters=4 | design effect=0.06 |
| langchain-mock+00000000 | r_at_5 | 0.827 | [0.801, 0.853] (question-level) | 0.0134 | 0.0032 | n_clusters=4 | design effect=0.06 |

## Citation accuracy zero-failure bound: mini

- custom-mock+00000000: 0 failures in 12 included questions (included n=12, excluded 0 citation-free questions; cited in all epochs: 12, in some epochs: 0). Exact Clopper-Pearson 95 percent upper bound on the per-question failure rate: 0.221 (rule of three approximation 3/n = 0.250).
- langchain-mock+00000000: 0 failures in 12 included questions (included n=12, excluded 0 citation-free questions; cited in all epochs: 12, in some epochs: 0). Exact Clopper-Pearson 95 percent upper bound on the per-question failure rate: 0.221 (rule of three approximation 3/n = 0.250).

## Framework equivalence (TOST): mini

- custom-mock+00000000 vs langchain-mock+00000000, p_at_5: mean diff +0.023, 90 percent CI [+0.013, +0.031], n=12: equivalent within plus or minus 0.10; the data support equivalence down to plus or minus 0.031.
- custom-mock+00000000 vs langchain-mock+00000000, r_at_5: mean diff +0.023, 90 percent CI [+0.013, +0.031], n=12: equivalent within plus or minus 0.10; the data support equivalence down to plus or minus 0.031.

## Variance decomposition and power: mini

- p_at_5 variance: between-question 0.00227, within-question 0.00031, ICC 0.88 (12 questions x 2 epochs).
- Error budget preview: the interval above is the statistical term only; template sensitivity and judge bias are systematic terms, scoped for v3.2.
- Minimum detectable p_at_5 difference at 80 percent power: 0.019 (normal approximation 0.016).

## Methods appendix

- Estimators: cluster bootstrap over cluster_id (10000 replicates); paired bootstrap on per-question epoch-mean differences; TOST at margin 0.10 absolute, alpha 0.05 per one-sided test, no multiplicity adjustment across P@5 and R@5 (pre-registered, design spec section 2, frozen 2026-06-11 before any WP5 data existed).
- Zero-failure bounding: a question succeeds only if zero hallucinated citations occurred across all epochs and all citations; collapsing epochs bounds the any-of-k failure rate, which also bounds the per-answer rate.
- Seed: 20260611. No wall-clock values appear in this report.
- Input table mini: 144 rows, content hash 5b5670437a65.

## README values

- mini_custom_mock_p_at_5_mean = 0.750
- mini_custom_mock_p_at_5_ci = [0.719, 0.780]
- mini_custom_mock_r_at_5_mean = 0.850
- mini_custom_mock_r_at_5_ci = [0.819, 0.880]
- mini_langchain_mock_p_at_5_mean = 0.727
- mini_langchain_mock_p_at_5_ci = [0.701, 0.753]
- mini_langchain_mock_r_at_5_mean = 0.827
- mini_langchain_mock_r_at_5_ci = [0.801, 0.853]
- mini_custom_mock_citation_n = 12
- mini_custom_mock_citation_upper = 0.221
- mini_custom_mock_citation_rule_of_three = 0.250
- mini_langchain_mock_citation_n = 12
- mini_langchain_mock_citation_upper = 0.221
- mini_langchain_mock_citation_rule_of_three = 0.250
- mini_custom_mock_vs_langchain_mock_p_at_5_tost = equivalent
- mini_custom_mock_vs_langchain_mock_p_at_5_support = 0.031
- mini_custom_mock_vs_langchain_mock_p_at_5_diff = +0.023
- mini_custom_mock_vs_langchain_mock_p_at_5_ci90 = [+0.013, +0.031]
- mini_custom_mock_vs_langchain_mock_p_at_5_ci95 = [+0.011, +0.032]
- mini_custom_mock_vs_langchain_mock_r_at_5_tost = equivalent
- mini_custom_mock_vs_langchain_mock_r_at_5_support = 0.031
- mini_custom_mock_vs_langchain_mock_r_at_5_diff = +0.023
- mini_custom_mock_vs_langchain_mock_r_at_5_ci90 = [+0.013, +0.031]
- mini_custom_mock_vs_langchain_mock_r_at_5_ci95 = [+0.011, +0.032]
- mini_significant_pairs_95 = custom_mock vs langchain_mock p_at_5; custom_mock vs langchain_mock r_at_5
- mini_significant_pairs_95_count = 2
- mini_icc_p_at_5 = 0.88
- mini_between_question_var_p_at_5 = 0.00227
- mini_mde_p_at_5_80 = 0.019
- mini_mde_p_at_5_80_normal = 0.016

