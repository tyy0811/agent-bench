# Statistics report

## Headline intervals: fastapi

| config | metric | mean | 95 percent interval (primary) | naive SE | clustered SE | n_clusters | design effect |
|---|---|---|---|---|---|---|---|
| custom-anthropic+0bc9cd53 | p_at_5 | 0.791 | [0.664, 0.917] (clustered) | 0.0566 | 0.0645 | n_clusters=12 | design effect=1.30 |
| custom-anthropic+0bc9cd53 | r_at_5 | 0.841 | [0.710, 0.971] (clustered) | 0.0662 | 0.0666 | n_clusters=12 | design effect=1.01 |
| custom-openai+470d79fa | p_at_5 | 0.718 | [0.610, 0.827] (clustered) | 0.0667 | 0.0554 | n_clusters=12 | design effect=0.69 |
| custom-openai+470d79fa | r_at_5 | 0.833 | [0.715, 0.951] (clustered) | 0.0719 | 0.0602 | n_clusters=12 | design effect=0.70 |
| langchain-anthropic+claude-haiku-4-5-20251001 | p_at_5 | 0.760 | [0.645, 0.875] (clustered) | 0.0606 | 0.0589 | n_clusters=12 | design effect=0.94 |
| langchain-anthropic+claude-haiku-4-5-20251001 | r_at_5 | 0.841 | [0.710, 0.971] (clustered) | 0.0662 | 0.0666 | n_clusters=12 | design effect=1.01 |
| langchain-openai+gpt-4o-mini | p_at_5 | 0.627 | [0.484, 0.770] (clustered) | 0.0683 | 0.0729 | n_clusters=12 | design effect=1.14 |
| langchain-openai+gpt-4o-mini | r_at_5 | 0.836 | [0.702, 0.971] (clustered) | 0.0716 | 0.0685 | n_clusters=12 | design effect=0.91 |

## Citation accuracy zero-failure bound: fastapi

- custom-anthropic+0bc9cd53: 0 failures in 21 included questions (included n=21, excluded 1 citation-free questions; cited in all epochs: 21, in some epochs: 0). Exact Clopper-Pearson 95 percent upper bound on the per-question failure rate: 0.133 (rule of three approximation 3/n = 0.143).
- custom-openai+470d79fa: 0 failures in 19 included questions (included n=19, excluded 3 citation-free questions; cited in all epochs: 19, in some epochs: 0). Exact Clopper-Pearson 95 percent upper bound on the per-question failure rate: 0.146 (rule of three approximation 3/n = 0.158).
- langchain-anthropic+claude-haiku-4-5-20251001: 0 failures in 20 included questions (included n=20, excluded 2 citation-free questions; cited in all epochs: 20, in some epochs: 0). Exact Clopper-Pearson 95 percent upper bound on the per-question failure rate: 0.139 (rule of three approximation 3/n = 0.150).
- langchain-openai+gpt-4o-mini: 0 failures in 20 included questions (included n=20, excluded 2 citation-free questions; cited in all epochs: 19, in some epochs: 1). Exact Clopper-Pearson 95 percent upper bound on the per-question failure rate: 0.139 (rule of three approximation 3/n = 0.150).

## Framework equivalence (TOST): fastapi

- custom-anthropic+0bc9cd53 vs langchain-anthropic+claude-haiku-4-5-20251001, p_at_5: mean diff +0.031, 90 percent CI [-0.013, +0.076], n=12: equivalent within plus or minus 0.10; the data support equivalence down to plus or minus 0.076.
- custom-anthropic+0bc9cd53 vs langchain-anthropic+claude-haiku-4-5-20251001, r_at_5: mean diff +0.000, 90 percent CI [+0.000, +0.000], n=12: equivalent within plus or minus 0.10; the data support equivalence down to plus or minus 0.000.
- custom-anthropic+0bc9cd53 vs langchain-openai+gpt-4o-mini, p_at_5: mean diff +0.164, 90 percent CI [+0.031, +0.317], n=12: equivalence not established at plus or minus 0.10; the data support only plus or minus 0.317.
- custom-anthropic+0bc9cd53 vs langchain-openai+gpt-4o-mini, r_at_5: mean diff +0.005, 90 percent CI [-0.076, +0.095], n=12: equivalent within plus or minus 0.10; the data support equivalence down to plus or minus 0.095.
- custom-openai+470d79fa vs langchain-anthropic+claude-haiku-4-5-20251001, p_at_5: mean diff -0.042, 90 percent CI [-0.105, +0.016], n=12: equivalence not established at plus or minus 0.10; the data support only plus or minus 0.105.
- custom-openai+470d79fa vs langchain-anthropic+claude-haiku-4-5-20251001, r_at_5: mean diff -0.008, 90 percent CI [-0.100, +0.068], n=12: equivalence not established at plus or minus 0.10; the data support only plus or minus 0.100.
- custom-openai+470d79fa vs langchain-openai+gpt-4o-mini, p_at_5: mean diff +0.091, 90 percent CI [-0.010, +0.202], n=12: equivalence not established at plus or minus 0.10; the data support only plus or minus 0.202.
- custom-openai+470d79fa vs langchain-openai+gpt-4o-mini, r_at_5: mean diff -0.003, 90 percent CI [-0.043, +0.032], n=12: equivalent within plus or minus 0.10; the data support equivalence down to plus or minus 0.043.

## Variance decomposition and power: fastapi

- p_at_5 variance: between-question 0.06051, within-question 0.00036, ICC 0.99 (22 questions x 5 epochs).
- Error budget preview: the interval above is the statistical term only; template sensitivity and judge bias are systematic terms, scoped for v3.2.
- Minimum detectable p_at_5 difference at 80 percent power: 0.110 (normal approximation 0.136).

## Refusal reliability (pass^k): fastapi

| config | k | single-run | pass^k | 95 percent interval | n_questions |
|---|---|---|---|---|---|
| custom-anthropic+0bc9cd53 | 5 | 0.560 | 0.400 | [0.053, 0.853] | 5 |
| custom-openai+470d79fa | 5 | 0.640 | 0.600 | [0.147, 0.947] | 5 |
| langchain-anthropic+claude-haiku-4-5-20251001 | 5 | 0.200 | 0.200 | [0.005, 0.716] | 5 |
| langchain-openai+gpt-4o-mini | 5 | 0.760 | 0.600 | [0.147, 0.947] | 5 |

## Methods appendix

- Estimators: cluster bootstrap over cluster_id (10000 replicates); paired bootstrap on per-question epoch-mean differences; TOST at margin 0.10 absolute, alpha 0.05 per one-sided test, no multiplicity adjustment across P@5 and R@5 (pre-registered, design spec section 2, frozen 2026-06-11 before any WP5 data existed).
- Zero-failure bounding: a question succeeds only if zero hallucinated citations occurred across all epochs and all citations; collapsing epochs bounds the any-of-k failure rate, which also bounds the per-answer rate.
- Seed: 20260611. No wall-clock values appear in this report.
- Input table fastapi: 1878 rows, content hash c28484520117.

## README values

- fastapi_custom_anthropic_p_at_5_mean = 0.791
- fastapi_custom_anthropic_p_at_5_ci = [0.664, 0.917]
- fastapi_custom_anthropic_r_at_5_mean = 0.841
- fastapi_custom_anthropic_r_at_5_ci = [0.710, 0.971]
- fastapi_custom_openai_p_at_5_mean = 0.718
- fastapi_custom_openai_p_at_5_ci = [0.610, 0.827]
- fastapi_custom_openai_r_at_5_mean = 0.833
- fastapi_custom_openai_r_at_5_ci = [0.715, 0.951]
- fastapi_langchain_anthropic_p_at_5_mean = 0.760
- fastapi_langchain_anthropic_p_at_5_ci = [0.645, 0.875]
- fastapi_langchain_anthropic_r_at_5_mean = 0.841
- fastapi_langchain_anthropic_r_at_5_ci = [0.710, 0.971]
- fastapi_langchain_openai_p_at_5_mean = 0.627
- fastapi_langchain_openai_p_at_5_ci = [0.484, 0.770]
- fastapi_langchain_openai_r_at_5_mean = 0.836
- fastapi_langchain_openai_r_at_5_ci = [0.702, 0.971]
- fastapi_custom_anthropic_citation_n = 21
- fastapi_custom_anthropic_citation_upper = 0.133
- fastapi_custom_anthropic_citation_rule_of_three = 0.143
- fastapi_custom_openai_citation_n = 19
- fastapi_custom_openai_citation_upper = 0.146
- fastapi_custom_openai_citation_rule_of_three = 0.158
- fastapi_langchain_anthropic_citation_n = 20
- fastapi_langchain_anthropic_citation_upper = 0.139
- fastapi_langchain_anthropic_citation_rule_of_three = 0.150
- fastapi_langchain_openai_citation_n = 20
- fastapi_langchain_openai_citation_upper = 0.139
- fastapi_langchain_openai_citation_rule_of_three = 0.150
- fastapi_custom_anthropic_vs_langchain_anthropic_p_at_5_tost = equivalent
- fastapi_custom_anthropic_vs_langchain_anthropic_p_at_5_support = 0.076
- fastapi_custom_anthropic_vs_langchain_anthropic_p_at_5_diff = +0.031
- fastapi_custom_anthropic_vs_langchain_anthropic_r_at_5_tost = equivalent
- fastapi_custom_anthropic_vs_langchain_anthropic_r_at_5_support = 0.000
- fastapi_custom_anthropic_vs_langchain_anthropic_r_at_5_diff = +0.000
- fastapi_custom_anthropic_vs_langchain_openai_p_at_5_tost = not established
- fastapi_custom_anthropic_vs_langchain_openai_p_at_5_support = 0.317
- fastapi_custom_anthropic_vs_langchain_openai_p_at_5_diff = +0.164
- fastapi_custom_anthropic_vs_langchain_openai_r_at_5_tost = equivalent
- fastapi_custom_anthropic_vs_langchain_openai_r_at_5_support = 0.095
- fastapi_custom_anthropic_vs_langchain_openai_r_at_5_diff = +0.005
- fastapi_custom_openai_vs_langchain_anthropic_p_at_5_tost = not established
- fastapi_custom_openai_vs_langchain_anthropic_p_at_5_support = 0.105
- fastapi_custom_openai_vs_langchain_anthropic_p_at_5_diff = -0.042
- fastapi_custom_openai_vs_langchain_anthropic_r_at_5_tost = not established
- fastapi_custom_openai_vs_langchain_anthropic_r_at_5_support = 0.100
- fastapi_custom_openai_vs_langchain_anthropic_r_at_5_diff = -0.008
- fastapi_custom_openai_vs_langchain_openai_p_at_5_tost = not established
- fastapi_custom_openai_vs_langchain_openai_p_at_5_support = 0.202
- fastapi_custom_openai_vs_langchain_openai_p_at_5_diff = +0.091
- fastapi_custom_openai_vs_langchain_openai_r_at_5_tost = equivalent
- fastapi_custom_openai_vs_langchain_openai_r_at_5_support = 0.043
- fastapi_custom_openai_vs_langchain_openai_r_at_5_diff = -0.003
- fastapi_significant_pairs_95 = custom_anthropic vs langchain_openai p_at_5
- fastapi_significant_pairs_95_count = 1
- fastapi_icc_p_at_5 = 0.99
- fastapi_between_question_var_p_at_5 = 0.06051
- fastapi_mde_p_at_5_80 = 0.110
- fastapi_mde_p_at_5_80_normal = 0.136

