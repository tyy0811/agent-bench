# k8s campaign eyeball record (2026-06-22)

## Campaign
- Paid, human-run. Full 25-question k8s corpus (`k8s_golden.json`), K=5 per config.
- Configs and run_ids (k8s corpus, custom-only -- langchain has no `--corpus`):
  - custom-openai-k8s+470d79fa: run_id 01KVQFMGKTWP6RCBTR9EXYNH69 (5 epochs, 463 rows)
  - custom-anthropic-k8s+0bc9cd53: run_id 01KVQGR1541AY32GPBQ1WK9WHG (5 epochs, 462 rows)
- Scale: 25 questions x 5 epochs x 2 configs = 250 question-runs, 0 errors, 0 null answers.

## Verified before lift (measure, do not trust "the run is done")
- Full corpus, not the pilot: question_ids are k8s_001..k8s_025 (n=25), zero `k8s_pilot_*` ids. The 6-question pilot golden was NOT used.
- One run_id per config, 5 epochs each (convert_envelopes refuses mixed run_ids; each dir is clean).
- No co-mingling: no stray smoke/dry-run dirs under results/epochs/ (the WP5 dup-config trap did not recur).
- Metric values in range (p_at_5/r_at_5/khr in [0,1], means ~0.74-0.96), no NaN/inf/error records.
- Row-count math: 23 in-scope x 3 metrics + ~22 citation_acc (cited-answer gate) + 2 refusal_correct ~= 93/epoch x 5 ~= 463/462. Matches.

## Headline (k8s, from docs/_generated/stats_report.md)
- p_at_5 means: custom-openai-k8s 0.814 [0.734, 0.894], custom-anthropic-k8s 0.823 [0.736, 0.910].
- r_at_5 means: custom-openai-k8s 0.957 [0.898, 1.000], custom-anthropic-k8s 0.943 [0.881, 1.000] (upper bounds ceiling-censored, see below).
- n_clusters=6 (CRAG question_types), design effects 1.32-2.82.
- Citation accuracy: 0 failures in 22 included questions both configs; Clopper-Pearson upper bound 0.127.

## Key finding: k8s carries real within-question variance (the reason the corpus earns its place)
- p_at_5 variance: between-question 0.03249, within-question 0.00209, ICC 0.94 -- against fastapi's near-deterministic ICC 0.99.
- Source: agentic retrieval (`max_iterations: 3`) with LLM-driven query reformulation varies retrieval on 4/25 questions across epochs; answers vary on 24-25/25. The k8s config comment already warned that LLM-driven query variance makes any refusal_threshold > 0.015 fragile.
- Consequence: the variance-decomposition, pass^k, and clustered-SE sections are MEANINGFUL on k8s, where they are nearly degenerate on the deterministic fastapi custom configs. This is the portfolio value of the second corpus -- the stats layer working on non-trivial noise, not a redundant clean copy.

## out_of_scope handling verified
- 2 out_of_scope questions (k8s_004, k8s_024, both question_type=false_premise) emit `refusal_correct`, not retrieval metrics.
- custom-anthropic-k8s k8s_004 refuses correctly in 3/5 epochs ([F,T,T,T,F]) -- genuine refusal-reliability variance -> pass^5 = 0.500 on n=2 (wide CI [0.013, 0.987], honest small-n). custom-openai-k8s refuses both all 5 epochs (pass^5 = 1.000).

## Ceiling-censoring: discovered and fixed before committing canonical numbers
- Observation: k8s r_at_5 normal-approximation CIs (mean 0.957 / 0.943, question-level SE) extended past the proportion ceiling -- raw upper bounds 1.015 and 1.006. A recall CI with an upper bound above 1.0 reads as a bug to a statistical reader, even though it is a known normal-approx-at-ceiling artifact.
- Fix (stats/report.py `_unit_ci`): clamp the headline 95% normal-approx interval to [0,1] for proportion metrics (every headline metric is a proportion), and emit an explicit "ceiling-censored" caution when a raw bound fell outside the unit interval. Applied at both render sites (headline table and README-values block) via one shared helper, so the existing block-mirrors-table consistency test verifies they stay in sync.
- Side effect (correct): the synthetic `failed_equivalence` fixture also has custom-mock r_at_5 ≈ 1.0, whose raw upper bound exceeded 1.0 but previously DISPLAYED as "1.000" by rounding -- the old renderer hid the censoring. Its golden was regenerated to carry the new caution; the only diff is the added line, no value drifted. The other three goldens (base, nonzero_failure, divergent_se) are byte-stable (no ceiling case).
- fastapi never approaches the recall ceiling (r_at_5 ~0.84), so the clamp is a no-op there: fastapi section byte-identical, all 35 README markers unchanged (checker 35/35).

## Framework equivalence (TOST): empty by construction, not a regression
k8s is custom-only (langchain eval has no `--corpus` flag), so there is no custom-vs-langchain framework pair to test. The TOST section header renders with no rows. This is structural; do not read the empty section as a missing/failed computation. A langchain-k8s comparison is not planned -- it would require adding corpus support to run_langchain_eval.py and another paid run, with no current trigger.

## Eyeball verdict
Report renders for both k8s configs: all applicable sections populated, no NaN/inf, intervals plausible and now ceiling-honest. k8s is purely additive to the committed report (fastapi byte-identical; checker 35/35). Data verified trustworthy and committable.

## Process note (timing, not methodology)
The agreed plan last turn was prep-don't-run: enablement now, paid execution deferred until a specific audience made a second corpus worth the spend. That trigger did not fire; the campaign ran because the prep made it one command away. The measurements are good and the variance finding retroactively justifies the run on the merits, but the timing was off-trigger -- cheap-to-execute eroded the deferral. Noted for the discipline, not as a defect in the data.

## Follow-ups (gate nothing)
1. README surfacing of k8s benchmark numbers + stat markers is now UNBLOCKED (the "do not surface until a real campaign lands" constraint is satisfied) but deliberately HELD as a separate, well-scoped agent-bench task -- not appended to this landing, to keep it out of an imminent context switch.
2. The earlier zero-variance TOST render fix (fastapi anthropic [0,0]) remains an open, gate-nothing render follow-up; the ceiling-censoring fix here is the same render-robustness bucket.
