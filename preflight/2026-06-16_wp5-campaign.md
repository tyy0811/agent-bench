# WP5 campaign eyeball record (2026-06-16)

## Campaign
- Run by Jane (paid, human-run). Gate 2 (k=1 smoke: custom-openai, langchain-openai) then Gate 3 (k=5, four configs).
- Configs and run_ids (fastapi corpus):
  - custom-openai+470d79fa: run_id 01KV7XEH72VJEAYEHMYVBKF9ZM (5 epochs, 465 rows)
  - custom-anthropic+0bc9cd53: run_id 01KV7YAYBVJH77X6NCVF50GVBC (5 epochs, 475 rows)
  - langchain-openai+gpt-4o-mini: run_id 01KV7ZV0JXVZ59C03KZFQT4W84 (5 epochs, 468 rows)
  - langchain-anthropic+claude-haiku-4-5-20251001: run_id 01KV8145TPQ40CT5109JSYQDD2 (5 epochs, 470 rows)
- Scale: 27 questions x 5 epochs x 4 configs = 540 question-runs, 0 errors.

## Headline (fastapi, from docs/_generated/stats_report.md)
- p_at_5 means: custom-openai 0.718, custom-anthropic 0.791, langchain-openai 0.627, langchain-anthropic 0.760.
- r_at_5 means: 0.833 / 0.841 / 0.836 / 0.841.
- n_clusters=12, design effects 0.67 to 1.30, ICC(p_at_5)=0.99, MDE(p_at_5) at 80 percent power = 0.110.
- Citation accuracy: 0 failures all configs; Clopper-Pearson upper bounds approx 0.13 to 0.15.

## Dup-config guard event (resolved)
- First evaluate-stats failed: load_tables refused to load because the Gate-2 smoke run dirs were still under results/epochs/, and the runbook convert loop (for run in results/epochs/*/) re-converted them into results/long/fastapi alongside the k=5 campaign. So custom-openai and langchain-openai each appeared under two run_ids (epochs [1] smoke plus [1-5] campaign). This is the PR #28/#29 dup-config guard working as designed.
- Resolution: moved (not deleted) the two smoke run dirs and their CSVs to /tmp/wp5-smoke-quarantine/. Reversible. Left exactly four clean campaign CSVs, one run_id each.
- Lesson: smoke and campaign artifacts must not co-mingle in results/epochs/ (the convert loop globs everything there). Clear results/epochs/ and results/long/<corpus>/ between the smoke gate and the full campaign, or move the smoke dirs out first.

## Anomaly verified: identical r_at_5 across the two Anthropic configs
Observation: custom-anthropic and langchain-anthropic have identical per-question r_at_5 on all 22 in-scope questions, producing a degenerate TOST CI [0.000, 0.000] (report line 26) and one scipy catastrophic-cancellation warning.

Verification (measure before trust, against the committed envelopes):
- Distinctness: four distinct run_ids and config_ids; custom-anthropic and langchain-anthropic have different run_ids and different retrieved_sources content hashes. Not a config compared to itself.
- Mechanism: retrieved_sources differ between the two configs on 16/22 questions; the distinct expected-hit count is identical on 22/22 (this makes recall identical, because retrieval_recall_at_k dedupes via set()); precision differs on 8/22 (this makes p_at_5 differ, because retrieval_precision_at_k counts duplicate chunks). Independent recompute of recall from retrieved_sources plus expected_sources matches the stored value on 22/22.
- Hand-verified q001, q003, q004 end to end (expected sources, both retrieved lists, recomputed vs stored p and r). Example q001 (expected fastapi_path_params.md): custom retrieved path_params in 2 of 5 slots (p=0.400), langchain in 1 slot (p=0.200), both recall 1.0.
- langchain-anthropic recall is epoch-constant (0/22 vary across its 5 epochs), so the epoch-mean equaling the deterministic custom value is structural, not an averaging fluke.

Verdict: genuine retrieval property. recall_at_5 saturates because both pipelines reliably surface the relevant source in top-5; precision varies with chunk composition. The [0,0] equivalence is a legitimate degenerate case (identical recall vectors), not fabricated equivalence and not a metric bug. Do not read it as a glitch.

## Deterministic-mode note
custom-* configs run --mode deterministic, so their 5 epochs are byte-identical (zero within-config epoch variance). within-question variance 0.00036, ICC 0.99. Expected.

## Eyeball verdict
Report renders for the four campaign configs (fastapi): all sections populated, no NaN or inf, intervals plausible. Data verified trustworthy and committable.

## Decision D2: legacy table excluded from the canonical report
At eyeball time the report also rendered a legacy section (single config custom-openai-legacy, TOST and variance correctly empty, valid degradation). Per decision D2 the WP1-scaffolding legacy table results/long/legacy/fastapi_postedit.csv was moved out of results/long/ so the canonical campaign report is the four configs only; the adapter's legacy-handling stays covered by the test suite. It regenerates into results/long/legacy/ via the Makefile legacy convert target if a legacy section is ever wanted, so this is reversible.

## Follow-ups (gate nothing; the report re-renders free from the committed tables)
1. TOST renderer: handle zero-variance paired diffs (annotate the degenerate identical-vector case; suppress the scipy warning). Own small PR.
2. WP5 runbook: add a line that smoke and campaign artifacts must not co-mingle in results/epochs/.
