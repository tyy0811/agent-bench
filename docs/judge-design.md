# Judge Layer — calibration writeup (v1.1.1)

## TL;DR

The v1 deliverable is a per-dimension LLM-judge layer (groundedness,
relevance, completeness) with anchored discrete rubrics, abstain
support, rubric permutation as a variance control, and a 2-judge
kappa-weighted jury. It supersedes the previous continuous-score
single-call judges. v1 was validated against a 30-item hand-labeled
calibration set spanning two corpora (FastAPI + Kubernetes); the
calibration surfaced six findings organized below as a methodology
arc rather than a flat ablation table. The interpretive headline:

- The shared retrieval stack does the heavy lifting on retrieval
  metrics (P@5, R@5, KHR vary < 0.12 across all four custom/LangChain
  × OpenAI/Anthropic configurations); the judge layer's value is in
  *measuring* the orchestrator's grounded-citation behavior, not in
  driving it.
- Calibration caught a published-rubric drift between human-grader
  and rubric-as-written (22/30 disagreements at v1.0); rubric
  clarification + re-labeling brought v1.1 inter-rater agreement to
  29/30 on groundedness.
- The 2-judge jury under v1's weighting pipeline fired both branches
  of the design doc's tracked risk simultaneously: the weights-source
  was a stub and the missing-weight fallback to 1.0 silently
  amplified an unweighted member. v1.1 fixed both; the corrected
  jury matches the calibrated single-judge baseline (κ 0.014 → 0.416
  on completeness, no API spend).
- A second-order finding the v1 design didn't anticipate: small
  models on 3-point ordinal scales with paraphrase semantics exhibit
  *at least two* distinct failure modes — one rubric-positional and
  prompt-engineering-fixable, one capacity-limited and only
  addressable by model selection. The 4A A/B against GPT-4o (full)
  is the empirical separator.
- A methodological observation that's the deepest finding of the
  calibration: Cohen's κ as a jury weight has a self-defeating
  property under intervention-induced marginal shifts. AC1 reads the
  signal correctly. v1.2 fix-list addresses this.

The closing position is *when not to use LLM-judge*: 3-point ordinal
scoring with paraphrase semantics is at the boundary where mid-tier
models (gpt-4o-mini class) exhibit capacity limits independent of
prompt engineering, and the right architectural choice is per-
dimension judge selection rather than further prompt iteration.

---

## 1. Methodology arc

The findings below are ordered as the calibration produced them, not
re-ordered for clarity. Each one has its own supporting evidence
file; the κ table at `docs/_generated/kappa_table.md` is the
quantitative summary; `DECISIONS.md` carries the per-decision
rationale that informs but doesn't repeat the writeup.

### 1.1 Rubric drift caught by frontier-model stress-test

The v1.0 hand-labeled calibration set (29 items, single-rater) ran
through a 90-cell Opus-4 stress-test (`measurements/2026-05-05-judge-
rubric-opus-stress.jsonl`, $0.20) against the published rubrics. The
test surfaced a 22/30 disagreement on groundedness — high enough to
indicate one of three things: (a) the rubric was wrong, (b) the
labels were wrong, (c) Opus was wrong.

Investigation localized the cause to a *scope mismatch* between the
rubric and the human-grader's labeling procedure. The groundedness
rubric scopes entailment to the *retrieval snippets* — a specific
binary check: every claim in the agent's answer must be entailed by
at least one retrieved snippet. The human grader had instead been
checking against *corpus documents* (which the snippets are drawn
from but which contain additional context). Under the corpus-
supported reading, claims like "useful for expensive operations like
database connections" pass; under the strict-snippets-only reading,
they fail.

The fix: the rubric was clarified with an explicit "must score 0"
reference-scope sentence, a trivial-inference clause with a
canonical-name carve-out (e.g., the snippet says "FastAPI's
`HTTPException`" and the answer says "the `HTTPException` class" —
that's still grounded), and three calibration anchors covering the
boundary cases (`q006` subtle embellishment, `k8s_006` dramatic
over-extension, `q021` trivial-inference positive).

22 v1.0 labels were flipped against the strict rubric. v1.1 inter-
rater agreement on groundedness rose to **29/30**. The methodology
note: *the rubric's reference scope was load-bearing for the dimension
to measure retrieval-grounded behavior rather than LLM general
knowledge*; relaxing it would have re-introduced the failure mode the
supersession was designed to remove.

**Why this matters for the writeup:** the strict-snippet groundedness
rubric is the v1 deliverable's identity. The benchmark is *zero
hallucinated citations on all API provider configurations* — that
claim is only meaningful under strict scope. Stress-testing the rubric
against a frontier model before publication is the cheap intervention
that catches the labeling-vs-rubric drift before the artifact ships.

### 1.2 CoT-before-score asymmetry across dimensions (tangent — see appendix)

The `baseline_no_cot` ablation row reached κ = 1.000 on completeness
— counterintuitive given the conventional CoT-helps-judging story —
but at n = 24 (vs n = 26 for `baseline`), and the no_cot row's
groundedness AC1 falls from 1.000 to 0.897, so the finding is real
but doesn't drive v1.1 design choices. The longer treatment with the
n = 24 caveat surfaced honestly is in **Appendix B — CoT-before-
score by dimension**.

### 1.3 v1 jury bug — two compounding weight-pipeline bugs

The v1 design doc's risks subsection listed *"jury κ worse than the
better individual judge — (a) kappa-weighting wrong, or (b) worse
judge drags mean"* as a tracked risk. The v1.0 calibration fired both
branches simultaneously.

The κ table row `jury_kappa_weighted` reads κ = 0.014 on
completeness, vs the single-judge `baseline` (Haiku) at κ = 0.416 —
a 30× regression. Per-member analysis from
`results/calibration_v1_judge_jury_kappa_weighted_members.jsonl`:

| Member | n | raw% | κ | AC1 |
|---|---|---|---|---|
| Haiku 4.5 alone (gold ⋈ pred) | 26 | 84.6% | +0.416 | +0.792 |
| gpt-4o-mini-2024-07-18 alone | 26 | 26.9% | +0.020 | +0.006 |
| Jury aggregate (v1) | 26 | 26.9% | +0.014 | +0.016 |

The jury aggregate matches gpt-4o-mini almost exactly. The mechanism
is not "weighted voting in the usual sense" but *missing-weight + tie-
break compounding*:

- `scripts/run_calibration.py::_load_weights_from_baseline` was a
  documented v1 stub returning `1.0` for every judge_id present in
  `baseline.json`. `baseline.json` contains only Haiku predictions
  (the baseline ablation is single-judge), so Haiku got `1.0` from
  the stub.
- gpt-4o-mini was not in the baseline file — its judge_id never
  appears there. v1's `Jury.score` had a fallback policy of
  `weights.get(judge_id, 1.0)` with a `logger.warning` for visibility.
  gpt-4o-mini got `1.0` from this fallback.
- Equal weights make a disputed (Haiku=2, gpt=1) cell aggregate as
  `(2 × 1 + 1 × 1) / 2 = 1.5`. The discretization rule
  (`_aggregate_scores`'s policy, mirrored in `_discretize_mean`) is
  *ties to lower*: `frac > 0.5 → ceil else floor`, and `0.5 > 0.5` is
  false, so 1.5 floors to 1. gpt-4o-mini's verdict wins every
  disputed cell.

The deeper structural point: weighting alone cannot rescue a
systematically miscalibrated member. Even held-out validation that
correctly assigned gpt-4o-mini's true low weight on completeness
would still let it dominate disputed ties unless its weight were
driven near zero — and at that point exclusion is more honest than
near-zero inclusion.

**v1.1 fix.** Two coordinated changes (single bundled commit, see
`ab0e054`):
- `agent_bench/evaluation/variance/jury.py`: missing-weight fallback
  to `1.0` → hard `ValueError`. v1.1 requires symmetric coverage in
  the weights source.
- `scripts/run_calibration.py::_load_weights_from_baseline` →
  `_compute_kappa_weights`: replaces the stub with real per-judge
  Cohen's κ on the dimension. Negative κ clipped to 0 (soft exclusion
  via weight). Hard-errors when any expected member is missing from
  the source.
- Configuration: `weights_source` re-pointed from
  `calibration_v1_judge_baseline.json` (Haiku-only, asymmetric) to
  `calibration_v1_judge_jury_kappa_weighted_members.jsonl` (sidecar
  from a prior jury run; both judges present). The source has
  documented circularity — weights are computed from the same
  calibration set used for κ reporting; v1.2 will use a held-out
  validation set.

**Re-aggregation (no API spend).** Re-running the existing 164
sidecar rows with κ-derived weights (Haiku 0.416, gpt-4o-mini 0.020):

| | n | raw% | κ |
|---|---|---|---|
| Jury (v1.0, broken)            | 26 | 26.9% | +0.014 |
| Jury (v1.1, corrected weights) | 26 | 84.6% | **+0.416** |
| Haiku-baseline (control)       | 26 | 84.6% | +0.416 |

The corrected jury matches the Haiku-baseline κ exactly. The
mechanism: with corrected weights, a disputed (Haiku=2, gpt=1) cell
aggregates as `(2 × 0.416 + 1 × 0.020) / 0.436 = 1.954`, frac 0.954 >
0.5, ceil to 2. Haiku's verdict wins. gpt-4o-mini's near-zero weight
correctly suppresses its verdict.

This is the **pre-committed Outcome 2** from the v1.1 jury-rescue
plan: jury matches baseline within ±0.05 → "soft exclusion via
weighting." The weighting suppresses the biased member to near-
irrelevance; the jury isn't *worse* than baseline, but it isn't
*doing meaningful work* either. The intervention is necessary but
not sufficient — the jury's value-add over single-judge depends on
the second judge being calibrated, which on completeness it isn't.

### 1.4 v1.1.1 prompt-positional intervention — one of two failure modes

The next investigation localized *why* gpt-4o-mini was so badly
miscalibrated on completeness. Confusion-matrix analysis (1A in the
investigation plan) on the existing sidecar showed:

- **17 of 19 disagreements** are gold=2/pred=1 (one-step-down)
- 1 is gold=2/pred=0, 1 is gold=1/pred=0
- **0 disagreements** are pred > gold

This is direction-aware structure, not balanced random labeling. The
probability of producing 19 same-direction disagreements by chance
under a balanced labeler is ~2⁻¹⁹. The bias is structural and
reproducible; gpt-4o-mini *consistently applies* a stricter standard
than the rubric specifies.

Reading the per-item reasoning surfaced an **extraction-vs-reasoning
split**: gpt-4o-mini's `evidence_quotes` field correctly extracts the
paraphrased coverage from the agent's answer, and then its `reasoning`
field denies that those quotes constitute coverage. The cleanest
example is `k8s_002` (Deployment vs StatefulSet) — gpt's
`evidence_quotes` literally contain the strings `"declarative
updates"` and `"sticky identity"`, while its `reasoning` says "the
answer does not explicitly mention 'declarative updates' and 'sticky
identity'." The score follows the reasoning, not the evidence. (Two
more examples in `measurements/2026-05-06-gpt4o-extraction-reasoning-
split.md`.)

The *intervention* that follows from this hypothesis: the model loses
the rubric's "paraphrase allowed" instruction across the rubric body,
the gold reference, the system answer, and its own reasoning step.
By the time it commits to a score, the literal-string-match standard
has displaced the rubric's permissive one. **Recency-positioning**
the paraphrase clause adjacent to the score instruction tests this:

```
{rubric body}
---
## Reference answer (gold)
{reference}
## Answer to score
{system_answer}
Note: a paraphrase that captures the same meaning as a gold-answer
point counts as covered. Score on content equivalence, not surface
form.
Score this answer against the rubric above. Respond with ONLY a {schema}.
```

**3A 5-item probe** (`q006`, `q011`, `k8s_002`, `k8s_006`, `k8s_018`,
$0.0013): 3/5 disputed items shifted 1 → 2 — at the binomial-
significance threshold per the pre-committed criteria. The protocol
triggered the full-26 re-run on gpt-4o-mini only (Haiku held as
control to make the v1.1 → v1.1.1 delta cleanly attributable).

**Full-26 re-run** (`scripts/_dev/rerun_completeness_v1_1_1.py`,
$0.0075):

| | n | raw% | κ | AC1 |
|---|---|---|---|---|
| v1.1   gpt-4o-mini | 26 | 26.9% | +0.020 | +0.006 |
| **v1.1.1 gpt-4o-mini** | 28 | **42.9%** | **+0.000** | **+0.232** |
| v1.1   Haiku (control) | 26 | 84.6% | +0.416 | +0.792 |

7 items shifted up (6 correct: gold=2/pred=1 → gold=2/pred=2 on
`q006`, `k8s_002`, `k8s_013`, `k8s_015`, `k8s_016`, `k8s_017`; 1
regression: `k8s_025` over-credited gold=1/pred=2). Net per-item
correctness delta: +5 items.

**Cohen's κ flat-lined** despite a 38× AC1 improvement and +16pp raw
agreement. This is the κ-as-weight degeneracy — section 1.6 below
covers the mechanism.

The intervention is real and partial: 5/19 disputed items recovered
via prompt positioning. 14 disagreements remained uncharacterized
after this step.

### 1.5 4A residual characterization — model-class-specific

The v1.1.1 result is interview-precarious framed as "fixed" (5/19 is
a partial fix, not a complete one). The right diagnostic for the
residual was the originally-deferred 4A: run a frontier-class model
on 5 of the 14 unchanged items at the same v1.1.1 prompt, and see
whether the residual is small-model-specific or rubric-under-
specified.

**4A** (`gpt-4o-2024-08-06`, items `k8s_006`, `k8s_018`, `q011`,
`q012`, `k8s_001`, $0.005–0.01): **5/5 scored correctly** — every
item that gpt-4o-mini got wrong at the v1.1.1 prompt, GPT-4o got
right at the same prompt. Clean A/B at fixed prompt varying only
the model.

The cleanest side-by-side is `k8s_018` (autoscaling/v2 vs v1). The
reference specifies three points: stable API version, memory metrics
support, custom metrics support. Both models receive the same
prompt:

- **gpt-4o-mini (score 1):** "It mentions some key points from the
  reference, including the stable version of `autoscaling/v2`,
  support for custom metrics, and memory metrics, but it does not
  explicitly state that the new fields in `autoscaling/v2` are
  preserved as annotations when using `autoscaling/v1`, nor does it
  mention the need to use `autoscaling/v2` directly for memory or
  custom metric scaling for a Deployment or StatefulSet."
- **gpt-4o (score 2):** "The answer covers all the key points from
  the reference. It mentions that the current stable version is
  autoscaling/v2, which supports scaling on memory and custom
  metrics, similar to the reference. It also notes that
  autoscaling/v1 only supports CPU-based scaling, aligning with the
  reference's points."

gpt-4o-mini's reasoning step **invents additional gold-criteria the
reference doesn't require** — "preserved as annotations," "use v2
directly for a Deployment or StatefulSet" — and deducts against
them. gpt-4o reads the reference's three points and scores against
exactly those. This is a **second, distinct failure mode** from the
1.4 finding:

- **Failure mode A (rubric-positional):** literal-match regression
  on paraphrased coverage. *Fixable* by recency-positioning the
  paraphrase clause. Recovers 5/19 items. (Section 1.4.)
- **Failure mode B (capacity-limited):** criteria-invention during
  the reasoning step — the model manufactures additional gold
  criteria the reference never specified, then deducts against them.
  *Not fixable* by the same prompt; demonstrably absent in gpt-4o.
  (This section.)

The v1.1.1 prompt addresses A but not B. B is what 4A characterizes.

### 1.6 κ-as-weight degeneracy — methodological observation

> **This section is the writeup's deepest finding.** The methodology
> arc 1.1–1.5 leads here: an intervention that improved a judge
> member at the per-cell level (raw 26.9% → 42.9%, AC1 0.006 → 0.232)
> was *silently excluded* from the jury aggregate by the weighting
> metric itself. The mechanism below generalizes beyond the v1.1.1
> instance and is what motivates v1.2 fix #5.

The v1.1.1 gpt-4o-mini result reveals a property of Cohen's κ as a
jury weight that the v1 design didn't anticipate: κ has a **self-
defeating property** under intervention-induced marginal shifts. An
intervention that improves a member can *lower* its weight even as
the member gets more accurate.

**Mechanism.** Cohen's κ = `(P_o - P_e) / (1 - P_e)`, where
`P_e = Σ_k P(gold=k) × P(pred=k)`. P_e is *not* invariant to the
predictor's marginal distribution. When a member's predictions
become more diverse — closer to gold's marginals — P_e rises in
lockstep with P_o. The numerator stays small, and κ deflates even
as raw accuracy improves.

**Empirical instance.** v1.1 gpt-4o-mini completeness pred dist:
`{0:2, 1:19, 2:5}` (concentrated at 1). v1.1.1 dist: `{0:4, 1:12,
2:12}` (more diverse, closer to gold's `{1:5, 2:23}`). Per-cell raw
accuracy 26.9% → 42.9%. AC1 (Gwet 2008, prevalence-robust):
0.006 → 0.232 (38×). Cohen's κ: 0.020 → 0.000.

`_compute_kappa_weights` clips κ < 0 to weight = 0. v1.1.1's
gpt-4o-mini κ = 0.000 → weight = 0.000 → contribution to jury
verdict is multiplied by zero. The improved member is invisible at
the aggregate level. **The κ table doesn't move at v1.1.1** despite
a real per-member improvement; the visible artifact disagrees with
the per-judge measurement.

Why this is non-obvious: in static conditions (no intervention,
fixed prompts), κ as weight is a sensible default. The self-
defeating property is invisible until you observe a real
intervention that shifts marginals. v1.0's calibration sweep
couldn't surface it because nothing was changing the marginals;
v1.1.1's intervention is the first time the calibration set has
produced an intervention-induced marginal shift.

The same prevalence trap is what motivates AC1 over κ on the
relevance and groundedness *reporting* rows of the κ table. The
v1.1.1 finding is that the same trap also affects κ when used as a
*weight*, with worse consequences: a reporting-degenerate κ is just
visually surprising; a weighting-degenerate κ silently excludes a
correctly-improved member from the aggregate.

**Implication.** The v1.2 fix-list (section 3) splits weighting and
reporting cleanly: per-dimension weight metric reusing the
`_DIM_METRIC` mapping already used for reporting. AC1 where κ
degenerates; κ where the gold's prevalence supports it.

### 1.7 Agreement uncertainty (v3.1 stats layer)

Every headline agreement number is a point estimate over a 26 to 30 item label
set, so it carries sampling uncertainty that a bare point hides. The v3.1
statistics layer (`stats/agreement.py`, a pure percentile bootstrap, 10000
replicates, seed 20260611) puts a 95 percent interval on each, joining the hand
labels against the v1.1 jury outputs through
`stats_adapters/calibration_agreement.py`. The point estimates reproduce
`docs/_generated/kappa_table.md` exactly, and so does the completeness upper
bound: both keep the degenerate resamples (every drawn item in one category,
where κ is undefined) as perfect agreement rather than dropping them, because
dropping those maximum-agreement draws would narrow the interval
anti-conservatively. The lower bound differs from the table only by the bootstrap
seed and replicate count.

| Dimension | Metric | Point | 95% CI | N |
|---|---|---|---|---|
| groundedness | AC1 | 1.000 | (1.000, 1.000) | 26 |
| relevance | AC1 | 1.000 | (1.000, 1.000) | 30 |
| completeness | κ | 0.416 | (-0.083, 0.866) | 26 |

The completeness κ of 0.416 gets an interval, not an excuse, and that interval
includes zero: at 26 items the data are consistent with no agreement above
chance. This is a descriptive interval, not a formal test of κ = 0; the point
estimate looks moderate, but the interval spans from no-better-than-chance to
strong, so a single headline number oversells it. κ collapsing on small
resamples is exactly the instability Gwet's AC1 was built to avoid, so where κ
degenerates AC1 is the more reliable read; that is why both are reported and the
κ resample convention is not worth over-engineering. The groundedness and
relevance intervals collapse to a point only because the v1.1 jury matched every
joined label on those dimensions: perfect observed agreement leaves the bootstrap
nothing to resample into disagreement, a small-sample artifact, not certainty.
The honest reading of all three is the same. With roughly 30 calibration items
the agreement estimates are directional, and the documented next step is a larger
labeled set, not a tighter claim on this one.

---

## 2. Position statement — when not to use LLM-judge

The combined findings support a sharper position than "small models
are bad at completeness." Two distinct failure modes were surfaced
on the same dimension, and they have different intervention classes:

|                    | Failure mode A (1.4) | Failure mode B (1.5) |
|--------------------|----------------------|----------------------|
| Mechanism          | Literal-match regression on paraphrased coverage | Criteria-invention during reasoning |
| Diagnostic         | 1A confusion matrix (17/19 disagreements one-step-down) | 4A A/B against gpt-4o (5/5 model-class swap fixes) |
| Intervention class | Rubric-positional prompt engineering | Model selection |
| Outcome            | Recovers 5/19 items | Recovers all 5 sampled at the same prompt |

The v1.1.1 prompt-positional fix exhausts what prompt engineering
can do on this rubric: the recency clause directs the model to
paraphrase semantics, and that's the only failure mode the
intervention can address. Iterating further on prompt design to
address criteria-invention would either (a) need a longer prompt
that re-explains the rubric's score levels in the score-decision
adjacency — which would cost tokens and likely confuse smaller
models more — or (b) require rubric simplification (binary instead
of 3-point), which is a v1.2 design change, not a tuning change.

**The structural answer for v1.2 is per-dimension judge selection.**
3-point ordinal completeness with paraphrase semantics is at the
boundary where mid-tier models exhibit capacity limits independent
of prompt engineering. Two defensible v1.2 paths:

1. **Exclude gpt-4o-mini from completeness scoring.** Per-dimension
   judge membership; jury reduces to single-judge Haiku on
   completeness; explicit and visible in the jury config (not
   emergent from κ-weight collapse).
2. **Replace gpt-4o-mini with GPT-4o on completeness.** Per-
   dimension judge selection; jury keeps two members; the second is
   a frontier-class model on the dimension that needs it.

The choice depends on cost budget. agent-bench's calibration scale
(~30 items × per-row × dimension-count) is trivially cheap on either
model; production deployment evaluating thousands of agent outputs
makes the trade-off material. For v1.2 the calibration cost
difference between the two paths is on the order of $0.15 per full
calibration sweep — well below the threshold where cost should
constrain the choice.

The honest interview answer to *"did you fix gpt-4o-mini on
completeness?"* is **no, deliberately**: the GPT-4o A/B showed the
residual bias is model-class-specific. The fix isn't another prompt
intervention; it's per-dimension judge selection. v1.1.1
demonstrated that rubric-engineering can address one of two failure
modes; the second one is what model choice is for.

**This generalizes beyond the specific dimension as a hypothesis the
v1 data is consistent with, not a claim the v1 data establishes.**
The empirical scope is narrow: 3-point ordinal × paraphrase ×
completeness, n = 26–28 items, one mid-tier model (gpt-4o-mini)
tested against one frontier model (gpt-4o) at the same prompt.

Within that scope, the combination of (multi-class discrimination) ×
(paraphrase tolerance) × (reasoning-induced elaboration latitude) is
at the capacity boundary where mid-tier models manufacture failure
modes that look like they should be prompt-tunable but aren't. Within
the same scope, frontier-class models on those dimensions; mid-tier
models on binary or strict-match dimensions where they perform
identically (groundedness AC1 = 1.000, relevance AC1 = 1.000 on the
same gpt-4o-mini that fails on completeness).

Whether this generalizes to other ordinal arities (4-point, 5-point),
other mid-tier models (Mistral, Sonnet, Gemini-Flash), or other
dimensions with paraphrase tolerance is *open* and worth replication
in v1.2. The v1 data is one mid-tier vs one frontier on one
dimension; the broader categorical claim ("don't use mid-tier on any
ordinal-with-paraphrase task") needs replication across model
families and ordinal arities before it's defensible as a general
recommendation.

---

## 3. v1.2 fix-list with empirical justification

Five items, ordered by methodology depth. Items 1–4 are escalations
of known v1 risks the calibration confirmed; item 5 is the new
finding from the v1.1.1 + 4A investigation.

### 3.1 Held-out jury weights

**v1 state.** v1.1 weights are computed on the same calibration set
used for κ reporting (circular). The pragmatic choice was driven by
N = 30 — splitting into a held-out subset would lose statistical
power on both halves.

**v1.2 fix.** A held-out 20-item validation set used solely for
jury-weight estimation; the 30-item calibration set retained for κ
reporting. Items selected by stratification across (corpus, gold-
class) so the validation set reflects the calibration set's
prevalence distribution.

**Empirical justification.** v1.1's circular weighting is documented
honestly (DECISIONS "v1.1 jury rescue" entry); a held-out set would
make the jury-weight numbers reproducible across calibration set
revisions without re-circularity.

### 3.2 Symmetric coverage / hard-error on missing weights — DONE in v1.1

The v1 silent fallback to `1.0` was the second of the two compounding
bugs in section 1.3. v1.1 made this a hard `ValueError` per
DECISIONS commit `ab0e054`. Listed here for completeness; closed.

### 3.3 Per-dimension judge membership

**v1 state.** Jury config declares members globally across all
dimensions (`configs/calibration/rows/jury_kappa_weighted.yaml`).
Weights are per-(member, dimension) but membership is per-jury.

**v1.2 fix.** Membership declared per-dimension in the jury config:

```yaml
jury:
  groundedness:
    - haiku
    - gpt-4o-mini
  relevance:
    - haiku
    - gpt-4o-mini
  completeness:
    - haiku            # gpt-4o-mini excluded; see writeup §1.5 + 4A
```

The exclusion is *visible* in the config, with a comment pointing
to the rationale. Not buried in code logic.

**Empirical justification.** 4A (writeup §1.5): GPT-4o handles 5/5
of the v1.1.1-residual items at the same prompt; gpt-4o-mini's
residual bias is model-class-specific (criteria-invention during
reasoning). v1.1's κ-as-weight handles this by collapsing the
member's weight to 0; v1.2 makes the exclusion explicit.

### 3.4 Per-dimension tie-break rule

`_discretize_mean` currently uses *ties to lower* (`floor + 1 if frac
> 0.5 else floor`) globally — selected for conservative behavior on
binary scales where "score 0 on uncertainty" matches the conservative
direction (hallucination, off-topic). v1.2 flips this per-dimension:
on 3-point completeness, "conservative" means scoring toward
*incomplete*, which is the wrong default given member miscalibration
already biases toward 1.

**This fix is independent of §3.5; even with correct AC1-weighted
aggregation, the global ties-to-lower default mis-handles ordinal
scales where the conservative direction differs from binary scales'
conservative direction.** Per-dimension tie-break is the *structural*
fix for ordinal asymmetry; per-dimension weight metric in §3.5 is the
*distributional* fix for prevalence-induced κ degeneracy. Different
defects, different fixes.

### 3.5 Per-dimension weight metric (NEW from v1.1.1)

**v1 state.** `_compute_kappa_weights` uses Cohen's κ for every
dimension. Section 1.6 demonstrated that κ has a self-defeating
property under intervention-induced marginal shifts — an
intervention that improves a member can lower its weight to zero,
silently excluding it from the aggregate.

**v1.2 fix.** Per-dimension weight metric reusing the `_DIM_METRIC`
mapping already used in
`agent_bench/evaluation/calibration/report.py`. Use AC1 (Gwet 2008)
where the dimension's gold prevalence makes κ degenerate;
κ where the gold's prevalence supports it. Same lookup, same per-
dimension policy at both reporting and weighting layers.

**Empirical justification.** v1.1.1's gpt-4o-mini intervention
(writeup §1.4 + 1.6): raw 26.9% → 42.9%, AC1 0.006 → 0.232 (38×),
κ 0.020 → 0.000. v1.1's `_compute_kappa_weights` clips the new κ at
zero, weight = 0, member silently excluded from the aggregate. AC1
as weight would have given the v1.1.1-improved member a non-zero
contribution proportional to its actual reliability, surfacing the
intervention's per-member improvement in the jury aggregate.

This is the writeup's deepest finding. The interaction between
Cohen's κ and prevalence-induced marginal skew is well-documented in
the κ-reporting literature — Gwet (2008) introduced AC1 specifically
to address it, and the κ table at `docs/_generated/kappa_table.md`
already uses AC1 over κ on relevance and groundedness for that
reason. *What's underexplored, to the author's knowledge,* is the
specific case where κ is used as a jury *weight* rather than as a
reporting statistic, and where an intervention shifts the predictor's
marginals while the gold's marginals stay fixed. v1.2's per-dimension
weight metric addresses this case structurally.

---

## 4. Closing position

The v1 calibration set — 30 hand-labeled items, two corpora, three
dimensions — was small enough that every finding above lived inside
single-digit item counts on the disputed surface. The fact that the
calibration produced six *separable* findings rather than one or two
flat κ numbers is itself a signal about evaluation design: a
calibration set sized to support stratified ablation (rubric × CoT ×
abstain × jury × prompt-positional × model-class) returns more per
item than a larger flat set used only for headline-κ reporting.

The methodology arc the calibration produced is reproducible from
the artifacts on disk:

- `docs/_generated/kappa_table.md` — the headline κ table, joined
  on `(item_id, dimension)` from
  `results/calibration_v1_judge_*.json` ⋈
  `measurements/2026-05-04-judge-calibration-labels.jsonl`. v1.1
  jury-rescue row visible at `jury_kappa_weighted_v1_1` (κ = 0.416,
  vs `jury_kappa_weighted` at κ = 0.014).
- `measurements/2026-05-05-judge-rubric-opus-stress.jsonl` — Opus-4
  stress-test that surfaced the rubric drift (§1.1).
- `measurements/2026-05-06-gpt4o-extraction-reasoning-split.md` —
  three side-by-side reasoning + evidence_quotes excerpts
  demonstrating the literal-match regression mechanism (§1.4).
- `measurements/2026-05-06-3a-paraphrase-recency-probe.jsonl` — the
  5-item probe artifact for the prompt-positional intervention
  (§1.4).
- `measurements/2026-05-06-4a-gpt4o-full-probe.jsonl` — GPT-4o A/B
  on the v1.1.1 residual; the empirical separator between the two
  failure modes (§1.5).
- `results/calibration_v1_judge_jury_kappa_weighted_v1_1_1_members.jsonl`
  — merged sidecar (v1.1 unchanged dims + v1.1.1 fresh gpt-4o-mini
  completeness rows). The data behind the per-member numbers in §1.4.
- `DECISIONS.md` — per-decision rationale for v1.1, v1.1.1, 3A, 4A.

**Total session API spend:** ~$0.013–0.018. v1.1 introduced no API
spend (re-aggregated existing predictions). v1.1.1 spent $0.0088 on
the prompt-positional intervention (5-item probe + 30-item full re-
run). 4A spent $0.005–0.01 on the diagnostic A/B.

**The v1 deliverable's position on when not to use LLM-judge:** mid-
tier models (gpt-4o-mini class) on 3-point ordinal scales with
paraphrase semantics exhibit capacity limits independent of prompt
engineering. The right architectural choice is per-dimension judge
selection, not iterative prompt tuning. Two defensible v1.2 paths
are listed in §3.3; the empirical evidence supports either one. The
choice between them depends on the cost of frontier inference at
production scale, which is a separate v1.2 decision.

---

## Appendix A — reproducer index

| Script | What it does | Cost |
|---|---|---|
| `scripts/_dev/reaggregate_jury_v1_1.py` | Re-aggregates the existing 164 sidecar rows with κ-derived weights; produces v1.1-corrected jury verdicts. Mirrors the production `Jury.score` aggregation logic offline. | $0.00 |
| `scripts/_dev/probe_3a_paraphrase_recency.py` | 5-item probe of the prompt-positional intervention on disputed completeness items; tests whether recency-positioning the paraphrase clause shifts gpt-4o-mini's verdicts. | $0.0013 |
| `scripts/_dev/rerun_completeness_v1_1_1.py` | Full-26 re-run of gpt-4o-mini completeness with the v1.1.1 production prompt. Haiku held as control. | $0.0075 |
| `scripts/_dev/probe_4a_gpt4o_full.py` | GPT-4o (full) A/B on 5 of the 14 v1.1.1-unchanged items at the same v1.1.1 prompt. Diagnostic for whether the residual is small-model-specific or rubric-under-specified. | $0.005–0.01 |

The production calibration runner (`scripts/run_calibration.py`) is
not in this list because it produces the headline κ table from the
canonical row configs; the `_dev` scripts above are one-off
diagnostics that produce the writeup's interpretive evidence.

---

## Appendix B — CoT-before-score by dimension

The `baseline_no_cot` ablation row (`use_cot=false`, schema requests
only the `score` field; reasoning + evidence_quotes omitted) shows a
per-dimension asymmetry that's interesting on its own but didn't
drive v1.1 design choices. Pulled out of the body to keep the
methodology arc focused on the v1.1 → v1.1.1 → 4A path.

| Dimension | baseline (CoT) | baseline_no_cot |
|---|---|---|
| completeness | κ = 0.416 (n = 26) | **κ = 1.000** (n = 24) |
| groundedness | AC1 = 1.000 (n = 26) | AC1 = 0.897 (n = 23) |
| relevance | AC1 = 0.964 (n = 29) | AC1 = 0.963 (n = 28) |

**Counterintuitive headline on completeness.** With CoT, the judge's
reasoning step over-emphasizes partial coverage and rationalizes
score = 1 ("the answer covers most of the points but misses
detail X") even when the gold's holistic reading is "covers the
points." Without CoT, the judge commits to a verdict against the
rubric directly, and the verdict aligns with the holistic reading.
The mechanism generalizes specifically to *ordinal scales with
permissive semantics* — where reasoning-induced elaboration can
manufacture grounds for downward verdicts.

**The n = 24 caveat.** `baseline_no_cot` excludes 2 cells (`q021`,
`k8s_012`) due to provider rate-limit retry exhaustion. Both were
gold = 2; neither was in `baseline`'s disagreement set. So the
agreement *isn't* selective in the misleading sense (the abstain set
isn't disproportionately drawn from `baseline`'s mistakes), but the
n = 24 vs n = 26 comparison is asymmetric across rows, and the
κ = 1.000 number is partly an abstain-exclusion artifact rather than
a pure counterfactual against `baseline`. The point estimate is real;
the bootstrap CI is wider than the table cell suggests.

**Why this didn't drive v1.1 design.** The no_cot row's groundedness
AC1 falls from 1.000 to 0.897 — meaningfully worse on the dimension
where CoT *does* help. Across dimensions: CoT helps on groundedness,
hurts on completeness, neutral on relevance. The right path is
*per-dimension* CoT selection (independent of v1.2 fix-list items
3.1–3.5; tracked separately as a v1.2 follow-up). Not included in
the §3 fix-list because the empirical evidence is partial (n = 24
caveat) and the asymmetric effect across dimensions makes a single
global change incorrect.

**Interview-readiness note.** A reader probing the κ table will see
the no_cot row's completeness κ = 1.000 and ask. The honest answer
is "interesting tangent, see appendix B, didn't change v1.1 design
choices because the asymmetry across dimensions doesn't support a
global flip." That answer is defensible because the appendix is
honest about the n = 24 caveat; it would not be defensible if the
body claimed CoT-before-score was load-bearing for v1's design.
