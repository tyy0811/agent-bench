# GPT-4o-mini extraction-vs-reasoning split — three calibration items

This artifact documents the specific failure mode behind v1.1 finding 3 (jury
κ regression on completeness): on each of the three representative
disputed cells (gold=2, Haiku=2, gpt-4o-mini=1), gpt-4o-mini's
`evidence_quotes` correctly extracts the paraphrased coverage from the
agent's answer, and then its `reasoning` field denies that those quotes
constitute coverage. The score follows the reasoning, not the evidence.

The mechanism is a *post-extraction reasoning regression*: the
structured-output discipline forces the model to commit to an extraction
step (which it does correctly), and then the reasoning step applies a
literal-string-match standard to the answer text — even though the
rubric says "paraphrase allowed". This is a known failure mode in
chain-of-thought judges and shows up more in smaller models because the
reasoning step has less capacity to integrate the rubric's "paraphrase
allowed" instruction with the literal-text comparison the model is
running on autopilot.

Source: `results/calibration_v1_judge_jury_kappa_weighted_members.jsonl`
filtered to `judge_id="gpt-4o-mini-2024-07-18_completeness"` joined to
gold labels in `measurements/2026-05-04-judge-calibration-labels.jsonl`.

---

## Item q006 — FastAPI dependency caching (gold=2, Haiku=2, gpt=1)

**Reference (gold) answer:** "FastAPI caches dependency results so each
dependency is called only once per request; caching can be disabled by
setting `use_cache=False` in the `Depends()` call."

### gpt-4o-mini evidence_quotes (extraction step — correct)

```
[
  "FastAPI implements automatic caching of dependency results within a single request.",
  "To disable caching and force a fresh call each time a dependency is needed, use the `use_cache` parameter set to `False`."
]
```

Both gold points are present in the extracted quotes: (1) per-request
caching, (2) `use_cache=False` mechanism.

### gpt-4o-mini reasoning (literal-match regression)

> "The answer provides a detailed explanation of how dependency caching
> works in FastAPI and how to disable it, but it does not explicitly
> mention that caching is enabled by default and that it can be disabled
> by setting `use_cache=False` in the `Depends()` call. While it covers
> the concept of caching and provides an example, it does not directly
> address the key points from the reference answer."

**The reasoning denies what the evidence_quotes confirm.** The model has
the strings "automatic caching of dependency results within a single
request" and "use the `use_cache` parameter set to `False`" in its own
extracted evidence, then writes "does not explicitly mention that caching
is enabled by default and that it can be disabled by setting
`use_cache=False`". Score=1 follows the reasoning.

---

## Item k8s_002 — Deployment vs StatefulSet (gold=2, Haiku=2, gpt=1)

**Reference (gold) answer:** "Deployment manages stateless pods with
declarative updates; StatefulSet maintains sticky identity for each pod
— stable network identifiers, stable persistent storage, ordered
operations."

### gpt-4o-mini evidence_quotes (extraction step — correct)

```
[
  "A Deployment manages a set of Pods for an application workload that does not maintain state and provides declarative updates; its Pods are interchangeable replicas.",
  "A StatefulSet, by contrast, maintains a sticky identity for each of its Pods — stable network identifiers, stable persistent storage, and ordered deployment/scaling."
]
```

The evidence_quotes literally contain the strings `"declarative updates"`
and `"sticky identity"` — the two reference phrases gpt's reasoning will
claim are missing.

### gpt-4o-mini reasoning (literal-match regression)

> "The answer provides a detailed comparison between Deployments and
> StatefulSets, covering key differences such as pod identity, naming,
> network identity, storage, ordering, and use cases. However, it does
> not explicitly mention the points about 'declarative updates' and
> 'sticky identity' for StatefulSets as stated in the reference."

**This is the cleanest evidence-vs-reasoning contradiction in the
calibration set.** The model quotes "declarative updates" and "sticky
identity" into evidence, then writes "does not explicitly mention
'declarative updates' and 'sticky identity'". The score field is then
1, following the reasoning.

---

## Item k8s_018 — autoscaling/v2 vs v1 (gold=2, Haiku=2, gpt=1)

**Reference (gold) answer:** "autoscaling/v2 is the current stable API
version; supports memory metrics beyond CPU-only autoscaling/v1; supports
custom metrics."

### gpt-4o-mini evidence_quotes (extraction step — correct)

```
[
  "The `autoscaling/v2` API version is the current stable version (stable since Kubernetes v1.23) that includes support for scaling on **custom metrics**.",
  "The `autoscaling/v2` API version includes support for scaling on **memory metrics**, which is not available in the older `autoscaling/v1` version.",
  "With `autoscaling/v2`, you can specify multiple metrics for a HorizontalPodAutoscaler to scale on."
]
```

All three reference points are extracted: stable version, memory metrics,
custom metrics.

### gpt-4o-mini reasoning (literal-match regression)

> "The answer covers some key points from the reference but does not
> fully address all of them. It mentions the stable version of
> `autoscaling/v2`, support for custom metrics, and memory metrics, but
> it does not explicitly state that the new fields in `autoscaling/v2`
> are preserved as annotations when using `autoscaling/v1`, nor does it
> mention the need to use `autoscaling/v2` directly for memory or custom
> metric scaling for a Deployment or StatefulSet. Therefore, it
> partially covers the reference points."

**Same pattern, with extra-credit deduction.** The reasoning
acknowledges the three reference points are covered ("It mentions the
stable version of `autoscaling/v2`, support for custom metrics, and
memory metrics") and then deducts for points the *reference does not
require* ("does not explicitly state that the new fields in
`autoscaling/v2` are preserved as annotations when using
`autoscaling/v1`"). The reference (per the gold annotation) requires
three points; gpt's reasoning invents a fourth and penalizes for it.

---

## Why this matters for the writeup

This isn't "GPT-4o-mini is bad at completeness." It's a sharper claim:
*the structured-output discipline forces correct extraction, but the
reasoning step regresses to a literal-match standard the rubric does
not specify*. That regression is dimension-specific (groundedness AC1 =
1.000, relevance AC1 = 1.000 on the same model) — it surfaces only on
the 3-point ordinal scale where "paraphrase allowed" is load-bearing.

Two consequences for evaluation framework design:

1. **Per-dimension judge selection matters more than per-judge selection.**
   gpt-4o-mini is fine for binary groundedness and saturated relevance;
   it's miscalibrated for paraphrase-tolerant ordinal completeness. v1's
   global "include in jury" decision flattens this.

2. **A judge's `reasoning` field can contradict its `evidence_quotes`
   field, and the score follows the reasoning.** Internal consistency
   between the two structured-output fields is not enforced by any
   provider's structured-output API; it's a property of the model's
   capability that varies across model sizes and dimensions. v1.2
   diagnostics should include an internal-consistency check (does the
   reasoning's score-direction match what the evidence_quotes would
   support?) as an additional signal beyond raw κ.
