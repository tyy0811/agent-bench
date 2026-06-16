# Canary detection schema

This documents the canary fixture contract used by the judge-detection harness
(`stats/detection.py`, `stats_adapters/canary.py`, `scripts/run_canary_eval.py`).

A canary is a known-defective answer injected into the judge pipeline so we can
measure how reliably the judges catch defects (detection efficiency) and how
often they flag clean answers (false-positive rate). This is the inject-and-
recover idea borrowed from detector calibration: you cannot trust a detector's
sensitivity claim without also measuring its background.

## Canary record

Each canary is one object in a JSON list. Fields:

| field | type | purpose |
|---|---|---|
| `id` | string | unique canary id, used as the judge `item_id` |
| `injection_type` | string | one of `ungrounded`, `absent_citation`, `incomplete` |
| `description` | string | one line on what the defect is (human notes) |
| `category` | string | golden-question category passed to the judges |
| `question` | string | the question the answer responds to |
| `answer` | string | the injected (defective) answer the judges score |
| `sources` | list[string] | citation targets present in the answer |
| `source_snippets` | list[string] | gold snippets the groundedness judge reads |
| `reference_answer` | string | gold answer the completeness judge reads |
| `expected_failing` | object | per-dimension ground truth (see below) |

`expected_failing` carries a boolean for every dimension:

```json
"expected_failing": {
  "groundedness": true,
  "completeness": false,
  "relevance": false,
  "citation_faithfulness": false
}
```

`true` means a defect was planted on that dimension and a correct judge should
flag it. `false` means the dimension was left clean; those cells are the false-
positive background. Multi-defect canaries are allowed (more than one `true`),
but the shipped fixture set plants exactly one defect per canary so the clean
background stays large.

## Injection types and the dimensions they target

| injection type | targets dimension | defect |
|---|---|---|
| `ungrounded` | groundedness | answer asserts facts absent from or contradicting the cited snippet |
| `absent_citation` | citation_faithfulness | claim is uncited, or attributed to a source that does not support it |
| `incomplete` | completeness | answer omits a required part of the reference answer |

### The relevance gap

There is no injection type that targets `relevance`. Relevance is reference-free
(it has no gold answer to violate), so a planted relevance defect cannot be
authored the same inject-and-recover way as the other three. Relevance therefore
has zero planted defects: its detection efficiency is not estimable (reported as
`n/a`), and every canary contributes to its clean background, so the harness
still reports a relevance false-positive rate as a control. This gap is by
design, not an omission.

## The flag rule

A verdict is counted as flagged-failing when:

```
flagged_failing = (not abstained) and (score < the dimension's top anchor)
```

The top anchor is the rubric's passing (maximum) score level: 1 for the binary
dimensions (groundedness, citation_faithfulness) and 2 for the three-point
dimensions (completeness, relevance). A partial three-point score is below the
ceiling, so it flags failing too. The rule is result-blind: it reads only the
judge's score against the rubric ceiling, never the `expected_failing` ground
truth, so it matches the production triage rule rather than an oracle.

`stats_adapters.canary.TOP_ANCHOR_BY_DIMENSION` holds the per-dimension anchors;
a drift-guard test pins them to the shipped rubric files.

## Metrics produced

Per dimension (`stats.detection.detection_by_dimension`):

- detection efficiency: flagged fraction among planted defects (sensitivity).
- false-positive rate: flagged fraction among clean cells (background).
- abstain rate: fraction of all cells where the judge abstained; an abstain is
  never a detection, so a judge that misses by abstaining is distinguishable
  from one that confidently passes a defect.

Both rates carry exact Clopper-Pearson 95 percent intervals because the per-
dimension counts are small.

## Predictions record

`run-judges` writes one record per canary and dimension. The detection report
consumes `item_id`, `dimension`, and `score` (an int level, or `Unknown` for an
abstain); the full judge provenance is also written but is not needed to render
the report.

```json
{"item_id": "canary_ungrounded_01", "dimension": "groundedness", "score": 0}
```

## Regenerating the report

- `make canary-report` is free and offline: it renders
  `docs/_generated/canary_detection.md` from a committed canary set and a
  committed predictions file. The shipped fixtures are a synthetic demonstration
  of the harness; the committed report's verdicts are simulated, not a
  measurement of any real judge.
- The paid path runs the real judges over a real canary set:
  `python scripts/run_canary_eval.py run-judges --canaries <set>.json --out <preds>.json`
  then `build-report`. This spends API budget (one judge call per canary and
  dimension) and is not run in tests or CI.
