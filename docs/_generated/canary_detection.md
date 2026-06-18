# Canary detection report

Provenance: golden-derived canary fixtures (tests/stats/fixtures/canary) with real corpus source passages; the judge verdicts are simulated, not a measurement of any real judge. Replace with real run-judges output for a measurement.

Each canary injects a known defect on one or more dimensions; the remaining dimensions are left clean and form the false-positive background. A verdict is flagged failing when the judge did not abstain and scored below the dimension's top anchor -- the result-blind production rule, which reads the score against the rubric ceiling, never the planted-defect label. Detection efficiency is the flagged fraction among planted defects; the false-positive rate is the flagged fraction among clean cells; both carry exact Clopper-Pearson 95 percent intervals because the per-dimension counts are small. Abstains are never detections and are reported separately, so a judge that misses defects by abstaining is distinguishable from one that confidently passes them.

| dimension | injection type(s) | planted | detected | detection efficiency | 95 percent interval | clean | false positives | false-positive rate | 95 percent interval | abstain rate |
|---|---|---|---|---|---|---|---|---|---|---|
| citation_faithfulness | absent_citation | 7 | 7 | 1.000 | [0.590, 1.000] | 13 | 0 | 0.000 | [0.000, 0.247] | 0.000 |
| completeness | incomplete | 6 | 6 | 1.000 | [0.541, 1.000] | 14 | 0 | 0.000 | [0.000, 0.232] | 0.000 |
| groundedness | ungrounded | 7 | 7 | 1.000 | [0.590, 1.000] | 13 | 0 | 0.000 | [0.000, 0.247] | 0.000 |
| relevance | none | 0 | 0 | n/a | n/a | 20 | 0 | 0.000 | [0.000, 0.168] | 0.000 |

Detection gap: no canary plants a defect on the relevance dimension, so its detection efficiency is not estimable (n/a); the false-positive rate is still measured on all 20 canaries as a clean-background control.

Canaries: 20. Input content hash: 161a27e01a35.

