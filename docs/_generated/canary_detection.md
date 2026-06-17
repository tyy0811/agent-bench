# Canary detection report

Provenance: synthetic demonstration fixtures (tests/stats/fixtures/canary); the judge verdicts are simulated, not a measurement of any real judge. Replace with a real canary set and real run-judges output for a measurement.

Each canary injects a known defect on one or more dimensions; the remaining dimensions are left clean and form the false-positive background. A verdict is flagged failing when the judge did not abstain and scored below the dimension's top anchor -- the result-blind production rule, which reads the score against the rubric ceiling, never the planted-defect label. Detection efficiency is the flagged fraction among planted defects; the false-positive rate is the flagged fraction among clean cells; both carry exact Clopper-Pearson 95 percent intervals because the per-dimension counts are small. Abstains are never detections and are reported separately, so a judge that misses defects by abstaining is distinguishable from one that confidently passes them.

| dimension | injection type(s) | planted | detected | detection efficiency | 95 percent interval | clean | false positives | false-positive rate | 95 percent interval | abstain rate |
|---|---|---|---|---|---|---|---|---|---|---|
| citation_faithfulness | absent_citation | 2 | 1 | 0.500 | [0.013, 0.987] | 4 | 0 | 0.000 | [0.000, 0.602] | 0.167 |
| completeness | incomplete | 2 | 2 | 1.000 | [0.158, 1.000] | 4 | 0 | 0.000 | [0.000, 0.602] | 0.000 |
| groundedness | ungrounded | 2 | 1 | 0.500 | [0.013, 0.987] | 4 | 1 | 0.250 | [0.006, 0.806] | 0.000 |
| relevance | none | 0 | 0 | n/a | n/a | 6 | 1 | 0.167 | [0.004, 0.641] | 0.000 |

Detection gap: no canary plants a defect on the relevance dimension, so its detection efficiency is not estimable (n/a); the false-positive rate is still measured on all 6 canaries as a clean-background control.

Canaries: 6. Input content hash: 9dc71c0aff72.

