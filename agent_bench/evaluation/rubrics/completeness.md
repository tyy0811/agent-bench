---
dimension: completeness
scale: three_point
reference_based: true
abstain_allowed: true
---

# Completeness (three-point)

Score how much of the gold reference answer is covered by the agent's
answer. This is reference-based — the judge sees the gold reference
and the agent's answer; score on **coverage of facts** in the
reference, not on additional facts the agent may have included.

The judge does not penalize the agent for adding correct extra detail
(that's a separate concern). Score only on what fraction of the
reference's points are present.

## Score 0

None of the reference's key points are present in the answer.

### Example A — answer addresses different facts

Reference: "StatefulSet pods receive ordinal indices, stable hostnames, and persistent storage."
Answer: "Kubernetes uses YAML manifests to declare resources."

Score=0 — none of the three reference points (ordinal, hostname, storage) appear.

### Example B — refusal that covers nothing

Reference: "The default port is 8080."
Answer: "I cannot find that information."

Score=0 — the reference's single point (port=8080) is not in the answer.

## Score 1

Some but not all of the reference's points are present.

### Example C — partial coverage

Reference: "StatefulSet pods receive ordinal indices, stable hostnames, and persistent storage."
Answer: "StatefulSet pods get ordinal indices."

Score=1 — one of three points covered.

### Example D — half a comparison

Reference: "Deployments manage stateless replicas; StatefulSets manage stateful pods with stable identities."
Answer: "Deployments manage stateless replicas with rolling updates."

Score=1 — Deployment side covered, StatefulSet side missing.

## Score 2

All of the reference's key points are present (paraphrase allowed).

### Example E — full coverage with paraphrase

Reference: "StatefulSet pods receive ordinal indices, stable hostnames, and persistent storage."
Answer: "Each pod gets an ordinal number, a stable DNS name, and storage that survives restarts."

Score=2 — all three points covered with paraphrase.

### Example F — full coverage of single-fact reference

Reference: "The default port is 8080."
Answer: "Port 8080."

Score=2 — the only reference point is covered.
