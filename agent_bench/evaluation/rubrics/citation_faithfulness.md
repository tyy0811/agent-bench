---
dimension: citation_faithfulness
scale: binary
reference_based: true
abstain_allowed: true
---

# Citation faithfulness (binary, all-or-nothing aggregation per item)

For each [source: X.md] citation in the answer, is the cited chunk's
content actually relevant to the claim it supports? This is stricter
than the deterministic citation_accuracy metric, which only checks
that the cited chunk_id appears in the retrieved set — citation
faithfulness checks the **relevance** of the chunk to the claim.

**Aggregation rule (item-level):** any unfaithful citation in the
answer → item score = 0. A single bad citation in a multi-citation
answer is a real failure that all-or-nothing surfaces; treating it as
partial would obscure the failure mode.

## Score 0

The cited chunk's content does not support the adjacent claim.

### Example A — citation drift

Claim: "The default port is 8080."
Cited chunk content: "The dashboard supports OAuth and SAML authentication."

Score=0 because the chunk talks about authentication, not the port.
The citation is misleading even though the claim happens to be true.

### Example B — wrong topic citation

Claim: "StatefulSet pods get ordinal indices."
Cited chunk content: "Deployments support rolling updates with maxSurge and maxUnavailable parameters."

Score=0 — the cited chunk is about Deployments, not StatefulSets.
The citation does not support the claim about StatefulSet ordinals.

## Score 1

The cited chunk's content directly supports the adjacent claim.

### Example C — single accurate citation

Claim: "The default port is 8080."
Cited chunk content: "The dashboard listens on port 8080 by default."

Score=1.

### Example D — paraphrase-supported citation

Claim: "Each pod has a stable hostname."
Cited chunk content: "StatefulSet pods receive hostnames derived from the StatefulSet name plus their ordinal, and these hostnames persist across reschedules."

Score=1 — the chunk supports the claim via paraphrase.
