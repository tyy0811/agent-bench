---
dimension: groundedness
scale: binary
reference_based: true
abstain_allowed: true
---

# Groundedness (binary)

Score whether **every claim** in the agent's answer is directly supported by
the gold source snippets attached to this item. Paraphrase is allowed; what
matters is factual entailment.

The judge sees only the gold snippets — not the retrieved chunks. A claim
that happens to be true in the world but is not entailed by the snippets
fails groundedness.

**When to abstain (`"Unknown"`)**: if the answer is a refusal ("I don't
know" / "not in the documentation") and there is nothing to ground, score
abstain rather than 1.

## Score 0

At least one claim in the answer is not supported by any snippet.

### Example A — answer adds an unsupported claim

Question: "What's the default port for the dashboard?"
Snippets: ["The dashboard listens on port 8080 by default."]
Answer: "The default port is 8080 and TLS is enabled out of the box."

Score=0 because the TLS claim has no support in the snippet. The strict-
conjunction rule applies: even a single unsupported claim fails the binary
groundedness rubric. The grounded portion of the answer doesn't redeem it.

### Example B — answer paraphrases incorrectly

Question: "How long do connections idle before timeout?"
Snippets: ["Idle connections are closed after 30 seconds."]
Answer: "Connections close after 30 minutes of inactivity."

Score=0 because the unit is wrong (minutes vs seconds). Paraphrase is
allowed but factual content must match.

## Score 1

Every claim in the answer is directly supported by at least one snippet.

### Example C — fully grounded one-fact answer

Question: "What's the default port?"
Snippets: ["The dashboard listens on port 8080 by default."]
Answer: "Port 8080."

Score=1 because the only claim is the port number, which is in the snippet.

### Example D — fully grounded multi-claim answer

Question: "What identity guarantees does a StatefulSet provide?"
Snippets: [
  "StatefulSet pods receive an ordinal index from 0 to N-1.",
  "Each pod gets a stable hostname based on the StatefulSet name and ordinal.",
  "Storage is persistent across pod restarts and reschedules."
]
Answer: "Pods are assigned ordinal indices, stable hostnames derived from
the StatefulSet name + ordinal, and storage that persists across restarts."

Score=1 because all three claims (ordinal indices, stable hostnames,
persistent storage) are each supported by one snippet.
