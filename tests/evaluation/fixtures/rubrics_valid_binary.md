---
dimension: groundedness
scale: binary
reference_based: true
abstain_allowed: true
---

# Groundedness (binary)

Score whether every claim in the answer is supported by the gold source snippets.

## Score 0

Answer contains at least one claim not supported by the snippets.

### Example A — answer cites unsupported fact

Question: "What's the default port?"
Snippets: ["The default is 8080."]
Answer: "The default is 8080 and supports TLS."

Score=0 because the TLS claim has no support in the snippet. The
unsupported claim is sufficient to fail groundedness regardless of
how many other claims are correctly grounded — this is the binary
rubric's strict-conjunction definition.

## Score 1

Every claim in the answer is supported by at least one snippet.

### Example B — fully grounded one-sentence answer

Question: "What's the default port?"
Snippets: ["The default is 8080."]
Answer: "The default port is 8080."

Score=1 because the only claim ("default port is 8080") is directly
supported by the snippet. Paraphrase is allowed; what matters is
factual entailment.
