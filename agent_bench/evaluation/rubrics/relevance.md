---
dimension: relevance
scale: three_point
reference_based: false
abstain_allowed: true
---

# Relevance (three-point)

Does the agent's answer address the user's question? This is reference-free
— the judge sees only the question and the answer, not gold snippets or a
reference answer. Score the topic-match, not the truth-value.

## Score 0

Off-topic. The answer addresses a different question, is unintelligible,
or is a refusal that does not engage with the question's premise.

### Example A — wrong topic

Question: "How do I deploy to Kubernetes?"
Answer: "Python virtual environments isolate dependencies between projects."

Score=0 — the answer is about Python venvs, not Kubernetes deployment.

### Example B — refusal that ignores the question

Question: "What's the default replica count for a StatefulSet?"
Answer: "I cannot help with that request."

Score=0 — the refusal does not engage with the StatefulSet topic. A
proper grounded refusal ("the documentation does not specify a default
replica count for StatefulSets") would score higher.

## Score 1

Partially relevant. The answer touches the question's topic but misses
the core ask, or addresses a related-but-different question.

### Example C — adjacent but off-target

Question: "How do I deploy a StatefulSet?"
Answer: "Kubernetes runs containerized workloads on a cluster of nodes."

Score=1 because it's about Kubernetes but doesn't address StatefulSet
deployment specifically.

### Example D — answers a sibling question

Question: "What's the difference between Deployment and StatefulSet?"
Answer: "A Deployment manages stateless replicas with rolling updates."

Score=1 because it describes Deployment but doesn't compare it to
StatefulSet — only half the question is addressed.

## Score 2

Directly addresses the question's core ask.

### Example E — on-target single-fact answer

Question: "What's the default port for kubelet?"
Answer: "Port 10250."

Score=2 because it directly answers the question.

### Example F — on-target comparison

Question: "What's the difference between Deployment and StatefulSet?"
Answer: "Deployments manage stateless, interchangeable pods with rolling
updates; StatefulSets manage stateful pods with stable identities,
ordered rollouts, and persistent per-pod storage."

Score=2 — both sides of the comparison are addressed.
