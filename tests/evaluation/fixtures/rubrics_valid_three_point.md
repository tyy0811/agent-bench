---
dimension: relevance
scale: three_point
reference_based: false
abstain_allowed: true
---

# Relevance (three-point)

Does the answer address the user's question?

## Score 0

Off-topic. Answer addresses a different question or is unintelligible.

### Example A — wrong topic

Question: "How do I deploy to Kubernetes?"
Answer: "Python virtual environments isolate dependencies."

Score=0 because the answer is about Python venvs, not deployment.

## Score 1

Partially relevant. Answer touches the question but misses the core ask.

### Example B — adjacent but off-target

Question: "How do I deploy to Kubernetes?"
Answer: "Kubernetes runs containerized workloads on a cluster of nodes."

Score=1 because it's about Kubernetes but doesn't say how to deploy.

## Score 2

Directly addresses the question.

### Example C — on-target

Question: "How do I deploy to Kubernetes?"
Answer: "Apply a Deployment manifest with kubectl apply -f deployment.yaml."

Score=2 because it gives a concrete deployment action.
