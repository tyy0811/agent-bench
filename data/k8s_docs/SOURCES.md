# Kubernetes Corpus Sources

**Status:** Placeholder — curation scheduled as a separate work session
outside the multi-corpus refactor.

**Target:** ~30–40 markdown files from kubernetes.io/docs covering the
concepts a technical reviewer would naturally type into the demo —
not comprehensive K8s coverage.

## Scope

**Include:**

- Core workload concepts: Pod, Deployment, StatefulSet, DaemonSet, Job,
  CronJob, ReplicaSet
- Networking: Service, Ingress, NetworkPolicy, EndpointSlice
- Config + state: ConfigMap, Secret, Volume, PersistentVolume, Namespace
- Access control: RBAC (Role, RoleBinding, ServiceAccount)
- Cross-referencing overview pages: "Connecting Applications with
  Services", "Workload Resources", "Services, Load Balancing, and
  Networking" — these stress the reranker because relevance spreads
  across multiple chunks per query

**Exclude:**

- Cluster administration deep-dives (etcd, kubelet, kube-apiserver
  internals) — wrong audience for a recruiter-facing demo
- Tutorials (long-form, chunk poorly, hurt retrieval precision)
- kubectl command reference and API reference — wrong shape for RAG,
  better served by `--help`
- Release notes and version history — no lasting value for Q&A

## Curation policy

This corpus targets **recruiter-likely questions**, not coverage. A
question about etcd raft internals will be correctly refused — the
refusal mechanism is part of the demo story, not a failure mode.

Each ingested file below must have:

- A URL (source of truth, for re-scraping if content drifts)
- A date pulled (provenance, for audit)
- A one-line rationale (why this page is in scope)

| URL | Date pulled | Rationale |
|-----|------------|-----------|
| _TBD_ | _TBD_ | _TBD_ |

See `docs/plans/2026-04-12-multi-corpus-refactor-design.md` section
"Corpus Curation — Kubernetes" for the full policy.

## Ingestion

Once curated files are in place, run:

```bash
make ingest-k8s
```

This populates `.cache/store_k8s/` with embeddings + BM25 index matching
the FastAPI corpus's chunker settings (recursive, 512-token chunks,
64-token overlap).
