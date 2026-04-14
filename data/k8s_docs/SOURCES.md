# Kubernetes Corpus Sources

**Status:** Locked at the category level (v1.1 Week 1 step 2). Per-page
URL verification and pull dates are deferred to step 4 ingestion per
pilot-first discipline — committing to 25 specific kubernetes.io URLs
in this session without a verification pass would invert the
"draft small, validate, then bulk" rule documented in the plan's
cross-cutting #8.

**Target:** ~25–30 markdown files from kubernetes.io/docs — enough to
support 25 golden questions at ~1 question per page with headroom for
multi-hop questions that draw on 2–4 pages each.

**Content license:** All kubernetes.io/docs content is licensed under
[CC BY 4.0](https://git.k8s.io/website/LICENSE). License verification
happens per page at step 4 pull time; any page whose license terms
differ from the site default is flagged in the table below and
reassessed against the honest-evaluation brand's licensing discipline
(same pattern the v1.1 plan uses for Lynx/HaluBench CC BY-NC).

## Scope

**Include:**

- Core workload concepts: Pod, Deployment, StatefulSet, DaemonSet,
  Job, CronJob, ReplicaSet, Init Containers, Pod Lifecycle
- Networking: Service, Ingress, NetworkPolicy, EndpointSlice, DNS
- Config + state: ConfigMap, Secret, Volumes, PersistentVolumes,
  Namespaces
- Scheduling + resources: Resource Management, Node Assignment,
  Taints and Tolerations, Node-pressure Eviction
- Access control: RBAC Authorization
- Health + autoscaling: Liveness/Readiness/Startup Probes,
  Horizontal Pod Autoscaling
- Security: Pod Security Admission, Pod Security Standards

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

Each ingested page below must have:

- A canonical kubernetes.io/docs URL (source of truth, for re-scraping
  if content drifts)
- A date pulled (provenance, for audit; verified at step 4)
- A one-line rationale (why this page is in scope)
- License confirmation (default CC BY 4.0 unless a per-page notice says
  otherwise)

## Locked category breakdown

| Category | Target pages | Rationale |
|---|---|---|
| Core workloads | 9 | Pod, Pod Lifecycle, Deployment, ReplicaSet, StatefulSet, DaemonSet, Job, CronJob, Init Containers. The reranker-stressing multi-hop questions will draw on 2–4 of these per question. |
| Networking | 5 | Service, Ingress, NetworkPolicy, EndpointSlice, DNS for Services and Pods. NetworkPolicy is already validated as the pilot_005 flavor-B false_premise target. |
| Config + state | 5 | ConfigMap, Secret, Volumes, Persistent Volumes, Namespaces. Supports `simple_w_condition` questions where the answer depends on configuration context (volume type, secret mount mode, namespace scoping). |
| Scheduling + resources | 4 | Resource Management for Pods and Containers, Assigning Pods to Nodes, Taints and Tolerations, Node-pressure Eviction (already pulled). Good source for `comparison` questions (e.g. taints vs affinity) and `time_sensitive` questions (feature-state-bound scheduler behavior). |
| Access control | 1 | RBAC Authorization. Single page supports 1–2 `simple` questions about RBAC primitives. Not the reranker-stressing category. |
| Health + autoscaling | 2 | Liveness/Readiness/Startup Probes, Horizontal Pod Autoscaling. HPA is a `time_sensitive` candidate (autoscaling/v2 stable state). |
| Security | 2 | Pod Security Admission (already pulled), Pod Security Standards. Pod Security Admission is the `simple_w_condition` stressor where answer depends on enforcement level (enforce / audit / warn). |
| **Total** | **28** | Supports 25 questions with 3 pages of headroom for multi-hop fan-out. |

## Already-pulled pages (8 from the pilot corpus)

These were pulled during the pilot work and are the empirical grounding
for the threshold calibration at 0.015 and the flavor-B discipline for
pilot_005. No re-pull required unless content drift is detected at
step 4 verification.

| File | Category | Best-known URL | Pilot evidence |
|---|---|---|---|
| `k8s_configmap.md` | Config + state | `https://kubernetes.io/docs/concepts/configuration/configmap/` | — |
| `k8s_deployment.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/controllers/deployment/` | — |
| `k8s_network_policies.md` | Networking | `https://kubernetes.io/docs/concepts/services-networking/network-policies/` | **pilot_005 flavor-B target** — contains "Anything TLS related (use a service mesh or ingress controller for this)" at chunk_index 63 |
| `k8s_node_pressure_eviction.md` | Scheduling + resources | `https://kubernetes.io/docs/concepts/scheduling-eviction/node-pressure-eviction/` | — |
| `k8s_pod_security_admission.md` | Security | `https://kubernetes.io/docs/concepts/security/pod-security-admission/` | — |
| `k8s_pods.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/pods/` | pilot_001 target (Pod IP + localhost communication) |
| `k8s_replicaset.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/` | — |
| `k8s_secret.md` | Config + state | `https://kubernetes.io/docs/concepts/configuration/secret/` | — |

## Pages to pull at step 4 (20 remaining)

**Core workloads (6 to add):**
- Pod Lifecycle
- StatefulSet
- DaemonSet
- Job
- CronJob
- Init Containers

**Networking (4 to add):**
- Service
- Ingress
- EndpointSlice
- DNS for Services and Pods

**Config + state (3 to add):**
- Volumes
- Persistent Volumes
- Namespaces

**Scheduling + resources (3 to add):**
- Resource Management for Pods and Containers
- Assigning Pods to Nodes
- Taints and Tolerations

**Access control (1 to add):**
- RBAC Authorization

**Health + autoscaling (2 to add):**
- Configure Liveness, Readiness and Startup Probes
- Horizontal Pod Autoscaling

**Security (1 to add):**
- Pod Security Standards

**Step 4 checklist per page:**
1. Resolve kubernetes.io/docs URL — use the best-known path in the
   table above as a starting point; confirm the page loads at that
   path; if redirected, update SOURCES.md with the final URL and
   a one-line note explaining the redirect.
2. Confirm CC BY 4.0 licensing (default); flag any exception.
3. Pull content using the same scraper used for the pilot 8 pages
   (matching format with inline markdown links and structured
   headings).
4. Record the pull date in the "date pulled" column.
5. Verify the one-line rationale still holds after reading the
   page — if the page content doesn't support any planned
   question (see `QUESTION_PLAN.md`), flag for replacement with a
   reasoned alternative.

## Ingestion

Once all 28 files are in `data/k8s_docs/`, run:

```bash
make ingest-k8s
```

This populates `.cache/store_k8s/` with embeddings + BM25 index
matching the FastAPI corpus's chunker settings (recursive, 512-token
chunks, 64-token overlap).

**Post-ingest validation (pilot-first):** Before authoring the full
25-question golden set, run 2–3 smoke queries against the ingested
store (e.g. `"what is a StatefulSet"`, `"how does HPA scale
replicas"`, `"what happens when a Pod is evicted"`) and confirm that
the retrieval returns sensible chunks from the expected pages. Any
query that surfaces irrelevant chunks or hits the refusal gate
indicates a chunk-boundary or content-coverage issue that should be
debugged before the golden-set authoring session.
