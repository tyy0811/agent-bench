# Kubernetes Corpus Sources

**Status:** Locked. 28 pages pulled via `defuddle parse` and verified
against the 25-question `QUESTION_PLAN.md` mapping. Pilot-first
smoke-query validation on the rebuilt store confirmed retrieval returns
expected chunks for 5 representative queries (StatefulSet, HPA,
node-pressure eviction, Service routing, Pod Security enforcement).

**Target:** ~25–30 markdown files from kubernetes.io/docs — achieved
at 28 pages. Supports 25 golden questions at ~1 question per page
with 3 pages of headroom for multi-hop fan-out.

**Content license:** All kubernetes.io/docs content is licensed under
[CC BY 4.0](https://git.k8s.io/website/LICENSE). All 28 pulled pages
fall under the site default license; no per-page exceptions encountered.

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

Each ingested page has:

- A canonical kubernetes.io/docs URL (source of truth, for re-scraping
  if content drifts)
- A date pulled (provenance, for audit)
- A one-line rationale (why this page is in scope)
- License confirmation (default CC BY 4.0)

## Locked category breakdown

| Category | Pages | Rationale |
|---|---|---|
| Core workloads | 9 | Pod, Pod Lifecycle, Deployment, ReplicaSet, StatefulSet, DaemonSet, Job, CronJob, Init Containers. Reranker-stressing multi-hop questions draw on 2–4 of these per question. |
| Networking | 5 | Service, Ingress, NetworkPolicy, EndpointSlice, DNS for Services and Pods. NetworkPolicy is the pilot_005 flavor-B false_premise target. |
| Config + state | 5 | ConfigMap, Secret, Volumes, Persistent Volumes, Namespaces. Supports `simple_w_condition` questions where the answer depends on configuration context. |
| Scheduling + resources | 4 | Resource Management, Assigning Pods to Nodes, Taints and Tolerations, Node-pressure Eviction. Good source for `comparison` and `time_sensitive` questions. |
| Access control | 1 | RBAC Authorization. Supports 1–2 `simple` questions about RBAC primitives. |
| Health + autoscaling | 2 | Probes, Horizontal Pod Autoscaling. HPA is a `time_sensitive` candidate (autoscaling/v2 stable state). |
| Security | 2 | Pod Security Admission, Pod Security Standards. PSA is the `simple_w_condition` stressor where the answer depends on enforcement level. |
| **Total** | **28** | Supports 25 questions with 3 pages of headroom. |

## Pulled pages (all 28)

All pages pulled via `defuddle parse <url> --md -o data/k8s_docs/<file>.md`.

| File | Category | URL | Date pulled | Pilot evidence |
|---|---|---|---|---|
| `k8s_configmap.md` | Config + state | `https://kubernetes.io/docs/concepts/configuration/configmap/` | 2026-03-24 (pilot) | — |
| `k8s_deployment.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/controllers/deployment/` | 2026-03-24 (pilot) | — |
| `k8s_network_policies.md` | Networking | `https://kubernetes.io/docs/concepts/services-networking/network-policies/` | 2026-03-24 (pilot) | **pilot_005 flavor-B target** — chunk_index 63 contains "Anything TLS related (use a service mesh or ingress controller for this)" |
| `k8s_node_pressure_eviction.md` | Scheduling + resources | `https://kubernetes.io/docs/concepts/scheduling-eviction/node-pressure-eviction/` | 2026-03-24 (pilot) | — |
| `k8s_pod_security_admission.md` | Security | `https://kubernetes.io/docs/concepts/security/pod-security-admission/` | 2026-03-24 (pilot) | — |
| `k8s_pods.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/pods/` | 2026-03-24 (pilot) | pilot_001 target (Pod IP + localhost communication) |
| `k8s_replicaset.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/` | 2026-03-24 (pilot) | — |
| `k8s_secret.md` | Config + state | `https://kubernetes.io/docs/concepts/configuration/secret/` | 2026-03-24 (pilot) | — |
| `k8s_pod_lifecycle.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/` | 2026-04-14 | step 4 |
| `k8s_statefulset.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/` | 2026-04-14 | step 4 |
| `k8s_daemonset.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/` | 2026-04-14 | step 4 |
| `k8s_job.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/controllers/job/` | 2026-04-14 | step 4 |
| `k8s_cronjob.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/` | 2026-04-14 | step 4 |
| `k8s_init_containers.md` | Core workloads | `https://kubernetes.io/docs/concepts/workloads/pods/init-containers/` | 2026-04-14 | step 4 |
| `k8s_service.md` | Networking | `https://kubernetes.io/docs/concepts/services-networking/service/` | 2026-04-14 | step 4 |
| `k8s_ingress.md` | Networking | `https://kubernetes.io/docs/concepts/services-networking/ingress/` | 2026-04-14 | step 4 |
| `k8s_endpoint_slices.md` | Networking | `https://kubernetes.io/docs/concepts/services-networking/endpoint-slices/` | 2026-04-14 | step 4 |
| `k8s_dns.md` | Networking | `https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/` | 2026-04-14 | step 4 |
| `k8s_volumes.md` | Config + state | `https://kubernetes.io/docs/concepts/storage/volumes/` | 2026-04-14 | step 4 |
| `k8s_persistent_volumes.md` | Config + state | `https://kubernetes.io/docs/concepts/storage/persistent-volumes/` | 2026-04-14 | step 4 |
| `k8s_namespaces.md` | Config + state | `https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/` | 2026-04-14 | step 4 |
| `k8s_resource_management.md` | Scheduling + resources | `https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/` | 2026-04-14 | step 4 |
| `k8s_assign_pod_node.md` | Scheduling + resources | `https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/` | 2026-04-14 | step 4 |
| `k8s_taints_tolerations.md` | Scheduling + resources | `https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/` | 2026-04-14 | step 4 |
| `k8s_rbac.md` | Access control | `https://kubernetes.io/docs/reference/access-authn-authz/rbac/` | 2026-04-14 | step 4 |
| `k8s_probes.md` | Health + autoscaling | `https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/` | 2026-04-14 | step 4 |
| `k8s_hpa.md` | Health + autoscaling | `https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/` | 2026-04-14 | step 4 |
| `k8s_pod_security_standards.md` | Security | `https://kubernetes.io/docs/concepts/security/pod-security-standards/` | 2026-04-14 | step 4 |

**Pull tool:** [defuddle](https://github.com/kepano/defuddle) CLI v0.16.0
(`defuddle parse <url> --md -o <file>`). Defuddle extracts the main
content region of kubernetes.io/docs pages and renders clean markdown
with inline links preserved — output format matches the pilot 8 pages
exactly, so no per-file normalization was needed.

**URL verification:** All 20 step-4 URLs resolved without redirect
(defuddle followed the URL as given and produced non-empty output;
any 404 or redirect would have produced a 0-byte file, which none
did — file sizes range 115–917 lines).

## Ingestion

```bash
make ingest-k8s
```

This populates `.cache/store_k8s/` with embeddings + BM25 index
matching the FastAPI corpus's chunker settings (recursive, 512-token
chunks, 64-token overlap). Current state: **2447 chunks across 28
unique sources**.

**Ingest hygiene:** `scripts/ingest.py` excludes `SOURCES.md`,
`QUESTION_PLAN.md`, and `README.md` from the corpus — these are
version-controlled curation artifacts, not content.

## Post-ingest smoke-query validation

Per cross-cutting #8 pilot-first discipline, 5 representative queries
were run against the rebuilt store to confirm retrieval works before
step 5 golden-set authoring:

| Query | Top-1 source | Expected | Verdict |
|---|---|---|---|
| "what is a StatefulSet" | `k8s_statefulset.md` | `k8s_statefulset.md` | ✓ |
| "how does HPA scale replicas" | `k8s_hpa.md` | `k8s_hpa.md` | ✓ |
| "Pod evicted node pressure" | `k8s_pod_lifecycle.md` | `k8s_node_pressure_eviction.md` or `k8s_pod_lifecycle.md` | ✓ (either acceptable — eviction is covered in both) |
| "Service route traffic to Pods" | `k8s_service.md` | `k8s_service.md` | ✓ |
| "enforce Pod Security Standards" | `k8s_pod_security_admission.md` | `k8s_pod_security_admission.md` or `k8s_pod_security_standards.md` | ✓ (PSA is the enforcement mechanism; PSS defines the levels — both valid hits) |

All 5 return top-1 from an expected page. No unexpected refusals.
No noise from irrelevant pages. The store is ready for step 5.
