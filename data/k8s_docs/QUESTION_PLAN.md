# K8s Golden Dataset — Question Plan

**Status:** Structural guide for Week 1 step 5 authoring (v1.1 plan).
This document defines the 25-question target distribution, per-type
source-page mapping, and authoring constraints. It does NOT contain
the 25 specific question texts — those are authored during step 5 in
a fresh session, per cross-cutting #8 pilot-first discipline.

**Upstream contracts:**
- Taxonomy: CRAG 8-type (Yang et al., NeurIPS 2024) — see DECISIONS.md
  "K8s golden dataset uses CRAG's 8-type taxonomy as the schema".
- Source pages: see `SOURCES.md` (28 pages, category-locked; 8 already
  pulled, 20 to pull at step 4).
- Schema: see `agent_bench/evaluation/harness.py` `GoldenQuestion`
  plus the v1.1 plan's methodology #3 source-attribution fields.
- Flavor A/B for `false_premise`: see DECISIONS.md "False-premise
  questions come in two flavors".

---

## Target distribution (25 questions total)

| CRAG type | Count | Schema field | Notes |
|---|---|---|---|
| `simple` | 5–6 | `question_type: "simple"` | Baseline retrieval: direct lookup in 1 page, 1–2 sentence answer. |
| `simple_w_condition` | 3–4 | `question_type: "simple_w_condition"` | Answer depends on a condition stated in the question (enforcement level, volume type, Pod phase). |
| `comparison` | 3–4 | `question_type: "comparison"` | Answer compares two concepts across 2 pages; reranker stress. |
| `multi_hop` | 5–6 | `question_type: "multi_hop"` | Answer synthesizes 2–4 pages; reranker-stressing by construction. |
| `false_premise` | 3–4 | `question_type: "false_premise"` | Grounded refusal stress. Flavor A (pure refusal) + flavor B (documented negative). |
| `set` / `aggregation` / `post_processing_heavy` | 0–3 | respective values | Optional. Include only if natural from corpus content. |
| **Total** | **25** | | |

**Orthogonal flag:** `time_sensitive: bool` on 2–3 questions. Does
NOT replace `question_type` — it's an independent property for
version-bounded content (feature state, API version migration,
deprecations).

---

## Per-type source-page mapping

Each row identifies the K8s concept pages a question of that type
should draw from. Multi-hop and comparison questions list multiple
pages intentionally.

### simple (5–6 slots)

Pool questions where a 1–2 sentence answer lives inside a single page.

| Candidate source | CRAG slot justification |
|---|---|
| `k8s_pods.md` | Pod IP semantics, container sharing, ephemeral containers |
| `k8s_deployment.md` | What a Deployment is, declarative update mechanic |
| `k8s_configmap.md` | What a ConfigMap is, immutable field |
| `k8s_secret.md` | What a Secret is, volume mount modes |
| RBAC Authorization *(step 4 page)* | RBAC primitive definitions (Role, RoleBinding, ClusterRole) |
| StatefulSet *(step 4 page)* | StatefulSet identity guarantees |
| DaemonSet *(step 4 page)* | One-per-node scheduling contract |
| Namespaces *(step 4 page)* | Namespace scoping for resources |

**Authoring rule:** Each `simple` question must have exactly one
expected source page and 1–2 source snippets. KHR target ≥ 0.60 on
the authored keywords.

### simple_w_condition (3–4 slots)

Pool questions where the answer explicitly depends on a condition
named in the question.

| Candidate source | Condition that shapes the answer |
|---|---|
| `k8s_pod_security_admission.md` | enforcement level: `enforce` / `audit` / `warn` |
| `k8s_secret.md` | mount mode: environment variable vs file in volume |
| Liveness/Readiness/Startup Probes *(step 4)* | probe type: liveness vs readiness vs startup |
| Volumes *(step 4)* | volume type: emptyDir vs configMap vs persistentVolumeClaim |
| Node-pressure Eviction (`k8s_node_pressure_eviction.md`) | resource under pressure: memory vs disk vs inodes |

**Authoring rule:** The condition must be named in the question
stem, not implied. The expected answer must change materially if the
condition flips. Example: "How is a Secret mounted as a volume
versus consumed as an environment variable?" is a valid
`simple_w_condition`; "How is a Secret mounted?" is `simple`.

### comparison (3–4 slots)

Pool questions where the answer explicitly compares two K8s concepts
that span 2 pages.

| Page pair | Concept compared |
|---|---|
| Deployment vs StatefulSet *(step 4)* | stateless vs stateful workload semantics |
| Deployment vs DaemonSet *(step 4)* | replica-count vs one-per-node scheduling |
| ConfigMap vs Secret | non-confidential vs confidential data, mount parity |
| Service vs Ingress *(step 4)* | L4 vs L7 exposure |
| Taints/Tolerations vs Node Affinity *(step 4)* | opt-out vs opt-in placement |
| Liveness vs Readiness probes *(step 4)* | restart vs traffic-routing semantics |

**Authoring rule:** The question must force retrieval from both
pages. Reranker stress is intentional — questions where BM25 would
find one side but miss the other are the target. Expected sources:
2 pages minimum.

### multi_hop (5–6 slots)

Pool questions where the answer synthesizes 2–4 pages. These are
the primary reranker stressors.

| Page set (example) | Hop path |
|---|---|
| Pod + Service + Ingress *(step 4)* | How external traffic reaches a Pod through Service → Ingress |
| Deployment + ReplicaSet + Pod | How a Deployment rollout changes the underlying ReplicaSet and Pod set |
| ConfigMap + Deployment | How a ConfigMap update propagates to Pods via env vars or mounted volume |
| HPA + Deployment + Metrics Server *(partial step 4)* | How HPA reads metrics and scales a Deployment |
| NetworkPolicy + Pod + Namespace *(partial step 4)* | How NetworkPolicy selectors resolve across namespaces |
| Job + Pod + Container lifecycle *(partial step 4)* | How a Job's completions and parallelism interact with Pod restart policy |

**Authoring rule:** Expected sources ≥ 2 pages. The question must
not be answerable from any single page alone. `source_chunk_ids`
must list at least one chunk from each expected page; partial
credit is granted in the evaluator if at least one expected chunk is
cited (see `agent_bench/evaluation/harness.py`).

### false_premise (3–4 slots)

Pool questions whose premise is wrong. Split across two flavors:

**Flavor A — pure refusal** (at least 1 slot):
- Premise targets a capability that does not exist in the K8s corpus
  (not in any pulled page).
- Example seed: "How do I configure Claude API rate limits in a
  Kubernetes Deployment?" (wrong domain — Claude API is not a K8s
  concept)
- Schema: `category: "out_of_scope"`, `expected_sources: []`,
  `source_snippets: []`.
- Evaluator expectation: answer contains refusal phrasing AND cites
  zero sources.

**Flavor B — documented negative** (at least 1 slot, ideally 2):
- Corpus contains an explicit negative statement (e.g.
  NetworkPolicy "Anything TLS related" limitation at chunk 63 of
  `k8s_network_policies.md`).
- Example already in pilot: `k8s_pilot_005` (NetworkPolicy mTLS).
- Schema: `category: "retrieval"`, `question_type: "false_premise"`,
  `expected_sources: [<negative-answer page>]`,
  `source_snippets: [<verbatim negative statement>]`.
- Evaluator expectation: answer reports the documented negative
  with citation, does NOT open with "the documentation does not
  provide instructions" phrasing (per pilot_005 Fix 1 + Fix 2
  revert analysis).

**Other flavor-B candidate pages for authoring:**
- Pod Security Standards — explicit statements about what each
  profile does NOT permit
- RBAC Authorization — explicit statements about what RBAC does NOT
  provide (e.g. no deny rules)
- NetworkPolicy — additional negative clauses beyond the pilot_005
  mTLS one

### set / aggregation / post_processing_heavy (0–3 slots)

Include only if a K8s page naturally supports the pattern:

- `set`: "Which Kubernetes resources can expose a Service?" (answer
  is a set drawn from the Service page). Include 0–1 of this type
  if a clean example emerges; otherwise leave slot empty.
- `aggregation`: Unlikely to fit K8s docs (docs describe concepts,
  not tabular data). Likely leave empty.
- `post_processing_heavy`: Unlikely to fit K8s docs. Likely leave
  empty.

**Default:** Leave 0–3 as **0**. Only author these if a question
emerges organically during step 5. Do not force-author to hit a
target count; the plan explicitly says "0–3, included only where
corpus content naturally supports".

---

## `time_sensitive` flag placement (2–3 questions)

Flag questions whose correct answer depends on K8s version state:

| Candidate | Why time-sensitive |
|---|---|
| HPA API version | `autoscaling/v1` vs `autoscaling/v2` — v2 stable since 1.23 |
| Pod Security Admission stability | "stable as of v1.25" — feature state in the page |
| PodSecurityPolicy removal | PSP removed in 1.25; migration path to PSA |

**Authoring rule:** Set `time_sensitive: true` on exactly 2–3
questions. Distribute across ≥2 different CRAG types (e.g. one
`simple`, one `simple_w_condition`) so the flag is not concentrated
in a single type. Each `time_sensitive` question must cite a
specific K8s version or feature state in the source snippet,
otherwise the flag is not load-bearing.

---

## Difficulty distribution

Loose guidance, not a hard constraint:

- `easy`: 8–10 questions — mostly `simple` and single-page
  `simple_w_condition`
- `medium`: 10–12 questions — `comparison`, most `multi_hop`,
  straightforward `false_premise`
- `hard`: 4–6 questions — deep `multi_hop`, flavor-B `false_premise`,
  `time_sensitive` + `multi_hop` combinations

The pilot's 6-question set is all `easy`/`medium`. Step 5 should add
the `hard` tier.

---

## Authoring checklist (per question)

For each of the 25 questions, the step 5 author must fill:

| Field | Required | Notes |
|---|---|---|
| `id` | yes | `k8s_<NNN>` zero-padded (e.g. `k8s_001`) |
| `question` | yes | Natural-language question in the voice of a recruiter or developer |
| `expected_answer_keywords` | yes | 3–6 keywords that MUST appear in a correct answer; drives `keyword_hit_rate` |
| `expected_sources` | yes | List of `.md` filenames from `SOURCES.md`; ≥1 for scoped questions, `[]` for flavor-A false-premise |
| `category` | yes | `retrieval` / `calculation` / `out_of_scope` |
| `difficulty` | yes | `easy` / `medium` / `hard` |
| `requires_calculator` | yes | `false` for all K8s questions (no calc tool use expected) |
| `reference_answer` | yes | 1–3 sentence answer used by the optional LLM judge |
| `question_type` | yes | CRAG taxonomy value (exactly one of the 8 canonical strings) |
| `time_sensitive` | yes | `bool`; `true` on exactly 2–3 questions |
| `source_chunk_ids` | yes | Content-hashed chunk IDs (stable across reindex); must be `[]` for flavor-A false-premise |
| `source_snippets` | yes | ~20 words verbatim per chunk; drift-detection field |
| `source_pages` | yes | Human-readable page anchor (e.g. `"concepts/workloads/pods"`) |
| `source_sections` | yes | Deepest heading containing the snippet |

**Deprecation note:** The pilot schema has `is_multi_hop: bool`.
Step 5 may retire this field in favor of `question_type == "multi_hop"`,
but only after confirming the evaluator's partial-credit logic
(`agent_bench/evaluation/harness.py:38`) is updated to read from
`question_type`. Do NOT remove `is_multi_hop` without the
corresponding harness update, or existing pilot questions will
break partial-credit scoring.

---

## Pilot-first validation before step 5 authoring

Before writing the 25 questions, step 5 author must:

1. Confirm the 20 new pages from step 4 are ingested and reachable
   via the pipeline (smoke-query test per `SOURCES.md`'s post-ingest
   validation).
2. Re-run `make evaluate` on the existing 6-question pilot dataset
   against the newly-expanded corpus. The pilot's existing questions
   must still pass their per-question gates — if adding 20 new
   pages drops pilot P@5 materially, investigate before adding more
   questions on top.
3. Hand-draft 2–3 questions first, run them through the pipeline,
   and confirm retrieval surfaces the expected chunks. This is the
   final pilot-first checkpoint before bulk authoring.

Only after these three checks pass does the step 5 author proceed
to the full 25-question authoring session.
