# Security

This document maps the agent-bench implementation against the OWASP LLM Applications Top 10 (2025). It is an honest mapping, not a coverage claim — every verdict cell for a guardrail we run carries a named residual risk or scope limit when OWASP's own 2025 text makes the limits explicit. The scope is a docs Q&A bot serving static curated corpora (FastAPI + Kubernetes) with no user ingestion, no fine-tuning, no authenticated sessions, and no side-effectful tools.

The implementation maps to the OWASP Appendix 1 reference architecture: user input → input guardrails → retrieval/tools → LLM → output guardrails → response. The agent-bench realization is diagrammed in the [README Security Architecture section](README.md#security-architecture); verdict cells below cross-link to the source files that implement each guardrail.

## Mapping summary

| Category | Verdict |
|---|---|
| LLM01 Prompt Injection | Addressed directly, named residual risk |
| LLM02 Sensitive Information Disclosure | Addressed directly, scope limit |
| LLM03 Supply Chain | Infrastructure layer, named gap |
| LLM04 Data and Model Poisoning | Out of scope |
| LLM05 Improper Output Handling | Addressed directly |
| LLM06 Excessive Agency | Addressed directly |
| LLM07 System Prompt Leakage | Addressed directly |
| LLM08 Vector and Embedding Weaknesses | Out of scope |
| LLM09 Misinformation | Addressed directly |
| LLM10 Unbounded Consumption | Infrastructure layer, named gap |

**Counts (verifiable in one eye-scan):** 6 addressed directly + 2 infrastructure layer + 2 out of scope = **10**.

## Detailed mapping

### LLM01 Prompt Injection

**Verdict:** Addressed directly with a named residual risk.

**Implementation:** Two-tier detection — Tier 1 heuristic regex (<1ms) covering role/identity hijacking, instruction override, system-prompt extraction, credential extraction, jailbreak keywords, and base64-nested payloads; Tier 2 optional DeBERTa classifier on Modal GPU for arbitration. Deployments without GPU run Tier 1 only; the two-tier design degrades to heuristic-only rather than failing closed. Grounded refusal via the retrieval-threshold gate bounds indirect injection. Static corpus + bounded `ToolRegistry` (only `search` + `calculator`, no side-effectful tools) + `max_iterations` cap bound blast radius. See [`agent_bench/security/injection_detector.py`](agent_bench/security/injection_detector.py), [`agent_bench/tools/registry.py`](agent_bench/tools/registry.py), and [DECISIONS.md § Why two-tier injection detection, not three](DECISIONS.md#why-two-tier-injection-detection-not-three).

**Residual risk:** novel injection patterns not caught by heuristics or classifier. OWASP notes that RAG and fine-tuning do not fully mitigate prompt injection; indirect injection through retrieved content remains a core risk class.

### LLM02 Sensitive Information Disclosure

**Verdict:** Addressed directly for the applicable scope.

**Implementation:** Regex PII redaction on every retrieved chunk (EMAIL, SSN, CREDIT_CARD, PHONE, IP_ADDRESS) with optional spaCy NER for PERSON/ORG; post-generation output validation with an always-on secret-format deny list (API key prefixes, bearer tokens, env-var assignments) and URL-against-retrieved-chunks check. See [`agent_bench/security/pii_redactor.py`](agent_bench/security/pii_redactor.py), [`agent_bench/security/output_validator.py`](agent_bench/security/output_validator.py), and [DECISIONS.md § Why regex + optional spaCy for PII, not a cloud API](DECISIONS.md#why-regex--optional-spacy-for-pii-not-a-cloud-api).

**Scope limit:** OWASP LLM02 spans access controls, training-data handling, user-consent transparency, and proprietary-information governance. This implementation addresses only response-time data surfaced to users — a narrower, output-side subset that does not cleanly map to any single one of the four; broader concerns would require architectural changes for multi-tenant or authenticated deployment.

### LLM03 Supply Chain

**Verdict:** Addressed at the infrastructure layer with a named gap.

**Implementation:** Dependencies pinned in [`pyproject.toml`](pyproject.toml); reproducible container via [`Dockerfile`](Dockerfile); models from official upstreams (HuggingFace, Modal) loaded in [`agent_bench/security/injection_detector.py`](agent_bench/security/injection_detector.py). No user-uploaded models, no third-party marketplaces.

**Named gap:** no SBOM, no signed model provenance; production would add signature checks at model ingestion.

### LLM05 Improper Output Handling

**Verdict:** Addressed directly.

**Implementation:** [`OutputValidator`](agent_bench/security/output_validator.py) runs four checks per response: PII detection, secret-format deny list, URL validation against retrieved chunks, configurable blocklist. Text-only output — no HTML rendering, no SQL, no code execution. See also [DECISIONS.md § Why three output validators, not four](DECISIONS.md#why-three-output-validators-not-four).

### LLM06 Excessive Agency

**Verdict:** Addressed directly.

**Implementation:** `max_iterations` cap in the [orchestrator](agent_bench/agents/orchestrator.py) bounds tool-use depth. [`ToolRegistry`](agent_bench/tools/registry.py) contains only `search_documents` (read-only) and `calculator` (pure arithmetic) — no write tools, no network, no code execution.

### LLM07 System Prompt Leakage

**Verdict:** Addressed directly.

**Implementation:** System prompt holds no credentials, no auth tokens, no multi-tenant structure — docs-Q&A instruction with corpus-label substitution only. [`RateLimitMiddleware`](agent_bench/serving/middleware.py) provides per-IP abuse protection; production layers authentication at the reverse-proxy. See also [DECISIONS.md § Why no authentication on API endpoints](DECISIONS.md#why-no-authentication-on-api-endpoints).

### LLM09 Misinformation

**Verdict:** Addressed directly.

**Implementation:** RRF retrieval-threshold gate — below `refusal_threshold`, the orchestrator emits a grounded refusal instead of synthesizing. Benchmarked on 27 FastAPI + 25 Kubernetes golden questions (citation-accuracy=1.00). See [DECISIONS.md § Why a relevance threshold for grounded refusal](DECISIONS.md#why-a-relevance-threshold-for-grounded-refusal).

### LLM10 Unbounded Consumption

**Verdict:** Addressed at the infrastructure layer with a named gap.

**Implementation:** Per-IP rate limit via [`RateLimitMiddleware`](agent_bench/serving/middleware.py) (10 req/min, configurable); `max_iterations` cap on the agent loop; provider timeouts via `ProviderTimeoutError` / `ProviderRateLimitError`.

**Named gap:** per-IP only, no per-user quota, no session budget ceiling. Production would add per-user tracking and a hard cost ceiling at the billing layer.

## What this doc is not

This is an application-layer mapping for the scope of a static-corpus docs Q&A bot. It does not replace network-level security, authentication, infrastructure hardening, formal threat modeling, or a production security review. It does not constitute OWASP certification or a coverage guarantee — only an honest, evidence-linked mapping of the guardrails this implementation actually runs.
