# agent-bench — LLM Security Hardening

**Theme:** Production-grade guardrails for agentic RAG systems
**Estimated effort:** 4–5 days
**Compute:** CPU locally + Modal GPU for classifier model

---

## Design Decisions (pre-implementation)

Five simplifications made during design review:

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Drop Tier 2 embedding similarity | General-purpose encoder (all-MiniLM-L6-v2) can't distinguish semantic similarity from intent similarity. "How do I ignore a field in Pydantic?" clusters near "ignore previous instructions" — threshold tuning would be perpetual. Two-tier (heuristic → classifier) is cleaner. |
| 2 | Make spaCy optional for PII | Regex covers high-risk PII (SSNs, credit cards, emails, phones). spaCy NER on technical text produces false positives ("FastAPI" as ORG, "Jordan" as PERSON). Optional import with graceful fallback + logged warning. |
| 3 | Drop `/admin/audit` query endpoint | Project has zero auth. Building API key auth for one endpoint while `/ask` remains open is inconsistent. JSONL + `jq` is how production audit logs actually get queried. |
| 4 | Drop length/format output check | Calculator returns short answers. Tech docs contain code blocks and JSON. "Suspiciously short" threshold would false-positive on day one. Keep three deterministic validators only. |
| 5 | Drop SQLite audit backend | No query endpoint consuming it. One storage codepath, one format. JSONL imports trivially into SQLite/DuckDB if queryability is needed later. |

---

## Features

### 1A. Prompt Injection Detection

Pre-retrieval guard that classifies user inputs as safe or potentially adversarial before they enter the RAG pipeline.

**Module:** `agent_bench/security/injection_detector.py`

**Two-tier detection:**

- **Tier 1 — Heuristic rules** (zero latency, runs locally): regex patterns for common injection signatures (`ignore previous instructions`, `you are now`, `system:`, role-switching patterns, base64-encoded payloads)
- **Tier 2 — DeBERTa classifier** (Modal GPU): fine-tuned `deepset/deberta-v3-base-injection` deployed as a serverless endpoint on Modal. Called only when Tier 1 doesn't match but input has characteristics worth checking (configurable). Modal cold-start is acceptable — Tier 1 handles the fast path, Tier 2 is the high-confidence arbiter.

**Returns:** `SecurityVerdict` dataclass:
```python
@dataclass
class SecurityVerdict:
    safe: bool
    tier: str           # "heuristic" | "classifier"
    confidence: float   # 1.0 for heuristic matches, model score for classifier
    matched_pattern: str | None  # regex pattern name for tier 1
```

**Configurable action on detection:** `block` (return 403 with explanation), `warn` (proceed but tag the audit log), or `flag` (proceed silently, log only)

**Configurable tier depth:** `tiers: [heuristic, classifier]` — deployments without GPU can run heuristic-only, which is honest and documented.

**Integration:** Wire into `/ask` and `/ask/stream` endpoints as middleware, before retrieval.

**Modal deployment:**

```python
# modal/injection_classifier.py
@app.cls(gpu="T4", image=image)
class InjectionClassifier:
    @modal.enter()
    def load(self):
        self.pipe = pipeline("text-classification",
                             model="deepset/deberta-v3-base-injection",
                             device="cuda")

    @modal.method()
    def classify(self, text: str) -> dict:
        result = self.pipe(text)[0]
        return {"label": result["label"], "score": result["score"]}
```

**Fallback story:** Without Modal/GPU → heuristic-only detection. Documented, not hidden.

**Test plan:**
- ~30 known injection prompts (Gandalf, HackAPrompt datasets)
- ~30 benign prompts including edge cases ("how do I ignore a field in Pydantic?", questions about security topics)
- Precision/recall report per tier
- Latency: Tier 1 local vs Tier 2 Modal round-trip
- Target: ≥0.85 precision (low false-positive rate matters more than recall for UX)

**Estimated effort:** 1.5–2 days

---

### 1B. PII Redaction in Retrieved Context

Post-retrieval, pre-generation filter that detects and masks PII in retrieved chunks before they enter the LLM context window.

**Module:** `agent_bench/security/pii_redactor.py`

**Detection methods:**
- **Regex-based (always active):** email addresses, phone numbers (international formats), SSNs, credit card patterns, IP addresses
- **NER (optional, off by default):** spaCy `en_core_web_sm` for PERSON, ORG, GPE entities. Requires `pip install spacy && python -m spacy download en_core_web_sm`. Graceful fallback if not installed:

```python
try:
    import spacy
    _NER_AVAILABLE = True
except ImportError:
    _NER_AVAILABLE = False

class PIIRedactor:
    def __init__(self, config: PIIConfig):
        self.use_ner = config.use_ner and _NER_AVAILABLE
        if config.use_ner and not _NER_AVAILABLE:
            logger.warning("pii.use_ner=true but spaCy not installed, falling back to regex-only")
```

**Redaction strategy:** Replace detected spans with typed placeholders (`[EMAIL_1]`, `[PERSON_2]`) — preserves answer coherence while removing PII. Placeholder mapping is deterministic within a request (same entity → same placeholder).

**Configuration:** Integrated into AppConfig via Pydantic:
```yaml
security:
  pii:
    enabled: true
    mode: redact          # redact | detect_only | passthrough
    redact_patterns:      # regex-based, always available
      - EMAIL
      - PHONE
      - SSN
      - CREDIT_CARD
      - IP_ADDRESS
    use_ner: false         # requires spaCy, off by default
    ner_entities:          # which spaCy entities to redact (if use_ner=true)
      - PERSON
```

**Integration:** Runs after FAISS+BM25+RRF+reranker, before context is assembled into LLM prompt.

**Returns metadata:** `{redactions_count: int, types_found: list[str]}` — surfaced in audit log.

**Test plan:**
- Synthetic documents with known PII patterns (all regex types)
- Verify redaction preserves answer coherence
- Verify placeholder determinism within a request
- Test both code paths: regex-only and regex+NER (NER tested in CI with spaCy in test deps)

**Estimated effort:** 1 day

---

### 1C. Structured Audit Logging

Append-only audit trail recording the full query → retrieval → generation → response chain for every request.

**Module:** `agent_bench/security/audit_logger.py`

**Log schema** (one JSON record per request):
```json
{
  "request_id": "uuid",
  "timestamp": "ISO-8601",
  "session_id": "str | null",
  "client_ip": "str (SHA-256 hashed)",
  "endpoint": "/ask",
  "input_query": "str",
  "injection_verdict": {"safe": true, "tier": "heuristic", "confidence": 0.98},
  "retrieved_chunks": ["doc_id_1", "doc_id_2"],
  "retrieval_scores": [0.87, 0.74],
  "pii_redactions": {"count": 2, "types": ["EMAIL"]},
  "llm_provider": "anthropic",
  "llm_model": "claude-haiku-4-5-20251001",
  "output_tokens": 342,
  "output_validation": {"passed": true, "violations": []},
  "grounded_refusal": false,
  "response_latency_ms": 1240,
  "error": null
}
```

**Storage:** JSONL only (`logs/audit.jsonl`). One codepath, one format.

**IP hashing:** SHA-256 hash client IPs before logging. Never store raw IPs. GDPR-aligned.

**Log rotation:** Configurable max file size, auto-rotate with timestamp suffix.

**Queryability:** Standard tools, not a custom endpoint:
```bash
# Find all requests where injection detection fired
jq 'select(.injection_verdict.safe == false)' logs/audit.jsonl

# Count PII redactions by type over the last 24h
jq 'select(.timestamp > "2025-03-30") | .pii_redactions.types[]' logs/audit.jsonl | sort | uniq -c

# Trace a full request chain by session
jq 'select(.session_id == "abc123")' logs/audit.jsonl
```

**Test plan:**
- Integration test: full pipeline request → verify audit record has all fields
- Verify IP hashing is irreversible (no raw IPs in any log)
- Test log rotation at configured size
- Test concurrent writes don't corrupt JSONL

**Estimated effort:** 1 day

---

### 1D. Output Validation Gate

Post-generation check that inspects LLM response before returning to user.

**Module:** `agent_bench/security/output_validator.py`

**Three deterministic checks:**

1. **PII leakage:** Run the same PII redactor (1B) on the generated response. If the LLM reconstructed PII that was redacted from context, block or redact. Reuses `PIIRedactor` — no new code.
2. **URL validation:** Any URLs in the response must appear in the retrieved chunks. Extends existing grounded-refusal logic. Prevents URL hallucination.
3. **Blocklist scan:** Configurable list of terms/patterns that should never appear in output (system prompt fragments, API key patterns, internal identifiers).

**Returns:** `OutputVerdict` dataclass:
```python
@dataclass
class OutputVerdict:
    passed: bool
    violations: list[str]
    action: str  # "pass" | "redact" | "block"
```

**On block:** Return generic safe response explaining output was filtered. Log violation in audit trail.

**Test plan:**
- PII leakage: inject PII into mock LLM response, verify caught
- URL hallucination: mock response with URL not in retrieved chunks, verify flagged
- Blocklist: inject system prompt fragment, verify caught
- Clean responses pass with negligible overhead

**Estimated effort:** 0.5–1 day

---

## Security Pipeline

```
User Input
    │
    ▼
┌──────────────────────┐
│  Injection Detection  │  Tier 1: heuristic regex (local, <1ms)
│  (pre-retrieval)      │  Tier 2: DeBERTa classifier (Modal GPU)
└──────────┬───────────┘
           │ safe
           ▼
┌──────────────────────┐
│  Retrieval            │  FAISS + BM25 + RRF + cross-encoder
│  (existing pipeline)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  PII Redaction        │  regex (always) + spaCy NER (optional)
│  (post-retrieval)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LLM Generation       │  OpenAI / Anthropic / vLLM (Modal)
│  (existing pipeline)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Output Validation    │  PII leakage + URL check + blocklist
│  (post-generation)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Audit Log            │  JSONL, IP-hashed, rotated
│  (every request)      │
└──────────┬───────────┘
           │
           ▼
       Response
```

---

## Configuration

All security config integrates into the existing Pydantic `AppConfig` system:

```yaml
# configs/default.yaml (additions)
security:
  injection:
    enabled: true
    action: block              # block | warn | flag
    tiers:
      - heuristic
      - classifier             # remove to run heuristic-only (no GPU)
    classifier_url: ""         # Modal endpoint URL, set via env var
  pii:
    enabled: true
    mode: redact               # redact | detect_only | passthrough
    redact_patterns: [EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS]
    use_ner: false
    ner_entities: [PERSON]
  output:
    enabled: true
    pii_check: true
    url_check: true
    blocklist: []              # patterns that must never appear in output
  audit:
    enabled: true
    path: logs/audit.jsonl
    max_size_mb: 100
    rotate: true
```

---

## New Dependencies

| Package | Purpose | Runs on | Required? |
|---------|---------|---------|-----------|
| `transformers` | DeBERTa injection classifier | Modal (T4 GPU) | No (Modal only) |
| `spacy` + `en_core_web_sm` | NER for PII detection | Local (CPU) | No (opt-in) |

All other features use stdlib (`re`, `hashlib`, `json`, `uuid`, `dataclasses`). Minimal local dependency footprint is deliberate.

---

## DECISIONS.md Additions

- **Why two-tier injection detection, not three:** Heuristics are fast and deterministic. DeBERTa classifier is the high-confidence arbiter. The embedding similarity middle tier was cut because a general-purpose encoder can't distinguish semantic similarity from intent similarity — the threshold between "ambiguous" and "suspicious" is an untunable hyperparameter. Two tiers degrade gracefully: without GPU, you get heuristic-only, which is honest and documented.
- **Why regex + optional spaCy for PII, not a cloud API:** Cost, latency, data residency. Regex covers the PII types with actual legal/compliance risk (SSNs, credit cards, emails). spaCy NER false-positive rate on technical text is unacceptable without domain tuning — kept optional with graceful fallback.
- **Why append-only JSONL for audit:** Simplicity, no external dependencies, compliance-friendly. One codepath, one format. JSONL imports trivially into SQLite/DuckDB — no bridges burned.
- **Why IP hashing:** GDPR alignment. SHA-256 is irreversible. Never store raw IPs.
- **Why Modal for the classifier:** Serverless GPU, no infra to manage, consistent with existing vLLM deployment pattern.
- **Why no audit query endpoint:** Project has zero auth. Building API key auth for one endpoint while `/ask` is open creates an inconsistency. `jq` on structured JSONL is how production audit logs get queried.
- **Why three output validators, not four:** Length/format sanity check false-positives on calculator answers (short) and tech doc responses (code blocks). The three remaining checks are deterministic with clear pass/fail semantics.

---

## README Section

A **Security Architecture** section will be added to README.md with the pipeline diagram and a summary of the guardrail design.

---

## Estimated Effort

| Feature | Effort |
|---------|--------|
| 1A. Injection Detection (heuristic + Modal classifier) | 1.5–2 days |
| 1B. PII Redaction (regex + optional NER) | 1 day |
| 1C. Audit Logging (JSONL, IP-hashed) | 1 day |
| 1D. Output Validation (3 checks) | 0.5–1 day |
| **Total** | **4–5 days** |
