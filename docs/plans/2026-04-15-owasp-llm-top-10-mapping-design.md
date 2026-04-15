# agent-bench — OWASP LLM Top 10 (2025) Mapping (Part A)

**Theme:** Honest documentation of OWASP LLM Top 10 (2025) coverage for the agent-bench implementation
**Estimated effort:** ~3.5 hours (plan budget 3–4 hours, hard cap 1 day)
**Scope:** Documentation-only. Four surfaces (SECURITY.md, DECISIONS.md entry, README tail, landing-page Security panel subtitle). Zero code changes to `agent_bench/security/*`, zero test changes, zero route changes.
**Plan reference:** `agent-bench v1.1 Implementation Plan` — Part A (OWASP LLM Top 10 (2025) Mapping). The 10-row mapping content and the mandatory language for LLM01/LLM02 verdict cells are locked by the plan; this design covers presentation, structure, cross-surface consistency, and execution order.

---

## Context

The plan's Part A deliverables are fully scoped:

1. `SECURITY.md` (~500–700 words) — honest mapping of the implementation against OWASP LLM Top 10 (2025), with named residual risks for LLM01 and scope limits for LLM02.
2. README one-line mention.
3. One sentence in the landing page.
4. DECISIONS.md entry.

The plan also locks the 10-verdict mapping table and specifies required language for the LLM01 and LLM02 verdict cells. What remains open and what this design resolves: presentation structure, placement on each surface, DECISIONS.md entry framing, cross-surface consistency discipline, execution order, and per-surface acceptance checks.

Real security code already exists in `agent_bench/security/` (two-tier injection detector, PII redactor, output validator with always-on secret-format deny list, audit logger), plus the rate limit middleware, bounded `ToolRegistry` (only `search` + `calculator`, no side-effectful tools), and orchestrator `max_iterations` cap. The existing README Security Architecture section already documents this pipeline with a diagram and four prose blocks. The DECISIONS.md file has entries for two-tier injection, regex-PII, JSONL audit, HMAC hashing, three-validators, no-auth, and monitor-mode validation — all available as evidence targets for cross-links in SECURITY.md.

---

## Locked decisions (from brainstorming)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | SECURITY.md uses a hybrid structure: 10-row × 2-column summary table at top (plain-text verdicts, LLM01/LLM02 rows carry their qualifier inline) + 8 H3 subsections under a single "Detailed mapping" H2 in LLM-numeric order. Out-of-scope LLM04/LLM08 live in the summary table only. | Dense 10×4 table fights the 500–700 word budget and makes mandatory LLM01/LLM02 language illegible in narrow cells. Per-category-subsection structure alone loses the scan-view that makes the 6+2+2=10 arithmetic verifiable at a glance. Hybrid gets both. |
| 2 | Uniform 3-part skeleton inside each H3: **Verdict** (one line) → **Implementation** (one paragraph with inline cross-links to `agent_bench/security/*.py` and DECISIONS.md anchors) → **Residual risk / Scope limit / Named gap** (one line, mandatory verbatim language for LLM01/LLM02). | Reader's eye learns the pattern by the second subsection; scanning becomes free. Inconsistency reads as compliance theater, which is the failure mode the brand is built to avoid. |
| 3 | Inline cross-links to source files and DECISIONS.md anchors in every Implementation cell. | Highest-leverage move for the "evidence not checklist" brand. A reviewer with the repo open should be able to click from claim to code in one hop. |
| 4 | Plain-text verdicts in the summary table (no badges, no checkmarks, no emoji). | "Addressed directly" / "Infrastructure layer" / "Out of scope" reads honest. Symbols would compress nicely but risk reading as security-badge theater, which the preamble disclaims. |
| 5 | Architecture-scope + OWASP Appendix 1 callout lives in the 1–2 paragraph preamble, not the summary table. Preamble references the README Security Architecture diagram rather than duplicating it. | Keeps the table clean for the 10×structure. Single source of truth for the architecture diagram (README), DECISIONS.md/SECURITY.md point at it. |
| 6 | LLM-numeric subsection ordering (LLM01 → LLM02 → LLM03 → LLM05 → LLM06 → LLM07 → LLM09 → LLM10), not strongest-first. | OWASP itself orders LLM01→LLM10; a reader cross-checking against the 2025 PDF expects this order. Strongest-first is exactly the framing compliance-theater docs adopt ("look at our best work first"), which the brand disclaims. |
| 7 | Single "Detailed mapping" H2 carrying all 8 subsections (not two H2s split by "Addressed directly" vs "Addressed at infrastructure layer"). | Two-H2 split implies the categorization is structurally important to the doc, but the actual categorization is carried by the verdict line inside each subsection and the summary table. Two H2s do redundant categorization at the cost of fragmenting the reader's scan. |
| 8 | DECISIONS.md entry framing: "Why named residual risks and scope limits, not 'fully mitigated' verdicts?" with explicit rejected-alternative naming in paragraph 1. | House-style paired thinking. Names the most interesting design choice (the discipline of refusing to say "mitigated" without a qualifier) and forces the body to cite OWASP's own language as the source of the discipline. |
| 9 | Responsibility split: SECURITY.md is the **what** (verdict cells); DECISIONS.md is the **why** (verdict discipline). One-way cross-link from DECISIONS.md → SECURITY.md with explicit split phrasing ("this entry covers why the verdict discipline takes the form it does"). | SECURITY.md = artifact reviewers read to evaluate the implementation; DECISIONS.md = artifact reviewers read to evaluate the thinking. Duplicating the 10-row mapping across both forces two sources of truth; first drift erodes the brand. One-way pointer goes the direction the why depends on the what but not vice versa. |
| 10 | README one-liner: joined into the existing Security Architecture section closing tail. Two drafting-time variants (single-sentence comma-list vs two-sentence equal-billing); selection by rhythm check at drafting time. | Preserves the existing four-item DECISIONS.md list (two-tier / regex-PII / JSONL / HMAC) which is load-bearing evidence that DECISIONS.md is substantive. Engineering Scope placement rejected because those bullets carry build artifacts, not documentation about build artifacts — miscategorization dilutes what the section is for. |
| 11 | Landing page one sentence: sub-text under `<h3>Security</h3>` in the dashboard right panel. Block-link (`<a>` wrapping the whole subtitle) with `aria-label="OWASP LLM Top 10 mapping in SECURITY.md"`, absolute GitHub URL, em-dash variant for mobile-wrap readability. | Only placement where the OWASP mention sits next to the guardrails it documents. New Finding card rejected (frame mismatch: Findings is "what the benchmark revealed about retrieval/orchestration," compliance mapping isn't that, plus résumé-theater failure mode on a recruiter-facing surface). Footer rejected (too quiet for an artifact a skim-reader should register). |
| 12 | LLM01 Implementation paragraph phrasing: "Deployments without GPU run Tier 1 only; the two-tier design degrades to heuristic-only rather than failing closed" (replaces weaker "by design" language). | Names the degradation behavior explicitly and uses "fails closed" security vocabulary that signals knowledge of the alternative posture. |
| 13 | Preamble scope-closing line: "These scope facts are referenced in the verdict cells below where they constrain what each guardrail must do." | Telegraphs that the scope statement ("no user ingestion, no fine-tuning, no authenticated sessions, no side-effectful tools") isn't decorative — it's the premise the verdicts depend on. Makes the doc read more like an argument and less like a checklist. |

---

## SECURITY.md structure (Section 1 of the design)

**Location:** top-level `/SECURITY.md` (GitHub auto-surfaces in the "Security" tab).

**Word budget:** 500–700 words, target ~620.

**Layout:**

```
# Security

<Preamble: 1 paragraph, ~60 words — honest mapping, scope statement>
<Architecture scope: 1 paragraph, ~70 words — OWASP Appendix 1 reference, pointer to README diagram, scope-closing line>

## Mapping summary
<10-row × 2-column table: Category | Verdict>

## Detailed mapping
### LLM01 Prompt Injection          (3-part skeleton)
### LLM02 Sensitive Information Disclosure (3-part skeleton)
### LLM03 Supply Chain              (3-part skeleton)
### LLM05 Improper Output Handling  (3-part skeleton)
### LLM06 Excessive Agency          (3-part skeleton)
### LLM07 System Prompt Leakage     (3-part skeleton)
### LLM09 Misinformation            (3-part skeleton)
### LLM10 Unbounded Consumption     (3-part skeleton)

## What this doc is not
<1 paragraph, ~50 words>
```

LLM04 and LLM08 are intentionally absent from the "Detailed mapping" H2 — they are "Out of scope" verdicts whose one-phrase treatment in the summary table is the entire treatment. Adding subsections for them would be compliance-theater padding.

### Preamble (structural draft; exact wording at drafting time)

> This document maps the agent-bench implementation against the OWASP LLM Applications Top 10 (2025). It is an honest mapping, not a coverage claim — every verdict cell for a guardrail we run carries a named residual risk or scope limit when OWASP's own 2025 text makes the limits explicit. The scope is a docs Q&A bot serving static curated corpora (FastAPI + Kubernetes) with no user ingestion, no fine-tuning, no authenticated sessions, and no side-effectful tools.

### Architecture-scope paragraph (structural draft)

> The implementation maps to the OWASP Appendix 1 reference architecture: user input → input guardrails → retrieval/tools → LLM → output guardrails → response. The agent-bench realization is diagrammed in [README Security Architecture section](README.md#security-architecture) — this document does not duplicate the diagram, and the verdict cells below cross-link to the specific source files that implement each guardrail. These scope facts are referenced in the verdict cells below where they constrain what each guardrail must do.

### Summary table (canonical; plain-text verdicts; LLM01/LLM02 qualifiers inline)

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

### H3 subsection skeleton

**2-part mandatory** (Verdict + Implementation) for every subsection. **3rd part mandatory** for the four categories where OWASP 2025 or the implementation scope makes an explicit limit (LLM01, LLM02, LLM03, LLM10). No invented 3rd part for LLM05, LLM06, LLM07, LLM09 — the honest-evaluation brand requires that we don't manufacture residual risks where OWASP's own text doesn't articulate them.

```
### LLMxx <Name>

**Verdict:** <one line>.

**Implementation:** <one paragraph with inline cross-links to source and DECISIONS.md>.

[optional third line — only for LLM01, LLM02, LLM03, LLM10]
**Residual risk:** <one line>.      ← LLM01 (OWASP says RAG does not fully mitigate)
**Scope limit:** <one line>.        ← LLM02 (narrower output-side scope than LLM02's four classes)
**Named gap:** <one line>.          ← LLM03, LLM10 (infrastructure layer, specific missing pieces)
```

LLM05, LLM06, LLM07, LLM09 use the 2-part skeleton only. Their verdicts read plain "Addressed directly" and their implementation paragraphs are the whole treatment — the guardrail is complete for the stated scope, and adding a synthetic "no residual risk" line would be reverse-compliance-theater (padding in the opposite direction).

### LLM01 Prompt Injection — canonical subsection (exact text)

> **Verdict:** Addressed directly with a named residual risk.
>
> **Implementation:** Two-tier detection — Tier 1 heuristic regex (local, <1ms) with ~20 pattern families covering role/identity hijacking, instruction override, system-prompt extraction, credential/env-var extraction, jailbreak keywords, and base64-nested payloads; Tier 2 optional DeBERTa classifier on Modal GPU for high-confidence arbitration. Deployments without GPU run Tier 1 only; the two-tier design degrades to heuristic-only rather than failing closed. Grounded refusal via the retrieval-threshold gate bounds indirect injection through retrieved content. Static corpus + bounded `ToolRegistry` (only `search` + `calculator`, no side-effectful tools) + `max_iterations` cap bound the blast radius. See [`agent_bench/security/injection_detector.py`](agent_bench/security/injection_detector.py), [`agent_bench/tools/registry.py`](agent_bench/tools/registry.py), and [DECISIONS.md § Why two-tier injection detection, not three](DECISIONS.md#why-two-tier-injection-detection-not-three).
>
> **Residual risk:** novel injection patterns not caught by heuristics or classifier. OWASP notes that RAG and fine-tuning do not fully mitigate prompt injection; indirect injection through retrieved content remains a core risk class.

### LLM02 Sensitive Information Disclosure — canonical subsection (exact text)

> **Verdict:** Addressed directly for the applicable scope.
>
> **Implementation:** Regex PII redaction on every retrieved chunk before it enters the LLM context window (EMAIL, SSN, CREDIT_CARD, PHONE, IP_ADDRESS) with optional spaCy NER for PERSON/ORG; post-generation output validation with an always-on secret-format deny list (OpenAI/Anthropic/Google/AWS/GitHub key prefixes, bearer tokens, env-var assignments) and URL-against-retrieved-chunks check. See [`agent_bench/security/pii_redactor.py`](agent_bench/security/pii_redactor.py), [`agent_bench/security/output_validator.py`](agent_bench/security/output_validator.py), and [DECISIONS.md § Why regex + optional spaCy for PII, not a cloud API](DECISIONS.md#why-regex--optional-spacy-for-pii-not-a-cloud-api).
>
> **Scope limit:** OWASP LLM02 spans access controls, training-data handling, user-consent transparency, and proprietary-information governance. This implementation addresses only response-time data surfaced to users; broader concerns would require architectural changes for multi-tenant or authenticated deployment.

### Remaining 6 subsections (content outline; drafted at implementation time)

- **LLM03 Supply Chain** → Infrastructure layer. Implementation: pinned deps (`pyproject.toml`), reproducible `Dockerfile`, official model sources (HuggingFace, Modal base images). **Named gap:** no formal SBOM, no signed model provenance.
- **LLM05 Improper Output Handling** → Addressed directly. Implementation: `OutputValidator` (PII leakage + secret deny-list + URL-against-retrieved + configurable blocklist); text-only output surface; no rendered HTML, no SQL generation, no executed code. Cross-link to `output_validator.py` and [DECISIONS.md § Why three output validators, not four](DECISIONS.md#why-three-output-validators-not-four).
- **LLM06 Excessive Agency** → Addressed directly. Implementation: `max_iterations` cap enforced in orchestrator; `ToolRegistry` contains only `search` and `calculator`; no side-effectful tools (no write, no network requests outside the corpus, no code execution). Cross-link to `agent_bench/tools/registry.py` and `agent_bench/agents/orchestrator.py`.
- **LLM07 System Prompt Leakage** → Addressed directly. Implementation: system prompt contains no credentials, no auth tokens, no multi-tenant role structure; access control enforced outside the LLM (rate limit middleware + future auth at infrastructure layer). Matches OWASP's own recommended pattern. Cross-link to [DECISIONS.md § Why no authentication on API endpoints](DECISIONS.md#why-no-authentication-on-api-endpoints).
- **LLM09 Misinformation** → Addressed directly. Implementation: grounded refusal via RRF retrieval-threshold gate — the project's core thesis and the source of the citation-accuracy=1.00 on API providers hero-tile number. Cross-link to [DECISIONS.md § Why a relevance threshold for grounded refusal](DECISIONS.md#why-a-relevance-threshold-for-grounded-refusal).
- **LLM10 Unbounded Consumption** → Infrastructure layer. Implementation: per-IP sliding-window rate limit (`RateLimitMiddleware`, 10 RPM default), `max_iterations` cap, provider timeouts via `ProviderTimeoutError` handling. **Named gap:** per-IP rate limit, not per-user cost ceiling; no budget cap per session.

### "What this doc is not" closing block (structural draft)

> This is an application-layer mapping for the scope of a static-corpus docs Q&A bot. It does not replace network-level security, authentication, infrastructure hardening, formal threat modeling, or a production security review. It does not constitute OWASP certification or a coverage guarantee — only an honest, evidence-linked mapping of the guardrails this implementation actually runs.

---

## Derived surfaces (Section 2 of the design)

All three surfaces are downstream of SECURITY.md canonical phrasing. Draft them in the execution order below.

### DECISIONS.md entry (~320 words, house style)

**Placement:** insert between "Why monitor mode for output validation, not gating?" (~line 334) and "Why additive SSE stage events?" (~line 336). Topical clustering with the other security-phase entries; not chronological file-end append.

**Header:** `## Why named residual risks and scope limits, not "fully mitigated" verdicts?`

**Body (structural draft; tighten paragraphs 2 and 3 by collapsing OWASP-restatement into verdict-explanation, target ~320 words total):**

Paragraph 1 — Named rejected alternative: the 10-row table could have been written as uniform "addressed" verdicts without qualifiers, arithmetically identical (same 6+2+2=10). Rejected because OWASP's own 2025 text is explicit about what an input guardrail can and cannot do, and writing a verdict that contradicts the source the mapping cites would be compliance theater.

Paragraph 2 — LLM01: OWASP 2025 states that RAG and fine-tuning do not fully mitigate prompt injection, and that indirect injection through retrieved content remains a core risk class. "Fully mitigated" is unsupportable for any system retrieving untrusted content into an LLM context window. Verdict reads "addressed directly with named residual risk"; residual-risk cell cites OWASP's own "do not fully mitigate" language verbatim.

Paragraph 3 — LLM02: OWASP 2025 defines LLM02 as spanning four concern classes — access controls, training-data handling, user-consent transparency, and proprietary-information governance. This implementation addresses a narrower output-side subset (regex PII redaction on retrieved chunks + output validation for PII leakage, secret formats, and URL hallucination) — not cleanly one of the four concern classes, but a narrower scope than any of them. Verdict reads "addressed directly for the applicable scope"; scope-limit cell enumerates the four concern classes verbatim and names what addressing the broader concerns would require (multi-tenant or authenticated architecture).

Paragraph 4 — Thesis (load-bearing, do not cut): the tension the entry resolves is honesty-vs-scannability. A mapping surfacing named residual risks and scope limits is longer and harder to skim than one with uniform "addressed" verdicts, but the scannable version over-claims relative to the cited source. Honest evaluation is the brand. Every verdict cell in SECURITY.md must survive a reviewer reading OWASP 2025 in a second tab.

Closing — Cross-link with explicit responsibility-split phrasing: "See [SECURITY.md § LLM01](SECURITY.md#llm01-prompt-injection) and [§ LLM02](SECURITY.md#llm02-sensitive-information-disclosure) for the verdict cells; this entry covers why the verdict discipline takes the form it does." Plus canonical-phrasing-discipline closing: the LLM01 "do not fully mitigate" phrasing and the LLM02 four-concern-class enumeration are canonical in SECURITY.md; the README tail link and the landing-page Security panel subtitle paraphrase this language but must preserve the named-residual-risk and scope-limit structure. A future edit that drifts one surface without the others creates a brand inconsistency visible in cross-surface diff.

**Framing discipline:** this entry uses **failure-mode framing** ("URL hallucination", "PII leakage") — describes what the guardrail prevents. SECURITY.md uses **mechanism framing** ("URL-against-retrieved-chunks check", "PIIRedactor on every retrieved chunk") — describes what the guardrail runs. The split is deliberate: SECURITY.md documents behavior, DECISIONS.md documents intent. A future edit must not swap the framings across surfaces.

### README one-liner (two variants; rhythm check at drafting time)

**Placement:** joined into the Security Architecture section closing tail (README.md line ~202), preserving the existing four-item DECISIONS.md list. No change to any other README section.

**Variant 1 (single sentence, comma-list tail):**
> See [SECURITY.md](SECURITY.md) for the OWASP LLM Top 10 (2025) mapping, and [DECISIONS.md](DECISIONS.md) for why we chose two-tier detection over three, regex-only PII by default, JSONL over SQLite for audit, and HMAC over plain SHA-256 for IP hashing.

**Variant 2 (two sentences, equal billing):**
> See [SECURITY.md](SECURITY.md) for the OWASP LLM Top 10 (2025) mapping. See [DECISIONS.md](DECISIONS.md) for why we chose two-tier detection over three, regex-only PII by default, JSONL over SQLite for audit, and HMAC over plain SHA-256 for IP hashing.

**Selection rule:** paste both into README in actual surrounding context at drafting time, pick on rhythm. Variant 2 wins if the section's closing cadence feels heavy; Variant 1 wins if the closing cadence is tight. Plan-wording compliance ("one-line mention") satisfied under either reading.

### Landing page Security panel subtitle

**HTML change** — insert one `<a>` element between `<h3>Security</h3>` and `<div class="security-badges">` in `agent_bench/serving/static/index.html` (~line 311):

```html
<div class="security-panel">
  <h3>Security</h3>
  <a class="security-panel-sub"
     href="https://github.com/tyy0811/agent-bench/blob/main/SECURITY.md"
     target="_blank"
     aria-label="OWASP LLM Top 10 mapping in SECURITY.md">
    Mapped against the OWASP LLM Top 10 (2025) &mdash; named residual risks for LLM01, scope limits for LLM02 &rarr; SECURITY.md
  </a>
  <div class="security-badges">
    ...
  </div>
</div>
```

**Block-link rationale:** larger click target, accessible, the whole subtitle semantically *is* the link. `aria-label` gives screen-reader users the same one-line semantic ("this is a link to the OWASP mapping doc") that sighted readers get — otherwise screen readers parse the em-dash and arrow awkwardly.

**Absolute GitHub URL rationale:** on HF Spaces, relative `SECURITY.md` would not resolve (FastAPI app only serves `index.html` + static assets). Matches the existing finding-card link pattern. Per the URL-reference-window memory, no rename of GitHub or HF URLs is planned during the application-reference window, so the absolute link is stable.

**CSS addition** — new rule, new class (`.security-panel-sub`) to avoid colliding with the existing `.sec-sub` class (which is already scoped inside `.sec-badge`):

```css
.security-panel-sub {
  display: block;
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.4;
  margin-top: -4px;
  margin-bottom: 10px;
  text-decoration: none;
}
.security-panel-sub:hover {
  color: var(--text);
  text-decoration: underline;
}
```

**Drafting-time check on `margin-top: -4px`:** negative margin is brittle under h3 font-size changes. Prefer adjusting h3 bottom-margin explicitly if the negative margin reads wrong in the actual rendered panel. Not a blocker; verify during drafting against the actual rendered panel.

**Mobile wrap check:** at 375px viewport width with `0.78rem` font size, the em-dash variant wraps cleaner than the comma-list version because the dash gives the parser a natural break point. Verify via browser devtools responsive mode during drafting.

---

## Canonical-phrasing discipline (Section 3.1)

One cross-surface constraint, enforced at pre-commit verification:

| Language fragment | Canonical in | Paraphrased in | Audit check |
|---|---|---|---|
| OWASP "do not fully mitigate" verbatim | SECURITY.md LLM01 Residual-risk cell | DECISIONS.md entry paragraph 2 (quoted) | Exact-string grep during pre-merge verification |
| LLM02 four-concern-class enumeration (access controls / training-data handling / user-consent transparency / proprietary-information governance) | SECURITY.md LLM02 Scope-limit cell | DECISIONS.md entry paragraph 3 (listed) | Exact list membership, same four items |
| "Named residual risks for LLM01, scope limits for LLM02" structural phrase | Landing-page Security panel subtitle | — (README tail is a pointer surface — refers readers to SECURITY.md but does not itself carry the structural phrase, so it is not audited for this phrase) | Structural grep: both "named residual risk" and "scope limit" appear in the landing-page subtitle |
| Mechanism framing (what runs) vs failure-mode framing (what it prevents) | SECURITY.md (mechanism) / DECISIONS.md (failure-mode) | — | Framings must not swap |

A drift — e.g., SECURITY.md's LLM01 cell losing the verbatim "do not fully mitigate" phrase, or DECISIONS.md's paragraph 3 dropping one of the four concern classes — is detectable in one diff pass.

---

## Execution order (Section 3.2)

**Strict order.** Each downstream surface paraphrases upstream phrasing; writing out of order risks pinning a downstream surface to a draft that never lands.

```
Step 1 ──▶ SECURITY.md (canonical source of all mandatory language)
Step 2 ──▶ DECISIONS.md entry (derives cross-link targets from SECURITY.md anchors)
Step 3 ──▶ README tail (derives the "OWASP LLM Top 10 (2025) mapping" pointer from SECURITY.md preamble)
Step 4 ──▶ Landing-page Security panel subtitle (derives the structural phrase from SECURITY.md verdict cells)
Step 5 ──▶ Cross-surface verification (canonical-phrasing table, per-surface acceptance, plan compliance)
```

**Time budget (3.5 hours inside plan's 3–4 hour soft budget; hard cap 1 day):**

| Step | Est. time | Notes |
|---|---|---|
| 1. SECURITY.md (~620 words, 8 subsections) | 90 min | Most writing happens here; LLM01/LLM02 cells take longest because exact language is prescribed |
| 2. DECISIONS.md entry (~320 words, tightened) | 45 min | Draft at ~390 words, trim paragraphs 2 and 3 by collapsing OWASP-restatement into verdict-explanation, lock paragraph 4 intact |
| 3. README tail (two variants, rhythm check) | 15 min | Both variants pasted in context, rhythm check, pick one |
| 4. Landing page HTML + CSS (one element + one rule) | 30 min | Insertion + CSS; start dev server and verify rendered subtitle at desktop + 375px viewport; aria-label manual check via browser devtools |
| 5. Cross-surface verification | 30 min | Canonical-phrasing table, per-surface acceptance, `make test` + `make lint`, commit |
| **Total** | **~3.5 hours** | Inside the 3–4 hour soft budget; 30-min slack pools at the end of Step 5 to absorb drafting-time surprises without touching the hard cap |

Slack is pooled, not spread across steps, because drafting-time surprises tend to cluster (one surface goes long, the others go normally).

---

## Per-surface acceptance checks (Section 3.3)

Exhaustive checks; cheap to run, load-bearing when missed.

### SECURITY.md

- [ ] Word count between 500 and 700
- [ ] 10-row summary table with 6 + 2 + 2 = 10 arithmetic visible in a single scan
- [ ] Summary table LLM01 row reads "Addressed directly, named residual risk"
- [ ] Summary table LLM02 row reads "Addressed directly, scope limit"
- [ ] Single "Detailed mapping" H2 contains exactly 8 subsections — LLM01, LLM02, LLM03, LLM05, LLM06, LLM07, LLM09, LLM10 — in that order. LLM04 and LLM08 (out of scope) are intentionally absent from this section; a future reader should recognize this as design intent, not omission.
- [ ] Each subsection follows the skeleton: 2-part mandatory (Verdict + Implementation) for all 8; 3rd part mandatory for LLM01 (Residual risk), LLM02 (Scope limit), LLM03 (Named gap), LLM10 (Named gap); LLM05/LLM06/LLM07/LLM09 use 2-part only with no synthetic 3rd line
- [ ] LLM01 subsection contains the exact phrase: "OWASP notes that RAG and fine-tuning do not fully mitigate prompt injection; indirect injection through retrieved content remains a core risk class"
- [ ] LLM01 subsection contains: "Deployments without GPU run Tier 1 only; the two-tier design degrades to heuristic-only rather than failing closed"
- [ ] LLM02 subsection enumerates all four OWASP LLM02 concern classes (access controls, training-data handling, user-consent transparency, proprietary-information governance)
- [ ] All 8 subsections contain at least one cross-link to `agent_bench/security/*.py`, `agent_bench/tools/registry.py`, or `agent_bench/agents/orchestrator.py`
- [ ] At least 5 subsections contain a DECISIONS.md anchor cross-link
- [ ] Preamble contains the scope statement ("no user ingestion, no fine-tuning, no authenticated sessions, no side-effectful tools") plus the closing line ("These scope facts are referenced in the verdict cells below where they constrain what each guardrail must do")
- [ ] Preamble references README Security Architecture diagram without duplicating it
- [ ] "What this doc is not" closing block present, ~50 words
- [ ] No occurrence of "fully mitigated", "coverage guarantee", "tests all 10", "OWASP-certified", or any other overclaiming phrase (per cross-cutting requirement #3 of the plan)

### DECISIONS.md entry

- [ ] Inserted between "Why monitor mode for output validation, not gating?" and "Why additive SSE stage events?"
- [ ] Header exactly: `## Why named residual risks and scope limits, not "fully mitigated" verdicts?`
- [ ] Word count ~320 ± 30
- [ ] Paragraph 1 names the rejected alternative (uniform "addressed" verdicts)
- [ ] Paragraph 2 quotes OWASP "do not fully mitigate" language (not paraphrased)
- [ ] Paragraph 3 enumerates all four LLM02 concern classes (same four as SECURITY.md canonical)
- [ ] Paragraph 4 (thesis) intact — honesty-vs-scannability tension + "honest evaluation is the brand"
- [ ] Cross-link to `SECURITY.md § LLM01` and `§ LLM02` with explicit responsibility-split phrasing ("this entry covers why the verdict discipline takes the form it does")
- [ ] Closing lines name the canonical-phrasing discipline and the cross-surface-diff detection mechanism
- [ ] Uses failure-mode framing ("URL hallucination", "PII leakage"), not mechanism framing

### README tail

- [ ] One of the two variants selected by rhythm check at drafting time
- [ ] Preserves the existing four-item DECISIONS.md list (two-tier / regex-PII / JSONL / HMAC) intact
- [ ] Link uses relative `SECURITY.md` (README renders on GitHub; relative resolves)
- [ ] No change to any other section of README — no new bullet in Engineering Scope, no new row in V1→V2→V3 table, no tagline edit

### Landing page subtitle

- [ ] One `<a class="security-panel-sub">` element inserted between `<h3>Security</h3>` and `<div class="security-badges">`
- [ ] Element is a block-link (whole subtitle wrapped in `<a>`)
- [ ] `href` is the absolute GitHub URL: `https://github.com/tyy0811/agent-bench/blob/main/SECURITY.md`
- [ ] `aria-label="OWASP LLM Top 10 mapping in SECURITY.md"` present
- [ ] `target="_blank"` present (matches existing finding-card link convention)
- [ ] Subtitle text uses em-dash variant: "Mapped against the OWASP LLM Top 10 (2025) — named residual risks for LLM01, scope limits for LLM02 → SECURITY.md"
- [ ] New `.security-panel-sub` CSS class added (not the existing `.sec-sub` which is scoped to badges)
- [ ] Dev server started (`make serve`); subtitle verified to render at desktop (1440px) + narrow (375px) viewport widths
- [ ] Wrap behavior at 375px verified clean (em-dash gives the natural break; no mid-phrase wraps)
- [ ] `margin-top: -4px` drafting-time check: prefer explicit h3 bottom-margin adjustment if the negative margin reads brittle in the actual rendered panel
- [ ] No change to any other element — Security badges panel, pipeline stages, retrieval panel, chat panel, findings cards all untouched

---

## Whole-task pre-commit verification (Section 3.4)

```
a. Read all four surfaces end-to-end in one sitting, in execution order.
b. Run the canonical-phrasing audit table (§ Canonical-phrasing discipline) —
   each row is a grep or exact-string check.
c. Verify cross-link targets resolve:
   - SECURITY.md anchors exist (LLM01 Prompt Injection, LLM02 Sensitive
     Information Disclosure, etc. — GitHub anchor-slugs match)
   - Source file cross-links point to existing files (agent_bench/security/
     injection_detector.py, agent_bench/security/pii_redactor.py,
     agent_bench/security/output_validator.py, agent_bench/tools/registry.py,
     agent_bench/agents/orchestrator.py)
   - DECISIONS.md anchor cross-links (two-tier injection, regex+spaCy, no
     authentication, three output validators, relevance threshold) — existing
     entries, anchor-slugs match
d. Run `make test` — confirm no regressions. No test changes expected;
   the landing-page HTML change is additive and the existing static-route
   tests assert on content presence, not structure.
e. Run `make lint` — confirm no new ruff/mypy issues (no Python changes,
   so this is a consistency check, not a content check).
f. Start `make serve`, open localhost:8000 in a browser, verify the
   subtitle renders as a muted sub-heading, clicking it opens the GitHub
   SECURITY.md in a new tab. Test narrow viewport via browser devtools
   responsive mode at 375px width.
g. Final brand-check: no overclaiming phrases anywhere across the four
   surfaces.
```

---

## Scope guardrails (Section 3.5)

- **No code changes to `agent_bench/security/*`.** Every guardrail mentioned in SECURITY.md already exists — this task documents behavior, it does not modify it.
- **No test changes.** No new security tests, no assertion updates. The existing 444-test suite is the baseline.
- **No changes to `agent_bench/serving/routes.py`**, no changes to `agent_bench/serving/app.py`, no changes to the security pipeline wiring.
- **No new routes, no new endpoints, no new SSE events.**
- **No DECISIONS.md edits outside the one new entry.** No retroactive phrasing updates to existing security-phase entries. A drafting-time observation of conflict with an existing entry is logged as a follow-up, not silently patched in this commit.
- **No landing-page changes outside the Security panel subtitle.** The pipeline stages, retrieval panel, findings cards, tiles, request log, and footer are untouched.
- **No Hugging Face Spaces metadata updates.** Per the HF-deploy reference memory, the Spaces README frontmatter is managed separately; this task does not cross that boundary.
- **No README changes outside the Security Architecture section's closing tail link.** No new bullets, no new tables, no new rows in V1→V2→V3, no tagline edits, no LinkedIn-prep rephrasing.

---

## Drafting-gap protocol (hybrid rule)

When drafting SECURITY.md reveals a mismatch between documented behavior and actual code, the resolution depends on which kind of gap it is:

**Phrasing / precision gap → proceed.** The code does what the doc describes; the doc just uses different vocabulary. Default: rename in the doc to match the code, log nothing, keep going. Example: SECURITY.md draft says "always-on secret-format deny list" and the code exposes the same mechanism as `_SECRET_PATTERNS` — rename in the doc to match, proceed.

**Behavioral gap → stop and surface.** The code does something the doc would mischaracterize. Log to a follow-up file (not a silent fix), decide outside the time budget whether to patch the doc or the code, then resume. Example: SECURITY.md draft says "regex PII redaction on every retrieved chunk before it enters the LLM context window" and drafting-time code review reveals redaction runs only on chunks above a relevance threshold — that's a different security claim, stop, log, decide later.

The honest-evaluation brand requires this distinction. Defaulting "fix the doc to match the code" on a behavioral gap is exactly the failure mode that produces SECURITY.md docs that don't survive a reviewer with the repo open. Defaulting "stop and surface" on a phrasing gap is exactly the failure mode that blows the time budget on drafting-time bikeshedding.

---

## Plan-compliance pointer

Plan compliance is verified at pre-commit per § Whole-task pre-commit verification; the canonical-phrasing table (§ Canonical-phrasing discipline) and per-surface acceptance checks (§ Per-surface acceptance checks) jointly cover every locked requirement in the plan's Part A section and every applicable cross-cutting requirement (#2 DECISIONS.md growth, #3 no-overclaiming, #4 interview-prep benefit).
