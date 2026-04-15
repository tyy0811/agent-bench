# Part A — OWASP LLM Top 10 (2025) Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the locked Part A design (`docs/plans/2026-04-15-owasp-llm-top-10-mapping-design.md`) across four surfaces — `SECURITY.md`, one new `DECISIONS.md` entry, one README tail line, one landing-page Security panel subtitle — producing an honest, evidence-linked mapping of the agent-bench implementation against the OWASP LLM Applications Top 10 (2025).

**Architecture:** Documentation-only. Four surfaces with a named canonical-phrasing discipline: `SECURITY.md` is canonical for all OWASP-cited language (LLM01 "do not fully mitigate" verbatim, LLM02 four-concern-class enumeration, LLM02 "narrower output-side subset" disclaimer); the other three surfaces paraphrase but preserve the named-residual-risk and scope-limit structure. Execution is strictly ordered — each downstream surface paraphrases upstream phrasing, so drafting out of order risks pinning a downstream surface to a draft that never lands. A **paired-review gate** fires between Task 2 (DECISIONS.md entry) and Task 3 (README tail) to satisfy cross-cutting requirement #9 from the v1.1 plan; Jane is the second reviewer and the DECISIONS.md commit does not land until the gate passes.

**Tech Stack:** Markdown (`SECURITY.md`, `DECISIONS.md`, `README.md`), HTML + CSS (`agent_bench/serving/static/index.html`). No Python code changes. No test changes. `make test` + `make lint` used only as regression checks in Task 5.

**Time budget:** ~3.5 hours inside the plan's 3–4 hour soft budget (hard cap 1 day). Slack is pooled at Task 5 to absorb drafting-time surprises.

**Branch:** `part-a/owasp-mapping` (already created, design doc already committed at `9a16483`).

---

## Scope guardrails (read before touching anything)

These constraints come from the design doc § Scope guardrails. They are non-negotiable for this plan.

- **No code changes to `agent_bench/security/*`.** Every guardrail mentioned in SECURITY.md already exists; this plan documents behavior, it does not modify it.
- **No test changes.** The existing 444-test suite is the baseline; no new security tests, no assertion updates.
- **No changes to `agent_bench/serving/routes.py`**, no changes to `agent_bench/serving/app.py`, no changes to the security-pipeline wiring.
- **No new routes, no new endpoints, no new SSE events.**
- **No DECISIONS.md edits outside the one new entry in Task 2.** A drafting-time observation of conflict with an existing entry is logged as a follow-up, not silently patched.
- **No landing-page changes outside the Security panel subtitle added in Task 4.** The pipeline stages, retrieval panel, findings cards, tiles, request log, and footer are untouched.
- **No Hugging Face Spaces metadata updates.** The Spaces README frontmatter is managed separately; this plan does not cross that boundary.
- **No README changes outside the Security Architecture section closing-tail line.** No new bullets, no new tables, no new rows in V1→V2→V3, no tagline edits.

## Drafting-gap protocol

If drafting reveals a mismatch between documented behavior and actual code:

- **Phrasing / precision gap → proceed.** Rename in the doc to match the code, log nothing, keep going. Example: SECURITY.md draft says "always-on secret-format deny list"; code has `_SECRET_PATTERNS` — rename in doc, proceed.
- **Behavioral gap → stop and surface.** Log to a follow-up note, decide outside the time budget whether to patch the doc or the code, then resume. Example: SECURITY.md draft says "PII redaction on every chunk"; code review reveals redaction only runs above a threshold — stop, log, decide later.

The budget-vs-canonical-content drift on SECURITY.md word count (flagged above, ~790 words vs 500–700 target) is a **precision gap** — proceed.

---

## Task 1: Create SECURITY.md

**Files:**
- Create: `/Users/zenith/Desktop/agent-bench/SECURITY.md` (top-level; GitHub auto-surfaces in "Security" tab)

**Estimated time:** ~90 minutes. Most of the plan's writing time lives here.

**Task context:** SECURITY.md is the canonical source for all OWASP-cited language. Every downstream surface (DECISIONS.md, README, landing page) paraphrases phrasing introduced here. The file is written in strict structural order — preamble → summary table → Detailed mapping H2 with 8 subsections in LLM-numeric order → "What this doc is not" closing. LLM04 and LLM08 (out of scope) appear only in the summary table, not as subsections — adding subsections for them would be compliance-theater padding.

**Exit criteria:**
- File exists at `/Users/zenith/Desktop/agent-bench/SECURITY.md`
- Word count target 500–700, ~790 acceptable (canonical content drift documented)
- Summary table has 10 rows with 6 + 2 + 2 = 10 arithmetic visible at a glance
- 8 H3 subsections under single `## Detailed mapping` H2, in order: LLM01, LLM02, LLM03, LLM05, LLM06, LLM07, LLM09, LLM10
- LLM01 subsection contains exact phrase: "OWASP notes that RAG and fine-tuning do not fully mitigate prompt injection; indirect injection through retrieved content remains a core risk class"
- LLM01 subsection contains exact phrase: "Deployments without GPU run Tier 1 only; the two-tier design degrades to heuristic-only rather than failing closed"
- LLM02 subsection enumerates all four OWASP concern classes (access controls, training-data handling, user-consent transparency, proprietary-information governance)
- LLM02 subsection contains the clarifying clause: "a narrower, output-side subset that does not cleanly map to any single one of the four"
- LLM01, LLM02, LLM03, LLM10 subsections use 3-part skeleton (Verdict + Implementation + Residual risk / Scope limit / Named gap)
- LLM05, LLM06, LLM07, LLM09 subsections use 2-part skeleton only (Verdict + Implementation; no synthetic 3rd line)
- All 8 subsections contain inline cross-links to source files in `agent_bench/` and at least 5 contain DECISIONS.md anchor cross-links
- No occurrence of overclaiming phrases: "fully mitigated", "coverage guarantee", "tests all 10", "OWASP-certified", "OWASP certification"
- Committed with message `docs(security): add SECURITY.md with OWASP LLM Top 10 (2025) mapping`

---

- [ ] **Step 1.1: Create file with preamble paragraph**

Create `/Users/zenith/Desktop/agent-bench/SECURITY.md` with the following opening content:

```markdown
# Security

This document maps the agent-bench implementation against the OWASP LLM Applications Top 10 (2025). It is an honest mapping, not a coverage claim — every verdict cell for a guardrail we run carries a named residual risk or scope limit when OWASP's own 2025 text makes the limits explicit. The scope is a docs Q&A bot serving static curated corpora (FastAPI + Kubernetes) with no user ingestion, no fine-tuning, no authenticated sessions, and no side-effectful tools.
```

- [ ] **Step 1.2: Add architecture-scope paragraph**

Append to `SECURITY.md`:

```markdown

The implementation maps to the OWASP Appendix 1 reference architecture: user input → input guardrails → retrieval/tools → LLM → output guardrails → response. The agent-bench realization is diagrammed in the [README Security Architecture section](README.md#security-architecture) — this document does not duplicate the diagram, and the verdict cells below cross-link to the specific source files that implement each guardrail. These scope facts are referenced in the verdict cells below where they constrain what each guardrail must do.
```

- [ ] **Step 1.3: Add summary table**

Append to `SECURITY.md`:

```markdown

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
```

Verify the arithmetic sentence is present and correct by scanning the Verdict column: "Addressed directly" should appear on rows LLM01, LLM02, LLM05, LLM06, LLM07, LLM09 (6 rows); "Infrastructure layer" on LLM03 and LLM10 (2 rows); "Out of scope" on LLM04 and LLM08 (2 rows).

- [ ] **Step 1.4: Add "Detailed mapping" H2 header**

Append to `SECURITY.md`:

```markdown

## Detailed mapping
```

- [ ] **Step 1.5: Add LLM01 Prompt Injection subsection (canonical text)**

Append to `SECURITY.md` — **exact text, do not paraphrase or tighten the Residual risk line**:

```markdown

### LLM01 Prompt Injection

**Verdict:** Addressed directly with a named residual risk.

**Implementation:** Two-tier detection — Tier 1 heuristic regex (local, <1ms) with ~20 pattern families covering role/identity hijacking, instruction override, system-prompt extraction, credential/env-var extraction, jailbreak keywords, and base64-nested payloads; Tier 2 optional DeBERTa classifier on Modal GPU for high-confidence arbitration. Deployments without GPU run Tier 1 only; the two-tier design degrades to heuristic-only rather than failing closed. Grounded refusal via the retrieval-threshold gate bounds indirect injection through retrieved content. Static corpus + bounded `ToolRegistry` (only `search` + `calculator`, no side-effectful tools) + `max_iterations` cap bound the blast radius. See [`agent_bench/security/injection_detector.py`](agent_bench/security/injection_detector.py), [`agent_bench/tools/registry.py`](agent_bench/tools/registry.py), and [DECISIONS.md § Why two-tier injection detection, not three](DECISIONS.md#why-two-tier-injection-detection-not-three).

**Residual risk:** novel injection patterns not caught by heuristics or classifier. OWASP notes that RAG and fine-tuning do not fully mitigate prompt injection; indirect injection through retrieved content remains a core risk class.
```

- [ ] **Step 1.6: Add LLM02 Sensitive Information Disclosure subsection (canonical text with new clause)**

Append to `SECURITY.md` — **exact text, the "narrower, output-side subset that does not cleanly map to any single one of the four" clause is mandatory and resolves the cross-surface drift flagged in the design self-review**:

```markdown

### LLM02 Sensitive Information Disclosure

**Verdict:** Addressed directly for the applicable scope.

**Implementation:** Regex PII redaction on every retrieved chunk before it enters the LLM context window (EMAIL, SSN, CREDIT_CARD, PHONE, IP_ADDRESS) with optional spaCy NER for PERSON/ORG; post-generation output validation with an always-on secret-format deny list (OpenAI/Anthropic/Google/AWS/GitHub key prefixes, bearer tokens, env-var assignments) and URL-against-retrieved-chunks check. See [`agent_bench/security/pii_redactor.py`](agent_bench/security/pii_redactor.py), [`agent_bench/security/output_validator.py`](agent_bench/security/output_validator.py), and [DECISIONS.md § Why regex + optional spaCy for PII, not a cloud API](DECISIONS.md#why-regex--optional-spacy-for-pii-not-a-cloud-api).

**Scope limit:** OWASP LLM02 spans access controls, training-data handling, user-consent transparency, and proprietary-information governance. This implementation addresses only response-time data surfaced to users — a narrower, output-side subset that does not cleanly map to any single one of the four; broader concerns would require architectural changes for multi-tenant or authenticated deployment.
```

- [ ] **Step 1.7: Add LLM03 Supply Chain subsection (3-part skeleton, infrastructure layer)**

Append to `SECURITY.md`:

```markdown

### LLM03 Supply Chain

**Verdict:** Addressed at the infrastructure layer with a named gap.

**Implementation:** Python dependencies pinned in [`pyproject.toml`](pyproject.toml); reproducible container image via [`Dockerfile`](Dockerfile); models sourced from official upstreams (HuggingFace for the DeBERTa injection classifier and the cross-encoder reranker, Modal base images for GPU serving). No user-uploaded models, no third-party model marketplaces.

**Named gap:** no formal SBOM, no signed model provenance. A production deployment would add explicit signature checks at model ingestion.
```

- [ ] **Step 1.8: Add LLM05 Improper Output Handling subsection (2-part skeleton)**

Append to `SECURITY.md`:

```markdown

### LLM05 Improper Output Handling

**Verdict:** Addressed directly.

**Implementation:** [`OutputValidator`](agent_bench/security/output_validator.py) runs four deterministic checks on every generated response: PII leakage detection (reuses `PIIRedactor` in detect-only mode), always-on secret-format deny list, URL validation against retrieved chunks, and configurable blocklist. Output surface is text-only — no rendered HTML, no SQL generation, no code execution. See also [DECISIONS.md § Why three output validators, not four](DECISIONS.md#why-three-output-validators-not-four).
```

- [ ] **Step 1.9: Add LLM06 Excessive Agency subsection (2-part skeleton)**

Append to `SECURITY.md`:

```markdown

### LLM06 Excessive Agency

**Verdict:** Addressed directly.

**Implementation:** `max_iterations` cap enforced in the [orchestrator](agent_bench/agents/orchestrator.py) bounds tool-use loop depth. The [`ToolRegistry`](agent_bench/tools/registry.py) contains only `search_documents` (read-only retrieval against the static corpus) and `calculator` (pure arithmetic evaluation) — no write tools, no external network, no code execution, no shell. Tool definitions are declared at app startup and are not mutable at request time.
```

- [ ] **Step 1.10: Add LLM07 System Prompt Leakage subsection (2-part skeleton)**

Append to `SECURITY.md`:

```markdown

### LLM07 System Prompt Leakage

**Verdict:** Addressed directly.

**Implementation:** The system prompt contains no credentials, no auth tokens, and no multi-tenant role structure — it is a docs-Q&A instruction with corpus-label variable substitution only. Access control is enforced outside the LLM: [`RateLimitMiddleware`](agent_bench/serving/middleware.py) provides per-IP abuse protection, and a production deployment would layer authentication at the reverse-proxy or API-gateway level. This matches OWASP's recommended pattern (controls outside the LLM, not in the prompt). See also [DECISIONS.md § Why no authentication on API endpoints](DECISIONS.md#why-no-authentication-on-api-endpoints).
```

- [ ] **Step 1.11: Add LLM09 Misinformation subsection (2-part skeleton)**

Append to `SECURITY.md`:

```markdown

### LLM09 Misinformation

**Verdict:** Addressed directly.

**Implementation:** Grounded refusal via the RRF retrieval-threshold gate — the project's core thesis. When the top retrieved chunk's RRF score falls below the per-corpus `refusal_threshold`, the orchestrator emits a grounded refusal (no hallucinated citation, no invented answer) rather than synthesizing. This mechanism is the source of the citation-accuracy=1.00 hero-tile number on API providers and is benchmarked across 27 FastAPI + 25 Kubernetes golden questions. See also [DECISIONS.md § Why a relevance threshold for grounded refusal](DECISIONS.md#why-a-relevance-threshold-for-grounded-refusal).
```

- [ ] **Step 1.12: Add LLM10 Unbounded Consumption subsection (3-part skeleton, infrastructure layer)**

Append to `SECURITY.md`:

```markdown

### LLM10 Unbounded Consumption

**Verdict:** Addressed at the infrastructure layer with a named gap.

**Implementation:** Per-IP sliding-window rate limit via [`RateLimitMiddleware`](agent_bench/serving/middleware.py) (10 requests per minute by default, configurable); `max_iterations` cap on the agent loop; provider timeouts enforced via `ProviderTimeoutError` / `ProviderRateLimitError` handling in the request middleware.

**Named gap:** the rate limit is per-IP, not per-user; there is no budget cap per session and no global cost ceiling. A production deployment would add per-user quota tracking and a hard cost ceiling at the infrastructure or billing layer.
```

- [ ] **Step 1.13: Add "What this doc is not" closing block**

Append to `SECURITY.md`:

```markdown

## What this doc is not

This is an application-layer mapping for the scope of a static-corpus docs Q&A bot. It does not replace network-level security, authentication, infrastructure hardening, formal threat modeling, or a production security review. It does not constitute OWASP certification or a coverage guarantee — only an honest, evidence-linked mapping of the guardrails this implementation actually runs.
```

- [ ] **Step 1.14: Verify word count**

Run:

```bash
wc -w /Users/zenith/Desktop/agent-bench/SECURITY.md
```

Expected: a number in the range `500`–`800`. Target was 500–700; ~790 is the planned ceiling due to canonical LLM01/LLM02 content density. If the count is >800, tighten LLM03/05/06/07/09/10 implementation paragraphs by removing redundant phrases (not by cutting cross-links). If the count is <500, the canonical content is missing something — re-check steps 1.5 and 1.6.

- [ ] **Step 1.15: Grep-verify LLM01 canonical phrases**

Run each of these and confirm each returns a hit:

```bash
grep -c "do not fully mitigate prompt injection" /Users/zenith/Desktop/agent-bench/SECURITY.md
grep -c "indirect injection through retrieved content remains a core risk class" /Users/zenith/Desktop/agent-bench/SECURITY.md
grep -c "degrades to heuristic-only rather than failing closed" /Users/zenith/Desktop/agent-bench/SECURITY.md
```

Expected: each command returns `1`. If any returns `0`, the canonical LLM01 text is missing or has been inadvertently paraphrased — re-check step 1.5 and restore the exact text.

- [ ] **Step 1.16: Grep-verify LLM02 canonical phrases**

Run:

```bash
grep -c "access controls, training-data handling, user-consent transparency, and proprietary-information governance" /Users/zenith/Desktop/agent-bench/SECURITY.md
grep -c "narrower, output-side subset that does not cleanly map to any single one of the four" /Users/zenith/Desktop/agent-bench/SECURITY.md
```

Expected: each returns `1`. If either returns `0`, re-check step 1.6.

- [ ] **Step 1.17: Grep-verify 8 subsections in numeric order**

Run:

```bash
grep -n "^### LLM" /Users/zenith/Desktop/agent-bench/SECURITY.md
```

Expected output (exactly 8 lines, in this order):

```
### LLM01 Prompt Injection
### LLM02 Sensitive Information Disclosure
### LLM03 Supply Chain
### LLM05 Improper Output Handling
### LLM06 Excessive Agency
### LLM07 System Prompt Leakage
### LLM09 Misinformation
### LLM10 Unbounded Consumption
```

If LLM04 or LLM08 appears as an `###` header, delete it — they belong in the summary table only. If any subsection is out of order, reorder.

- [ ] **Step 1.18: Grep-verify no overclaiming phrases**

Run:

```bash
grep -iE "fully mitigated|coverage guarantee|tests all 10|OWASP.certified|OWASP certification" /Users/zenith/Desktop/agent-bench/SECURITY.md
```

Expected: no output (exit code 1). If any match fires, remove the offending phrase. **Note:** "OWASP certification" appears legitimately in the "What this doc is not" closing block as `"does not constitute OWASP certification"` — that's a disclaimer, not an overclaim. The grep above uses the literal phrase without negation, so it will match the disclaimer; inspect any hit for whether it is a disclaimer (good) or a claim (bad). Rephrase to "does not constitute an OWASP Top 10 certification" if the grep-based audit needs the phrase itself absent.

If the inspection reveals only the disclaimer hit, step passes.

- [ ] **Step 1.19: Verify cross-links resolve to existing files**

Run:

```bash
ls /Users/zenith/Desktop/agent-bench/agent_bench/security/injection_detector.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/security/pii_redactor.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/security/output_validator.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/tools/registry.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/agents/orchestrator.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/serving/middleware.py \
   /Users/zenith/Desktop/agent-bench/pyproject.toml \
   /Users/zenith/Desktop/agent-bench/Dockerfile
```

Expected: all 8 files listed (no "No such file or directory" errors). If any file is missing, the cross-link in SECURITY.md is dead — either fix the path or remove the link.

Also verify DECISIONS.md anchors exist for each `DECISIONS.md#...` cross-link in SECURITY.md:

```bash
grep -E "^## (Why two-tier injection detection|Why regex \+ optional spaCy|Why three output validators|Why no authentication|Why a relevance threshold)" /Users/zenith/Desktop/agent-bench/DECISIONS.md
```

Expected: 5 lines, one for each cited DECISIONS.md section. If any is missing, the anchor is dead — inspect the DECISIONS.md header for the exact title and update the SECURITY.md cross-link.

- [ ] **Step 1.20: Commit**

```bash
cd /Users/zenith/Desktop/agent-bench
git add SECURITY.md
git commit -m "$(cat <<'EOF'
docs(security): add SECURITY.md with OWASP LLM Top 10 (2025) mapping

Honest mapping of the agent-bench implementation against the OWASP
LLM Applications Top 10 (2025). Part A of the v1.1 implementation
plan.

Structure: 1-2 paragraph preamble + 10-row summary table with
6+2+2=10 arithmetic + single "Detailed mapping" H2 with 8 H3
subsections in LLM-numeric order (LLM01, LLM02, LLM03, LLM05,
LLM06, LLM07, LLM09, LLM10 — LLM04 and LLM08 intentionally absent
from Detailed mapping as they are out-of-scope per the summary
table) + "What this doc is not" closing.

LLM01 carries a named residual risk citing OWASP's own "do not
fully mitigate" language. LLM02 carries a scope limit enumerating
all four OWASP concern classes and naming the narrower output-side
subset this implementation actually addresses. LLM03 and LLM10
carry named infrastructure-layer gaps. LLM05/06/07/09 use the
2-part skeleton without a synthetic 3rd line.

All 8 subsections cross-link to source files in agent_bench/ and
at least 5 cross-link to DECISIONS.md anchors.

Design doc: docs/plans/2026-04-15-owasp-llm-top-10-mapping-design.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -1
```

Expected: commit lands, `git log --oneline -1` shows the new commit SHA with `docs(security): add SECURITY.md with OWASP LLM Top 10 (2025) mapping` message.

---

## Task 2: Add DECISIONS.md entry + paired-review gate

**Files:**
- Modify: `/Users/zenith/Desktop/agent-bench/DECISIONS.md` (insert one new entry between existing sections)

**Estimated time:** ~45 minutes drafting + ~10–15 minutes paired-review gate.

**Task context:** The DECISIONS.md entry is the **why** artifact; SECURITY.md is the **what** artifact. The entry is inserted topically clustered with the other security-phase entries (not appended at file end). House-style paired thinking: name the rejected alternative, cite OWASP verbatim, explain the tension, close with a cross-link that uses the explicit responsibility-split phrasing. Uses **failure-mode framing** ("URL hallucination", "PII leakage") not mechanism framing (SECURITY.md holds the mechanism framing). Target ~320 words.

**The paired-review gate** (satisfies cross-cutting #9 of the v1.1 plan) fires after the entry is drafted and before its commit lands. Jane reviews with the OWASP 2025 PDF open in a second tab and confirms three checks. Commit does not land until the gate passes.

**Exit criteria:**
- DECISIONS.md has a new entry inserted between "Why monitor mode for output validation, not gating?" and "Why additive SSE stage events?"
- Entry header is exactly `## Why named residual risks and scope limits, not "fully mitigated" verdicts?`
- Entry word count ~320 ± 30
- Paragraph 1 names the rejected alternative (uniform "addressed" verdicts)
- Paragraph 2 quotes OWASP "do not fully mitigate" language verbatim (not paraphrased)
- Paragraph 3 enumerates all four LLM02 concern classes and contains the "narrower output-side subset" disclaim
- Paragraph 4 (thesis) intact — "honesty-vs-scannability" + "honest evaluation is the brand"
- Closing contains cross-link with explicit responsibility-split phrasing ("this entry covers why the verdict discipline takes the form it does")
- Closing contains the canonical-phrasing discipline note
- Uses failure-mode framing ("URL hallucination", "PII leakage"), not mechanism framing
- **Paired-review gate passed** — Jane has confirmed three OWASP citation checks
- Committed with message `docs(decisions): add entry on named residual risks and scope limits verdict discipline`

---

- [ ] **Step 2.1: Identify insertion point in DECISIONS.md**

Run:

```bash
grep -n "^## Why monitor mode for output validation, not gating?" /Users/zenith/Desktop/agent-bench/DECISIONS.md
grep -n "^## Why additive SSE stage events?" /Users/zenith/Desktop/agent-bench/DECISIONS.md
```

Expected: two line numbers, one for each header. The new entry goes between them — specifically, after the last line of the "Why monitor mode" section (before the next `## ` header). Record the line number of the `## Why additive SSE stage events?` header for the insertion point.

- [ ] **Step 2.2: Draft paragraph 1 (rejected alternative)**

In a scratch buffer or directly into DECISIONS.md at the insertion point, draft:

```markdown
## Why named residual risks and scope limits, not "fully mitigated" verdicts?

The OWASP LLM Top 10 (2025) mapping could have been written as a 10-row table where LLM01 and LLM02 read as "addressed" without qualifiers — shorter, cleaner-looking, and arithmetically identical to the current version in the sense that both produce the same 6+2+2=10 category counts. Rejected because OWASP's own 2025 text is explicit about what an input guardrail can and cannot do, and writing a verdict that contradicts the source the mapping cites would be compliance theater.
```

- [ ] **Step 2.3: Draft paragraph 2 (LLM01 — OWASP verbatim)**

Append to the entry:

```markdown

LLM01 Prompt Injection — OWASP 2025 states that RAG and fine-tuning do not fully mitigate prompt injection, and that indirect injection through retrieved content remains a core risk class. "Fully mitigated" is unsupportable for any system retrieving untrusted content into an LLM context window, which is every RAG system including this one. The LLM01 verdict reads "addressed directly with named residual risk"; the residual-risk cell cites OWASP's own "do not fully mitigate" language verbatim.
```

- [ ] **Step 2.4: Draft paragraph 3 (LLM02 — four-concern-class enumeration + narrower-subset disclaim)**

Append to the entry:

```markdown

LLM02 Sensitive Information Disclosure — OWASP 2025 defines LLM02 as spanning four concern classes: access controls, training-data handling, user-consent transparency, and proprietary-information governance. This implementation addresses a narrower output-side subset (regex PII redaction on retrieved chunks + output validation for PII leakage, secret formats, and URL hallucination) — not cleanly one of the four concern classes, but a narrower scope than any of them. The verdict reads "addressed directly for the applicable scope"; the scope-limit cell enumerates the four concern classes verbatim and names what addressing the broader concerns would require (multi-tenant or authenticated architecture).
```

- [ ] **Step 2.5: Draft paragraph 4 (thesis — load-bearing, do not cut on tightening)**

Append to the entry:

```markdown

The tension the entry resolves is honesty-vs-scannability: a mapping that surfaces named residual risks and scope limits is longer and harder to skim than one with uniform "addressed" verdicts, but the scannable version over-claims relative to the cited source. Honest evaluation is the brand. Every verdict cell in SECURITY.md must survive a reviewer reading OWASP 2025 in a second tab.
```

- [ ] **Step 2.6: Draft closing (cross-link + canonical-phrasing discipline)**

Append to the entry:

```markdown

See [SECURITY.md § LLM01 Prompt Injection](SECURITY.md#llm01-prompt-injection) and [§ LLM02 Sensitive Information Disclosure](SECURITY.md#llm02-sensitive-information-disclosure) for the verdict cells; this entry covers why the verdict discipline takes the form it does. The LLM01 "do not fully mitigate" phrasing and the LLM02 four-concern-class enumeration are canonical in SECURITY.md; the README tail link and the landing-page Security panel subtitle paraphrase this language but must preserve the named-residual-risk and scope-limit structure. A future edit that drifts one surface without the others creates a brand inconsistency visible in cross-surface diff.
```

- [ ] **Step 2.7: Verify word count and tighten if over**

Run:

```bash
awk '/^## Why named residual risks and scope limits/,/^## Why additive SSE stage events/' /Users/zenith/Desktop/agent-bench/DECISIONS.md | wc -w
```

Expected: between 290 and 360 words (target ~320). If over 360, tighten paragraphs 2 and 3 by collapsing the OWASP-restatement sentence into the verdict-explanation sentence — **do not cut paragraph 4, it is the thesis and load-bearing**. If under 290, the entry is under-developed; re-check that all four paragraphs + closing are present.

- [ ] **Step 2.8: Verify failure-mode framing (not mechanism framing)**

Run:

```bash
awk '/^## Why named residual risks and scope limits/,/^## Why additive SSE stage events/' /Users/zenith/Desktop/agent-bench/DECISIONS.md | grep -cE "URL hallucination|PII leakage"
awk '/^## Why named residual risks and scope limits/,/^## Why additive SSE stage events/' /Users/zenith/Desktop/agent-bench/DECISIONS.md | grep -cE "URL-against-retrieved-chunks check|PIIRedactor on every retrieved chunk"
```

Expected: first command returns `≥1` (failure-mode phrases present), second command returns `0` (mechanism phrases absent). If the second returns `≥1`, the entry has accidentally adopted SECURITY.md's mechanism framing — rewrite the offending sentence in failure-mode terms.

- [ ] **Step 2.9: Verify insertion placement**

Run:

```bash
grep -A 1 "^## Why named residual risks and scope limits" /Users/zenith/Desktop/agent-bench/DECISIONS.md | head -3
grep -B 2 "^## Why named residual risks and scope limits" /Users/zenith/Desktop/agent-bench/DECISIONS.md | head -3
```

Expected: the header immediately before the new entry is `## Why monitor mode for output validation, not gating?` (or content from that section); the header immediately after is `## Why additive SSE stage events?`. If the new entry is at file end or in the wrong cluster, move it.

- [ ] **Step 2.10: PAIRED-REVIEW GATE — pause and surface to Jane**

**Do NOT commit until this gate passes.** This satisfies cross-cutting requirement #9 of the v1.1 plan.

Surface the drafted entry to Jane with the OWASP LLM Top 10 (2025) PDF open in a second tab: https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf

Jane runs three checks:

1. **LLM01 verbatim**: the phrase "RAG and fine-tuning do not fully mitigate prompt injection" (or close equivalent expressing the same claim) appears in the OWASP 2025 PDF's LLM01 section, AND the phrase "indirect injection through retrieved content" is a supportable description of a core risk class named in LLM01.
2. **LLM02 four-concern-class enumeration**: the OWASP 2025 PDF's LLM02 section identifies access controls, training-data handling, user-consent transparency, and proprietary-information governance (or the same four under slightly different names) as the concern classes that LLM02 spans.
3. **SECURITY.md LLM01 and LLM02 cells survive PDF cross-reference**: Jane reads the SECURITY.md LLM01 Residual-risk cell and LLM02 Scope-limit cell with the OWASP PDF open, and confirms neither cell contradicts or overclaims relative to the cited source.

**If all three checks pass:** Jane confirms `gate: PASS`. Proceed to Step 2.11.

**If any check fails:** stop. Log the discrepancy. Do not commit the entry. Decide with Jane outside the time budget whether to patch the entry (if the draft is wrong) or update the design doc (if the canonical phrasing in SECURITY.md needs to change). Re-run the gate after any fix.

Same failure class as the CRAG-taxonomy error caught on 2026-04-14: an unverified citation propagates across surfaces because the four-surface discipline amplifies it. The gate's cost (~10–15 min) is cheap insurance against that failure mode.

- [ ] **Step 2.11: Commit (only after gate passes)**

```bash
cd /Users/zenith/Desktop/agent-bench
git add DECISIONS.md
git commit -m "$(cat <<'EOF'
docs(decisions): add entry on named residual risks and scope limits verdict discipline

New entry in DECISIONS.md documenting why the OWASP LLM Top 10
(2025) mapping in SECURITY.md uses named residual risks and scope
limits on LLM01 and LLM02 rather than uniform "fully mitigated"
verdicts. Topically clustered with the other security-phase
entries (between "Why monitor mode for output validation, not
gating?" and "Why additive SSE stage events?").

Named the rejected alternative (uniform "addressed" verdicts),
quotes OWASP 2025 "do not fully mitigate" language verbatim,
enumerates all four LLM02 concern classes, and names the
honesty-vs-scannability tension the entry resolves. Cross-link
to SECURITY.md uses the explicit responsibility-split phrasing:
SECURITY.md is the WHAT (verdict cells), this entry is the WHY
(verdict discipline).

Uses failure-mode framing ("URL hallucination", "PII leakage");
mechanism framing ("URL-against-retrieved-chunks check", etc.)
lives in SECURITY.md. The split is deliberate.

Paired-review gate (cross-cutting requirement #9) passed: OWASP
2025 verbatim phrases and four-concern-class enumeration
cross-checked against the official PDF before commit.

Design doc: docs/plans/2026-04-15-owasp-llm-top-10-mapping-design.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -1
```

Expected: commit lands.

---

## Task 3: Add README one-liner (Security Architecture closing tail)

**Files:**
- Modify: `/Users/zenith/Desktop/agent-bench/README.md` (single line edit near line 202)

**Estimated time:** ~15 minutes (most of which is the rhythm-check between the two variants).

**Task context:** The README already has a substantial "Security Architecture" section ending with the line "See [DECISIONS.md](DECISIONS.md) for why we chose two-tier detection over three, regex-only PII by default, JSONL over SQLite for audit, and HMAC over plain SHA-256 for IP hashing." The OWASP pointer joins this existing tail — either as a comma-joined clause (Variant 1) or as a preceding sentence (Variant 2). Rhythm check at drafting time; pick whichever breathes better in context. Plan-wording compliance ("one-line mention") is satisfied under either reading.

**Exit criteria:**
- README Security Architecture section closing tail is updated with the chosen variant
- Existing four-item DECISIONS.md list (two-tier / regex-PII / JSONL / HMAC) is intact
- No other README section is touched
- Committed with message `docs(readme): link SECURITY.md OWASP mapping from Security Architecture tail`

---

- [ ] **Step 3.1: Read the current closing tail**

Run:

```bash
grep -n "See \[DECISIONS.md\]" /Users/zenith/Desktop/agent-bench/README.md
```

Expected: one hit, in the Security Architecture section (around line ~202). Record the line number.

- [ ] **Step 3.2: Read the surrounding paragraph for rhythm context**

Read `/Users/zenith/Desktop/agent-bench/README.md` offset ~195, limit 15, to see the "application-layer security pipeline" disclaimer paragraph + the closing tail + what follows. The rhythm check is about whether the section closes tight (favors Variant 1) or feels heavy (favors Variant 2).

- [ ] **Step 3.3: Prepare Variant 1 (single sentence, comma-list tail)**

Variant 1:

```markdown
See [SECURITY.md](SECURITY.md) for the OWASP LLM Top 10 (2025) mapping, and [DECISIONS.md](DECISIONS.md) for why we chose two-tier detection over three, regex-only PII by default, JSONL over SQLite for audit, and HMAC over plain SHA-256 for IP hashing.
```

- [ ] **Step 3.4: Prepare Variant 2 (two sentences, equal billing)**

Variant 2:

```markdown
See [SECURITY.md](SECURITY.md) for the OWASP LLM Top 10 (2025) mapping. See [DECISIONS.md](DECISIONS.md) for why we chose two-tier detection over three, regex-only PII by default, JSONL over SQLite for audit, and HMAC over plain SHA-256 for IP hashing.
```

- [ ] **Step 3.5: Pick a variant on rhythm**

Read the two variants in the actual surrounding context (the paragraph before is the "application-layer security pipeline" disclaimer). Pick the one that reads cleaner. Rough heuristic: if the section already closes on a short, declarative sentence, Variant 1 (comma-join) stays tight; if the section closes on a longer sentence that already feels weighty, Variant 2 (two short sentences) breaks the cadence more cleanly. **There is no wrong answer between the two — this is a drafting-time rhythm choice, not a design choice.**

- [ ] **Step 3.6: Apply the chosen variant**

Replace the current `See [DECISIONS.md]...` line (identified in step 3.1) with the chosen variant. Use the Edit tool with the exact old string (the current one-line `See [DECISIONS.md] for why we chose...`) and the chosen variant as the new string.

- [ ] **Step 3.7: Verify the four-item DECISIONS.md list is intact**

Run:

```bash
grep "two-tier detection over three" /Users/zenith/Desktop/agent-bench/README.md
grep "regex-only PII by default" /Users/zenith/Desktop/agent-bench/README.md
grep "JSONL over SQLite for audit" /Users/zenith/Desktop/agent-bench/README.md
grep "HMAC over plain SHA-256" /Users/zenith/Desktop/agent-bench/README.md
```

Expected: each command returns `1` hit. If any returns `0`, the four-item list was accidentally truncated — re-apply the edit.

- [ ] **Step 3.8: Verify no other README section was touched**

Run:

```bash
cd /Users/zenith/Desktop/agent-bench
git diff README.md | head -30
```

Expected: the diff shows exactly one changed line (the tail) — no changes to Engineering Scope, no changes to V1→V2→V3 table, no tagline edit, no other section. If the diff shows unintended changes, revert them.

- [ ] **Step 3.9: Verify SECURITY.md link target**

Run:

```bash
grep -o "SECURITY.md" /Users/zenith/Desktop/agent-bench/README.md | head -5
```

Expected: at least one hit. The link is relative (`SECURITY.md`, not an absolute URL) because README renders on GitHub and relative resolves.

- [ ] **Step 3.10: Commit**

```bash
cd /Users/zenith/Desktop/agent-bench
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): link SECURITY.md OWASP mapping from Security Architecture tail

Adds a one-line pointer to SECURITY.md from the Security
Architecture section closing tail. The existing four-item
DECISIONS.md list (two-tier / regex-PII / JSONL / HMAC) is
preserved intact — cutting it to make room for the OWASP
pointer would have traded a credibility signal for a
navigation pointer, a net loss.

Engineering Scope bullets deliberately not touched — those
carry build artifacts, not documentation about build artifacts.

Design doc: docs/plans/2026-04-15-owasp-llm-top-10-mapping-design.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands.

---

## Task 4: Add landing page Security panel subtitle

**Files:**
- Modify: `/Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html` (add one HTML element + one CSS rule inside the existing `<style>` block)

**Estimated time:** ~30 minutes (most of which is viewport rendering verification).

**Task context:** The landing page's Security panel (right column of the dashboard) currently renders only the `<h3>Security</h3>` heading plus three live badges. The OWASP pointer is a new sub-text element inserted between the h3 and the badges row. Block-link (whole subtitle wrapped in `<a>`), absolute GitHub URL (relative won't resolve on HF Spaces), `aria-label` for screen readers, em-dash variant for mobile-wrap readability, `target="_blank"` matching the existing finding-card pattern.

**Exit criteria:**
- One `<a class="security-panel-sub">` element inserted between `<h3>Security</h3>` and `<div class="security-badges">`
- Element is a block-link (whole subtitle is the link target)
- `href="https://github.com/tyy0811/agent-bench/blob/main/SECURITY.md"` (absolute)
- `aria-label="OWASP LLM Top 10 mapping in SECURITY.md"` present
- `target="_blank"` present
- Subtitle text uses em-dash variant: "Mapped against the OWASP LLM Top 10 (2025) &mdash; named residual risks for LLM01, scope limits for LLM02 &rarr; SECURITY.md"
- New `.security-panel-sub` CSS class added in the `<style>` block (distinct from the existing `.sec-sub` class which is scoped inside `.sec-badge`)
- Dev server (`make serve`) started and subtitle verified to render at 1440px and 375px viewport widths
- No other element in `index.html` touched
- Committed with message `docs(landing): add OWASP mapping subtitle to Security panel`

---

- [ ] **Step 4.1: Locate the existing Security panel h3**

Run:

```bash
grep -n '<h3>Security</h3>' /Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html
```

Expected: one hit around line ~312. Record the line number.

- [ ] **Step 4.2: Locate the existing `.sec-sub` CSS class to confirm the new class name doesn't collide**

Run:

```bash
grep -n '\.sec-sub' /Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html
```

Expected: one or more hits, all inside the `<style>` block, scoped to `.sec-badge`. The new `.security-panel-sub` class is distinct.

- [ ] **Step 4.3: Locate a good insertion point for the new CSS rule**

Run:

```bash
grep -n '/\* Security badges \*/' /Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html
```

Expected: one hit around line ~135, inside the `<style>` block. The new `.security-panel-sub` rule goes near this block to keep security-related styles clustered.

- [ ] **Step 4.4: Add the new CSS rule**

Use the Edit tool to insert the following rule into the `<style>` block, immediately after the existing `/* Security badges */` comment and the `.security-panel` rule that follows it. The exact anchor is the `.security-panel` rule's closing brace.

Anchor text (approximate; verify exact text before editing):

```css
.security-panel{background:var(--panel-bg);border:1px solid var(--panel-border);border-radius:12px;padding:16px}
```

New CSS to insert **immediately after** that line:

```css
.security-panel-sub{display:block;color:var(--muted);font-size:0.78rem;line-height:1.4;margin-top:-4px;margin-bottom:10px;text-decoration:none}
.security-panel-sub:hover{color:var(--text);text-decoration:underline}
```

- [ ] **Step 4.5: Add the HTML subtitle element**

Use the Edit tool to insert the following element **immediately after** `<h3>Security</h3>` and **before** `<div class="security-badges">`.

Exact old string (3 lines):

```html
      <div class="security-panel">
        <h3>Security</h3>
        <div class="security-badges">
```

Exact new string (4 lines, preserving indentation):

```html
      <div class="security-panel">
        <h3>Security</h3>
        <a class="security-panel-sub" href="https://github.com/tyy0811/agent-bench/blob/main/SECURITY.md" target="_blank" aria-label="OWASP LLM Top 10 mapping in SECURITY.md">Mapped against the OWASP LLM Top 10 (2025) &mdash; named residual risks for LLM01, scope limits for LLM02 &rarr; SECURITY.md</a>
        <div class="security-badges">
```

- [ ] **Step 4.6: Verify the HTML insertion**

Run:

```bash
grep -c 'class="security-panel-sub"' /Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html
grep -c 'aria-label="OWASP LLM Top 10 mapping in SECURITY.md"' /Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html
grep -c 'github.com/tyy0811/agent-bench/blob/main/SECURITY.md' /Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html
grep -c '&mdash; named residual risks for LLM01, scope limits for LLM02' /Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html
```

Expected: each command returns exactly `1`. If any returns `0`, the insertion failed; re-check the edit.

- [ ] **Step 4.7: Start the dev server for visual verification**

```bash
cd /Users/zenith/Desktop/agent-bench
make serve
```

Expected: FastAPI starts on `http://localhost:8000`. This runs in the background for the next few steps; do not stop it until step 4.11.

- [ ] **Step 4.8: Verify desktop viewport render (1440px)**

Open `http://localhost:8000` in a browser. Confirm:

- The Security panel in the dashboard right column now shows the new subtitle line below the `Security` heading and above the three badges.
- The subtitle renders as muted text (gray, `0.78rem`).
- The subtitle is clickable and opens `https://github.com/tyy0811/agent-bench/blob/main/SECURITY.md` in a new tab (the link target doesn't exist yet on main's `SECURITY.md`, which is fine — Task 5 pushes the branch and the link resolves after merge).
- Hover state changes color and adds an underline.
- The em-dash and right-arrow Unicode characters render cleanly.

If any of these fail, stop the server, fix the HTML/CSS, and re-test.

- [ ] **Step 4.9: Verify narrow viewport render (375px)**

In browser devtools, switch to responsive mode at 375px width. Confirm:

- The subtitle wraps to 2–3 lines at this width.
- The em-dash provides a natural wrap break so the phrase "named residual risks for LLM01" and "scope limits for LLM02" don't split mid-phrase.
- The subtitle stays readable (no font-size collapse, no overflow).

If the wrap reads poorly, the em-dash variant may still be the best option — but flag it as a known limitation of the 0.78rem font size at narrow viewport.

- [ ] **Step 4.10: Drafting-time check on `margin-top: -4px`**

Inspect the rendered panel in browser devtools. The negative top margin pulls the subtitle closer to the h3. If this reads brittle (e.g., if it overlaps the h3 baseline, or if it looks intentional vs accidental-negative-space), prefer an explicit adjustment to the h3 bottom margin instead. In that case, replace `margin-top:-4px` in the CSS rule with `margin-top:0` and add a new rule:

```css
.security-panel h3{margin-bottom:4px}
```

If the negative margin reads fine, leave it.

- [ ] **Step 4.11: Stop the dev server**

```
Ctrl+C in the terminal running make serve
```

- [ ] **Step 4.12: Verify no other element was touched**

Run:

```bash
cd /Users/zenith/Desktop/agent-bench
git diff agent_bench/serving/static/index.html | head -50
```

Expected: the diff shows exactly two insertions — the new CSS rule (2 lines) and the new HTML element (1 line). No changes to the pipeline stages, retrieval panel, findings cards, tiles, request log, footer, or any other element. If the diff shows unintended changes, revert them.

- [ ] **Step 4.13: Commit**

```bash
cd /Users/zenith/Desktop/agent-bench
git add agent_bench/serving/static/index.html
git commit -m "$(cat <<'EOF'
docs(landing): add OWASP mapping subtitle to Security panel

Adds a block-link subtitle under the Security panel h3 that
points to the top-level SECURITY.md. Uses an absolute GitHub
URL (relative path wouldn't resolve on HF Spaces), an
aria-label for screen-reader users, and the em-dash variant
for cleaner mobile wrap at 375px viewport.

The subtitle names the honesty qualifier ("named residual
risks for LLM01, scope limits for LLM02") inline so a
skim-reader understands before clicking that SECURITY.md is
an honest mapping, not a checkbox. Without the qualifier,
the subtitle would read as compliance theater.

New .security-panel-sub CSS class (distinct from the existing
.sec-sub class which is scoped to .sec-badge). No other
landing-page element touched.

Design doc: docs/plans/2026-04-15-owasp-llm-top-10-mapping-design.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands.

---

## Task 5: Cross-surface verification and push

**Files:** None modified (unless verification surfaces a drift; then fix + commit + re-verify).

**Estimated time:** ~30 minutes. This task absorbs the pooled slack for the whole plan.

**Task context:** The four surfaces are all landed. This task is the pre-merge verification pass — canonical-phrasing audit, cross-link target resolution, `make test` + `make lint` regression check, final no-overclaiming scan, and push to origin. No commits in the normal case; drift-repair commits only if a check fails.

**Exit criteria:**
- Canonical-phrasing table from design doc §3.1 fully audited
- All cross-link targets resolve (source files exist, DECISIONS.md anchors exist, SECURITY.md anchors exist)
- `make test` passes with no regressions (baseline: 444 tests)
- `make lint` passes
- No overclaiming phrases anywhere across the four surfaces
- SECURITY.md word count within planned range (500–800)
- DECISIONS.md entry word count within planned range (290–360)
- Branch `part-a/owasp-mapping` pushed to `origin`
- Ready for PR to `main`

---

- [ ] **Step 5.1: Read all four surfaces end-to-end in execution order**

Read in order:

1. `/Users/zenith/Desktop/agent-bench/SECURITY.md` — full file
2. The new DECISIONS.md entry only (the "Why named residual risks and scope limits..." section)
3. The README Security Architecture section closing tail (one or two lines)
4. The landing page Security panel block (h3 + new subtitle + badges)

While reading, check that the four surfaces describe the same mapping with consistent vocabulary: named residual risks, scope limits, four concern classes, narrower output-side subset, etc.

- [ ] **Step 5.2: Canonical-phrasing audit — LLM01 verbatim**

Run:

```bash
grep -l "do not fully mitigate prompt injection" \
  /Users/zenith/Desktop/agent-bench/SECURITY.md \
  /Users/zenith/Desktop/agent-bench/DECISIONS.md
```

Expected: both files listed. SECURITY.md is canonical; DECISIONS.md paraphrases/quotes. If only one hits, the canonical-phrasing discipline has drifted — inspect and repair.

- [ ] **Step 5.3: Canonical-phrasing audit — LLM02 four-concern-class enumeration**

Run:

```bash
grep -l "access controls, training-data handling, user-consent transparency, and proprietary-information governance" \
  /Users/zenith/Desktop/agent-bench/SECURITY.md \
  /Users/zenith/Desktop/agent-bench/DECISIONS.md
```

Expected: both files listed.

- [ ] **Step 5.4: Canonical-phrasing audit — "narrower output-side subset" disclaim**

Run:

```bash
grep -l "narrower, output-side subset that does not cleanly map to any single one of the four" /Users/zenith/Desktop/agent-bench/SECURITY.md
grep "narrower output-side subset" /Users/zenith/Desktop/agent-bench/DECISIONS.md
```

Expected: first command lists SECURITY.md; second command shows at least one hit in DECISIONS.md with matching disclaim semantics (DECISIONS.md version may phrase as "narrower output-side subset" without the comma — that's fine, the structural claim matches).

- [ ] **Step 5.5: Canonical-phrasing audit — structural phrase on landing page**

Run:

```bash
grep "named residual risks for LLM01, scope limits for LLM02" /Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html
```

Expected: one hit. This is the structural phrase the landing subtitle carries; if absent, Task 4 dropped the required wording.

- [ ] **Step 5.6: Framing-discipline audit — SECURITY.md uses mechanism framing, DECISIONS.md uses failure-mode framing**

Run:

```bash
# SECURITY.md should use mechanism framing
grep -c "URL-against-retrieved-chunks check\|always-on secret-format deny list" /Users/zenith/Desktop/agent-bench/SECURITY.md
# DECISIONS.md new entry should use failure-mode framing (extract the entry first)
awk '/^## Why named residual risks and scope limits/,/^## Why additive SSE stage events/' /Users/zenith/Desktop/agent-bench/DECISIONS.md | grep -c "URL hallucination\|PII leakage"
```

Expected: first command ≥ 1 (mechanism phrases in SECURITY.md), second command ≥ 1 (failure-mode phrases in DECISIONS.md). If either returns 0, the framings may have swapped — inspect and repair.

- [ ] **Step 5.7: Cross-link verification — source files exist**

Run:

```bash
ls /Users/zenith/Desktop/agent-bench/agent_bench/security/injection_detector.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/security/pii_redactor.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/security/output_validator.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/tools/registry.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/agents/orchestrator.py \
   /Users/zenith/Desktop/agent-bench/agent_bench/serving/middleware.py \
   /Users/zenith/Desktop/agent-bench/pyproject.toml \
   /Users/zenith/Desktop/agent-bench/Dockerfile
```

Expected: all 8 files listed, no errors.

- [ ] **Step 5.8: Cross-link verification — DECISIONS.md anchors exist**

Run:

```bash
grep -E "^## (Why two-tier injection detection|Why regex \+ optional spaCy|Why three output validators|Why no authentication|Why a relevance threshold)" /Users/zenith/Desktop/agent-bench/DECISIONS.md
```

Expected: 5 lines, one for each anchor SECURITY.md cross-links to.

- [ ] **Step 5.9: Cross-link verification — SECURITY.md anchors exist for DECISIONS.md entry**

Run:

```bash
grep -E "^### (LLM01 Prompt Injection|LLM02 Sensitive Information Disclosure)" /Users/zenith/Desktop/agent-bench/SECURITY.md
```

Expected: 2 lines, one for each anchor the DECISIONS.md entry cross-links to.

- [ ] **Step 5.10: Run the test suite**

```bash
cd /Users/zenith/Desktop/agent-bench
make test
```

Expected: 444 tests pass (or the current baseline count, whichever is higher — new tests may have landed on main since the design doc was written). **Failure mode to watch for:** if the landing-page HTML insertion breaks a static-route test that asserts on the HTML structure, investigate whether the assertion is structural-fragile (needs updating to account for the new element) or behavioral (the insertion actually broke something). Structural-fragile assertions can be tightened to check presence, not ordering; behavioral breaks mean the edit is wrong. Do not land the fix if the test break is behavioral.

- [ ] **Step 5.11: Run the linter**

```bash
cd /Users/zenith/Desktop/agent-bench
make lint
```

Expected: ruff + mypy pass with no new issues. Since no Python code changed, this is a consistency check only.

- [ ] **Step 5.12: Final no-overclaiming scan across all four surfaces**

Run:

```bash
grep -iE "fully mitigated|coverage guarantee|tests all 10|OWASP.certified" \
  /Users/zenith/Desktop/agent-bench/SECURITY.md \
  /Users/zenith/Desktop/agent-bench/DECISIONS.md \
  /Users/zenith/Desktop/agent-bench/README.md \
  /Users/zenith/Desktop/agent-bench/agent_bench/serving/static/index.html
```

Expected: no output. If anything matches, inspect for disclaimer-vs-claim context (SECURITY.md's "What this doc is not" legitimately contains "does not constitute OWASP certification" — that's a disclaimer). Non-disclaimer matches must be rephrased.

- [ ] **Step 5.13: Verify word counts**

```bash
wc -w /Users/zenith/Desktop/agent-bench/SECURITY.md
awk '/^## Why named residual risks and scope limits/,/^## Why additive SSE stage events/' /Users/zenith/Desktop/agent-bench/DECISIONS.md | wc -w
```

Expected: SECURITY.md count between 500 and 800 (target ~790, per the drafting-time drift); DECISIONS.md entry count between 290 and 360 (target ~320).

- [ ] **Step 5.14: Push to origin**

```bash
cd /Users/zenith/Desktop/agent-bench
git log --oneline -6
git push origin part-a/owasp-mapping
```

Expected: `git log` shows 5 commits on this branch (1 design doc + 1 self-review fixes + 1 SECURITY.md + 1 DECISIONS.md entry + 1 README tail + 1 landing subtitle = 6 commits on `part-a/owasp-mapping` beyond main). Push succeeds; origin/part-a/owasp-mapping updated.

- [ ] **Step 5.15: Ready for PR**

Surface the branch state to Jane with a short summary:

- Branch: `part-a/owasp-mapping`
- Commits: 1 design doc + 1 self-review + 4 implementation surface commits = 6 total
- PR target: `main`
- PR body should reference the design doc + the locked Part A design
- No HF Spaces push (Part A is not a deploy)

Do not open the PR yourself; surface the state and let Jane decide whether to open the PR now or wait.

---

## Plan-level self-review

Before marking this plan ready for execution, verify:

1. **Spec coverage.** Every locked decision from the design doc's "Locked decisions (from brainstorming)" table (13 entries) is reflected in at least one task step. Cross-cutting #9 (paired-review gate) is covered in Task 2 step 2.10.

2. **No placeholders.** The LLM01 and LLM02 canonical subsections contain exact text; the DECISIONS.md entry paragraphs contain exact text; the README variants contain exact text; the landing HTML and CSS contain exact text. No "TBD", no "TODO", no "fill in later".

3. **Type consistency.** Class names (`.security-panel-sub` vs `.sec-sub`), file paths (`agent_bench/security/injection_detector.py`, etc.), and cross-link anchor slugs (`#llm01-prompt-injection`, `#why-two-tier-injection-detection-not-three`, etc.) are consistent across tasks.

4. **Execution order.** Tasks 1 → 2 → 3 → 4 → 5 is strict per design doc §3.2. Task 2's paired-review gate blocks Task 3. No task depends on work from a later task.

5. **Budget drift documentation.** The SECURITY.md word count overshoot (500–700 target → ~790 realistic) is flagged at the top of this plan and again in Task 1 exit criteria. The overshoot is a precision/phrasing drift per the hybrid drafting-gap protocol and does not require design-doc revision.

6. **Scope guardrails.** No task modifies `agent_bench/security/*`, no task modifies tests, no task changes routes/app/wiring, no task edits Hugging Face metadata, no task modifies README outside the Security Architecture closing tail.
