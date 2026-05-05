---
dimension: groundedness
scale: binary
reference_based: true
abstain_allowed: true
---

# Groundedness (binary)

Score whether **every claim** in the agent's answer is entailed by the gold
source snippets attached to this item. Paraphrase is allowed; what matters
is content equivalence, not surface form.

## Reference scope (strict, clarified in v1.1)

Reference scope is the **gold snippets only**, not the broader corpus, not
the retrieved chunks, not the LLM's general domain knowledge. A claim that
is factually correct in the world but not entailed by the snippets **must
score 0**. The "must" forecloses the "well, mostly grounded" reading: a
single ungrounded claim in an otherwise solid answer fails the binary
rubric.

The strict-entailment posture is a methodological choice. A claim that is
correct because the model happened to know it isn't grounded — it's lucky.
Strict-snippet groundedness measures *retrieval-grounded behavior*, not
LLM general knowledge passing through a RAG harness.

## Trivial inference is entailment

Some surface-form variations of a snippet's content are entailment, not
new claims. The test is **content equivalence**, not surface form:

- **Paraphrase.** "X causes Y" ↔ "Y is caused by X".
- **Unit conversion.** "600 seconds" ↔ "10 minutes".
- **Syntactic variation.** Pluralization, tense, voice, declarative ↔ imperative.
- **Canonical name of the snippet's concept.** When the snippet describes
  a field, header, or API element by configuration syntax (e.g., a
  `max_age` table row), the canonical name (`Access-Control-Max-Age` HTTP
  header) is the same content in different surface form. This is a
  separate carve-out from pure paraphrase: it admits domain knowledge
  tightly bound to the snippet's referent.

> **v1.2 debt.** The trivial-inference clause — especially the
> canonical-name carve-out — is the strictest-rubric concession most
> likely to require revision in v1.2. If labelers find themselves
> applying it broadly to rescue answers from score-0, the clause is
> too permissive and should be tightened.

**When to abstain (`"Unknown"`)**: if the answer is a refusal ("I don't
know" / "not in the documentation") and there is nothing to ground, score
abstain rather than 1.

## Score 0

At least one claim in the answer is not entailed by any snippet, after
applying the trivial-inference clause.

### Example A — calibration anchor `k8s_006` (dramatic over-extension)

Question: "What is a ConfigMap in Kubernetes and what kind of data should you store in it?"

Snippet: "A ConfigMap is an API object used to store non-confidential data in key-value pairs."

Answer (excerpted): The agent gives a comprehensive multi-section answer
covering (i) the definition, (ii) three consumption methods (env vars,
command-line args, volumes), (iii) a warning not to store
passwords/tokens/certificates, (iv) a recommendation to use Secrets
instead, and (v) details about `data` and `binaryData` fields.

Thinking trace: Score = 0. Only the definition (i) is entailed by the
snippet. Claims (ii)–(v) are factually correct against the underlying
`k8s_configmap.md` doc, but **none are entailed by the one-sentence
snippet**. The snippet does not describe consumption methods, security
guidance, or schema fields. The strict-conjunction rule applies: even
though most of the answer is well-supported by the broader corpus, the
gold-snippet scope is what the rubric measures, and the answer goes
dramatically beyond it.

### Example B — calibration anchor `q006` (subtle embellishment)

Question: "How does dependency caching work in FastAPI, and how can you disable it?"

Snippet: "By default, if the same dependency is used multiple times within
a single request (e.g., both a route and a sub-dependency use
`Depends(get_db)`), FastAPI caches the result and calls the dependency
only once. To disable caching and force a fresh call each time, use
`Depends(get_db, use_cache=False)`."

Answer (excerpted): Most of the answer is entailed by the snippet — the
default-caching behavior, the once-per-request semantics, the
`use_cache=False` disable mechanism, and an example. But the answer also
states that caching is "particularly useful for expensive operations like
database connections."

Thinking trace: Score = 0. The "particularly useful for expensive
operations like database connections" claim is **not entailed** by the
snippet. The snippet describes *what* caching does (a mechanism); the
claim adds *why* it is useful (a use-case argument the snippet does not
make). The example uses `Depends(get_db)` as an identifier, not as a
normative argument that caching is useful for database connections. This
is the subtle case: most of the answer is good, but a single embellishment
that imports general LLM knowledge ("expensive operations like database
connections") fails strict snippet entailment. This is the boundary case
the rubric exists to catch — easier to mis-label than `k8s_006` because
the failure is one claim in the middle of a mostly-grounded answer.

## Score 1

Every claim in the answer is entailed by at least one snippet, including
trivial inferences (paraphrase, unit conversion, syntactic variation,
canonical-name substitution).

### Example C — calibration anchor `q021` (trivial inference + canonical name)

Question: "If the CORS max_age is 600 seconds, how many minutes does the browser cache preflight results?"

Snippet (table row): `| max_age | 600 | Seconds the browser caches preflight results |`

Answer: "If the CORS `max_age` is 600 seconds, the browser caches
preflight results for 10 minutes. The `max_age` directive in the
`Access-Control-Max-Age` header specifies the maximum time (in seconds)
that the results of a preflight request can be cached. So 600 seconds
÷ 60 seconds/minute = 10 minutes."

Thinking trace: Score = 1. Every claim is entailed under the
trivial-inference clause:

- "max_age=600 → 10 minutes" — unit conversion.
- "Access-Control-Max-Age header" — canonical-name carve-out: the snippet
  describes `max_age` by its configuration syntax; the answer uses the
  canonical HTTP header name for the same field.
- "specifies the maximum time (in seconds)" — paraphrase of "Seconds the
  browser caches preflight results".
- "600 ÷ 60 = 10 minutes" — arithmetic, the same trivial-inference class
  as unit conversion.

The canonical-name carve-out is doing the heaviest lifting in this
example. Without it, "Access-Control-Max-Age" would be ungrounded
(domain knowledge not in the snippet text). With it, the answer is a
clean strict-snippet pass. This is exactly the v1.2-debt sentence above
— if many future labels rescue score-1 via canonical-name appeals, the
clause is over-rescuing and should be tightened.
