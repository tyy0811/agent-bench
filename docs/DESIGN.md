# agent-bench — Design Document

> Evaluation-first agentic RAG system with one provider, one domain, one API, and one benchmark report — built from API primitives on a CPU-only laptop.

Based on V3 spec with 7 refinements from design review (2026-03-24).

---

## Scope Lock

| Decision | Choice |
|----------|--------|
| LLM backend | OpenAI (`gpt-4o-mini`) + `MockProvider` for tests + `AnthropicProvider` stub |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers, CPU) |
| Vector store | FAISS (CPU) + BM25, fused via Reciprocal Rank Fusion |
| API framework | FastAPI |
| Validation | Pydantic v2 |
| Testing | pytest + httpx (async test client) |
| CI | GitHub Actions — full deterministic test suite |
| Containerization | Docker + docker-compose |
| Logging | structlog (JSON structured logging) |
| Domain | Technical documentation Q&A (markdown) |
| Corpus | ~15-20 curated markdown files (e.g., FastAPI tutorial pages) |
| Async strategy | Async provider internals, sync user-facing behavior |
| Citation format | Structured `sources` list in JSON + `[source: filename.md]` inline |

### Non-goals (V1)

No LangChain/LlamaIndex, no fine-tuning, no frontend UI, no cloud deploy, no third-party observability, no GPU, no streaming, no persistent memory/conversation DB, no `/upload` endpoint, no second domain, no second provider implementation, no conversation sessions.

---

## Repository Structure

```
agent-bench/
├── pyproject.toml
├── Makefile
├── README.md
├── DECISIONS.md
├── .github/workflows/ci.yaml
├── configs/
│   ├── default.yaml
│   └── tasks/tech_docs.yaml
├── data/tech_docs/                  # ~15-20 curated markdown files
├── agent_bench/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── provider.py            # LLM provider abstraction
│   │   ├── config.py              # Pydantic settings
│   │   └── types.py               # Shared type definitions
│   ├── agents/
│   │   ├── __init__.py
│   │   └── orchestrator.py        # Tool-use loop (no memory.py in V1)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── base.py
│   │   ├── search.py
│   │   └── calculator.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── store.py
│   │   └── retriever.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── harness.py
│   │   ├── metrics.py
│   │   ├── datasets/tech_docs_golden.json
│   │   └── report.py
│   └── serving/
│       ├── __init__.py
│       ├── app.py
│       ├── routes.py
│       ├── schemas.py
│       └── middleware.py
├── scripts/
│   ├── ingest.py
│   ├── evaluate.py
│   └── benchmark.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_provider.py
│   ├── test_tools.py
│   ├── test_rag.py
│   ├── test_agent.py
│   └── test_serving.py
└── docker/
    ├── Dockerfile
    └── docker-compose.yaml
```

---

## Data Flow

```
Client → FastAPI (/ask) → Middleware (request_id, timing)
  → Orchestrator.run(question, top_k, strategy)
    → messages = [system_prompt, user_question]
    → Loop (max 3 iterations):
        → OpenAI.complete(messages, tools=[search_documents, calculator])
        → If tool_calls: execute via ToolRegistry → append tool results to messages
        → If no tool_calls: break (final answer)
    → If max iterations hit: one final complete() without tools → force text answer
    → Return AgentResponse(answer, sources, metadata)
  → Serialize to AskResponse
→ Client
```

Three endpoints: `POST /ask`, `GET /health`, `GET /metrics`. No CRUD, no sessions, no auth.

Three singletons at startup: ToolRegistry, HybridStore (loaded from disk), OpenAI client.

---

## Provider Abstraction

```python
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=1024) -> CompletionResponse: ...
    @abstractmethod
    def format_tools(self, tools: list[ToolDefinition]) -> list[dict]: ...
```

Three implementations:
1. **OpenAIProvider** — full implementation, `gpt-4o-mini` default
2. **MockProvider** — deterministic responses for tests (returns tool_calls on first call, final answer when tool results present)
3. **AnthropicProvider** — raises `NotImplementedError("planned for V2")`

### OpenAI-specific details

- Message mapping: internal Role enum → OpenAI role strings
- Tool calls: `choice.message.tool_calls` → `list[ToolCall]`
- Arguments parsing: `json.loads(tc.function.arguments)` with try/except for malformed JSON → empty dict fallback
- Cost: `(input_tokens * input_cost_per_mtok + output_tokens * output_cost_per_mtok) / 1_000_000`, pricing from config YAML
- Latency: `time.perf_counter()` around the API call
- Errors: `openai.APITimeoutError` → domain exception. No retries in V1.
- `tool_choice: "auto"` — let the model decide

### MockProvider keying

Checks whether messages contain `Role.TOOL` entries:
- No tool results present → return canned response with `tool_calls`
- Tool results present → return canned final answer with no `tool_calls`
- Returns realistic `TokenUsage` for cost-tracking tests

---

## RAG Pipeline

### Chunk model (flattened)

```python
class Chunk(BaseModel):
    id: str              # hash of content + source
    content: str
    source: str          # bare filename, e.g. "fastapi_path_params.md"
    chunk_index: int
    metadata: dict
```

`chunk.source` must match golden dataset `expected_sources` exactly (bare filename, no path prefix).

### Chunker

Two strategies, configured via `chunk_size` (512) and `chunk_overlap` (64):
- **Recursive:** splits on `\n\n` → `\n` → `. ` → space
- **Fixed-size:** character-count splits with overlap

### Embedder

- `SentenceTransformer('all-MiniLM-L6-v2')`, loaded once at init
- Output: `np.ndarray` shape `(384,)` per chunk
- Disk cache: `hash(content)` → `.cache/embeddings/{hash}.npy`

### Store (FAISS + BM25 + RRF)

- FAISS `IndexFlatIP` on L2-normalized vectors (= cosine similarity)
- BM25 via `rank_bm25.BM25Okapi`, tokenized with `re.findall(r'\w+', text.lower())`
- `add(chunks)` writes to both indices
- `search(query, top_k, strategy)` where strategy = "semantic" | "keyword" | "hybrid"

**RRF fusion:**
```
dense_results  = faiss.search(query_embedding, k=candidates_per_system)  # 10
sparse_results = bm25.get_top_n(tokenized_query, k=candidates_per_system)  # 10
For each unique chunk: rrf_score = Σ 1/(60 + rank_in_system)
Sort by rrf_score descending, return top_k (5)
```

- `save()`/`load()`: FAISS via `faiss.write_index`/`read_index`, BM25 via pickle, chunks via JSON
- No delete in V1. Rebuild on re-ingest.

### Retriever

Thin glue: query string → embedder → store.search() → `list[SearchResult]`.

---

## Tool System

### Interface

```python
class Tool(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema
    @abstractmethod
    async def execute(self, **kwargs) -> ToolOutput: ...
```

### SearchTool

- Input: `query: str`, optional `top_k: int = 5`
- Calls `retriever.search(query, top_k)`
- Formats results as numbered passages with filename attribution:
  ```
  [1] (fastapi_path_params.md): Path parameters are defined using curly braces...
  [2] (fastapi_query_params.md): Query parameters are automatically parsed...
  ```
- Returns `ToolOutput(success=True, result=formatted, metadata={"sources": [filenames]})`

### CalculatorTool

- Input: `expression: str`
- Uses `simpleeval.simple_eval()` (blocks import, exec, eval, attribute access by default)
- Wrapped in try/except:
  ```python
  try:
      result = simple_eval(expression)
      return ToolOutput(success=True, result=str(result))
  except Exception:
      return ToolOutput(success=False, result=f"Could not evaluate: {expression}")
  ```

### Registry

- Dict-based. `register(tool)`, `execute(name, **kwargs)`, `get_definitions()`
- Unknown tool name → `ToolOutput(success=False, result="Unknown tool: {name}")`

---

## Orchestrator

```python
async def run(self, question, system_prompt, top_k, strategy) -> AgentResponse:
    messages = [system, user]
    tools = registry.get_definitions()
    all_sources, tools_used = [], []
    total_usage = TokenUsage(0, 0, 0.0)

    for iteration in range(max_iterations):
        response = await provider.complete(messages, tools=tools)
        # Manual accumulation (no operator overloading on Pydantic model)
        total_usage.input_tokens += response.usage.input_tokens
        total_usage.output_tokens += response.usage.output_tokens
        total_usage.estimated_cost_usd += response.usage.estimated_cost_usd

        if not response.tool_calls:
            return AgentResponse(answer=response.content, sources=dedup(all_sources), ...)

        messages.append(assistant_msg_with_tool_calls)
        for tc in response.tool_calls:
            result = await registry.execute(tc.name, **tc.arguments)
            messages.append(Message(role=TOOL, content=result.result, tool_call_id=tc.id))
            tools_used.append(tc.name)
            if "sources" in result.metadata:
                all_sources.extend(result.metadata["sources"])

    # Max iterations hit — force a text answer without tools
    response = await provider.complete(messages, tools=None)
    return AgentResponse(answer=response.content, sources=dedup(all_sources), ...)
```

No `memory.py` in V1. The `messages` list is local to this function. Every `/ask` request is stateless.

---

## Serving Layer

### Schemas

```python
class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    retrieval_strategy: Literal["semantic", "keyword", "hybrid"] = "hybrid"

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    metadata: ResponseMetadata  # provider, model, iterations, tools_used, latency_ms, token_usage, request_id
```

No `conversation_id`. No persistent sessions in V1.

### App factory

Initializes singletons (embedder, store, retriever, registry, provider, orchestrator), attaches to `app.state`.

### Middleware

- `X-Request-ID` (uuid4) on every response
- structlog: method, path, status, latency_ms, request_id
- Provider timeout → 504
- Unexpected exceptions → 500 with request_id

### MetricsCollector

In-process: `deque(maxlen=1000)` of latencies, request count, error count, total cost. Percentiles computed on demand. Resets on restart.

---

## Evaluation Harness

### Golden dataset (25 questions)

- 20 positive: 8 easy (single chunk), 8 medium (2-3 chunks), 4 hard (multi-source)
- 5 negative: out-of-scope, expects grounded refusal
- 3+ calculator questions among the 20 positive

Written in two passes: 10 on Day 4 (after seeing retrieval), 15 on Day 7 (after seeing actual system behavior).

### Deterministic metrics (free, CI-safe)

- `retrieval_precision_at_k(retrieved, expected, k=5)`
- `retrieval_recall_at_k(retrieved, expected, k=5)`
- `keyword_hit_rate(answer, keywords)`
- `source_presence_rate(response)` — has at least one source?
- `grounded_refusal_rate(answer, category, expected_sources)` — out-of-scope → refuses + no sources
- `citation_accuracy(answer, sources)` — regex `\[source: (.+?)\]`, check against structured sources list
- `calculator_used_when_expected(response, requires_calculator)`
- `tool_call_count(response)`

### LLM-judge metrics (costs money, manual)

- `answer_faithfulness(answer, chunks, judge)` → 0.0-1.0
- `answer_correctness(answer, reference, judge)` → 0.0-1.0

Judge prompt ends with: `Respond with ONLY a JSON object: {"score": 0.8, "reasoning": "brief explanation"}`. Parse with `json.loads()`. If parsing fails, return `None`. Log reasoning for failure analysis.

### Benchmark report (`docs/benchmark_report.md`)

Tables: aggregate, by category, by difficulty, chunking comparison.
Failure analysis: 3 worst queries with root cause (manual, informed by judge reasoning).
Config snapshot: full YAML dumped for reproducibility.

---

## Testing (31 tests, all deterministic)

### Fixtures (`conftest.py`)

- `mock_provider`: MockProvider with realistic TokenUsage
- `mock_embedder`: replaces SentenceTransformer with `np.random.RandomState(seed).randn(n, 384)` normalized to unit length. Deterministic, no model download.
- `sample_chunks`: 5-10 Chunk objects with known content/sources
- `test_store`: HybridStore populated with sample_chunks via mock_embedder
- `test_registry`: SearchTool (backed by test_retriever) + CalculatorTool

### Test files

| File | Tests | Coverage |
|------|-------|----------|
| `test_provider.py` | 6 | MockProvider responses, format_tools schema, cost calc, stub raises |
| `test_tools.py` | 6 | Registry CRUD, search results, calculator valid/invalid, JSON Schema |
| `test_rag.py` | 9 | Chunker strategies, embedder shape/cache, store search/RRF/empty/roundtrip |
| `test_agent.py` | 4 | AgentResponse fields, max_iterations, source accumulation, deterministic output |
| `test_serving.py` | 6 | /ask valid/invalid, /health, /metrics, request_id header, timeout 504 |

All tests use MockProvider + mock_embedder. No API keys. No model downloads. CI runs full suite.

---

## Changes from V3 Spec

| # | Change | Why |
|---|--------|-----|
| 1 | Drop `agents/memory.py` | No `conversation_id` → cross-request memory is a contract you can't honor |
| 2 | Fix build-backend to `setuptools.build_meta` | Legacy backend breaks editable installs |
| 3 | Wrap `simpleeval` errors in try/except | Prevents agent loop crash on malformed expressions |
| 4 | Split golden dataset: 10 on Day 4, 15 on Day 7 | Better questions informed by real retrieval behavior |
| 5 | `json.loads` safety net on tool_call arguments | Handles rare OpenAI malformed JSON |
| 6 | Toolless final call on max-iterations fallback | Clean answer instead of raw tool result string |
| 7 | JSON-structured LLM judge output with reasoning | Reliable parsing + free root-cause hints |

### Implementation details locked in

- SearchTool formats: `[1] (filename.md): content...`
- BM25 tokenizer: `re.findall(r'\w+', text.lower())`
- `Chunk.source` = bare filename matching golden dataset `expected_sources`
- Manual token accumulation (no Pydantic operator overloading)
- `mock_embedder` fixture with seeded deterministic vectors

---

## Build Sequence

| Day | Focus | Gate |
|-----|-------|------|
| 1 | Repo + provider + config | `make install && make test` green |
| 2 | Tools + registry | Registration, dispatch, schema generation pass |
| 3 | RAG core (chunker, embedder, store) | chunk → embed → store → retrieve works |
| 4 | RAG e2e + ingest + 10 golden questions | **GATE: known query → right chunk. P@5 ≥ 0.5** |
| 5 | Orchestrator wired to tools + RAG | Agent answers questions e2e using search + LLM |
| 6 | Serving layer | `curl POST /ask` returns valid AskResponse |
| 7 | Eval harness + 15 more golden questions + benchmark | **GATE: `docs/benchmark_report.md` with real numbers + failure analysis** |
| 8 | README + DECISIONS.md | README can sell the project |
| 9 | Docker | `docker-compose up → curl /ask` works |
| 10 | Buffer | Everything green |
