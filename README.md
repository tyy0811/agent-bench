# agent-bench

> Evaluation-first agentic RAG system built from API primitives.
> One provider, one domain, one API, one benchmark report.

## Benchmark Results

| Metric | Value |
|--------|-------|
| Retrieval P@5 | _run with real provider_ |
| Retrieval R@5 | _run with real provider_ |
| Citation Accuracy | _run with real provider_ |
| Grounded Refusal | _/5_ |
| Answer Faithfulness | _run with real provider_ |
| Latency p50 | _run with real provider_ |
| Cost per query | _run with real provider_ |

[Full benchmark report](docs/benchmark_report.md) (currently MockProvider proof-of-structure; real numbers after `make evaluate-fast` with API key)

## Quick Start

```bash
# Install (uses the pinned interpreter from Makefile)
make install

# Ingest the documentation corpus
make ingest

# Start the API server
make serve

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define a path parameter in FastAPI?"}'
```

### With Docker

```bash
OPENAI_API_KEY=sk-... docker-compose -f docker/docker-compose.yaml up --build
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define a path parameter in FastAPI?"}'
```

## Architecture

```
Client
  |
  v
POST /ask  ──>  Middleware (request_id, timing, error handling)
  |
  v
Orchestrator  ──>  Loop (max 3 iterations):
  |                   |
  |                   v
  |              LLM Provider (OpenAI gpt-4o-mini)
  |                   |
  |              tool_calls?  ──yes──>  Tool Registry
  |                   |                     |
  |                   no               search_documents ──> Retriever
  |                   |                     |                    |
  |                   v                calculator         Embedder + HybridStore
  |              Final answer                            (FAISS + BM25 + RRF)
  |
  v
AskResponse { answer, sources[], metadata }
```

## What This Demonstrates

- **Agentic architecture**: Iterative tool-use loop with plan, execute, verify — max 3 iterations with toolless fallback
- **RAG pipeline**: Hybrid retrieval via Reciprocal Rank Fusion (FAISS dense + BM25 sparse), two chunking strategies (recursive + fixed-size)
- **Provider abstraction**: Swap LLM backend via config. OpenAI implemented, Anthropic stubbed, MockProvider for deterministic tests
- **Evaluation infrastructure**: 27-question golden dataset with negative/out-of-scope cases, 8 deterministic metrics + 2 LLM-judge metrics, failure analysis, benchmark report
- **Production patterns**: FastAPI, Docker, structlog structured logging, Pydantic v2 validation, CI with 97 deterministic tests, request-level metrics

## What This Is Not

- Not a framework (it's a focused demonstration)
- Not cloud-deployed (Docker-local is the scope)
- Not GPU-dependent (runs on a CPU laptop)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Ask a question, get answer with sources |
| `/health` | GET | Store stats, provider status, uptime |
| `/metrics` | GET | Request count, latency p50/p95, cost |

### POST /ask

```json
{
  "question": "How do I define a path parameter in FastAPI?",
  "top_k": 5,
  "retrieval_strategy": "hybrid"
}
```

Response:

```json
{
  "answer": "Path parameters in FastAPI are defined using curly braces...",
  "sources": [{"source": "fastapi_path_params.md"}],
  "metadata": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "iterations": 2,
    "tools_used": ["search_documents"],
    "latency_ms": 1234.5,
    "token_usage": {"input_tokens": 500, "output_tokens": 150, "estimated_cost_usd": 0.0002},
    "request_id": "abc-123"
  }
}
```

## Evaluation

```bash
# Deterministic metrics only (free, CI-safe)
make evaluate-fast

# Deterministic + LLM-judge metrics (costs money)
make evaluate-full

# Generate benchmark report
make benchmark
```

The golden dataset (`agent_bench/evaluation/datasets/tech_docs_golden.json`) contains 27 hand-crafted questions:
- 19 retrieval: 8 easy (single chunk), 7 medium (multi-chunk), 4 hard (multi-source)
- 3 calculation: questions requiring the calculator tool
- 5 out-of-scope: questions testing grounded refusal (answer not in corpus)

Metrics measured: Retrieval P@5, R@5, keyword hit rate, source citation rate, citation accuracy, grounded refusal rate, calculator accuracy, latency, cost.

## Testing

```bash
make test    # 97 deterministic tests, no API keys needed
make lint    # ruff + mypy
```

All tests use MockProvider + MockEmbeddingModel. No API keys. No model downloads. CI-safe.

## Project Structure

```
agent_bench/
  core/       # Provider abstraction, config, types
  agents/     # Orchestrator (tool-use loop, no persistent memory)
  tools/      # Registry, search_documents, calculator
  rag/        # Chunker, embedder, FAISS+BM25 store, retriever
  evaluation/ # Harness, metrics, report generator, golden dataset
  serving/    # FastAPI app, routes, schemas, middleware
```

## Design Decisions

See [DECISIONS.md](DECISIONS.md) for rationale on:
- Building from primitives (no LangChain)
- Reciprocal Rank Fusion over score normalization
- One provider in V1 with interface for extensibility
- Negative evaluation cases for grounded refusal
- Deterministic eval + optional LLM judge

## V2 Roadmap

- [ ] Second provider (Anthropic Claude)
- [ ] Cross-encoder reranking (feature-flagged, config ready)
- [ ] Research paper domain (PDF ingestion)
- [ ] Streaming responses
- [ ] Conversation sessions with SQLite persistence + conversation_id
- [ ] Provider comparison benchmark
