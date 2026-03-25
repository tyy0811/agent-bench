---
title: agent-bench
emoji: "🔍"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# agent-bench

![CI](https://github.com/tyy0811/agent-bench/actions/workflows/ci.yaml/badge.svg)

Agentic RAG system with a 27-question evaluation harness, hybrid retrieval (FAISS + BM25 + RRF), tool use, and zero hallucinated citations — built from API primitives.

Built as a portfolio project demonstrating AI engineering depth: provider abstraction, evaluation infrastructure, production patterns (FastAPI, Docker, CI, structured logging).

`120 tests` | `27-question benchmark` | `$0.0004/query` | `Docker ready` | `CI green`

## Benchmark Results

Evaluated on 27 hand-crafted questions using **gpt-4o-mini** ($0.0004/query) over 16 FastAPI documentation files. Provider is swappable via config — Anthropic Claude stubbed for V2.

| Metric | Value | Notes |
|--------|-------|-------|
| Citation Accuracy | **1.00** | Zero hallucinated citations |
| Keyword Hit Rate | **0.89** | Expected facts present in answer |
| Retrieval R@5 | **0.83** | Expected sources found in top 5 |
| Retrieval P@5 | **0.70** | Hybrid RRF (FAISS + BM25) |
| Calculator Accuracy | **2/3** | LLM sometimes skips tool use |
| Grounded Refusal | **0/5** | LLM never refuses — top V2 priority |
| Latency p50 | 4,690 ms | gpt-4o-mini, single iteration |
| Cost per query | $0.0004 | ~$0.01 for full 27-question eval |

[Full benchmark report with failure analysis](docs/benchmark_report.md) | [Design decisions](DECISIONS.md)

## Live Demo

**https://nomearod-agentbench.hf.space** (Hugging Face Spaces — first request after idle may take ~30s for cold start)

```bash
# In-scope question (expect answer with sources)
curl -X POST https://nomearod-agentbench.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define a path parameter in FastAPI?"}'

# Out-of-scope question (expect grounded refusal)
curl -X POST https://nomearod-agentbench.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I cook pasta?"}'

# Health check
curl https://nomearod-agentbench.hf.space/health
```

## Quick Start (Local)

```bash
make install    # Install dependencies
make ingest     # Chunk + embed 16 FastAPI docs into FAISS + BM25
make serve      # Start FastAPI server on :8000
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define a path parameter in FastAPI?"}'
```

### With Docker

```bash
OPENAI_API_KEY=sk-... docker-compose -f docker/docker-compose.yaml up --build
```

## Architecture

```mermaid
flowchart LR
    Client -->|POST /ask| MW[Middleware<br/>request_id, timing, errors]
    MW --> Orch[Orchestrator<br/>max 3 iterations]
    Orch --> LLM[OpenAI gpt-4o-mini]
    LLM -->|tool_calls| Reg[Tool Registry]
    Reg --> Search[search_documents]
    Reg --> Calc[calculator]
    Search --> Store[Hybrid Store<br/>FAISS + BM25 + RRF]
    LLM -->|no tool_calls| Resp[AskResponse<br/>answer + sources + metadata]
```

## What This Demonstrates

- **Agentic architecture**: Iterative tool-use loop — max 3 iterations with toolless fallback, no LangChain or LlamaIndex
- **RAG pipeline**: Hybrid retrieval via Reciprocal Rank Fusion (FAISS dense + BM25 sparse), two chunking strategies (recursive + fixed-size)
- **Provider abstraction**: Swap LLM backend via config. OpenAI + Anthropic implemented, MockProvider for deterministic tests
- **Evaluation infrastructure**: 27-question golden dataset with negative/out-of-scope cases, 8 deterministic metrics + 2 LLM-judge metrics, failure analysis
- **Production patterns**: FastAPI, Docker, CI/CD (GitHub Actions), Fly.io deployment, rate limiting, provider retry with backoff, structlog structured logging, Pydantic v2 validation, 120 deterministic tests

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
make evaluate-fast   # Deterministic metrics only (needs API key)
make evaluate-full   # + LLM-judge metrics (costs more)
make benchmark       # Generate markdown report from results
```

The golden dataset contains 27 hand-crafted questions:
- 19 retrieval: 8 easy (single chunk), 7 medium (multi-chunk), 4 hard (multi-source)
- 3 calculation: questions requiring the calculator tool
- 5 out-of-scope: questions testing grounded refusal (answer not in corpus)

## Testing

```bash
make test    # 120 deterministic tests, no API keys needed
make lint    # ruff + mypy
```

All tests use MockProvider + MockEmbeddingModel. No API keys. No model downloads. CI-safe.

## Design Decisions

See [DECISIONS.md](DECISIONS.md) for rationale on building from primitives, RRF over score normalization, negative evaluation cases, deterministic eval + optional LLM judge, and more.

## V1 → V2 Improvements

| Feature | V1 | V2 | Skill Demonstrated |
|---------|----|----|-------------------|
| Grounded refusal | 0/5 | Threshold gate | Trust & safety |
| Retrieval precision | RRF only | RRF + cross-encoder | Reranking |
| Provider resilience | None | Retry + backoff | Error handling |
| Rate limiting | None | 10 RPM per IP | API hardening |
| Cloud deployment | None | HF Spaces (Docker) | Docker → production |
| CI/CD | None | GitHub Actions | Automated quality gates |

See [DECISIONS.md](DECISIONS.md) for the reasoning behind each design choice.

## Roadmap

- [x] Streaming responses (SSE for final synthesis)
- [x] SQLite conversation sessions
- [x] Anthropic provider (config swap: `provider.default: anthropic`)

*CPU-only, single-domain. Framework scales to larger corpora and additional providers.*
