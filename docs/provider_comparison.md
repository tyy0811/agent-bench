# Provider Comparison — OpenAI vs Anthropic

Evaluated on the same 27-question golden dataset over 16 FastAPI documentation files.
Both providers use the same RAG pipeline: hybrid retrieval (FAISS + BM25 + RRF),
cross-encoder reranking, grounded refusal threshold, and identical system prompt.

**The only difference is the LLM provider.** Everything else is controlled.

## Models

| Provider | Model | Context | Pricing (input/output per 1M tokens) |
|----------|-------|---------|--------------------------------------|
| OpenAI | gpt-4o-mini | 128K | $0.15 / $0.60 |
| Anthropic | claude-haiku-4-5 | 200K | $0.80 / $4.00 |

## Retrieval Metrics

| Metric | OpenAI gpt-4o-mini | Anthropic claude-haiku | Delta |
|--------|-------------------|----------------------|-------|
| Retrieval P@5 | 0.70 | **0.74** | +0.04 |
| Retrieval R@5 | 0.83 | **0.84** | +0.01 |
| Keyword Hit Rate | 0.89 | **0.92** | +0.03 |

Haiku outperforms gpt-4o-mini on all retrieval metrics. The improvement
in P@5 (0.70 → 0.74) suggests Haiku generates more precise search queries,
which the cross-encoder reranker then amplifies.

## Cost

| Metric | OpenAI gpt-4o-mini | Anthropic claude-haiku |
|--------|-------------------|----------------------|
| Cost per query | **$0.0004** | $0.0007 |
| Full eval (27 questions) | **~$0.01** | ~$0.02 |

OpenAI is ~1.75x cheaper per query. Both are negligible for a demo.

## Qualitative Observations

- **Tool use**: Both providers correctly use the `search_documents` tool on retrieval
  questions and the `calculator` tool on calculation questions.
- **Refusal**: Both providers follow the system prompt instruction to refuse when the
  search tool returns "No relevant documents found." The refusal threshold gate fires
  identically since it operates on retrieval scores before the LLM is invoked.
- **Citation format**: Both providers follow the `[source: filename.md]` citation format
  specified in the system prompt.
- **Answer quality**: Haiku tends to produce more structured answers (numbered lists,
  code examples) while gpt-4o-mini is more concise. Both are accurate.

## How to Reproduce

```bash
# OpenAI evaluation (default config)
OPENAI_API_KEY=sk-... python scripts/evaluate.py --mode deterministic

# Anthropic evaluation
ANTHROPIC_API_KEY=sk-ant-... python scripts/evaluate.py --config configs/anthropic.yaml --mode deterministic
```

## Takeaway

The provider abstraction works as designed — switching from OpenAI to Anthropic is a
single config change (`provider.default: anthropic`). The orchestrator, tools, evaluation
harness, and serving layer are completely unchanged. Both providers produce competitive
results on the same benchmark.
