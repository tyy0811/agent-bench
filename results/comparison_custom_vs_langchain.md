# Custom Pipeline vs. LangChain Baseline

Both pipelines use identical configuration: hybrid retrieval (FAISS + BM25 + RRF), cross-encoder reranking enabled, 27-question golden dataset, temperature 0, max 3 tool-use iterations. The only variable is the orchestration layer: custom tool-calling loop vs. LangChain `AgentExecutor` with `create_tool_calling_agent`.

Custom pipeline numbers are from `docs/provider_comparison.md` (V2, reranker enabled). LangChain numbers are from runs using the same `configs/default.yaml` (reranker enabled).

## OpenAI gpt-4o-mini

| Metric | Custom | LangChain | Delta |
|--------|--------|-----------|-------|
| Retrieval P@5 | **0.70** | 0.64 | -0.06 |
| Retrieval R@5 | 0.83 | **0.86** | +0.03 |
| Keyword Hit Rate | **0.89** | 0.85 | -0.04 |
| Citation Accuracy | 1.00 | 1.00 | tied |
| Calculator Accuracy | 2/3 | **3/3** | +1 |
| Cost per query | $0.0004 | $0.0003 | -$0.0001 |

## Anthropic claude-haiku-4-5

| Metric | Custom | LangChain | Delta |
|--------|--------|-----------|-------|
| Retrieval P@5 | 0.74 | **0.75** | +0.01 |
| Retrieval R@5 | **0.84** | **0.84** | tied |
| Keyword Hit Rate | **0.92** | 0.91 | -0.01 |
| Citation Accuracy | 1.00 | 1.00 | tied |
| Calculator Accuracy | **3/3** | **3/3** | tied |
| Cost per query | **$0.0007** | $0.0046 | +$0.0039 |

## Cross-Framework Summary (all 4 configurations)

| Metric | Custom OpenAI | Custom Anthropic | LC OpenAI | LC Anthropic |
|--------|--------------|-----------------|-----------|-------------|
| P@5 | 0.70 | 0.74 | 0.64 | **0.75** |
| R@5 | 0.83 | **0.84** | **0.86** | **0.84** |
| KHR | 0.89 | **0.92** | 0.85 | 0.91 |
| Citation Acc | 1.00 | 1.00 | 1.00 | 1.00 |
| Calc | 2/3 | **3/3** | **3/3** | **3/3** |
| Cost/query | **$0.0004** | $0.0007 | **$0.0003** | $0.0046 |

## Key Takeaways

1. **Retrieval quality is comparable across all configurations.** The shared retrieval stack dominates — differences are within ~0.10 on P@5/R@5 and come down to how each orchestration layer formats prompts and how each LLM formulates search queries.

2. **Answer quality is close.** Keyword hit rate is within 7 points across all four, citation accuracy is 1.00 everywhere (zero hallucinated citations), and calculator use is correct in 3 of 4 configurations.

3. **Zero hallucinated citations across all four configurations.** The retrieval-grounded approach works regardless of orchestration layer or provider.

4. **Anthropic slightly outperforms OpenAI on retrieval precision** in both custom (0.74 vs 0.70) and LangChain (0.75 vs 0.64), while OpenAI is cheaper per query.

## Limitations

- Latency is not directly compared because the custom and LangChain runs were executed at different times on the same machine. Network conditions, API server load, and local resource contention differ between runs.
- Token cost for LangChain Anthropic ($0.0046/query) is higher than custom Anthropic ($0.0007/query), likely because the `AgentExecutor` makes additional LLM calls for intermediate reasoning steps. This reflects a real framework cost difference.
