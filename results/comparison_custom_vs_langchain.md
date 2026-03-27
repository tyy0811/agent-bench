# Custom Pipeline vs. LangChain Baseline

Both pipelines use the same retrieval stack (FAISS + BM25 + RRF + cross-encoder reranker) and the same 27-question golden dataset. The only difference is the orchestration layer: custom tool-calling loop vs. LangChain's `AgentExecutor` with `create_tool_calling_agent`.

## OpenAI gpt-4o-mini

| Metric | Custom | LangChain | Delta |
|--------|--------|-----------|-------|
| Retrieval P@5 | **0.70** | 0.64 | -0.06 |
| Retrieval R@5 | 0.83 | **0.86** | +0.03 |
| Keyword Hit Rate | **0.89** | 0.85 | -0.04 |
| Citation Accuracy | 1.00 | 1.00 | tied |
| Calculator Accuracy | 2/3 | **3/3** | +1 |
| Latency p50 | **4,690 ms** | 10,118 ms | +5,428 ms |
| Cost per query | $0.0004 | $0.0003 | -$0.0001 |

## Anthropic claude-haiku-4-5

| Metric | Custom | LangChain | Delta |
|--------|--------|-----------|-------|
| Retrieval P@5 | 0.74 | **0.75** | +0.01 |
| Retrieval R@5 | **0.84** | **0.84** | tied |
| Keyword Hit Rate | **0.92** | 0.91 | -0.01 |
| Citation Accuracy | 1.00 | 1.00 | tied |
| Calculator Accuracy | **3/3** | **3/3** | tied |
| Latency p50 | **4,690 ms** | 7,165 ms | +2,475 ms |
| Cost per query | $0.0007 | $0.0046 | +$0.0039 |

## Cross-Framework Summary (all 4 configurations)

| Metric | Custom OpenAI | Custom Anthropic | LC OpenAI | LC Anthropic |
|--------|--------------|-----------------|-----------|-------------|
| P@5 | 0.70 | 0.74 | 0.64 | **0.75** |
| R@5 | 0.83 | **0.84** | **0.86** | **0.84** |
| KHR | 0.89 | **0.92** | 0.85 | 0.91 |
| Citation Acc | 1.00 | 1.00 | 1.00 | 1.00 |
| Calc | 2/3 | **3/3** | **3/3** | **3/3** |
| Latency p50 | **4,690 ms** | **4,690 ms** | 10,118 ms | 7,165 ms |
| Cost/query | **$0.0004** | $0.0007 | **$0.0003** | $0.0046 |

## Key Takeaways

1. **Retrieval quality is comparable across all configurations.** The shared retrieval stack dominates — differences are within ~0.10 on P@5/R@5 and come down to how each LLM formulates search queries.

2. **Latency is the biggest differentiator.** The custom pipeline runs at ~2x lower latency than LangChain at p50, due to framework overhead in prompt formatting, callback chains, and intermediate step serialization.

3. **Zero hallucinated citations across all four configurations.** Citation accuracy is 1.00 everywhere — the retrieval-grounded approach works regardless of orchestration layer.

4. **Anthropic slightly outperforms OpenAI on retrieval precision** in both custom (0.74 vs 0.70) and LangChain (0.75 vs 0.64), while OpenAI is cheaper per query.
