# Provider Comparison: API vs Self-Hosted

Evaluated on the same 27-question golden dataset over 16 FastAPI documentation files.
All providers use hybrid retrieval (FAISS + BM25 + RRF), cross-encoder reranking,
grounded refusal threshold, and identical system prompt.

**Note:** The self-hosted config differs from API configs in two ways to accommodate
the 7B model's smaller context window (8192 tokens) and weaker instruction following:
`max_iterations=1` (vs 3) and `top_k=3` (vs 5). This means the self-hosted row is
**not a controlled comparison** — it reflects realistic operating constraints for a
7B model, not an apples-to-apples provider swap. The API providers are directly
comparable to each other.

## Results

| Provider | Model | Iterations | top_k | P@5 | R@5 | Citation Acc | Latency p50 (ms) | Cost/query |
|----------|-------|-----------|-------|-----|-----|--------------|-------------------|------------|
| OpenAI (API) | gpt-4o-mini | 3 | 5 | 0.70 | 0.83 | 1.00 | 4,690 | $0.0004 |
| Anthropic (API) | claude-haiku-4-5 | 3 | 5 | 0.74 | 0.84 | 1.00 | 5,120 | $0.0007 |
| Self-hosted (Modal) | Mistral-7B-Instruct-v0.3 | 1 | 3 | 0.05 | 0.05 | 0.14 | 6,709 | $0.0031 |

## Why Mistral-7B Scores So Differently

The gap between API providers and self-hosted Mistral-7B compounds from three factors,
ordered by causal priority:

**1. No native tool calling (upstream of everything else).** vLLM 0.6.6 with Mistral-7B
doesn't support OpenAI-format `tool_calls`. The provider falls back to injecting tool
descriptions into the system prompt and parsing JSON from the model's text output.
Mistral-7B frequently produces malformed JSON or calls tools with vague queries like
`"search"` instead of `"FastAPI dependency injection"` — so the retrieval stage gets
garbage queries, and P@5 collapses to 0.05.

**2. Forced single iteration (context window constraint).** API providers get 3 iterations
(call tool, read result, refine, repeat). Mistral-7B is limited to 1 because each iteration
adds ~2K tokens of tool results, and the 8K context window fills up. One shot at picking the
right tool and query, no opportunity to refine.

**3. Weak instruction following (residual even when 1 and 2 don't bite).** Even when
Mistral-7B calls the right tool with a reasonable query, it struggles to follow the citation
format (`[source: filename.md]`) specified in the system prompt. Citation accuracy is 0.14 —
it's not hallucinating sources, it's mostly omitting them.

**Keyword hit rate (0.61) is the interesting signal.** The model *sometimes* generates queries
with relevant keywords, meaning it has semantic understanding of the questions but can't
translate that into well-formed tool calls. This is exactly the gap between "understands
language" and "can operate as an agent."

**Cost is counterintuitively higher.** Self-hosted Mistral-7B costs $0.0031/query vs
$0.0004 for OpenAI gpt-4o-mini — despite being self-hosted. Modal A10G time is billed
per GPU-second, and Mistral-7B takes longer per query while producing worse results.
The cost advantage of self-hosted only materializes at high throughput with batched
requests and sustained GPU utilization, not at single-query evaluation scale.

## Infrastructure

| Config | Cold start | Warm latency p50 | GPU | Infra |
|--------|-----------|-------------------|-----|-------|
| OpenAI | N/A | 4,690 ms | N/A | Managed API |
| Anthropic | N/A | 5,120 ms | N/A | Managed API |
| Self-hosted (Modal) | ~90s | 6,709 ms | A10G (24GB) | Serverless GPU |

## How to Reproduce

```bash
# OpenAI evaluation
OPENAI_API_KEY=sk-... python scripts/evaluate.py --mode deterministic

# Anthropic evaluation
ANTHROPIC_API_KEY=sk-ant-... python scripts/evaluate.py --config configs/anthropic.yaml --mode deterministic

# Self-hosted evaluation (requires Modal deployment + HF secret)
pip install -e ".[modal]"
modal secret create huggingface-secret HF_TOKEN=hf_...
modal deploy modal/serve_vllm.py
export MODAL_VLLM_URL=https://your--agent-bench-vllm-serve.modal.run/v1
python scripts/evaluate.py --config configs/selfhosted_modal.yaml --mode deterministic

# All providers at once
make benchmark-all
```

## Known Limitations & Future Work

The self-hosted benchmark is not a controlled comparison. Three specific constraints
disadvantage Mistral-7B beyond its inherent model quality:

1. **Prompt-based tool calling (fixable).** vLLM 0.6.6 was pinned to work around
   `huggingface_hub` and `transformers` dependency conflicts. Newer vLLM versions (0.8+)
   support native Mistral tool calling via `--enable-auto-tool-choice --tool-call-parser mistral`.
   This would eliminate the malformed-JSON failure mode that drives P@5 to 0.05.

2. **Artificially low context window (fixable).** Mistral-7B supports 32K context natively,
   but `max_model_len` is set to 8K to fit A10G memory at `gpu_memory_utilization=0.85`.
   Bumping to 16K (with `0.90` utilization) would likely allow restoring `max_iterations=3`
   and `top_k=5` to match the API configs — making the comparison controlled.

3. **Model scale (architectural).** Even with fixes 1 and 2, a 7B model will underperform
   gpt-4o-mini and claude-haiku on multi-step agentic tasks. A fairer model-size comparison
   would use Mixtral-8x7B or Llama-3-70B (requiring A100 80GB). This would refine the
   model-size floor estimate but not change the architectural conclusion.

## Takeaway

The provider abstraction works as designed — switching providers is a single config change.
API models dominate on quality metrics, but the self-hosted path demonstrates end-to-end
inference serving: vLLM on Modal (serverless A10G), OpenAI-compatible endpoint, identical
evaluation harness. The quality gap is a combination of model scale and infrastructure
constraints, both of which are documented and addressable.

---

Generated by `modal/run_benchmark.py`
