# Benchmark Results — Technical Documentation Q&A

**Provider:** MockProvider (deterministic) | **Corpus:** 16 markdown files

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| Retrieval P@5 | 0.00 |
| Retrieval R@5 | 0.00 |
| Keyword Hit Rate | 0.03 |
| Source Citation Rate | 22/22 |
| Citation Accuracy | 0.00 |
| Grounded Refusal Rate | 0/5 |
| Calculator Accuracy | 0/3 |
| Latency p50 | 1 ms |
| Latency p95 | 1 ms |
| Cost per query | $0.0003 |

## By Category

| Category | Count | P@5 | R@5 | Keyword Hit | Refusal |
|----------|-------|-----|-----|-------------|---------|
| retrieval | 19 | 0.00 | 0.00 | 0.04 | n/a |
| calculation | 3 | 0.00 | 0.00 | 0.00 | n/a |
| out_of_scope | 5 | n/a | n/a | n/a | 0/5 |

## By Difficulty

| Difficulty | Count | P@5 | R@5 | Keyword Hit |
|-----------|-------|-----|-----|-------------|
| easy | 13 | 0.00 | 0.00 | 0.09 |
| medium | 10 | 0.00 | 0.00 | 0.00 |
| hard | 4 | 0.00 | 0.00 | 0.00 |

## Chunking Strategy Comparison

| Strategy | Note |
|----------|------|
| Recursive (default) | Used for this benchmark run |
| Fixed-size | Available via `--chunk-strategy fixed` in ingest. Re-run evaluation to compare. |

_To generate a comparison, run `make ingest` with each strategy and `make evaluate-fast` for each, then compare the results JSON files._

## Failure Analysis (3 worst queries)

**q001: "How do you define a path parameter in FastAPI?"**
- Retrieval P@5: 0.00
- Retrieval R@5: 0.00
- Keyword Hit Rate: 0.75
- Retrieved: ['fastapi_query_params.md', 'fastapi_query_params.md', 'fastapi_query_params.md']
- Root cause: MockProvider returned canned answer — retrieval worked but answer text doesn't match expected sources

**q002: "What is the default page size for pagination in FastAPI and what is the maximum allowed?"**
- Retrieval P@5: 0.00
- Retrieval R@5: 0.00
- Keyword Hit Rate: 0.00
- Retrieved: ['fastapi_query_params.md', 'fastapi_query_params.md', 'fastapi_query_params.md']
- Root cause: MockProvider canned response does not target this question's expected sources

**q003: "How does FastAPI handle CORS and what is the default max_age for preflight caching?"**
- Retrieval P@5: 0.00
- Retrieval R@5: 0.00
- Keyword Hit Rate: 0.00
- Retrieved: ['fastapi_query_params.md', 'fastapi_query_params.md', 'fastapi_query_params.md']
- Root cause: MockProvider canned response does not target this question's expected sources

## Per-Question Results

| ID | Cat | Diff | P@5 | R@5 | KHR | Citation | Refusal | Calc |
|----|-----|------|-----|-----|-----|----------|---------|------|
| q001 | retrieval | easy | 0.00 | 0.00 | 0.75 | 0.00 | PASS | PASS |
| q002 | retrieval | easy | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q003 | retrieval | easy | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q004 | retrieval | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q005 | retrieval | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q006 | retrieval | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q007 | calculation | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | FAIL |
| q008 | out_of_scope | easy | n/a | n/a | 0.00 | n/a | FAIL | PASS |
| q009 | out_of_scope | easy | n/a | n/a | 0.00 | n/a | FAIL | PASS |
| q010 | out_of_scope | easy | n/a | n/a | 0.00 | n/a | FAIL | PASS |
| q011 | retrieval | easy | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q012 | retrieval | easy | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q013 | retrieval | easy | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q014 | retrieval | easy | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q015 | retrieval | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q016 | retrieval | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q017 | retrieval | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q018 | retrieval | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q019 | retrieval | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q020 | calculation | medium | 0.00 | 0.00 | 0.00 | 0.00 | PASS | FAIL |
| q021 | calculation | easy | 0.00 | 0.00 | 0.00 | 0.00 | PASS | FAIL |
| q022 | retrieval | hard | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q023 | retrieval | hard | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q024 | retrieval | hard | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q025 | retrieval | hard | 0.00 | 0.00 | 0.00 | 0.00 | PASS | PASS |
| q026 | out_of_scope | easy | n/a | n/a | 0.00 | n/a | FAIL | PASS |
| q027 | out_of_scope | easy | n/a | n/a | 0.00 | n/a | FAIL | PASS |

## Configuration Snapshot

```yaml
agent:
  max_iterations: 3
  temperature: 0.0
embedding:
  cache_dir: .cache/embeddings
  model: all-MiniLM-L6-v2
evaluation:
  golden_dataset: agent_bench/evaluation/datasets/tech_docs_golden.json
  judge_provider: openai
provider:
  default: openai
  models:
    claude-sonnet-4-20250514:
      input_cost_per_mtok: 3.0
      output_cost_per_mtok: 15.0
    gpt-4o-mini:
      input_cost_per_mtok: 0.15
      output_cost_per_mtok: 0.6
rag:
  chunking:
    chunk_overlap: 64
    chunk_size: 512
    strategy: recursive
  reranker:
    enabled: false
  retrieval:
    candidates_per_system: 10
    rrf_k: 60
    strategy: hybrid
    top_k: 5
  store_path: .cache/store
serving:
  host: 0.0.0.0
  port: 8000
  request_timeout_seconds: 30
```

