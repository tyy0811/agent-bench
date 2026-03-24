# Benchmark Results — Technical Documentation Q&A

**Provider:** openai | **Corpus:** 16 markdown files

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| Retrieval P@5 | 0.70 |
| Retrieval R@5 | 0.83 |
| Keyword Hit Rate | 0.89 |
| Source Citation Rate | 20/22 |
| Citation Accuracy | 1.00 |
| Grounded Refusal Rate | 0/5 |
| Calculator Accuracy | 2/3 |
| Latency p50 | 4,690 ms |
| Latency p95 | 14,991 ms |
| Cost per query | $0.0004 |

## By Category

| Category | Count | P@5 | R@5 | Keyword Hit | Refusal |
|----------|-------|-----|-----|-------------|---------|
| retrieval | 19 | 0.76 | 0.91 | 0.88 | n/a |
| calculation | 3 | 0.33 | 0.33 | 0.92 | n/a |
| out_of_scope | 5 | n/a | n/a | n/a | 0/5 |

## By Difficulty

| Difficulty | Count | P@5 | R@5 | Keyword Hit |
|-----------|-------|-----|-----|-------------|
| easy | 13 | 0.50 | 0.88 | 0.84 |
| medium | 10 | 0.78 | 0.90 | 0.90 |
| hard | 4 | 0.90 | 0.58 | 0.94 |

## Chunking Strategy Comparison

| Strategy | Note |
|----------|------|
| Recursive (default) | Used for this benchmark run |
| Fixed-size | Available via `--chunk-strategy fixed` in ingest. Re-run evaluation to compare. |

_To generate a comparison, run `make ingest` with each strategy and `make evaluate-fast` for each, then compare the results JSON files._

## Failure Analysis (3 worst queries)

**q007: "If a paginated endpoint returns 20 items per page and there are 10,000 items total, how many total pages are there? And if the page size is changed to 30, how many pages would there be?"**
- Retrieval P@5: 0.00 | R@5: 0.00 | KHR: 0.75
- Retrieved: []
- Root cause: The LLM answered the calculation from its parametric knowledge without calling search_documents first. The answer was correct (keywords hit 0.75) but no retrieval occurred, so P@5/R@5 are zero. Fix: stronger system prompt forcing search before calculation, or a retrieval-first orchestrator policy.

**q021: "If the CORS max_age is 600 seconds, how many minutes does the browser cache preflight results?"**
- Retrieval P@5: 0.00 | R@5: 0.00 | KHR: 1.00
- Retrieved: []
- Root cause: Same pattern as q007 — the LLM computed 600/60=10 from parametric knowledge, skipping retrieval entirely. The answer was fully correct (KHR 1.00) but ungrounded. Fix: same as above — enforce tool use before answering.

**q001: "How do you define a path parameter in FastAPI?"**
- Retrieval P@5: 0.20 | R@5: 1.00 | KHR: 0.75
- Retrieved: ['fastapi_query_params.md', 'fastapi_path_params.md', 'fastapi_query_params.md']
- Root cause: BM25 ranked fastapi_query_params.md higher than fastapi_path_params.md due to shared vocabulary ("parameters", "FastAPI"). The correct source was retrieved (R@5 = 1.00) but at rank 2, diluting precision. Fix: cross-encoder reranking or query-specific term weighting would help disambiguate.

## Per-Question Results

| ID | Cat | Diff | P@5 | R@5 | KHR | Citation | Refusal | Calc |
|----|-----|------|-----|-----|-----|----------|---------|------|
| q001 | retrieval | easy | 0.20 | 1.00 | 0.75 | 1.00 | PASS | PASS |
| q002 | retrieval | easy | 1.00 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q003 | retrieval | easy | 0.80 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q004 | retrieval | medium | 1.00 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q005 | retrieval | medium | 1.00 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q006 | retrieval | medium | 1.00 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q007 | calculation | medium | 0.00 | 0.00 | 0.75 | 1.00 | PASS | PASS |
| q008 | out_of_scope | easy | n/a | n/a | 0.67 | n/a | FAIL | PASS |
| q009 | out_of_scope | easy | n/a | n/a | 0.67 | n/a | FAIL | PASS |
| q010 | out_of_scope | easy | n/a | n/a | 0.67 | n/a | FAIL | PASS |
| q011 | retrieval | easy | 0.80 | 1.00 | 0.67 | 1.00 | PASS | PASS |
| q012 | retrieval | easy | 0.60 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q013 | retrieval | easy | 0.20 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q014 | retrieval | easy | 0.40 | 1.00 | 0.33 | 1.00 | PASS | PASS |
| q015 | retrieval | medium | 0.60 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q016 | retrieval | medium | 0.60 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q017 | retrieval | medium | 0.80 | 1.00 | 0.50 | 1.00 | PASS | PASS |
| q018 | retrieval | medium | 0.80 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q019 | retrieval | medium | 1.00 | 1.00 | 0.75 | 1.00 | PASS | PASS |
| q020 | calculation | medium | 1.00 | 1.00 | 1.00 | 1.00 | PASS | FAIL |
| q021 | calculation | easy | 0.00 | 0.00 | 1.00 | 1.00 | PASS | PASS |
| q022 | retrieval | hard | 0.60 | 0.33 | 0.75 | 1.00 | PASS | PASS |
| q023 | retrieval | hard | 1.00 | 0.67 | 1.00 | 1.00 | PASS | PASS |
| q024 | retrieval | hard | 1.00 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q025 | retrieval | hard | 1.00 | 0.33 | 1.00 | 1.00 | PASS | PASS |
| q026 | out_of_scope | easy | n/a | n/a | 0.67 | n/a | FAIL | PASS |
| q027 | out_of_scope | easy | n/a | n/a | 0.67 | n/a | FAIL | PASS |

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

