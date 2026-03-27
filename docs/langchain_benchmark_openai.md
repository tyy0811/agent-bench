# Benchmark Results — Technical Documentation Q&A

**Provider:** langchain-openai | **Corpus:** 16 markdown files

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| Retrieval P@5 | 0.64 |
| Retrieval R@5 | 0.86 |
| Keyword Hit Rate | 0.85 |
| Source Citation Rate | 20/22 |
| Citation Accuracy | 1.00 |
| Grounded Refusal Rate | 0/5 |
| Calculator Accuracy | 3/3 |
| Latency p50 | 10,118 ms |
| Latency p95 | 18,084 ms |
| Cost per query | $0.0003 |

## By Category

| Category | Count | P@5 | R@5 | Keyword Hit | Refusal |
|----------|-------|-----|-----|-------------|---------|
| retrieval | 19 | 0.68 | 0.95 | 0.89 | n/a |
| calculation | 3 | 0.33 | 0.33 | 0.58 | n/a |
| out_of_scope | 5 | n/a | n/a | n/a | 0/5 |

## By Difficulty

| Difficulty | Count | P@5 | R@5 | Keyword Hit |
|-----------|-------|-----|-----|-------------|
| easy | 13 | 0.55 | 0.88 | 0.84 |
| medium | 10 | 0.66 | 0.90 | 0.82 |
| hard | 4 | 0.75 | 0.75 | 0.94 |

## Chunking Strategy Comparison

| Strategy | Note |
|----------|------|
| Recursive (default) | Used for this benchmark run |
| Fixed-size | Available via `--chunk-strategy fixed` in ingest. Re-run evaluation to compare. |

_To generate a comparison, run `make ingest` with each strategy and `make evaluate-fast` for each, then compare the results JSON files._

## Failure Analysis (3 worst queries)

**q007: "If a paginated endpoint returns 20 items per page and there are 10,000 items total, how many total pages are there? And if the page size is changed to 30, how many pages would there be?"**
- Retrieval P@5: 0.00
- Retrieval R@5: 0.00
- Keyword Hit Rate: 0.75
- Retrieved: []
- Root cause: Retrieved sources did not match expected sources

**q021: "If the CORS max_age is 600 seconds, how many minutes does the browser cache preflight results?"**
- Retrieval P@5: 0.00
- Retrieval R@5: 0.00
- Keyword Hit Rate: 1.00
- Retrieved: []
- Root cause: Retrieved sources did not match expected sources

**q014: "What testing tools does FastAPI use, and what class provides the test client?"**
- Retrieval P@5: 0.20
- Retrieval R@5: 1.00
- Keyword Hit Rate: 0.33
- Retrieved: ['fastapi_testing.md', 'fastapi_openapi.md', 'fastapi_intro.md']
- Root cause: _(manual analysis needed)_

## Per-Question Results

| ID | Cat | Diff | P@5 | R@5 | KHR | Citation | Refusal | Calc |
|----|-----|------|-----|-----|-----|----------|---------|------|
| q001 | retrieval | easy | 0.40 | 1.00 | 0.75 | 1.00 | PASS | PASS |
| q002 | retrieval | easy | 0.80 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q003 | retrieval | easy | 1.00 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q004 | retrieval | medium | 1.00 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q005 | retrieval | medium | 1.00 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q006 | retrieval | medium | 0.60 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q007 | calculation | medium | 0.00 | 0.00 | 0.75 | 1.00 | PASS | PASS |
| q008 | out_of_scope | easy | n/a | n/a | 0.67 | n/a | FAIL | PASS |
| q009 | out_of_scope | easy | n/a | n/a | 0.00 | n/a | FAIL | PASS |
| q010 | out_of_scope | easy | n/a | n/a | 0.67 | n/a | FAIL | PASS |
| q011 | retrieval | easy | 0.60 | 1.00 | 0.67 | 1.00 | PASS | PASS |
| q012 | retrieval | easy | 0.80 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q013 | retrieval | easy | 0.60 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q014 | retrieval | easy | 0.20 | 1.00 | 0.33 | 1.00 | PASS | PASS |
| q015 | retrieval | medium | 0.20 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q016 | retrieval | medium | 0.40 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q017 | retrieval | medium | 0.80 | 1.00 | 0.75 | 1.00 | PASS | PASS |
| q018 | retrieval | medium | 0.80 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q019 | retrieval | medium | 0.80 | 1.00 | 0.75 | 1.00 | PASS | PASS |
| q020 | calculation | medium | 1.00 | 1.00 | 0.00 | 1.00 | PASS | PASS |
| q021 | calculation | easy | 0.00 | 0.00 | 1.00 | 1.00 | PASS | PASS |
| q022 | retrieval | hard | 0.40 | 0.33 | 0.75 | 1.00 | PASS | PASS |
| q023 | retrieval | hard | 0.80 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q024 | retrieval | hard | 0.80 | 1.00 | 1.00 | 1.00 | PASS | PASS |
| q025 | retrieval | hard | 1.00 | 0.67 | 1.00 | 1.00 | PASS | PASS |
| q026 | out_of_scope | easy | n/a | n/a | 0.67 | n/a | FAIL | PASS |
| q027 | out_of_scope | easy | n/a | n/a | 0.67 | n/a | FAIL | PASS |

