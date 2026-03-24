# Retrieval Gate Check

**Store:** 207 chunks, 16 sources

| ID | Category | Expected Source | Top-5 Sources | Recall@5 | Result |
|-----|----------|----------------|---------------|----------|--------|
| q001 | retrieval | fastapi_path_params.md | fastapi_path_params.md, fastapi_query_params.md, fastapi_request_body.md | 1.00 | PASS |
| q002 | retrieval | fastapi_pagination.md | fastapi_pagination.md, fastapi_path_params.md | 1.00 | PASS |
| q003 | retrieval | fastapi_middleware.md | fastapi_middleware.md | 1.00 | PASS |
| q004 | retrieval | fastapi_security.md | fastapi_security.md | 1.00 | PASS |
| q005 | retrieval | fastapi_deployment.md | fastapi_deployment.md | 1.00 | PASS |
| q006 | retrieval | fastapi_dependencies.md | fastapi_dependencies.md | 1.00 | PASS |
| q007 | calculation | fastapi_pagination.md | fastapi_pagination.md | 1.00 | PASS |
| q008 | out_of_scope | (none) | fastapi_deployment.md, fastapi_intro.md, fastapi_openapi.md | n/a | N/A |
| q009 | out_of_scope | (none) | fastapi_websockets.md, fastapi_background_tasks.md | n/a | N/A |
| q010 | out_of_scope | (none) | fastapi_openapi.md, fastapi_response_model.md | n/a | N/A |

**Avg Recall@5 (positive only):** 1.00
**Gate:** PASS (threshold >= 0.5)
