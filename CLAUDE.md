## v3.1 statistics layer rules
- stats/ is a pure package: stdlib + numpy + scipy + pandas only. Never import agent-bench modules inside stats/. Adapters live in stats_adapters/ and are the only boundary.
- Never run make targets that cost money (evaluate-full, calibrate, evaluate-judges, evaluate-langchain) or anything needing real API keys. Use fixtures and MockProvider.
- All randomness is seeded and tests pin seeds.
- Numerical tests assert against independently computed reference values with provenance comments (statsmodels or R, tool and version stated).
- Do not modify existing tests or existing eval logic; add alongside.
- ruff and mypy clean before finishing. No em or en dashes in docs. Reports regenerate with one command.
- End every session by answering: what previously agreed scope does this change violate?
