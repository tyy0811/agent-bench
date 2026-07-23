# Cost claim re-verification, 2026-07

Re-verification of the single-run cost finding that the LangChain Anthropic
pipeline costs about 6.6x per query versus the custom Anthropic pipeline
($0.0046 vs $0.0007, results/comparison_custom_vs_langchain.md). The claim
was measured months before this check; the question is whether it still
describes current library versions.

## Verdict

The claim is correct as originally scoped but must be version-stamped:

- **As measured: the pre-1.0 AgentExecutor stack.** The comparison ran
  LangChain `AgentExecutor` with `create_tool_calling_agent` over
  `langchain-anthropic`'s `ChatAnthropic`. The exact installed minor
  versions at measurement time were not captured (pyproject pins ranges,
  no lockfile); the surviving repo environment resolves to 0.3.28/0.3.22
  at re-verification time.
- **As of langchain 1.3.14 (current on 2026-07-22): the measured adapter path
  no longer exists.** `from langchain.agents import AgentExecutor` raises
  ImportError; the 1.x line replaced the AgentExecutor agent stack with the
  LangGraph-based `create_agent`. The 6.6x figure therefore cannot be
  reproduced or refuted on current langchain; it is a statement about the
  pre-1.0 line, which is what this repo pins (`langchain>=0.2.0,<1.0.0`
  in pyproject.toml).
- **Mechanism narrowed, not resolved.** A zero-API structural probe (below)
  shows both arms resend their fixed prompt and tool-schema material on
  every iteration with similar fixed overhead (within 2 percent in a canned
  two-call trajectory), so per-iteration payload resending does not explain
  the cost gap. What does remains unresolved without per-call live traces:
  extra LLM calls or iterations (the original artifact's stated hypothesis,
  qualified there as "likely"), different real response and output lengths,
  or accumulated history differences are all still possible.

## Versions

| Package | Repo env at re-verification (pre-1.0 line) | Current (2026-07-22, isolated venv) |
|---------|---------------------------------------------|--------------------------------------|
| langchain | 0.3.28 | 1.3.14 |
| langchain-anthropic | 0.3.22 | 1.5.0 |
| langchain-openai | 0.3.35 | 1.4.0 |
| langchain-core | 0.3.83 | 1.5.0 |
| anthropic | repo env | 0.117.1 |
| Python | 3.11.4 | 3.11.4 |

Isolated venv: `.venv-phase2/` (gitignored), built with
`pip install -e . && pip install -U langchain langchain-anthropic langchain-openai`.
The upgrade violates the repo's `<1.0.0` pin by construction; that conflict is
the point of the check.

Provenance caveat: the original measurement predates this check and did not
record package versions, so the left column is the surviving environment's
resolution of the range pins, not a lock captured at measurement time. What
is certain is that the measurement used the pre-1.0 AgentExecutor stack,
because that is the only path the code has ever had.

## Method

### 1. Adapter-path existence under current versions

```
.venv-phase2/bin/python -c "from agent_bench.langchain_baseline.agent import create_langchain_agent"
ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'
```

Outcome (c) of the pre-declared re-verification plan: structural change.

### 2. Zero-API structural probe (pinned 0.3.x env)

The anthropic SDK's `Messages.create` (sync and async) was monkeypatched to
capture every request payload and return canned responses: call 1 returns a
`tool_use` block for `search_documents`, call 2 returns an `end_turn` text
block. One question was driven through both arms with the same two tools
and byte-identical canned tool output (five formatted passages), forcing an
identical two-call trajectory. No network; fake API key. ChatAnthropic was
forced through its non-streaming path (`disable_streaming=True`); payload
structure is identical either way.

The two arms carry the same two tools (search_documents, calculator) but
each arm's own serialization of them: the custom arm's Anthropic-format
schemas total 751 chars, LangChain's conversion 649. The system prompts
also differ by construction (446 chars custom template vs 532 chars
LangChain default). What is byte-identical across arms is the canned tool
OUTPUT and the forced trajectory; payload differences beyond that are
framework structure.

Per-call capture, custom arm (Orchestrator + AnthropicProvider):

```
{"call": 1, "system_chars": 446, "n_tools": 2, "tools_json_chars": 751, "n_messages": 1, "messages_json_chars": 78}
{"call": 2, "system_chars": 446, "n_tools": 2, "tools_json_chars": 751, "n_messages": 3, "messages_json_chars": 2592}
```

Per-call capture, LangChain arm (AgentExecutor + create_tool_calling_agent):

```
{"call": 1, "system_chars": 532, "n_tools": 2, "tools_json_chars": 649, "n_messages": 1, "messages_json_chars": 78}
{"call": 2, "system_chars": 532, "n_tools": 2, "tools_json_chars": 649, "n_messages": 3, "messages_json_chars": 2611}
```

Reading: fixed per-call overhead (system + tools, resent every iteration) is
1197 chars for the custom arm and 1181 for LangChain, a difference under 2
percent in this canned two-call trajectory. Message-history growth is
likewise equivalent given identical forced behavior (2592 vs 2611 chars).
Two calls each. Conclusion: fixed payload overhead does not explain the
6.6x gap. What does cannot be determined from this probe; distinguishing
extra calls, longer real outputs, or accumulated-history differences
requires per-call live traces from a paid run.

## What was deliberately not run

The magnitude re-measurement (both arms, same 27-question set, single run,
pre-1.0 env) requires live Anthropic calls and is excluded from agent
sessions by the repo's paid boundary. It is optional given the verdict
above (the claim is version-stamped either way), and if wanted it is one
probe plus two commands in the repo env, roughly $0.15 total at the
original per-query costs:

```
python scripts/run_langchain_eval.py --provider anthropic --max-questions 1   # probe one item first
python scripts/run_langchain_eval.py --provider anthropic                     # ~27 x $0.0046
python scripts/evaluate.py --config configs/anthropic.yaml --mode deterministic  # custom arm, same set
```

Mode note: the custom arm must run `--mode deterministic`. Full mode
constructs an LLM judge provider and makes one judge call per question,
which is spend outside this comparison and would also contaminate the
cost-per-query readout. Probe-first limitation: `scripts/evaluate.py` has
no `--max-questions` flag, so the one-item probe covers only the LangChain
arm; the custom arm's first live call is the full 27-question run (~$0.02
at the original per-query cost).

If the refreshed ratio differs materially from 6.6x, regenerate
`agent_bench/serving/static/reveal_anchor.json` through
`scripts/build_reveal_anchor.py` after updating
`results/comparison_custom_vs_langchain.md`; the README prose figure must
then be updated in the same change (it is currently plain text, pinned by
prose review rather than a checker).

## Surfaces carrying the claim, and how each is stamped

- `README.md` key-insight blockquote: adjacent version-stamp paragraph
  appended inside the blockquote; the 6.6x number itself is unchanged.
- Dashboard (`agent_bench/serving/static/index.html`): version note added
  to the reveal cost caption, the meta-strip chip, and the cost finding
  card. The card's original mechanism sentence (extra re-sends per
  iteration) predated the probe and was contradicted by it; on Jane's
  instruction (2026-07-22) it now states the probe result and that the
  remaining candidate mechanisms need per-call live traces.
- `agent_bench/serving/static/reveal_anchor.json`: unchanged; provenance
  string "single-run" is pinned by tests and the ratio derives from the
  comparison artifact's cost row, which is unchanged.
- `results/comparison_custom_vs_langchain.md`: dated version-stamp
  addendum appended; original text and numbers untouched.

## Probe script

Archived inline for reproducibility; run with the repo env python
(`/usr/local/opt/python@3.11/bin/python3.11`) as `probe_payloads.py custom`
or `probe_payloads.py langchain`.

```python
"""Zero-API structural probe for the 6.6x cost-claim mechanism."""

import asyncio
import json
import os
import sys

os.environ["ANTHROPIC_API_KEY"] = "fake-key-structural-probe"

CAPTURED: list[dict] = []

QUESTION = "How does FastAPI handle dependency injection?"
CANNED_PASSAGE = (
    "FastAPI provides a Depends() function for dependency injection. "
    "Dependencies are declared as function parameters and resolved per request. "
) * 3


def _canned_message(first_call: bool):
    from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

    if first_call:
        content = [
            ToolUseBlock(
                type="tool_use",
                id="toolu_probe_01",
                name="search_documents",
                input={"query": "dependency injection"},
            )
        ]
        stop_reason = "tool_use"
    else:
        content = [
            TextBlock(
                type="text",
                text=(
                    "FastAPI uses Depends() for dependency injection "
                    "[source: fastapi_dependencies.md]."
                ),
            )
        ]
        stop_reason = "end_turn"
    return Message(
        id="msg_probe_01",
        content=content,
        model="claude-haiku-4-5-20251001",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=1000, output_tokens=50),
    )


def _capture(kwargs: dict):
    CAPTURED.append(kwargs)
    return _canned_message(first_call=(len(CAPTURED) == 1))


def _install_patch():
    from anthropic.resources.messages import AsyncMessages, Messages

    async def fake_acreate(self, **kwargs):
        return _capture(kwargs)

    def fake_create(self, **kwargs):
        return _capture(kwargs)

    AsyncMessages.create = fake_acreate
    Messages.create = fake_create


def _report():
    print(f"calls: {len(CAPTURED)}")
    for i, kw in enumerate(CAPTURED, 1):
        system = kw.get("system") or ""
        system_s = system if isinstance(system, str) else json.dumps(system)
        tools = kw.get("tools") or []
        messages = kw.get("messages") or []
        print(json.dumps({
            "call": i,
            "model": kw.get("model"),
            "system_chars": len(system_s),
            "n_tools": len(tools),
            "tools_json_chars": len(json.dumps(tools, default=str)),
            "n_messages": len(messages),
            "messages_json_chars": len(json.dumps(messages, default=str)),
            "max_tokens": kw.get("max_tokens"),
        }))


async def run_custom():
    from agent_bench.agents.orchestrator import Orchestrator
    from agent_bench.core.prompts import format_system_prompt
    from agent_bench.core.provider import AnthropicProvider
    from agent_bench.tools.base import ToolOutput
    from agent_bench.tools.calculator import CalculatorTool
    from agent_bench.tools.registry import ToolRegistry
    from agent_bench.tools.search import SearchTool

    class CannedSearchTool(SearchTool):
        def __init__(self):  # real name/description/parameters, no retriever
            pass

        async def execute(self, **kwargs):
            formatted = "\n\n".join(
                f"[{i}] (fastapi_dependencies.md): {CANNED_PASSAGE}"
                for i in range(1, 6)
            )
            return ToolOutput(success=True, result=formatted, metadata={
                "sources": ["fastapi_dependencies.md"],
                "ranked_sources": ["fastapi_dependencies.md"] * 5,
                "source_chunks": [CANNED_PASSAGE] * 5,
            })

    registry = ToolRegistry()
    registry.register(CannedSearchTool())
    registry.register(CalculatorTool())
    orch = Orchestrator(
        provider=AnthropicProvider(), registry=registry, max_iterations=3
    )
    await orch.run(QUESTION, system_prompt=format_system_prompt("FastAPI"))
    _report()


async def run_langchain():
    from langchain_core.documents import Document

    import agent_bench.langchain_baseline.agent as lc_agent_mod
    from agent_bench.langchain_baseline.agent import create_langchain_agent
    from agent_bench.langchain_baseline.tools import (
        LangChainSearchTool,
        create_calculator_tool,
    )

    _orig = lc_agent_mod.ChatAnthropic
    lc_agent_mod.ChatAnthropic = lambda **kw: _orig(disable_streaming=True, **kw)

    class StubRetriever:
        async def ainvoke(self, query):
            return [
                Document(
                    page_content=CANNED_PASSAGE,
                    metadata={"source": "fastapi_dependencies.md"},
                )
                for _ in range(5)
            ]

    search = LangChainSearchTool(StubRetriever())
    tools = [search.as_tool(), create_calculator_tool()]
    agent = create_langchain_agent(tools, provider="anthropic", max_iterations=3)
    await agent.ainvoke({"input": QUESTION})
    _report()


if __name__ == "__main__":
    _install_patch()
    if sys.argv[1] == "custom":
        asyncio.run(run_custom())
    else:
        asyncio.run(run_langchain())
```
