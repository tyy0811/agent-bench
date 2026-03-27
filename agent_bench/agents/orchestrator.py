"""Agent orchestrator: iterative tool-use loop.

The orchestrator builds a local message list per request.
No cross-request state. No memory.py in V1.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agent_bench.core.provider import LLMProvider
from agent_bench.core.types import (
    Message,
    Role,
    TokenUsage,
)
from agent_bench.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from agent_bench.serving.schemas import StreamEvent


class SourceReference(BaseModel):
    source: str


class AgentResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    ranked_sources: list[str] = Field(default_factory=list)
    source_chunks: list[str] = Field(default_factory=list)
    iterations: int
    tools_used: list[str] = Field(default_factory=list)
    usage: TokenUsage
    provider: str = ""
    model: str = ""
    latency_ms: float


class Orchestrator:
    """Iterative tool-use agent loop.

    Flow:
    1. Build messages = [system_prompt, user_question]
    2. Loop up to max_iterations:
       - Call provider.complete(messages, tools)
       - If no tool_calls → return answer
       - Execute each tool call, append results to messages
    3. If max iterations hit → one final complete() without tools
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_iterations: int = 3,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.max_iterations = max_iterations
        self.temperature = temperature

    async def run(
        self,
        question: str,
        system_prompt: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        history: list[dict] | None = None,
    ) -> AgentResponse:
        start = time.perf_counter()

        # Request-level retrieval settings — local variables only, no self mutation
        req_top_k = top_k
        req_strategy = strategy

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
        ]
        # Insert prior conversation history between system prompt and new question
        if history:
            for turn in history:
                role = Role.USER if turn["role"] == "user" else Role.ASSISTANT
                messages.append(Message(role=role, content=turn["content"]))
        messages.append(Message(role=Role.USER, content=question))
        tools = self.registry.get_definitions()
        all_sources: list[str] = []
        all_ranked_sources: list[str] = []
        all_source_chunks: list[str] = []
        tools_used: list[str] = []
        total_usage = TokenUsage(input_tokens=0, output_tokens=0, estimated_cost_usd=0.0)

        for iteration in range(self.max_iterations):
            response = await self.provider.complete(
                messages, tools=tools, temperature=self.temperature
            )
            # Manual accumulation (no operator overloading on Pydantic model)
            total_usage.input_tokens += response.usage.input_tokens
            total_usage.output_tokens += response.usage.output_tokens
            total_usage.estimated_cost_usd += response.usage.estimated_cost_usd

            if not response.tool_calls:
                # Final answer — no more tools needed
                latency = (time.perf_counter() - start) * 1000
                return AgentResponse(
                    answer=response.content,
                    sources=_dedup_sources(all_sources),
                    ranked_sources=all_ranked_sources,
                    source_chunks=all_source_chunks,
                    iterations=iteration + 1,
                    tools_used=tools_used,
                    usage=total_usage,
                    provider=response.provider,
                    model=response.model,
                    latency_ms=latency,
                )

            # Append assistant message with tool calls
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            # Execute each tool call, append results
            for tc in response.tool_calls:
                kwargs = dict(tc.arguments)
                # Inject request-level retrieval settings for search tool
                # All local — no shared state mutation
                if tc.name == "search_documents":
                    kwargs.setdefault("top_k", req_top_k)
                    kwargs["_strategy"] = req_strategy
                result = await self.registry.execute(tc.name, **kwargs)
                messages.append(Message(role=Role.TOOL, content=result.result, tool_call_id=tc.id))
                tools_used.append(tc.name)
                if "sources" in result.metadata:
                    all_sources.extend(result.metadata["sources"])
                if "ranked_sources" in result.metadata:
                    all_ranked_sources.extend(result.metadata["ranked_sources"])
                if "source_chunks" in result.metadata:
                    all_source_chunks.extend(result.metadata["source_chunks"])

        # Max iterations hit — force a text answer without tools
        response = await self.provider.complete(messages, tools=None, temperature=self.temperature)
        total_usage.input_tokens += response.usage.input_tokens
        total_usage.output_tokens += response.usage.output_tokens
        total_usage.estimated_cost_usd += response.usage.estimated_cost_usd

        latency = (time.perf_counter() - start) * 1000
        return AgentResponse(
            answer=response.content,
            sources=_dedup_sources(all_sources),
            ranked_sources=all_ranked_sources,
            source_chunks=all_source_chunks,
            iterations=self.max_iterations,
            tools_used=tools_used,
            usage=total_usage,
            provider=response.provider,
            model=response.model,
            latency_ms=latency,
        )


    async def run_stream(
        self,
        question: str,
        system_prompt: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        history: list[dict] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the final synthesis. Tool-use iterations are NOT streamed.

        Tool calls (retrieval, calculator) are fast (~100ms each). The slow
        part is the final LLM synthesis (~3-4s). Streaming only the final
        answer keeps the tool-use loop simple and deterministic.
        """
        from agent_bench.serving.schemas import StreamEvent

        req_top_k = top_k
        req_strategy = strategy

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
        ]
        if history:
            for turn in history:
                role = Role.USER if turn["role"] == "user" else Role.ASSISTANT
                messages.append(Message(role=role, content=turn["content"]))
        messages.append(Message(role=Role.USER, content=question))
        tools = self.registry.get_definitions()
        all_sources: list[str] = []

        # Step 1: Run tool-use loop normally (non-streamed)
        used_tools = False
        for _ in range(self.max_iterations):
            response = await self.provider.complete(
                messages, tools=tools, temperature=self.temperature
            )
            if not response.tool_calls:
                break

            used_tools = True
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )
            for tc in response.tool_calls:
                kwargs = dict(tc.arguments)
                if tc.name == "search_documents":
                    kwargs.setdefault("top_k", req_top_k)
                    kwargs["_strategy"] = req_strategy
                result = await self.registry.execute(tc.name, **kwargs)
                messages.append(
                    Message(role=Role.TOOL, content=result.result, tool_call_id=tc.id)
                )
                if "sources" in result.metadata:
                    all_sources.extend(result.metadata["sources"])

        # Step 2: Emit sources
        yield StreamEvent(
            type="sources",
            sources=[{"source": s} for s in dict.fromkeys(all_sources)],
        )

        # Step 3: Stream the final synthesis
        if used_tools:
            # Tools were used — need a fresh streaming call to synthesize
            async for chunk in self.provider.stream_complete(
                messages, temperature=self.temperature
            ):
                yield StreamEvent(type="chunk", content=chunk)
        else:
            # No tools needed — response already has the answer, emit it
            # without a redundant second LLM call
            yield StreamEvent(type="chunk", content=response.content)

        yield StreamEvent(type="done")


def _dedup_sources(sources: list[str]) -> list[SourceReference]:
    """Deduplicate source filenames, preserving order."""
    seen: set[str] = set()
    result: list[SourceReference] = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            result.append(SourceReference(source=s))
    return result
