"""Agent orchestrator: iterative tool-use loop.

The orchestrator builds a local message list per request.
No cross-request state. No memory.py in V1.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from agent_bench.core.provider import LLMProvider
from agent_bench.core.types import (
    Message,
    Role,
    TokenUsage,
)
from agent_bench.tools.registry import ToolRegistry


class SourceReference(BaseModel):
    source: str


class AgentResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    iterations: int
    tools_used: list[str] = Field(default_factory=list)
    usage: TokenUsage
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
    ) -> AgentResponse:
        start = time.perf_counter()

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=question),
        ]
        tools = self.registry.get_definitions()
        all_sources: list[str] = []
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
                    iterations=iteration + 1,
                    tools_used=tools_used,
                    usage=total_usage,
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
                result = await self.registry.execute(tc.name, **tc.arguments)
                messages.append(Message(role=Role.TOOL, content=result.result, tool_call_id=tc.id))
                tools_used.append(tc.name)
                if "sources" in result.metadata:
                    all_sources.extend(result.metadata["sources"])

        # Max iterations hit — force a text answer without tools
        response = await self.provider.complete(messages, tools=None, temperature=self.temperature)
        total_usage.input_tokens += response.usage.input_tokens
        total_usage.output_tokens += response.usage.output_tokens
        total_usage.estimated_cost_usd += response.usage.estimated_cost_usd

        latency = (time.perf_counter() - start) * 1000
        return AgentResponse(
            answer=response.content,
            sources=_dedup_sources(all_sources),
            iterations=self.max_iterations,
            tools_used=tools_used,
            usage=total_usage,
            latency_ms=latency,
        )


def _dedup_sources(sources: list[str]) -> list[SourceReference]:
    """Deduplicate source filenames, preserving order."""
    seen: set[str] = set()
    result: list[SourceReference] = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            result.append(SourceReference(source=s))
    return result
