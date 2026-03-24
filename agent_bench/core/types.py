"""Shared type definitions used across agent-bench."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class Message(BaseModel):
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class CompletionResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage
    provider: str
    model: str
    latency_ms: float
