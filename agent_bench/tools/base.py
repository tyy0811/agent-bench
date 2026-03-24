"""Base tool interface and output model."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from agent_bench.core.types import ToolDefinition


class ToolOutput(BaseModel):
    success: bool
    result: str
    metadata: dict = Field(default_factory=dict)


class Tool(ABC):
    """Abstract base for all tools the agent can invoke."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the tool's arguments

    @abstractmethod
    async def execute(self, **kwargs: object) -> ToolOutput:
        """Execute the tool with the given arguments."""
        ...

    def definition(self) -> ToolDefinition:
        """Return a ToolDefinition for provider format_tools()."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
