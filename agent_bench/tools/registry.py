"""Tool registry: register, retrieve, and dispatch tools by name."""

from __future__ import annotations

from agent_bench.core.types import ToolDefinition
from agent_bench.tools.base import Tool, ToolOutput


class ToolRegistry:
    """Dict-based tool registry."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Retrieve a tool by name, or None if not found."""
        return self._tools.get(name)

    def get_definitions(self) -> list[ToolDefinition]:
        """Return ToolDefinitions for all registered tools."""
        return [tool.definition() for tool in self._tools.values()]

    async def execute(self, name: str, **kwargs: object) -> ToolOutput:
        """Execute a tool by name. Returns failure output for unknown tools."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutput(success=False, result=f"Unknown tool: {name}")
        return await tool.execute(**kwargs)
