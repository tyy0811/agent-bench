"""Calculator tool: safe math expression evaluation via simpleeval."""

from __future__ import annotations

from simpleeval import simple_eval

from agent_bench.tools.base import Tool, ToolOutput


class CalculatorTool(Tool):
    """Evaluate mathematical expressions safely."""

    name = "calculator"
    description = "Evaluate a mathematical expression. Supports +, -, *, /, **, %, and parentheses."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    }

    async def execute(self, **kwargs: object) -> ToolOutput:
        expression = str(kwargs.get("expression", ""))
        if not expression:
            return ToolOutput(success=False, result="No expression provided")
        try:
            result = simple_eval(expression)
            return ToolOutput(success=True, result=str(result))
        except Exception:
            return ToolOutput(success=False, result=f"Could not evaluate: {expression}")
