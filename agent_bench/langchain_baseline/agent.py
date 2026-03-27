"""LangChain tool-calling agent factory.

Uses native function calling (not ReAct text parsing) for a fair
apples-to-apples comparison with the custom pipeline.
"""

from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

_DEFAULT_SYSTEM_PROMPT = (
    "You are a technical documentation assistant. You have access to tools "
    "that let you search a documentation corpus and perform calculations.\n\n"
    "Rules:\n"
    "- Use search_documents to find relevant information before answering.\n"
    "- Base your answer ONLY on the retrieved documents.\n"
    "- Cite sources inline as [source: filename.md] for each claim.\n"
    "- If the documents don't contain the answer, respond with: "
    '"The documentation does not contain information about this topic."\n'
    "- Use calculator for any numerical computations.\n"
    "- Be concise and precise."
)


def create_langchain_agent(
    tools: list[BaseTool],
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    max_iterations: int = 5,
) -> AgentExecutor:
    """Create a LangChain tool-calling agent.

    Args:
        tools: LangChain tools for the agent.
        provider: "openai" or "anthropic".
        model: Model name override. Defaults to gpt-4o-mini / claude-haiku-4-5-20251001.
        temperature: LLM temperature (0.0 for reproducibility).
        system_prompt: System prompt. Defaults to the tech_docs task prompt.
        max_iterations: Max tool-use iterations before forcing a final answer.
    """
    if provider == "openai":
        llm = ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)
    elif provider == "anthropic":
        llm = ChatAnthropic(
            model=model or "claude-haiku-4-5-20251001", temperature=temperature
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt or _DEFAULT_SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
