"""Tests for LangChain agent factory."""

from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import AgentExecutor
from langchain_core.tools import StructuredTool

from agent_bench.langchain_baseline.agent import create_langchain_agent


def _make_dummy_tool():
    return StructuredTool.from_function(
        func=lambda query: "result",
        name="test_tool",
        description="A test tool",
    )


@patch("agent_bench.langchain_baseline.agent.ChatOpenAI")
def test_creates_agent_executor_openai(mock_chat):
    mock_chat.return_value = MagicMock()
    tool = _make_dummy_tool()

    executor = create_langchain_agent(
        tools=[tool],
        provider="openai",
    )

    assert isinstance(executor, AgentExecutor)
    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs.kwargs["temperature"] == 0.0


@patch("agent_bench.langchain_baseline.agent.ChatAnthropic")
def test_creates_agent_executor_anthropic(mock_chat):
    mock_chat.return_value = MagicMock()
    tool = _make_dummy_tool()

    executor = create_langchain_agent(
        tools=[tool],
        provider="anthropic",
    )

    assert isinstance(executor, AgentExecutor)
    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args
    assert call_kwargs.kwargs["model_name"] == "claude-haiku-4-5-20251001"


@patch("agent_bench.langchain_baseline.agent.ChatOpenAI")
def test_custom_model_override(mock_chat):
    mock_chat.return_value = MagicMock()
    tool = _make_dummy_tool()

    create_langchain_agent(
        tools=[tool],
        provider="openai",
        model="gpt-4o",
    )

    call_kwargs = mock_chat.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"


def test_unknown_provider_raises():
    tool = _make_dummy_tool()
    with pytest.raises(ValueError, match="Unknown provider"):
        create_langchain_agent(tools=[tool], provider="unknown")


@patch("agent_bench.langchain_baseline.agent.ChatOpenAI")
def test_uses_custom_system_prompt(mock_chat):
    mock_chat.return_value = MagicMock()
    tool = _make_dummy_tool()

    executor = create_langchain_agent(
        tools=[tool],
        provider="openai",
        system_prompt="Custom prompt here",
    )

    assert isinstance(executor, AgentExecutor)
