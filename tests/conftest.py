"""Shared test fixtures."""

import pytest

from agent_bench.core.provider import MockProvider


@pytest.fixture
def mock_provider() -> MockProvider:
    """MockProvider instance for deterministic testing."""
    return MockProvider()
