"""Pure statistics package for the v3.1 layer.

Depends only on the standard library, numpy, scipy, and pandas.
Never imports agent-bench modules (guardrail 1); tests/stats/test_isolation.py
enforces this with a meta-path blocker and a source scan.
"""

from stats.schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
