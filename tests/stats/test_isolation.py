"""Guardrail 1 enforcement: stats/ never imports agent-bench.

Two mechanisms (design spec section 8): a meta-path blocker that fails any
agent_bench import while importing every stats submodule, and a source scan.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

import stats


class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "agent_bench" or fullname.startswith("agent_bench."):
            raise ImportError("stats/ must not import agent_bench (guardrail 1)")
        return None


def test_every_stats_submodule_imports_with_agent_bench_blocked():
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        for mod in pkgutil.walk_packages(stats.__path__, prefix="stats."):
            module = importlib.import_module(mod.name)
            importlib.reload(module)
    finally:
        sys.meta_path.remove(blocker)


def test_source_scan_finds_no_agent_bench_string():
    root = Path(stats.__file__).parent
    offenders = [p for p in root.rglob("*.py") if "agent_bench" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_blocker_actually_rejects_agent_bench():
    # Positive control: prove the blocker mechanism fires. Without this, the
    # import loop above passes even with a no-op blocker (no stats module imports
    # agent_bench today), so a dead blocker would give false assurance.
    blocker = _Blocker()
    with pytest.raises(ImportError):
        blocker.find_spec("agent_bench")
    with pytest.raises(ImportError):
        blocker.find_spec("agent_bench.evaluation.metrics")
    assert blocker.find_spec("numpy") is None  # non-agent_bench passes through
