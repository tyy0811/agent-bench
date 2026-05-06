"""Smoke + dispatch tests for scripts/run_calibration.py.

Two failure modes this guards against:

1. Silent broken imports inside cmd_generate_outputs. The runner has no
   module-level test coverage; a missing symbol like build_default_registry
   will pass CI and fail at first invocation. test_module_imports asserts
   the runner is importable.

2. Mixed-corpus calibration items routed to the wrong store. The spec
   includes both k8s and fastapi questions. test_dispatch_routes_per_corpus
   verifies each item goes to the orchestrator built for its corpus, and
   test_unknown_corpus_raises verifies a clear error if the spec drifts
   from the corpora the runner builds.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _StubProvider:
    def __init__(self, *args, **kwargs):
        pass


class _StubEmbedder:
    def __init__(self, *args, **kwargs):
        pass


class _StubOrchestrator:
    """Records each run() call and returns a synthetic AgentResponse-shaped
    object so cmd_generate_outputs can compute its hash + write its record.
    """

    def __init__(self, corpus_name: str, calls: list) -> None:
        self.corpus_name = corpus_name
        self.calls = calls

    async def run(self, *, question: str, system_prompt: str):
        self.calls.append({"corpus": self.corpus_name, "question": question})

        class _Source:
            def __init__(self, s: str) -> None:
                self.source = s

        class _Resp:
            answer = f"[{self.corpus_name}] answer to: {question}"
            sources = [_Source(f"{self.corpus_name}/doc.md")]
            ranked_sources = [f"{self.corpus_name}/doc.md"]
            source_chunks = ["chunk text"]

        return _Resp()


def test_module_imports():
    """Importing the runner must not raise. Catches broken symbol references
    inside the module before they cost a calibration run."""
    mod = importlib.import_module("scripts.run_calibration")
    assert hasattr(mod, "cmd_generate_outputs")
    assert hasattr(mod, "_build_corpus_orchestrator")


async def test_dispatch_routes_per_corpus(monkeypatch, tmp_path):
    runner = importlib.import_module("scripts.run_calibration")

    monkeypatch.setattr(
        "agent_bench.core.provider.AnthropicProvider", _StubProvider
    )
    monkeypatch.setattr("agent_bench.rag.embedder.Embedder", _StubEmbedder)

    calls: list = []
    built_corpora: list = []

    def fake_builder(cfg, corpus_name, embedder, provider):
        built_corpora.append(corpus_name)
        return _StubOrchestrator(corpus_name, calls)

    monkeypatch.setattr(runner, "_build_corpus_orchestrator", fake_builder)

    out_path = tmp_path / "system_outputs.json"
    monkeypatch.setattr(runner, "SYSTEM_OUTPUTS", out_path)

    await runner.cmd_generate_outputs(concurrency=2)

    assert sorted(built_corpora) == ["fastapi", "k8s"]

    spec = json.loads(runner.CALIBRATION_SPEC.read_text())
    expected_corpus_by_id = {i["id"]: i["corpus"] for i in spec["items"]}

    records = json.loads(out_path.read_text())
    assert len(records) == len(expected_corpus_by_id)

    seen_ids = set()
    for rec in records:
        item_id = rec["item_id"]
        seen_ids.add(item_id)
        expected = expected_corpus_by_id[item_id]
        assert rec["corpus"] == expected
        assert rec["answer"].startswith(f"[{expected}]")
        assert rec["sources"] == [f"{expected}/doc.md"]
        assert isinstance(rec["system_output_hash"], str)
        assert len(rec["system_output_hash"]) == 64

    assert seen_ids == set(expected_corpus_by_id.keys())

    by_corpus: dict[str, int] = {}
    for c in calls:
        by_corpus[c["corpus"]] = by_corpus.get(c["corpus"], 0) + 1
    expected_counts: dict[str, int] = {}
    for cor in expected_corpus_by_id.values():
        expected_counts[cor] = expected_counts.get(cor, 0) + 1
    assert by_corpus == expected_counts


async def test_unknown_corpus_raises(monkeypatch, tmp_path):
    runner = importlib.import_module("scripts.run_calibration")

    monkeypatch.setattr(
        "agent_bench.core.provider.AnthropicProvider", _StubProvider
    )
    monkeypatch.setattr("agent_bench.rag.embedder.Embedder", _StubEmbedder)

    calls: list = []

    def fake_builder(cfg, corpus_name, embedder, provider):
        return _StubOrchestrator(corpus_name, calls)

    monkeypatch.setattr(runner, "_build_corpus_orchestrator", fake_builder)
    monkeypatch.setattr(
        runner, "SYSTEM_OUTPUTS", tmp_path / "system_outputs.json"
    )

    spec = json.loads(runner.CALIBRATION_SPEC.read_text())
    bogus_spec = {
        "items": [
            {**spec["items"][0], "corpus": "phantom_corpus"},
        ]
    }
    bogus_spec_path = tmp_path / "calibration_v1.json"
    bogus_spec_path.write_text(json.dumps(bogus_spec))
    monkeypatch.setattr(runner, "CALIBRATION_SPEC", bogus_spec_path)

    with pytest.raises(KeyError) as excinfo:
        await runner.cmd_generate_outputs(concurrency=1)

    msg = str(excinfo.value)
    assert "phantom_corpus" in msg
    assert "not in cfg.corpora" in msg
    assert spec["items"][0]["id"] in msg


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_compute_kappa_weights_returns_real_kappa(monkeypatch, tmp_path):
    """Two judges over 4 items with known agreement patterns. Cohen's κ
    must come out positive for the high-agreement judge and zero for the
    chance-agreement judge.
    """
    runner = importlib.import_module("scripts.run_calibration")

    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "item_id": f"i{n}",
                "dimension": "completeness",
                "score": gold,
                "abstained": False,
                "system_output_hash": f"h{n}",
            }
            for n, gold in enumerate([2, 2, 1, 1])
        ],
    )
    monkeypatch.setattr(runner, "LABELS_PATH", labels_path)

    sidecar = tmp_path / "predictions.jsonl"
    rows = []
    for n, gold in enumerate([2, 2, 1, 1]):
        rows.append(
            {
                "judge_id": "good_completeness",
                "system_output_hash": f"h{n}",
                "score": gold,
            }
        )
        rows.append(
            {
                "judge_id": "bad_completeness",
                "system_output_hash": f"h{n}",
                "score": 1,
            }
        )
    _write_jsonl(sidecar, rows)

    weights = runner._compute_kappa_weights(
        sidecar,
        "completeness",
        expected_judge_ids={"good_completeness", "bad_completeness"},
    )
    assert weights["good_completeness"] == pytest.approx(1.0)
    assert weights["bad_completeness"] == 0.0  # negative κ clipped to 0


def test_compute_kappa_weights_hard_errors_on_missing_member(monkeypatch, tmp_path):
    """Asymmetric coverage in the weights source must hard-error, not
    silently return partial weights — that was the v1 bug that masked
    gpt-4o-mini's exclusion."""
    runner = importlib.import_module("scripts.run_calibration")

    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "item_id": "i0",
                "dimension": "completeness",
                "score": 2,
                "abstained": False,
                "system_output_hash": "h0",
            }
        ],
    )
    monkeypatch.setattr(runner, "LABELS_PATH", labels_path)

    sidecar = tmp_path / "predictions.jsonl"
    _write_jsonl(
        sidecar,
        [
            {
                "judge_id": "haiku_completeness",
                "system_output_hash": "h0",
                "score": 2,
            }
        ],
    )

    with pytest.raises(ValueError, match="symmetric coverage"):
        runner._compute_kappa_weights(
            sidecar,
            "completeness",
            expected_judge_ids={"haiku_completeness", "gpt4o_completeness"},
        )
