"""Canary runner tests. Everything here is keyless: the judge layer is injected
as MockJudge, so no API calls happen. The committed-report test pins the
offline regeneration to the artifact under docs/_generated/.
"""

import json
import sys
from pathlib import Path

from agent_bench.evaluation.judges.base import MockJudge, ScoreResult

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

import run_canary_eval  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "canary"


def _score(value: object) -> ScoreResult:
    return ScoreResult(
        reasoning="prebaked",
        evidence_quotes=[],
        score=value,  # type: ignore[arg-type]
        judge_id="mock_judge",
        rubric_version="rubric-abc",
        system_output_hash="hash-def",
        cost_usd=0.0,
        latency_ms=0.0,
    )


_CANARIES = [
    {
        "id": "c1",
        "injection_type": "ungrounded",
        "category": "retrieval",
        "question": "q1?",
        "answer": "a1",
        "sources": ["a.md"],
        "source_snippets": ["snippet"],
        "reference_answer": "ref",
        "expected_failing": {
            "groundedness": True,
            "completeness": False,
            "relevance": False,
            "citation_faithfulness": False,
        },
    },
    {
        "id": "c2",
        "injection_type": "incomplete",
        "category": "retrieval",
        "question": "q2?",
        "answer": "a2",
        "sources": ["b.md"],
        "source_snippets": ["snippet2"],
        "reference_answer": "ref2",
        "expected_failing": {
            "groundedness": False,
            "completeness": True,
            "relevance": False,
            "citation_faithfulness": False,
        },
    },
]

# Per-dimension, per-canary simulated scores. Distinct per dimension so the test
# proves score_canaries routes the right judge to the right dimension.
_VERDICTS = {
    "groundedness": {"c1": 0, "c2": 1},
    "completeness": {"c1": 2, "c2": 1},
    "relevance": {"c1": 2, "c2": 2},
    "citation_faithfulness": {"c1": 1, "c2": "Unknown"},
}


def _mock_judges() -> dict:
    return {
        dim: MockJudge({cid: _score(val) for cid, val in per_canary.items()})
        for dim, per_canary in _VERDICTS.items()
    }


async def test_score_canaries_runs_each_judge_on_each_dimension():
    preds = await run_canary_eval.score_canaries(_CANARIES, _mock_judges())
    assert len(preds) == len(_CANARIES) * len(run_canary_eval.DIMENSIONS)
    keyed = {(p["item_id"], p["dimension"]): p["score"] for p in preds}
    # Routing: each dimension's judge verdict lands under that dimension.
    assert keyed[("c1", "groundedness")] == 0
    assert keyed[("c1", "completeness")] == 2
    assert keyed[("c2", "citation_faithfulness")] == "Unknown"
    # Predictions carry judge provenance from the ScoreResult model dump.
    assert all("rubric_version" in p for p in preds)


async def test_cmd_run_judges_writes_predictions_file(tmp_path, monkeypatch):
    canaries_path = tmp_path / "canaries.json"
    canaries_path.write_text(json.dumps(_CANARIES))
    out_path = tmp_path / "preds.json"
    # Inject MockJudges in place of the paid real-judge construction.
    monkeypatch.setattr(run_canary_eval, "build_judges", lambda provider, model_id: _mock_judges())

    await run_canary_eval.cmd_run_judges(canaries_path, out_path, provider="mock", model_id="mock")

    written = json.loads(out_path.read_text())
    assert len(written) == len(_CANARIES) * len(run_canary_eval.DIMENSIONS)
    assert {p["dimension"] for p in written} == set(run_canary_eval.DIMENSIONS)


def test_cmd_build_report_writes_markdown(tmp_path):
    out_path = tmp_path / "canary_detection.md"
    run_canary_eval.cmd_build_report(
        FIXTURES / "canaries.json",
        FIXTURES / "predictions.json",
        out_path,
        provenance="test provenance",
    )
    md = out_path.read_text()
    assert "# Canary detection report" in md
    assert "test provenance" in md
    for dim in run_canary_eval.DIMENSIONS:
        assert dim in md


def test_committed_report_matches_regeneration():
    # Guards docs/_generated/canary_detection.md against drift: regenerating from
    # the committed fixtures with the committed provenance must reproduce it
    # byte for byte (render is a pure function of its inputs).
    from stats_adapters import canary

    canaries = json.loads((FIXTURES / "canaries.json").read_text())
    predictions = json.loads((FIXTURES / "predictions.json").read_text())
    expected = canary.build_report(
        canaries, predictions, provenance=run_canary_eval.DEFAULT_PROVENANCE
    )
    committed = (REPO / "docs/_generated/canary_detection.md").read_text()
    assert committed == expected, "stale canary_detection.md; run `make canary-report`"
