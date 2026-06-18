"""Canary runner tests. Everything here is keyless: the judge layer is injected
as MockJudge, so no API calls happen. The committed-report test pins the
offline regeneration to the artifact under docs/_generated/.
"""

import json
import sys
from pathlib import Path

import pytest

from agent_bench.core.types import CompletionResponse, TokenUsage
from agent_bench.evaluation.judges.base import MockJudge, Rubric, ScoreResult
from agent_bench.evaluation.judges.citation_faithfulness import CitationFaithfulnessJudge

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

import run_canary_eval  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "canary"
CORPUS_DIRS = (REPO / "data" / "tech_docs", REPO / "data" / "k8s_docs")


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
        "source_chunks": ["chunk a"],
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
        "source_chunks": ["chunk b"],
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


# --- Finding 1: the canary citation path must be real-judge-ready ---
# The harness was only exercised with MockJudge; these drive the REAL
# CitationFaithfulnessJudge through a stub JSON provider (no API) to prove the
# cited chunk reaches the judge, and to pin the no-citation blind spot that
# dictates how absent_citation canaries must be authored.

CITATION_RUBRIC = REPO / "agent_bench/evaluation/rubrics/citation_faithfulness.md"


class _StubProvider:
    """Returns a fixed JSON verdict so a real judge can be driven without API
    calls. Records every prompt it is sent."""

    def __init__(self, score: object) -> None:
        self._score = score
        self.prompts: list[str] = []

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=1024):
        self.prompts.append(messages[0].content)
        return CompletionResponse(
            content=json.dumps({"reasoning": "stub", "evidence_quotes": [], "score": self._score}),
            tool_calls=[],
            usage=TokenUsage(input_tokens=1, output_tokens=1, estimated_cost_usd=0.0),
            provider="stub",
            model="stub",
            latency_ms=0.0,
        )


def _canary(canary_id: str) -> dict:
    canaries = json.loads((FIXTURES / "canaries.json").read_text())
    return next(c for c in canaries if c["id"] == canary_id)


def _citation_judge(provider: _StubProvider) -> CitationFaithfulnessJudge:
    return CitationFaithfulnessJudge(
        judge_provider=provider,  # type: ignore[arg-type]
        rubric=Rubric.from_markdown_file(CITATION_RUBRIC),
        model_id="stub",
    )


def test_shipped_canaries_carry_source_chunks_aligned_to_sources():
    # The citation judge maps citations via zip(sources, source_chunks); empty or
    # misaligned chunks make it score against no evidence. Every shipped canary
    # must carry one non-empty chunk per source.
    for c in json.loads((FIXTURES / "canaries.json").read_text()):
        _item, output = run_canary_eval._build_item_and_output(c)
        assert len(output.source_chunks) == len(output.sources), c["id"]
        assert all(chunk.strip() for chunk in output.source_chunks), c["id"]


def test_shipped_canaries_use_real_corpus_sources_and_passages():
    """Canaries used with real judges must cite source files and evidence text
    that exist in the ingested corpora, not synthetic fixture names.
    """

    def source_text(source: str) -> str:
        matches = [path / source for path in CORPUS_DIRS if (path / source).exists()]
        assert matches, f"canary source {source!r} is not in any corpus directory"
        return matches[0].read_text()

    for c in json.loads((FIXTURES / "canaries.json").read_text()):
        source_docs = {source: source_text(source) for source in c["sources"]}
        for source, chunk in zip(c["sources"], c["source_chunks"], strict=True):
            assert chunk in source_docs[source], f"{c['id']} chunk is not copied from {source}"
        for snippet in c["source_snippets"]:
            assert any(snippet in doc for doc in source_docs.values()), (
                f"{c['id']} snippet is not copied from its cited corpus sources"
            )


def test_build_item_and_output_rejects_misaligned_source_chunks():
    bad = {**_canary("canary_absent_citation_02"), "source_chunks": []}
    with pytest.raises(ValueError, match="source_chunks"):
        run_canary_eval._build_item_and_output(bad)


async def test_real_citation_judge_sees_the_cited_chunk():
    # Drive the REAL judge over an absent_citation canary; the cited chunk text
    # must reach the judge's prompt (proving source_chunks flows through), and the
    # stub's unfaithful verdict must propagate to the aggregate.
    rec = _canary("canary_absent_citation_02")
    item, output = run_canary_eval._build_item_and_output(rec)
    stub = _StubProvider(0)
    result = await _citation_judge(stub).score(item, output)
    assert stub.prompts, "judge never evaluated a citation pair"
    assert any(rec["source_chunks"][0] in p for p in stub.prompts), (
        "cited chunk text never reached the judge prompt"
    )
    assert result.score == 0  # unfaithful verdict propagates (all-or-nothing)


async def test_real_citation_judge_passes_an_uncited_answer():
    # Characterizes the production judge's blind spot: a no-citation answer is
    # scored 1 (vacuously faithful) with no provider call. This is why an
    # absent_citation canary must cite a source that does NOT support the claim,
    # not omit the citation entirely.
    _item, output = run_canary_eval._build_item_and_output(
        {
            "id": "uncited",
            "answer": "A plain answer with no citation marker.",
            "sources": ["a.md"],
            "source_chunks": ["some chunk"],
            "source_snippets": ["snippet"],
            "reference_answer": "ref",
            "category": "retrieval",
        }
    )
    stub = _StubProvider(0)
    result = await _citation_judge(stub).score(_item, output)
    assert result.score == 1 and stub.prompts == []
