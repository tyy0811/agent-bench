"""Tests for extended golden dataset schema."""

import json
from pathlib import Path

from agent_bench.evaluation.harness import (
    load_golden_dataset,
)


def test_legacy_flat_list_still_loads(tmp_path):
    """Existing flat-list format continues to work."""
    data = [
        {
            "id": "q001",
            "question": "Test?",
            "expected_answer_keywords": ["test"],
            "expected_sources": ["doc.md"],
            "category": "retrieval",
            "difficulty": "easy",
            "requires_calculator": False,
        }
    ]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    qs = load_golden_dataset(path)
    assert len(qs) == 1
    assert qs[0].id == "q001"
    assert qs[0].source_chunk_ids == []  # default empty list


def test_nested_header_format_loads(tmp_path):
    """New format with corpus/version/snapshot_date header."""
    data = {
        "corpus": "k8s",
        "version": "v1.31",
        "snapshot_date": "2026-04-15",
        "chunker": {
            "strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 64,
        },
        "questions": [
            {
                "id": "k8s_001",
                "question": "Diff between Deployment and StatefulSet?",
                "expected_answer_keywords": ["deployment", "statefulset"],
                "expected_sources": ["k8s_deployment.md", "k8s_statefulset.md"],
                "category": "retrieval",
                "difficulty": "hard",
                "requires_calculator": False,
                "source_chunk_ids": ["abc123", "def456"],
                "source_snippets": ["A Deployment ...", "StatefulSet ..."],
                "question_type": "comparison",
                "is_multi_hop": True,
            }
        ],
    }
    path = tmp_path / "k8s_golden.json"
    path.write_text(json.dumps(data))
    qs = load_golden_dataset(path)
    assert len(qs) == 1
    assert qs[0].source_chunk_ids == ["abc123", "def456"]
    assert qs[0].is_multi_hop is True
    assert qs[0].question_type == "comparison"


def test_existing_fastapi_dataset_still_loads():
    """The real FastAPI dataset loads without error."""
    path = Path("agent_bench/evaluation/datasets/tech_docs_golden.json")
    qs = load_golden_dataset(path)
    assert len(qs) >= 20
    # All questions get default empty lists for new fields
    for q in qs:
        assert q.source_chunk_ids == []
        assert q.source_snippets == []


def test_unknown_format_raises(tmp_path):
    """Dict without 'questions' key raises ValueError."""
    import pytest

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"foo": "bar"}))
    with pytest.raises(ValueError, match="Unrecognized golden dataset format"):
        load_golden_dataset(path)


def test_k8s_pilot_dataset_loads():
    """The 6-question K8s pilot dataset parses and holds schema invariants.

    Pre-ingestion: source_chunk_ids is empty on every pilot. source_pages and
    source_sections are populated and index-aligned with source_snippets. This
    is the authoring shape the backfill script consumes in Commit 7.
    """
    path = Path("agent_bench/evaluation/datasets/k8s_golden_pilot.json")
    qs = load_golden_dataset(path)

    # (a) all 6 pilots parse
    assert len(qs) == 6
    ids = [q.id for q in qs]
    assert ids == [f"k8s_pilot_{i:03d}" for i in range(1, 7)]

    # (b) source_pages and source_sections are populated on every pilot
    # (Variant B false-premise means no pilot has empty sources)
    for q in qs:
        assert len(q.source_pages) > 0, f"{q.id}: source_pages is empty"
        assert len(q.source_snippets) > 0, f"{q.id}: source_snippets is empty"

    # (c) at least one pilot has source_sections[i] == "" (intro edge case)
    intro_edge_pilots = [q.id for q in qs if "" in q.source_sections]
    assert len(intro_edge_pilots) >= 1, (
        "expected at least one pilot testing the intro-content edge case "
        "(snippet above first H2/H3, source_sections[i] == '')"
    )

    # (d) source_chunk_ids is empty pre-ingestion on every pilot
    for q in qs:
        assert q.source_chunk_ids == [], (
            f"{q.id}: source_chunk_ids should be empty pre-backfill"
        )

    # (e) index-alignment invariant across all four parallel lists
    # source_chunk_ids is allowed to be empty OR length-matched (pre/post backfill)
    for q in qs:
        n = len(q.source_snippets)
        assert len(q.source_pages) == n, (
            f"{q.id}: source_pages length {len(q.source_pages)} != "
            f"source_snippets length {n}"
        )
        assert len(q.source_sections) == n, (
            f"{q.id}: source_sections length {len(q.source_sections)} != "
            f"source_snippets length {n}"
        )
        assert len(q.source_chunk_ids) in (0, n), (
            f"{q.id}: source_chunk_ids length {len(q.source_chunk_ids)} "
            f"must be 0 (pre-backfill) or {n} (post-backfill)"
        )


def test_nested_format_ignores_unknown_header_fields(tmp_path):
    """Extra header keys alongside 'questions' don't break the loader."""
    data = {
        "corpus": "k8s",
        "version": "v1.31",
        "snapshot_date": "2026-04-15",
        "custom_future_field": {"anything": [1, 2, 3]},
        "author": "someone",
        "questions": [
            {
                "id": "q1",
                "question": "Test?",
                "expected_answer_keywords": ["test"],
                "expected_sources": ["doc.md"],
                "category": "retrieval",
                "difficulty": "easy",
                "requires_calculator": False,
            }
        ],
    }
    path = tmp_path / "headerish.json"
    path.write_text(json.dumps(data))
    qs = load_golden_dataset(path)
    assert len(qs) == 1
    assert qs[0].id == "q1"
