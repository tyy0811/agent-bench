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
