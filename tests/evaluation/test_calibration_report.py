"""Tests for generate_kappa_table — joins, hash-mismatch raise, strict, abstain flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog

from agent_bench.evaluation.calibration.report import generate_kappa_table


def _write_predictions(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def _write_labels(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records))


def _pred(
    item_id: str, dim: str, score, sys_hash: str = "h1", reasoning: str = ""
) -> dict:
    return {
        "item_id": item_id,
        "dimension": dim,
        "score": score,
        "judge_id": "claude-haiku-4-5_" + dim,
        "rubric_version": "abc",
        "system_output_hash": sys_hash,
        "prompt_seed": 0,
        "cost_usd": 0.001,
        "latency_ms": 100.0,
        "reasoning": reasoning,
        "evidence_quotes": [],
    }


def _lbl(item_id: str, dim: str, score, sys_hash: str = "h1") -> dict:
    return {
        "item_id": item_id,
        "dimension": dim,
        "score": score,
        "abstained": score == "Unknown",
        "notes": "",
        "label_timestamp": "2026-05-04T00:00:00Z",
        "system_output_hash": sys_hash,
    }


class TestHashMismatch:
    def test_raises_with_first_item_detail_and_full_list(self, tmp_path):
        preds = [_pred("i1", "groundedness", 1, sys_hash="A")]
        labels = [_lbl("i1", "groundedness", 1, sys_hash="B")]
        _write_predictions(
            tmp_path / "results" / "calibration_v1_judge_baseline.json", preds
        )
        _write_labels(tmp_path / "labels.jsonl", labels)
        with pytest.raises(ValueError) as exc_info:
            generate_kappa_table(
                predictions_glob=str(
                    tmp_path / "results" / "calibration_v1_judge_*.json"
                ),
                labels_path=str(tmp_path / "labels.jsonl"),
                output_path=str(tmp_path / "kappa.md"),
            )
        msg = str(exc_info.value)
        assert "i1" in msg
        assert "A" in msg and "B" in msg

    def test_hash_mismatch_raises_in_strict_mode_too(self, tmp_path):
        preds = [_pred("i1", "groundedness", 1, sys_hash="A")]
        labels = [_lbl("i1", "groundedness", 1, sys_hash="B")]
        _write_predictions(
            tmp_path / "results" / "calibration_v1_judge_baseline.json", preds
        )
        _write_labels(tmp_path / "labels.jsonl", labels)
        with pytest.raises(ValueError):
            generate_kappa_table(
                predictions_glob=str(
                    tmp_path / "results" / "calibration_v1_judge_*.json"
                ),
                labels_path=str(tmp_path / "labels.jsonl"),
                output_path=str(tmp_path / "kappa.md"),
                strict=True,
            )


class TestMissingPredictionLabel:
    def test_default_warns_and_excludes(self, tmp_path):
        preds = [
            _pred("i1", "groundedness", 1),
            _pred("i3", "groundedness", 0),
            _pred("i4", "groundedness", 1),
        ]
        labels = [
            _lbl("i1", "groundedness", 1),
            _lbl("i2", "groundedness", 0),  # label without prediction
            _lbl("i3", "groundedness", 0),
            _lbl("i4", "groundedness", 1),
        ]
        _write_predictions(
            tmp_path / "results" / "calibration_v1_judge_baseline.json", preds
        )
        _write_labels(tmp_path / "labels.jsonl", labels)
        with structlog.testing.capture_logs() as logs:
            generate_kappa_table(
                predictions_glob=str(
                    tmp_path / "results" / "calibration_v1_judge_*.json"
                ),
                labels_path=str(tmp_path / "labels.jsonl"),
                output_path=str(tmp_path / "kappa.md"),
            )
        assert (tmp_path / "kappa.md").exists()
        assert any(
            entry.get("event") == "calibration_report_missing" for entry in logs
        ), f"no missing-warning log in {logs!r}"

    def test_strict_raises_on_missing_prediction(self, tmp_path):
        preds = [_pred("i1", "groundedness", 1)]
        labels = [
            _lbl("i1", "groundedness", 1),
            _lbl("i2", "groundedness", 0),
        ]
        _write_predictions(
            tmp_path / "results" / "calibration_v1_judge_baseline.json", preds
        )
        _write_labels(tmp_path / "labels.jsonl", labels)
        with pytest.raises(ValueError, match="missing"):
            generate_kappa_table(
                predictions_glob=str(
                    tmp_path / "results" / "calibration_v1_judge_*.json"
                ),
                labels_path=str(tmp_path / "labels.jsonl"),
                output_path=str(tmp_path / "kappa.md"),
                strict=True,
            )


class TestAbstainRateFlag:
    def _setup(self, tmp_path: Path, abstain_count: int) -> Path:
        preds = []
        labels = []
        for i in range(30):
            score: int | str = "Unknown" if i < abstain_count else 1
            reasoning = (
                "schema_parse_failed_after_retry: x" if score == "Unknown" else ""
            )
            preds.append(
                _pred(f"i{i}", "groundedness", score, reasoning=reasoning)
            )
            # Half of non-abstain labels score 0 to ensure variance
            label_score = 0 if (score == 1 and i % 2 == 0) else 1
            labels.append(_lbl(f"i{i}", "groundedness", label_score))
        _write_predictions(
            tmp_path / "results" / "calibration_v1_judge_baseline.json", preds
        )
        _write_labels(tmp_path / "labels.jsonl", labels)
        out = tmp_path / "kappa.md"
        generate_kappa_table(
            predictions_glob=str(
                tmp_path / "results" / "calibration_v1_judge_*.json"
            ),
            labels_path=str(tmp_path / "labels.jsonl"),
            output_path=str(out),
        )
        return out

    def test_at_20_percent_boundary_does_not_fire(self, tmp_path):
        # 6/30 = exactly 20% — flag is ">"  (strictly greater), so not fired.
        out = self._setup(tmp_path, abstain_count=6)
        assert "high abstain rate" not in out.read_text().lower()

    def test_above_20_percent_fires(self, tmp_path):
        # 7/30 = 23.3% — flag fires
        out = self._setup(tmp_path, abstain_count=7)
        text = out.read_text().lower()
        assert "high abstain rate" in text
        assert "schema parse" in text


class TestSidecarSkipped:
    def test_members_json_sidecar_excluded_from_table(self, tmp_path):
        """Regression: per-member sidecar files (matching '_members.*' in
        basename) must not contaminate the κ table even when their extension
        matches the predictions glob. The contract is keyed off the basename
        marker, not the extension.
        """
        # Real prediction file
        preds = [_pred("i1", "groundedness", 1)]
        labels = [_lbl("i1", "groundedness", 1)]
        _write_predictions(
            tmp_path / "results" / "calibration_v1_judge_baseline.json", preds
        )

        # Hypothetical sidecar file that happens to end in .json (would
        # normally be .jsonl but the contract should not depend on that).
        # If the report didn't skip this file, the per-member records inside
        # would be parsed as aggregate predictions and skew the κ stats.
        sidecar_pred_shape = [_pred("i1", "groundedness", 0)]  # opposite score
        _write_predictions(
            tmp_path / "results" / "calibration_v1_judge_jury_members.json",
            sidecar_pred_shape,
        )

        _write_labels(tmp_path / "labels.jsonl", labels)
        out = tmp_path / "kappa.md"
        generate_kappa_table(
            predictions_glob=str(
                tmp_path / "results" / "calibration_v1_judge_*.json"
            ),
            labels_path=str(tmp_path / "labels.jsonl"),
            output_path=str(out),
        )
        text = out.read_text()
        # Aggregate row from baseline.json should appear; sidecar's "jury_members"
        # label should NOT appear as a row in the table.
        assert "baseline" in text
        assert "jury_members" not in text


class TestKappaUndefined:
    def test_renders_dash_with_footnote(self, tmp_path):
        # All same label → degenerate; report renders ' — '
        preds = [_pred(f"i{i}", "groundedness", 1) for i in range(5)]
        labels = [_lbl(f"i{i}", "groundedness", 1) for i in range(5)]
        _write_predictions(
            tmp_path / "results" / "calibration_v1_judge_baseline.json", preds
        )
        _write_labels(tmp_path / "labels.jsonl", labels)
        out = tmp_path / "kappa.md"
        generate_kappa_table(
            predictions_glob=str(
                tmp_path / "results" / "calibration_v1_judge_*.json"
            ),
            labels_path=str(tmp_path / "labels.jsonl"),
            output_path=str(out),
        )
        text = out.read_text()
        assert " — " in text
