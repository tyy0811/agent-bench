"""Tests for structured audit logging."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_bench.security.audit_logger import AuditLogger


class TestAuditLogger:
    def test_log_creates_file(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=str(log_path))
        logger.log({"request_id": "test-1", "endpoint": "/ask"})
        assert log_path.exists()

    def test_log_appends_jsonl(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=str(log_path))
        logger.log({"request_id": "r1"})
        logger.log({"request_id": "r2"})
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["request_id"] == "r1"
        assert json.loads(lines[1])["request_id"] == "r2"

    def test_log_adds_timestamp(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=str(log_path))
        logger.log({"request_id": "r1"})
        record = json.loads(log_path.read_text().strip())
        assert "timestamp" in record

    def test_hash_ip(self):
        logger = AuditLogger(path="/dev/null")
        hashed = logger.hash_ip("192.168.1.1")
        # Deterministic
        assert hashed == logger.hash_ip("192.168.1.1")
        # Not the raw IP
        assert "192.168.1.1" not in hashed
        # SHA-256 hex = 64 chars
        assert len(hashed) == 64

    def test_hash_ip_different_inputs(self):
        logger = AuditLogger(path="/dev/null")
        assert logger.hash_ip("10.0.0.1") != logger.hash_ip("10.0.0.2")

    def test_log_rotation(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        # 1 byte max size to force rotation on second write
        logger = AuditLogger(path=str(log_path), max_size_bytes=1, rotate=True)
        logger.log({"request_id": "r1"})
        logger.log({"request_id": "r2"})
        # Original file should still exist with latest record
        assert log_path.exists()
        # Rotated file should exist
        rotated = list(tmp_path.glob("audit.jsonl.*"))
        assert len(rotated) >= 1

    def test_no_rotation_when_disabled(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=str(log_path), max_size_bytes=1, rotate=False)
        logger.log({"request_id": "r1"})
        logger.log({"request_id": "r2"})
        rotated = list(tmp_path.glob("audit.jsonl.*"))
        assert len(rotated) == 0

    def test_creates_parent_directories(self, tmp_path):
        log_path = tmp_path / "nested" / "dir" / "audit.jsonl"
        logger = AuditLogger(path=str(log_path))
        logger.log({"request_id": "r1"})
        assert log_path.exists()

    def test_multiple_rotations_no_data_loss(self, tmp_path):
        """Multiple rotations in the same second must not overwrite each other."""
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=str(log_path), max_size_bytes=1, rotate=True)
        logger.log({"request_id": "r1"})
        logger.log({"request_id": "r2"})
        logger.log({"request_id": "r3"})
        # All 3 records must survive: 2 in rotated files, 1 in active log
        rotated = list(tmp_path.glob("audit.jsonl.*"))
        assert len(rotated) == 2
        all_records = []
        for f in [log_path, *rotated]:
            for line in f.read_text().strip().split("\n"):
                all_records.append(json.loads(line)["request_id"])
        assert sorted(all_records) == ["r1", "r2", "r3"]

    def test_hash_ip_different_keys_produce_different_hashes(self):
        """Different HMAC keys produce different hashes for the same IP."""
        logger_a = AuditLogger(path="/dev/null", hmac_key="key-a")
        logger_b = AuditLogger(path="/dev/null", hmac_key="key-b")
        assert logger_a.hash_ip("192.168.1.1") != logger_b.hash_ip("192.168.1.1")

    def test_hash_ip_stable_with_same_key(self):
        """Same HMAC key produces consistent hashes across instances."""
        logger_a = AuditLogger(path="/dev/null", hmac_key="stable-key")
        logger_b = AuditLogger(path="/dev/null", hmac_key="stable-key")
        assert logger_a.hash_ip("10.0.0.1") == logger_b.hash_ip("10.0.0.1")

    def test_multi_instance_rotation_no_data_loss(self, tmp_path):
        """Two logger instances rotating the same file must not overwrite each other."""
        log_path = tmp_path / "audit.jsonl"
        logger_a = AuditLogger(path=str(log_path), max_size_bytes=1, rotate=True)
        logger_b = AuditLogger(path=str(log_path), max_size_bytes=1, rotate=True)
        logger_a.log({"request_id": "r1"})
        logger_b.log({"request_id": "r2"})
        logger_a.log({"request_id": "r3"})
        # All 3 records must survive across rotated files + active log
        all_records = []
        for f in tmp_path.glob("audit.jsonl*"):
            for line in f.read_text().strip().split("\n"):
                if line:
                    all_records.append(json.loads(line)["request_id"])
        assert sorted(all_records) == ["r1", "r2", "r3"]

    def test_no_hmac_key_logs_warning(self, tmp_path, capsys):
        """Default-constructed logger warns about non-stable IP hashing."""
        os.environ.pop("AUDIT_HMAC_KEY", None)
        AuditLogger(path=str(tmp_path / "audit.jsonl"))
        captured = capsys.readouterr()
        assert "audit_hmac_key_missing" in captured.out
