"""Tests for structured audit logging."""

from __future__ import annotations

import json
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
