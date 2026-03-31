"""Append-only structured audit logging.

Writes one JSON record per line to a JSONL file. Supports log rotation
and IP hashing (SHA-256) for GDPR compliance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path


class AuditLogger:
    """Append-only JSONL audit logger with optional rotation."""

    def __init__(
        self,
        path: str = "logs/audit.jsonl",
        max_size_bytes: int = 100 * 1024 * 1024,  # 100 MB
        rotate: bool = True,
    ) -> None:
        self.path = Path(path)
        self.max_size_bytes = max_size_bytes
        self.rotate = rotate
        self._lock = threading.Lock()

    def log(self, record: dict) -> None:
        """Append a record to the audit log.

        Adds a timestamp if not present. Thread-safe.
        """
        if "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            if self.rotate and self.path.exists():
                if self.path.stat().st_size >= self.max_size_bytes:
                    self._rotate()

            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def hash_ip(self, ip: str) -> str:
        """Hash an IP address with SHA-256. Irreversible."""
        return hashlib.sha256(ip.encode()).hexdigest()

    def _rotate(self) -> None:
        """Rotate the current log file by appending a timestamp suffix."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        rotated = self.path.with_name(f"{self.path.name}.{ts}")
        shutil.move(str(self.path), str(rotated))
