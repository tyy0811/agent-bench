"""Append-only structured audit logging.

Writes one JSON record per line to a JSONL file. Supports log rotation
and HMAC-SHA256 IP hashing for GDPR compliance.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
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
        hmac_key: str = "",
    ) -> None:
        self.path = Path(path)
        self.max_size_bytes = max_size_bytes
        self.rotate = rotate
        self._lock = threading.Lock()
        self._rotate_seq = 0
        # HMAC key: explicit arg > env var > random per-process key
        key_str = hmac_key or os.environ.get("AUDIT_HMAC_KEY", "")
        if key_str:
            self._hmac_key = key_str.encode()
        else:
            self._hmac_key = os.urandom(32)

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
                    self._do_rotate()

            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def hash_ip(self, ip: str) -> str:
        """HMAC-SHA256 hash an IP address. Keyed and irreversible."""
        return hmac.new(self._hmac_key, ip.encode(), hashlib.sha256).hexdigest()

    def _do_rotate(self) -> None:
        """Rotate the current log file with a unique suffix."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._rotate_seq += 1
        rotated = self.path.with_name(f"{self.path.name}.{ts}.{self._rotate_seq}")
        shutil.move(str(self.path), str(rotated))
