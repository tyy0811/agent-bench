"""SQLite-backed conversation persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ConversationStore:
    """Store and retrieve conversation history keyed by session_id."""

    def __init__(self, db_path: str = "data/conversations.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON conversations(session_id)
            """)

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a message to a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    content,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(metadata or {}),
                ),
            )

    def get_history(
        self, session_id: str, max_turns: int = 10
    ) -> list[dict]:
        """Get recent conversation history for a session.

        Returns up to max_turns * 2 messages (user + assistant pairs),
        ordered chronologically.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT role, content FROM conversations
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (session_id, max_turns * 2),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM conversations"
            ).fetchall()
        return [r[0] for r in rows]

    def delete_session(self, session_id: str) -> None:
        """Delete all messages for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (session_id,),
            )
