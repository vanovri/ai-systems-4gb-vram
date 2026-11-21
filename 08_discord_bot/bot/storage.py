"""Lightweight SQLite persistence store for scheduled user reminders."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ReminderRecord:
    """Persisted reminder row."""

    id: int
    user_id: int
    channel_id: int
    message: str
    created_at: float
    due_at: float
    fired: bool
    fired_at: float | None = None


class ReminderStore:
    """SQLite-backed reminder queue surviving bot restarts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    due_at REAL NOT NULL,
                    fired INTEGER NOT NULL DEFAULT 0,
                    fired_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_pending ON reminders (fired, due_at)"
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ReminderRecord:
        return ReminderRecord(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            channel_id=int(row["channel_id"]),
            message=str(row["message"]),
            created_at=float(row["created_at"]),
            due_at=float(row["due_at"]),
            fired=bool(row["fired"]),
            fired_at=float(row["fired_at"]) if row["fired_at"] is not None else None,
        )

    def add_reminder(
        self,
        user_id: int,
        channel_id: int,
        message: str,
        delay_seconds: float,
        now: float | None = None,
    ) -> ReminderRecord:
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")
        cleaned = message.strip()
        if not cleaned:
            raise ValueError("Reminder message cannot be empty")

        ts = time.time() if now is None else float(now)
        due = ts + float(delay_seconds)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO reminders (user_id, channel_id, message, created_at, due_at, fired)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (int(user_id), int(channel_id), cleaned, ts, due),
            )
            row_id = int(cur.lastrowid)
        return ReminderRecord(
            id=row_id,
            user_id=int(user_id),
            channel_id=int(channel_id),
            message=cleaned,
            created_at=ts,
            due_at=due,
            fired=False,
            fired_at=None,
        )

    def get_by_id(self, reminder_id: int) -> ReminderRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reminders WHERE id = ?", (int(reminder_id),)).fetchone()
        return self._row_to_record(row) if row is not None else None

    def get_pending(self) -> list[ReminderRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE fired = 0 ORDER BY due_at ASC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_due(self, now: float | None = None) -> list[ReminderRecord]:
        ts = time.time() if now is None else float(now)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE fired = 0 AND due_at <= ? ORDER BY due_at ASC",
                (ts,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_user_reminders(
        self, user_id: int, *, include_fired: bool = False
    ) -> list[ReminderRecord]:
        query = "SELECT * FROM reminders WHERE user_id = ?"
        if not include_fired:
            query += " AND fired = 0"
        query += " ORDER BY due_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, (int(user_id),)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def mark_fired(self, reminder_id: int, now: float | None = None) -> bool:
        ts = time.time() if now is None else float(now)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE reminders SET fired = 1, fired_at = ? WHERE id = ? AND fired = 0",
                (ts, int(reminder_id)),
            )
            return cur.rowcount > 0

    def cancel_reminder(self, reminder_id: int, user_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM reminders WHERE id = ? AND user_id = ? AND fired = 0",
                (int(reminder_id), int(user_id)),
            )
            return cur.rowcount > 0

    def count_pending(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM reminders WHERE fired = 0").fetchone()
            return int(row["cnt"]) if row is not None else 0
