"""Local SQLite query cache to avoid redundant search API calls."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from research_agent.models import SearchResult


class ResearchCache:
    """Persistent SQLite cache keyed by (backend, normalized_query, max_results)."""

    def __init__(self, db_path: Path | str = ".cache/research_cache.sqlite3", ttl_seconds: int = 86_400) -> None:
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    cache_key TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    query TEXT NOT NULL,
                    max_results INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def make_key(backend: str, query: str, max_results: int) -> str:
        normalized = " ".join(query.strip().lower().split())
        return f"{backend.lower()}::{max_results}::{normalized}"

    def get(self, backend: str, query: str, max_results: int) -> list[SearchResult] | None:
        key = self.make_key(backend, query, max_results)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, created_at FROM search_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None

        payload_json, created_at = row
        if self.ttl_seconds > 0 and (time.time() - float(created_at)) > self.ttl_seconds:
            self.delete(backend, query, max_results)
            return None

        raw_items = json.loads(payload_json)
        return [SearchResult.model_validate(item) for item in raw_items]

    def set(
        self,
        backend: str,
        query: str,
        max_results: int,
        results: list[SearchResult],
        *,
        created_at: float | None = None,
    ) -> None:
        key = self.make_key(backend, query, max_results)
        payload = json.dumps([r.model_dump() for r in results], ensure_ascii=False)
        ts = time.time() if created_at is None else created_at
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO search_cache
                    (cache_key, backend, query, max_results, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, backend, query.strip(), max_results, payload, ts),
            )
            conn.commit()

    def delete(self, backend: str, query: str, max_results: int) -> None:
        key = self.make_key(backend, query, max_results)
        with self._connect() as conn:
            conn.execute("DELETE FROM search_cache WHERE cache_key = ?", (key,))
            conn.commit()

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()
        return int(row[0]) if row else 0
