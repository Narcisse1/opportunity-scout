"""SQLite state: which URLs were processed, known opportunities, and page snapshots.

State is what makes the agent incremental: re-runs only pay tokens for new pages, and the
competitor monitor can diff a page against what it looked like last time.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_urls (
    url TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS opportunities (
    key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT,
    type TEXT,
    deadline TEXT,
    score INTEGER,
    verdict TEXT,
    payload TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    url TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_url ON snapshots(url, fetched_at);
"""

# Where an opportunity sits in the founder's application pipeline.
PIPELINE_STATUSES = ("new", "saved", "applying", "submitted", "won", "lost", "dismissed")

MIGRATIONS = {
    "status": "ALTER TABLE opportunities ADD COLUMN status TEXT NOT NULL DEFAULT 'new'",
    "notes": "ALTER TABLE opportunities ADD COLUMN notes TEXT NOT NULL DEFAULT ''",
}


def content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(opportunities)")}
        with self.conn:
            for column, ddl in MIGRATIONS.items():
                if column not in columns:
                    self.conn.execute(ddl)

    def close(self) -> None:
        self.conn.close()

    # --- processed URLs -------------------------------------------------------
    def is_seen(self, url: str) -> bool:
        with self._lock:
            row = self.conn.execute("SELECT 1 FROM seen_urls WHERE url = ?", (url,)).fetchone()
        return row is not None

    def mark_seen(self, url: str, now: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO seen_urls(url, first_seen) VALUES (?, ?)", (url, now)
            )

    # --- opportunities --------------------------------------------------------
    def has_opportunity(self, key: str) -> bool:
        with self._lock:
            row = self.conn.execute("SELECT 1 FROM opportunities WHERE key = ?", (key,)).fetchone()
        return row is not None

    def upsert_opportunity(self, key: str, record: dict[str, Any], now: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO opportunities
                    (key, name, url, type, deadline, score, verdict, payload, first_seen, updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    url = excluded.url, type = excluded.type, deadline = excluded.deadline,
                    score = excluded.score, verdict = excluded.verdict,
                    payload = excluded.payload, updated = excluded.updated
                """,
                (
                    key,
                    record["name"],
                    record.get("url"),
                    record.get("type"),
                    record.get("deadline"),
                    record.get("score"),
                    record.get("verdict"),
                    json.dumps(record, default=str),
                    now,
                    now,
                ),
            )

    def list_opportunities(
        self, min_score: int = 0, statuses: list[str] | None = None
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT payload, status, notes, updated FROM opportunities "
            "WHERE COALESCE(score, 0) >= ?"
        )
        params: list[Any] = [min_score]
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            params += statuses
        sql += " ORDER BY COALESCE(deadline, '9999-12-31') ASC, score DESC"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [
            {**json.loads(payload), "status": status, "notes": notes, "updated": updated}
            for payload, status, notes, updated in rows
        ]

    # --- application pipeline -------------------------------------------------
    def set_status(self, key: str, status: str, notes: str | None, now: str) -> bool:
        if status not in PIPELINE_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._lock, self.conn:
            cur = self.conn.execute(
                "UPDATE opportunities SET status = ?, notes = COALESCE(?, notes), updated = ? "
                "WHERE key = ?",
                (status, notes, now, key),
            )
        return cur.rowcount > 0

    # --- page snapshots -------------------------------------------------------
    def latest_snapshot(self, url: str) -> tuple[str, str] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT content_hash, content FROM snapshots WHERE url = ? "
                "ORDER BY fetched_at DESC, rowid DESC LIMIT 1",
                (url,),
            ).fetchone()
        return (row[0], row[1]) if row else None

    def add_snapshot(self, url: str, content: str, now: str) -> str:
        digest = content_hash(content)
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO snapshots(url, fetched_at, content_hash, content) VALUES (?, ?, ?, ?)",
                (url, now, digest, content),
            )
        return digest
