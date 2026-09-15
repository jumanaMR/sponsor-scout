"""Persistent posting store, backed by SQLite.

SQLite because this has to survive a process exit (a scheduled agent is a
fresh process every run) without asking anyone to stand up a database. It's
in the standard library, it's a single file you can copy or delete, and it
handles the one query pattern the app needs — filter by country and
sponsorship label — without an index server.

When this moves to a real deployment the swap is Postgres + pgvector, and
the interface below is deliberately small enough that only this file
changes.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sponsor_scout.sources.base import Posting

SCHEMA = """
CREATE TABLE IF NOT EXISTS postings (
    id                    TEXT PRIMARY KEY,
    company               TEXT NOT NULL,
    role_title            TEXT,
    country               TEXT,
    location              TEXT,
    source                TEXT,
    source_url            TEXT,
    date_posted           TEXT,
    text                  TEXT,
    provider_sponsorship  INTEGER,
    sponsorship_label     TEXT,
    evidence              TEXT,
    first_seen            TEXT NOT NULL,
    notified              INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_postings_country ON postings(country);
CREATE INDEX IF NOT EXISTS idx_postings_label   ON postings(sponsorship_label);
CREATE INDEX IF NOT EXISTS idx_postings_notified ON postings(notified);
"""


class PostingStore:
    def __init__(self, path: str | Path = "data/postings.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- writes ------------------------------------------------------------

    def upsert_many(self, records: Iterable[Dict[str, Any]]) -> List[str]:
        """Insert postings, ignoring ones already stored. Returns the ids
        that were genuinely new — that list IS the alert, so 'new' has to
        mean 'not seen on any previous run', not 'in this batch'."""
        new_ids: List[str] = []
        with closing(self._connect()) as conn:
            for record in records:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO postings
                        (id, company, role_title, country, location, source, source_url,
                         date_posted, text, provider_sponsorship, sponsorship_label,
                         evidence, first_seen, notified)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                    """,
                    (
                        record["id"], record.get("company", ""), record.get("role_title", ""),
                        record.get("country", ""), record.get("location", ""),
                        record.get("source", ""), record.get("source_url", ""),
                        record.get("date_posted", ""), record.get("text", ""),
                        _bool_to_int(record.get("provider_sponsorship")),
                        record.get("sponsorship_label", ""),
                        json.dumps(record.get("evidence", [])),
                        record.get("first_seen", ""),
                    ),
                )
                if cursor.rowcount:
                    new_ids.append(record["id"])
            conn.commit()
        return new_ids

    def mark_notified(self, ids: Iterable[str]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        with closing(self._connect()) as conn:
            conn.executemany("UPDATE postings SET notified = 1 WHERE id = ?", [(i,) for i in ids])
            conn.commit()
            return len(ids)

    def delete_all(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM postings")
            conn.commit()

    # -- reads -------------------------------------------------------------

    def all_postings(
        self,
        country: Optional[str] = None,
        label: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM postings"
        clauses, params = [], []
        if country and country.lower() != "all":
            clauses.append("country = ?")
            params.append(country)
        if label and label.lower() != "all":
            clauses.append("sponsorship_label = ?")
            params.append(label)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY date_posted DESC, company ASC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            return [_row_to_dict(r) for r in conn.execute(query, params).fetchall()]

    def unnotified(self, limit: int = 100) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM postings WHERE notified = 0 ORDER BY date_posted DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def countries(self) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT country,
                       COUNT(*) AS total,
                       SUM(CASE WHEN sponsorship_label = 'likely sponsoring' THEN 1 ELSE 0 END) AS sponsoring
                FROM postings GROUP BY country ORDER BY sponsoring DESC, total DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total,
                       COUNT(DISTINCT country) AS countries,
                       COUNT(DISTINCT company) AS companies,
                       SUM(CASE WHEN sponsorship_label = 'likely sponsoring' THEN 1 ELSE 0 END) AS sponsoring
                FROM postings
                """
            ).fetchone()
        return {k: (row[k] or 0) for k in row.keys()}

    def __len__(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]


def _bool_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return 1 if value else 0


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    if data.get("provider_sponsorship") is not None:
        data["provider_sponsorship"] = bool(data["provider_sponsorship"])
    try:
        data["evidence"] = json.loads(data.get("evidence") or "[]")
    except json.JSONDecodeError:
        data["evidence"] = []
    data["notified"] = bool(data.get("notified"))
    return data


def posting_to_record(posting: Posting, label: str, evidence: List[str], first_seen: str) -> Dict[str, Any]:
    record = posting.to_dict()
    record.update({"sponsorship_label": label, "evidence": evidence, "first_seen": first_seen})
    return record
