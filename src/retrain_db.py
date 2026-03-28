"""
SQLite persistence for retraining uploads: each saved file gets a row (label, path, name, time).
Training still reads images from disk; the DB satisfies audit / coursework "save to database" expectations.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS retrain_upload (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    file_path TEXT NOT NULL,
    original_filename TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrain_upload_label ON retrain_upload(label);
CREATE INDEX IF NOT EXISTS idx_retrain_upload_created ON retrain_upload(created_at);
"""

_lock = threading.Lock()


def init_db(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(str(path), timeout=30.0)
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()


def insert_upload_row(
    db_path: str,
    label: int,
    file_path: str,
    original_filename: Optional[str],
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        try:
            cur = conn.execute(
                """
                INSERT INTO retrain_upload (label, file_path, original_filename, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (label, file_path, original_filename, created_at),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def upload_summary(db_path: str) -> Dict[str, Any]:
    with _lock:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT COUNT(*) AS c FROM retrain_upload").fetchone()
            total = int(row["c"]) if row else 0
            by_label: Dict[str, int] = {"0": 0, "1": 0}
            for r in conn.execute("SELECT label, COUNT(*) AS c FROM retrain_upload GROUP BY label"):
                by_label[str(int(r["label"]))] = int(r["c"])
            return {"total": total, "by_label": by_label}
        finally:
            conn.close()


def list_recent_uploads(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(limit, 500))
    with _lock:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT id, label, file_path, original_filename, created_at
                FROM retrain_upload
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
