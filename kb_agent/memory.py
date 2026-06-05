from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from . import db
from .utils import stable_id


def put_memory(
    db_path: Path,
    scope: str,
    type_: str,
    subject_key: str,
    content: str,
    importance: float = 0.5,
    confidence: float = 1.0,
    ttl: Optional[float] = None,
    refs: str = "",
) -> Dict[str, object]:
    conn = db.connect(db_path)
    db.init_db(conn)
    now = time.time()
    memory_id = stable_id("mem", scope, type_, subject_key)
    try:
        conn.execute(
            """
            INSERT INTO memory_items(
                memory_id, scope, type, subject_key, content, refs, ttl,
                importance, confidence, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                content = excluded.content,
                refs = excluded.refs,
                ttl = excluded.ttl,
                importance = excluded.importance,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (
                memory_id,
                scope,
                type_,
                subject_key,
                content,
                refs,
                ttl,
                importance,
                confidence,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"memory_id": memory_id, "scope": scope, "subject_key": subject_key}


def search_memory(db_path: Path, query: str, scope: Optional[str] = None, top_k: int = 8) -> List[Dict[str, object]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    like = f"%{query}%"
    params: List[object] = [like, like, like]
    scope_filter = ""
    if scope:
        scope_filter = "AND scope = ?"
        params.append(scope)
    params.append(top_k)
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM memory_items
            WHERE (content LIKE ? OR subject_key LIKE ? OR type LIKE ?)
              {scope_filter}
              AND (ttl IS NULL OR ttl > ?)
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            params[:-1] + [time.time(), params[-1]],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

