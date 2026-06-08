from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import db
from .models import EvidencePacket, SearchResult
from .utils import compact_whitespace


def fts_query(text: str) -> str:
    raw_tokens = re.findall(r"[\w\u4e00-\u9fff]+", text, flags=re.UNICODE)
    tokens: List[str] = []
    for token in raw_tokens:
        tokens.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            tokens.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    if not tokens:
        return '""'
    unique_tokens = []
    seen = set()
    for token in tokens:
        cleaned = token.replace('"', "")
        if cleaned and cleaned not in seen:
            unique_tokens.append(cleaned)
            seen.add(cleaned)
    return " OR ".join(unique_tokens[:16])


def search_nodes(
    db_path: Path,
    query: str,
    doc_id: Optional[str] = None,
    top_k: int = 8,
) -> List[SearchResult]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        return _search_nodes_conn(conn, query, doc_id=doc_id, top_k=top_k)
    finally:
        conn.close()


def search_documents(db_path: Path, query: str, top_k: int = 8) -> List[Dict[str, object]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        match = fts_query(query)
        rows = conn.execute(
            """
            WITH matches AS (
              SELECT doc_id, bm25(doc_nodes_fts) AS score
              FROM doc_nodes_fts
              WHERE doc_nodes_fts MATCH ?
              ORDER BY score ASC
              LIMIT 200
            )
            SELECT d.doc_id, d.title, d.path, d.file_type, d.summary,
                   COUNT(*) AS node_matches, MIN(matches.score) AS score
            FROM matches
            JOIN documents d ON d.doc_id = matches.doc_id
            WHERE d.status = 'ready'
            GROUP BY d.doc_id
            ORDER BY score ASC, node_matches DESC
            LIMIT ?
            """,
            (match, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = _fallback_doc_search(conn, query, top_k)
    finally:
        conn.close()
    return [dict(row) for row in rows]


def get_evidence(
    db_path: Path,
    doc_id: str,
    node_ids: Iterable[str],
) -> List[EvidencePacket]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        return db.get_evidence_packets(conn, doc_id, node_ids)
    finally:
        conn.close()


def _search_nodes_conn(
    conn,
    query: str,
    doc_id: Optional[str] = None,
    top_k: int = 8,
) -> List[SearchResult]:  # type: ignore[no-untyped-def]
    match = fts_query(query)
    params: List[object] = [match]
    doc_filter = ""
    if doc_id:
        doc_filter = "AND f.doc_id = ?"
        params.append(doc_id)
    params.append(top_k)
    try:
        rows = conn.execute(
            f"""
            SELECT f.node_id, f.doc_id, d.title, d.path, n.node_path, n.heading,
                   snippet(doc_nodes_fts, 4, '[', ']', ' ... ', 24) AS snippet,
                   bm25(doc_nodes_fts) AS score,
                   n.page_start, n.page_end
            FROM doc_nodes_fts f
            JOIN doc_nodes n ON n.node_id = f.node_id
            JOIN documents d ON d.doc_id = f.doc_id
            WHERE doc_nodes_fts MATCH ? {doc_filter}
              AND d.status = 'ready'
              AND n.type != 'document'
              AND COALESCE(n.text, '') != ''
            ORDER BY score ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        rows = _fallback_node_search(conn, query, doc_id, top_k)
    return [
        SearchResult(
            doc_id=row["doc_id"],
            node_id=row["node_id"],
            title=row["title"],
            path=row["path"],
            node_path=row["node_path"],
            heading=row["heading"],
            snippet=compact_whitespace(row["snippet"] or ""),
            score=float(row["score"] or 0.0),
            page_start=row["page_start"],
            page_end=row["page_end"],
        )
        for row in rows
    ]


def _fallback_node_search(conn, query: str, doc_id: Optional[str], top_k: int):  # type: ignore[no-untyped-def]
    like = f"%{query}%"
    params: List[object] = [like, like, like]
    doc_filter = ""
    if doc_id:
        doc_filter = "AND n.doc_id = ?"
        params.append(doc_id)
    params.append(top_k)
    return conn.execute(
        f"""
        SELECT n.node_id, n.doc_id, d.title, d.path, n.node_path, n.heading,
               substr(COALESCE(NULLIF(n.text, ''), n.summary), 1, 500) AS snippet,
               0.0 AS score,
               n.page_start, n.page_end
        FROM doc_nodes n
        JOIN documents d ON d.doc_id = n.doc_id
        WHERE (n.text LIKE ? OR n.summary LIKE ? OR n.heading LIKE ?)
          {doc_filter}
          AND d.status = 'ready'
          AND n.type != 'document'
          AND COALESCE(n.text, '') != ''
        ORDER BY n.order_index ASC
        LIMIT ?
        """,
        params,
    ).fetchall()


def _fallback_doc_search(conn, query: str, top_k: int):  # type: ignore[no-untyped-def]
    like = f"%{query}%"
    return conn.execute(
        """
        SELECT d.doc_id, d.title, d.path, d.file_type, d.summary,
               1 AS node_matches, 0.0 AS score
        FROM documents d
        WHERE d.status = 'ready'
          AND (d.title LIKE ? OR d.summary LIKE ? OR d.path LIKE ?)
        ORDER BY d.updated_at DESC
        LIMIT ?
        """,
        (like, like, like, top_k),
    ).fetchall()
