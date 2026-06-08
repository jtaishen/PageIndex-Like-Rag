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
        rows = list(conn.execute(
            """
            WITH matches AS (
              SELECT doc_id, bm25(doc_nodes_fts) AS score
              FROM doc_nodes_fts
              WHERE doc_nodes_fts MATCH ?
              ORDER BY score ASC
              LIMIT 200
            )
            SELECT d.doc_id, d.title, d.path, d.file_type, d.summary, d.abstract, d.keywords,
                   COUNT(*) AS node_matches, MIN(matches.score) AS score
            FROM matches
            JOIN documents d ON d.doc_id = matches.doc_id
            WHERE d.status = 'ready'
            GROUP BY d.doc_id
            ORDER BY score ASC, node_matches DESC
            LIMIT ?
            """,
            (match, top_k),
        ).fetchall())
        if len(rows) < top_k:
            seen = {row["doc_id"] for row in rows}
            for row in _fallback_doc_search(conn, query, top_k * 2):
                if row["doc_id"] not in seen:
                    rows.append(row)
                    seen.add(row["doc_id"])
                if len(rows) >= top_k:
                    break
    except sqlite3.OperationalError:
        rows = list(_fallback_doc_search(conn, query, top_k))
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
        rows = list(conn.execute(
            f"""
            SELECT f.node_id, f.doc_id, d.title, d.path, n.node_path, n.heading, n.type,
                   snippet(doc_nodes_fts, 4, '[', ']', ' ... ', 24) AS snippet,
                   bm25(doc_nodes_fts) AS score,
                   n.page_start, n.page_end, n.order_index
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
        ).fetchall())
        if len(rows) < top_k:
            seen = {row["node_id"] for row in rows}
            for row in _fallback_node_search(conn, query, doc_id, top_k * 3):
                if row["node_id"] not in seen:
                    rows.append(row)
                    seen.add(row["node_id"])
                if len(rows) >= top_k * 2:
                    break
    except sqlite3.OperationalError:
        rows = list(_fallback_node_search(conn, query, doc_id, top_k))
    rows = _rank_node_rows(rows, query)[:top_k]
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
    terms = _fallback_terms(query)
    conditions = []
    params: List[object] = []
    for term in terms:
        like = f"%{term}%"
        conditions.append("(n.text LIKE ? OR n.summary LIKE ? OR n.heading LIKE ? OR n.node_path LIKE ?)")
        params.extend([like, like, like, like])
    doc_filter = ""
    if doc_id:
        doc_filter = "AND n.doc_id = ?"
        params.append(doc_id)
    params.append(top_k)
    where_terms = " OR ".join(conditions) if conditions else "1 = 0"
    return conn.execute(
        f"""
        SELECT n.node_id, n.doc_id, d.title, d.path, n.node_path, n.heading, n.type,
               substr(COALESCE(NULLIF(n.text, ''), n.summary), 1, 500) AS snippet,
               0.0 AS score,
               n.page_start, n.page_end, n.order_index
        FROM doc_nodes n
        JOIN documents d ON d.doc_id = n.doc_id
        WHERE ({where_terms})
          {doc_filter}
          AND d.status = 'ready'
          AND n.type != 'document'
          AND COALESCE(n.text, '') != ''
        ORDER BY n.order_index ASC
        LIMIT ?
        """,
        params,
    ).fetchall()


def _rank_node_rows(rows, query: str):  # type: ignore[no-untyped-def]
    terms = _fallback_terms(query)

    def key(row):  # type: ignore[no-untyped-def]
        text = " ".join(str(row[name] or "") for name in ("heading", "node_path", "snippet"))
        penalty = 0
        if row["type"] in {"document", "page"}:
            penalty += 3
        if row["type"] == "reference" and not any(term in query for term in ("参考", "引用", "reference")):
            penalty += 3
        if _looks_like_front_matter(text):
            penalty += 3
        if any(term and term in text for term in terms):
            penalty -= 3
        if row["type"] in {"section", "abstract"}:
            penalty -= 1
        return (penalty, float(row["score"] or 0.0), int(row["order_index"] or 0))

    return sorted(rows, key=key)


def _fallback_terms(query: str) -> List[str]:
    terms: List[str] = []
    known_phrases = [
        "主要研究内容",
        "研究内容",
        "主要贡献",
        "研究背景",
        "研究目的",
        "任务规划",
        "任务分配",
        "参考文献",
        "关键词",
        "摘要",
        "结论",
    ]
    for phrase in known_phrases:
        if phrase in query:
            terms.append(phrase)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{5,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    seen = set()
    result = []
    stopwords = {"这篇", "论文", "什么", "是什么", "主要", "研究"}
    for term in terms:
        cleaned = term.strip()
        if not cleaned or cleaned in stopwords or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= 10:
            break
    return result or [query]


def _looks_like_front_matter(text: str) -> bool:
    compacted = text.replace(" ", "")
    front_matter_tokens = ("目录", "学院", "专业", "研究生", "指导教师", "答辩", "分类号", "学号", "密级")
    return any(token in compacted for token in front_matter_tokens) or bool(re.search(r"(\.{4,}|…{2,})", text))


def _fallback_doc_search(conn, query: str, top_k: int):  # type: ignore[no-untyped-def]
    like = f"%{query}%"
    return conn.execute(
        """
        SELECT d.doc_id, d.title, d.path, d.file_type, d.summary, d.abstract, d.keywords,
               1 AS node_matches, 0.0 AS score
        FROM documents d
        WHERE d.status = 'ready'
          AND (
            d.title LIKE ?
            OR d.abstract LIKE ?
            OR d.keywords LIKE ?
            OR d.summary LIKE ?
            OR d.path LIKE ?
          )
        ORDER BY d.updated_at DESC
        LIMIT ?
        """,
        (like, like, like, like, like, top_k),
    ).fetchall()
