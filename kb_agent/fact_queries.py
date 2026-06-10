from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import db
from .artifacts import get_artifact
from .fact_utils import excerpt, is_table_source, json_value, query_terms
from .utils import compact_whitespace, unique_strings


def get_claims(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, "claims.json", version_id=version_id)["content"]


def get_entities(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, "entities.json", version_id=version_id)["content"]


def get_relations(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, "relations.json", version_id=version_id)["content"]


def get_fact_graph(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, "fact_graph.json", version_id=version_id)["content"]


def fact_search(
    db_path: Path,
    query: str,
    *,
    doc_ids: Optional[List[str]] = None,
    fact_type: Optional[str] = None,
    source: str = "all",
    min_confidence: float = 0.0,
    top_k: int = 20,
) -> Dict[str, Any]:
    fact_kind = (fact_type or "").strip().lower()
    if fact_kind and fact_kind not in {"claim", "entity", "relation"}:
        raise ValueError("fact type must be one of: claim, entity, relation")
    source_filter = (source or "all").strip().lower()
    if source_filter not in {"all", "text", "table"}:
        raise ValueError("source must be one of: all, text, table")
    terms = query_terms(query)
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        items: List[Dict[str, Any]] = []
        if fact_kind in {"", "claim"}:
            items.extend(_search_claim_rows(conn, terms, doc_ids, source_filter, min_confidence))
        if fact_kind in {"", "entity"}:
            items.extend(_search_entity_rows(conn, terms, doc_ids, source_filter, min_confidence))
        if fact_kind in {"", "relation"}:
            items.extend(_search_relation_rows(conn, terms, doc_ids, source_filter, min_confidence))
    finally:
        conn.close()
    ranked = sorted(items, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("fact_id") or "")))[: max(1, top_k)]
    return {
        "schema": "fact_search.v1",
        "query": query,
        "type": fact_kind or "all",
        "source": source_filter,
        "min_confidence": min_confidence,
        "doc_ids": doc_ids or [],
        "top_k": top_k,
        "count": len(ranked),
        "items": ranked,
    }


def fact_coverage_summary(db_path: Path, *, doc_id: Optional[str] = None) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        counts = db.paper_fact_counts(conn, doc_id=doc_id)
        source_counts = _fact_source_counts(conn, doc_id=doc_id)
    finally:
        conn.close()
    total = counts["claim_count"] + counts["entity_count"] + counts["relation_count"]
    return {
        "schema": "fact_coverage.v1",
        "doc_id": doc_id or "",
        "total_fact_count": total,
        **counts,
        **source_counts,
        "table_backed_fact_rate": round(source_counts["table_backed_fact_count"] / max(1, total), 4),
    }


def fact_summary_for_doc(db_path: Path, doc_id: str) -> Dict[str, Any]:
    try:
        claims = get_claims(db_path, doc_id)
        entities = get_entities(db_path, doc_id)
        relations = get_relations(db_path, doc_id)
    except (FileNotFoundError, KeyError, ValueError):
        return {"schema": "fact_summary.v1", "doc_id": doc_id, "available": False}
    table_claims = [
        item
        for item in (claims.get("claims") or [])
        if isinstance(item, dict) and is_table_source(str(item.get("source") or ""))
    ]
    table_entities = [
        item
        for item in (entities.get("entities") or [])
        if isinstance(item, dict) and is_table_source(str(item.get("source") or ""))
    ]
    table_relations = [
        item
        for item in (relations.get("relations") or [])
        if isinstance(item, dict) and is_table_source(str(item.get("source") or ""))
    ]
    return {
        "schema": "fact_summary.v1",
        "doc_id": doc_id,
        "available": True,
        "claim_count": int(claims.get("count") or 0),
        "entity_count": int(entities.get("count") or 0),
        "relation_count": int(relations.get("count") or 0),
        "table_backed_fact_count": len(table_claims) + len(table_entities) + len(table_relations),
        "table_claim_count": len(table_claims),
        "table_entity_count": len(table_entities),
        "table_relation_count": len(table_relations),
        "top_claims": [
            {
                "claim_id": item.get("claim_id"),
                "type": item.get("type"),
                "text": excerpt(str(item.get("text") or ""), 180),
            }
            for item in (claims.get("claims") or [])[:5]
            if isinstance(item, dict)
        ],
        "top_entities": [
            {
                "entity_id": item.get("entity_id"),
                "type": item.get("type"),
                "name": item.get("name"),
            }
            for item in (entities.get("entities") or [])[:8]
            if isinstance(item, dict)
        ],
        "top_table_entities": [
            {
                "entity_id": item.get("entity_id"),
                "type": item.get("type"),
                "name": item.get("name"),
                "confidence": item.get("confidence"),
            }
            for item in table_entities[:8]
        ],
        "top_table_relations": [
            {
                "relation_id": item.get("relation_id"),
                "type": item.get("type"),
                "text": excerpt(str(item.get("text") or ""), 180),
                "confidence": item.get("confidence"),
            }
            for item in table_relations[:8]
        ],
    }


def _search_claim_rows(
    conn,
    terms: List[str],
    doc_ids: Optional[List[str]],
    source: str,
    min_confidence: float,
) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    conditions, params = _like_conditions("text", terms)
    doc_filter = _doc_filter(doc_ids, params)
    source_filter = _source_filter(source, params)
    confidence_filter = _confidence_filter(min_confidence, params)
    rows = conn.execute(
        f"""
        SELECT claim_id AS fact_id, 'claim' AS fact_type, doc_id, version_id, node_id,
               claim_type AS type, text, page_range, confidence, source, evidence_json
        FROM paper_claims
        WHERE ({conditions}) {doc_filter} {source_filter} {confidence_filter}
        LIMIT 200
        """,
        params,
    ).fetchall()
    return [_fact_row(dict(row), terms) for row in rows]


def _search_entity_rows(
    conn,
    terms: List[str],
    doc_ids: Optional[List[str]],
    source: str,
    min_confidence: float,
) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    conditions, params = _like_conditions("name", terms)
    doc_filter = _doc_filter(doc_ids, params)
    source_filter = _source_filter(source, params)
    confidence_filter = _confidence_filter(min_confidence, params)
    rows = conn.execute(
        f"""
        SELECT entity_id AS fact_id, 'entity' AS fact_type, doc_id, version_id, node_id,
               entity_type AS type, name AS text, page_range, confidence, source, evidence_json
        FROM paper_entities
        WHERE ({conditions}) {doc_filter} {source_filter} {confidence_filter}
        LIMIT 200
        """,
        params,
    ).fetchall()
    return [_fact_row(dict(row), terms) for row in rows]


def _search_relation_rows(
    conn,
    terms: List[str],
    doc_ids: Optional[List[str]],
    source: str,
    min_confidence: float,
) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    conditions, params = _like_conditions("text || ' ' || subject_name || ' ' || object_name", terms)
    doc_filter = _doc_filter(doc_ids, params)
    source_filter = _source_filter(source, params)
    confidence_filter = _confidence_filter(min_confidence, params)
    rows = conn.execute(
        f"""
        SELECT relation_id AS fact_id, 'relation' AS fact_type, doc_id, version_id, node_id,
               relation_type AS type, text, subject_name, object_name, page_range,
               confidence, source, evidence_json
        FROM paper_relations
        WHERE ({conditions}) {doc_filter} {source_filter} {confidence_filter}
        LIMIT 200
        """,
        params,
    ).fetchall()
    return [_fact_row(dict(row), terms) for row in rows]


def _like_conditions(column_sql: str, terms: List[str]) -> tuple[str, List[Any]]:
    conditions = []
    params: List[Any] = []
    for term in terms:
        conditions.append(f"{column_sql} LIKE ?")
        params.append(f"%{term}%")
    return " OR ".join(conditions) if conditions else "1 = 0", params


def _doc_filter(doc_ids: Optional[List[str]], params: List[Any]) -> str:
    clean = unique_strings(doc_ids or [])
    if not clean:
        return ""
    placeholders = ",".join("?" for _ in clean)
    params.extend(clean)
    return f"AND doc_id IN ({placeholders})"


def _source_filter(source: str, params: List[Any]) -> str:
    if source == "table":
        params.append("%table%")
        return "AND source LIKE ?"
    if source == "text":
        params.append("%table%")
        return "AND source NOT LIKE ?"
    return ""


def _confidence_filter(min_confidence: float, params: List[Any]) -> str:
    try:
        threshold = float(min_confidence)
    except (TypeError, ValueError):
        threshold = 0.0
    if threshold <= 0:
        return ""
    params.append(threshold)
    return "AND confidence >= ?"


def _fact_row(row: Dict[str, Any], terms: List[str]) -> Dict[str, Any]:
    text = compact_whitespace(str(row.get("text") or ""))
    haystack = text + " " + str(row.get("subject_name") or "") + " " + str(row.get("object_name") or "")
    score = sum(1 for term in terms if term and term in haystack)
    return {
        "fact_id": row.get("fact_id") or "",
        "fact_type": row.get("fact_type") or "",
        "doc_id": row.get("doc_id") or "",
        "version_id": row.get("version_id") or "",
        "node_id": row.get("node_id") or "",
        "type": row.get("type") or "",
        "text": excerpt(text, 240),
        "subject_name": row.get("subject_name") or "",
        "object_name": row.get("object_name") or "",
        "page_range": json_value(row.get("page_range"), []),
        "confidence": float(row.get("confidence") or 0.0),
        "source": row.get("source") or "",
        "source_kind": "table" if is_table_source(str(row.get("source") or "")) else "text",
        "evidence": json_value(row.get("evidence_json"), {}),
        "score": score,
    }


def _fact_source_counts(conn, *, doc_id: Optional[str] = None) -> Dict[str, int]:  # type: ignore[no-untyped-def]
    params: List[Any] = []
    where = ""
    if doc_id:
        where = "WHERE doc_id = ?"
        params.append(doc_id)
    table_count = 0
    text_count = 0
    for table in ("paper_claims", "paper_entities", "paper_relations"):
        prefix = f"{where} AND" if where else "WHERE"
        table_row = conn.execute(
            f"SELECT COUNT(*) AS count FROM {table} {prefix} source LIKE ?",
            [*params, "%table%"],
        ).fetchone()
        text_row = conn.execute(
            f"SELECT COUNT(*) AS count FROM {table} {prefix} source NOT LIKE ?",
            [*params, "%table%"],
        ).fetchone()
        table_count += int(table_row["count"] or 0)
        text_count += int(text_row["count"] or 0)
    return {
        "table_backed_fact_count": table_count,
        "text_backed_fact_count": text_count,
    }
