from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .utils import compact_whitespace, stable_id


SENSITIVE_KEYS = {
    "answer",
    "excerpt",
    "evidence",
    "tree_search_trace",
    "review_draft",
    "content",
    "raw_text",
    "text",
    "body",
}


def write_query_log(
    db_path: Path,
    *,
    operation: str,
    query: str,
    intent: str = "",
    search_mode: str = "",
    status: str = "ok",
    task_id: str = "",
    docs_used: Optional[Iterable[str]] = None,
    nodes_used: Optional[Iterable[str]] = None,
    latency_ms: float = 0.0,
    warnings: Optional[Iterable[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    feedback: str = "",
) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        result = insert_query_log(
            conn,
            operation=operation,
            query=query,
            intent=intent,
            search_mode=search_mode,
            status=status,
            task_id=task_id,
            docs_used=docs_used,
            nodes_used=nodes_used,
            latency_ms=latency_ms,
            warnings=warnings,
            metrics=metrics,
            feedback=feedback,
        )
        conn.commit()
        return result
    finally:
        conn.close()


def insert_query_log(
    conn: Any,
    *,
    operation: str,
    query: str,
    intent: str = "",
    search_mode: str = "",
    status: str = "ok",
    task_id: str = "",
    docs_used: Optional[Iterable[str]] = None,
    nodes_used: Optional[Iterable[str]] = None,
    latency_ms: float = 0.0,
    warnings: Optional[Iterable[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    feedback: str = "",
) -> Dict[str, Any]:
    now = time.time()
    clean_docs = _unique_strings(docs_used or [])
    clean_nodes = _unique_strings(nodes_used or [])
    clean_warnings = _unique_strings(warnings or [])
    clean_metrics = sanitize_metrics(metrics or {})
    query_id = stable_id("query", operation, query, now, length=12)
    conn.execute(
        """
        INSERT OR REPLACE INTO query_logs(
            query_id, operation, intent, query, search_mode, status, task_id,
            docs_used, nodes_used, latency_ms, warnings, metrics_json,
            feedback, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            operation,
            intent,
            query,
            search_mode,
            status,
            task_id,
            json.dumps(clean_docs, ensure_ascii=False),
            json.dumps(clean_nodes, ensure_ascii=False),
            float(latency_ms or 0.0),
            json.dumps(clean_warnings, ensure_ascii=False),
            json.dumps(clean_metrics, ensure_ascii=False),
            compact_whitespace(feedback)[:600],
            now,
        ),
    )
    return {
        "schema": "query_log_write.v1",
        "query_id": query_id,
        "operation": operation,
        "status": status,
        "warning_count": len(clean_warnings),
    }


def list_query_logs(
    db_path: Path,
    *,
    limit: int = 20,
    operation: Optional[str] = None,
    intent: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    filters = []
    params: List[Any] = []
    if operation:
        filters.append("operation = ?")
        params.append(operation)
    if intent:
        filters.append("intent = ?")
        params.append(intent)
    if status:
        filters.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(max(1, limit))
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM query_logs
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        items = [_row_to_log_item(dict(row)) for row in rows]
    finally:
        conn.close()
    return {
        "schema": "query_log_list.v1",
        "limit": limit,
        "filters": {"operation": operation, "intent": intent, "status": status},
        "count": len(items),
        "items": items,
    }


def query_stats(db_path: Path, *, since_days: Optional[float] = None) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    params: List[Any] = []
    filters = []
    if since_days is not None:
        filters.append("created_at >= ?")
        params.append(time.time() - since_days * 86400)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    try:
        rows = conn.execute(f"SELECT * FROM query_logs {where}", params).fetchall()
        items = [_row_to_log_item(dict(row)) for row in rows]
        feedback_rows = [
            dict(row)
            for row in conn.execute(f"SELECT * FROM feedback_items {where}", params).fetchall()
        ]
    finally:
        conn.close()

    operation_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    warning_counts: Dict[str, int] = {}
    latencies: List[float] = []
    fallback_count = 0
    no_evidence_count = 0
    for item in items:
        _inc(operation_counts, str(item.get("operation") or "unknown"))
        _inc(mode_counts, str(item.get("search_mode") or "unknown"))
        _inc(status_counts, str(item.get("status") or "unknown"))
        latencies.append(float(item.get("latency_ms") or 0.0))
        warnings = item.get("warnings") or []
        metrics = item.get("metrics") or {}
        for warning in warnings:
            _inc(warning_counts, str(warning))
        if any("fallback" in str(warning) for warning in warnings) or metrics.get("fallback_used"):
            fallback_count += 1
        if metrics.get("evidence_count") == 0 or "no_tree_evidence" in warnings or "no_search_results" in warnings:
            no_evidence_count += 1

    feedback_label_counts: Dict[str, int] = {}
    feedback_mode_counts: Dict[str, int] = {}
    feedback_ratings: List[int] = []
    for row in feedback_rows:
        rating = int(row.get("rating") or 0)
        if rating:
            feedback_ratings.append(rating)
        _inc(feedback_label_counts, str(row.get("label") or "unlabeled"))
        _inc(feedback_mode_counts, str(row.get("preferred_search_mode") or "unspecified"))

    query_count = len(items)
    return {
        "schema": "query_stats.v1",
        "since_days": since_days,
        "query_count": query_count,
        "operation_counts": operation_counts,
        "search_mode_counts": mode_counts,
        "status_counts": status_counts,
        "avg_latency_ms": round(sum(latencies) / query_count, 3) if query_count else 0.0,
        "failure_rate": round(status_counts.get("failed", 0) / query_count, 4) if query_count else 0.0,
        "fallback_rate": round(fallback_count / query_count, 4) if query_count else 0.0,
        "no_evidence_rate": round(no_evidence_count / query_count, 4) if query_count else 0.0,
        "top_warnings": _top_counts(warning_counts),
        "feedback_count": len(feedback_rows),
        "avg_feedback_rating": round(sum(feedback_ratings) / len(feedback_ratings), 3) if feedback_ratings else 0.0,
        "low_rating_count": sum(1 for value in feedback_ratings if value <= 2),
        "feedback_label_counts": feedback_label_counts,
        "feedback_search_mode_counts": feedback_mode_counts,
    }


def sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return _sanitize_value(metrics)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in SENSITIVE_KEYS:
                continue
            result[normalized_key] = _sanitize_value(item)
        return result
    if isinstance(value, list):
        if len(value) > 40:
            value = value[:40]
        return [_sanitize_value(item) for item in value if not _looks_sensitive(item)]
    if isinstance(value, str):
        return compact_whitespace(value)[:500]
    return value


def _row_to_log_item(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_id": row.get("query_id"),
        "operation": row.get("operation") or "",
        "intent": row.get("intent") or "",
        "query": row.get("query") or "",
        "search_mode": row.get("search_mode") or "",
        "status": row.get("status") or "",
        "task_id": row.get("task_id") or "",
        "docs_used": _json_list(row.get("docs_used")),
        "nodes_used": _json_list(row.get("nodes_used")),
        "latency_ms": row.get("latency_ms") or 0.0,
        "warnings": _json_list(row.get("warnings")),
        "metrics": _json_dict(row.get("metrics_json")),
        "feedback": row.get("feedback") or "",
        "created_at": row.get("created_at"),
    }


def _json_list(value: Any) -> List[Any]:
    if not value:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    return payload if isinstance(payload, list) else []


def _json_dict(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _looks_sensitive(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return any(key in item for key in SENSITIVE_KEYS)


def _unique_strings(values: Iterable[Any]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _inc(counts: Dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _top_counts(counts: Dict[str, int], limit: int = 10) -> List[Dict[str, Any]]:
    return [
        {"warning": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
