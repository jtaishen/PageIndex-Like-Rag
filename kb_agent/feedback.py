from __future__ import annotations

import json
import re
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .config import DATA_DIR
from .query_log import query_stats
from .utils import compact_whitespace, stable_id, write_json


COMMENT_LIMIT = 600
SENSITIVE_COMMENT_PATTERNS = [
    re.compile(r"\bnode_id\s*=", re.IGNORECASE),
    re.compile(r"\bpage_range\s*=", re.IGNORECASE),
    re.compile(r"\bexcerpt\s*=", re.IGNORECASE),
    re.compile(r"\bevidence\s*=", re.IGNORECASE),
    re.compile(r"\braw_text\b", re.IGNORECASE),
    re.compile(r"\btree_search_trace\b", re.IGNORECASE),
    re.compile(r"\breview_draft\b", re.IGNORECASE),
    re.compile(r"\bsection_drafts\b", re.IGNORECASE),
    re.compile(r"raw_text\.txt", re.IGNORECASE),
    re.compile(r"body\.md", re.IGNORECASE),
]


def put_feedback(
    db_path: Path,
    *,
    query: str = "",
    query_id: str = "",
    operation: str = "",
    rating: int = 0,
    label: str = "",
    comment: str = "",
    expected_doc_ids: Optional[Iterable[str]] = None,
    expected_node_ids: Optional[Iterable[str]] = None,
    expected_keywords: Optional[Iterable[str]] = None,
    preferred_search_mode: str = "",
) -> Dict[str, Any]:
    rating_value = int(rating)
    if rating_value < 1 or rating_value > 5:
        raise ValueError("rating must be between 1 and 5")

    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        log_row = None
        if query_id:
            log_row = conn.execute("SELECT * FROM query_logs WHERE query_id = ?", (query_id,)).fetchone()
        if log_row is not None:
            query = query or str(log_row["query"] or "")
            operation = operation or str(log_row["operation"] or "")
        query = compact_whitespace(query)
        if not query:
            raise ValueError("query is required when query_id cannot provide it")

        warnings: List[str] = []
        clean_comment, comment_status = sanitize_feedback_comment(comment, warnings)
        now = time.time()
        feedback_id = stable_id("feedback", query_id, operation, query, label, now, length=12)
        clean_docs = _unique_strings(expected_doc_ids or [])
        clean_nodes = _unique_strings(expected_node_ids or [])
        clean_keywords = _unique_strings(expected_keywords or [])
        preferred_mode = _preferred_mode(preferred_search_mode)
        conn.execute(
            """
            INSERT OR REPLACE INTO feedback_items(
                feedback_id, query_id, operation, query, rating, label, comment,
                expected_doc_ids, expected_node_ids, expected_keywords,
                preferred_search_mode, warnings, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                compact_whitespace(query_id),
                compact_whitespace(operation),
                query,
                rating_value,
                compact_whitespace(label),
                clean_comment,
                json.dumps(clean_docs, ensure_ascii=False),
                json.dumps(clean_nodes, ensure_ascii=False),
                json.dumps(clean_keywords, ensure_ascii=False),
                preferred_mode,
                json.dumps(warnings, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()
        item = _row_to_feedback_item(
            {
                "feedback_id": feedback_id,
                "query_id": query_id,
                "operation": operation,
                "query": query,
                "rating": rating_value,
                "label": label,
                "comment": clean_comment,
                "expected_doc_ids": json.dumps(clean_docs, ensure_ascii=False),
                "expected_node_ids": json.dumps(clean_nodes, ensure_ascii=False),
                "expected_keywords": json.dumps(clean_keywords, ensure_ascii=False),
                "preferred_search_mode": preferred_mode,
                "warnings": json.dumps(warnings, ensure_ascii=False),
                "created_at": now,
                "updated_at": now,
            }
        )
    finally:
        conn.close()

    return {
        "schema": "feedback_write.v1",
        "feedback_id": feedback_id,
        "accepted": True,
        "comment_status": comment_status,
        "warnings": warnings,
        "item": item,
    }


def list_feedback(
    db_path: Path,
    *,
    limit: int = 20,
    operation: Optional[str] = None,
    label: Optional[str] = None,
    rating: Optional[int] = None,
    min_rating: Optional[int] = None,
) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    filters: List[str] = []
    params: List[Any] = []
    if operation:
        filters.append("operation = ?")
        params.append(operation)
    if label:
        filters.append("label = ?")
        params.append(label)
    if rating is not None:
        filters.append("rating = ?")
        params.append(int(rating))
    if min_rating is not None:
        filters.append("rating >= ?")
        params.append(int(min_rating))
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(max(1, int(limit)))
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM feedback_items
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        items = [_row_to_feedback_item(dict(row)) for row in rows]
    finally:
        conn.close()
    return {
        "schema": "feedback_list.v1",
        "limit": limit,
        "filters": {
            "operation": operation,
            "label": label,
            "rating": rating,
            "min_rating": min_rating,
        },
        "count": len(items),
        "items": items,
    }


def build_eval_set_from_feedback(
    db_path: Path,
    *,
    output_path: Optional[Path] = None,
    min_rating: int = 4,
    label: Optional[str] = None,
    operation: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    feedback = list_feedback(
        db_path,
        limit=limit,
        operation=operation,
        label=label,
        min_rating=min_rating,
    )
    queries = []
    skipped = 0
    for item in feedback["items"]:
        expected_doc_ids = _unique_strings(item.get("expected_doc_ids") or [])
        expected_node_ids = _unique_strings(item.get("expected_node_ids") or [])
        expected_keywords = _unique_strings(item.get("expected_keywords") or [])
        if not (expected_doc_ids or expected_node_ids or expected_keywords):
            skipped += 1
            continue
        queries.append(
            {
                "query": item.get("query") or "",
                "intent": "",
                "expected_doc_ids": expected_doc_ids,
                "expected_node_ids": expected_node_ids,
                "expected_node_keywords": expected_keywords,
                "source_feedback_id": item.get("feedback_id"),
                "query_id": item.get("query_id") or "",
                "operation": item.get("operation") or "",
                "preferred_search_mode": item.get("preferred_search_mode") or "",
            }
        )

    created_at = time.time()
    payload = {
        "schema": "search_eval_set.v1",
        "source": "feedback_items",
        "created_at": created_at,
        "min_rating": min_rating,
        "label": label or "",
        "operation": operation or "",
        "query_count": len(queries),
        "skipped_feedback_count": skipped,
        "queries": queries,
    }
    path = output_path or _default_eval_set_path(created_at)
    write_json(path, payload)
    return {**payload, "path": str(path)}


def eval_dashboard(db_path: Path, *, since_days: Optional[float] = None, output_format: str = "json") -> Dict[str, Any]:
    from .search_profile import latest_search_tuning_reports, list_search_profiles

    created_at = time.time()
    stats = query_stats(db_path, since_days=since_days)
    feedback = feedback_summary(db_path, since_days=since_days)
    latest_reports = _latest_eval_reports(limit=8)
    tuning_reports = latest_search_tuning_reports(limit=5)
    profiles = list_search_profiles()
    recommendations = _dashboard_recommendations(stats, feedback)
    dashboard = {
        "schema": "eval_dashboard.v1",
        "since_days": since_days,
        "created_at": created_at,
        "query_stats": stats,
        "feedback_summary": feedback,
        "latest_eval_reports": latest_reports,
        "latest_search_tuning": tuning_reports,
        "search_profiles": profiles,
        "recommendations": recommendations,
    }
    out_dir = DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval_dashboard_{int(created_at)}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    html_path = out_dir / f"{stem}.html"
    write_json(json_path, dashboard)
    md_path.write_text(_dashboard_markdown(dashboard), encoding="utf-8")
    html_path.write_text(_dashboard_html(dashboard), encoding="utf-8")
    format_path = {
        "json": json_path,
        "md": md_path,
        "html": html_path,
    }.get((output_format or "json").strip().lower(), json_path)
    return {**dashboard, "path": str(format_path), "json_path": str(json_path), "md_path": str(md_path), "html_path": str(html_path)}


def feedback_summary(db_path: Path, *, since_days: Optional[float] = None) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    params: List[Any] = []
    filters: List[str] = []
    if since_days is not None:
        filters.append("created_at >= ?")
        params.append(time.time() - since_days * 86400)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    try:
        rows = [dict(row) for row in conn.execute(f"SELECT * FROM feedback_items {where}", params).fetchall()]
    finally:
        conn.close()
    items = [_row_to_feedback_item(row) for row in rows]
    rating_values = [int(item.get("rating") or 0) for item in items]
    label_counts: Dict[str, int] = {}
    operation_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    warning_counts: Dict[str, int] = {}
    for item in items:
        _inc(label_counts, str(item.get("label") or "unlabeled"))
        _inc(operation_counts, str(item.get("operation") or "unknown"))
        _inc(mode_counts, str(item.get("preferred_search_mode") or "unspecified"))
        for warning in item.get("warnings") or []:
            _inc(warning_counts, str(warning))
    return {
        "schema": "feedback_summary.v1",
        "since_days": since_days,
        "feedback_count": len(items),
        "avg_rating": round(sum(rating_values) / len(rating_values), 3) if rating_values else 0.0,
        "low_rating_count": sum(1 for value in rating_values if value and value <= 2),
        "label_counts": label_counts,
        "operation_counts": operation_counts,
        "preferred_search_mode_counts": mode_counts,
        "top_warnings": _top_counts(warning_counts),
    }


def sanitize_feedback_comment(comment: str, warnings: List[str]) -> tuple[str, str]:
    clean = compact_whitespace(comment or "")
    if not clean:
        return "", "empty"
    if any(pattern.search(clean) for pattern in SENSITIVE_COMMENT_PATTERNS):
        warnings.append("comment_rejected:paper_asset_boundary")
        return "", "rejected"
    if len(clean) > COMMENT_LIMIT:
        warnings.append("comment_truncated")
        return clean[:COMMENT_LIMIT], "truncated"
    return clean, "accepted"


def _row_to_feedback_item(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "feedback_id": row.get("feedback_id") or "",
        "query_id": row.get("query_id") or "",
        "operation": row.get("operation") or "",
        "query": row.get("query") or "",
        "rating": int(row.get("rating") or 0),
        "label": row.get("label") or "",
        "comment": row.get("comment") or "",
        "expected_doc_ids": _json_list(row.get("expected_doc_ids")),
        "expected_node_ids": _json_list(row.get("expected_node_ids")),
        "expected_keywords": _json_list(row.get("expected_keywords")),
        "preferred_search_mode": row.get("preferred_search_mode") or "",
        "warnings": _json_list(row.get("warnings")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _default_eval_set_path(created_at: float) -> Path:
    out_dir = DATA_DIR / "eval_sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"feedback_eval_{int(created_at)}.json"


def _latest_eval_reports(limit: int = 8) -> List[Dict[str, Any]]:
    out_dir = DATA_DIR / "eval"
    if not out_dir.exists():
        return []
    reports = []
    for path in sorted(out_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reports.append(
            {
                "path": str(path),
                "schema": payload.get("schema") or "",
                "status": payload.get("status") or "",
                "query_count": payload.get("query_count", 0),
                "created_at": payload.get("created_at"),
                "warnings": payload.get("warnings") or [],
            }
        )
        if len(reports) >= limit:
            break
    return reports


def _dashboard_recommendations(stats: Dict[str, Any], feedback: Dict[str, Any]) -> List[str]:
    recommendations = []
    if stats.get("no_evidence_rate", 0) > 0:
        recommendations.append("复盘无证据查询，补充 expected_doc_ids 或 expected_node_ids 后生成评测集。")
    if stats.get("fallback_rate", 0) > 0:
        recommendations.append("检查 hybrid fallback 查询，必要时刷新 embedding 或切换 search_mode。")
    if feedback.get("low_rating_count", 0) > 0:
        recommendations.append("将低评分反馈转成评测用例，并比较 hybrid/tree/fts 的召回差异。")
    if not feedback.get("feedback_count"):
        recommendations.append("尚无人工反馈；建议从最近 query-log 中挑选代表性失败案例记录反馈。")
    return recommendations


def _dashboard_markdown(dashboard: Dict[str, Any]) -> str:
    stats = dashboard.get("query_stats") or {}
    feedback = dashboard.get("feedback_summary") or {}
    reports = dashboard.get("latest_eval_reports") or []
    lines = [
        "# KB Eval Dashboard",
        "",
        f"- schema: `{dashboard.get('schema')}`",
        f"- since_days: `{dashboard.get('since_days')}`",
        f"- query_count: `{stats.get('query_count', 0)}`",
        f"- feedback_count: `{feedback.get('feedback_count', 0)}`",
        f"- avg_feedback_rating: `{feedback.get('avg_rating', 0.0)}`",
        f"- fallback_rate: `{stats.get('fallback_rate', 0.0)}`",
        f"- no_evidence_rate: `{stats.get('no_evidence_rate', 0.0)}`",
        "",
        "## Feedback Labels",
    ]
    for label, count in sorted((feedback.get("label_counts") or {}).items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Latest Eval Reports"])
    for report in reports:
        lines.append(f"- `{report.get('schema')}` status=`{report.get('status')}` path=`{report.get('path')}`")
    lines.extend(["", "## Search Tuning"])
    for report in dashboard.get("latest_search_tuning") or []:
        lines.append(f"- default=`{report.get('default_mode')}` path=`{report.get('path')}`")
    active = ((dashboard.get("search_profiles") or {}).get("active") or {})
    if active:
        lines.extend(["", "## Active Search Profile", f"- `{active.get('name')}` path=`{active.get('path')}`"])
    lines.extend(["", "## Recommendations"])
    for item in dashboard.get("recommendations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _dashboard_html(dashboard: Dict[str, Any]) -> str:
    stats = dashboard.get("query_stats") or {}
    feedback = dashboard.get("feedback_summary") or {}
    profiles = dashboard.get("search_profiles") or {}
    active = profiles.get("active") or {}
    cards = [
        ("Queries", stats.get("query_count", 0)),
        ("Fallback Rate", stats.get("fallback_rate", 0.0)),
        ("No Evidence Rate", stats.get("no_evidence_rate", 0.0)),
        ("Feedback", feedback.get("feedback_count", 0)),
        ("Avg Rating", feedback.get("avg_rating", 0.0)),
        ("Low Ratings", feedback.get("low_rating_count", 0)),
    ]
    card_html = "\n".join(
        f"<section class='card'><div class='label'>{escape(str(label))}</div><div class='value'>{escape(str(value))}</div></section>"
        for label, value in cards
    )
    labels_html = _count_table(feedback.get("label_counts") or {}, "Feedback Labels")
    modes_html = _count_table(stats.get("search_mode_counts") or {}, "Search Modes")
    tuning_html = _list_section(
        "Search Tuning",
        [
            f"default={item.get('default_mode', '')} path={item.get('path', '')}"
            for item in dashboard.get("latest_search_tuning") or []
        ],
    )
    reports_html = _list_section(
        "Latest Eval Reports",
        [
            f"{item.get('schema', '')} status={item.get('status', '')} path={item.get('path', '')}"
            for item in dashboard.get("latest_eval_reports") or []
        ],
    )
    recommendations_html = _list_section("Recommendations", dashboard.get("recommendations") or [])
    active_html = ""
    if active:
        active_html = (
            "<section class='panel'><h2>Active Search Profile</h2>"
            f"<p><strong>{escape(str(active.get('name') or ''))}</strong></p>"
            f"<p>{escape(str(active.get('path') or ''))}</p></section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>KB Eval Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; background: #f7f8fa; }}
    header {{ padding: 28px 32px 16px; background: #ffffff; border-bottom: 1px solid #d9dee7; }}
    h1 {{ margin: 0; font-size: 26px; letter-spacing: 0; }}
    main {{ padding: 24px 32px 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card, .panel {{ background: #ffffff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
    .label {{ font-size: 12px; color: #64748b; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    h2 {{ font-size: 16px; margin: 0 0 10px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ border-top: 1px solid #edf1f5; padding: 7px 0; font-size: 13px; }}
    ul {{ padding-left: 18px; margin: 0; }}
    li {{ margin: 6px 0; font-size: 13px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>KB Eval Dashboard</h1>
    <p>Generated at {escape(str(dashboard.get('created_at') or ''))}</p>
  </header>
  <main>
    <div class="grid">{card_html}</div>
    <div class="panels">
      {labels_html}
      {modes_html}
      {active_html}
      {tuning_html}
      {reports_html}
      {recommendations_html}
    </div>
  </main>
</body>
</html>
"""


def _count_table(counts: Dict[str, Any], title: str) -> str:
    rows = "\n".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(str(value))}</td></tr>"
        for key, value in sorted(counts.items())
    )
    return f"<section class='panel'><h2>{escape(title)}</h2><table>{rows}</table></section>"


def _list_section(title: str, items: List[Any]) -> str:
    rows = "\n".join(f"<li>{escape(str(item))}</li>" for item in items)
    return f"<section class='panel'><h2>{escape(title)}</h2><ul>{rows}</ul></section>"


def _json_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def _preferred_mode(value: str) -> str:
    text = compact_whitespace(value)
    return text if text in {"hybrid", "tree", "fts"} else ""


def _unique_strings(values: Iterable[Any]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = compact_whitespace(str(value))
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _inc(counts: Dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _top_counts(counts: Dict[str, int], limit: int = 10) -> List[Dict[str, Any]]:
    return [
        {"item": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
