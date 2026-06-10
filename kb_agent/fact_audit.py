from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .artifacts import get_artifact, get_citation_map
from .config import DATA_DIR
from .utils import compact_whitespace, read_json as _read_json, stable_id, unique_strings as _unique_strings, write_json


AUDIT_SCHEMA = "fact_audit.v1"
CONFLICT_SCHEMA = "fact_conflicts.v1"
EVAL_DIR = DATA_DIR / "eval"

POSITIVE_TERMS = ("提升", "提高", "增加", "上升", "优于", "改善", "增强", "支持", "有效", "更高", "较高")
NEGATIVE_TERMS = ("降低", "下降", "减少", "弱于", "低于", "差于", "不足", "限制", "失败", "无效", "退化", "更低", "较低")
ANCHOR_TERMS = (
    "任务完成率",
    "任务成功率",
    "成功率",
    "响应时间",
    "负载均衡",
    "通信开销",
    "计算开销",
    "鲁棒性",
    "任务规划",
    "任务分配",
    "协同调度",
    "动态角色",
    "本文方法",
    "基线方法",
)


def audit_facts(
    db_path: Path,
    *,
    doc_ids: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    write_report: bool = True,
) -> Dict[str, Any]:
    clean_doc_ids = _unique_strings(doc_ids or [])
    if doc_ids is not None and not clean_doc_ids:
        rows: List[Dict[str, Any]] = []
    else:
        rows = _load_fact_rows(db_path, clean_doc_ids or None)
    if not clean_doc_ids and doc_ids is None:
        clean_doc_ids = _unique_strings(row["doc_id"] for row in rows if row.get("doc_id"))

    duplicate_groups = _duplicate_groups(rows)
    low_confidence = [row for row in rows if float(row.get("confidence") or 0.0) < min_confidence]
    no_evidence = [row for row in rows if not row.get("node_id")]
    conflicts = _detect_conflicts(rows)
    table_mismatches = [item for item in conflicts if item.get("kind") == "table_text_mismatch"]
    citation_gaps = _citation_gaps(db_path, clean_doc_ids, rows)
    artifact_summaries = _artifact_summaries(db_path, clean_doc_ids)
    warnings = _audit_warnings(rows, duplicate_groups, low_confidence, no_evidence, conflicts, citation_gaps)
    created_at = time.time()
    audit_id = stable_id("fact_audit", ",".join(clean_doc_ids), len(rows), created_at, length=12)
    report = {
        "schema": AUDIT_SCHEMA,
        "audit_id": audit_id,
        "doc_ids": clean_doc_ids,
        "doc_count": len(clean_doc_ids),
        "min_confidence": min_confidence,
        "total_fact_count": len(rows),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_fact_count": sum(max(0, int(item.get("count") or 0) - 1) for item in duplicate_groups),
        "duplicates": duplicate_groups[:50],
        "low_confidence_count": len(low_confidence),
        "low_confidence_facts": [_fact_ref(item) for item in low_confidence[:100]],
        "no_evidence_count": len(no_evidence),
        "no_evidence_facts": [_fact_ref(item) for item in no_evidence[:100]],
        "conflict_count": len(conflicts),
        "high_severity_conflict_count": sum(1 for item in conflicts if item.get("severity") == "high"),
        "conflicts": conflicts[:100],
        "table_text_mismatch_count": len(table_mismatches),
        "table_text_mismatches": table_mismatches[:50],
        "citation_gap_count": len(citation_gaps),
        "citation_gaps": citation_gaps,
        "artifact_summaries": artifact_summaries,
        "status": "needs_review" if warnings else "passed",
        "warnings": warnings,
        "created_at": created_at,
    }
    if write_report:
        json_path = EVAL_DIR / f"fact_audit_{audit_id}.json"
        md_path = EVAL_DIR / f"fact_audit_{audit_id}.md"
        write_json(json_path, report)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_audit_markdown(report), encoding="utf-8")
        return {**report, "path": str(json_path), "md_path": str(md_path)}
    return report


def get_fact_conflicts(
    db_path: Path,
    *,
    doc_ids: Optional[List[str]] = None,
    severity: Optional[str] = None,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    severity_filter = compact_whitespace(severity or "").lower()
    if severity_filter and severity_filter not in {"low", "medium", "high"}:
        raise ValueError("severity must be one of: low, medium, high")
    audit = audit_facts(db_path, doc_ids=doc_ids, min_confidence=min_confidence, write_report=True)
    conflicts = [item for item in audit.get("conflicts") or [] if not severity_filter or item.get("severity") == severity_filter]
    return {
        "schema": CONFLICT_SCHEMA,
        "audit_id": audit.get("audit_id"),
        "audit_path": audit.get("path", ""),
        "doc_ids": audit.get("doc_ids") or [],
        "severity": severity_filter or "",
        "count": len(conflicts),
        "high_severity_count": sum(1 for item in conflicts if item.get("severity") == "high"),
        "conflicts": conflicts,
        "warnings": audit.get("warnings") or [],
        "created_at": time.time(),
    }


def fact_audit_summary(
    db_path: Path,
    *,
    doc_ids: Optional[List[str]] = None,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    audit = audit_facts(db_path, doc_ids=doc_ids, min_confidence=min_confidence, write_report=False)
    return {
        "schema": "fact_audit_summary.v1",
        "doc_ids": audit.get("doc_ids") or [],
        "total_fact_count": audit.get("total_fact_count", 0),
        "duplicate_group_count": audit.get("duplicate_group_count", 0),
        "low_confidence_count": audit.get("low_confidence_count", 0),
        "no_evidence_count": audit.get("no_evidence_count", 0),
        "conflict_count": audit.get("conflict_count", 0),
        "high_severity_conflict_count": audit.get("high_severity_conflict_count", 0),
        "table_text_mismatch_count": audit.get("table_text_mismatch_count", 0),
        "citation_gap_count": audit.get("citation_gap_count", 0),
        "top_conflicts": (audit.get("conflicts") or [])[:5],
        "warnings": audit.get("warnings") or [],
    }


def fact_conflict_summary(
    db_path: Path,
    query: str = "",
    *,
    doc_ids: Optional[List[str]] = None,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    audit = audit_facts(db_path, doc_ids=doc_ids, min_confidence=min_confidence, write_report=False)
    terms = _query_terms(query)
    conflicts = audit.get("conflicts") or []
    if terms:
        conflicts = [
            item
            for item in conflicts
            if any(term in compact_whitespace(f"{item.get('anchor', '')} {item.get('reason', '')}") for term in terms)
        ] or audit.get("conflicts") or []
    conflicts = conflicts[:8]
    return {
        "schema": "fact_conflict_summary.v1",
        "doc_ids": audit.get("doc_ids") or [],
        "query": compact_whitespace(query),
        "conflict_count": audit.get("conflict_count", 0),
        "high_severity_conflict_count": audit.get("high_severity_conflict_count", 0),
        "items": conflicts,
        "warnings": audit.get("warnings") or [],
    }


def latest_fact_audit_report(limit: int = 1) -> List[Dict[str, Any]]:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(EVAL_DIR.glob("fact_audit_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = _read_json(path, {})
        if payload.get("schema") != AUDIT_SCHEMA:
            continue
        items.append(
            {
                "path": str(path),
                "schema": payload.get("schema") or "",
                "status": payload.get("status") or "",
                "audit_id": payload.get("audit_id") or "",
                "doc_count": payload.get("doc_count", 0),
                "total_fact_count": payload.get("total_fact_count", 0),
                "conflict_count": payload.get("conflict_count", 0),
                "high_severity_conflict_count": payload.get("high_severity_conflict_count", 0),
                "table_text_mismatch_count": payload.get("table_text_mismatch_count", 0),
                "citation_gap_count": payload.get("citation_gap_count", 0),
                "warnings": payload.get("warnings") or [],
                "created_at": payload.get("created_at"),
            }
        )
        if len(items) >= limit:
            break
    return items


def _load_fact_rows(db_path: Path, doc_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        return [
            *_select_claims(conn, doc_ids),
            *_select_entities(conn, doc_ids),
            *_select_relations(conn, doc_ids),
        ]
    finally:
        conn.close()


def _select_claims(conn, doc_ids: Optional[List[str]]) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    doc_filter, params = _doc_filter(doc_ids)
    rows = conn.execute(
        f"""
        SELECT claim_id AS fact_id, 'claim' AS fact_type, doc_id, version_id, node_id,
               claim_type AS type, text, normalized_text AS normalized_key,
               page_range, confidence, source, evidence_json, '' AS subject_name, '' AS object_name
        FROM paper_claims {doc_filter}
        """,
        params,
    ).fetchall()
    return [_normalize_fact(dict(row)) for row in rows]


def _select_entities(conn, doc_ids: Optional[List[str]]) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    doc_filter, params = _doc_filter(doc_ids)
    rows = conn.execute(
        f"""
        SELECT entity_id AS fact_id, 'entity' AS fact_type, doc_id, version_id, node_id,
               entity_type AS type, name AS text, normalized_name AS normalized_key,
               page_range, confidence, source, evidence_json, '' AS subject_name, '' AS object_name
        FROM paper_entities {doc_filter}
        """,
        params,
    ).fetchall()
    return [_normalize_fact(dict(row)) for row in rows]


def _select_relations(conn, doc_ids: Optional[List[str]]) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    doc_filter, params = _doc_filter(doc_ids)
    rows = conn.execute(
        f"""
        SELECT relation_id AS fact_id, 'relation' AS fact_type, doc_id, version_id, node_id,
               relation_type AS type, text, '' AS normalized_key, page_range,
               confidence, source, evidence_json, subject_name, object_name
        FROM paper_relations {doc_filter}
        """,
        params,
    ).fetchall()
    return [_normalize_fact(dict(row)) for row in rows]


def _doc_filter(doc_ids: Optional[List[str]]) -> tuple[str, List[Any]]:
    clean = _unique_strings(doc_ids or [])
    if not clean:
        return "", []
    return f"WHERE doc_id IN ({','.join('?' for _ in clean)})", clean


def _normalize_fact(row: Dict[str, Any]) -> Dict[str, Any]:
    text = compact_whitespace(str(row.get("text") or ""))
    subject = compact_whitespace(str(row.get("subject_name") or ""))
    obj = compact_whitespace(str(row.get("object_name") or ""))
    combined = compact_whitespace(f"{text} {subject} {obj}")
    source = str(row.get("source") or "")
    item = {
        "fact_id": row.get("fact_id") or "",
        "fact_type": row.get("fact_type") or "",
        "doc_id": row.get("doc_id") or "",
        "version_id": row.get("version_id") or "",
        "node_id": row.get("node_id") or "",
        "type": row.get("type") or "",
        "text": _excerpt(text, 220),
        "subject_name": subject,
        "object_name": obj,
        "normalized_key": _fact_key(row.get("normalized_key") or combined),
        "page_range": _json_value(row.get("page_range"), []),
        "confidence": float(row.get("confidence") or 0.0),
        "source": source,
        "source_kind": "table" if _is_table_source(source) else "text",
        "evidence": _json_value(row.get("evidence_json"), {}),
        "polarity": _polarity(combined),
        "anchors": _anchors(row.get("type") or "", combined, subject, obj),
    }
    return item


def _duplicate_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        parts = [
            row.get("doc_id") or "",
            row.get("fact_type") or "",
            row.get("type") or "",
            row.get("normalized_key") or "",
        ]
        if row.get("type") in {"cites", "citation"}:
            parts.append(row.get("node_id") or "")
        key = "|".join(parts)
        if key.endswith("|"):
            continue
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in groups.items():
        if len(items) <= 1:
            continue
        result.append(
            {
                "key": key,
                "count": len(items),
                "fact_ids": [item["fact_id"] for item in items[:8]],
                "doc_ids": _unique_strings(item["doc_id"] for item in items),
                "sample": [_fact_ref(item) for item in items[:3]],
            }
        )
    return sorted(result, key=lambda item: (-int(item["count"]), item["key"]))


def _detect_conflicts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anchored: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("polarity") == "neutral":
            continue
        for anchor in row.get("anchors") or []:
            anchored.setdefault(anchor, []).append(row)
    conflicts: Dict[str, Dict[str, Any]] = {}
    for anchor, items in anchored.items():
        for left, right in combinations(items, 2):
            if left["fact_id"] == right["fact_id"] or not _opposes(left, right):
                continue
            kind = "table_text_mismatch" if left["doc_id"] == right["doc_id"] and left["source_kind"] != right["source_kind"] else "cross_doc_conflict"
            if kind == "cross_doc_conflict" and left["doc_id"] == right["doc_id"]:
                continue
            conflict = _conflict_item(kind, anchor, left, right)
            conflicts[conflict["conflict_id"]] = conflict
    return sorted(conflicts.values(), key=lambda item: (_severity_rank(item["severity"]), item["anchor"], item["conflict_id"]))


def _conflict_item(kind: str, anchor: str, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    severity = _severity(kind, left, right)
    reason = f"{kind}:{anchor}:{left.get('polarity')}_vs_{right.get('polarity')}"
    return {
        "conflict_id": stable_id("conflict", kind, anchor, left["fact_id"], right["fact_id"], length=12),
        "kind": kind,
        "severity": severity,
        "anchor": anchor,
        "reason": reason,
        "left": _fact_ref(left),
        "right": _fact_ref(right),
    }


def _fact_ref(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fact_id": item.get("fact_id") or "",
        "fact_type": item.get("fact_type") or "",
        "type": item.get("type") or "",
        "doc_id": item.get("doc_id") or "",
        "version_id": item.get("version_id") or "",
        "node_id": item.get("node_id") or "",
        "page_range": item.get("page_range") or [],
        "confidence": item.get("confidence", 0.0),
        "source": item.get("source") or "",
        "source_kind": item.get("source_kind") or "",
        "summary": _excerpt(str(item.get("text") or item.get("subject_name") or ""), 160),
    }


def _citation_gaps(db_path: Path, doc_ids: List[str], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    relation_counts: Dict[str, int] = {}
    for row in rows:
        if row.get("fact_type") == "relation" and row.get("type") == "cites":
            relation_counts[row["doc_id"]] = relation_counts.get(row["doc_id"], 0) + 1
    gaps = []
    for doc_id in doc_ids:
        try:
            citation_map = get_citation_map(db_path, doc_id)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        reference_count = len(citation_map.get("references") or [])
        relation_count = _unique_citation_relation_count(citation_map.get("relations") or [])
        db_cites = relation_counts.get(doc_id, 0)
        if reference_count and db_cites == 0:
            gaps.append(
                {
                    "doc_id": doc_id,
                    "reason": "missing_citation_fact_relations",
                    "reference_count": reference_count,
                    "citation_map_relation_count": relation_count,
                    "fact_cite_relation_count": db_cites,
                }
            )
        elif relation_count > db_cites:
            gaps.append(
                {
                    "doc_id": doc_id,
                    "reason": "partial_citation_fact_relations",
                    "reference_count": reference_count,
                    "citation_map_relation_count": relation_count,
                    "fact_cite_relation_count": db_cites,
                }
            )
    return gaps


def _unique_citation_relation_count(relations: Iterable[Any]) -> int:
    keys = set()
    for item in relations:
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("ref_id") or "")
        node_id = str(item.get("node_id") or "")
        if ref_id and node_id:
            keys.add((ref_id, node_id))
    return len(keys)


def _artifact_summaries(db_path: Path, doc_ids: List[str]) -> List[Dict[str, Any]]:
    summaries = []
    for doc_id in doc_ids:
        fact_graph = _artifact_content(db_path, doc_id, "fact_graph.json", {})
        table_content = _artifact_content(db_path, doc_id, "table_content.json", {})
        citation_map = _artifact_content(db_path, doc_id, "citation_map.json", {})
        summaries.append(
            {
                "doc_id": doc_id,
                "fact_graph_nodes": len(fact_graph.get("nodes") or []),
                "fact_graph_edges": len(fact_graph.get("edges") or []),
                "table_content_count": table_content.get("count", 0),
                "citation_reference_count": len(citation_map.get("references") or []),
                "citation_relation_count": len(citation_map.get("relations") or []),
            }
        )
    return summaries


def _artifact_content(db_path: Path, doc_id: str, name: str, default: Any) -> Any:
    try:
        return get_artifact(db_path, doc_id, name)["content"]
    except (FileNotFoundError, KeyError, ValueError):
        return default


def _audit_warnings(
    rows: List[Dict[str, Any]],
    duplicate_groups: List[Dict[str, Any]],
    low_confidence: List[Dict[str, Any]],
    no_evidence: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
    citation_gaps: List[Dict[str, Any]],
) -> List[str]:
    warnings = []
    if not rows:
        warnings.append("no_facts_to_audit")
    if duplicate_groups:
        warnings.append("duplicate_facts")
    if low_confidence:
        warnings.append("low_confidence_facts")
    if no_evidence:
        warnings.append("facts_without_evidence")
    if conflicts:
        warnings.append("fact_conflicts_detected")
    if any(item.get("kind") == "table_text_mismatch" for item in conflicts):
        warnings.append("table_text_mismatch")
    if citation_gaps:
        warnings.append("citation_relation_gaps")
    return warnings


def _audit_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Fact Audit",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- audit_id: `{report.get('audit_id')}`",
        f"- doc_count: `{report.get('doc_count')}`",
        f"- total_fact_count: `{report.get('total_fact_count')}`",
        f"- conflict_count: `{report.get('conflict_count')}`",
        f"- high_severity_conflict_count: `{report.get('high_severity_conflict_count')}`",
        f"- table_text_mismatch_count: `{report.get('table_text_mismatch_count')}`",
        f"- citation_gap_count: `{report.get('citation_gap_count')}`",
        "",
        "## Warnings",
    ]
    for warning in report.get("warnings") or []:
        lines.append(f"- `{warning}`")
    return "\n".join(lines) + "\n"


def _fact_key(value: Any) -> str:
    text = compact_whitespace(str(value or "")).lower()
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")[:120]


def _polarity(text: str) -> str:
    positive = any(term in text for term in POSITIVE_TERMS)
    negative = any(term in text for term in NEGATIVE_TERMS)
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    if positive and negative:
        first_positive = min((text.find(term) for term in POSITIVE_TERMS if term in text), default=len(text))
        first_negative = min((text.find(term) for term in NEGATIVE_TERMS if term in text), default=len(text))
        return "positive" if first_positive < first_negative else "negative"
    return "neutral"


def _anchors(fact_type: str, text: str, subject: str, obj: str) -> List[str]:
    anchors = [term for term in ANCHOR_TERMS if term in text]
    for value in (subject, obj):
        if value and value in text:
            anchors.append(value)
    if not anchors and fact_type in {"metric", "method", "model", "dataset"}:
        anchors.append(_excerpt(text, 40))
    return _unique_strings(_fact_key(anchor) or anchor for anchor in anchors)


def _opposes(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return {left.get("polarity"), right.get("polarity")} == {"positive", "negative"}


def _severity(kind: str, left: Dict[str, Any], right: Dict[str, Any]) -> str:
    confidence = min(float(left.get("confidence") or 0.0), float(right.get("confidence") or 0.0))
    table_involved = left.get("source_kind") == "table" or right.get("source_kind") == "table"
    if confidence >= 0.75 and (kind == "table_text_mismatch" or table_involved):
        return "high"
    if confidence >= 0.65 or table_involved:
        return "medium"
    return "low"


def _severity_rank(value: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(value, 3)


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not value:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _query_terms(query: str) -> List[str]:
    text = compact_whitespace(query)
    return [term for term in ANCHOR_TERMS if term in text]


def _is_table_source(source: str) -> bool:
    return "table" in source.lower()


def _excerpt(text: str, limit: int) -> str:
    value = compact_whitespace(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "..."
