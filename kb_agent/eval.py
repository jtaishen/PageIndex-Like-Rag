from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .artifacts import get_parse_quality
from .config import DATA_DIR
from .memory import evaluate_memory_write, resume_task
from .search import search_documents, search_nodes
from .tasks import get_task_artifact
from .utils import compact_whitespace, unique_strings as _unique_strings, write_json


def eval_search(
    db_path: Path,
    queries_path: Path,
    search_mode: str = "hybrid",
    top_k: int = 5,
    compare_modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if isinstance(raw_queries, list):
        queries = raw_queries
        eval_set_schema = "json_list"
    elif isinstance(raw_queries, dict) and isinstance(raw_queries.get("queries"), list):
        queries = raw_queries["queries"]
        eval_set_schema = str(raw_queries.get("schema") or "search_eval_set")
    else:
        raise ValueError("Search eval file must be a JSON list or an object with a queries list.")

    modes = _unique_strings(compare_modes or [search_mode])
    mode_results = {
        mode: _eval_search_mode(db_path, queries, queries_path, mode, top_k)
        for mode in modes
    }
    primary = mode_results[modes[0]]
    mode_rankings = _mode_rankings(mode_results)
    report = {
        "schema": "search_eval.v2",
        "queries_path": str(queries_path),
        "eval_set_schema": eval_set_schema,
        "search_mode": search_mode,
        "compare_modes": modes,
        "top_k": top_k,
        "query_count": primary["query_count"],
        "doc_recall_at_k": primary["doc_recall_at_k"],
        "node_recall_at_k": primary["node_recall_at_k"],
        "node_keyword_hit_rate": primary["node_keyword_hit_rate"],
        "evidence_precision": primary["evidence_precision"],
        "mrr": primary["mrr"],
        "evidence_count": primary["evidence_count"],
        "fallback_count": primary["fallback_count"],
        "weak_parse_quality_count": primary["weak_parse_quality_count"],
        "mode_results": mode_results,
        "best_mode_by_node_recall": _best_mode(mode_results, "node_recall_at_k"),
        "mode_rankings": mode_rankings,
        "per_intent_metrics": _per_intent_metrics(mode_results),
        "created_at": time.time(),
    }
    path = _write_eval_report("search_eval", report)
    return {**report, "path": str(path)}


def eval_review(db_path: Path, task_id: str) -> Dict[str, Any]:
    warnings: List[str] = []
    review_report = _task_artifact_content(db_path, task_id, "review_report.json", warnings)
    citation_check = _task_artifact_content(db_path, task_id, "citation_check.json", warnings)
    outline = _task_artifact_content(db_path, task_id, "review_outline.json", warnings)

    sections = outline.get("sections") or []
    section_count = len(sections)
    drafted_section_count = int(review_report.get("drafted_section_count") or 0)
    coverage_score = float(
        review_report.get("citation_coverage_score")
        or citation_check.get("coverage_score")
        or 0.0
    )
    missing_refs = citation_check.get("missing_refs") or []
    unsupported = citation_check.get("unsupported_paragraphs") or []
    status = "passed" if coverage_score >= 0.8 and not missing_refs and not unsupported else "needs_review"
    if missing_refs:
        warnings.append("missing_refs")
    if unsupported:
        warnings.append("unsupported_paragraphs")
    if section_count and drafted_section_count < section_count:
        warnings.append("incomplete_section_drafts")

    report = {
        "schema": "review_eval.v1",
        "task_id": task_id,
        "status": status,
        "section_count": section_count,
        "drafted_section_count": drafted_section_count,
        "citation_coverage_score": coverage_score,
        "missing_ref_count": len(missing_refs),
        "unsupported_paragraph_count": len(unsupported),
        "warnings": _unique_strings([*warnings, *review_report.get("warnings", [])]),
        "created_at": time.time(),
    }
    path = _write_eval_report(f"review_eval_{task_id}", report)
    return {**report, "path": str(path)}


def eval_memory(db_path: Path) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    now = time.time()
    try:
        rows = [dict(row) for row in conn.execute("SELECT * FROM memory_items").fetchall()]
    finally:
        conn.close()

    expired = [row for row in rows if row.get("ttl") is not None and float(row.get("ttl") or 0.0) <= now]
    duplicate_groups = _duplicate_memory_groups(rows)
    pollution = []
    for row in rows:
        gate = evaluate_memory_write(
            str(row.get("scope") or ""),
            str(row.get("type") or ""),
            str(row.get("subject_key") or ""),
            str(row.get("content") or ""),
            confidence=float(row.get("confidence") or 1.0),
        )
        if gate.get("reason") == "paper_asset_boundary":
            pollution.append(
                {
                    "memory_id": row.get("memory_id"),
                    "scope": row.get("scope"),
                    "type": row.get("type"),
                    "subject_key": row.get("subject_key"),
                }
            )
    resumed = resume_task(db_path)
    warnings = []
    if expired:
        warnings.append("expired_memory_items")
    if duplicate_groups:
        warnings.append("duplicate_memory_subjects")
    if pollution:
        warnings.append("suspected_memory_pollution")
    if not resumed.get("current_task") and not resumed.get("remembered_tasks"):
        warnings.append("no_resumable_task_memory")

    report = {
        "schema": "memory_eval.v1",
        "status": "needs_review" if warnings else "passed",
        "memory_count": len(rows),
        "expired_count": len(expired),
        "duplicate_subject_count": len(duplicate_groups),
        "suspected_pollution_count": len(pollution),
        "suspected_pollution": pollution,
        "resume_available": bool(resumed.get("current_task") or resumed.get("remembered_tasks")),
        "warnings": warnings,
        "created_at": time.time(),
    }
    path = _write_eval_report("memory_eval", report)
    return {**report, "path": str(path)}


def eval_facts(db_path: Path, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    clean_doc_ids = _unique_strings(doc_ids or [])
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        rows = _fact_rows(conn, clean_doc_ids or None)
        if not clean_doc_ids:
            clean_doc_ids = _unique_strings(str(row.get("doc_id") or "") for row in rows if row.get("doc_id"))
            if not clean_doc_ids:
                clean_doc_ids = [
                    str(row["doc_id"])
                    for row in conn.execute("SELECT doc_id FROM documents WHERE status = 'ready'").fetchall()
                ]
    finally:
        conn.close()

    total = len(rows)
    claim_count = sum(1 for row in rows if row.get("fact_type") == "claim")
    entity_count = sum(1 for row in rows if row.get("fact_type") == "entity")
    relation_count = sum(1 for row in rows if row.get("fact_type") == "relation")
    no_node = sum(1 for row in rows if not row.get("node_id"))
    with_node = total - no_node
    low_confidence = sum(1 for row in rows if float(row.get("confidence") or 0.0) < 0.5)
    table_backed = sum(1 for row in rows if "table" in str(row.get("source") or "").lower())
    duplicates = _duplicate_fact_groups(rows)
    weak_docs = []
    for doc_id in clean_doc_ids:
        try:
            quality = get_parse_quality(db_path, doc_id)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        quality_warnings = quality.get("quality_warnings") or []
        if quality.get("quality_level") == "weak" or any(
            warning in quality_warnings
            for warning in ("page_only_tree", "weak_layout_blocks", "weak_table_parse", "missing_table_content")
        ):
            weak_docs.append(
                {
                    "doc_id": doc_id,
                    "quality_level": quality.get("quality_level"),
                    "quality_warnings": quality_warnings,
                    "table_parse_score": quality.get("table_parse_score"),
                }
            )

    warnings: List[str] = []
    if total == 0:
        warnings.append("no_facts")
    if no_node:
        warnings.append("facts_without_node_id")
    if low_confidence:
        warnings.append("low_confidence_facts")
    if duplicates:
        warnings.append("duplicate_facts")
    if weak_docs:
        warnings.append("weak_parse_quality_for_facts")
    if table_backed == 0 and total:
        warnings.append("no_table_backed_facts")

    report = {
        "schema": "fact_eval.v1",
        "status": "needs_review" if warnings else "passed",
        "doc_ids": clean_doc_ids,
        "doc_count": len(clean_doc_ids),
        "total_fact_count": total,
        "claim_count": claim_count,
        "entity_count": entity_count,
        "relation_count": relation_count,
        "evidence_coverage_rate": round(with_node / max(1, total), 4),
        "low_confidence_count": low_confidence,
        "low_confidence_rate": round(low_confidence / max(1, total), 4),
        "duplicate_group_count": len(duplicates),
        "duplicate_rate": round(sum(group["count"] - 1 for group in duplicates) / max(1, total), 4),
        "no_node_id_count": no_node,
        "table_backed_fact_count": table_backed,
        "table_backed_fact_rate": round(table_backed / max(1, total), 4),
        "weak_parse_doc_count": len(weak_docs),
        "weak_parse_docs": weak_docs,
        "duplicate_groups": duplicates[:20],
        "warnings": warnings,
        "created_at": time.time(),
    }
    path = _write_eval_report("fact_eval", report)
    return {**report, "path": str(path)}


def _eval_search_mode(
    db_path: Path,
    queries: List[Any],
    queries_path: Path,
    search_mode: str,
    top_k: int,
) -> Dict[str, Any]:
    items = []
    doc_recall_values: List[float] = []
    node_recall_values: List[float] = []
    mrr_values: List[float] = []
    precision_values: List[float] = []
    keyword_hits = 0
    keyword_total = 0
    evidence_count = 0
    fallback_count = 0
    weak_parse_quality_count = 0

    for raw_item in queries:
        if not isinstance(raw_item, dict):
            continue
        query = str(raw_item.get("query") or "").strip()
        if not query:
            continue
        expected_docs = [str(item) for item in raw_item.get("expected_doc_ids") or []]
        expected_nodes = [str(item) for item in raw_item.get("expected_node_ids") or []]
        expected_keywords = [str(item) for item in raw_item.get("expected_node_keywords") or []]

        docs = search_documents(db_path, query, top_k=top_k, search_mode=search_mode)
        nodes = search_nodes(db_path, query, top_k=top_k, search_mode=search_mode)
        result_doc_ids = [str(item.get("doc_id")) for item in docs]
        result_node_ids = [node.node_id for node in nodes]
        node_text = "\n".join(
            compact_whitespace(f"{node.title} {node.node_path} {node.heading} {node.snippet}")
            for node in nodes
        )

        doc_recall = _recall(result_doc_ids, expected_docs)
        node_recall = _recall(result_node_ids, expected_nodes)
        mrr = _mrr(result_doc_ids, expected_docs)
        precision = _precision(result_node_ids, expected_nodes)
        doc_recall_values.append(doc_recall)
        node_recall_values.append(node_recall)
        mrr_values.append(mrr)
        precision_values.append(precision)
        keyword_hit = all(keyword in node_text for keyword in expected_keywords) if expected_keywords else True
        if expected_keywords:
            keyword_total += 1
            if keyword_hit:
                keyword_hits += 1
        evidence_count += len(nodes)
        fallback_used = any("fts_fallback" in node.rank_reason for node in nodes)
        if fallback_used:
            fallback_count += 1
        for doc_id in set(result_doc_ids):
            try:
                quality = get_parse_quality(db_path, doc_id)
            except (KeyError, FileNotFoundError, ValueError):
                continue
            if quality.get("quality_level") == "weak":
                weak_parse_quality_count += 1

        items.append(
            {
                "query": query,
                "intent": raw_item.get("intent") or "",
                "expected_doc_ids": expected_docs,
                "expected_node_ids": expected_nodes,
                "expected_node_keywords": expected_keywords,
                "result_doc_ids": result_doc_ids,
                "result_node_ids": result_node_ids,
                "doc_recall_at_k": doc_recall,
                "node_recall_at_k": node_recall,
                "evidence_precision": precision,
                "mrr": mrr,
                "node_keyword_hit": keyword_hit,
                "fallback_used": fallback_used,
            }
        )

    return {
        "schema": "search_eval_mode.v1",
        "queries_path": str(queries_path),
        "search_mode": search_mode,
        "top_k": top_k,
        "query_count": len(items),
        "doc_recall_at_k": _average(doc_recall_values),
        "node_recall_at_k": _average(node_recall_values),
        "node_keyword_hit_rate": (keyword_hits / keyword_total) if keyword_total else 1.0,
        "evidence_precision": _average(precision_values),
        "mrr": _average(mrr_values),
        "evidence_count": evidence_count,
        "fallback_count": fallback_count,
        "weak_parse_quality_count": weak_parse_quality_count,
        "items": items,
    }


def _task_artifact_content(db_path: Path, task_id: str, name: str, warnings: List[str]) -> Dict[str, Any]:
    try:
        payload = get_task_artifact(db_path, task_id, name)["content"]
    except (FileNotFoundError, ValueError, KeyError) as exc:
        warnings.append(f"missing_task_artifact:{name}:{exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _fact_rows(conn, doc_ids: Optional[List[str]]) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    params: List[Any] = []
    filter_sql = ""
    if doc_ids:
        placeholders = ",".join("?" for _ in doc_ids)
        filter_sql = f"WHERE doc_id IN ({placeholders})"
        params.extend(doc_ids)
    queries = [
        (
            "claim",
            f"""
            SELECT 'claim' AS fact_type, claim_id AS fact_id, doc_id, version_id, node_id,
                   claim_type AS type, normalized_text AS key_text, confidence, source, evidence_json
            FROM paper_claims {filter_sql}
            """,
        ),
        (
            "entity",
            f"""
            SELECT 'entity' AS fact_type, entity_id AS fact_id, doc_id, version_id, node_id,
                   entity_type AS type, normalized_name AS key_text, confidence, source, evidence_json
            FROM paper_entities {filter_sql}
            """,
        ),
        (
            "relation",
            f"""
            SELECT 'relation' AS fact_type, relation_id AS fact_id, doc_id, version_id, node_id,
                   relation_type AS type, subject_name || '->' || object_name AS key_text,
                   confidence, source, evidence_json
            FROM paper_relations {filter_sql}
            """,
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for _, query in queries:
        rows.extend(dict(row) for row in conn.execute(query, params).fetchall())
    return rows


def _duplicate_fact_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        parts = [
            str(row.get("doc_id") or ""),
            str(row.get("fact_type") or ""),
            str(row.get("type") or ""),
            compact_whitespace(str(row.get("key_text") or "")).lower(),
        ]
        if str(row.get("type") or "") in {"cites", "citation"}:
            parts.append(str(row.get("node_id") or ""))
        key = "|".join(parts)
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in groups.items():
        if len(items) > 1:
            result.append(
                {
                    "key": key,
                    "count": len(items),
                    "fact_ids": [str(item.get("fact_id") or "") for item in items[:8]],
                }
            )
    return sorted(result, key=lambda item: (-int(item["count"]), item["key"]))


def _duplicate_memory_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = "|".join(str(row.get(name) or "") for name in ("scope", "type", "subject_key"))
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in groups.items():
        if len(items) > 1:
            result.append({"key": key, "count": len(items)})
    return result


def _write_eval_report(prefix: str, report: Dict[str, Any]) -> Path:
    out_dir = DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_{int(report['created_at'])}.json"
    write_json(out_path, report)
    return out_path


def _recall(results: List[str], expected: List[str]) -> float:
    if not expected:
        return 1.0
    hits = len(set(results) & set(expected))
    return hits / len(set(expected))


def _precision(results: List[str], expected: List[str]) -> float:
    if not expected:
        return 1.0
    if not results:
        return 0.0
    hits = len(set(results) & set(expected))
    return hits / len(set(results))


def _mrr(results: List[str], expected: List[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    for index, doc_id in enumerate(results, start=1):
        if doc_id in expected_set:
            return 1.0 / index
    return 0.0


def _average(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _best_mode(mode_results: Dict[str, Dict[str, Any]], metric: str) -> str:
    if not mode_results:
        return ""
    return max(mode_results.items(), key=lambda item: float(item[1].get(metric) or 0.0))[0]


def _mode_rankings(mode_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rankings = []
    for mode, result in mode_results.items():
        query_count = max(1, int(result.get("query_count") or 0))
        fallback_rate = float(result.get("fallback_count") or 0) / query_count
        weak_rate = float(result.get("weak_parse_quality_count") or 0) / query_count
        score = _mode_score(
            float(result.get("doc_recall_at_k") or 0.0),
            float(result.get("node_recall_at_k") or 0.0),
            float(result.get("evidence_precision") or 0.0),
            float(result.get("mrr") or 0.0),
            float(result.get("node_keyword_hit_rate") or 0.0),
            fallback_rate,
            weak_rate,
        )
        rankings.append(
            {
                "search_mode": mode,
                "score": score,
                "doc_recall_at_k": result.get("doc_recall_at_k") or 0.0,
                "node_recall_at_k": result.get("node_recall_at_k") or 0.0,
                "evidence_precision": result.get("evidence_precision") or 0.0,
                "mrr": result.get("mrr") or 0.0,
                "fallback_rate": round(fallback_rate, 4),
                "weak_parse_quality_rate": round(weak_rate, 4),
            }
        )
    return sorted(rankings, key=lambda item: (-float(item["score"]), str(item["search_mode"])))


def _per_intent_metrics(mode_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for mode, result in mode_results.items():
        for item in result.get("items") or []:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent") or "unknown")
            grouped.setdefault(intent, {}).setdefault(mode, []).append(item)
    metrics: Dict[str, Dict[str, Any]] = {}
    for intent, by_mode in grouped.items():
        mode_metrics = {}
        for mode, items in by_mode.items():
            count = len(items)
            fallback_rate = sum(1 for item in items if item.get("fallback_used")) / count if count else 0.0
            mode_metrics[mode] = {
                "query_count": count,
                "doc_recall_at_k": _average([float(item.get("doc_recall_at_k") or 0.0) for item in items]),
                "node_recall_at_k": _average([float(item.get("node_recall_at_k") or 0.0) for item in items]),
                "evidence_precision": _average([float(item.get("evidence_precision") or 0.0) for item in items]),
                "mrr": _average([float(item.get("mrr") or 0.0) for item in items]),
                "fallback_rate": round(fallback_rate, 4),
            }
        metrics[intent] = {
            "modes": mode_metrics,
            "recommended_mode": _recommended_intent_mode(mode_metrics),
        }
    return metrics


def _recommended_intent_mode(mode_metrics: Dict[str, Dict[str, Any]]) -> str:
    if not mode_metrics:
        return ""
    ranked = []
    for mode, item in mode_metrics.items():
        score = _mode_score(
            float(item.get("doc_recall_at_k") or 0.0),
            float(item.get("node_recall_at_k") or 0.0),
            float(item.get("evidence_precision") or 0.0),
            float(item.get("mrr") or 0.0),
            1.0,
            float(item.get("fallback_rate") or 0.0),
            0.0,
        )
        ranked.append((mode, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def _mode_score(
    doc_recall: float,
    node_recall: float,
    precision: float,
    mrr: float,
    keyword_hit: float,
    fallback_rate: float,
    weak_rate: float,
) -> float:
    return round(
        doc_recall * 0.3
        + node_recall * 0.35
        + precision * 0.15
        + mrr * 0.15
        + keyword_hit * 0.05
        - fallback_rate * 0.05
        - weak_rate * 0.03,
        6,
    )
