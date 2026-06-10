from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .artifacts import get_parse_quality
from .config import DATA_DIR
from .fact_audit import fact_conflict_summary
from .facts import fact_search
from .feedback import list_feedback
from .knowledge_graph import graph_summary
from .query import classify_query
from .query_log import list_query_logs
from .search import build_search_report, search_documents, search_nodes
from .utils import compact_whitespace, read_json as _read_json, stable_id, unique_strings as _unique_strings, write_json


SUITE_DIR = DATA_DIR / "eval_suites"
EVAL_DIR = DATA_DIR / "eval"
SUITE_SCHEMA = "eval_suite.v1"
BENCHMARK_SCHEMA = "benchmark_report.v1"
FAILURE_SCHEMA = "failure_analysis.v1"
CASE_SCHEMA = "case_study.v1"


def create_eval_suite(
    db_path: Path,
    name: str,
    *,
    input_json: Optional[Path] = None,
    from_feedback: bool = False,
    from_query_log: bool = False,
    doc_ids: Optional[List[str]] = None,
    limit: int = 100,
    min_rating: int = 4,
) -> Dict[str, Any]:
    suite_name = _safe_name(name)
    queries: List[Dict[str, Any]] = []
    sources = []
    if input_json:
        queries.extend(_queries_from_json(input_json))
        sources.append(f"json:{input_json}")
    if from_feedback:
        queries.extend(_queries_from_feedback(db_path, limit=limit, min_rating=min_rating))
        sources.append("feedback_items")
    if from_query_log:
        queries.extend(_queries_from_query_logs(db_path, limit=limit))
        sources.append("query_logs")
    if doc_ids:
        queries.extend(_queries_from_doc_ids(doc_ids))
        sources.append("doc_ids")
    queries = _dedupe_queries(_normalize_query_item(item) for item in queries)
    created_at = time.time()
    suite_id = stable_id("suite", suite_name, created_at, length=12)
    payload = {
        "schema": SUITE_SCHEMA,
        "suite_id": suite_id,
        "name": suite_name,
        "sources": sources,
        "query_count": len(queries),
        "queries": queries,
        "created_at": created_at,
        "warnings": [] if queries else ["empty_eval_suite"],
    }
    path = _suite_path(suite_name)
    write_json(path, payload)
    return {**payload, "path": str(path)}


def list_eval_suites() -> Dict[str, Any]:
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(SUITE_DIR.glob("*.json")):
        payload = _read_json(path, {})
        if payload.get("schema") != SUITE_SCHEMA:
            continue
        items.append(
            {
                "name": payload.get("name") or path.stem,
                "suite_id": payload.get("suite_id") or "",
                "query_count": payload.get("query_count", 0),
                "path": str(path),
                "created_at": payload.get("created_at"),
                "warnings": payload.get("warnings") or [],
            }
        )
    return {"schema": "eval_suite_list.v1", "count": len(items), "items": items}


def get_eval_suite(name: str) -> Dict[str, Any]:
    path = _suite_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Eval suite not found: {name}")
    payload = _read_json(path, {})
    if payload.get("schema") != SUITE_SCHEMA:
        raise ValueError(f"Unsupported eval suite schema in {path}")
    return {**payload, "path": str(path)}


def run_benchmark(
    db_path: Path,
    suite_name: str,
    *,
    compare_modes: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    suite = get_eval_suite(suite_name)
    modes = _unique_strings(compare_modes or ["fts", "hybrid", "tree", "auto"])
    queries = [item for item in suite.get("queries") or [] if isinstance(item, dict)]
    mode_results = {mode: _benchmark_mode(db_path, queries, mode, top_k) for mode in modes}
    created_at = time.time()
    benchmark_id = stable_id("benchmark", suite.get("name"), ",".join(modes), created_at, length=12)
    report = {
        "schema": BENCHMARK_SCHEMA,
        "benchmark_id": benchmark_id,
        "suite_name": suite.get("name") or suite_name,
        "suite_id": suite.get("suite_id") or "",
        "suite_path": suite.get("path") or "",
        "compare_modes": modes,
        "top_k": top_k,
        "query_count": len(queries),
        "mode_results": mode_results,
        "best_mode_by_node_recall": _best_mode(mode_results, "node_recall_at_k"),
        "best_mode_by_score": _best_mode(mode_results, "benchmark_score"),
        "summary": _benchmark_summary(mode_results),
        "warnings": _benchmark_warnings(mode_results),
        "created_at": created_at,
    }
    json_path = EVAL_DIR / f"benchmark_{benchmark_id}.json"
    md_path = EVAL_DIR / f"benchmark_{benchmark_id}.md"
    write_json(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_benchmark_markdown(report), encoding="utf-8")
    return {**report, "path": str(json_path), "md_path": str(md_path)}


def analyze_failures(db_path: Path, benchmark_id: str) -> Dict[str, Any]:
    del db_path
    report, report_path = _load_benchmark(benchmark_id)
    failures = []
    reason_counts: Dict[str, int] = {}
    for mode, result in (report.get("mode_results") or {}).items():
        for item in result.get("items") or []:
            reasons = _failure_reasons(item, mode)
            if not reasons:
                continue
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            failures.append(
                {
                    "mode": mode,
                    "query": item.get("query") or "",
                    "category": item.get("category") or "",
                    "reasons": reasons,
                    "result_doc_ids": item.get("result_doc_ids") or [],
                    "result_node_ids": item.get("result_node_ids") or [],
                    "warnings": item.get("warnings") or [],
                }
            )
    actions = _next_actions(reason_counts)
    created_at = time.time()
    payload = {
        "schema": FAILURE_SCHEMA,
        "benchmark_id": report.get("benchmark_id") or benchmark_id,
        "benchmark_path": str(report_path),
        "status": "needs_review" if failures else "passed",
        "failure_count": len(failures),
        "reason_counts": reason_counts,
        "failures": failures[:200],
        "next_actions": actions,
        "created_at": created_at,
    }
    analysis_path = EVAL_DIR / f"failure_analysis_{report.get('benchmark_id') or benchmark_id}.json"
    next_path = EVAL_DIR / f"next_actions_{report.get('benchmark_id') or benchmark_id}.json"
    write_json(analysis_path, payload)
    write_json(
        next_path,
        {
            "schema": "next_actions.v1",
            "benchmark_id": payload["benchmark_id"],
            "items": actions,
            "created_at": created_at,
        },
    )
    return {**payload, "path": str(analysis_path), "next_actions_path": str(next_path)}


def generate_case_study(
    db_path: Path,
    query: str,
    *,
    doc_ids: Optional[List[str]] = None,
    compare_modes: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    modes = _unique_strings(compare_modes or ["hybrid", "tree"])
    clean_doc_ids = _unique_strings(doc_ids or [])
    profile = classify_query(query, use_llm=False)
    mode_reports = {}
    for mode in modes:
        report = build_search_report(
            db_path,
            query,
            doc_id=clean_doc_ids[0] if len(clean_doc_ids) == 1 else None,
            top_k=top_k,
            search_mode=mode,
        )
        mode_reports[mode] = _case_mode_summary(report)
    facts = fact_search(db_path, query, doc_ids=clean_doc_ids or None, top_k=8)
    conflicts = fact_conflict_summary(db_path, query, doc_ids=clean_doc_ids or None)
    graph = graph_summary(db_path, doc_ids=clean_doc_ids or None, include_conflicts=True)
    created_at = time.time()
    case_id = stable_id("case", query, ",".join(clean_doc_ids), created_at, length=12)
    payload = {
        "schema": CASE_SCHEMA,
        "case_id": case_id,
        "query": compact_whitespace(query),
        "doc_ids": clean_doc_ids,
        "compare_modes": modes,
        "query_profile": profile,
        "mode_reports": mode_reports,
        "fact_matches": _fact_summary(facts),
        "fact_conflicts": conflicts,
        "claim_graph": graph,
        "evidence_summary": _case_evidence_summary(mode_reports),
        "answer_outline": _answer_outline(mode_reports, facts),
        "warnings": _case_warnings(mode_reports, facts, conflicts, graph),
        "created_at": created_at,
    }
    json_path = EVAL_DIR / f"case_study_{case_id}.json"
    md_path = EVAL_DIR / f"case_study_{case_id}.md"
    write_json(json_path, payload)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_case_markdown(payload), encoding="utf-8")
    return {**payload, "path": str(json_path), "md_path": str(md_path)}


def latest_benchmark_reports(limit: int = 5) -> List[Dict[str, Any]]:
    return _latest_reports("benchmark_*.json", BENCHMARK_SCHEMA, limit)


def latest_case_studies(limit: int = 5) -> List[Dict[str, Any]]:
    return _latest_reports("case_study_*.json", CASE_SCHEMA, limit)


def latest_failure_analyses(limit: int = 5) -> List[Dict[str, Any]]:
    return _latest_reports("failure_analysis_*.json", FAILURE_SCHEMA, limit)


def _benchmark_mode(db_path: Path, queries: List[Dict[str, Any]], mode: str, top_k: int) -> Dict[str, Any]:
    items = []
    doc_recall_values: List[float] = []
    node_recall_values: List[float] = []
    precision_values: List[float] = []
    mrr_values: List[float] = []
    table_hits = 0
    table_expected = 0
    fallback_count = 0
    weak_count = 0
    trace_scores: List[float] = []
    for raw in queries:
        query = str(raw.get("query") or "").strip()
        if not query:
            continue
        expected_docs = _string_list(raw.get("expected_doc_ids"))
        expected_nodes = _string_list(raw.get("expected_node_ids"))
        expected_keywords = _string_list(raw.get("expected_keywords") or raw.get("expected_node_keywords"))
        expected_sources = _string_list(raw.get("expected_fact_sources"))
        docs = search_documents(db_path, query, top_k=top_k, search_mode=mode)
        nodes = search_nodes(db_path, query, top_k=top_k, search_mode=mode)
        result_doc_ids = [str(item.get("doc_id") or "") for item in docs if item.get("doc_id")]
        result_node_ids = [result.node_id for result in nodes]
        node_text = " ".join(compact_whitespace(f"{node.title} {node.node_path} {node.heading} {node.snippet}") for node in nodes)
        doc_recall = _recall(result_doc_ids, expected_docs)
        node_recall = _recall(result_node_ids, expected_nodes)
        precision = _precision(result_node_ids, expected_nodes)
        mrr = _mrr(result_doc_ids, expected_docs)
        keyword_hit = all(keyword in node_text for keyword in expected_keywords) if expected_keywords else True
        facts = fact_search(db_path, query, doc_ids=expected_docs or None, top_k=8)
        table_hit = any(item.get("source_kind") == "table" for item in facts.get("items") or [])
        if "table" in expected_sources:
            table_expected += 1
            if table_hit:
                table_hits += 1
        fallback_used = any("fallback" in str(node.rank_reason) for node in nodes)
        if fallback_used:
            fallback_count += 1
        weak_docs = _weak_docs(db_path, _unique_strings([*result_doc_ids, *expected_docs]))
        if weak_docs:
            weak_count += 1
        trace_score = 0.0
        warnings: List[str] = []
        if mode == "tree":
            trace_score, trace_warnings = _tree_trace_completeness(db_path, query, top_k)
            warnings.extend(trace_warnings)
            trace_scores.append(trace_score)
        if not nodes:
            warnings.append("no_evidence_nodes")
        if fallback_used:
            warnings.append("search_fallback")
        if weak_docs:
            warnings.append("weak_parse_quality")
        if "table" in expected_sources and not table_hit:
            warnings.append("missing_table_fact_hit")
        doc_recall_values.append(doc_recall)
        node_recall_values.append(node_recall)
        precision_values.append(precision)
        mrr_values.append(mrr)
        items.append(
            {
                "query": query,
                "intent": raw.get("intent") or "",
                "category": raw.get("category") or "",
                "expected_doc_ids": expected_docs,
                "expected_node_ids": expected_nodes,
                "expected_keywords": expected_keywords,
                "expected_fact_sources": expected_sources,
                "result_doc_ids": result_doc_ids,
                "result_node_ids": result_node_ids,
                "doc_recall_at_k": doc_recall,
                "node_recall_at_k": node_recall,
                "evidence_precision": precision,
                "mrr": mrr,
                "keyword_hit": keyword_hit,
                "table_fact_hit": table_hit,
                "fallback_used": fallback_used,
                "weak_parse_doc_ids": weak_docs,
                "trace_completeness": trace_score,
                "warnings": _unique_strings(warnings),
            }
        )
    query_count = len(items)
    table_rate = table_hits / table_expected if table_expected else 1.0
    fallback_rate = fallback_count / query_count if query_count else 0.0
    weak_rate = weak_count / max(1, query_count)
    trace_completeness = _average(trace_scores) if trace_scores else (1.0 if mode != "tree" else 0.0)
    benchmark_score = round(
        _average(doc_recall_values) * 0.25
        + _average(node_recall_values) * 0.35
        + _average(precision_values) * 0.15
        + _average(mrr_values) * 0.1
        + table_rate * 0.1
        + trace_completeness * 0.05
        - fallback_rate * 0.05
        - weak_rate * 0.03,
        6,
    )
    return {
        "schema": "benchmark_mode.v1",
        "search_mode": mode,
        "top_k": top_k,
        "query_count": query_count,
        "doc_recall_at_k": _average(doc_recall_values),
        "node_recall_at_k": _average(node_recall_values),
        "evidence_precision": _average(precision_values),
        "mrr": _average(mrr_values),
        "table_backed_fact_hit_rate": table_rate,
        "fallback_rate": round(fallback_rate, 4),
        "weak_parse_rate": round(weak_rate, 4),
        "trace_completeness": round(trace_completeness, 4),
        "benchmark_score": benchmark_score,
        "items": items,
    }


def _queries_from_json(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path, [])
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        return [item for item in payload["queries"] if isinstance(item, dict)]
    raise ValueError("input_json must be a JSON list or an object with a queries list")


def _queries_from_feedback(db_path: Path, *, limit: int, min_rating: int) -> List[Dict[str, Any]]:
    feedback = list_feedback(db_path, limit=limit, min_rating=min_rating)
    queries = []
    for item in feedback.get("items") or []:
        queries.append(
            {
                "query": item.get("query") or "",
                "intent": "",
                "category": item.get("operation") or "feedback",
                "expected_doc_ids": item.get("expected_doc_ids") or [],
                "expected_node_ids": item.get("expected_node_ids") or [],
                "expected_keywords": item.get("expected_keywords") or [],
                "expected_fact_sources": [],
                "source_feedback_id": item.get("feedback_id") or "",
            }
        )
    return queries


def _queries_from_query_logs(db_path: Path, *, limit: int) -> List[Dict[str, Any]]:
    logs = list_query_logs(db_path, limit=limit)
    queries = []
    for item in logs.get("items") or []:
        queries.append(
            {
                "query": item.get("query") or "",
                "intent": item.get("intent") or "",
                "category": item.get("operation") or "query_log",
                "expected_doc_ids": item.get("docs_used") or [],
                "expected_node_ids": item.get("nodes_used") or [],
                "expected_keywords": [],
                "expected_fact_sources": [],
                "source_query_id": item.get("query_id") or "",
            }
        )
    return queries


def _queries_from_doc_ids(doc_ids: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "query": f"文档 {doc_id} 的主要研究内容是什么？",
            "intent": "summary",
            "category": "doc_smoke",
            "expected_doc_ids": [doc_id],
            "expected_node_ids": [],
            "expected_keywords": [],
            "expected_fact_sources": [],
        }
        for doc_id in _unique_strings(doc_ids)
    ]


def _normalize_query_item(item: Dict[str, Any]) -> Dict[str, Any]:
    query = compact_whitespace(str(item.get("query") or ""))
    return {
        "query": query,
        "intent": compact_whitespace(str(item.get("intent") or "")),
        "category": compact_whitespace(str(item.get("category") or item.get("operation") or "")),
        "expected_doc_ids": _string_list(item.get("expected_doc_ids")),
        "expected_node_ids": _string_list(item.get("expected_node_ids")),
        "expected_keywords": _string_list(item.get("expected_keywords") or item.get("expected_node_keywords")),
        "expected_fact_sources": _string_list(item.get("expected_fact_sources")),
        "source_feedback_id": compact_whitespace(str(item.get("source_feedback_id") or "")),
        "source_query_id": compact_whitespace(str(item.get("source_query_id") or item.get("query_id") or "")),
    }


def _case_mode_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    trace = report.get("tree_search_trace") if isinstance(report.get("tree_search_trace"), dict) else {}
    results = [item for item in report.get("results") or [] if isinstance(item, dict)]
    return {
        "schema": "case_mode_summary.v1",
        "requested_search_mode": report.get("requested_search_mode") or "",
        "resolved_search_mode": report.get("resolved_search_mode") or "",
        "effective_search_mode": report.get("effective_search_mode") or "",
        "warning_count": len(report.get("warnings") or []),
        "warnings": report.get("warnings") or [],
        "document_count": len(report.get("documents") or []),
        "result_count": len(results),
        "results": [_safe_result(item) for item in results[:8]],
        "fact_matches": _fact_summary(report.get("fact_matches") or {}),
        "tree_trace": _trace_summary(trace),
    }


def _safe_result(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": item.get("doc_id") or "",
        "node_id": item.get("node_id") or "",
        "title": item.get("title") or "",
        "node_path": item.get("node_path") or "",
        "heading": item.get("heading") or "",
        "page_range": [item.get("page_start"), item.get("page_end")],
        "score": item.get("score"),
        "rank_reason": item.get("rank_reason") or "",
    }


def _trace_summary(trace: Dict[str, Any]) -> Dict[str, Any]:
    if not trace:
        return {}
    return {
        "schema": trace.get("schema") or "",
        "query_profile": trace.get("query_profile") or {},
        "expanded_node_count": len(trace.get("expanded_nodes") or []),
        "selected_path_count": len(trace.get("selected_paths") or []),
        "evidence_count": len(trace.get("evidence") or []),
        "fallback_reason": trace.get("fallback_reason") or "",
        "warnings": trace.get("warnings") or [],
    }


def _fact_summary(facts: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    for item in (facts.get("items") or [])[:8]:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "fact_id": item.get("fact_id") or "",
                "fact_type": item.get("fact_type") or "",
                "type": item.get("type") or "",
                "text": compact_whitespace(str(item.get("text") or ""))[:180],
                "doc_id": item.get("doc_id") or "",
                "node_id": item.get("node_id") or "",
                "source_kind": item.get("source_kind") or "",
                "confidence": item.get("confidence"),
            }
        )
    return {
        "schema": "fact_search_summary.v1",
        "available": bool(facts.get("available", True)),
        "count": facts.get("count", len(items)),
        "table_backed_count": sum(1 for item in items if item.get("source_kind") == "table")
        or facts.get("table_backed_count", 0),
        "items": items,
        "warnings": facts.get("warnings") or [],
    }


def _tree_trace_completeness(db_path: Path, query: str, top_k: int) -> tuple[float, List[str]]:
    report = build_search_report(db_path, query, top_k=top_k, search_mode="tree")
    trace = report.get("tree_search_trace") or {}
    checks = [
        bool(trace.get("query_profile")),
        bool(trace.get("expanded_nodes")),
        bool(trace.get("selected_paths")),
        bool(trace.get("evidence")),
    ]
    warnings = [str(item) for item in report.get("warnings") or []]
    return round(sum(1 for item in checks if item) / len(checks), 4), warnings


def _weak_docs(db_path: Path, doc_ids: List[str]) -> List[str]:
    result = []
    for doc_id in doc_ids:
        try:
            quality = get_parse_quality(db_path, doc_id)
        except (KeyError, FileNotFoundError, ValueError):
            continue
        warnings = quality.get("quality_warnings") or []
        if quality.get("quality_level") == "weak" or any(
            warning in warnings for warning in ("page_only_tree", "weak_layout_blocks", "weak_table_parse")
        ):
            result.append(doc_id)
    return _unique_strings(result)


def _failure_reasons(item: Dict[str, Any], mode: str) -> List[str]:
    reasons = []
    if item.get("expected_doc_ids") and float(item.get("doc_recall_at_k") or 0.0) < 1.0:
        reasons.append("doc_routing_miss")
    if item.get("expected_node_ids") and float(item.get("node_recall_at_k") or 0.0) < 1.0:
        reasons.append("node_recall_miss")
    if not item.get("result_node_ids"):
        reasons.append("evidence_coverage_insufficient")
    if item.get("fallback_used"):
        reasons.append("search_fallback")
    if item.get("weak_parse_doc_ids"):
        reasons.append("weak_parse_quality")
    if "table" in (item.get("expected_fact_sources") or []) and not item.get("table_fact_hit"):
        reasons.append("table_facts_missing")
    if mode == "tree" and float(item.get("trace_completeness") or 0.0) < 0.75:
        reasons.append("tree_trace_incomplete")
    if item.get("expected_doc_ids") == [] and item.get("expected_node_ids") == [] and item.get("expected_keywords") == []:
        reasons.append("low_signal_eval_case")
    return _unique_strings(reasons)


def _next_actions(reason_counts: Dict[str, int]) -> List[str]:
    mapping = {
        "doc_routing_miss": "补充 doc_card/abstract/keywords 或刷新 embedding 后重跑 benchmark。",
        "node_recall_miss": "检查章节树和 tree value function，必要时补充 expected_node_keywords。",
        "evidence_coverage_insufficient": "检查 query intent、search mode 和解析质量，确认是否需要重新同步论文。",
        "search_fallback": "刷新 semantic embedding，或显式比较 fts/hybrid/tree 的差异。",
        "weak_parse_quality": "对弱解析 PDF 使用 docling 重建工件，或人工确认 page-only tree 风险。",
        "table_facts_missing": "重新运行 extract-facts，检查 table_content/table_summaries 是否包含实验指标。",
        "tree_trace_incomplete": "复核 tree-search trace、节点 summary 和 value function 权重。",
        "low_signal_eval_case": "为评测样例补充 expected_doc_ids、expected_node_ids 或 expected_keywords。",
    }
    return [mapping[key] for key, _ in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])) if key in mapping]


def _case_evidence_summary(mode_reports: Dict[str, Any]) -> Dict[str, Any]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for mode, report in mode_reports.items():
        for item in report.get("results") or []:
            key = f"{item.get('doc_id')}:{item.get('node_id')}"
            if not item.get("node_id") or key in by_key:
                continue
            by_key[key] = {**item, "modes": [mode]}
        for item in report.get("results") or []:
            key = f"{item.get('doc_id')}:{item.get('node_id')}"
            if key in by_key and mode not in by_key[key]["modes"]:
                by_key[key]["modes"].append(mode)
    return {"schema": "case_evidence_summary.v1", "count": len(by_key), "items": list(by_key.values())[:12]}


def _answer_outline(mode_reports: Dict[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
    evidence_count = sum(len(report.get("results") or []) for report in mode_reports.values())
    fact_items = facts.get("items") or []
    return {
        "schema": "answer_outline.v1",
        "status": "ready" if evidence_count else "needs_more_evidence",
        "claim": f"当前可基于 {evidence_count} 个检索节点和 {len(fact_items)} 条事实候选组织回答。",
        "recommended_next_step": "人工查看 evidence packet 后再生成正式回答。" if evidence_count else "补充检索或重新解析文档。",
    }


def _case_warnings(
    mode_reports: Dict[str, Any],
    facts: Dict[str, Any],
    conflicts: Optional[Dict[str, Any]] = None,
    graph: Optional[Dict[str, Any]] = None,
) -> List[str]:
    warnings = []
    for report in mode_reports.values():
        warnings.extend(report.get("warnings") or [])
    warnings.extend(facts.get("warnings") or [])
    warnings.extend((conflicts or {}).get("warnings") or [])
    warnings.extend((graph or {}).get("warnings") or [])
    if (conflicts or {}).get("conflict_count", 0) > 0:
        warnings.append(f"fact_conflicts:{(conflicts or {}).get('conflict_count')}")
    if (graph or {}).get("conflict_count", 0) > 0:
        warnings.append(f"claim_graph_conflicts:{(graph or {}).get('conflict_count')}")
    if (graph or {}).get("isolated_fact_count", 0) > 0:
        warnings.append(f"claim_graph_isolated_facts:{(graph or {}).get('isolated_fact_count')}")
    if not any((report.get("results") or []) for report in mode_reports.values()):
        warnings.append("case_has_no_evidence")
    return _unique_strings(warnings)


def _load_benchmark(benchmark_id: str) -> tuple[Dict[str, Any], Path]:
    raw = str(benchmark_id)
    path = Path(raw).expanduser()
    candidates = []
    if path.exists():
        candidates.append(path)
    candidates.extend(EVAL_DIR.glob(f"benchmark_{raw}.json"))
    candidates.extend(EVAL_DIR.glob("benchmark_*.json"))
    for candidate in candidates:
        payload = _read_json(candidate, {})
        if payload.get("schema") == BENCHMARK_SCHEMA and (
            payload.get("benchmark_id") == raw or candidate == path or raw in candidate.stem
        ):
            return payload, candidate
    raise FileNotFoundError(f"Benchmark report not found: {benchmark_id}")


def _latest_reports(pattern: str, schema: str, limit: int) -> List[Dict[str, Any]]:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(EVAL_DIR.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = _read_json(path, {})
        if payload.get("schema") != schema:
            continue
        items.append(
            {
                "path": str(path),
                "schema": payload.get("schema") or "",
                "status": payload.get("status") or "",
                "benchmark_id": payload.get("benchmark_id") or "",
                "case_id": payload.get("case_id") or "",
                "suite_name": payload.get("suite_name") or "",
                "query_count": payload.get("query_count", 0),
                "failure_count": payload.get("failure_count", 0),
                "best_mode_by_score": payload.get("best_mode_by_score") or "",
                "warnings": payload.get("warnings") or [],
                "created_at": payload.get("created_at"),
            }
        )
        if len(items) >= limit:
            break
    return items


def _benchmark_summary(mode_results: Dict[str, Any]) -> Dict[str, Any]:
    return {
        mode: {
            "benchmark_score": result.get("benchmark_score", 0.0),
            "doc_recall_at_k": result.get("doc_recall_at_k", 0.0),
            "node_recall_at_k": result.get("node_recall_at_k", 0.0),
            "table_backed_fact_hit_rate": result.get("table_backed_fact_hit_rate", 0.0),
            "trace_completeness": result.get("trace_completeness", 0.0),
        }
        for mode, result in mode_results.items()
    }


def _benchmark_warnings(mode_results: Dict[str, Any]) -> List[str]:
    warnings = []
    for mode, result in mode_results.items():
        if not result.get("query_count"):
            warnings.append(f"empty_mode_result:{mode}")
        if float(result.get("fallback_rate") or 0.0) > 0:
            warnings.append(f"fallback_used:{mode}")
        if float(result.get("weak_parse_rate") or 0.0) > 0:
            warnings.append(f"weak_parse_quality:{mode}")
    return _unique_strings(warnings)


def _benchmark_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Report",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- benchmark_id: `{report.get('benchmark_id')}`",
        f"- suite: `{report.get('suite_name')}`",
        f"- query_count: `{report.get('query_count')}`",
        f"- best_mode_by_score: `{report.get('best_mode_by_score')}`",
        "",
        "## Modes",
    ]
    for mode, result in (report.get("mode_results") or {}).items():
        lines.append(
            f"- `{mode}` score={result.get('benchmark_score')} doc_recall={result.get('doc_recall_at_k')} "
            f"node_recall={result.get('node_recall_at_k')} table_hit={result.get('table_backed_fact_hit_rate')}"
        )
    return "\n".join(lines) + "\n"


def _case_markdown(case: Dict[str, Any]) -> str:
    lines = [
        "# Case Study",
        "",
        f"- schema: `{case.get('schema')}`",
        f"- case_id: `{case.get('case_id')}`",
        f"- query: {case.get('query')}",
        f"- modes: `{', '.join(case.get('compare_modes') or [])}`",
        f"- evidence_count: `{(case.get('evidence_summary') or {}).get('count', 0)}`",
        f"- fact_conflict_count: `{(case.get('fact_conflicts') or {}).get('conflict_count', 0)}`",
        f"- claim_graph_id: `{(case.get('claim_graph') or {}).get('graph_id', '')}`",
        f"- claim_graph_conflict_count: `{(case.get('claim_graph') or {}).get('conflict_count', 0)}`",
        f"- claim_graph_isolated_fact_count: `{(case.get('claim_graph') or {}).get('isolated_fact_count', 0)}`",
        f"- warnings: `{', '.join(case.get('warnings') or [])}`",
    ]
    return "\n".join(lines) + "\n"


def _suite_path(name: str) -> Path:
    return SUITE_DIR / f"{_safe_name(name)}.json"


def _safe_name(value: str) -> str:
    text = compact_whitespace(value)
    if not text:
        raise ValueError("suite name is required")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or stable_id("suite", text, length=8)


def _dedupe_queries(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for item in items:
        query = str(item.get("query") or "")
        if not query or query in seen:
            continue
        seen.add(query)
        result.append(item)
    return result[:500]


def _recall(results: List[str], expected: List[str]) -> float:
    if not expected:
        return 1.0
    return len(set(results) & set(expected)) / len(set(expected))


def _precision(results: List[str], expected: List[str]) -> float:
    if not expected:
        return 1.0
    if not results:
        return 0.0
    return len(set(results) & set(expected)) / len(set(results))


def _mrr(results: List[str], expected: List[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    for index, value in enumerate(results, start=1):
        if value in expected_set:
            return 1.0 / index
    return 0.0


def _average(values: List[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _best_mode(mode_results: Dict[str, Any], metric: str) -> str:
    if not mode_results:
        return ""
    return max(mode_results.items(), key=lambda item: float(item[1].get(metric) or 0.0))[0]


def _string_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return _unique_strings(str(item) for item in value)
    return _unique_strings(str(value).split(","))
