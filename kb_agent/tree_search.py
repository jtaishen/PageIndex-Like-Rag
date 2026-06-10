from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import db
from .llm import LLMError, generate_json_object
from .models import EvidencePacket, SearchResult
from .query import classify_query
from .query_log import insert_query_log
from .utils import compact_whitespace, first_words


FLAT_SEARCH_MODES = {"hybrid", "fts"}
TREE_MEMORY_TYPES = {"preference", "project_rule", "setting", "default"}


def tree_search(
    db_path: Path,
    doc_id: str,
    query: str,
    *,
    budget: int = 8,
    use_llm: bool = True,
    require_llm: bool = False,
    search_mode: str = "hybrid",
) -> Dict[str, Any]:
    started = time.time()
    flat_mode = _resolve_flat_search_mode(search_mode)
    conn = db.connect(db_path)
    db.init_db(conn)
    llm_error = ""
    fallback_reason = ""
    try:
        doc = _get_document(conn, doc_id)
        rows = _node_rows(conn, doc_id)
        if not rows:
            trace = _empty_trace(doc_id, query, budget, "no_doc_nodes")
            _log_query(conn, trace, started)
            conn.commit()
            return trace

        profile = classify_query(query, use_llm=use_llm, require_llm=require_llm)
        quality = _parse_quality(conn, doc_id)
        preferences = _memory_preferences(conn)
        flat_signals = _flat_candidate_signals(db_path, doc_id, query, budget, flat_mode)
        scored = _score_nodes(rows, query, profile, flat_signals, quality, preferences)
        children = _children_by_parent(rows)
        subtree_scores = _subtree_scores(rows, children, scored)
        selected_ids = _deterministic_selection(rows, scored, budget)
        llm_decisions: Dict[str, Any] = {}
        llm_selected_count = 0
        llm_warning_count = 0

        if use_llm:
            try:
                llm_decisions = _llm_select_nodes(query, profile, rows, scored, selected_ids, preferences, budget)
                llm_ids = _valid_llm_node_ids(llm_decisions, rows)
                llm_selected_count = len(llm_ids)
                raw_llm_warnings = llm_decisions.get("warnings") or []
                if isinstance(raw_llm_warnings, str):
                    raw_llm_warnings = [raw_llm_warnings]
                llm_warning_count = len(_unique_strings(raw_llm_warnings))
                if llm_ids:
                    selected_ids = _merge_selected_ids(llm_ids, selected_ids, budget)
            except LLMError as exc:
                if require_llm:
                    raise
                llm_error = str(exc)
                fallback_reason = f"llm_unavailable:{llm_error}"
                profile["warnings"] = _unique_strings([*profile.get("warnings", []), fallback_reason])
        else:
            profile["warnings"] = _unique_strings([*profile.get("warnings", []), "llm_disabled"])

        selected_ids = _expand_sections_to_evidence(rows, children, scored, selected_ids, budget)
        expanded_nodes, selected_paths = _build_trace_paths(rows, children, scored, subtree_scores, selected_ids)
        results = _search_results(doc, rows, scored, flat_signals, selected_ids)
        evidence = _evidence_for_selected(conn, doc_id, selected_ids, scored)
        warnings = _trace_warnings(rows, quality, evidence, fallback_reason)
        trace = {
            "schema": "tree_search_trace.v1",
            "doc_id": doc_id,
            "query": query,
            "query_profile": profile,
            "requested_search_mode": search_mode,
            "effective_search_mode": flat_mode,
            "budget": budget,
            "use_llm": use_llm,
            "llm_used": bool(use_llm and not llm_error and (llm_decisions or profile.get("source") == "llm")),
            "llm_selected_count": llm_selected_count,
            "llm_warning_count": llm_warning_count,
            "llm_error": llm_error,
            "fallback_reason": fallback_reason,
            "memory_preferences": _preference_refs(preferences),
            "expanded_nodes": expanded_nodes,
            "selected_paths": selected_paths,
            "llm_decisions": llm_decisions,
            "warnings": _unique_strings([*profile.get("warnings", []), *warnings]),
            "results": [result.__dict__ for result in results],
            "evidence": evidence,
            "latency_ms": round((time.time() - started) * 1000, 3),
        }
        _log_query(conn, trace, started)
        conn.commit()
        return trace
    finally:
        conn.close()


def tree_search_for_query(
    db_path: Path,
    query: str,
    *,
    doc_id: Optional[str] = None,
    top_k: int = 8,
    use_llm: bool = False,
    require_llm: bool = False,
    search_mode: str = "hybrid",
) -> Dict[str, Any]:
    if doc_id:
        return tree_search(
            db_path,
            doc_id,
            query,
            budget=top_k,
            use_llm=use_llm,
            require_llm=require_llm,
            search_mode=search_mode,
        )

    from .search import search_documents

    route_mode = _resolve_flat_search_mode(search_mode)
    doc_limit = max(1, min(4, top_k))
    docs = search_documents(db_path, query, top_k=doc_limit, search_mode=route_mode)
    traces = []
    evidence: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    per_doc_budget = max(2, math.ceil(top_k / max(1, len(docs))))
    for item in docs:
        trace = tree_search(
            db_path,
            str(item["doc_id"]),
            query,
            budget=per_doc_budget,
            use_llm=use_llm,
            require_llm=require_llm,
            search_mode=route_mode,
        )
        traces.append(trace)
        evidence.extend(trace.get("evidence") or [])
        results.extend(trace.get("results") or [])
    results.sort(key=lambda item: -float(item.get("score") or 0.0))
    evidence_by_node = {str(item.get("node_id")): item for item in evidence}
    ordered_evidence = [evidence_by_node[str(item.get("node_id"))] for item in results if str(item.get("node_id")) in evidence_by_node]
    warnings = _unique_strings(
        warning
        for trace in traces
        for warning in trace.get("warnings", [])
    )
    if not docs:
        warnings.append("no_routed_documents")
    return {
        "schema": "tree_search_multi_trace.v1",
        "query": query,
        "doc_id": None,
        "requested_search_mode": search_mode,
        "effective_search_mode": route_mode,
        "top_k": top_k,
        "routed_documents": docs,
        "traces": traces,
        "warnings": warnings,
        "results": results[:top_k],
        "evidence": ordered_evidence[:top_k],
    }


def tree_search_results(
    db_path: Path,
    query: str,
    *,
    doc_id: Optional[str] = None,
    top_k: int = 8,
    search_mode: str = "hybrid",
) -> List[SearchResult]:
    trace = tree_search_for_query(
        db_path,
        query,
        doc_id=doc_id,
        top_k=top_k,
        use_llm=False,
        search_mode=search_mode,
    )
    return [
        SearchResult(**item)
        for item in trace.get("results", [])[:top_k]
    ]


def _get_document(conn: Any, doc_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM documents WHERE doc_id = ? AND status = 'ready'",
        (doc_id,),
    ).fetchone()
    if not row:
        raise KeyError(f"Document not found or not ready: {doc_id}")
    return dict(row)


def _node_rows(conn: Any, doc_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT n.*, d.title, d.path
        FROM doc_nodes n
        JOIN documents d ON d.doc_id = n.doc_id
        WHERE n.doc_id = ?
        ORDER BY n.order_index ASC
        """,
        (doc_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _flat_candidate_signals(
    db_path: Path,
    doc_id: str,
    query: str,
    budget: int,
    search_mode: str,
) -> Dict[str, Dict[str, Any]]:
    from .search import search_nodes

    try:
        flat = search_nodes(
            db_path,
            query,
            doc_id=doc_id,
            top_k=max(budget * 5, 20),
            search_mode=_resolve_flat_search_mode(search_mode),
        )
    except Exception as exc:
        return {"__error__": {"warning": f"flat_search_failed:{exc}"}}
    signals: Dict[str, Dict[str, Any]] = {}
    for rank, result in enumerate(flat, start=1):
        signals[result.node_id] = {
            "rank": rank,
            "rank_score": round(0.25 / math.sqrt(rank), 8),
            "fts_score": result.fts_score,
            "vector_score": result.vector_score,
            "hybrid_score": result.hybrid_score,
            "rank_reason": result.rank_reason,
        }
    return signals


def _score_nodes(
    rows: List[Dict[str, Any]],
    query: str,
    profile: Dict[str, Any],
    flat_signals: Dict[str, Dict[str, Any]],
    quality: Dict[str, Any],
    preferences: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    scored: Dict[str, Dict[str, Any]] = {}
    terms = [str(term) for term in profile.get("focus_terms", [])]
    preferred_types = set(str(item) for item in profile.get("preferred_node_types", []))
    target_sections = [str(item) for item in profile.get("target_sections", [])]
    for row in rows:
        node_id = str(row["node_id"])
        signal = flat_signals.get(node_id, {})
        node_type = str(row.get("type") or "")
        text = _node_search_text(row)
        heading_path = compact_whitespace(f"{row.get('heading') or ''} {row.get('node_path') or ''}")
        term_hits = _term_hits(terms, text)
        heading_hits = _term_hits(terms, heading_path)
        section_hits = _term_hits(target_sections, heading_path)
        components = {
            "search_signal": float(signal.get("rank_score") or 0.0),
            "vector_signal": min(0.12, max(0.0, float(signal.get("vector_score") or 0.0)) * 0.12),
            "term_match": min(0.28, term_hits * 0.035 + heading_hits * 0.025),
            "intent_type": 0.12 if node_type in preferred_types else 0.0,
            "section_match": min(0.14, section_hits * 0.045),
            "leaf_bonus": 0.06 if row.get("text") and node_type not in {"document", "page"} else 0.0,
            "memory_preference": _memory_boost(preferences, text),
            "quality_penalty": _quality_penalty(quality),
            "noise_penalty": _noise_penalty(row, str(profile.get("intent") or ""), query),
        }
        score = round(sum(components.values()), 8)
        scored[node_id] = {
            "score": score,
            "components": components,
            "term_hits": term_hits,
            "section_hits": section_hits,
            "rank_reason": _rank_reason(components, signal),
        }
    return scored


def _deterministic_selection(rows: List[Dict[str, Any]], scored: Dict[str, Dict[str, Any]], budget: int) -> List[str]:
    candidates = [
        row for row in rows
        if row.get("type") != "document" and _node_search_text(row)
    ]
    candidates.sort(
        key=lambda row: (
            -float(scored.get(str(row["node_id"]), {}).get("score") or 0.0),
            0 if row.get("text") else 1,
            int(row.get("order_index") or 0),
        )
    )
    selected = []
    seen_parent = set()
    for row in candidates:
        node_id = str(row["node_id"])
        if not row.get("text") and len(selected) >= max(1, budget // 2):
            continue
        parent_id = str(row.get("parent_id") or "")
        if parent_id in seen_parent and len(selected) < max(2, budget // 2):
            continue
        selected.append(node_id)
        if parent_id:
            seen_parent.add(parent_id)
        if len(selected) >= budget:
            break
    return selected


def _llm_select_nodes(
    query: str,
    profile: Dict[str, Any],
    rows: List[Dict[str, Any]],
    scored: Dict[str, Dict[str, Any]],
    selected_ids: List[str],
    preferences: List[Dict[str, Any]],
    budget: int,
) -> Dict[str, Any]:
    by_id = {str(row["node_id"]): row for row in rows}
    candidate_ids = _merge_selected_ids(selected_ids, _top_node_ids(rows, scored, limit=max(budget * 2, 12)), max(budget * 3, 18))
    candidates = [
        {
            "node_id": node_id,
            "type": by_id[node_id].get("type"),
            "node_path": by_id[node_id].get("node_path"),
            "score": scored[node_id]["score"],
            "components": scored[node_id]["components"],
            "summary": first_words(_node_search_text(by_id[node_id]), 70),
        }
        for node_id in candidate_ids
        if node_id in by_id
    ]
    system_prompt = (
        "你是 PageIndex-like 论文树检索路由器。只能根据候选节点、查询意图和评分选择证据路径，"
        "不要编造 node_id。必须返回 JSON object，不要返回 Markdown。"
    )
    user_prompt = "\n".join(
        [
            f"查询：{query}",
            f"query_profile: {json.dumps(profile, ensure_ascii=False)}",
            f"memory_preferences: {json.dumps(_preference_refs(preferences), ensure_ascii=False)}",
            f"budget: {budget}",
            "候选节点：",
            json.dumps(candidates, ensure_ascii=False),
            "请返回：",
            '{"selected_node_ids":[],"rationale":[],"warnings":[]}',
        ]
    )
    payload = generate_json_object(system_prompt, user_prompt)
    payload.setdefault("candidate_count", len(candidates))
    return payload


def _valid_llm_node_ids(payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    valid = {str(row["node_id"]) for row in rows}
    raw = payload.get("selected_node_ids") or payload.get("selected") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return _unique_strings(str(item) for item in raw if str(item) in valid)


def _expand_sections_to_evidence(
    rows: List[Dict[str, Any]],
    children: Dict[str, List[Dict[str, Any]]],
    scored: Dict[str, Dict[str, Any]],
    selected_ids: List[str],
    budget: int,
) -> List[str]:
    by_id = {str(row["node_id"]): row for row in rows}
    expanded = []
    for node_id in selected_ids:
        row = by_id.get(node_id)
        if not row:
            continue
        if row.get("text") or not children.get(node_id):
            expanded.append(node_id)
            continue
        descendants = _descendants(node_id, children)
        descendants.sort(key=lambda item: -float(scored.get(str(item["node_id"]), {}).get("score") or 0.0))
        leaf = next((item for item in descendants if item.get("text")), None)
        expanded.append(str((leaf or row)["node_id"]))
    return _unique_strings(expanded)[:budget]


def _build_trace_paths(
    rows: List[Dict[str, Any]],
    children: Dict[str, List[Dict[str, Any]]],
    scored: Dict[str, Dict[str, Any]],
    subtree_scores: Dict[str, float],
    selected_ids: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_id = {str(row["node_id"]): row for row in rows}
    expanded_nodes = []
    selected_paths = []
    seen_expanded = set()
    for node_id in selected_ids:
        path = _ancestor_path(by_id, node_id)
        selected_paths.append(
            {
                "node_id": node_id,
                "node_path": by_id[node_id].get("node_path") if node_id in by_id else "",
                "score": scored.get(node_id, {}).get("score", 0.0),
                "score_components": scored.get(node_id, {}).get("components", {}),
                "path_node_ids": path,
            }
        )
        for index, parent_id in enumerate(path[:-1]):
            if parent_id in seen_expanded:
                continue
            seen_expanded.add(parent_id)
            chosen_child_id = path[index + 1]
            candidates = []
            for child in sorted(children.get(parent_id, []), key=lambda row: -subtree_scores.get(str(row["node_id"]), 0.0))[:6]:
                child_id = str(child["node_id"])
                candidates.append(
                    {
                        "node_id": child_id,
                        "type": child.get("type"),
                        "heading": child.get("heading"),
                        "node_path": child.get("node_path"),
                        "subtree_score": subtree_scores.get(child_id, 0.0),
                        "score": scored.get(child_id, {}).get("score", 0.0),
                    }
                )
            expanded_nodes.append(
                {
                    "node_id": parent_id,
                    "node_path": by_id[parent_id].get("node_path") if parent_id in by_id else "",
                    "chosen_child_id": chosen_child_id,
                    "candidate_children": candidates,
                }
            )
    return expanded_nodes, selected_paths


def _search_results(
    doc: Dict[str, Any],
    rows: List[Dict[str, Any]],
    scored: Dict[str, Dict[str, Any]],
    flat_signals: Dict[str, Dict[str, Any]],
    selected_ids: List[str],
) -> List[SearchResult]:
    by_id = {str(row["node_id"]): row for row in rows}
    results = []
    for node_id in selected_ids:
        row = by_id.get(node_id)
        if not row:
            continue
        signal = flat_signals.get(node_id, {})
        score = float(scored.get(node_id, {}).get("score") or 0.0)
        results.append(
            SearchResult(
                doc_id=str(doc["doc_id"]),
                node_id=node_id,
                title=str(doc["title"]),
                path=str(doc["path"]),
                node_path=str(row.get("node_path") or ""),
                heading=str(row.get("heading") or ""),
                snippet=first_words(_node_search_text(row), 70),
                score=score,
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                fts_score=_optional_float(signal.get("fts_score")),
                vector_score=_optional_float(signal.get("vector_score")),
                hybrid_score=score,
                rank_reason=str(scored.get(node_id, {}).get("rank_reason") or "tree:value"),
            )
        )
    return results


def _evidence_for_selected(
    conn: Any,
    doc_id: str,
    selected_ids: List[str],
    scored: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    packets = db.get_evidence_packets(conn, doc_id, selected_ids)
    by_id: Dict[str, EvidencePacket] = {packet.node_id: packet for packet in packets}
    evidence = []
    for node_id in selected_ids:
        packet = by_id.get(node_id)
        if not packet:
            continue
        item = packet.to_dict()
        score = float(scored.get(node_id, {}).get("score") or 0.0)
        item["confidence"] = round(max(0.35, min(0.95, 0.55 + score * 0.7)), 3)
        item["tree_score"] = score
        item["score_components"] = scored.get(node_id, {}).get("components", {})
        item["rank_reason"] = scored.get(node_id, {}).get("rank_reason", "tree:value")
        evidence.append(item)
    return evidence


def _parse_quality(conn: Any, doc_id: str) -> Dict[str, Any]:
    card = db.get_doc_card(conn, doc_id) or {}
    quality = card.get("parse_quality") if isinstance(card, dict) else {}
    return quality if isinstance(quality, dict) else {}


def _memory_preferences(conn: Any, limit: int = 8) -> List[Dict[str, Any]]:
    placeholders = ",".join("?" for _ in TREE_MEMORY_TYPES)
    rows = conn.execute(
        f"""
        SELECT memory_id, scope, type, subject_key, content, refs, importance, confidence, updated_at
        FROM memory_items
        WHERE type IN ({placeholders})
          AND (ttl IS NULL OR ttl > ?)
        ORDER BY importance DESC, updated_at DESC
        LIMIT ?
        """,
        [*sorted(TREE_MEMORY_TYPES), time.time(), limit],
    ).fetchall()
    return [dict(row) for row in rows]


def _preference_refs(preferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs = []
    for item in preferences:
        content = compact_whitespace(str(item.get("content") or ""))
        refs.append(
            {
                "memory_id": item.get("memory_id"),
                "scope": item.get("scope"),
                "type": item.get("type"),
                "subject_key": item.get("subject_key"),
                "summary": first_words(content, 24),
                "importance": item.get("importance"),
                "confidence": item.get("confidence"),
            }
        )
    return refs


def _children_by_parent(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    children: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        parent_id = row.get("parent_id")
        if parent_id:
            children.setdefault(str(parent_id), []).append(row)
    return children


def _subtree_scores(
    rows: List[Dict[str, Any]],
    children: Dict[str, List[Dict[str, Any]]],
    scored: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    by_id = {str(row["node_id"]): row for row in rows}
    cache: Dict[str, float] = {}

    def score(node_id: str) -> float:
        if node_id in cache:
            return cache[node_id]
        own = float(scored.get(node_id, {}).get("score") or 0.0)
        child_scores = [score(str(child["node_id"])) for child in children.get(node_id, [])]
        cache[node_id] = max([own, *child_scores] or [own])
        return cache[node_id]

    for node_id in by_id:
        score(node_id)
    return cache


def _descendants(node_id: str, children: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    result = []
    stack = list(children.get(node_id, []))
    while stack:
        row = stack.pop(0)
        result.append(row)
        stack.extend(children.get(str(row["node_id"]), []))
    return result


def _ancestor_path(by_id: Dict[str, Dict[str, Any]], node_id: str) -> List[str]:
    path = []
    current = node_id
    while current and current in by_id:
        path.append(current)
        parent = by_id[current].get("parent_id")
        current = str(parent) if parent else ""
    return list(reversed(path))


def _node_search_text(row: Dict[str, Any]) -> str:
    return compact_whitespace(
        " ".join(
            str(row.get(name) or "")
            for name in ("heading", "node_path", "summary", "text")
        )
    )


def _term_hits(terms: Iterable[str], text: str) -> int:
    compacted = text.lower().replace(" ", "")
    hits = 0
    for term in terms:
        cleaned = str(term).lower().replace(" ", "")
        if cleaned and cleaned in compacted:
            hits += 1
    return hits


def _memory_boost(preferences: List[Dict[str, Any]], text: str) -> float:
    if not preferences:
        return 0.0
    boost = 0.0
    compacted = text.lower()
    for item in preferences:
        content = str(item.get("content") or "").lower()
        terms = [token for token in ("实验", "局限", "方法", "引用", "页码", "结论", "安全", "记忆") if token in content]
        if any(term in compacted for term in terms):
            boost += 0.025
    return min(0.08, boost)


def _quality_penalty(quality: Dict[str, Any]) -> float:
    penalty = 0.0
    if quality.get("quality_level") == "weak":
        penalty -= 0.05
    if quality.get("page_only_tree"):
        penalty -= 0.04
    if quality.get("missing_abstract"):
        penalty -= 0.015
    return penalty


def _noise_penalty(row: Dict[str, Any], intent: str, query: str) -> float:
    node_type = str(row.get("type") or "")
    text = _node_search_text(row)
    penalty = 0.0
    if node_type == "document":
        penalty -= 0.2
    if node_type == "page":
        penalty -= 0.08
    if node_type == "reference" and intent != "citation" and not any(token in query for token in ("引用", "参考", "文献")):
        penalty -= 0.08
    compacted = text.replace(" ", "")
    if any(token in compacted for token in ("目录", "学号", "指导教师", "分类号", "密级", "网络首发", "引用格式")):
        penalty -= 0.08
    return penalty


def _rank_reason(components: Dict[str, float], signal: Dict[str, Any]) -> str:
    reasons = ["tree:value"]
    if signal.get("rank_reason"):
        reasons.append(str(signal["rank_reason"]))
    for key, value in components.items():
        if value > 0.0001:
            reasons.append(f"{key}+")
        elif value < -0.0001:
            reasons.append(f"{key}-")
    return ",".join(_unique_strings(reasons))


def _top_node_ids(rows: List[Dict[str, Any]], scored: Dict[str, Dict[str, Any]], limit: int) -> List[str]:
    sorted_rows = sorted(rows, key=lambda row: -float(scored.get(str(row["node_id"]), {}).get("score") or 0.0))
    return [str(row["node_id"]) for row in sorted_rows if row.get("type") != "document"][:limit]


def _merge_selected_ids(primary: Iterable[str], fallback: Iterable[str], limit: int) -> List[str]:
    return _unique_strings([*primary, *fallback])[:limit]


def _trace_warnings(
    rows: List[Dict[str, Any]],
    quality: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    fallback_reason: str,
) -> List[str]:
    warnings = []
    if quality.get("page_only_tree"):
        warnings.append("page_only_tree")
    if quality.get("quality_level") == "weak":
        warnings.append("weak_parse_quality")
    if not any(row.get("type") in {"section", "abstract", "keywords"} for row in rows):
        warnings.append("missing_structured_sections")
    if not evidence:
        warnings.append("no_tree_evidence")
    if fallback_reason:
        warnings.append(fallback_reason)
    return warnings


def _empty_trace(doc_id: str, query: str, budget: int, warning: str) -> Dict[str, Any]:
    return {
        "schema": "tree_search_trace.v1",
        "doc_id": doc_id,
        "query": query,
        "query_profile": classify_query(query, use_llm=False),
        "requested_search_mode": "hybrid",
        "effective_search_mode": "hybrid",
        "budget": budget,
        "use_llm": False,
        "llm_error": "",
        "fallback_reason": warning,
        "memory_preferences": [],
        "expanded_nodes": [],
        "selected_paths": [],
        "llm_decisions": {},
        "warnings": [warning],
        "results": [],
        "evidence": [],
        "latency_ms": 0.0,
    }


def _log_query(conn: Any, trace: Dict[str, Any], started: float) -> None:
    profile = trace.get("query_profile") or {}
    docs_used = _unique_strings(
        str(item.get("doc_id") or trace.get("doc_id") or "")
        for item in trace.get("results", [])
    )
    nodes_used = _unique_strings(
        str(item.get("node_id") or "")
        for item in trace.get("results", [])
    )
    insert_query_log(
        conn,
        operation="tree-search",
        query=str(trace.get("query") or ""),
        intent=str(profile.get("intent") or "unknown"),
        search_mode=str(trace.get("requested_search_mode") or trace.get("effective_search_mode") or ""),
        status="ok" if trace.get("evidence") else "empty",
        docs_used=docs_used,
        nodes_used=nodes_used,
        latency_ms=round((time.time() - started) * 1000, 3),
        warnings=trace.get("warnings") or [],
        metrics={
            "schema": trace.get("schema"),
            "budget": trace.get("budget") or trace.get("top_k"),
            "effective_search_mode": trace.get("effective_search_mode"),
            "result_count": len(trace.get("results") or []),
            "evidence_count": len(trace.get("evidence") or []),
            "fallback_used": bool(trace.get("fallback_reason")),
        },
    )


def _resolve_flat_search_mode(search_mode: str) -> str:
    mode = (search_mode or "hybrid").strip().lower()
    if mode == "tree":
        return "hybrid"
    if mode not in FLAT_SEARCH_MODES:
        choices = ", ".join(sorted([*FLAT_SEARCH_MODES, "tree"]))
        raise ValueError(f"Unsupported search_mode '{search_mode}'. Expected one of: {choices}")
    return mode


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique_strings(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
