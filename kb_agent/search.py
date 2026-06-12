from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import db
from .answer_plan import answer_plan_counts, build_answer_plan
from .embeddings import (
    EmbeddingError,
    cosine_similarity,
    get_embedding_provider,
    semantic_index_status,
    vector_from_json,
)
from .models import EvidencePacket, SearchResult
from .query import classify_query
from .utils import compact_whitespace, unique_strings as _unique_strings


SEARCH_MODES = {"hybrid", "fts", "tree", "auto"}


def resolve_search_mode(search_mode: str = "hybrid") -> str:
    mode = (search_mode or "hybrid").strip().lower()
    if mode not in SEARCH_MODES:
        choices = ", ".join(sorted(SEARCH_MODES))
        raise ValueError(f"Unsupported search_mode '{search_mode}'. Expected one of: {choices}")
    return mode


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
    search_mode: str = "hybrid",
) -> List[SearchResult]:
    mode = resolve_search_mode(search_mode)
    auto_resolution = None
    if mode == "auto":
        auto_resolution = _auto_resolution(db_path, query)
        mode = str(auto_resolution.get("resolved_search_mode") or "hybrid")
    if mode == "tree":
        from .tree_search import tree_search_results

        return _tag_auto_results(
            tree_search_results(db_path, query, doc_id=doc_id, top_k=top_k, search_mode="hybrid"),
            auto_resolution,
        )
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        return _tag_auto_results(
            _search_nodes_conn(conn, query, doc_id=doc_id, top_k=top_k, search_mode=mode),
            auto_resolution,
        )
    finally:
        conn.close()


def search_documents(
    db_path: Path,
    query: str,
    top_k: int = 8,
    search_mode: str = "hybrid",
) -> List[Dict[str, object]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        mode = resolve_search_mode(search_mode)
        auto_resolution = None
        if mode == "auto":
            auto_resolution = _auto_resolution(db_path, query)
            mode = str(auto_resolution.get("resolved_search_mode") or "hybrid")
        if mode == "tree":
            mode = "hybrid"
        if mode == "hybrid":
            rows = _search_documents_hybrid_conn(conn, query, top_k)
            if rows:
                return _tag_auto_docs(
                    _augment_documents_with_routing(conn, rows, query, top_k=top_k),
                    auto_resolution,
                )
        docs = [dict(row) for row in _search_documents_fts_conn(conn, query, top_k)]
        return _tag_auto_docs(_augment_documents_with_routing(conn, docs, query, top_k=top_k), auto_resolution)
    finally:
        conn.close()


def build_search_report(
    db_path: Path,
    query: str,
    doc_id: Optional[str] = None,
    top_k: int = 8,
    search_mode: str = "hybrid",
) -> Dict[str, object]:
    mode = resolve_search_mode(search_mode)
    auto_resolution = None
    requested_mode = mode
    if mode == "auto":
        auto_resolution = _auto_resolution(db_path, query)
        mode = str(auto_resolution.get("resolved_search_mode") or "hybrid")
    if mode == "tree":
        from .tree_search import tree_search_for_query

        trace = tree_search_for_query(db_path, query, doc_id=doc_id, top_k=top_k, use_llm=False, search_mode="hybrid")
        trace["auto_resolution"] = auto_resolution or {}
        trace["resolved_search_mode"] = mode
        docs = trace.get("routed_documents", []) or _docs_from_tree_results(trace.get("results", []))
        docs, routing = _augment_documents_with_routing_path(db_path, docs, query, top_k=top_k, doc_id=doc_id)
        fact_matches = _fact_matches(db_path, query, doc_id, top_k=5)
        answer_plan = build_answer_plan(query, (fact_matches.get("claim_frame_matches") or {}), trace.get("evidence") or [])
        return {
            "schema": "search_report.v1",
            "query": query,
            "doc_id": doc_id,
            "requested_search_mode": requested_mode,
            "resolved_search_mode": mode,
            "effective_search_mode": "tree",
            "top_k": top_k,
            "warnings": _unique_strings([*trace.get("warnings", []), *((auto_resolution or {}).get("warnings") or []), *answer_plan.get("warnings", [])]),
            "auto_resolution": auto_resolution or {},
            "embedding_status": _safe_embedding_status(db_path),
            "hybrid_query_profile": _hybrid_query_profile(query),
            "documents": docs,
            "document_routing": routing,
            "results": trace.get("results", []),
            "tree_search_trace": trace,
            "fact_matches": fact_matches,
            "answer_plan": answer_plan,
            **answer_plan_counts(answer_plan),
        }
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        results = _search_nodes_conn(conn, query, doc_id=doc_id, top_k=top_k, search_mode=mode)
        warnings = _search_report_warnings(results, mode, db_path)
        docs = _docs_for_results(conn, results)
        docs = _augment_documents_with_routing(conn, docs, query, top_k=top_k, doc_id=doc_id, results=results)
        routing = _document_routing_report(docs)
    finally:
        conn.close()
    fact_matches = _fact_matches(db_path, query, doc_id, top_k=5)
    answer_plan = build_answer_plan(query, fact_matches.get("claim_frame_matches") or {})
    return {
        "schema": "search_report.v1",
        "query": query,
        "doc_id": doc_id,
        "requested_search_mode": requested_mode,
        "resolved_search_mode": mode,
        "effective_search_mode": "fts" if any("fts_fallback" in item.rank_reason for item in results) else mode,
        "top_k": top_k,
        "warnings": _unique_strings([*warnings, *((auto_resolution or {}).get("warnings") or []), *answer_plan.get("warnings", [])]),
        "auto_resolution": auto_resolution or {},
        "embedding_status": _safe_embedding_status(db_path),
        "hybrid_query_profile": _hybrid_query_profile(query),
        "documents": docs,
        "document_routing": routing,
        "results": [result.__dict__ for result in results],
        "fact_matches": fact_matches,
        "answer_plan": answer_plan,
        **answer_plan_counts(answer_plan),
    }


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
    search_mode: str = "hybrid",
) -> List[SearchResult]:  # type: ignore[no-untyped-def]
    mode = resolve_search_mode(search_mode)
    candidate_limit = max(top_k * 4, 20)
    fts_rows = _rank_node_rows(_fts_node_rows(conn, query, doc_id, candidate_limit), query)
    if mode == "fts":
        return _rows_to_results(fts_rows[:top_k], reason_prefix="fts")

    try:
        vector_rows = _vector_node_rows(conn, query, doc_id, candidate_limit)
    except EmbeddingError as exc:
        return _rows_to_results(fts_rows[:top_k], reason_prefix=f"fts_fallback:{exc}")
    if not vector_rows:
        return _rows_to_results(fts_rows[:top_k], reason_prefix="fts_fallback:no_embedding_index")

    quality_by_doc = _quality_by_doc_id(conn, [str(row["doc_id"]) for row in [*fts_rows, *vector_rows]])
    merged = _merge_hybrid_rows(fts_rows, vector_rows, query, quality_by_doc)
    return _rows_to_results(merged[:top_k], reason_prefix="hybrid")


def _search_documents_hybrid_conn(conn, query: str, top_k: int) -> List[Dict[str, object]]:  # type: ignore[no-untyped-def]
    node_results = _search_nodes_conn(conn, query, top_k=max(top_k * 6, 20), search_mode="hybrid")
    grouped: Dict[str, Dict[str, object]] = {}
    for rank, result in enumerate(node_results, start=1):
        item = grouped.setdefault(
            result.doc_id,
            {
                "doc_id": result.doc_id,
                "node_matches": 0,
                "score": 0.0,
                "hybrid_score": 0.0,
                "rank_reason": result.rank_reason,
                "best_node_id": result.node_id,
            },
        )
        item["node_matches"] = int(item["node_matches"]) + 1
        score = float(result.hybrid_score if result.hybrid_score is not None else result.score)
        item["hybrid_score"] = max(float(item["hybrid_score"]), score)
        item["score"] = -float(item["hybrid_score"]) if score > 0 else result.score
        if rank == 1 or score >= float(item["hybrid_score"]):
            item["best_node_id"] = result.node_id
            item["rank_reason"] = result.rank_reason

    if not grouped:
        return []

    doc_ids = list(grouped.keys())
    placeholders = ",".join("?" for _ in doc_ids)
    rows = conn.execute(
        f"""
        SELECT d.doc_id, d.title, d.path, d.file_type, d.summary, d.abstract, d.keywords
        FROM documents d
        WHERE d.doc_id IN ({placeholders}) AND d.status = 'ready'
        """,
        doc_ids,
    ).fetchall()
    docs = []
    for row in rows:
        item = dict(row)
        item.update(grouped[row["doc_id"]])
        docs.append(item)
    docs.sort(key=lambda item: (-float(item.get("hybrid_score") or 0.0), -int(item.get("node_matches") or 0)))
    return docs[:top_k]


def _search_documents_fts_conn(conn, query: str, top_k: int):  # type: ignore[no-untyped-def]
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
        return rows
    except sqlite3.OperationalError:
        return list(_fallback_doc_search(conn, query, top_k))


def _fts_node_rows(conn, query: str, doc_id: Optional[str], top_k: int) -> List[Dict[str, object]]:  # type: ignore[no-untyped-def]
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
    return [_row_dict(row) for row in rows]


def _vector_node_rows(conn, query: str, doc_id: Optional[str], top_k: int) -> List[Dict[str, object]]:  # type: ignore[no-untyped-def]
    provider = get_embedding_provider()
    query_vector = provider.embed(query)
    embedding_rows = db.get_node_embedding_rows(conn, provider=provider.name, model=provider.model, doc_id=doc_id)
    if not embedding_rows:
        return []
    rows: List[Dict[str, object]] = []
    for embedding_row in embedding_rows:
        vector = vector_from_json(str(embedding_row["vector_json"] or "[]"))
        vector_score = cosine_similarity(query_vector, vector)
        if vector_score <= 0:
            continue
        text = str(embedding_row["text"] or embedding_row["summary"] or embedding_row["heading"] or "")
        rows.append(
            {
                "node_id": embedding_row["node_id"],
                "doc_id": embedding_row["doc_id"],
                "title": embedding_row["title"],
                "path": embedding_row["path"],
                "node_path": embedding_row["node_path"],
                "heading": embedding_row["heading"],
                "type": embedding_row["type"],
                "snippet": compact_whitespace(text[:500]),
                "score": -vector_score,
                "vector_score": vector_score,
                "page_start": embedding_row["page_start"],
                "page_end": embedding_row["page_end"],
                "order_index": embedding_row["order_index"],
            }
        )
    rows.sort(key=lambda item: (-float(item["vector_score"]), int(item["order_index"] or 0)))
    return rows[:top_k]


def _merge_hybrid_rows(
    fts_rows: List[Dict[str, object]],
    vector_rows: List[Dict[str, object]],
    query: str,
    quality_by_doc: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    reasons: Dict[str, List[str]] = {}
    profile = _hybrid_query_profile(query)
    for rank, row in enumerate(fts_rows, start=1):
        item = merged.setdefault(str(row["node_id"]), dict(row))
        item["fts_rank"] = rank
        item["fts_score"] = float(row.get("score") or 0.0)
        reasons.setdefault(str(row["node_id"]), []).append("fts")
    for rank, row in enumerate(vector_rows, start=1):
        item = merged.setdefault(str(row["node_id"]), dict(row))
        item["vector_rank"] = rank
        item["vector_score"] = float(row.get("vector_score") or 0.0)
        for key, value in row.items():
            item.setdefault(key, value)
        reasons.setdefault(str(row["node_id"]), []).append("vector")

    terms = _fallback_terms(query)
    for node_id, item in merged.items():
        fts_rank = item.get("fts_rank")
        vector_rank = item.get("vector_rank")
        score = 0.0
        fts_contribution = 0.0
        vector_contribution = 0.0
        if fts_rank:
            fts_contribution = float(profile["fts_weight"]) / (60.0 + int(fts_rank))
            score += fts_contribution
        if vector_rank:
            vector_contribution = float(profile["vector_weight"]) / (60.0 + int(vector_rank))
            score += vector_contribution
        penalty, rerank_reasons = _rerank_penalty(item, terms, query, quality_by_doc)
        conflict = _hybrid_conflict_reason(fts_rank, vector_rank)
        if conflict:
            rerank_reasons.append(f"hybrid_conflict:{conflict}")
        rerank_reasons.append(f"weight:{profile['weighting_mode']}")
        item["query_intent"] = profile["intent"]
        item["hybrid_weighting"] = profile["weighting_mode"]
        item["fts_contribution"] = round(fts_contribution, 8)
        item["vector_contribution"] = round(vector_contribution, 8)
        item["hybrid_conflict"] = conflict
        item["hybrid_score"] = round(score - penalty, 8)
        item["rank_reason"] = ",".join([*reasons.get(node_id, []), *rerank_reasons])
    return sorted(
        merged.values(),
        key=lambda item: (-float(item.get("hybrid_score") or 0.0), int(item.get("order_index") or 0)),
    )


def _rerank_penalty(
    row: Dict[str, object],
    terms: List[str],
    query: str,
    quality_by_doc: Dict[str, Dict[str, object]],
) -> Tuple[float, List[str]]:
    text = " ".join(str(row.get(name) or "") for name in ("heading", "node_path", "snippet"))
    reasons: List[str] = []
    penalty = 0.0
    if row.get("type") in {"document", "page"}:
        penalty += 0.004
        reasons.append("page_penalty")
    if row.get("type") == "reference" and not any(term in query for term in ("参考", "引用", "reference")):
        penalty += 0.006
        reasons.append("reference_penalty")
    if _looks_like_front_matter(text):
        penalty += 0.006
        reasons.append("front_matter_penalty")
    if any(term and term in text for term in terms):
        penalty -= 0.006
        reasons.append("exact_term_boost")
    if row.get("type") in {"section", "abstract"}:
        penalty -= 0.002
        reasons.append("structure_boost")
    quality = quality_by_doc.get(str(row.get("doc_id") or "")) or {}
    if quality.get("quality_level") == "weak" or quality.get("page_only_tree"):
        penalty += 0.004
        reasons.append("parse_quality_penalty")
    return penalty, reasons


def _hybrid_query_profile(query: str) -> Dict[str, object]:
    profile = classify_query(query, use_llm=False)
    intent = str(profile.get("intent") or "qa")
    normalized = query.lower()
    exact_metric = bool(re.search(r"\d", query)) or any(
        term in normalized
        for term in ("指标", "数值", "成功率", "准确率", "召回率", "表格", "表 ", "table", "metric", "score", "%")
    )
    if intent == "citation":
        weighting_mode = "fts_reference_boost"
        fts_weight = 1.25
        vector_weight = 0.75
    elif exact_metric or intent == "experiment":
        weighting_mode = "fts_metric_boost"
        fts_weight = 1.15
        vector_weight = 0.85
    elif intent in {"method", "limitation", "compare", "review", "qa"}:
        weighting_mode = "semantic_vector_boost"
        fts_weight = 0.85
        vector_weight = 1.15
    else:
        weighting_mode = "balanced"
        fts_weight = 1.0
        vector_weight = 1.0
    return {
        "schema": "hybrid_query_profile.v1",
        "intent": intent,
        "weighting_mode": weighting_mode,
        "fts_weight": fts_weight,
        "vector_weight": vector_weight,
        "exact_metric": exact_metric,
    }


def _hybrid_conflict_reason(fts_rank: object, vector_rank: object) -> str:
    fts_value = _optional_float(fts_rank)
    vector_value = _optional_float(vector_rank)
    if fts_value is None and vector_value is not None:
        return "vector_without_fts"
    if fts_value is not None and vector_value is None:
        return "fts_without_vector"
    if fts_value is not None and vector_value is not None and abs(fts_value - vector_value) >= 8:
        return "rank_divergence"
    return ""


def _rank_node_rows(rows: List[Dict[str, object]], query: str) -> List[Dict[str, object]]:
    terms = _fallback_terms(query)

    def key(row: Dict[str, object]) -> tuple:
        text = " ".join(str(row.get(name) or "") for name in ("heading", "node_path", "snippet"))
        penalty = 0
        if row.get("type") in {"document", "page"}:
            penalty += 3
        if row.get("type") == "reference" and not any(term in query for term in ("参考", "引用", "reference")):
            penalty += 3
        if _looks_like_front_matter(text):
            penalty += 3
        if any(term and term in text for term in terms):
            penalty -= 3
        if row.get("type") in {"section", "abstract"}:
            penalty -= 1
        return (penalty, float(row.get("score") or 0.0), int(row.get("order_index") or 0))

    return sorted(rows, key=key)


def _rows_to_results(rows: List[Dict[str, object]], reason_prefix: str) -> List[SearchResult]:
    results = []
    for row in rows:
        rank_reason = str(row.get("rank_reason") or reason_prefix)
        if reason_prefix not in rank_reason:
            rank_reason = f"{reason_prefix},{rank_reason}"
        score = row.get("hybrid_score")
        if score is None:
            score = row.get("score") or 0.0
        results.append(
            SearchResult(
                doc_id=str(row["doc_id"]),
                node_id=str(row["node_id"]),
                title=str(row["title"]),
                path=str(row["path"]),
                node_path=str(row["node_path"]),
                heading=str(row["heading"] or ""),
                snippet=compact_whitespace(str(row.get("snippet") or "")),
                score=float(score or 0.0),
                page_start=row.get("page_start"),  # type: ignore[arg-type]
                page_end=row.get("page_end"),  # type: ignore[arg-type]
                fts_score=_optional_float(row.get("fts_score", row.get("score"))),
                vector_score=_optional_float(row.get("vector_score")),
                hybrid_score=_optional_float(row.get("hybrid_score")),
                rank_reason=rank_reason,
                query_intent=str(row.get("query_intent") or ""),
                hybrid_weighting=str(row.get("hybrid_weighting") or ""),
                fts_contribution=_optional_float(row.get("fts_contribution")),
                vector_contribution=_optional_float(row.get("vector_contribution")),
                hybrid_conflict=str(row.get("hybrid_conflict") or ""),
            )
        )
    return results


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
    front_matter_tokens = (
        "目录",
        "学院",
        "专业",
        "研究生",
        "指导教师",
        "答辩",
        "分类号",
        "学号",
        "密级",
        "网络首发",
        "收稿日期",
        "引用格式",
        "出版确认",
        "doi",
        "issn",
    )
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


def _quality_by_doc_id(conn, doc_ids: List[str]) -> Dict[str, Dict[str, object]]:  # type: ignore[no-untyped-def]
    unique = sorted({doc_id for doc_id in doc_ids if doc_id})
    if not unique:
        return {}
    placeholders = ",".join("?" for _ in unique)
    rows = conn.execute(
        f"SELECT doc_id, card_json FROM doc_cards WHERE doc_id IN ({placeholders})",
        unique,
    ).fetchall()
    result: Dict[str, Dict[str, object]] = {}
    for row in rows:
        try:
            card = json.loads(row["card_json"])
        except json.JSONDecodeError:
            continue
        quality = card.get("parse_quality")
        if isinstance(quality, dict):
            result[row["doc_id"]] = quality
    return result


def _docs_for_results(conn, results: List[SearchResult]) -> List[Dict[str, object]]:  # type: ignore[no-untyped-def]
    doc_ids = sorted({result.doc_id for result in results})
    if not doc_ids:
        return []
    placeholders = ",".join("?" for _ in doc_ids)
    rows = conn.execute(
        f"""
        SELECT doc_id, title, path, file_type, summary, abstract, keywords
        FROM documents
        WHERE doc_id IN ({placeholders})
        """,
        doc_ids,
    ).fetchall()
    matches = {doc_id: 0 for doc_id in doc_ids}
    for result in results:
        matches[result.doc_id] += 1
    return [{**dict(row), "node_matches": matches.get(row["doc_id"], 0)} for row in rows]


ROUTING_FIELD_WEIGHTS = {
    "title": 3.0,
    "abstract": 2.5,
    "keywords": 2.0,
    "description": 1.6,
    "method_summary": 1.4,
    "innovation_summary": 1.2,
    "limitation_summary": 1.0,
}


def _augment_documents_with_routing_path(
    db_path: Path,
    docs: List[Dict[str, object]],
    query: str,
    *,
    top_k: int,
    doc_id: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        augmented = _augment_documents_with_routing(conn, docs, query, top_k=top_k, doc_id=doc_id)
        return augmented, _document_routing_report(augmented)
    finally:
        conn.close()


def _augment_documents_with_routing(
    conn,
    docs: List[Dict[str, object]],
    query: str,
    *,
    top_k: int,
    doc_id: Optional[str] = None,
    results: Optional[List[SearchResult]] = None,
) -> List[Dict[str, object]]:  # type: ignore[no-untyped-def]
    existing: Dict[str, Dict[str, object]] = {}
    existing_rank: Dict[str, int] = {}
    for rank, doc in enumerate(docs, start=1):
        current = dict(doc)
        current_doc_id = str(current.get("doc_id") or "")
        if not current_doc_id:
            continue
        existing[current_doc_id] = current
        existing_rank[current_doc_id] = rank

    node_signals = _node_route_signals(results or [])
    route_rows = _doc_card_route_rows(conn, doc_id=doc_id)
    routed: Dict[str, Dict[str, object]] = {}
    for row in route_rows:
        current_doc_id = str(row.get("doc_id") or "")
        if not current_doc_id:
            continue
        base = existing.get(current_doc_id, dict(row))
        card = _json_dict(row.get("card_json"))
        merged = _merge_doc_card_fields(base, card)
        route = _document_route_explanation(merged, query, node_signals.get(current_doc_id))
        if current_doc_id not in existing and route["routing_score"] <= 0:
            continue
        merged.update(route)
        routed[current_doc_id] = merged

    for current_doc_id, doc in existing.items():
        if current_doc_id in routed:
            continue
        route = _document_route_explanation(doc, query, node_signals.get(current_doc_id))
        doc.update(route)
        routed[current_doc_id] = doc

    items = list(routed.values())
    items.sort(
        key=lambda item: (
            -float(item.get("routing_score") or 0.0),
            existing_rank.get(str(item.get("doc_id") or ""), 10_000),
            str(item.get("title") or ""),
        )
    )
    return items[:top_k]


def _doc_card_route_rows(conn, doc_id: Optional[str] = None) -> List[Dict[str, object]]:  # type: ignore[no-untyped-def]
    params: List[object] = []
    doc_filter = ""
    if doc_id:
        doc_filter = "AND d.doc_id = ?"
        params.append(doc_id)
    rows = conn.execute(
        f"""
        SELECT d.doc_id, d.title, d.path, d.file_type, d.summary, d.abstract, d.keywords,
               c.card_json
        FROM documents d
        LEFT JOIN doc_cards c ON c.doc_id = d.doc_id
        WHERE d.status = 'ready' {doc_filter}
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _merge_doc_card_fields(doc: Dict[str, object], card: Dict[str, object]) -> Dict[str, object]:
    merged = dict(doc)
    merged.pop("card_json", None)
    for field in ("description", "method_summary", "innovation_summary", "limitation_summary", "summary_source"):
        value = card.get(field)
        if value and not merged.get(field):
            merged[field] = value
    if card.get("keywords") and not merged.get("keywords"):
        merged["keywords"] = card.get("keywords")
    if card.get("abstract") and not merged.get("abstract"):
        merged["abstract"] = card.get("abstract")
    return merged


def _document_route_explanation(
    doc: Dict[str, object],
    query: str,
    node_signal: Optional[Dict[str, object]],
) -> Dict[str, object]:
    terms = _routing_terms(query)
    fields = _routing_fields(doc)
    field_hits: Dict[str, List[str]] = {}
    score = 0.0
    for field, text in fields.items():
        hits = _matching_terms(text, terms)
        if not hits:
            continue
        field_hits[field] = hits
        score += ROUTING_FIELD_WEIGHTS.get(field, 1.0) * max(1, len(hits))

    selection_reasons = [f"field:{field}" for field in field_hits]
    node_matches = 0
    if node_signal:
        node_matches = int(node_signal.get("node_matches") or 0)
        if node_matches:
            score += min(3.0, 0.7 * node_matches)
            reason = str(node_signal.get("rank_reason") or "node_match")
            selection_reasons.append(f"node:{reason}")
        hybrid_score = _optional_float(node_signal.get("hybrid_score") or node_signal.get("score"))
        if hybrid_score is not None and hybrid_score > 0:
            score += min(2.0, hybrid_score * 40.0)

    fallback_reason = ""
    if field_hits and not node_matches:
        fallback_reason = "doc_card_field_match"
    elif node_matches and not field_hits:
        fallback_reason = "node_match_without_doc_card_hit"
    elif not field_hits and not node_matches:
        fallback_reason = "no_route_signal"

    return {
        "field_hits": field_hits,
        "routing_score": round(score, 3),
        "selection_reasons": _unique_strings(selection_reasons),
        "fallback_reason": fallback_reason,
        "node_route_signal": node_signal or {},
    }


def _document_routing_report(docs: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "schema": "document_routing.v1",
        "count": len(docs),
        "items": [
            {
                "doc_id": item.get("doc_id", ""),
                "title": item.get("title", ""),
                "field_hits": item.get("field_hits") or {},
                "routing_score": item.get("routing_score", 0.0),
                "selection_reasons": item.get("selection_reasons") or [],
                "fallback_reason": item.get("fallback_reason", ""),
                "node_matches": item.get("node_matches", 0),
                "rank_reason": item.get("rank_reason", ""),
                "node_route_signal": item.get("node_route_signal") or {},
            }
            for item in docs
        ],
    }


def _node_route_signals(results: List[SearchResult]) -> Dict[str, Dict[str, object]]:
    signals: Dict[str, Dict[str, object]] = {}
    for result in results:
        item = signals.setdefault(
            result.doc_id,
            {
                "node_matches": 0,
                "rank_reason": result.rank_reason,
                "score": result.score,
                "hybrid_score": result.hybrid_score,
                "query_intent": result.query_intent,
                "hybrid_weighting": result.hybrid_weighting,
                "fts_contribution": result.fts_contribution,
                "vector_contribution": result.vector_contribution,
                "hybrid_conflict": result.hybrid_conflict,
            },
        )
        item["node_matches"] = int(item["node_matches"]) + 1
        current_score = _optional_float(item.get("hybrid_score") or item.get("score")) or 0.0
        next_score = _optional_float(result.hybrid_score or result.score) or 0.0
        if next_score >= current_score:
            item["rank_reason"] = result.rank_reason
            item["score"] = result.score
            item["hybrid_score"] = result.hybrid_score
            item["query_intent"] = result.query_intent
            item["hybrid_weighting"] = result.hybrid_weighting
            item["fts_contribution"] = result.fts_contribution
            item["vector_contribution"] = result.vector_contribution
            item["hybrid_conflict"] = result.hybrid_conflict
    return signals


def _routing_fields(doc: Dict[str, object]) -> Dict[str, str]:
    return {
        "title": str(doc.get("title") or ""),
        "abstract": str(doc.get("abstract") or ""),
        "keywords": _keywords_text(doc.get("keywords")),
        "description": str(doc.get("description") or doc.get("summary") or ""),
        "method_summary": str(doc.get("method_summary") or ""),
        "innovation_summary": str(doc.get("innovation_summary") or ""),
        "limitation_summary": str(doc.get("limitation_summary") or ""),
    }


def _routing_terms(query: str) -> List[str]:
    terms = list(_fallback_terms(query))
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    result: List[str] = []
    seen = set()
    for term in terms:
        cleaned = term.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _matching_terms(text: str, terms: List[str]) -> List[str]:
    haystack = text.lower()
    matches = []
    for term in terms:
        if term and term in haystack:
            matches.append(term)
    return matches[:6]


def _keywords_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list):
            return " ".join(str(item) for item in parsed)
        return value
    return ""


def _json_dict(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _docs_from_tree_results(results: object) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    if not isinstance(results, list):
        return []
    for item in results:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "")
        if not doc_id:
            continue
        doc = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": item.get("title") or doc_id,
                "path": item.get("path") or "",
                "node_matches": 0,
                "score": item.get("score"),
                "rank_reason": item.get("rank_reason") or "tree:value",
                "best_node_id": item.get("node_id"),
            },
        )
        doc["node_matches"] = int(doc["node_matches"]) + 1
        if float(item.get("score") or 0.0) > float(doc.get("score") or 0.0):
            doc["score"] = item.get("score")
            doc["best_node_id"] = item.get("node_id")
            doc["rank_reason"] = item.get("rank_reason") or doc["rank_reason"]
    return list(grouped.values())


def _search_report_warnings(results: List[SearchResult], mode: str, db_path: Path) -> List[str]:
    warnings: List[str] = []
    if mode == "hybrid" and any("fts_fallback" in result.rank_reason for result in results):
        warnings.append("hybrid_fallback_to_fts")
    status = _safe_embedding_status(db_path)
    if mode == "hybrid" and not status.get("ready"):
        warnings.append("missing_embedding_index")
    if not results:
        warnings.append("no_search_results")
    return _unique_strings(warnings)


def _fact_matches(db_path: Path, query: str, doc_id: Optional[str], top_k: int) -> Dict[str, object]:
    claim_frame_matches: Dict[str, object] = {
        "schema": "claim_frame_search.v1",
        "available": False,
        "count": 0,
        "items": [],
        "warnings": ["claim_frame_search_not_run"],
    }
    try:
        from .claim_frames import search_claim_frames

        claim_frame_matches = search_claim_frames(db_path, query, doc_ids=[doc_id] if doc_id else None, top_k=top_k)
    except Exception as exc:
        claim_frame_matches = {
            "schema": "claim_frame_search.v1",
            "available": False,
            "count": 0,
            "items": [],
            "warnings": [f"claim_frame_search_unavailable:{exc}"],
        }
    try:
        from .facts import fact_search

        result = fact_search(db_path, query, doc_ids=[doc_id] if doc_id else None, top_k=top_k)
    except Exception as exc:
        return {
            "schema": "fact_search_summary.v1",
            "available": False,
            "count": 0,
            "items": [],
            "claim_frame_matches": claim_frame_matches,
            "warnings": [f"fact_search_unavailable:{exc}"],
        }
    return {
        "schema": "fact_search_summary.v1",
        "available": True,
        "count": result.get("count", 0),
        "items": result.get("items", [])[:top_k],
        "claim_frame_matches": claim_frame_matches,
        "table_backed_count": sum(1 for item in result.get("items", [])[:top_k] if item.get("source_kind") == "table"),
        "text_backed_count": sum(1 for item in result.get("items", [])[:top_k] if item.get("source_kind") != "table"),
        "warnings": [],
    }


def _safe_embedding_status(db_path: Path) -> Dict[str, object]:
    try:
        return semantic_index_status(db_path)
    except EmbeddingError as exc:
        return {
            "schema": "semantic_index_status.v1",
            "ready": False,
            "error": str(exc),
        }


def _auto_resolution(db_path: Path, query: str) -> Dict[str, object]:
    from .search_profile import resolve_auto_search_mode

    return resolve_auto_search_mode(db_path, query)


def _tag_auto_results(results: List[SearchResult], auto_resolution: Optional[Dict[str, object]]) -> List[SearchResult]:
    if not auto_resolution:
        return results
    profile = str(auto_resolution.get("profile_name") or "none")
    resolved = str(auto_resolution.get("resolved_search_mode") or "hybrid")
    for result in results:
        result.rank_reason = f"auto:{profile}:{resolved},{result.rank_reason}"
    return results


def _tag_auto_docs(docs: List[Dict[str, object]], auto_resolution: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
    if not auto_resolution:
        return docs
    profile = str(auto_resolution.get("profile_name") or "none")
    resolved = str(auto_resolution.get("resolved_search_mode") or "hybrid")
    for doc in docs:
        reason = str(doc.get("rank_reason") or resolved)
        doc["rank_reason"] = f"auto:{profile}:{resolved},{reason}"
        doc["auto_resolution"] = auto_resolution
    return docs


def _row_dict(row) -> Dict[str, object]:  # type: ignore[no-untyped-def]
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
