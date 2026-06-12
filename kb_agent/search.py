from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import db
from .answer_plan import answer_plan_counts, build_answer_plan
from .embeddings import EmbeddingError, get_embedding_provider, semantic_index_status
from .models import EvidencePacket, SearchResult
from .search_core import (
    augment_documents_with_routing as _augment_documents_with_routing,
    augment_documents_with_routing_path as _augment_documents_with_routing_path,
    docs_for_results as _docs_for_results,
    docs_from_tree_results as _docs_from_tree_results,
    document_routing_report as _document_routing_report,
    fts_query,
    hybrid_query_profile as _hybrid_query_profile,
    search_documents_fts_conn as _search_documents_fts_conn,
    search_documents_hybrid_conn as _search_documents_hybrid_core_conn,
    search_nodes_flat_conn as _search_nodes_core_conn,
)
from .utils import unique_strings as _unique_strings


SEARCH_MODES = {"hybrid", "fts", "tree", "auto"}


def resolve_search_mode(search_mode: str = "hybrid") -> str:
    mode = (search_mode or "hybrid").strip().lower()
    if mode not in SEARCH_MODES:
        choices = ", ".join(sorted(SEARCH_MODES))
        raise ValueError(f"Unsupported search_mode '{search_mode}'. Expected one of: {choices}")
    return mode


def _search_nodes_conn(conn, query: str, doc_id: Optional[str] = None, top_k: int = 8, search_mode: str = "hybrid") -> List[SearchResult]:  # type: ignore[no-untyped-def]
    return _search_nodes_core_conn(
        conn,
        query,
        doc_id=doc_id,
        top_k=top_k,
        search_mode=search_mode,
        embedding_provider_factory=get_embedding_provider,
    )


def _search_documents_hybrid_conn(conn, query: str, top_k: int) -> List[Dict[str, object]]:  # type: ignore[no-untyped-def]
    return _search_documents_hybrid_core_conn(conn, query, top_k, embedding_provider_factory=get_embedding_provider)


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
