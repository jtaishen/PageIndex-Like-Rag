from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .artifacts import get_artifact, get_doc_card, list_artifacts
from .claim_frame_builder import (
    CLAIM_FRAME_SCHEMA,
    LLM_ENHANCE_FRAME_LIMIT,
    LLM_ENHANCE_UNIT_LIMIT,
    MAX_CLAIM_CHARS,
    build_claim_frames,
    claim_frames_payload as _builder_claim_frames_payload,
    claim_type as _builder_claim_type,
    confidence as _builder_confidence,
    count_by_field as _builder_count_by_field,
    dedupe_frames as _builder_dedupe_frames,
    enhance_frames_with_llm as _builder_enhance_frames_with_llm,
    frame_record as _builder_frame_record,
    infer_frame_fields as _builder_infer_frame_fields,
)
from .claim_frame_evidence import (
    evidence_units_from_artifacts,
    source_artifact_ids,
    unit_by_id as _unit_by_id,
    unit_by_node_id as _unit_by_node_id,
    unit_by_source_id as _unit_by_source_id,
)
from .claim_frame_quality import LOW_FRAME_QUALITY_SCORE, MIN_FRAME_QUALITY_SCORE
from .claim_frame_search import (
    claim_frame_search_item,
    frame_match_score as _search_frame_match_score,
    matched_frame_fields as _search_matched_frame_fields,
    normalize_support_status as _search_normalize_support_status,
    query_terms as _search_query_terms,
    rank_claim_frame_items,
)
from .claim_frame_verifier import (
    CLAIM_FRAME_VERIFIER_RESULT_SCHEMA,
    CLAIM_FRAME_VERIFIER_SCHEMA,
    merge_count_dicts as _verifier_merge_count_dicts,
    semantic_result as _verifier_semantic_result,
    semantic_support_for_frame as _verifier_semantic_support_for_frame,
    sync_claim_frames_with_verifier as _verifier_sync_claim_frames_with_verifier,
    verifier_totals as _verifier_totals,
    verify_claim_frames_payload as _verifier_claim_frames_payload,
)
from .llm import LLMError, generate_json_object
from .utils import unique_strings as _unique_strings, write_json


EVIDENCE_UNITS_ARTIFACT = "evidence_units.json"
CLAIM_FRAMES_ARTIFACT = "claim_frames.json"
CLAIM_FRAME_VERIFIER_ARTIFACT = "claim_frame_verifier.json"

EVIDENCE_UNIT_SCHEMA = "evidence_units.v1"


def extract_evidence_units(db_path: Path, doc_id: str, *, force: bool = False) -> Dict[str, Any]:
    listing = list_artifacts(db_path, doc_id)
    artifact_dir = Path(str(listing["artifact_dir"]))
    version_id = str(listing["version_id"])
    path = artifact_dir / EVIDENCE_UNITS_ARTIFACT
    if path.exists() and not force:
        payload = _read_json(path, {})
        if payload.get("schema") == EVIDENCE_UNIT_SCHEMA:
            return {
                "schema": "evidence_unit_extraction_result.v1",
                "doc_id": doc_id,
                "version_id": version_id,
                "skipped": True,
                "path": str(path),
                "evidence_units": payload,
            }

    nodes = _artifact_content(db_path, doc_id, "node_index.jsonl", [])
    if not isinstance(nodes, list):
        nodes = []
    table_summaries = _artifact_content(db_path, doc_id, "table_summaries.json", {})
    figures = _artifact_content(db_path, doc_id, "figures.json", {})
    reference_sections = _artifact_content(db_path, doc_id, "reference_sections.json", {})
    citation_map = _artifact_content(db_path, doc_id, "citation_map.json", {})
    units = evidence_units_from_artifacts(doc_id, version_id, nodes, table_summaries, figures, reference_sections, citation_map)
    warnings: List[str] = []
    if not units:
        warnings.append("no_evidence_units")
    payload = {
        "schema": EVIDENCE_UNIT_SCHEMA,
        "status": "extracted" if units else "partial",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(units),
        "units": units,
        "unit_type_counts": _count_by_field(units, "unit_type"),
        "source_kind_counts": _count_by_field(units, "source_kind"),
        "warnings": warnings,
        "created_at": time.time(),
    }
    write_json(path, payload)
    return {
        "schema": "evidence_unit_extraction_result.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "skipped": False,
        "path": str(path),
        "evidence_units": payload,
    }


def get_evidence_units(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, EVIDENCE_UNITS_ARTIFACT, version_id=version_id)["content"]


def extract_claim_frames(
    db_path: Path,
    doc_id: str,
    *,
    force: bool = False,
    use_llm: bool = True,
    require_llm: bool = False,
) -> Dict[str, Any]:
    listing = list_artifacts(db_path, doc_id)
    artifact_dir = Path(str(listing["artifact_dir"]))
    version_id = str(listing["version_id"])
    path = artifact_dir / CLAIM_FRAMES_ARTIFACT
    if path.exists() and not force:
        payload = _read_json(path, {})
        if payload.get("schema") == CLAIM_FRAME_SCHEMA:
            return {
                "schema": "claim_frame_extraction_result.v1",
                "doc_id": doc_id,
                "version_id": version_id,
                "skipped": True,
                "path": str(path),
                "claim_frames": payload,
            }

    evidence_units = _ensure_evidence_units(db_path, doc_id, force=force)
    units = evidence_units.get("units") or []
    unit_by_node = _unit_by_node_id(units)
    unit_by_id = _unit_by_id(units)
    unit_by_source_id = _unit_by_source_id(units)
    card = _safe_doc_card(db_path, doc_id)
    claims = _artifact_content(db_path, doc_id, "claims.json", {})
    innovation = _artifact_content(db_path, doc_id, "innovation.json", {})
    table_summaries = _artifact_content(db_path, doc_id, "table_summaries.json", {})
    citation_map = _artifact_content(db_path, doc_id, "citation_map.json", {})
    warnings: List[str] = []
    frames = build_claim_frames(
        doc_id,
        version_id,
        card,
        claims,
        innovation,
        table_summaries,
        citation_map,
        unit_by_node,
        unit_by_id,
        unit_by_source_id,
    )
    if not frames:
        warnings.append("no_claim_frames")

    llm_error = ""
    llm_used = False
    llm_metadata: Dict[str, Any] = {}
    if use_llm and frames:
        try:
            frames, llm_metadata = _enhance_frames_with_llm(card, frames, units)
            llm_used = True
            warnings.extend(llm_metadata.get("enhancement_warnings") or [])
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            warnings.append(f"llm_unavailable:{exc.error_type}")

    payload = _claim_frames_payload(
        doc_id,
        version_id,
        frames,
        evidence_unit_count=len(units),
        warnings=warnings,
        llm_used=llm_used,
        llm_error=llm_error,
        llm_metadata=llm_metadata,
    )
    verifier = _verify_claim_frames_payload(db_path, doc_id, payload, evidence_units)
    _sync_claim_frames_with_verifier(payload, verifier)
    write_json(path, payload)
    write_json(artifact_dir / CLAIM_FRAME_VERIFIER_ARTIFACT, verifier)
    return {
        "schema": "claim_frame_extraction_result.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "skipped": False,
        "path": str(path),
        "verifier_path": str(artifact_dir / CLAIM_FRAME_VERIFIER_ARTIFACT),
        "claim_frames": payload,
        "verifier": verifier,
    }


def get_claim_frames(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, CLAIM_FRAMES_ARTIFACT, version_id=version_id)["content"]


def verify_claim_frames(db_path: Path, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    selected_doc_ids = _ready_doc_ids(db_path) if doc_ids is None else _unique_strings(doc_ids)
    documents = []
    warnings: List[str] = []
    for doc_id in selected_doc_ids:
        try:
            claim_frames = _ensure_claim_frames(db_path, doc_id, force=False, use_llm=False)
            evidence_units = _ensure_evidence_units(db_path, doc_id, force=False)
            report = _verify_claim_frames_payload(db_path, doc_id, claim_frames, evidence_units)
            artifact_dir = Path(str(list_artifacts(db_path, doc_id)["artifact_dir"]))
            write_json(artifact_dir / CLAIM_FRAME_VERIFIER_ARTIFACT, report)
            documents.append(report)
        except Exception as exc:
            warnings.append(f"claim_frame_verify_failed:{doc_id}:{exc}")
            documents.append(
                {
                    "schema": CLAIM_FRAME_VERIFIER_SCHEMA,
                    "doc_id": doc_id,
                    "status": "failed",
                    "frame_count": 0,
                    "verified_frame_rate": 0.0,
                    "warnings": [str(exc)],
                }
            )
    totals = _verifier_totals(documents)
    return {
        "schema": CLAIM_FRAME_VERIFIER_RESULT_SCHEMA,
        "status": "passed"
        if not warnings and totals["unsupported_frame_count"] == 0 and totals["contradicted_frame_count"] == 0
        else "needs_review",
        "doc_count": len(documents),
        "documents": documents,
        **totals,
        "warnings": _unique_strings(warnings),
        "created_at": time.time(),
    }


def claim_frame_summary_for_doc(db_path: Path, doc_id: str) -> Dict[str, Any]:
    try:
        frames = get_claim_frames(db_path, doc_id)
        verifier = _artifact_content(db_path, doc_id, CLAIM_FRAME_VERIFIER_ARTIFACT, {})
        evidence_units = _ensure_evidence_units(db_path, doc_id, force=False)
    except (FileNotFoundError, KeyError, ValueError):
        return {"schema": "claim_frame_summary.v1", "doc_id": doc_id, "available": False}
    items = frames.get("frames") or []
    units = evidence_units.get("units") or []
    return {
        "schema": "claim_frame_summary.v1",
        "doc_id": doc_id,
        "available": True,
        "evidence_unit_count": evidence_units.get("count", len(units)),
        "source_kind_counts": evidence_units.get("source_kind_counts") or _count_by_field(units, "source_kind"),
        "frame_count": len(items),
        "verified_frame_rate": verifier.get("verified_frame_rate", 0.0),
        "unsupported_frame_count": verifier.get("unsupported_frame_count", 0),
        "trace_status_counts": verifier.get("trace_status_counts") or _count_by_field(items, "trace_status"),
        "support_status_counts": verifier.get("support_status_counts") or _count_by_field(items, "support_status"),
        "semantic_support_status_counts": verifier.get("semantic_support_status_counts") or _count_by_field(items, "semantic_support_status"),
        "semantic_supported_frame_rate": verifier.get("semantic_supported_frame_rate", 0.0),
        "semantic_verified_frame_count": verifier.get("semantic_verified_frame_count", 0),
        "partial_supported_frame_count": verifier.get("partial_supported_frame_count", 0),
        "related_only_frame_count": verifier.get("related_only_frame_count", 0),
        "contradicted_frame_count": verifier.get("contradicted_frame_count", 0),
        "insufficient_evidence_frame_count": verifier.get("insufficient_evidence_frame_count", 0),
        "citation_risk_counts": verifier.get("citation_risk_counts") or _count_by_field(items, "citation_risk"),
        "missing_evidence_unit_count": verifier.get("missing_evidence_unit_count", 0),
        "missing_node_count": verifier.get("missing_node_count", 0),
        "missing_source_count": verifier.get("missing_source_count", 0),
        "citation_gap_count": verifier.get("citation_gap_count", 0),
        "low_quality_frame_count": verifier.get("low_quality_frame_count", frames.get("low_quality_frame_count", 0)),
        "noisy_frame_count": verifier.get("noisy_frame_count", frames.get("noisy_frame_count", 0)),
        "ignored_noise_frame_count": verifier.get("ignored_noise_frame_count", 0),
        "top_frame_noise_reasons": verifier.get("top_frame_noise_reasons") or frames.get("top_frame_noise_reasons") or [],
        "type_counts": frames.get("claim_type_counts") or {},
        "top_frames": [
            {
                "frame_id": item.get("frame_id"),
                "claim_type": item.get("claim_type"),
                "short_claim": item.get("short_claim"),
                "trace_status": item.get("trace_status", ""),
                "support_status": item.get("support_status"),
                "support_reason": item.get("support_reason", ""),
                "semantic_support_status": item.get("semantic_support_status", "not_checked"),
                "semantic_support_score": item.get("semantic_support_score", 0.0),
                "semantic_support_reason": item.get("semantic_support_reason", ""),
                "citation_risk": item.get("citation_risk", "not_checked"),
                "primary_evidence_unit_ids": item.get("primary_evidence_unit_ids") or [],
                "weak_evidence_unit_ids": item.get("weak_evidence_unit_ids") or [],
                "contradictory_evidence_unit_ids": item.get("contradictory_evidence_unit_ids") or [],
                "source": item.get("source", ""),
                "quality_score": item.get("quality_score", 0.0),
                "frame_quality": item.get("frame_quality", ""),
                "noise_reasons": item.get("noise_reasons") or [],
                "evidence_unit_ids": item.get("evidence_unit_ids") or [],
                "confidence": item.get("confidence"),
                "warnings": item.get("warnings") or [],
            }
            for item in sorted((item for item in items if isinstance(item, dict)), key=lambda item: (-float(item.get("quality_score") or 0.0), str(item.get("frame_id") or "")))[:6]
            if isinstance(item, dict)
        ],
        "warnings": _unique_strings([*(frames.get("warnings") or []), *(verifier.get("warnings") or [])]),
    }


def search_claim_frames(
    db_path: Path,
    query: str,
    *,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    terms = _query_terms(query)
    selected_doc_ids = _ready_doc_ids(db_path) if doc_ids is None else _unique_strings(doc_ids)
    items: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for doc_id in selected_doc_ids:
        try:
            frames_payload = get_claim_frames(db_path, doc_id)
            verifier = _artifact_content(db_path, doc_id, CLAIM_FRAME_VERIFIER_ARTIFACT, {})
        except (FileNotFoundError, KeyError, ValueError):
            continue
        verified_by_id = {
            str(item.get("frame_id") or ""): item
            for item in verifier.get("items") or []
            if isinstance(item, dict)
        }
        for frame in frames_payload.get("frames") or []:
            if not isinstance(frame, dict):
                continue
            item = claim_frame_search_item(frame, verified_by_id.get(str(frame.get("frame_id") or "")) or {}, query, terms)
            if item:
                items.append(item)
    ranked = rank_claim_frame_items(items, top_k)
    return {
        "schema": "claim_frame_search.v1",
        "query": query,
        "doc_ids": doc_ids or [],
        "available": bool(ranked),
        "count": len(ranked),
        "items": ranked,
        "warnings": warnings,
    }


def _ensure_evidence_units(db_path: Path, doc_id: str, *, force: bool) -> Dict[str, Any]:
    try:
        payload = get_evidence_units(db_path, doc_id)
        if payload.get("schema") == EVIDENCE_UNIT_SCHEMA and not force:
            return payload
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return extract_evidence_units(db_path, doc_id, force=True)["evidence_units"]


def _ensure_claim_frames(db_path: Path, doc_id: str, *, force: bool, use_llm: bool) -> Dict[str, Any]:
    try:
        payload = get_claim_frames(db_path, doc_id)
        if payload.get("schema") == CLAIM_FRAME_SCHEMA and not force:
            return payload
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return extract_claim_frames(db_path, doc_id, force=True, use_llm=use_llm, require_llm=False)["claim_frames"]


def _artifact_content(db_path: Path, doc_id: str, name: str, default: Any) -> Any:
    try:
        return get_artifact(db_path, doc_id, name)["content"]
    except (FileNotFoundError, KeyError, ValueError):
        return default


def _safe_doc_card(db_path: Path, doc_id: str) -> Dict[str, Any]:
    try:
        return get_doc_card(db_path, doc_id)
    except (FileNotFoundError, KeyError, ValueError):
        return {"doc_id": doc_id, "title": doc_id}


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _ready_doc_ids(db_path: Path) -> List[str]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        return [str(row["doc_id"]) for row in db.list_documents(conn) if row["status"] == "ready"]
    finally:
        conn.close()


def _frame_record(
    doc_id: str,
    version_id: str,
    claim_type: str,
    text: str,
    evidence_unit_ids: List[str],
    *,
    source: str,
    source_claim_ids: List[str],
    confidence: float,
    index: int,
    problem: str = "",
    method: str = "",
    binding_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return _builder_frame_record(
        doc_id,
        version_id,
        claim_type,
        text,
        evidence_unit_ids,
        source=source,
        source_claim_ids=source_claim_ids,
        confidence=confidence,
        index=index,
        problem=problem,
        method=method,
        binding_warnings=binding_warnings,
    )


def _enhance_frames_with_llm(
    card: Dict[str, Any],
    frames: List[Dict[str, Any]],
    units: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    return _builder_enhance_frames_with_llm(card, frames, units, json_generator=generate_json_object)


def _claim_frames_payload(
    doc_id: str,
    version_id: str,
    frames: List[Dict[str, Any]],
    *,
    evidence_unit_count: int,
    warnings: List[str],
    llm_used: bool,
    llm_error: str,
    llm_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return _builder_claim_frames_payload(
        doc_id,
        version_id,
        frames,
        evidence_unit_count=evidence_unit_count,
        warnings=warnings,
        llm_used=llm_used,
        llm_error=llm_error,
        llm_metadata=llm_metadata,
    )


def _verify_claim_frames_payload(
    db_path: Path,
    doc_id: str,
    claim_frames: Dict[str, Any],
    evidence_units: Dict[str, Any],
) -> Dict[str, Any]:
    nodes = _artifact_content(db_path, doc_id, "node_index.jsonl", [])
    node_ids = {str(node.get("node_id") or "") for node in nodes if isinstance(node, dict)}
    source_ids = source_artifact_ids(
        doc_id,
        lambda lookup_doc_id, name, default: _artifact_content(db_path, lookup_doc_id, name, default),
    )
    citation_map = _artifact_content(db_path, doc_id, "citation_map.json", {})
    return _verifier_claim_frames_payload(
        doc_id,
        claim_frames,
        evidence_units,
        node_ids=node_ids,
        source_ids=source_ids,
        citation_map=citation_map,
    )


def _sync_claim_frames_with_verifier(payload: Dict[str, Any], verifier: Dict[str, Any]) -> None:
    _verifier_sync_claim_frames_with_verifier(payload, verifier)


def _semantic_support_for_frame(frame: Dict[str, Any], units: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _verifier_semantic_support_for_frame(frame, units)


def _semantic_result(
    status: str,
    score: float,
    reason: str,
    primary_ids: List[str],
    weak_ids: List[str],
    contradictory_ids: List[str],
) -> Dict[str, Any]:
    return _verifier_semantic_result(status, score, reason, primary_ids, weak_ids, contradictory_ids)


def _claim_type(raw: str) -> str:
    return _builder_claim_type(raw)


def _infer_frame_fields(text: str, claim_type: str) -> Dict[str, str]:
    return _builder_infer_frame_fields(text, claim_type)


def _confidence(value: Any, default: float) -> float:
    return _builder_confidence(value, default)


def _normalize_support_status(value: str) -> str:
    return _search_normalize_support_status(value)


def _dedupe_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _builder_dedupe_frames(frames)


def _count_by_field(items: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return _builder_count_by_field(items, field)


def _query_terms(query: str) -> List[str]:
    return _search_query_terms(query)


def _frame_match_score(
    frame: Dict[str, Any],
    terms: List[str],
    *,
    query: str = "",
    support_status: str = "",
    semantic_support_status: str = "",
) -> float:
    return _search_frame_match_score(
        frame,
        terms,
        query=query,
        support_status=support_status,
        semantic_support_status=semantic_support_status,
    )


def _matched_frame_fields(frame: Dict[str, Any], terms: List[str]) -> Dict[str, List[str]]:
    return _search_matched_frame_fields(frame, terms)


def _merge_count_dicts(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return _verifier_merge_count_dicts(items)
