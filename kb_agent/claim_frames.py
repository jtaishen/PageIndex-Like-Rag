from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .artifacts import get_artifact, get_doc_card, list_artifacts
from .claim_frame_evidence import (
    evidence_unit_ids_for_claim as _evidence_unit_ids_for_claim,
    evidence_units_from_artifacts,
    source_artifact_ids,
    unit_by_id as _unit_by_id,
    unit_by_node_id as _unit_by_node_id,
    unit_by_source_id as _unit_by_source_id,
    unit_ids_for_node as _unit_ids_for_node,
    unit_ids_for_source as _unit_ids_for_source,
)
from .claim_frame_quality import (
    LOW_FRAME_QUALITY_SCORE,
    MIN_FRAME_QUALITY_SCORE,
    existing_frame_quality,
    frame_fallback_reason,
    frame_quality,
    frame_quality_summary,
    frame_selection_reasons,
    query_allows_weak_frames,
    should_skip_low_quality_frame,
    top_frame_noise_reasons,
    top_noise_reasons,
)
from .llm import LLMError, generate_json_object, llm_payload_metadata
from .text_quality import short_research_text
from .utils import compact_whitespace, excerpt as _excerpt, stable_id, unique_strings as _unique_strings, write_json


EVIDENCE_UNITS_ARTIFACT = "evidence_units.json"
CLAIM_FRAMES_ARTIFACT = "claim_frames.json"
CLAIM_FRAME_VERIFIER_ARTIFACT = "claim_frame_verifier.json"

EVIDENCE_UNIT_SCHEMA = "evidence_units.v1"
CLAIM_FRAME_SCHEMA = "claim_frames.v1"
CLAIM_FRAME_VERIFIER_SCHEMA = "claim_frame_verifier.v1"
CLAIM_FRAME_VERIFIER_RESULT_SCHEMA = "claim_frame_verifier_result.v1"

MAX_CLAIM_CHARS = 240
LLM_ENHANCE_FRAME_LIMIT = 24
LLM_ENHANCE_UNIT_LIMIT = 60

CLAIM_TYPE_MAP = {
    "gap": "problem",
    "problem": "problem",
    "contribution": "method",
    "method": "method",
    "approach": "method",
    "result": "result",
    "experiment": "result",
    "limitation": "limitation",
    "limit": "limitation",
    "citation": "citation",
    "cites": "citation",
    "reference": "citation",
}

METRIC_TERMS = (
    "任务完成率",
    "任务成功率",
    "成功率",
    "准确率",
    "召回率",
    "响应时间",
    "负载均衡",
    "通信开销",
    "计算开销",
    "鲁棒性",
    "score",
    "accuracy",
    "recall",
    "latency",
)
SEMANTIC_SUPPORT_STATUSES = {
    "semantically_supported",
    "partially_supported",
    "related_only",
    "contradicted",
    "insufficient_evidence",
    "not_checked",
}
CITATION_RISKS = {
    "safe",
    "needs_qualification",
    "needs_more_evidence",
    "conflicting_evidence",
    "not_checked",
}
SEMANTIC_STOP_TERMS = {
    "本文",
    "研究",
    "提出",
    "方法",
    "模型",
    "系统",
    "算法",
    "实验",
    "结果",
    "表明",
    "采用",
    "通过",
    "基于",
    "实现",
    "问题",
    "相关",
    "进行",
    "具有",
    "paper",
    "method",
    "model",
    "system",
    "result",
    "study",
}
GENERIC_RELATED_TERMS = {
    "任务",
    "务规",
    "规划",
    "务机",
    "器人",
    "智能",
    "能体",
    "任务规划",
    "服务",
    "机器人",
    "服务机器人",
    "智能体",
    "多智能体",
    "论文",
    "研究",
    "问题",
    "场景",
    "系统",
}
CONFLICT_TERMS = (
    "未验证",
    "未能",
    "没有",
    "无法",
    "不能",
    "不支持",
    "不足",
    "失败",
    "相反",
    "缺乏",
    "无显著",
    "受限",
    "局限",
    "风险",
    "not verified",
    "not support",
    "unsupported",
    "insufficient",
    "fail",
    "failed",
    "failure",
    "lack",
    "limited",
    "contradict",
)


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
    frames = _frames_from_claims(doc_id, version_id, claims, unit_by_node, unit_by_id, unit_by_source_id)
    frames.extend(_frames_from_innovations(doc_id, version_id, innovation, unit_by_node, unit_by_id, unit_by_source_id, start_index=len(frames)))
    frames.extend(_frames_from_table_summaries(doc_id, version_id, table_summaries, unit_by_node, unit_by_source_id, start_index=len(frames)))
    frames.extend(_frames_from_citations(doc_id, version_id, card, citation_map, unit_by_node, unit_by_source_id, start_index=len(frames)))
    frames = _dedupe_frames(frames)
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
            verifier_item = verified_by_id.get(str(frame.get("frame_id") or "")) or {}
            support_status = str(verifier_item.get("support_status") or frame.get("support_status") or "")
            support_status = _normalize_support_status(support_status)
            semantic_support_status = _normalize_semantic_support_status(
                str(verifier_item.get("semantic_support_status") or frame.get("semantic_support_status") or "")
            )
            citation_risk = _normalize_citation_risk(str(verifier_item.get("citation_risk") or frame.get("citation_risk") or ""))
            score = _frame_match_score(
                frame,
                terms,
                query=query,
                support_status=support_status,
                semantic_support_status=semantic_support_status,
            )
            if score <= 0:
                continue
            items.append(
                {
                    "frame_id": frame.get("frame_id"),
                    "doc_id": frame.get("doc_id"),
                    "claim_type": frame.get("claim_type"),
                    "short_claim": frame.get("short_claim"),
                    "trace_status": verifier_item.get("trace_status") or frame.get("trace_status") or "",
                    "support_status": support_status,
                    "support_reason": verifier_item.get("support_reason") or frame.get("support_reason", ""),
                    "semantic_support_status": semantic_support_status,
                    "semantic_support_score": verifier_item.get("semantic_support_score", frame.get("semantic_support_score", 0.0)),
                    "semantic_support_reason": verifier_item.get("semantic_support_reason") or frame.get("semantic_support_reason", ""),
                    "citation_risk": citation_risk,
                    "primary_evidence_unit_ids": verifier_item.get("primary_evidence_unit_ids") or frame.get("primary_evidence_unit_ids") or [],
                    "weak_evidence_unit_ids": verifier_item.get("weak_evidence_unit_ids") or frame.get("weak_evidence_unit_ids") or [],
                    "contradictory_evidence_unit_ids": verifier_item.get("contradictory_evidence_unit_ids")
                    or frame.get("contradictory_evidence_unit_ids")
                    or [],
                    "evidence_unit_ids": frame.get("evidence_unit_ids") or [],
                    "source_claim_ids": frame.get("source_claim_ids") or [],
                    "confidence": frame.get("confidence"),
                    "quality_score": frame.get("quality_score", 0.0),
                    "frame_quality": frame.get("frame_quality", ""),
                    "noise_reasons": frame.get("noise_reasons") or [],
                    "score": round(score, 3),
                    "matched_fields": _matched_frame_fields(frame, terms),
                    "selection_reasons": _unique_strings(
                        [
                            *frame_selection_reasons(frame, support_status),
                            f"semantic:{semantic_support_status}",
                            f"citation_risk:{citation_risk}",
                        ]
                    ),
                    "fallback_reason": frame_fallback_reason(frame, support_status, query),
                    "warnings": _unique_strings([*(frame.get("warnings") or []), *(verifier_item.get("warnings") or [])]),
                }
            )
    ranked = sorted(items, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("frame_id") or "")))[: max(1, top_k)]
    return {
        "schema": "claim_frame_search.v1",
        "query": query,
        "doc_ids": doc_ids or [],
        "available": bool(ranked),
        "count": len(ranked),
        "items": ranked,
        "warnings": warnings,
    }


def _frames_from_claims(
    doc_id: str,
    version_id: str,
    claims_payload: Any,
    unit_by_node: Dict[str, List[Dict[str, Any]]],
    unit_by_id: Dict[str, Dict[str, Any]],
    unit_by_source_id: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    frames = []
    for index, claim in enumerate((claims_payload or {}).get("claims") or []):
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("text") or claim.get("claim") or "")
        if not text:
            continue
        claim_type = _claim_type(str(claim.get("claim_type") or claim.get("type") or ""))
        evidence_unit_ids, binding_warnings = _evidence_unit_ids_for_claim(claim, unit_by_node, unit_by_id, unit_by_source_id)
        frames.append(
            _frame_record(
                doc_id,
                version_id,
                claim_type,
                text,
                evidence_unit_ids,
                source="claim",
                source_claim_ids=[str(claim.get("claim_id") or "")],
                confidence=_confidence(claim.get("confidence"), 0.64),
                index=index,
                binding_warnings=binding_warnings,
            )
        )
    return [item for item in frames if item]


def _frames_from_innovations(
    doc_id: str,
    version_id: str,
    innovation_payload: Any,
    unit_by_node: Dict[str, List[Dict[str, Any]]],
    unit_by_id: Dict[str, Dict[str, Any]],
    unit_by_source_id: Dict[str, List[Dict[str, Any]]],
    *,
    start_index: int,
) -> List[Dict[str, Any]]:
    frames = []
    for offset, item in enumerate((innovation_payload or {}).get("items") or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("claim") or item.get("title") or item.get("approach") or "")
        if not text:
            continue
        claim_type = _claim_type(str(item.get("type") or "contribution"))
        evidence_unit_ids, binding_warnings = _evidence_unit_ids_for_claim(item, unit_by_node, unit_by_id, unit_by_source_id)
        frames.append(
            _frame_record(
                doc_id,
                version_id,
                claim_type,
                text,
                evidence_unit_ids,
                source="innovation",
                source_claim_ids=[stable_id("innovation", doc_id, offset, text, length=14)],
                confidence=_confidence(item.get("confidence"), 0.58),
                index=start_index + offset,
                problem=str(item.get("problem") or ""),
                method=str(item.get("approach") or ""),
                binding_warnings=binding_warnings,
            )
        )
    return [item for item in frames if item]


def _frames_from_table_summaries(
    doc_id: str,
    version_id: str,
    table_payload: Any,
    unit_by_node: Dict[str, List[Dict[str, Any]]],
    unit_by_source_id: Dict[str, List[Dict[str, Any]]],
    *,
    start_index: int,
) -> List[Dict[str, Any]]:
    frames = []
    for offset, table in enumerate((table_payload or {}).get("table_summaries") or []):
        if not isinstance(table, dict):
            continue
        text = compact_whitespace(f"{table.get('caption') or ''} {table.get('summary') or ''}")
        if not text:
            continue
        node_id = str(table.get("node_id") or table.get("source_node_id") or "")
        source_id = str(table.get("table_id") or table.get("id") or "")
        evidence_unit_ids = _unique_strings([*_unit_ids_for_node(node_id, unit_by_node), *_unit_ids_for_source(source_id, unit_by_source_id)])
        frames.append(
            _frame_record(
                doc_id,
                version_id,
                "result",
                text,
                evidence_unit_ids,
                source="table_summary",
                source_claim_ids=[str(table.get("table_id") or stable_id("table", doc_id, offset, text, length=14))],
                confidence=_confidence(table.get("confidence"), 0.6),
                index=start_index + offset,
            )
        )
    return [item for item in frames if item]


def _frames_from_citations(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    citation_payload: Any,
    unit_by_node: Dict[str, List[Dict[str, Any]]],
    unit_by_source_id: Dict[str, List[Dict[str, Any]]],
    *,
    start_index: int,
) -> List[Dict[str, Any]]:
    frames = []
    title = str(card.get("title") or doc_id)
    raw_items = []
    for name in ("relations", "in_text_citations"):
        for item in (citation_payload or {}).get(name) or []:
            if isinstance(item, dict):
                raw_items.append(item)
    if not raw_items:
        for item in (citation_payload or {}).get("references") or []:
            if isinstance(item, dict):
                raw_items.append(item)
    for offset, item in enumerate(raw_items[:20]):
        ref_id = str(item.get("ref_id") or item.get("reference_id") or item.get("id") or "")
        raw = str(item.get("raw") or item.get("title") or ref_id)
        if not ref_id and not raw:
            continue
        node_id = str(item.get("node_id") or "")
        source_ids = [ref_id, str(item.get("citation_id") or ""), str(item.get("source_id") or "")]
        evidence_unit_ids = _unique_strings(
            [
                *_unit_ids_for_node(node_id, unit_by_node),
                *[unit_id for source_id in source_ids for unit_id in _unit_ids_for_source(source_id, unit_by_source_id)],
            ]
        )
        text = f"{title} 引用了 {ref_id or raw}。" if node_id else f"{title} 的参考文献包含 {ref_id or raw}。"
        frames.append(
            _frame_record(
                doc_id,
                version_id,
                "citation",
                text,
                evidence_unit_ids,
                source="citation_map",
                source_claim_ids=[ref_id or stable_id("reference", doc_id, raw, length=14)],
                confidence=_confidence(item.get("confidence"), 0.62 if node_id else 0.5),
                index=start_index + offset,
            )
        )
    return [item for item in frames if item]


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
    short_claim = short_research_text(text, MAX_CLAIM_CHARS)
    if not short_claim:
        return {}
    inferred = _infer_frame_fields(short_claim, claim_type)
    if problem:
        inferred["problem"] = short_research_text(problem, 180)
    if method:
        inferred["method"] = short_research_text(method, 180)
    clean_evidence = _unique_strings(evidence_unit_ids)
    quality = frame_quality(short_claim, clean_evidence, source=source, claim_type=claim_type, confidence=confidence)
    if should_skip_low_quality_frame(source, clean_evidence, quality):
        return {}
    warnings = [] if clean_evidence else ["missing_evidence_unit"]
    warnings.extend(binding_warnings or [])
    warnings.extend(f"noise:{reason}" for reason in quality["noise_reasons"])
    if quality["quality_score"] < LOW_FRAME_QUALITY_SCORE:
        warnings.append("low_quality_frame")
    trace_status = "partial" if clean_evidence and binding_warnings else ("verified" if clean_evidence else "missing")
    support_status = "structurally_supported" if trace_status == "verified" else ("unchecked" if trace_status == "partial" else "unsupported")
    return {
        "frame_id": stable_id("cf", version_id, claim_type, short_claim, ",".join(clean_evidence), index, length=14),
        "doc_id": doc_id,
        "version_id": version_id,
        "claim_type": claim_type,
        "short_claim": short_claim,
        "problem": inferred["problem"],
        "method": inferred["method"],
        "dataset_or_setting": inferred["dataset_or_setting"],
        "metric_or_signal": inferred["metric_or_signal"],
        "result_or_gain": inferred["result_or_gain"],
        "limitation": inferred["limitation"],
        "evidence_unit_ids": clean_evidence,
        "source_claim_ids": _unique_strings(source_claim_ids),
        "source": source,
        "confidence": round(confidence, 3),
        "trace_status": trace_status,
        "support_status": support_status,
        "support_reason": "evidence_units_bound" if clean_evidence else "missing_evidence_unit",
        "quality_score": quality["quality_score"],
        "frame_quality": quality["frame_quality"],
        "noise_reasons": quality["noise_reasons"],
        "warnings": _unique_strings(warnings),
    }


def _enhance_frames_with_llm(
    card: Dict[str, Any],
    frames: List[Dict[str, Any]],
    units: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    system_prompt = (
        "你是严谨的论文 ClaimFrame 结构化助手。只基于给定短 claim 和 evidence 摘要补全字段，"
        "不要新增 frame，不要输出长正文。必须返回 JSON object。"
    )
    payload_frames = [
        {
            "frame_id": frame["frame_id"],
            "claim_type": frame["claim_type"],
            "short_claim": frame["short_claim"],
            "evidence_unit_ids": frame.get("evidence_unit_ids") or [],
        }
        for frame in frames[:LLM_ENHANCE_FRAME_LIMIT]
    ]
    payload_units = [
        {
            "unit_id": unit.get("unit_id"),
            "unit_type": unit.get("unit_type"),
            "summary": unit.get("summary"),
            "keywords": unit.get("keywords") or [],
        }
        for unit in units[:LLM_ENHANCE_UNIT_LIMIT]
    ]
    user_prompt = "\n".join(
        [
            f"title: {card.get('title') or ''}",
            f"description: {card.get('description') or card.get('summary') or ''}",
            "返回格式：",
            '{"frames":[{"frame_id":"","problem":"","method":"","dataset_or_setting":"",'
            '"metric_or_signal":"","result_or_gain":"","limitation":"","confidence":0.0,"warnings":[]}]}',
            "claim_frames:",
            json.dumps(payload_frames, ensure_ascii=False),
            "evidence_units:",
            json.dumps(payload_units, ensure_ascii=False),
        ]
    )
    payload = generate_json_object(system_prompt, user_prompt, operation="claim_frames", stage="enhance")
    by_id = {str(frame.get("frame_id") or ""): frame for frame in frames}
    for raw in payload.get("frames") or []:
        if not isinstance(raw, dict):
            continue
        frame = by_id.get(str(raw.get("frame_id") or ""))
        if not frame:
            continue
        for field in ("problem", "method", "dataset_or_setting", "metric_or_signal", "result_or_gain", "limitation"):
            value = short_research_text(raw.get(field), 180)
            if value:
                frame[field] = value
        if raw.get("confidence") is not None:
            frame["confidence"] = round(max(0.0, min(1.0, _confidence(raw.get("confidence"), frame.get("confidence", 0.6)))), 3)
        quality = frame_quality(
            str(frame.get("short_claim") or ""),
            [str(item) for item in frame.get("evidence_unit_ids") or []],
            source=str(frame.get("source") or ""),
            claim_type=str(frame.get("claim_type") or ""),
            confidence=_confidence(frame.get("confidence"), 0.6),
        )
        frame["quality_score"] = quality["quality_score"]
        frame["frame_quality"] = quality["frame_quality"]
        frame["noise_reasons"] = quality["noise_reasons"]
        frame["warnings"] = _unique_strings([*(frame.get("warnings") or []), *[str(item) for item in raw.get("warnings") or []]])
    metadata = llm_payload_metadata(payload)
    truncated_frames = len(frames) > LLM_ENHANCE_FRAME_LIMIT
    truncated_units = len(units) > LLM_ENHANCE_UNIT_LIMIT
    enhancement_warnings = []
    if truncated_frames:
        enhancement_warnings.append("llm_frame_enhancement_truncated")
    if truncated_units:
        enhancement_warnings.append("llm_unit_context_truncated")
    metadata["llm_enhancement"] = {
        "used": True,
        "enhanced_frame_limit": LLM_ENHANCE_FRAME_LIMIT,
        "total_frame_count": len(frames),
        "context_unit_limit": LLM_ENHANCE_UNIT_LIMIT,
        "total_unit_count": len(units),
        "truncated": truncated_frames or truncated_units,
    }
    metadata["enhancement_warnings"] = enhancement_warnings
    return list(by_id.values()), metadata


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
    units = evidence_units.get("units") or []
    unit_by_id = {str(unit.get("unit_id") or ""): unit for unit in units if isinstance(unit, dict)}
    citation_map = _artifact_content(db_path, doc_id, "citation_map.json", {})
    items = []
    verified_count = 0
    unsupported_count = 0
    trace_status_counts: Dict[str, int] = {}
    support_status_counts: Dict[str, int] = {}
    semantic_status_counts: Dict[str, int] = {}
    citation_risk_counts: Dict[str, int] = {}
    semantic_verified_count = 0
    partial_supported_count = 0
    related_only_count = 0
    contradicted_count = 0
    insufficient_evidence_count = 0
    low_confidence_count = 0
    low_quality_count = 0
    noisy_count = 0
    ignored_noise_count = 0
    missing_unit_count = 0
    missing_node_count = 0
    missing_source_count = 0
    citation_gap_count = 0
    for frame in claim_frames.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        warnings = list(frame.get("warnings") or [])
        evidence_unit_ids = _unique_strings(frame.get("evidence_unit_ids") or [])
        existing_units = [unit_by_id[unit_id] for unit_id in evidence_unit_ids if unit_id in unit_by_id]
        missing_units = [unit_id for unit_id in evidence_unit_ids if unit_id not in unit_by_id]
        computed_quality = existing_frame_quality(frame, evidence_unit_ids)
        quality_score = _confidence(computed_quality.get("quality_score"), 0.6)
        noise_reasons = _unique_strings(computed_quality.get("noise_reasons") or [])
        if quality_score < LOW_FRAME_QUALITY_SCORE:
            low_quality_count += 1
            warnings.append("low_quality_frame")
        if noise_reasons:
            noisy_count += 1
            warnings.extend(f"noise:{reason}" for reason in noise_reasons)
        if missing_units:
            missing_unit_count += len(missing_units)
            warnings.append("missing_evidence_unit")
        missing_nodes = [
            str(unit.get("node_id") or "")
            for unit in existing_units
            if unit.get("node_id") and str(unit.get("node_id")) not in node_ids
        ]
        if missing_nodes:
            missing_node_count += len(missing_nodes)
            warnings.append("missing_evidence_node")
        missing_sources = [
            str(unit.get("source_id") or "")
            for unit in existing_units
            if unit.get("source_kind") not in {"", "node"} and unit.get("source_id") and str(unit.get("source_id")) not in source_ids
        ]
        if missing_sources:
            missing_source_count += len(missing_sources)
            warnings.append("missing_evidence_source")
        confidence = _confidence(frame.get("confidence"), 0.0)
        if confidence < 0.5:
            low_confidence_count += 1
            warnings.append("low_confidence_frame")
        if frame.get("claim_type") == "citation" and not ((citation_map or {}).get("references") or (citation_map or {}).get("relations")):
            citation_gap_count += 1
            warnings.append("citation_frame_without_citation_map")
        if existing_units and not missing_units and not missing_nodes and not missing_sources:
            trace_status = "verified"
            support_status = "structurally_supported"
            support_reason = "evidence_units_verified"
            verified_count += 1
        elif evidence_unit_ids:
            trace_status = "partial"
            support_status = "unchecked"
            support_reason = "partial_evidence_unit_match"
        else:
            trace_status = "missing"
            support_status = "unsupported"
            support_reason = "no_evidence_unit_found"
            unsupported_count += 1
            warnings.append("unsupported_frame")
        if trace_status == "missing" and (quality_score < MIN_FRAME_QUALITY_SCORE or noise_reasons):
            support_reason = "low_quality_or_noise_without_evidence"
            ignored_noise_count += 1
        semantic = _semantic_support_for_frame(frame, existing_units)
        semantic_status = str(semantic["semantic_support_status"])
        citation_risk = str(semantic["citation_risk"])
        if semantic_status == "semantically_supported":
            semantic_verified_count += 1
        elif semantic_status == "partially_supported":
            partial_supported_count += 1
        elif semantic_status == "related_only":
            related_only_count += 1
        elif semantic_status == "contradicted":
            contradicted_count += 1
            warnings.append("semantic_contradiction")
        elif semantic_status == "insufficient_evidence":
            insufficient_evidence_count += 1
            warnings.append("semantic_support_insufficient")
        trace_status_counts[trace_status] = trace_status_counts.get(trace_status, 0) + 1
        support_status_counts[support_status] = support_status_counts.get(support_status, 0) + 1
        semantic_status_counts[semantic_status] = semantic_status_counts.get(semantic_status, 0) + 1
        citation_risk_counts[citation_risk] = citation_risk_counts.get(citation_risk, 0) + 1
        items.append(
            {
                "frame_id": frame.get("frame_id"),
                "claim_type": frame.get("claim_type"),
                "trace_status": trace_status,
                "support_status": support_status,
                "support_reason": support_reason,
                "semantic_support_status": semantic_status,
                "semantic_support_score": semantic["semantic_support_score"],
                "semantic_support_reason": semantic["semantic_support_reason"],
                "primary_evidence_unit_ids": semantic["primary_evidence_unit_ids"],
                "weak_evidence_unit_ids": semantic["weak_evidence_unit_ids"],
                "contradictory_evidence_unit_ids": semantic["contradictory_evidence_unit_ids"],
                "citation_risk": citation_risk,
                "evidence_unit_count": len(existing_units),
                "missing_evidence_unit_ids": missing_units,
                "missing_node_ids": _unique_strings(missing_nodes),
                "missing_source_ids": _unique_strings(missing_sources),
                "confidence": frame.get("confidence"),
                "quality_score": round(quality_score, 3),
                "frame_quality": computed_quality.get("frame_quality", ""),
                "noise_reasons": noise_reasons,
                "warnings": _unique_strings(warnings),
            }
        )
    frame_count = len(items)
    warnings = []
    if unsupported_count:
        warnings.append("unsupported_claim_frames")
    if missing_unit_count:
        warnings.append("missing_evidence_units")
    if low_confidence_count:
        warnings.append("low_confidence_claim_frames")
    if low_quality_count:
        warnings.append("low_quality_claim_frames")
    if ignored_noise_count:
        warnings.append("ignored_noise_claim_frames")
    if citation_gap_count:
        warnings.append("citation_frame_gaps")
    if contradicted_count:
        warnings.append("semantic_contradicted_claim_frames")
    if insufficient_evidence_count:
        warnings.append("semantic_insufficient_evidence")
    return {
        "schema": CLAIM_FRAME_VERIFIER_SCHEMA,
        "doc_id": doc_id,
        "version_id": claim_frames.get("version_id") or evidence_units.get("version_id") or "",
        "status": "passed" if frame_count and not warnings else ("partial" if frame_count else "skipped"),
        "frame_count": frame_count,
        "verified_frame_count": verified_count,
        "verified_frame_rate": round(verified_count / max(1, frame_count), 4),
        "unsupported_frame_count": unsupported_count,
        "trace_status_counts": trace_status_counts,
        "support_status_counts": support_status_counts,
        "semantic_support_status_counts": semantic_status_counts,
        "semantic_verified_frame_count": semantic_verified_count,
        "semantic_supported_frame_rate": round(semantic_verified_count / max(1, frame_count), 4),
        "partial_supported_frame_count": partial_supported_count,
        "related_only_frame_count": related_only_count,
        "contradicted_frame_count": contradicted_count,
        "insufficient_evidence_frame_count": insufficient_evidence_count,
        "citation_risk_counts": citation_risk_counts,
        "low_confidence_frame_count": low_confidence_count,
        "low_quality_frame_count": low_quality_count,
        "noisy_frame_count": noisy_count,
        "ignored_noise_frame_count": ignored_noise_count,
        "missing_evidence_unit_count": missing_unit_count,
        "missing_node_count": missing_node_count,
        "missing_source_count": missing_source_count,
        "citation_gap_count": citation_gap_count,
        "quality_summary": frame_quality_summary(claim_frames.get("frames") or []),
        "top_frame_noise_reasons": top_frame_noise_reasons(claim_frames.get("frames") or []),
        "items": items,
        "warnings": warnings,
        "created_at": time.time(),
    }


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
    return {
        "schema": CLAIM_FRAME_SCHEMA,
        "status": "extracted" if frames else "partial",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(frames),
        "evidence_unit_count": evidence_unit_count,
        "claim_type_counts": _count_by_field(frames, "claim_type"),
        "trace_status_counts": _count_by_field(frames, "trace_status"),
        "support_status_counts": _count_by_field(frames, "support_status"),
        "quality_summary": frame_quality_summary(frames),
        "noisy_frame_count": sum(1 for frame in frames if frame.get("noise_reasons")),
        "low_quality_frame_count": sum(1 for frame in frames if _confidence(frame.get("quality_score"), 0.6) < LOW_FRAME_QUALITY_SCORE),
        "top_frame_noise_reasons": top_frame_noise_reasons(frames),
        "frames": frames,
        "llm_used": llm_used,
        "llm_error": llm_error,
        "llm_enhancement": llm_metadata.get("llm_enhancement") or {"used": False, "truncated": False},
        "llm_metadata": llm_metadata,
        "warnings": _unique_strings([*warnings, *[warning for frame in frames for warning in frame.get("warnings", [])]]),
        "created_at": time.time(),
    }


def _sync_claim_frames_with_verifier(payload: Dict[str, Any], verifier: Dict[str, Any]) -> None:
    items_by_id = {str(item.get("frame_id") or ""): item for item in verifier.get("items") or [] if isinstance(item, dict)}
    frames = [frame for frame in payload.get("frames") or [] if isinstance(frame, dict)]
    for frame in frames:
        item = items_by_id.get(str(frame.get("frame_id") or ""))
        if not item:
            continue
        for field in (
            "trace_status",
            "support_status",
            "support_reason",
            "semantic_support_status",
            "semantic_support_score",
            "semantic_support_reason",
            "primary_evidence_unit_ids",
            "weak_evidence_unit_ids",
            "contradictory_evidence_unit_ids",
            "citation_risk",
            "quality_score",
            "frame_quality",
            "noise_reasons",
        ):
            if field in item:
                frame[field] = item[field]
        frame["warnings"] = _unique_strings([*(frame.get("warnings") or []), *(item.get("warnings") or [])])
    payload["trace_status_counts"] = _count_by_field(frames, "trace_status")
    payload["support_status_counts"] = _count_by_field(frames, "support_status")
    payload["semantic_support_status_counts"] = _count_by_field(frames, "semantic_support_status")
    payload["semantic_verified_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "semantically_supported")
    payload["semantic_supported_frame_rate"] = round(payload["semantic_verified_frame_count"] / max(1, len(frames)), 4)
    payload["partial_supported_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "partially_supported")
    payload["related_only_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "related_only")
    payload["contradicted_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "contradicted")
    payload["insufficient_evidence_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "insufficient_evidence")
    payload["citation_risk_counts"] = _count_by_field(frames, "citation_risk")
    payload["quality_summary"] = frame_quality_summary(frames)
    payload["noisy_frame_count"] = sum(1 for frame in frames if frame.get("noise_reasons"))
    payload["low_quality_frame_count"] = sum(1 for frame in frames if _confidence(frame.get("quality_score"), 0.6) < LOW_FRAME_QUALITY_SCORE)
    payload["top_frame_noise_reasons"] = top_frame_noise_reasons(frames)
    payload["warnings"] = _unique_strings([*(payload.get("warnings") or []), *[warning for frame in frames for warning in frame.get("warnings", [])]])


def _semantic_support_for_frame(frame: Dict[str, Any], units: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not frame.get("short_claim"):
        return _semantic_result("not_checked", 0.0, "missing_short_claim", [], [], [])
    if not units:
        return _semantic_result("insufficient_evidence", 0.0, "no_evidence_units", [], [], [])

    claim_text = _frame_semantic_text(frame)
    claim_terms = _semantic_terms(claim_text)
    core_terms = _core_claim_terms(frame, claim_terms)
    if not claim_terms:
        return _semantic_result("not_checked", 0.0, "no_claim_terms", [], [], [])

    unit_scores = []
    contradictory_ids: List[str] = []
    for unit in units:
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            continue
        unit_text = _evidence_unit_text(unit)
        unit_terms = _semantic_terms(unit_text)
        claim_hits = [term for term in claim_terms if term in unit_terms]
        core_hits = [term for term in core_terms if term in unit_terms]
        claim_coverage = len(claim_hits) / max(1, len(claim_terms))
        core_coverage = len(core_hits) / max(1, len(core_terms))
        confidence = _confidence(unit.get("confidence"), 0.6)
        score = min(1.0, 0.55 * claim_coverage + 0.35 * core_coverage + 0.10 * confidence)
        conflict = _evidence_conflicts_with_claim(frame, unit_text, claim_hits + core_hits)
        if conflict:
            contradictory_ids.append(unit_id)
        unit_scores.append(
            {
                "unit_id": unit_id,
                "score": score,
                "core_coverage": core_coverage,
                "claim_hits": claim_hits,
                "core_hits": core_hits,
                "conflict": conflict,
            }
        )

    if contradictory_ids:
        best_conflict = max((item["score"] for item in unit_scores if item["conflict"]), default=0.2)
        return _semantic_result("contradicted", best_conflict, "conflicting_evidence_signal", [], [], contradictory_ids)

    primary = [
        str(item["unit_id"])
        for item in unit_scores
        if float(item["score"]) >= 0.5 and (item["core_hits"] or len(item["claim_hits"]) >= 3)
    ]
    weak = [
        str(item["unit_id"])
        for item in unit_scores
        if item["unit_id"] not in primary and (float(item["score"]) >= 0.18 or item["claim_hits"] or item["core_hits"])
    ]
    best_score = max((float(item["score"]) for item in unit_scores), default=0.0)
    best_core_coverage = max((float(item["core_coverage"]) for item in unit_scores), default=0.0)
    has_specific_core_hit = any(_specific_semantic_hits(item["core_hits"]) for item in unit_scores)
    strong_core_hit = any(len(_specific_semantic_hits(item["core_hits"])) >= 2 for item in unit_scores)
    if primary and best_score >= 0.58 and best_core_coverage >= 0.65 and strong_core_hit:
        return _semantic_result("semantically_supported", best_score, "primary_evidence_overlap", primary, weak, [])
    if primary or has_specific_core_hit:
        return _semantic_result("partially_supported", best_score, "partial_core_overlap", primary, weak, [])
    if weak:
        return _semantic_result("related_only", best_score, "topic_related_only", [], weak, [])
    return _semantic_result("insufficient_evidence", 0.0, "no_semantic_evidence_match", [], [], [])


def _semantic_result(
    status: str,
    score: float,
    reason: str,
    primary_ids: List[str],
    weak_ids: List[str],
    contradictory_ids: List[str],
) -> Dict[str, Any]:
    normalized = _normalize_semantic_support_status(status)
    return {
        "semantic_support_status": normalized,
        "semantic_support_score": round(max(0.0, min(1.0, float(score or 0.0))), 3),
        "semantic_support_reason": reason,
        "primary_evidence_unit_ids": _unique_strings(primary_ids)[:8],
        "weak_evidence_unit_ids": _unique_strings(weak_ids)[:8],
        "contradictory_evidence_unit_ids": _unique_strings(contradictory_ids)[:8],
        "citation_risk": _citation_risk_for_semantic_status(normalized),
    }


def _frame_semantic_text(frame: Dict[str, Any]) -> str:
    return compact_whitespace(
        " ".join(
            str(frame.get(field) or "")
            for field in (
                "short_claim",
                "problem",
                "method",
                "dataset_or_setting",
                "metric_or_signal",
                "result_or_gain",
                "limitation",
            )
        )
    )


def _evidence_unit_text(unit: Dict[str, Any]) -> str:
    return compact_whitespace(
        " ".join(
            [
                str(unit.get("summary") or ""),
                str(unit.get("text_excerpt") or ""),
                " ".join(str(item) for item in unit.get("keywords") or []),
                str(unit.get("node_path") or ""),
                str(unit.get("unit_type") or ""),
                str(unit.get("source_kind") or ""),
            ]
        )
    )


def _core_claim_terms(frame: Dict[str, Any], fallback_terms: List[str]) -> List[str]:
    claim_type = str(frame.get("claim_type") or "")
    fields = ["method"] if claim_type == "method" else []
    if claim_type == "result":
        fields = ["result_or_gain", "metric_or_signal", "dataset_or_setting"]
    elif claim_type == "limitation":
        fields = ["limitation", "problem"]
    elif claim_type == "problem":
        fields = ["problem"]
    elif claim_type == "citation":
        fields = ["short_claim"]
    terms = _semantic_terms(" ".join(str(frame.get(field) or "") for field in fields))
    return terms or fallback_terms[:8]


def _semantic_terms(text: str) -> List[str]:
    terms = []
    for term in _query_terms(compact_whitespace(text)):
        normalized = term.lower()
        if normalized in SEMANTIC_STOP_TERMS or term in SEMANTIC_STOP_TERMS:
            continue
        if len(term) < 2:
            continue
        terms.append(term)
    return _unique_strings(terms)[:24]


def _specific_semantic_hits(hits: List[str]) -> List[str]:
    return [term for term in hits if term not in GENERIC_RELATED_TERMS and term.lower() not in GENERIC_RELATED_TERMS]


def _evidence_conflicts_with_claim(frame: Dict[str, Any], evidence_text: str, matched_terms: List[str]) -> bool:
    if not matched_terms:
        return False
    if str(frame.get("claim_type") or "") == "limitation":
        return False
    claim_text = _frame_semantic_text(frame)
    if _has_conflict_signal(claim_text):
        return False
    return _has_conflict_signal(evidence_text)


def _has_conflict_signal(text: str) -> bool:
    haystack = compact_whitespace(text).lower()
    return any(term.lower() in haystack for term in CONFLICT_TERMS)


def _normalize_semantic_support_status(value: str) -> str:
    return value if value in SEMANTIC_SUPPORT_STATUSES else "not_checked"


def _normalize_citation_risk(value: str) -> str:
    return value if value in CITATION_RISKS else "not_checked"


def _citation_risk_for_semantic_status(status: str) -> str:
    if status == "semantically_supported":
        return "safe"
    if status == "partially_supported":
        return "needs_qualification"
    if status in {"related_only", "insufficient_evidence"}:
        return "needs_more_evidence"
    if status == "contradicted":
        return "conflicting_evidence"
    return "not_checked"


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


def _claim_type(raw: str) -> str:
    normalized = compact_whitespace(raw).lower()
    if normalized in CLAIM_TYPE_MAP:
        return CLAIM_TYPE_MAP[normalized]
    if any(token in normalized for token in ("limit", "局限", "不足")):
        return "limitation"
    if any(token in normalized for token in ("result", "实验", "结果", "提升")):
        return "result"
    if any(token in normalized for token in ("method", "方法", "模型", "框架", "算法")):
        return "method"
    if any(token in normalized for token in ("problem", "gap", "问题", "挑战")):
        return "problem"
    return "claim"


def _infer_frame_fields(text: str, claim_type: str) -> Dict[str, str]:
    result = {
        "problem": "",
        "method": "",
        "dataset_or_setting": "",
        "metric_or_signal": "",
        "result_or_gain": "",
        "limitation": "",
    }
    if claim_type == "problem" or any(token in text for token in ("问题", "挑战", "不足", "瓶颈", "解决")):
        result["problem"] = _excerpt(text, 180)
    if claim_type == "method" or any(token in text for token in ("提出", "方法", "算法", "模型", "框架", "机制")):
        result["method"] = _excerpt(text, 180)
    if claim_type == "result" or any(token in text for token in ("实验", "结果", "提升", "优于", "降低", "验证")):
        result["result_or_gain"] = _excerpt(text, 180)
    if claim_type == "limitation" or any(token in text for token in ("局限", "不足", "限制", "未来工作", "展望")):
        result["limitation"] = _excerpt(text, 180)
    setting_terms = [term for term in ("数据集", "场景", "环境", "家庭", "仿真", "真实", "多智能体", "服务机器人") if term in text]
    if setting_terms:
        result["dataset_or_setting"] = "、".join(setting_terms[:6])
    metric_terms = [term for term in METRIC_TERMS if term.lower() in text.lower()]
    if metric_terms:
        result["metric_or_signal"] = "、".join(metric_terms[:6])
    return result


def _confidence(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_support_status(value: str) -> str:
    if value == "supported":
        return "structurally_supported"
    if value == "partial":
        return "unchecked"
    return value


def _dedupe_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for frame in frames:
        key = (
            str(frame.get("claim_type") or ""),
            compact_whitespace(str(frame.get("short_claim") or "")),
            tuple(frame.get("evidence_unit_ids") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(frame)
    return result


def _count_by_field(items: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(field) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _query_terms(query: str) -> List[str]:
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    return _unique_strings(terms)[:16]


def _frame_match_score(
    frame: Dict[str, Any],
    terms: List[str],
    *,
    query: str = "",
    support_status: str = "",
    semantic_support_status: str = "",
) -> float:
    quality_score = _confidence(frame.get("quality_score"), 0.6)
    status = _normalize_support_status(support_status or str(frame.get("support_status") or ""))
    semantic_status = _normalize_semantic_support_status(semantic_support_status or str(frame.get("semantic_support_status") or ""))
    if not query_allows_weak_frames(query) and (quality_score < MIN_FRAME_QUALITY_SCORE or status == "unsupported"):
        return 0.0
    text_by_field = {
        "short_claim": str(frame.get("short_claim") or ""),
        "problem": str(frame.get("problem") or ""),
        "method": str(frame.get("method") or ""),
        "dataset_or_setting": str(frame.get("dataset_or_setting") or ""),
        "metric_or_signal": str(frame.get("metric_or_signal") or ""),
        "result_or_gain": str(frame.get("result_or_gain") or ""),
        "limitation": str(frame.get("limitation") or ""),
    }
    weights = {
        "short_claim": 2.0,
        "problem": 1.2,
        "method": 1.5,
        "dataset_or_setting": 1.0,
        "metric_or_signal": 1.0,
        "result_or_gain": 1.3,
        "limitation": 1.2,
    }
    score = 0.0
    for field, text in text_by_field.items():
        haystack = text.lower()
        hits = sum(1 for term in terms if term.lower() in haystack)
        score += weights[field] * hits
    score += 0.2 * len(frame.get("evidence_unit_ids") or [])
    score += 0.3 * _confidence(frame.get("confidence"), 0.0)
    score += 0.5 * quality_score
    if status == "structurally_supported":
        score += 0.4
    elif status == "unchecked":
        score += 0.15
    elif status == "unsupported":
        score -= 1.2 if not query_allows_weak_frames(query) else 0.3
    if semantic_status == "semantically_supported":
        score += 0.6
    elif semantic_status == "partially_supported":
        score += 0.25
    elif semantic_status == "related_only":
        score -= 0.5 if not query_allows_weak_frames(query) else 0.15
    elif semantic_status == "insufficient_evidence":
        score -= 1.0 if not query_allows_weak_frames(query) else 0.3
    elif semantic_status == "contradicted":
        score -= 2.0 if not query_allows_weak_frames(query) else 0.5
    if frame.get("noise_reasons"):
        score -= 0.3 * len(frame.get("noise_reasons") or [])
    return score


def _matched_frame_fields(frame: Dict[str, Any], terms: List[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for field in ("short_claim", "problem", "method", "dataset_or_setting", "metric_or_signal", "result_or_gain", "limitation"):
        text = str(frame.get(field) or "").lower()
        hits = [term for term in terms if term.lower() in text]
        if hits:
            result[field] = hits[:6]
    return result


def _verifier_totals(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    frame_count = sum(int(doc.get("frame_count") or 0) for doc in documents)
    verified = sum(int(doc.get("verified_frame_count") or 0) for doc in documents)
    trace_status_counts = _merge_count_dicts(doc.get("trace_status_counts") or {} for doc in documents)
    support_status_counts = _merge_count_dicts(doc.get("support_status_counts") or {} for doc in documents)
    semantic_status_counts = _merge_count_dicts(doc.get("semantic_support_status_counts") or {} for doc in documents)
    return {
        "frame_count": frame_count,
        "verified_frame_count": verified,
        "verified_frame_rate": round(verified / max(1, frame_count), 4),
        "unsupported_frame_count": sum(int(doc.get("unsupported_frame_count") or 0) for doc in documents),
        "trace_status_counts": trace_status_counts,
        "support_status_counts": support_status_counts,
        "semantic_support_status_counts": semantic_status_counts,
        "semantic_verified_frame_count": sum(int(doc.get("semantic_verified_frame_count") or 0) for doc in documents),
        "semantic_supported_frame_rate": round(
            sum(int(doc.get("semantic_verified_frame_count") or 0) for doc in documents) / max(1, frame_count),
            4,
        ),
        "partial_supported_frame_count": sum(int(doc.get("partial_supported_frame_count") or 0) for doc in documents),
        "related_only_frame_count": sum(int(doc.get("related_only_frame_count") or 0) for doc in documents),
        "contradicted_frame_count": sum(int(doc.get("contradicted_frame_count") or 0) for doc in documents),
        "insufficient_evidence_frame_count": sum(int(doc.get("insufficient_evidence_frame_count") or 0) for doc in documents),
        "citation_risk_counts": _merge_count_dicts(doc.get("citation_risk_counts") or {} for doc in documents),
        "low_confidence_frame_count": sum(int(doc.get("low_confidence_frame_count") or 0) for doc in documents),
        "low_quality_frame_count": sum(int(doc.get("low_quality_frame_count") or 0) for doc in documents),
        "noisy_frame_count": sum(int(doc.get("noisy_frame_count") or 0) for doc in documents),
        "ignored_noise_frame_count": sum(int(doc.get("ignored_noise_frame_count") or 0) for doc in documents),
        "missing_evidence_unit_count": sum(int(doc.get("missing_evidence_unit_count") or 0) for doc in documents),
        "missing_node_count": sum(int(doc.get("missing_node_count") or 0) for doc in documents),
        "missing_source_count": sum(int(doc.get("missing_source_count") or 0) for doc in documents),
        "citation_gap_count": sum(int(doc.get("citation_gap_count") or 0) for doc in documents),
        "top_frame_noise_reasons": top_noise_reasons(documents),
    }


def _merge_count_dicts(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value or 0)
    return counts
