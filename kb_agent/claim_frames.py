from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .artifacts import get_artifact, get_doc_card, list_artifacts
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

MAX_UNIT_EXCERPT_CHARS = 360
MAX_UNIT_SUMMARY_CHARS = 180
MAX_CLAIM_CHARS = 240

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
    units = _evidence_units_from_nodes(doc_id, version_id, nodes)
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
    card = _safe_doc_card(db_path, doc_id)
    claims = _artifact_content(db_path, doc_id, "claims.json", {})
    innovation = _artifact_content(db_path, doc_id, "innovation.json", {})
    table_summaries = _artifact_content(db_path, doc_id, "table_summaries.json", {})
    citation_map = _artifact_content(db_path, doc_id, "citation_map.json", {})
    warnings: List[str] = []
    frames = _frames_from_claims(doc_id, version_id, claims, unit_by_node)
    frames.extend(_frames_from_innovations(doc_id, version_id, innovation, unit_by_node, start_index=len(frames)))
    frames.extend(_frames_from_table_summaries(doc_id, version_id, table_summaries, unit_by_node, start_index=len(frames)))
    frames.extend(_frames_from_citations(doc_id, version_id, card, citation_map, unit_by_node, start_index=len(frames)))
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
    write_json(path, payload)
    verifier = _verify_claim_frames_payload(db_path, doc_id, payload, evidence_units)
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
        "status": "passed" if not warnings and totals["unsupported_frame_count"] == 0 else "needs_review",
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
    except (FileNotFoundError, KeyError, ValueError):
        return {"schema": "claim_frame_summary.v1", "doc_id": doc_id, "available": False}
    items = frames.get("frames") or []
    return {
        "schema": "claim_frame_summary.v1",
        "doc_id": doc_id,
        "available": True,
        "frame_count": len(items),
        "verified_frame_rate": verifier.get("verified_frame_rate", 0.0),
        "unsupported_frame_count": verifier.get("unsupported_frame_count", 0),
        "type_counts": frames.get("claim_type_counts") or {},
        "top_frames": [
            {
                "frame_id": item.get("frame_id"),
                "claim_type": item.get("claim_type"),
                "short_claim": item.get("short_claim"),
                "support_status": item.get("support_status"),
                "quality_score": item.get("quality_score", 0.0),
                "frame_quality": item.get("frame_quality", ""),
                "evidence_unit_ids": item.get("evidence_unit_ids") or [],
                "confidence": item.get("confidence"),
            }
            for item in items[:6]
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
            score = _frame_match_score(frame, terms, query=query, support_status=support_status)
            if score <= 0:
                continue
            items.append(
                {
                    "frame_id": frame.get("frame_id"),
                    "doc_id": frame.get("doc_id"),
                    "claim_type": frame.get("claim_type"),
                    "short_claim": frame.get("short_claim"),
                    "support_status": support_status,
                    "support_reason": verifier_item.get("support_reason") or frame.get("support_reason", ""),
                    "evidence_unit_ids": frame.get("evidence_unit_ids") or [],
                    "source_claim_ids": frame.get("source_claim_ids") or [],
                    "confidence": frame.get("confidence"),
                    "quality_score": frame.get("quality_score", 0.0),
                    "frame_quality": frame.get("frame_quality", ""),
                    "noise_reasons": frame.get("noise_reasons") or [],
                    "score": round(score, 3),
                    "matched_fields": _matched_frame_fields(frame, terms),
                    "selection_reasons": frame_selection_reasons(frame, support_status),
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


def _evidence_units_from_nodes(doc_id: str, version_id: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units = []
    seen = set()
    for node in sorted((item for item in nodes if isinstance(item, dict)), key=lambda item: int(item.get("order_index") or 0)):
        node_id = str(node.get("node_id") or "")
        if not node_id or node_id in seen:
            continue
        kind = str(node.get("kind") or node.get("type") or "paragraph")
        if kind == "document":
            continue
        text = compact_whitespace(str(node.get("text") or node.get("summary") or node.get("heading") or ""))
        if not text:
            continue
        seen.add(node_id)
        unit_type = _unit_type(kind, str(node.get("node_path") or ""), str(node.get("heading") or ""))
        source_kind = _source_kind(unit_type, str(node.get("node_path") or ""), str(node.get("heading") or ""))
        warnings = []
        if not node.get("text"):
            warnings.append("summary_only_unit")
        confidence = 0.78 if node.get("text") else 0.68
        if source_kind in {"table", "figure", "reference"}:
            confidence = max(0.62, confidence - 0.05)
        units.append(
            {
                "unit_id": stable_id("eu", version_id, node_id, unit_type, length=14),
                "doc_id": doc_id,
                "version_id": version_id,
                "node_id": node_id,
                "unit_type": unit_type,
                "node_path": str(node.get("node_path") or ""),
                "page_range": _page_range(node),
                "source_kind": source_kind,
                "text_excerpt": _excerpt(text, MAX_UNIT_EXCERPT_CHARS),
                "summary": _excerpt(str(node.get("summary") or text), MAX_UNIT_SUMMARY_CHARS),
                "keywords": _short_keywords(node.get("keywords") or [], text),
                "confidence": round(confidence, 3),
                "warnings": warnings,
            }
        )
    return units


def _frames_from_claims(
    doc_id: str,
    version_id: str,
    claims_payload: Any,
    unit_by_node: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    frames = []
    for index, claim in enumerate((claims_payload or {}).get("claims") or []):
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("text") or claim.get("claim") or "")
        if not text:
            continue
        claim_type = _claim_type(str(claim.get("claim_type") or claim.get("type") or ""))
        evidence_unit_ids = _evidence_unit_ids_for_claim(claim, unit_by_node)
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
            )
        )
    return [item for item in frames if item]


def _frames_from_innovations(
    doc_id: str,
    version_id: str,
    innovation_payload: Any,
    unit_by_node: Dict[str, List[Dict[str, Any]]],
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
        evidence_unit_ids = []
        for evidence in item.get("evidence") or []:
            if isinstance(evidence, dict):
                evidence_unit_ids.extend(_unit_ids_for_node(str(evidence.get("node_id") or ""), unit_by_node))
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
            )
        )
    return [item for item in frames if item]


def _frames_from_table_summaries(
    doc_id: str,
    version_id: str,
    table_payload: Any,
    unit_by_node: Dict[str, List[Dict[str, Any]]],
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
        frames.append(
            _frame_record(
                doc_id,
                version_id,
                "result",
                text,
                _unit_ids_for_node(node_id, unit_by_node),
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
        text = f"{title} 引用了 {ref_id or raw}。" if node_id else f"{title} 的参考文献包含 {ref_id or raw}。"
        frames.append(
            _frame_record(
                doc_id,
                version_id,
                "citation",
                text,
                _unit_ids_for_node(node_id, unit_by_node),
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
    warnings.extend(f"noise:{reason}" for reason in quality["noise_reasons"])
    if quality["quality_score"] < LOW_FRAME_QUALITY_SCORE:
        warnings.append("low_quality_frame")
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
        "support_status": "supported" if clean_evidence else "unsupported",
        "support_reason": "evidence_units_bound" if clean_evidence else "missing_evidence_unit",
        "quality_score": quality["quality_score"],
        "frame_quality": quality["frame_quality"],
        "noise_reasons": quality["noise_reasons"],
        "warnings": warnings,
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
        for frame in frames[:24]
    ]
    payload_units = [
        {
            "unit_id": unit.get("unit_id"),
            "unit_type": unit.get("unit_type"),
            "summary": unit.get("summary"),
            "keywords": unit.get("keywords") or [],
        }
        for unit in units[:60]
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
    return list(by_id.values()), llm_payload_metadata(payload)


def _verify_claim_frames_payload(
    db_path: Path,
    doc_id: str,
    claim_frames: Dict[str, Any],
    evidence_units: Dict[str, Any],
) -> Dict[str, Any]:
    nodes = _artifact_content(db_path, doc_id, "node_index.jsonl", [])
    node_ids = {str(node.get("node_id") or "") for node in nodes if isinstance(node, dict)}
    units = evidence_units.get("units") or []
    unit_by_id = {str(unit.get("unit_id") or ""): unit for unit in units if isinstance(unit, dict)}
    citation_map = _artifact_content(db_path, doc_id, "citation_map.json", {})
    items = []
    verified_count = 0
    unsupported_count = 0
    low_confidence_count = 0
    low_quality_count = 0
    noisy_count = 0
    ignored_noise_count = 0
    missing_unit_count = 0
    missing_node_count = 0
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
        confidence = _confidence(frame.get("confidence"), 0.0)
        if confidence < 0.5:
            low_confidence_count += 1
            warnings.append("low_confidence_frame")
        if frame.get("claim_type") == "citation" and not ((citation_map or {}).get("references") or (citation_map or {}).get("relations")):
            citation_gap_count += 1
            warnings.append("citation_frame_without_citation_map")
        if existing_units and not missing_units and not missing_nodes:
            support_status = "supported"
            support_reason = "evidence_units_verified"
            verified_count += 1
        elif existing_units:
            support_status = "partial"
            support_reason = "partial_evidence_unit_match"
        elif quality_score < MIN_FRAME_QUALITY_SCORE or noise_reasons:
            support_status = "ignored_noise"
            support_reason = "low_quality_or_noise_without_evidence"
            ignored_noise_count += 1
        else:
            support_status = "unsupported"
            support_reason = "no_evidence_unit_found"
            unsupported_count += 1
            warnings.append("unsupported_frame")
        items.append(
            {
                "frame_id": frame.get("frame_id"),
                "claim_type": frame.get("claim_type"),
                "support_status": support_status,
                "support_reason": support_reason,
                "evidence_unit_count": len(existing_units),
                "missing_evidence_unit_ids": missing_units,
                "missing_node_ids": _unique_strings(missing_nodes),
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
    return {
        "schema": CLAIM_FRAME_VERIFIER_SCHEMA,
        "doc_id": doc_id,
        "version_id": claim_frames.get("version_id") or evidence_units.get("version_id") or "",
        "status": "passed" if frame_count and not warnings else ("partial" if frame_count else "skipped"),
        "frame_count": frame_count,
        "verified_frame_count": verified_count,
        "verified_frame_rate": round(verified_count / max(1, frame_count), 4),
        "unsupported_frame_count": unsupported_count,
        "low_confidence_frame_count": low_confidence_count,
        "low_quality_frame_count": low_quality_count,
        "noisy_frame_count": noisy_count,
        "ignored_noise_frame_count": ignored_noise_count,
        "missing_evidence_unit_count": missing_unit_count,
        "missing_node_count": missing_node_count,
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
        "quality_summary": frame_quality_summary(frames),
        "noisy_frame_count": sum(1 for frame in frames if frame.get("noise_reasons")),
        "low_quality_frame_count": sum(1 for frame in frames if _confidence(frame.get("quality_score"), 0.6) < LOW_FRAME_QUALITY_SCORE),
        "top_frame_noise_reasons": top_frame_noise_reasons(frames),
        "frames": frames,
        "llm_used": llm_used,
        "llm_error": llm_error,
        "llm_metadata": llm_metadata,
        "warnings": _unique_strings([*warnings, *[warning for frame in frames for warning in frame.get("warnings", [])]]),
        "created_at": time.time(),
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


def _unit_by_node_id(units: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for unit in units:
        node_id = str(unit.get("node_id") or "")
        if node_id:
            result.setdefault(node_id, []).append(unit)
    return result


def _unit_ids_for_node(node_id: str, unit_by_node: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    if not node_id:
        return []
    return [str(unit.get("unit_id") or "") for unit in unit_by_node.get(node_id, []) if unit.get("unit_id")]


def _evidence_unit_ids_for_claim(claim: Dict[str, Any], unit_by_node: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    node_ids = [str(claim.get("node_id") or "")]
    evidence = claim.get("evidence")
    if isinstance(evidence, dict):
        node_ids.append(str(evidence.get("node_id") or ""))
    return _unique_strings(unit_id for node_id in node_ids for unit_id in _unit_ids_for_node(node_id, unit_by_node))


def _unit_type(kind: str, node_path: str, heading: str) -> str:
    raw = f"{kind} {node_path} {heading}".lower()
    if kind in {"abstract", "keywords", "reference", "figure", "table", "paragraph", "section", "subsection"}:
        return kind
    if any(token in raw for token in ("reference", "参考文献", "引用")):
        return "reference"
    if any(token in raw for token in ("figure", "fig.", "图 ")):
        return "figure"
    if any(token in raw for token in ("table", "表 ")):
        return "table"
    if int(_safe_int_from_level(raw)) >= 2:
        return "subsection"
    if kind in {"page"}:
        return "paragraph"
    return "section" if kind == "section" else "paragraph"


def _safe_int_from_level(text: str) -> int:
    match = re.search(r"\blevel[:=](\d+)", text)
    return int(match.group(1)) if match else 0


def _source_kind(unit_type: str, node_path: str, heading: str) -> str:
    text = f"{unit_type} {node_path} {heading}".lower()
    if unit_type in {"table", "figure", "reference"}:
        return unit_type
    if any(token in text for token in ("table", "表 ")):
        return "table"
    if any(token in text for token in ("figure", "fig.", "图 ")):
        return "figure"
    if any(token in text for token in ("reference", "参考文献", "引用")):
        return "reference"
    return "node"


def _page_range(node: Dict[str, Any]) -> List[Optional[int]]:
    return [node.get("page_start"), node.get("page_end")]


def _short_keywords(value: Any, text: str) -> List[str]:
    items = []
    if isinstance(value, list):
        items.extend(str(item) for item in value)
    elif isinstance(value, str):
        items.extend(re.split(r"[,，;；\s]+", value))
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text):
        items.append(token)
    return _unique_strings(items)[:12]


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


def _frame_match_score(frame: Dict[str, Any], terms: List[str], *, query: str = "", support_status: str = "") -> float:
    quality_score = _confidence(frame.get("quality_score"), 0.6)
    status = support_status or str(frame.get("support_status") or "")
    if not query_allows_weak_frames(query) and (quality_score < MIN_FRAME_QUALITY_SCORE or status in {"unsupported", "ignored_noise"}):
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
    if status == "supported":
        score += 0.4
    elif status == "partial":
        score += 0.15
    elif status in {"unsupported", "ignored_noise"}:
        score -= 1.2 if not query_allows_weak_frames(query) else 0.3
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
    return {
        "frame_count": frame_count,
        "verified_frame_count": verified,
        "verified_frame_rate": round(verified / max(1, frame_count), 4),
        "unsupported_frame_count": sum(int(doc.get("unsupported_frame_count") or 0) for doc in documents),
        "low_confidence_frame_count": sum(int(doc.get("low_confidence_frame_count") or 0) for doc in documents),
        "low_quality_frame_count": sum(int(doc.get("low_quality_frame_count") or 0) for doc in documents),
        "noisy_frame_count": sum(int(doc.get("noisy_frame_count") or 0) for doc in documents),
        "ignored_noise_frame_count": sum(int(doc.get("ignored_noise_frame_count") or 0) for doc in documents),
        "missing_evidence_unit_count": sum(int(doc.get("missing_evidence_unit_count") or 0) for doc in documents),
        "missing_node_count": sum(int(doc.get("missing_node_count") or 0) for doc in documents),
        "citation_gap_count": sum(int(doc.get("citation_gap_count") or 0) for doc in documents),
        "top_frame_noise_reasons": top_noise_reasons(documents),
    }
