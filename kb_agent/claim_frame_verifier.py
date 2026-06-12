from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set
import time

from .claim_frame_builder import confidence, count_by_field
from .claim_frame_quality import (
    LOW_FRAME_QUALITY_SCORE,
    MIN_FRAME_QUALITY_SCORE,
    existing_frame_quality,
    frame_quality_summary,
    top_frame_noise_reasons,
    top_noise_reasons,
)
from .claim_frame_search import (
    citation_risk_for_semantic_status,
    normalize_citation_risk,
    normalize_semantic_support_status,
    query_terms,
)
from .utils import compact_whitespace, unique_strings as _unique_strings


CLAIM_FRAME_VERIFIER_SCHEMA = "claim_frame_verifier.v1"
CLAIM_FRAME_VERIFIER_RESULT_SCHEMA = "claim_frame_verifier_result.v1"

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


def verify_claim_frames_payload(
    doc_id: str,
    claim_frames: Dict[str, Any],
    evidence_units: Dict[str, Any],
    *,
    node_ids: Set[str],
    source_ids: Set[str],
    citation_map: Dict[str, Any],
) -> Dict[str, Any]:
    units = evidence_units.get("units") or []
    unit_by_id = {str(unit.get("unit_id") or ""): unit for unit in units if isinstance(unit, dict)}
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
        quality_score = confidence(computed_quality.get("quality_score"), 0.6)
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
        frame_confidence = confidence(frame.get("confidence"), 0.0)
        if frame_confidence < 0.5:
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
        semantic = semantic_support_for_frame(frame, existing_units)
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


def sync_claim_frames_with_verifier(payload: Dict[str, Any], verifier: Dict[str, Any]) -> None:
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
    payload["trace_status_counts"] = count_by_field(frames, "trace_status")
    payload["support_status_counts"] = count_by_field(frames, "support_status")
    payload["semantic_support_status_counts"] = count_by_field(frames, "semantic_support_status")
    payload["semantic_verified_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "semantically_supported")
    payload["semantic_supported_frame_rate"] = round(payload["semantic_verified_frame_count"] / max(1, len(frames)), 4)
    payload["partial_supported_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "partially_supported")
    payload["related_only_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "related_only")
    payload["contradicted_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "contradicted")
    payload["insufficient_evidence_frame_count"] = sum(1 for frame in frames if frame.get("semantic_support_status") == "insufficient_evidence")
    payload["citation_risk_counts"] = count_by_field(frames, "citation_risk")
    payload["quality_summary"] = frame_quality_summary(frames)
    payload["noisy_frame_count"] = sum(1 for frame in frames if frame.get("noise_reasons"))
    payload["low_quality_frame_count"] = sum(1 for frame in frames if confidence(frame.get("quality_score"), 0.6) < LOW_FRAME_QUALITY_SCORE)
    payload["top_frame_noise_reasons"] = top_frame_noise_reasons(frames)
    payload["warnings"] = _unique_strings([*(payload.get("warnings") or []), *[warning for frame in frames for warning in frame.get("warnings", [])]])


def semantic_support_for_frame(frame: Dict[str, Any], units: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not frame.get("short_claim"):
        return semantic_result("not_checked", 0.0, "missing_short_claim", [], [], [])
    if not units:
        return semantic_result("insufficient_evidence", 0.0, "no_evidence_units", [], [], [])

    claim_text = _frame_semantic_text(frame)
    claim_terms = _semantic_terms(claim_text)
    core_terms = _core_claim_terms(frame, claim_terms)
    if not claim_terms:
        return semantic_result("not_checked", 0.0, "no_claim_terms", [], [], [])

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
        unit_confidence = confidence(unit.get("confidence"), 0.6)
        score = min(1.0, 0.55 * claim_coverage + 0.35 * core_coverage + 0.10 * unit_confidence)
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
        return semantic_result("contradicted", best_conflict, "conflicting_evidence_signal", [], [], contradictory_ids)

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
        return semantic_result("semantically_supported", best_score, "primary_evidence_overlap", primary, weak, [])
    if primary or has_specific_core_hit:
        return semantic_result("partially_supported", best_score, "partial_core_overlap", primary, weak, [])
    if weak:
        return semantic_result("related_only", best_score, "topic_related_only", [], weak, [])
    return semantic_result("insufficient_evidence", 0.0, "no_semantic_evidence_match", [], [], [])


def semantic_result(
    status: str,
    score: float,
    reason: str,
    primary_ids: List[str],
    weak_ids: List[str],
    contradictory_ids: List[str],
) -> Dict[str, Any]:
    normalized = normalize_semantic_support_status(status)
    return {
        "semantic_support_status": normalized,
        "semantic_support_score": round(max(0.0, min(1.0, float(score or 0.0))), 3),
        "semantic_support_reason": reason,
        "primary_evidence_unit_ids": _unique_strings(primary_ids)[:8],
        "weak_evidence_unit_ids": _unique_strings(weak_ids)[:8],
        "contradictory_evidence_unit_ids": _unique_strings(contradictory_ids)[:8],
        "citation_risk": normalize_citation_risk(citation_risk_for_semantic_status(normalized)),
    }


def verifier_totals(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    frame_count = sum(int(doc.get("frame_count") or 0) for doc in documents)
    verified = sum(int(doc.get("verified_frame_count") or 0) for doc in documents)
    trace_status_counts = merge_count_dicts(doc.get("trace_status_counts") or {} for doc in documents)
    support_status_counts = merge_count_dicts(doc.get("support_status_counts") or {} for doc in documents)
    semantic_status_counts = merge_count_dicts(doc.get("semantic_support_status_counts") or {} for doc in documents)
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
        "citation_risk_counts": merge_count_dicts(doc.get("citation_risk_counts") or {} for doc in documents),
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


def merge_count_dicts(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value or 0)
    return counts


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
    for term in query_terms(compact_whitespace(text)):
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
