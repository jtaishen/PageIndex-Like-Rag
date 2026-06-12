from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .claim_frame_quality import (
    MIN_FRAME_QUALITY_SCORE,
    frame_fallback_reason,
    frame_selection_reasons,
    query_allows_weak_frames,
)
from .utils import compact_whitespace, unique_strings as _unique_strings


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


def claim_frame_search_item(
    frame: Dict[str, Any],
    verifier_item: Dict[str, Any],
    query: str,
    terms: List[str],
) -> Optional[Dict[str, Any]]:
    support_status = str(verifier_item.get("support_status") or frame.get("support_status") or "")
    support_status = normalize_support_status(support_status)
    semantic_support_status = normalize_semantic_support_status(
        str(verifier_item.get("semantic_support_status") or frame.get("semantic_support_status") or "")
    )
    citation_risk = normalize_citation_risk(str(verifier_item.get("citation_risk") or frame.get("citation_risk") or ""))
    score = frame_match_score(
        frame,
        terms,
        query=query,
        support_status=support_status,
        semantic_support_status=semantic_support_status,
    )
    if score <= 0:
        return None
    return {
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
        "matched_fields": matched_frame_fields(frame, terms),
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


def rank_claim_frame_items(items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("frame_id") or "")))[: max(1, top_k)]


def query_terms(query: str) -> List[str]:
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    return _unique_strings(terms)[:16]


def frame_match_score(
    frame: Dict[str, Any],
    terms: List[str],
    *,
    query: str = "",
    support_status: str = "",
    semantic_support_status: str = "",
) -> float:
    quality_score = _confidence(frame.get("quality_score"), 0.6)
    status = normalize_support_status(support_status or str(frame.get("support_status") or ""))
    semantic_status = normalize_semantic_support_status(semantic_support_status or str(frame.get("semantic_support_status") or ""))
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


def matched_frame_fields(frame: Dict[str, Any], terms: List[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for field in ("short_claim", "problem", "method", "dataset_or_setting", "metric_or_signal", "result_or_gain", "limitation"):
        text = str(frame.get(field) or "").lower()
        hits = [term for term in terms if term.lower() in text]
        if hits:
            result[field] = hits[:6]
    return result


def normalize_support_status(value: str) -> str:
    if value == "supported":
        return "structurally_supported"
    if value == "partial":
        return "unchecked"
    return value


def normalize_semantic_support_status(value: str) -> str:
    return value if value in SEMANTIC_SUPPORT_STATUSES else "not_checked"


def normalize_citation_risk(value: str) -> str:
    return value if value in CITATION_RISKS else "not_checked"


def citation_risk_for_semantic_status(status: str) -> str:
    if status == "semantically_supported":
        return "safe"
    if status == "partially_supported":
        return "needs_qualification"
    if status in {"related_only", "insufficient_evidence"}:
        return "needs_more_evidence"
    if status == "contradicted":
        return "conflicting_evidence"
    return "not_checked"


def _confidence(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
