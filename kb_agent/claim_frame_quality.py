from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .text_quality import has_research_signal, research_noise_reasons
from .utils import compact_whitespace, unique_strings as _unique_strings


MIN_FRAME_QUALITY_SCORE = 0.35
LOW_FRAME_QUALITY_SCORE = 0.5


def frame_quality(
    short_claim: str,
    evidence_unit_ids: List[str],
    *,
    source: str,
    claim_type: str,
    confidence: float,
) -> Dict[str, Any]:
    noise_reasons = research_noise_reasons(short_claim, source=source)
    score = 0.5 + 0.2 * _confidence(confidence, 0.6)
    if evidence_unit_ids:
        score += 0.18
    else:
        score -= 0.22
    if source in {"claim", "innovation"}:
        score += 0.08
    if source in {"table_summary", "citation_map"} and not evidence_unit_ids:
        score -= 0.12
    if claim_type == "citation":
        score -= 0.04
    if has_research_signal(short_claim):
        score += 0.08
    else:
        score -= 0.08
    score -= min(0.3, 0.12 * len(noise_reasons))
    score = round(max(0.0, min(1.0, score)), 3)
    return {
        "quality_score": score,
        "frame_quality": "high" if score >= 0.75 else ("medium" if score >= LOW_FRAME_QUALITY_SCORE else "low"),
        "noise_reasons": noise_reasons,
    }


def existing_frame_quality(frame: Dict[str, Any], evidence_unit_ids: List[str]) -> Dict[str, Any]:
    if frame.get("quality_score") is not None:
        score = _confidence(frame.get("quality_score"), 0.6)
        quality = str(frame.get("frame_quality") or ("high" if score >= 0.75 else ("medium" if score >= LOW_FRAME_QUALITY_SCORE else "low")))
        return {
            "quality_score": score,
            "frame_quality": quality,
            "noise_reasons": _unique_strings(frame.get("noise_reasons") or []),
        }
    return frame_quality(
        str(frame.get("short_claim") or ""),
        evidence_unit_ids,
        source=str(frame.get("source") or ""),
        claim_type=str(frame.get("claim_type") or ""),
        confidence=_confidence(frame.get("confidence"), 0.6),
    )


def should_skip_low_quality_frame(source: str, evidence_unit_ids: List[str], quality: Dict[str, Any]) -> bool:
    if evidence_unit_ids:
        return False
    if source not in {"table_summary", "citation_map"}:
        return False
    return _confidence(quality.get("quality_score"), 0.0) < MIN_FRAME_QUALITY_SCORE and bool(quality.get("noise_reasons"))


def frame_quality_summary(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [_confidence(frame.get("quality_score"), 0.6) for frame in frames if isinstance(frame, dict)]
    return {
        "schema": "claim_frame_quality_summary.v1",
        "frame_count": len(frames),
        "avg_quality_score": round(sum(scores) / max(1, len(scores)), 4),
        "quality_counts": _count_by_field(frames, "frame_quality"),
        "noisy_frame_count": sum(1 for frame in frames if frame.get("noise_reasons")),
        "low_quality_frame_count": sum(1 for frame in frames if _confidence(frame.get("quality_score"), 0.6) < LOW_FRAME_QUALITY_SCORE),
        "top_noise_reasons": top_frame_noise_reasons(frames),
    }


def top_frame_noise_reasons(frames: Iterable[Dict[str, Any]]) -> List[str]:
    counts: Dict[str, int] = {}
    for frame in frames:
        for reason in frame.get("noise_reasons") or []:
            key = str(reason)
            counts[key] = counts.get(key, 0) + 1
    return [reason for reason, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:6]]


def top_noise_reasons(documents: Iterable[Dict[str, Any]]) -> List[str]:
    counts: Dict[str, int] = {}
    for doc in documents:
        for item in doc.get("items") or []:
            if not isinstance(item, dict):
                continue
            for reason in item.get("noise_reasons") or []:
                key = str(reason)
                counts[key] = counts.get(key, 0) + 1
    return [reason for reason, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:6]]


def query_allows_weak_frames(query: str) -> bool:
    lowered = compact_whitespace(query).lower()
    return any(
        term in lowered
        for term in (
            "局限",
            "不足",
            "风险",
            "缺口",
            "未支持",
            "unsupported",
            "验证",
            "证据",
            "warning",
            "噪声",
        )
    )


def frame_selection_reasons(frame: Dict[str, Any], support_status: str) -> List[str]:
    reasons = []
    if support_status:
        reasons.append(f"support:{support_status}")
    quality = frame.get("frame_quality")
    if quality:
        reasons.append(f"quality:{quality}")
    if frame.get("evidence_unit_ids"):
        reasons.append("evidence_unit_match")
    if frame.get("noise_reasons"):
        reasons.append("noise_penalty")
    return _unique_strings(reasons)


def frame_fallback_reason(frame: Dict[str, Any], support_status: str, query: str) -> str:
    if support_status == "supported":
        return ""
    if support_status == "partial":
        return "partial_evidence"
    if support_status in {"unsupported", "ignored_noise"} and query_allows_weak_frames(query):
        return "weak_frame_allowed_by_query"
    if frame.get("noise_reasons"):
        return "noise_or_front_matter_penalty"
    return "missing_evidence_unit"


def _count_by_field(items: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(field) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _confidence(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
