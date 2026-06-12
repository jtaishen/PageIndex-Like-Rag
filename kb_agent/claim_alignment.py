from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .utils import compact_whitespace, first_words, stable_id, unique_strings


ALIGNMENT_SCHEMA = "claim_alignment.v1"
RELATION_SCHEMA = "claim_relations.v1"
RELATION_TYPES = {
    "cites",
    "supports",
    "contradicts",
    "same_dataset",
    "same_metric",
    "improves_over",
    "ablation_of",
    "limitation_of",
}
CONTRADICTION_STATUSES = {
    "supports",
    "contradicts",
    "incomparable",
}
STOP_TERMS = {
    "本文",
    "研究",
    "方法",
    "系统",
    "模型",
    "算法",
    "实验",
    "结果",
    "提出",
    "通过",
    "基于",
    "任务",
    "规划",
    "问题",
}


def build_claim_alignment(query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = _normalized_frames(contexts)
    groups: List[Dict[str, Any]] = []
    for frame in frames:
        target = _best_group(groups, frame)
        if target is None:
            groups.append(_new_group(query, frame))
        else:
            target["claims"].append(frame)
            target["terms"] = unique_strings([*target.get("terms", []), *frame["terms"]])[:20]
    normalized = [_finalize_group(query, group) for group in groups]
    return {
        "schema": ALIGNMENT_SCHEMA,
        "query": query,
        "group_count": len(normalized),
        "groups": normalized,
        "method_family_groups": _groups_by_type(normalized, {"method", "claim"}),
        "conflicting_claim_groups": [group for group in normalized if group.get("support_level") == "conflicting"],
        "research_gap_candidates": _research_gap_candidates(normalized),
        "warnings": _alignment_warnings(normalized),
    }


def build_claim_relations(alignment: Dict[str, Any]) -> Dict[str, Any]:
    relations = []
    comparability_checks = []
    for group in alignment.get("groups") or []:
        claims = [item for item in group.get("claims") or [] if isinstance(item, dict)]
        for index, source in enumerate(claims):
            for target in claims[index + 1 :]:
                relation, check = _relation_for_pair(source, target, group)
                if relation:
                    relations.append(relation)
                if check:
                    comparability_checks.append(check)
    type_counts: Dict[str, int] = {}
    for relation in relations:
        relation_type = str(relation.get("relation_type") or "")
        type_counts[relation_type] = type_counts.get(relation_type, 0) + 1
    classification_counts: Dict[str, int] = {}
    for check in comparability_checks:
        status = str(check.get("contradiction_status") or "")
        if status not in CONTRADICTION_STATUSES:
            continue
        classification_counts[status] = classification_counts.get(status, 0) + 1
    return {
        "schema": RELATION_SCHEMA,
        "relation_count": len(relations),
        "type_counts": type_counts,
        "relations": relations,
        "comparability_checks": comparability_checks,
        "comparability_check_count": len(comparability_checks),
        "conflict_classification_counts": classification_counts,
        "incomparable_pair_count": classification_counts.get("incomparable", 0),
        "warnings": ["claim_relations_empty"] if not relations else [],
    }


def claim_alignment_summary(alignment: Dict[str, Any], relations: Dict[str, Any]) -> Dict[str, Any]:
    groups = alignment.get("groups") or []
    return {
        "schema": "claim_alignment_summary.v1",
        "available": alignment.get("schema") == ALIGNMENT_SCHEMA,
        "group_count": len(groups),
        "method_family_group_count": len(alignment.get("method_family_groups") or []),
        "conflicting_group_count": len(alignment.get("conflicting_claim_groups") or []),
        "research_gap_count": len(alignment.get("research_gap_candidates") or []),
        "relation_count": int(relations.get("relation_count") or 0),
        "relation_type_counts": relations.get("type_counts") or {},
        "conflict_classification_counts": relations.get("conflict_classification_counts") or {},
        "incomparable_pair_count": int(relations.get("incomparable_pair_count") or 0),
        "warnings": unique_strings([*(alignment.get("warnings") or []), *(relations.get("warnings") or [])]),
    }


def claim_alignment_rollup(summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rollup = {
        "available": False,
        "group_count": 0,
        "method_family_group_count": 0,
        "conflicting_group_count": 0,
        "research_gap_count": 0,
        "relation_count": 0,
        "relation_type_counts": {},
        "conflict_classification_counts": {},
        "incomparable_pair_count": 0,
        "warnings": [],
    }
    type_counts: Dict[str, int] = {}
    classification_counts: Dict[str, int] = {}
    warnings: List[str] = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        rollup["available"] = bool(rollup["available"] or summary.get("available"))
        for field in (
            "group_count",
            "method_family_group_count",
            "conflicting_group_count",
            "research_gap_count",
            "relation_count",
            "incomparable_pair_count",
        ):
            rollup[field] += int(summary.get(field) or 0)
        for key, value in (summary.get("relation_type_counts") or {}).items():
            type_counts[str(key)] = type_counts.get(str(key), 0) + int(value or 0)
        for key, value in (summary.get("conflict_classification_counts") or {}).items():
            classification_counts[str(key)] = classification_counts.get(str(key), 0) + int(value or 0)
        warnings.extend(str(item) for item in summary.get("warnings") or [] if str(item))
    rollup["relation_type_counts"] = type_counts
    rollup["conflict_classification_counts"] = classification_counts
    rollup["warnings"] = unique_strings(warnings)
    return rollup


def review_alignment_sections(alignment: Dict[str, Any], relations: Dict[str, Any]) -> Dict[str, Any]:
    groups = alignment.get("groups") or []
    return {
        "claim_alignment_summary": claim_alignment_summary(alignment, relations),
        "method_lineage": _method_lineage(alignment, relations),
        "evidence_patterns": _groups_by_type(groups, {"result", "citation"}),
        "limitation_groups": _groups_by_type(groups, {"limitation"}),
        "research_gap_candidates": alignment.get("research_gap_candidates") or [],
    }


def _normalized_frames(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frames = []
    for context in contexts:
        for frame in (context.get("claim_frames") or {}).get("frames") or []:
            if not isinstance(frame, dict):
                continue
            frame_id = str(frame.get("frame_id") or "")
            if not frame_id:
                continue
            text = _frame_text(frame)
            terms = _terms(text)
            frames.append(
                {
                    "doc_id": str(context.get("doc_id") or ""),
                    "title": str(context.get("title") or ""),
                    "frame_id": frame_id,
                    "claim_type": str(frame.get("claim_type") or "claim"),
                    "short_claim": first_words(compact_whitespace(str(frame.get("short_claim") or "")), 56),
                    "method": str(frame.get("method") or ""),
                    "dataset_or_setting": str(frame.get("dataset_or_setting") or ""),
                    "metric_or_signal": str(frame.get("metric_or_signal") or ""),
                    "result_or_gain": str(frame.get("result_or_gain") or ""),
                    "limitation": str(frame.get("limitation") or ""),
                    "semantic_support_status": str(frame.get("semantic_support_status") or "not_checked"),
                    "citation_risk": str(frame.get("citation_risk") or "not_checked"),
                    "evidence_unit_ids": [str(item) for item in frame.get("evidence_unit_ids") or [] if str(item)],
                    "primary_evidence_unit_ids": [str(item) for item in frame.get("primary_evidence_unit_ids") or [] if str(item)],
                    "terms": terms,
                    "metric_terms": _terms(str(frame.get("metric_or_signal") or "")),
                    "setting_terms": _terms(str(frame.get("dataset_or_setting") or "")),
                    "polarity": _polarity(frame),
                }
            )
    return frames


def _new_group(query: str, frame: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": query,
        "dominant_claim_type": frame["claim_type"],
        "terms": list(frame["terms"]),
        "claims": [frame],
    }


def _best_group(groups: List[Dict[str, Any]], frame: Dict[str, Any]) -> Dict[str, Any] | None:
    best = None
    best_score = 0.0
    for group in groups:
        if group.get("dominant_claim_type") != frame["claim_type"]:
            continue
        score = _overlap_score(group.get("terms") or [], frame["terms"])
        if score > best_score:
            best = group
            best_score = score
    return best if best_score >= 0.18 else None


def _finalize_group(query: str, group: Dict[str, Any]) -> Dict[str, Any]:
    claims = group.get("claims") or []
    claim_type = str(group.get("dominant_claim_type") or "claim")
    terms = list(group.get("terms") or [])
    group_id = stable_id("ca", query, claim_type, "|".join(terms[:8]), length=12)
    support_level = _support_level(claims)
    evidence_ids = unique_strings(
        evidence_id
        for claim in claims
        for evidence_id in (claim.get("primary_evidence_unit_ids") or claim.get("evidence_unit_ids") or [])
    )[:10]
    return {
        "group_id": group_id,
        "topic": " ".join(terms[:6]) or query,
        "dominant_claim_type": claim_type,
        "claim_frame_ids": [claim["frame_id"] for claim in claims],
        "doc_ids": unique_strings(claim["doc_id"] for claim in claims),
        "support_level": support_level,
        "alignment_reason": _alignment_reason(claim_type),
        "primary_evidence_unit_ids": evidence_ids,
        "claims": [_public_claim(claim) for claim in claims],
        "warnings": _group_warnings(claims, support_level),
    }


def _relation_for_pair(source: Dict[str, Any], target: Dict[str, Any], group: Dict[str, Any]) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    if source.get("frame_id") == target.get("frame_id") or source.get("doc_id") == target.get("doc_id"):
        return None, None
    claim_type = str(group.get("dominant_claim_type") or "")
    comparable = _comparable(source, target, claim_type)
    dimensions = _alignment_dimensions(source, target, claim_type)
    check_base = _comparability_check(source, target, group, dimensions)
    if not comparable:
        return None, {**check_base, "comparability_status": "incomparable", "contradiction_status": "incomparable", "reason": "metric_or_setting_mismatch"}
    if _conflict_risk(source) or _conflict_risk(target) or _opposite_polarity(source, target):
        return (
            _relation(source, target, "contradicts", "comparable_conflict_signal", 0.76, dimensions),
            {**check_base, "comparability_status": "comparable", "contradiction_status": "contradicts", "reason": "comparable_conflict_signal"},
        )
    if claim_type == "method":
        return (
            _relation(source, target, "supports", "method_family_overlap", 0.72, dimensions),
            {**check_base, "comparability_status": "comparable", "contradiction_status": "supports", "reason": "method_family_overlap"},
        )
    if claim_type == "result":
        if source.get("metric_terms") and target.get("metric_terms"):
            if _has_improvement_signal(source) or _has_improvement_signal(target):
                return (
                    _relation(source, target, "improves_over", "same_metric_improvement_signal", 0.68, dimensions),
                    {**check_base, "comparability_status": "comparable", "contradiction_status": "supports", "reason": "same_metric_improvement_signal"},
                )
            if _overlap_score(source.get("metric_terms") or [], target.get("metric_terms") or []) > 0:
                return (
                    _relation(source, target, "same_metric", "metric_terms_overlap", 0.66, dimensions),
                    {**check_base, "comparability_status": "comparable", "contradiction_status": "supports", "reason": "metric_terms_overlap"},
                )
        if _overlap_score(source.get("setting_terms") or [], target.get("setting_terms") or []) > 0:
            return (
                _relation(source, target, "same_dataset", "dataset_or_setting_overlap", 0.64, dimensions),
                {**check_base, "comparability_status": "comparable", "contradiction_status": "supports", "reason": "dataset_or_setting_overlap"},
            )
        return (
            _relation(source, target, "supports", "result_terms_overlap", 0.62, dimensions),
            {**check_base, "comparability_status": "comparable", "contradiction_status": "supports", "reason": "result_terms_overlap"},
        )
    if claim_type == "limitation":
        return (
            _relation(source, target, "limitation_of", "limitation_terms_overlap", 0.62, dimensions),
            {**check_base, "comparability_status": "comparable", "contradiction_status": "supports", "reason": "limitation_terms_overlap"},
        )
    return (
        _relation(source, target, "supports", "claim_terms_overlap", 0.6, dimensions),
        {**check_base, "comparability_status": "comparable", "contradiction_status": "supports", "reason": "claim_terms_overlap"},
    )


def _relation(
    source: Dict[str, Any],
    target: Dict[str, Any],
    relation_type: str,
    reason: str,
    confidence: float,
    dimensions: List[str],
) -> Dict[str, Any]:
    safe_type = relation_type if relation_type in RELATION_TYPES else "supports"
    return {
        "source_frame_id": source["frame_id"],
        "target_frame_id": target["frame_id"],
        "source_doc_id": source["doc_id"],
        "target_doc_id": target["doc_id"],
        "relation_type": safe_type,
        "reason": reason,
        "confidence": confidence,
        "alignment_dimensions": dimensions,
        "evidence_unit_ids": unique_strings([*(source.get("evidence_unit_ids") or []), *(target.get("evidence_unit_ids") or [])])[:10],
        "warnings": [] if safe_type == relation_type else ["relation_type_normalized_to_supports"],
    }


def _comparability_check(
    source: Dict[str, Any],
    target: Dict[str, Any],
    group: Dict[str, Any],
    dimensions: List[str],
) -> Dict[str, Any]:
    return {
        "group_id": group.get("group_id"),
        "claim_type": group.get("dominant_claim_type"),
        "source_frame_id": source["frame_id"],
        "target_frame_id": target["frame_id"],
        "source_doc_id": source["doc_id"],
        "target_doc_id": target["doc_id"],
        "alignment_dimensions": dimensions,
        "evidence_unit_ids": unique_strings([*(source.get("evidence_unit_ids") or []), *(target.get("evidence_unit_ids") or [])])[:10],
    }


def _public_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": claim["doc_id"],
        "title": claim["title"],
        "frame_id": claim["frame_id"],
        "claim_type": claim["claim_type"],
        "short_claim": claim["short_claim"],
        "method": claim["method"],
        "dataset_or_setting": claim["dataset_or_setting"],
        "metric_or_signal": claim["metric_or_signal"],
        "result_or_gain": claim["result_or_gain"],
        "limitation": claim["limitation"],
        "terms": claim["terms"],
        "metric_terms": claim["metric_terms"],
        "setting_terms": claim["setting_terms"],
        "semantic_support_status": claim["semantic_support_status"],
        "citation_risk": claim["citation_risk"],
        "evidence_unit_ids": claim["evidence_unit_ids"],
        "primary_evidence_unit_ids": claim["primary_evidence_unit_ids"],
        "polarity": claim["polarity"],
    }


def _frame_text(frame: Dict[str, Any]) -> str:
    return compact_whitespace(
        " ".join(
            str(frame.get(field) or "")
            for field in ("short_claim", "problem", "method", "dataset_or_setting", "metric_or_signal", "result_or_gain", "limitation")
        )
    )


def _terms(text: str) -> List[str]:
    result: List[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", compact_whitespace(text)):
        if token in STOP_TERMS or token.lower() in STOP_TERMS:
            continue
        is_cjk = bool(re.fullmatch(r"[\u4e00-\u9fff]{2,}", token))
        if not is_cjk or len(token) <= 8:
            result.append(token)
        if is_cjk and len(token) >= 4:
            result.extend(token[index : index + 2] for index in range(0, len(token) - 1) if token[index : index + 2] not in STOP_TERMS)
    return unique_strings(result)[:24]


def _overlap_score(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, min(len(left_set), len(right_set)))


def _support_level(claims: List[Dict[str, Any]]) -> str:
    if any(_conflict_risk(claim) for claim in claims):
        return "conflicting"
    if any(claim.get("semantic_support_status") == "semantically_supported" and claim.get("citation_risk") == "safe" for claim in claims):
        return "strong"
    if any(claim.get("semantic_support_status") == "partially_supported" for claim in claims):
        return "qualified"
    return "insufficient"


def _alignment_reason(claim_type: str) -> str:
    return {
        "method": "method_family_overlap",
        "result": "metric_or_result_overlap",
        "limitation": "limitation_overlap",
        "problem": "problem_overlap",
        "citation": "citation_context_overlap",
    }.get(claim_type, "claim_terms_overlap")


def _group_warnings(claims: List[Dict[str, Any]], support_level: str) -> List[str]:
    warnings = []
    if len({claim.get("doc_id") for claim in claims}) < 2:
        warnings.append("single_doc_alignment_group")
    if support_level == "conflicting":
        warnings.append("alignment_conflicting_claims")
    if support_level == "insufficient":
        warnings.append("alignment_insufficient_evidence")
    return warnings


def _alignment_warnings(groups: List[Dict[str, Any]]) -> List[str]:
    warnings = []
    if not groups:
        warnings.append("claim_alignment_empty")
    if any(group.get("support_level") == "conflicting" for group in groups):
        warnings.append("claim_alignment_conflicts")
    if any(group.get("support_level") == "insufficient" for group in groups):
        warnings.append("claim_alignment_insufficient_evidence")
    return warnings


def _groups_by_type(groups: List[Dict[str, Any]], claim_types: set[str]) -> List[Dict[str, Any]]:
    return [_group_summary(group) for group in groups if group.get("dominant_claim_type") in claim_types]


def _research_gap_candidates(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_group_summary(group) for group in groups if group.get("support_level") == "insufficient"]


def _group_summary(group: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "group_id": group.get("group_id"),
        "topic": group.get("topic"),
        "dominant_claim_type": group.get("dominant_claim_type"),
        "claim_frame_ids": group.get("claim_frame_ids") or [],
        "doc_ids": group.get("doc_ids") or [],
        "support_level": group.get("support_level"),
        "primary_evidence_unit_ids": group.get("primary_evidence_unit_ids") or [],
        "warnings": group.get("warnings") or [],
    }


def _method_lineage(alignment: Dict[str, Any], relations: Dict[str, Any]) -> Dict[str, Any]:
    relation_items = [
        relation
        for relation in relations.get("relations") or []
        if relation.get("relation_type") in {"cites", "improves_over", "ablation_of"}
        or "method_family" in (relation.get("alignment_dimensions") or [])
    ]
    return {
        "schema": "method_lineage.v1",
        "groups": alignment.get("method_family_groups") or [],
        "relations": relation_items,
        "relation_count": len(relation_items),
    }


def _comparable(source: Dict[str, Any], target: Dict[str, Any], claim_type: str) -> bool:
    if claim_type in {"method", "limitation", "problem", "claim"}:
        return True
    metric_overlap = _overlap_score(source.get("metric_terms") or [], target.get("metric_terms") or [])
    setting_terms = [*(source.get("setting_terms") or []), *(target.get("setting_terms") or [])]
    setting_overlap = _overlap_score(source.get("setting_terms") or [], target.get("setting_terms") or [])
    if claim_type == "result":
        if source.get("metric_terms") and target.get("metric_terms"):
            return metric_overlap > 0 or setting_overlap > 0
        return metric_overlap > 0 or setting_overlap > 0 or not setting_terms
    return metric_overlap > 0 or setting_overlap > 0


def _alignment_dimensions(source: Dict[str, Any], target: Dict[str, Any], claim_type: str) -> List[str]:
    dimensions = []
    if claim_type == "method":
        dimensions.append("method_family")
    if claim_type in {"problem", "gap", "claim"}:
        dimensions.append("problem")
    if claim_type == "limitation":
        dimensions.append("limitation")
    if claim_type == "result":
        dimensions.append("gain")
    if _overlap_score(source.get("metric_terms") or [], target.get("metric_terms") or []) > 0:
        dimensions.append("metric")
    if _overlap_score(source.get("setting_terms") or [], target.get("setting_terms") or []) > 0:
        dimensions.append("dataset")
    return unique_strings(dimensions)


def _polarity(frame: Dict[str, Any]) -> str:
    text = _frame_text(frame)
    if any(term in text for term in ("不足", "失败", "不能", "未验证", "缺乏", "下降", "降低任务")):
        return "negative"
    if any(term in text for term in ("提升", "提高", "优于", "改进", "增强", "降低响应", "减少")):
        return "positive"
    if str(frame.get("claim_type") or "") == "limitation":
        return "negative"
    return "neutral"


def _opposite_polarity(source: Dict[str, Any], target: Dict[str, Any]) -> bool:
    return {source.get("polarity"), target.get("polarity")} == {"positive", "negative"}


def _conflict_risk(frame: Dict[str, Any]) -> bool:
    return frame.get("semantic_support_status") == "contradicted" or frame.get("citation_risk") == "conflicting_evidence"


def _has_improvement_signal(frame: Dict[str, Any]) -> bool:
    text = " ".join(str(frame.get(field) or "") for field in ("short_claim", "result_or_gain"))
    return any(term in text for term in ("优于", "improve", "outperform", "提升", "提高", "改进"))
