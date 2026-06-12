from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .utils import compact_whitespace, first_words, unique_strings as _unique_strings


ANSWER_PLAN_SCHEMA = "answer_plan.v1"
ANSWERABILITY_VALUES = {
    "answerable",
    "partially_answerable",
    "conflicting",
    "insufficient_evidence",
}


def build_answer_plan(
    query: str,
    claim_frame_matches: Optional[Any] = None,
    evidence_items: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    max_items_per_bucket: int = 8,
) -> Dict[str, Any]:
    items = _claim_items(claim_frame_matches, evidence_items)
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "strong_claims": [],
        "qualified_claims": [],
        "related_claims": [],
        "conflicting_claims": [],
        "insufficient_claims": [],
        "unchecked_claims": [],
    }
    warnings: List[str] = []
    for item in items:
        normalized = _normalize_claim_item(item)
        if not normalized.get("short_claim") and not normalized.get("frame_id"):
            continue
        bucket = _bucket_for_claim(normalized)
        buckets[bucket].append(normalized)

    for name, bucket_items in buckets.items():
        buckets[name] = _dedupe_claims(bucket_items)[:max_items_per_bucket]

    if not any(buckets.values()):
        warnings.append("answer_plan_no_claim_frames")
    if buckets["conflicting_claims"]:
        warnings.append("answer_plan_conflicting_claims")
    if buckets["insufficient_claims"] and not (buckets["strong_claims"] or buckets["qualified_claims"]):
        warnings.append("answer_plan_insufficient_evidence")
    if buckets["unchecked_claims"]:
        warnings.append("answer_plan_unchecked_claims")

    answerability = _answerability(buckets)
    answer_policy = _answer_policy(answerability, buckets)
    return {
        "schema": ANSWER_PLAN_SCHEMA,
        "query": query,
        "answerability": answerability,
        "answer_policy": answer_policy,
        **buckets,
        "strong_claim_count": len(buckets["strong_claims"]),
        "qualified_claim_count": len(buckets["qualified_claims"]),
        "related_claim_count": len(buckets["related_claims"]),
        "conflicting_claim_count": len(buckets["conflicting_claims"]),
        "insufficient_claim_count": len(buckets["insufficient_claims"]),
        "unchecked_claim_count": len(buckets["unchecked_claims"]),
        "warnings": _unique_strings(warnings),
    }


def answer_plan_summary(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    plan = plan or {}
    return {
        "schema": "answer_plan_summary.v1",
        "available": plan.get("schema") == ANSWER_PLAN_SCHEMA,
        "answerability": plan.get("answerability") or "insufficient_evidence",
        "answer_policy": plan.get("answer_policy") or "",
        "strong_claim_count": int(plan.get("strong_claim_count") or 0),
        "qualified_claim_count": int(plan.get("qualified_claim_count") or 0),
        "related_claim_count": int(plan.get("related_claim_count") or 0),
        "conflicting_claim_count": int(plan.get("conflicting_claim_count") or 0),
        "insufficient_claim_count": int(plan.get("insufficient_claim_count") or 0),
        "unchecked_claim_count": int(plan.get("unchecked_claim_count") or 0),
        "warnings": plan.get("warnings") or [],
    }


def answer_plan_counts(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary = answer_plan_summary(plan)
    return {
        "answerability": summary["answerability"],
        "answer_policy": summary["answer_policy"],
        "strong_claim_count": summary["strong_claim_count"],
        "qualified_claim_count": summary["qualified_claim_count"],
        "related_claim_count": summary["related_claim_count"],
        "conflicting_claim_count": summary["conflicting_claim_count"],
        "insufficient_claim_count": summary["insufficient_claim_count"],
        "unchecked_claim_count": summary["unchecked_claim_count"],
    }


def build_answer_plan_from_evidence(query: str, evidence_items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    claim_items = [
        item
        for item in evidence_items
        if isinstance(item, dict) and (item.get("claim_frame_id") or item.get("frame_id") or item.get("source") == "claim_frame")
    ]
    return build_answer_plan(query, claim_items, max_items_per_bucket=12)


def answer_plan_warning_tags(summary: Dict[str, Any]) -> List[str]:
    warnings = list(summary.get("warnings") or [])
    if int(summary.get("conflicting_claim_count") or 0) > 0:
        warnings.append(f"answer_plan_conflicts:{summary.get('conflicting_claim_count')}")
    if int(summary.get("insufficient_claim_count") or 0) > 0 and not int(summary.get("strong_claim_count") or 0):
        warnings.append(f"answer_plan_insufficient_evidence:{summary.get('insufficient_claim_count')}")
    if str(summary.get("answerability") or "") in {"conflicting", "insufficient_evidence"}:
        warnings.append(f"answerability:{summary.get('answerability')}")
    return _unique_strings(warnings)


def answer_plan_open_questions(summary: Dict[str, Any]) -> List[str]:
    questions = []
    if int(summary.get("conflicting_claim_count") or 0) > 0:
        questions.append("部分 ClaimFrame 存在 conflicting_evidence，需要回到 EvidenceUnit 核验后再作为结论。")
    if int(summary.get("insufficient_claim_count") or 0) > 0 and not int(summary.get("strong_claim_count") or 0):
        questions.append("当前主题缺少可强支撑 ClaimFrame，需要补充 EvidenceUnit 或重新抽取事实。")
    if int(summary.get("unchecked_claim_count") or 0) > 0:
        questions.append("部分候选 ClaimFrame 尚未完成语义验证，不应直接作为强结论。")
    return questions


def task_semantic_score_adjustment(status: str) -> float:
    return {
        "semantically_supported": 0.55,
        "partially_supported": 0.2,
        "related_only": -0.35,
        "insufficient_evidence": -0.6,
        "not_checked": -0.05,
    }.get(status, -0.05)


def _claim_items(claim_frame_matches: Optional[Any], evidence_items: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if isinstance(claim_frame_matches, dict):
        raw_items = claim_frame_matches.get("items") or []
        if isinstance(raw_items, list):
            items.extend(item for item in raw_items if isinstance(item, dict))
    elif isinstance(claim_frame_matches, list):
        items.extend(item for item in claim_frame_matches if isinstance(item, dict))
    if evidence_items:
        items.extend(
            item
            for item in evidence_items
            if isinstance(item, dict) and (item.get("claim_frame_id") or item.get("frame_id") or item.get("source") == "claim_frame")
        )
    return items


def _normalize_claim_item(item: Dict[str, Any]) -> Dict[str, Any]:
    frame_id = str(item.get("frame_id") or item.get("claim_frame_id") or "")
    short_claim = compact_whitespace(
        str(item.get("short_claim") or item.get("claim") or item.get("summary") or item.get("excerpt") or "")
    )
    evidence_unit_ids = _string_list(item.get("evidence_unit_ids"))
    primary_ids = _string_list(item.get("primary_evidence_unit_ids"))
    return {
        "doc_id": str(item.get("doc_id") or ""),
        "title": str(item.get("title") or ""),
        "frame_id": frame_id,
        "short_claim": first_words(short_claim, 60),
        "claim_type": str(item.get("claim_type") or item.get("evidence_type") or ""),
        "semantic_support_status": _semantic_status(str(item.get("semantic_support_status") or "")),
        "citation_risk": _citation_risk(str(item.get("citation_risk") or "")),
        "evidence_unit_ids": evidence_unit_ids,
        "primary_evidence_unit_ids": primary_ids,
        "weak_evidence_unit_ids": _string_list(item.get("weak_evidence_unit_ids")),
        "contradictory_evidence_unit_ids": _string_list(item.get("contradictory_evidence_unit_ids")),
        "node_id": str(item.get("node_id") or ""),
        "score": _score(item),
    }


def _bucket_for_claim(item: Dict[str, Any]) -> str:
    status = str(item.get("semantic_support_status") or "not_checked")
    risk = str(item.get("citation_risk") or "not_checked")
    if status == "semantically_supported" and risk == "safe":
        return "strong_claims"
    if status == "partially_supported" and risk == "needs_qualification":
        return "qualified_claims"
    if status == "contradicted" or risk == "conflicting_evidence":
        return "conflicting_claims"
    if status == "insufficient_evidence":
        return "insufficient_claims"
    if status == "related_only":
        return "related_claims"
    return "unchecked_claims"


def _answerability(buckets: Dict[str, List[Dict[str, Any]]]) -> str:
    if buckets["conflicting_claims"]:
        return "conflicting"
    if buckets["strong_claims"]:
        return "answerable"
    if buckets["qualified_claims"]:
        return "partially_answerable"
    return "insufficient_evidence"


def _answer_policy(answerability: str, buckets: Dict[str, List[Dict[str, Any]]]) -> str:
    if answerability == "conflicting":
        return "优先说明冲突证据，不把 conflicting_evidence 写成确定结论。"
    if answerability == "answerable":
        if buckets["qualified_claims"]:
            return "可使用 strong_claims 作正式结论，qualified_claims 需带限定语。"
        return "可使用 strong_claims 作正式结论，并保留证据 ID。"
    if answerability == "partially_answerable":
        return "只能给出带限定语的回答，避免使用强确定表述。"
    return "证据不足；应提示需要更多 EvidenceUnit 或重新抽取 ClaimFrame。"


def _dedupe_claims(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in sorted(items, key=lambda value: (-float(value.get("score") or 0.0), str(value.get("frame_id") or ""))):
        key = (
            str(item.get("doc_id") or ""),
            str(item.get("frame_id") or ""),
            str(item.get("short_claim") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return _unique_strings(str(item) for item in value if str(item))
    if value:
        return [str(value)]
    return []


def _semantic_status(value: str) -> str:
    allowed = {
        "semantically_supported",
        "partially_supported",
        "related_only",
        "contradicted",
        "insufficient_evidence",
        "not_checked",
    }
    return value if value in allowed else "not_checked"


def _citation_risk(value: str) -> str:
    allowed = {"safe", "needs_qualification", "needs_more_evidence", "conflicting_evidence", "not_checked"}
    return value if value in allowed else "not_checked"


def _score(item: Dict[str, Any]) -> float:
    for field in ("score", "semantic_support_score", "confidence"):
        if item.get(field) in {None, ""}:
            continue
        try:
            return round(float(item.get(field)), 3)
        except (TypeError, ValueError):
            continue
    return 0.0
