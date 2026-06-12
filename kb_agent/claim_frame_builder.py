from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from .claim_frame_evidence import (
    evidence_unit_ids_for_claim,
    unit_ids_for_node,
    unit_ids_for_source,
)
from .claim_frame_quality import (
    LOW_FRAME_QUALITY_SCORE,
    frame_quality,
    frame_quality_summary,
    should_skip_low_quality_frame,
    top_frame_noise_reasons,
)
from .llm import llm_payload_metadata
from .text_quality import short_research_text
from .utils import compact_whitespace, excerpt as _excerpt, stable_id, unique_strings as _unique_strings


CLAIM_FRAME_SCHEMA = "claim_frames.v1"
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

JsonGenerator = Callable[..., Dict[str, object]]


def build_claim_frames(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    claims: Any,
    innovation: Any,
    table_summaries: Any,
    citation_map: Any,
    unit_by_node: Dict[str, List[Dict[str, Any]]],
    unit_by_id: Dict[str, Dict[str, Any]],
    unit_by_source_id: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    frames = frames_from_claims(doc_id, version_id, claims, unit_by_node, unit_by_id, unit_by_source_id)
    frames.extend(
        frames_from_innovations(
            doc_id,
            version_id,
            innovation,
            unit_by_node,
            unit_by_id,
            unit_by_source_id,
            start_index=len(frames),
        )
    )
    frames.extend(frames_from_table_summaries(doc_id, version_id, table_summaries, unit_by_node, unit_by_source_id, start_index=len(frames)))
    frames.extend(frames_from_citations(doc_id, version_id, card, citation_map, unit_by_node, unit_by_source_id, start_index=len(frames)))
    return dedupe_frames(frames)


def frames_from_claims(
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
        frame_claim_type = claim_type(str(claim.get("claim_type") or claim.get("type") or ""))
        evidence_unit_ids, binding_warnings = evidence_unit_ids_for_claim(claim, unit_by_node, unit_by_id, unit_by_source_id)
        frames.append(
            frame_record(
                doc_id,
                version_id,
                frame_claim_type,
                text,
                evidence_unit_ids,
                source="claim",
                source_claim_ids=[str(claim.get("claim_id") or "")],
                confidence=confidence(claim.get("confidence"), 0.64),
                index=index,
                binding_warnings=binding_warnings,
            )
        )
    return [item for item in frames if item]


def frames_from_innovations(
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
        frame_claim_type = claim_type(str(item.get("type") or "contribution"))
        evidence_unit_ids, binding_warnings = evidence_unit_ids_for_claim(item, unit_by_node, unit_by_id, unit_by_source_id)
        frames.append(
            frame_record(
                doc_id,
                version_id,
                frame_claim_type,
                text,
                evidence_unit_ids,
                source="innovation",
                source_claim_ids=[stable_id("innovation", doc_id, offset, text, length=14)],
                confidence=confidence(item.get("confidence"), 0.58),
                index=start_index + offset,
                problem=str(item.get("problem") or ""),
                method=str(item.get("approach") or ""),
                binding_warnings=binding_warnings,
            )
        )
    return [item for item in frames if item]


def frames_from_table_summaries(
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
        evidence_unit_ids = _unique_strings([*unit_ids_for_node(node_id, unit_by_node), *unit_ids_for_source(source_id, unit_by_source_id)])
        frames.append(
            frame_record(
                doc_id,
                version_id,
                "result",
                text,
                evidence_unit_ids,
                source="table_summary",
                source_claim_ids=[str(table.get("table_id") or stable_id("table", doc_id, offset, text, length=14))],
                confidence=confidence(table.get("confidence"), 0.6),
                index=start_index + offset,
            )
        )
    return [item for item in frames if item]


def frames_from_citations(
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
                *unit_ids_for_node(node_id, unit_by_node),
                *[unit_id for source_id in source_ids for unit_id in unit_ids_for_source(source_id, unit_by_source_id)],
            ]
        )
        text = f"{title} 引用了 {ref_id or raw}。" if node_id else f"{title} 的参考文献包含 {ref_id or raw}。"
        frames.append(
            frame_record(
                doc_id,
                version_id,
                "citation",
                text,
                evidence_unit_ids,
                source="citation_map",
                source_claim_ids=[ref_id or stable_id("reference", doc_id, raw, length=14)],
                confidence=confidence(item.get("confidence"), 0.62 if node_id else 0.5),
                index=start_index + offset,
            )
        )
    return [item for item in frames if item]


def frame_record(
    doc_id: str,
    version_id: str,
    claim_type_value: str,
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
    inferred = infer_frame_fields(short_claim, claim_type_value)
    if problem:
        inferred["problem"] = short_research_text(problem, 180)
    if method:
        inferred["method"] = short_research_text(method, 180)
    clean_evidence = _unique_strings(evidence_unit_ids)
    quality = frame_quality(short_claim, clean_evidence, source=source, claim_type=claim_type_value, confidence=confidence)
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
        "frame_id": stable_id("cf", version_id, claim_type_value, short_claim, ",".join(clean_evidence), index, length=14),
        "doc_id": doc_id,
        "version_id": version_id,
        "claim_type": claim_type_value,
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


def enhance_frames_with_llm(
    card: Dict[str, Any],
    frames: List[Dict[str, Any]],
    units: List[Dict[str, Any]],
    *,
    json_generator: JsonGenerator,
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
    payload = json_generator(system_prompt, user_prompt, operation="claim_frames", stage="enhance")
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
            frame["confidence"] = round(max(0.0, min(1.0, confidence(raw.get("confidence"), frame.get("confidence", 0.6)))), 3)
        quality = frame_quality(
            str(frame.get("short_claim") or ""),
            [str(item) for item in frame.get("evidence_unit_ids") or []],
            source=str(frame.get("source") or ""),
            claim_type=str(frame.get("claim_type") or ""),
            confidence=confidence(frame.get("confidence"), 0.6),
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


def claim_frames_payload(
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
        "claim_type_counts": count_by_field(frames, "claim_type"),
        "trace_status_counts": count_by_field(frames, "trace_status"),
        "support_status_counts": count_by_field(frames, "support_status"),
        "quality_summary": frame_quality_summary(frames),
        "noisy_frame_count": sum(1 for frame in frames if frame.get("noise_reasons")),
        "low_quality_frame_count": sum(1 for frame in frames if confidence(frame.get("quality_score"), 0.6) < LOW_FRAME_QUALITY_SCORE),
        "top_frame_noise_reasons": top_frame_noise_reasons(frames),
        "frames": frames,
        "llm_used": llm_used,
        "llm_error": llm_error,
        "llm_enhancement": llm_metadata.get("llm_enhancement") or {"used": False, "truncated": False},
        "llm_metadata": llm_metadata,
        "warnings": _unique_strings([*warnings, *[warning for frame in frames for warning in frame.get("warnings", [])]]),
        "created_at": time.time(),
    }


def claim_type(raw: str) -> str:
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


def infer_frame_fields(text: str, claim_type_value: str) -> Dict[str, str]:
    result = {
        "problem": "",
        "method": "",
        "dataset_or_setting": "",
        "metric_or_signal": "",
        "result_or_gain": "",
        "limitation": "",
    }
    if claim_type_value == "problem" or any(token in text for token in ("问题", "挑战", "不足", "瓶颈", "解决")):
        result["problem"] = _excerpt(text, 180)
    if claim_type_value == "method" or any(token in text for token in ("提出", "方法", "算法", "模型", "框架", "机制")):
        result["method"] = _excerpt(text, 180)
    if claim_type_value == "result" or any(token in text for token in ("实验", "结果", "提升", "优于", "降低", "验证")):
        result["result_or_gain"] = _excerpt(text, 180)
    if claim_type_value == "limitation" or any(token in text for token in ("局限", "不足", "限制", "未来工作", "展望")):
        result["limitation"] = _excerpt(text, 180)
    setting_terms = [term for term in ("数据集", "场景", "环境", "家庭", "仿真", "真实", "多智能体", "服务机器人") if term in text]
    if setting_terms:
        result["dataset_or_setting"] = "、".join(setting_terms[:6])
    metric_terms = [term for term in METRIC_TERMS if term.lower() in text.lower()]
    if metric_terms:
        result["metric_or_signal"] = "、".join(metric_terms[:6])
    return result


def confidence(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def dedupe_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def count_by_field(items: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(field) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts
