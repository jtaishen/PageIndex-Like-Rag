from __future__ import annotations

import re
from typing import Any, Dict, List

from .text_quality import short_research_text
from .utils import compact_whitespace, unique_strings


SCHEMA_FIELDS = (
    "normalized_subject",
    "method_family",
    "dataset",
    "metric",
    "claimed_gain",
    "condition",
    "polarity",
)

METHOD_FAMILY_PATTERNS = (
    ("llm_planning", ("大语言模型", "语言模型", "llm", "large language model", "prompt")),
    ("multi_agent_planning", ("多智能体", "multi-agent", "multi agent", "协同规划", "协同调度")),
    ("task_planning_framework", ("任务规划", "任务分解", "工具调用", "技能库", "规划框架")),
    ("optimization_algorithm", ("优化", "调度", "负载均衡", "重分配", "算法")),
    ("tree_search_rag", ("树检索", "tree search", "rag", "文档树")),
)

METRIC_PATTERNS = (
    "任务完成率",
    "任务成功率",
    "成功率",
    "准确率",
    "召回率",
    "响应时间",
    "通信开销",
    "计算开销",
    "负载均衡",
    "鲁棒性",
    "accuracy",
    "recall",
    "latency",
    "score",
)

SETTING_PATTERNS = (
    "数据集",
    "benchmark",
    "dataset",
    "场景",
    "环境",
    "家庭",
    "仿真",
    "真实",
    "服务机器人",
    "多智能体",
)


def normalize_claim_frame_fields(frame: Dict[str, Any]) -> Dict[str, Any]:
    text = _frame_text(frame)
    dataset_or_setting = _short(frame.get("dataset_or_setting"))
    metric_or_signal = _short(frame.get("metric_or_signal"))
    result_or_gain = _short(frame.get("result_or_gain"))
    normalized = {
        "normalized_subject": _normalized_subject(frame, text),
        "method_family": _method_family(frame, text),
        "dataset": _dataset(dataset_or_setting),
        "metric": _metric(metric_or_signal or text),
        "claimed_gain": _claimed_gain(result_or_gain, text),
        "condition": _condition(dataset_or_setting, text),
        "polarity": _polarity(frame, text),
    }
    warnings = _normalization_warnings(frame, normalized)
    normalized["normalization_warnings"] = warnings
    return normalized


def apply_claim_frame_normalization(frame: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_claim_frame_fields(frame)
    frame.update({field: normalized[field] for field in SCHEMA_FIELDS})
    frame["normalization_warnings"] = normalized["normalization_warnings"]
    return frame


def _normalized_subject(frame: Dict[str, Any], text: str) -> str:
    for field in ("problem", "method", "short_claim"):
        value = _short(frame.get(field), 120)
        if value:
            return value
    return _short(text, 120)


def _method_family(frame: Dict[str, Any], text: str) -> str:
    raw = _short(frame.get("method")) or text
    raw_lower = raw.lower()
    for family, patterns in METHOD_FAMILY_PATTERNS:
        if any(pattern.lower() in raw_lower for pattern in patterns):
            return family
    if str(frame.get("claim_type") or "") == "method":
        return "method_unspecified"
    return ""


def _dataset(value: str) -> str:
    if not value:
        return ""
    explicit = _extract_after_label(value, ("数据集", "dataset", "benchmark"))
    return explicit or value


def _metric(value: str) -> str:
    hits = [term for term in METRIC_PATTERNS if term.lower() in value.lower()]
    deduped = []
    for term in unique_strings(hits):
        if any(term != other and term in other for other in hits):
            continue
        deduped.append(term)
    return "、".join(deduped[:6])


def _claimed_gain(result_or_gain: str, text: str) -> str:
    if result_or_gain:
        return result_or_gain
    if any(term in text for term in ("提升", "提高", "优于", "降低", "减少", "改进", "增强")):
        return _short(text, 120)
    return ""


def _condition(dataset_or_setting: str, text: str) -> str:
    if dataset_or_setting:
        return dataset_or_setting
    hits = [term for term in SETTING_PATTERNS if term.lower() in text.lower()]
    return "、".join(unique_strings(hits)[:6])


def _polarity(frame: Dict[str, Any], text: str) -> str:
    if str(frame.get("claim_type") or "") == "limitation":
        return "negative"
    if any(term in text for term in ("不足", "失败", "不能", "未验证", "缺乏", "下降", "无显著", "低于")):
        return "negative"
    if any(term in text for term in ("提升", "提高", "优于", "改进", "增强", "降低响应", "减少开销", "减少")):
        return "positive"
    return "neutral"


def _normalization_warnings(frame: Dict[str, Any], normalized: Dict[str, str]) -> List[str]:
    warnings = []
    claim_type = str(frame.get("claim_type") or "")
    if claim_type == "method" and not normalized["method_family"]:
        warnings.append("missing_method_family")
    if claim_type == "result" and not normalized["metric"]:
        warnings.append("missing_metric")
    if not normalized["normalized_subject"]:
        warnings.append("missing_normalized_subject")
    return warnings


def _frame_text(frame: Dict[str, Any]) -> str:
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


def _extract_after_label(value: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        match = re.search(rf"{re.escape(label)}[:：\\s]*([^,，;；。]+)", value, re.IGNORECASE)
        if match:
            return _short(match.group(1), 80)
    return ""


def _short(value: Any, limit: int = 120) -> str:
    return short_research_text(value, limit)
