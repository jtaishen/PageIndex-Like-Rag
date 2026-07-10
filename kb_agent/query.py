from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .llm import LLMError, generate_json_object
from .llm_policies import structured_json_generator
from .utils import compact_whitespace, string_list as _string_list, unique_strings as _unique_strings


INTENT_SPECS: Dict[str, Dict[str, List[str]]] = {
    "method": {
        "triggers": ["方法", "算法", "模型", "框架", "机制", "流程", "设计", "怎么做", "如何实现", "method", "algorithm", "model", "framework", "design"],
        "preferred_node_types": ["section", "paragraph", "abstract"],
        "target_sections": ["方法", "算法", "模型", "框架", "研究内容", "主要贡献", "系统设计", "method", "algorithm", "model", "framework"],
    },
    "experiment": {
        "triggers": ["实验", "评测", "评价", "指标", "结果", "基线", "数据集", "消融", "experiment", "evaluation", "result", "metric", "baseline"],
        "preferred_node_types": ["section", "paragraph", "table", "figure"],
        "target_sections": ["实验", "评测", "结果", "数据", "指标", "验证", "experiment", "evaluation", "result"],
    },
    "limitation": {
        "triggers": ["局限", "不足", "限制", "失败", "未来工作", "展望", "问题", "limitation", "weakness", "future work"],
        "preferred_node_types": ["section", "paragraph"],
        "target_sections": ["局限", "不足", "讨论", "结论", "展望", "未来工作", "limitation", "discussion", "conclusion"],
    },
    "citation": {
        "triggers": ["引用", "参考文献", "文献", "来源", "cite", "citation", "reference"],
        "preferred_node_types": ["reference", "paragraph", "section"],
        "target_sections": ["参考文献", "相关工作", "引用", "文献"],
    },
    "compare": {
        "triggers": ["比较", "对比", "区别", "差异", "异同", "相同", "不同", "compare", "comparison", "difference"],
        "preferred_node_types": ["section", "paragraph", "abstract"],
        "target_sections": ["方法", "实验", "结果", "结论", "贡献", "局限", "method", "experiment", "result", "conclusion"],
    },
    "review": {
        "triggers": ["综述", "研究现状", "进展", "梳理", "总结", "归纳", "路线", "survey", "review", "overview"],
        "preferred_node_types": ["abstract", "section", "paragraph"],
        "target_sections": ["摘要", "绪论", "相关工作", "方法", "结论", "展望", "abstract", "introduction", "related work", "method"],
    },
    "qa": {
        "triggers": [],
        "preferred_node_types": ["abstract", "section", "paragraph"],
        "target_sections": ["摘要", "研究内容", "方法", "结论"],
    },
}


def classify_query(
    query: str,
    *,
    use_llm: bool = True,
    require_llm: bool = False,
) -> Dict[str, Any]:
    """Build a stable query profile with optional LLM enhancement."""
    base = _rule_based_profile(query)
    if not use_llm:
        base["warnings"] = _unique_strings([*base["warnings"], "llm_disabled"])
        return base
    try:
        payload = _classify_with_llm(query, base)
        return _normalize_llm_profile(query, base, payload)
    except LLMError as exc:
        if require_llm:
            raise
        base["warnings"] = _unique_strings([*base["warnings"], f"llm_unavailable:{exc}"])
        base["llm_error"] = str(exc)
        return base


def focus_terms(query: str, extra_terms: Optional[Iterable[str]] = None, limit: int = 14) -> List[str]:
    terms: List[str] = []
    known_phrases = [
        "主要研究内容",
        "研究内容",
        "主要贡献",
        "研究背景",
        "研究目的",
        "任务规划",
        "任务分配",
        "协同调度",
        "动态角色",
        "参考文献",
        "关键词",
        "摘要",
        "结论",
        "局限",
        "实验结果",
    ]
    for phrase in known_phrases:
        if phrase in query:
            terms.append(phrase)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{5,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    if extra_terms:
        terms.extend(str(term) for term in extra_terms)
    stopwords = {
        "这篇", "论文", "什么", "是什么", "主要", "研究", "内容", "一下", "进行", "哪些",
        "the", "does", "what", "which", "paper", "use", "uses", "used",
    }
    result = []
    seen = set()
    for term in terms:
        cleaned = compact_whitespace(str(term))
        normalized = cleaned.lower() if cleaned.isascii() else cleaned
        if not cleaned or normalized in stopwords or normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result or [compact_whitespace(query)]


def rule_query_profile(query: str) -> Dict[str, Any]:
    """Return the deterministic profile used before optional LLM tree routing."""
    return _rule_based_profile(query)


def _rule_based_profile(query: str) -> Dict[str, Any]:
    normalized = compact_whitespace(query)
    intent = _detect_intent(normalized)
    spec = INTENT_SPECS[intent]
    warnings: List[str] = []
    if len(normalized) < 4:
        warnings.append("short_query")
    return {
        "schema": "query_profile.v1",
        "query": query,
        "intent": intent,
        "focus_terms": focus_terms(query, spec["target_sections"]),
        "preferred_node_types": list(spec["preferred_node_types"]),
        "target_sections": list(spec["target_sections"]),
        "filters": {},
        "source": "rule",
        "warnings": warnings,
        "llm_error": "",
    }


def _detect_intent(query: str) -> str:
    scores = {intent: 0 for intent in INTENT_SPECS}
    for intent, spec in INTENT_SPECS.items():
        for trigger in spec["triggers"]:
            if trigger and trigger.lower() in query.lower():
                scores[intent] += 1
    priority = ["compare", "review", "citation", "limitation", "experiment", "method"]
    for intent in priority:
        if scores[intent] > 0:
            return intent
    return "qa"


def _classify_with_llm(query: str, base: Dict[str, Any]) -> Dict[str, object]:
    system_prompt = (
        "你是论文知识库的查询意图分类器。只能输出 JSON object，不要输出 Markdown。"
        "intent 必须是 qa/method/experiment/limitation/citation/compare/review 之一。"
    )
    user_prompt = "\n".join(
        [
            f"用户查询：{query}",
            f"规则初判：{base}",
            "请返回：",
            '{"intent":"","focus_terms":[],"preferred_node_types":[],"target_sections":[],"filters":{},"warnings":[]}',
        ]
    )
    return structured_json_generator(
        "query_classification",
        "classify",
        json_generator=generate_json_object,
    )(system_prompt, user_prompt)


def _normalize_llm_profile(query: str, base: Dict[str, Any], payload: Dict[str, object]) -> Dict[str, Any]:
    intent = str(payload.get("intent") or base["intent"]).strip().lower()
    if intent not in INTENT_SPECS:
        intent = str(base["intent"])
    spec = INTENT_SPECS[intent]
    focus = _string_list(payload.get("focus_terms")) or base["focus_terms"]
    preferred = _string_list(payload.get("preferred_node_types")) or spec["preferred_node_types"]
    target = _string_list(payload.get("target_sections")) or spec["target_sections"]
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    warnings = _unique_strings([*base.get("warnings", []), *_string_list(payload.get("warnings"))])
    return {
        "schema": "query_profile.v1",
        "query": query,
        "intent": intent,
        "focus_terms": focus_terms(query, focus),
        "preferred_node_types": _unique_strings(preferred),
        "target_sections": _unique_strings(target),
        "filters": filters,
        "source": "llm",
        "warnings": warnings,
        "llm_error": "",
    }
