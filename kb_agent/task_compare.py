from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .llm import LLMError, generate_json_object, llm_payload_metadata
from .task_evidence import evidence_confidence, normalize_evidence_refs
from .task_payloads import confidence, find_by_doc_id, find_by_id, llm_diagnostics, string_list, string_value
from .task_prompting import excerpt, format_contexts_for_prompt, format_dimension_contexts_for_prompt
from .utils import unique_strings


JsonGenerator = Callable[[str, str], Dict[str, object]]


@dataclass(frozen=True)
class ComparisonBuildResult:
    matrix: Dict[str, Any]
    llm_error: str
    llm_diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class ComparisonDimensionBuildResult:
    dimension: Dict[str, Any]
    llm_diagnostics: Dict[str, Any]


def build_comparison_matrix(
    query: str,
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
    dimensions: List[Dict[str, Any]],
    *,
    warnings: List[str],
    use_llm: bool = True,
    require_llm: bool = False,
    json_generator: JsonGenerator | None = None,
) -> ComparisonBuildResult:
    json_generator = json_generator or generate_json_object
    llm_error = ""
    diagnostics = llm_diagnostics("disabled" if not use_llm else "fallback_rule")
    if use_llm:
        try:
            matrix, diagnostics = _compare_with_dimension_llm(
                query,
                contexts,
                evidence_by_dimension,
                dimensions,
                warnings=warnings,
                json_generator=json_generator,
            )
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            warnings.append(f"llm_unavailable:{llm_error}")
            matrix = _rule_based_comparison(query, contexts, evidence_by_dimension, dimensions, warnings)
            diagnostics = llm_diagnostics("fallback_rule", error=exc)
    else:
        warnings.append("llm_disabled")
        matrix = _rule_based_comparison(query, contexts, evidence_by_dimension, dimensions, warnings)
    matrix["llm_diagnostics"] = diagnostics
    return ComparisonBuildResult(matrix=matrix, llm_error=llm_error, llm_diagnostics=diagnostics)


def build_comparison_dimension(
    query: str,
    dimension: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    evidence_by_doc: Dict[str, List[Dict[str, Any]]],
    *,
    json_generator: JsonGenerator | None = None,
) -> ComparisonDimensionBuildResult:
    """Generate and normalize exactly one comparison dimension."""
    generator = json_generator or generate_json_object
    dimension_id = str(dimension["id"])
    evidence_by_dimension = {dimension_id: evidence_by_doc}
    payload = _compare_dimension_with_llm(query, dimension, contexts, evidence_by_dimension, generator)
    metadata = llm_payload_metadata(payload)
    return ComparisonDimensionBuildResult(
        dimension=_normalize_comparison_dimension_payload(payload, dimension, contexts, evidence_by_dimension),
        llm_diagnostics=llm_diagnostics("staged_dimension_json", metadata=metadata),
    )


def _compare_with_dimension_llm(
    query: str,
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
    dimensions: List[Dict[str, Any]],
    *,
    warnings: List[str],
    json_generator: JsonGenerator,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    rule_matrix = _rule_based_comparison(query, contexts, evidence_by_dimension, dimensions, warnings)
    rule_dimensions = {str(item.get("id") or ""): item for item in rule_matrix.get("dimensions") or []}
    normalized_dimensions = []
    fallback_dimensions: List[str] = []
    timeout_count = 0
    retry_count = 0
    repair_used = False
    first_error_type = ""
    last_error: LLMError | None = None
    for dimension in dimensions:
        dimension_id = str(dimension["id"])
        try:
            payload = _compare_dimension_with_llm(query, dimension, contexts, evidence_by_dimension, json_generator)
            metadata = llm_payload_metadata(payload)
            retry_count += int(metadata.get("retry_count") or 0)
            repair_used = repair_used or bool(metadata.get("repair_used"))
            normalized = _normalize_comparison_dimension_payload(payload, dimension, contexts, evidence_by_dimension)
            normalized_dimensions.append(normalized)
        except LLMError as exc:
            last_error = exc
            if not first_error_type:
                first_error_type = exc.error_type
            retry_count += int(exc.metadata.get("retry_count") or 0)
            repair_used = repair_used or bool(exc.metadata.get("repair_used"))
            if exc.error_type == "request_timeout":
                timeout_count += 1
            fallback_dimensions.append(dimension_id)
            fallback = dict(rule_dimensions.get(dimension_id) or {})
            fallback["warnings"] = unique_strings([*fallback.get("warnings", []), f"llm_dimension_unavailable:{exc.error_type}"])
            normalized_dimensions.append(fallback)
    success_count = len(dimensions) - len(fallback_dimensions)
    if success_count == 0:
        error = last_error or LLMError("DeepSeek dimension compare failed.", error_type="all_compare_dimensions_failed")
        raise LLMError(
            "DeepSeek compare dimension extraction failed.",
            error_type=error.error_type,
            metadata={
                "dimension_success_count": 0,
                "dimension_timeout_count": timeout_count,
                "fallback_dimensions": fallback_dimensions,
                "retry_count": retry_count,
                "repair_used": repair_used,
                "first_error_type": first_error_type,
            },
        ) from error
    matrix_warnings = unique_strings(
        [
            *warnings,
            *(rule_matrix.get("warnings") or [] if fallback_dimensions else []),
            *(["dimension_json_partial"] if fallback_dimensions else []),
        ]
    )
    mode = "dimension_json" if not fallback_dimensions else "dimension_partial"
    diagnostics = llm_diagnostics(
        mode,
        metadata={
            "retry_count": retry_count,
            "repair_used": repair_used,
            "error_type": "",
            "first_error_type": first_error_type,
        },
        fallback_dimensions=fallback_dimensions,
        extra={
            "dimension_count": len(dimensions),
            "dimension_success_count": success_count,
            "dimension_timeout_count": timeout_count,
        },
    )
    return (
        {
            "schema": "comparison_matrix.v1",
            "status": "extracted" if not fallback_dimensions else "partial",
            "source": "llm_dimension",
            "query": query,
            "dimensions": normalized_dimensions,
            "open_questions": _comparison_open_questions(normalized_dimensions),
            "warnings": matrix_warnings,
            "created_at": time.time(),
        },
        diagnostics,
    )


def _compare_dimension_with_llm(
    query: str,
    dimension: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
    json_generator: JsonGenerator,
) -> Dict[str, object]:
    dimension_id = str(dimension["id"])
    system_prompt = (
        "你是严谨的论文比较分析助手。只比较一个指定维度，只能引用给定 node_id，"
        "不要编造，不要输出长正文。必须返回 JSON object，不要 Markdown。"
    )
    user_prompt = "\n".join(
        [
            f"比较主题：{query}",
            f"dimension_id: {dimension_id}",
            f"dimension_name: {dimension['name']}",
            "返回格式：",
            '{"id":"","synthesis":"","overlaps":[],"differences":[],'
            '"cells":[{"doc_id":"","claim":"","evidence":[],"confidence":0.0,"warnings":[]}],"warnings":[]}',
            "",
            "论文与本维度证据：",
            *format_dimension_contexts_for_prompt(contexts, dimension_id, evidence_by_dimension),
        ]
    )
    return json_generator(system_prompt, user_prompt)


def _compare_with_llm(
    query: str,
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
    dimensions: List[Dict[str, Any]],
    json_generator: JsonGenerator,
) -> Dict[str, object]:
    system_prompt = (
        "你是严谨的论文比较分析助手。只能基于给定论文卡片、创新点和证据节点比较，"
        "不要编造。必须返回 JSON object，不要返回 Markdown。"
    )
    user_prompt = "\n".join(
        [
            f"比较主题：{query}",
            "请按固定维度输出比较矩阵。返回格式：",
            '{"dimensions":[{"id":"","synthesis":"","overlaps":[],"differences":[],'
            '"cells":[{"doc_id":"","claim":"","evidence":[],"confidence":0.0,"warnings":[]}]}],'
            '"open_questions":[],"warnings":[]}',
            "",
            "固定维度：",
            *[f"- {item['id']}: {item['name']}" for item in dimensions],
            "",
            "论文与证据：",
            *format_contexts_for_prompt(contexts, evidence_by_dimension, dimensions),
        ]
    )
    return json_generator(system_prompt, user_prompt)


def _normalize_comparison_dimension_payload(
    payload: Dict[str, object],
    dimension: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    raw_dimension = payload.get("dimension") if isinstance(payload.get("dimension"), dict) else payload
    if not isinstance(raw_dimension, dict):
        raw_dimension = {}
    dimension_id = str(dimension["id"])
    raw_cells = raw_dimension.get("cells") if isinstance(raw_dimension, dict) else []
    cells = []
    for context in contexts:
        raw_cell = find_by_doc_id(raw_cells, context["doc_id"])
        fallback_evidence = evidence_by_dimension.get(dimension_id, {}).get(context["doc_id"], [])
        evidence = normalize_evidence_refs(raw_cell.get("evidence") if raw_cell else None, fallback_evidence)
        cell_warnings = string_list(raw_cell.get("warnings") if raw_cell else None)
        if not evidence:
            cell_warnings.append(f"missing_evidence:{dimension_id}:{context['doc_id']}")
        cells.append(
            {
                "doc_id": context["doc_id"],
                "title": context["title"],
                "claim": string_value(raw_cell.get("claim") if raw_cell else "") or _dimension_claim(context, dimension_id, evidence),
                "evidence": evidence,
                "evidence_count": len(evidence),
                "confidence": confidence(raw_cell.get("confidence") if raw_cell else None, evidence_confidence(evidence)),
                "warnings": unique_strings(cell_warnings),
            }
        )
    return {
        "id": dimension_id,
        "name": dimension["name"],
        "synthesis": string_value(raw_dimension.get("synthesis")) or _dimension_synthesis("", dimension, cells),
        "overlaps": string_list(raw_dimension.get("overlaps")) or _rule_overlaps(dimension_id, cells),
        "differences": string_list(raw_dimension.get("differences")) or _rule_differences(dimension_id, cells),
        "cells": cells,
        "warnings": unique_strings([*string_list(raw_dimension.get("warnings")), *[warning for cell in cells for warning in cell["warnings"]]]),
    }


def _normalize_comparison_payload(
    payload: Dict[str, object],
    query: str,
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
    dimensions: List[Dict[str, Any]],
    *,
    source: str,
    warnings: List[str],
) -> Dict[str, Any]:
    raw_dimensions = payload.get("dimensions") if isinstance(payload, dict) else []
    normalized_dimensions = []
    for dimension in dimensions:
        dimension_id = str(dimension["id"])
        raw_dimension = find_by_id(raw_dimensions, dimension_id)
        cells = []
        raw_cells = raw_dimension.get("cells") if isinstance(raw_dimension, dict) else []
        for context in contexts:
            raw_cell = find_by_doc_id(raw_cells, context["doc_id"])
            fallback_evidence = evidence_by_dimension.get(dimension_id, {}).get(context["doc_id"], [])
            evidence = normalize_evidence_refs(raw_cell.get("evidence") if raw_cell else None, fallback_evidence)
            cell_warnings = string_list(raw_cell.get("warnings") if raw_cell else None)
            if not evidence:
                cell_warnings.append(f"missing_evidence:{dimension_id}:{context['doc_id']}")
            cells.append(
                {
                    "doc_id": context["doc_id"],
                    "title": context["title"],
                    "claim": string_value(raw_cell.get("claim") if raw_cell else "") or _dimension_claim(context, dimension_id, evidence),
                    "evidence": evidence,
                    "evidence_count": len(evidence),
                    "confidence": confidence(raw_cell.get("confidence") if raw_cell else None, evidence_confidence(evidence)),
                    "warnings": unique_strings(cell_warnings),
                }
            )
        normalized_dimensions.append(
            {
                "id": dimension_id,
                "name": dimension["name"],
                "synthesis": string_value(raw_dimension.get("synthesis") if raw_dimension else "")
                or _dimension_synthesis(query, dimension, cells),
                "overlaps": string_list(raw_dimension.get("overlaps") if raw_dimension else None) or _rule_overlaps(dimension_id, cells),
                "differences": string_list(raw_dimension.get("differences") if raw_dimension else None) or _rule_differences(dimension_id, cells),
                "cells": cells,
                "warnings": unique_strings([warning for cell in cells for warning in cell["warnings"]]),
            }
        )
    return {
        "schema": "comparison_matrix.v1",
        "status": "extracted",
        "source": source,
        "query": query,
        "dimensions": normalized_dimensions,
        "open_questions": string_list(payload.get("open_questions")),
        "warnings": unique_strings([*warnings, *string_list(payload.get("warnings"))]),
        "created_at": time.time(),
    }


def _rule_based_comparison(
    query: str,
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
    dimensions: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    normalized_dimensions = []
    for dimension in dimensions:
        dimension_id = str(dimension["id"])
        cells = []
        for context in contexts:
            evidence = evidence_by_dimension.get(dimension_id, {}).get(context["doc_id"], [])
            cell_warnings = [] if evidence else [f"missing_evidence:{dimension_id}:{context['doc_id']}"]
            cells.append(
                {
                    "doc_id": context["doc_id"],
                    "title": context["title"],
                    "claim": _dimension_claim(context, dimension_id, evidence),
                    "evidence": evidence,
                    "evidence_count": len(evidence),
                    "confidence": evidence_confidence(evidence),
                    "warnings": cell_warnings,
                }
            )
        normalized_dimensions.append(
            {
                "id": dimension_id,
                "name": dimension["name"],
                "synthesis": _dimension_synthesis(query, dimension, cells),
                "overlaps": _rule_overlaps(dimension_id, cells),
                "differences": _rule_differences(dimension_id, cells),
                "cells": cells,
                "warnings": unique_strings([warning for cell in cells for warning in cell["warnings"]]),
            }
        )
    return {
        "schema": "comparison_matrix.v1",
        "status": "partial",
        "source": "rule",
        "query": query,
        "dimensions": normalized_dimensions,
        "open_questions": _comparison_open_questions(normalized_dimensions),
        "warnings": unique_strings([*warnings, "rule_based_comparison"]),
        "created_at": time.time(),
    }


def _dimension_claim(context: Dict[str, Any], dimension_id: str, evidence: List[Dict[str, Any]]) -> str:
    facts = context.get("facts") or {}
    if dimension_id == "innovation_overlap":
        claims = [
            excerpt(str(item.get("claim") or item.get("title") or ""), 180)
            for item in context.get("innovation", {}).get("items", [])
            if isinstance(item, dict) and (item.get("claim") or item.get("title"))
        ]
        if claims:
            return "；".join(claims[:2])
        fact_claims = [str(item.get("text") or "") for item in facts.get("top_claims", []) if isinstance(item, dict)]
        if fact_claims:
            return "；".join(excerpt(item, 180) for item in fact_claims[:2])
    if dimension_id == "limitations":
        limitations = context.get("innovation", {}).get("limitations") or []
        if limitations:
            return "；".join(excerpt(str(item), 160) for item in limitations[:2])
    if dimension_id == "evaluation_protocol":
        table_relations = [str(item.get("text") or "") for item in facts.get("top_table_relations", []) if isinstance(item, dict)]
        table_entities = [str(item.get("name") or "") for item in facts.get("top_table_entities", []) if isinstance(item, dict)]
        if table_relations:
            return "；".join(excerpt(item, 180) for item in table_relations[:2])
        if table_entities:
            return f"表格事实包含：{'、'.join(excerpt(item, 60) for item in table_entities[:4])}。"
    if dimension_id == "evidence_strength":
        quality = context.get("quality") or {}
        citation_count = len((context.get("citation_map") or {}).get("references") or [])
        table_fact_count = int(facts.get("table_backed_fact_count") or 0)
        return (
            f"章节数 {quality.get('section_count', 0)}，参考文献 {citation_count}，"
            f"表格事实 {table_fact_count} 条，质量告警 {len(quality.get('quality_warnings') or [])} 个。"
        )
    if evidence:
        return excerpt(str(evidence[0].get("excerpt") or ""), 260)
    description = str(context.get("description") or context.get("abstract") or "")
    return excerpt(description, 220) if description else "当前证据不足，无法稳定归纳。"


def _dimension_synthesis(query: str, dimension: Dict[str, Any], cells: List[Dict[str, Any]]) -> str:
    with_evidence = [cell for cell in cells if cell["evidence_count"] > 0]
    if not with_evidence:
        return f"围绕“{query}”的{dimension['name']}维度缺少可用证据。"
    return f"{dimension['name']}维度已有 {len(with_evidence)} 篇论文的证据，适合做带引用的进一步比较。"


def _rule_overlaps(dimension_id: str, cells: List[Dict[str, Any]]) -> List[str]:
    with_evidence = [cell for cell in cells if cell["evidence_count"] > 0]
    if len(with_evidence) < 2:
        return ["当前证据不足，暂不能稳定判断重叠点。"]
    if dimension_id == "method_paradigm":
        return ["多篇论文都围绕任务规划方法、系统框架或算法流程展开。"]
    if dimension_id == "problem_setting":
        return ["多篇论文都关注任务规划或任务分配中的动态性、协同性和约束处理。"]
    if dimension_id == "innovation_overlap":
        return ["多篇论文都声称在规划流程、任务分配或协同机制上有方法贡献。"]
    return ["多篇论文在该维度均检索到相关证据，可继续人工精读确认重叠关系。"]


def _rule_differences(dimension_id: str, cells: List[Dict[str, Any]]) -> List[str]:
    with_evidence = [cell for cell in cells if cell["evidence_count"] > 0]
    if len(with_evidence) < 2:
        return ["当前证据不足，暂不能稳定判断差异点。"]
    if dimension_id == "evidence_strength":
        return ["不同论文的章节完整度、参考文献数量和质量告警不同，证据强度需要分开标注。"]
    return ["规则版仅依据证据节点列出候选差异，语义差异需要 LLM 或人工复核。"]


def _comparison_open_questions(dimensions: List[Dict[str, Any]]) -> List[str]:
    questions = []
    for dimension in dimensions:
        if dimension.get("warnings"):
            questions.append(f"{dimension['name']}维度缺少部分论文证据，需要补充检索或人工确认。")
    return unique_strings(questions)
