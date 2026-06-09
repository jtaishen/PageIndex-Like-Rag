from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .artifacts import get_citation_map, get_doc_card, get_innovations, get_parse_quality
from .config import DEFAULT_DB_PATH, PROJECT_ROOT
from .fact_audit import fact_audit_summary
from .facts import fact_summary_for_doc
from .insights import extract_doc_insights
from .llm import LLMError, generate_json_object
from .query_log import write_query_log
from .search import get_evidence, search_documents, search_nodes
from .tree_search import tree_search
from .utils import compact_whitespace, stable_id, write_json


COMPARE_DIMENSIONS = [
    {
        "id": "problem_setting",
        "name": "问题设定",
        "search_terms": ["问题", "挑战", "研究背景", "任务规划", "任务分配"],
    },
    {
        "id": "method_paradigm",
        "name": "方法范式",
        "search_terms": ["方法", "算法", "模型", "框架", "规划"],
    },
    {
        "id": "evaluation_protocol",
        "name": "数据与评测",
        "search_terms": ["实验", "评测", "指标", "结果", "数据"],
    },
    {
        "id": "innovation_overlap",
        "name": "创新点重叠",
        "search_terms": ["创新", "贡献", "提出", "研究内容", "主要贡献"],
    },
    {
        "id": "limitations",
        "name": "局限与失败模式",
        "search_terms": ["局限", "不足", "失败", "展望", "未来工作"],
    },
    {
        "id": "evidence_strength",
        "name": "证据强度",
        "search_terms": ["实验结果", "评估", "参考文献", "结论", "验证"],
    },
]

REVIEW_SECTIONS = [
    {
        "section_id": "background_problem",
        "title": "研究背景与问题定义",
        "purpose": "界定主题范围、核心任务和主要挑战。",
        "search_terms": ["研究背景", "问题", "挑战", "任务规划", "任务分配"],
    },
    {
        "section_id": "method_paradigms",
        "title": "方法范式与系统框架",
        "purpose": "归纳不同论文采用的方法范式、系统组件和规划流程。",
        "search_terms": ["方法", "框架", "模型", "算法", "规划流程"],
    },
    {
        "section_id": "coordination_mechanisms",
        "title": "任务分解、分配与协同机制",
        "purpose": "整理任务分解、动态分配、调度和协同执行机制。",
        "search_terms": ["任务分解", "任务分配", "调度", "协同", "重分配"],
    },
    {
        "section_id": "evaluation_evidence",
        "title": "实验评测与证据强度",
        "purpose": "比较实验设置、评价指标、结果和证据充分性。",
        "search_terms": ["实验", "评测", "指标", "结果", "基线"],
    },
    {
        "section_id": "limitations_future",
        "title": "局限性与未来方向",
        "purpose": "汇总已有工作的不足、失败模式和后续研究机会。",
        "search_terms": ["局限", "不足", "展望", "未来工作", "限制"],
    },
]

TASK_ARTIFACT_WHITELIST = {
    "manifest.json",
    "selected_papers.json",
    "comparison_matrix.json",
    "review_outline.json",
    "review_draft.md",
    "citation_check.json",
    "review_report.json",
    "open_questions.json",
    "next_actions.json",
}
TASK_ID_RE = re.compile(r"^task_[0-9a-f]{12}$")


def compare_papers(
    db_path: Path,
    query: str,
    *,
    doc_ids: Optional[List[str]] = None,
    top_k_docs: int = 5,
    use_llm: bool = True,
    require_llm: bool = False,
    search_mode: str = "hybrid",
) -> Dict[str, Any]:
    started = time.time()
    selected = _select_papers(db_path, query, doc_ids, top_k_docs, search_mode)
    contexts, prepare_warnings = _prepare_paper_contexts(db_path, selected)
    audit = fact_audit_summary(db_path, doc_ids=[context["doc_id"] for context in contexts])
    evidence_by_dimension = _collect_dimension_evidence(db_path, query, contexts, COMPARE_DIMENSIONS, search_mode)
    warnings = [*prepare_warnings, *_fact_audit_warning_tags(audit)]
    if len(contexts) < 2:
        warnings.append("insufficient_papers_for_comparison")

    llm_error = ""
    if use_llm:
        try:
            payload = _compare_with_llm(query, contexts, evidence_by_dimension)
            matrix = _normalize_comparison_payload(
                payload,
                query,
                contexts,
                evidence_by_dimension,
                source="llm",
                warnings=warnings,
            )
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            warnings.append(f"llm_unavailable:{llm_error}")
            matrix = _rule_based_comparison(query, contexts, evidence_by_dimension, warnings)
    else:
        warnings.append("llm_disabled")
        matrix = _rule_based_comparison(query, contexts, evidence_by_dimension, warnings)

    coverage = _matrix_coverage(matrix)
    _apply_fact_audit_to_comparison(matrix, audit)
    matrix["evidence_coverage"] = coverage
    matrix["warnings"] = _unique_strings([*matrix.get("warnings", []), *coverage["warnings"]])
    matrix["status"] = "partial" if matrix["warnings"] or matrix.get("source") == "rule" else "extracted"

    task_id = _new_task_id("compare", query, [context["doc_id"] for context in contexts])
    matrix["task_id"] = task_id
    selected_artifact = _selected_papers_artifact(task_id, "compare", query, contexts)
    open_questions = _open_questions_artifact(task_id, matrix.get("open_questions", []), coverage, matrix["warnings"])
    next_actions = _next_actions_artifact(task_id, "compare", coverage, matrix["warnings"])
    manifest = _manifest(task_id, "compare", query, matrix["status"], matrix["warnings"])
    paths = _write_task_artifacts(
        db_path,
        task_id,
        manifest=manifest,
        selected_papers=selected_artifact,
        comparison_matrix=matrix,
        open_questions=open_questions,
        next_actions=next_actions,
    )
    _log_task_query(
        db_path,
        operation="compare",
        query=query,
        search_mode=search_mode,
        task_id=task_id,
        contexts=contexts,
        status=matrix["status"],
        warnings=matrix["warnings"],
        coverage=coverage,
        evidence=_flatten_dimension_evidence(evidence_by_dimension),
        started=started,
        llm_error=llm_error,
    )
    return {
        "schema": "task_result.v1",
        "task_id": task_id,
        "task_type": "compare",
        "status": matrix["status"],
        "query": query,
        "selected_papers": selected_artifact,
        "comparison_matrix": matrix,
        "open_questions": open_questions,
        "next_actions": next_actions,
        "artifact_paths": paths,
        "llm_error": llm_error,
    }


def generate_review_plan(
    db_path: Path,
    topic: str,
    *,
    doc_ids: Optional[List[str]] = None,
    top_k_docs: int = 8,
    use_llm: bool = True,
    require_llm: bool = False,
    search_mode: str = "hybrid",
) -> Dict[str, Any]:
    started = time.time()
    selected = _select_papers(db_path, topic, doc_ids, top_k_docs, search_mode)
    contexts, prepare_warnings = _prepare_paper_contexts(db_path, selected)
    audit = fact_audit_summary(db_path, doc_ids=[context["doc_id"] for context in contexts])
    section_evidence = _collect_section_evidence(db_path, topic, contexts, search_mode)
    warnings = [*prepare_warnings, *_fact_audit_warning_tags(audit)]
    if not contexts:
        warnings.append("no_selected_papers")

    llm_error = ""
    if use_llm:
        try:
            payload = _review_plan_with_llm(topic, contexts, section_evidence)
            outline = _normalize_review_payload(
                payload,
                topic,
                contexts,
                section_evidence,
                source="llm",
                warnings=warnings,
            )
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            warnings.append(f"llm_unavailable:{llm_error}")
            outline = _rule_based_review_plan(topic, contexts, section_evidence, warnings)
    else:
        warnings.append("llm_disabled")
        outline = _rule_based_review_plan(topic, contexts, section_evidence, warnings)

    coverage = _outline_coverage(outline, section_evidence)
    _apply_fact_audit_to_review(outline, audit)
    outline["evidence_coverage"] = coverage
    outline["warnings"] = _unique_strings([*outline.get("warnings", []), *coverage["warnings"]])
    outline["status"] = "partial" if outline["warnings"] or outline.get("source") == "rule" else "extracted"

    task_id = _new_task_id("review", topic, [context["doc_id"] for context in contexts])
    outline["task_id"] = task_id
    selected_artifact = _selected_papers_artifact(task_id, "review", topic, contexts)
    section_artifacts = {
        section_id: _section_evidence_artifact(task_id, section_id, topic, evidence)
        for section_id, evidence in section_evidence.items()
    }
    open_questions = _open_questions_artifact(task_id, outline.get("open_questions", []), coverage, outline["warnings"])
    next_actions = _next_actions_artifact(task_id, "review", coverage, outline["warnings"])
    manifest = _manifest(task_id, "review", topic, outline["status"], outline["warnings"])
    paths = _write_task_artifacts(
        db_path,
        task_id,
        manifest=manifest,
        selected_papers=selected_artifact,
        review_outline=outline,
        section_evidence=section_artifacts,
        open_questions=open_questions,
        next_actions=next_actions,
    )
    _log_task_query(
        db_path,
        operation="generate-review",
        query=topic,
        search_mode=search_mode,
        task_id=task_id,
        contexts=contexts,
        status=outline["status"],
        warnings=outline["warnings"],
        coverage=coverage,
        evidence=[item for items in section_evidence.values() for item in items],
        started=started,
        llm_error=llm_error,
    )
    return {
        "schema": "task_result.v1",
        "task_id": task_id,
        "task_type": "review",
        "status": outline["status"],
        "topic": topic,
        "selected_papers": selected_artifact,
        "review_outline": outline,
        "section_evidence": section_artifacts,
        "open_questions": open_questions,
        "next_actions": next_actions,
        "artifact_paths": paths,
        "llm_error": llm_error,
    }


def get_task_artifact(db_path: Path, task_id: str, name: str) -> Dict[str, Any]:
    if not _valid_task_artifact_name(name):
        raise ValueError(f"Unsupported task artifact name: {name}")
    if task_id != "current" and not TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"Unsupported task id: {task_id}")
    root = _task_state_root(db_path)
    if task_id == "current":
        path = root / "current_task.json"
    else:
        path = root / task_id / name
    if not path.exists():
        raise FileNotFoundError(f"Task artifact not found: {path}")
    text = path.read_text(encoding="utf-8")
    content: Any = text
    if path.suffix == ".json":
        content = json.loads(text)
    return {
        "task_id": task_id,
        "name": name,
        "path": str(path),
        "content": content,
    }


def _select_papers(
    db_path: Path,
    query: str,
    doc_ids: Optional[List[str]],
    top_k_docs: int,
    search_mode: str,
) -> List[Dict[str, Any]]:
    if doc_ids:
        return [{"doc_id": doc_id, "score": None, "node_matches": None} for doc_id in _unique_strings(doc_ids)]
    route_mode = "hybrid" if search_mode == "tree" else search_mode
    return search_documents(db_path, query, top_k=max(1, top_k_docs), search_mode=route_mode)


def _prepare_paper_contexts(db_path: Path, selected: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    contexts = []
    warnings: List[str] = []
    for item in selected:
        doc_id = str(item.get("doc_id") or "")
        if not doc_id:
            continue
        try:
            card = get_doc_card(db_path, doc_id)
            quality = get_parse_quality(db_path, doc_id)
            innovation, insight_warnings = _read_or_extract_insights(db_path, doc_id)
            citation_map = get_citation_map(db_path, doc_id)
            facts = fact_summary_for_doc(db_path, doc_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            warnings.append(f"paper_prepare_failed:{doc_id}:{exc}")
            continue
        warnings.extend(insight_warnings)
        contexts.append(
            {
                "doc_id": doc_id,
                "title": card.get("title") or doc_id,
                "path": card.get("path") or "",
                "abstract": card.get("abstract") or "",
                "description": card.get("description") or card.get("summary") or "",
                "keywords": card.get("keywords") or [],
                "quality": quality,
                "innovation": innovation,
                "citation_map": citation_map,
                "facts": facts,
                "route_score": item.get("score"),
                "node_matches": item.get("node_matches"),
            }
        )
    return contexts, _unique_strings(warnings)


def _read_or_extract_insights(db_path: Path, doc_id: str) -> tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    try:
        innovation = get_innovations(db_path, doc_id)
        citation_map = get_citation_map(db_path, doc_id)
    except (FileNotFoundError, KeyError, ValueError):
        innovation = {}
        citation_map = {}
    if innovation.get("schema") == "innovation.v1" and citation_map.get("schema") == "citation_map.v1":
        return innovation, warnings
    result = extract_doc_insights(db_path, doc_id, force=True, use_llm=False)
    warnings.append(f"insights_rule_refreshed:{doc_id}")
    return result["innovation"], warnings


def _collect_dimension_evidence(
    db_path: Path,
    query: str,
    contexts: List[Dict[str, Any]],
    dimensions: List[Dict[str, Any]],
    search_mode: str,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for dimension in dimensions:
        dimension_id = str(dimension["id"])
        result[dimension_id] = {}
        terms = " ".join(str(term) for term in dimension["search_terms"])
        search_query = f"{query} {terms}"
        for context in contexts:
            evidence = _search_doc_evidence(db_path, context["doc_id"], search_query, top_k=4, search_mode=search_mode)
            if not evidence:
                evidence = _innovation_evidence_for_dimension(context, dimension_id)[:3]
            result[dimension_id][context["doc_id"]] = evidence[:4]
    return result


def _collect_section_evidence(
    db_path: Path,
    topic: str,
    contexts: List[Dict[str, Any]],
    search_mode: str,
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for section in REVIEW_SECTIONS:
        section_id = str(section["section_id"])
        terms = " ".join(str(term) for term in section["search_terms"])
        search_query = f"{topic} {terms}"
        evidence: List[Dict[str, Any]] = []
        for context in contexts:
            evidence.extend(_search_doc_evidence(db_path, context["doc_id"], search_query, top_k=3, search_mode=search_mode))
        result[section_id] = _dedupe_evidence(evidence)[:12]
    return result


def _search_doc_evidence(db_path: Path, doc_id: str, query: str, top_k: int, search_mode: str = "hybrid") -> List[Dict[str, Any]]:
    if search_mode == "tree":
        trace = tree_search(db_path, doc_id, query, budget=top_k, use_llm=False, search_mode="hybrid")
        return _dedupe_evidence(list(trace.get("evidence") or []))
    results = search_nodes(db_path, query, doc_id=doc_id, top_k=top_k, search_mode=search_mode)
    packets = []
    for result in results:
        packets.extend(packet.to_dict() for packet in get_evidence(db_path, result.doc_id, [result.node_id]))
    return _dedupe_evidence(packets)


def _compare_with_llm(
    query: str,
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
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
            *[f"- {item['id']}: {item['name']}" for item in COMPARE_DIMENSIONS],
            "",
            "论文与证据：",
            *_format_contexts_for_prompt(contexts, evidence_by_dimension),
        ]
    )
    return generate_json_object(system_prompt, user_prompt)


def _review_plan_with_llm(
    topic: str,
    contexts: List[Dict[str, Any]],
    section_evidence: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, object]:
    system_prompt = (
        "你是严谨的论文综述规划助手。只能基于给定论文和证据节点生成综述大纲，"
        "不要写完整正文，不要编造。必须返回 JSON object，不要返回 Markdown。"
    )
    user_prompt = "\n".join(
        [
            f"综述主题：{topic}",
            "请生成综述规划。返回格式：",
            '{"title":"","scope":"","sections":[{"section_id":"","title":"","purpose":"",'
            '"paper_ids":[],"evidence":[],"warnings":[]}],"open_questions":[],"warnings":[]}',
            "",
            "建议章节：",
            *[f"- {item['section_id']}: {item['title']}，{item['purpose']}" for item in REVIEW_SECTIONS],
            "",
            "论文：",
            *_format_papers_for_prompt(contexts),
            "",
            "章节证据：",
            *_format_section_evidence_for_prompt(section_evidence),
        ]
    )
    return generate_json_object(system_prompt, user_prompt)


def _rule_based_comparison(
    query: str,
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
    warnings: List[str],
) -> Dict[str, Any]:
    dimensions = []
    for dimension in COMPARE_DIMENSIONS:
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
                    "confidence": _evidence_confidence(evidence),
                    "warnings": cell_warnings,
                }
            )
        dimensions.append(
            {
                "id": dimension_id,
                "name": dimension["name"],
                "synthesis": _dimension_synthesis(query, dimension, cells),
                "overlaps": _rule_overlaps(dimension_id, cells),
                "differences": _rule_differences(dimension_id, cells),
                "cells": cells,
                "warnings": _unique_strings([warning for cell in cells for warning in cell["warnings"]]),
            }
        )
    return {
        "schema": "comparison_matrix.v1",
        "status": "partial",
        "source": "rule",
        "query": query,
        "dimensions": dimensions,
        "open_questions": _comparison_open_questions(dimensions),
        "warnings": _unique_strings([*warnings, "rule_based_comparison"]),
        "created_at": time.time(),
    }


def _normalize_comparison_payload(
    payload: Dict[str, object],
    query: str,
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
    *,
    source: str,
    warnings: List[str],
) -> Dict[str, Any]:
    raw_dimensions = payload.get("dimensions") if isinstance(payload, dict) else []
    dimensions = []
    for dimension in COMPARE_DIMENSIONS:
        dimension_id = str(dimension["id"])
        raw_dimension = _find_by_id(raw_dimensions, dimension_id)
        cells = []
        raw_cells = raw_dimension.get("cells") if isinstance(raw_dimension, dict) else []
        for context in contexts:
            raw_cell = _find_by_doc_id(raw_cells, context["doc_id"])
            fallback_evidence = evidence_by_dimension.get(dimension_id, {}).get(context["doc_id"], [])
            evidence = _normalize_evidence_refs(raw_cell.get("evidence") if raw_cell else None, fallback_evidence)
            cell_warnings = _string_list(raw_cell.get("warnings") if raw_cell else None)
            if not evidence:
                cell_warnings.append(f"missing_evidence:{dimension_id}:{context['doc_id']}")
            cells.append(
                {
                    "doc_id": context["doc_id"],
                    "title": context["title"],
                    "claim": _string_value(raw_cell.get("claim") if raw_cell else "") or _dimension_claim(context, dimension_id, evidence),
                    "evidence": evidence,
                    "evidence_count": len(evidence),
                    "confidence": _confidence(raw_cell.get("confidence") if raw_cell else None, _evidence_confidence(evidence)),
                    "warnings": _unique_strings(cell_warnings),
                }
            )
        dimensions.append(
            {
                "id": dimension_id,
                "name": dimension["name"],
                "synthesis": _string_value(raw_dimension.get("synthesis") if raw_dimension else "")
                or _dimension_synthesis(query, dimension, cells),
                "overlaps": _string_list(raw_dimension.get("overlaps") if raw_dimension else None) or _rule_overlaps(dimension_id, cells),
                "differences": _string_list(raw_dimension.get("differences") if raw_dimension else None) or _rule_differences(dimension_id, cells),
                "cells": cells,
                "warnings": _unique_strings([warning for cell in cells for warning in cell["warnings"]]),
            }
        )
    return {
        "schema": "comparison_matrix.v1",
        "status": "extracted",
        "source": source,
        "query": query,
        "dimensions": dimensions,
        "open_questions": _string_list(payload.get("open_questions")),
        "warnings": _unique_strings([*warnings, *_string_list(payload.get("warnings"))]),
        "created_at": time.time(),
    }


def _rule_based_review_plan(
    topic: str,
    contexts: List[Dict[str, Any]],
    section_evidence: Dict[str, List[Dict[str, Any]]],
    warnings: List[str],
) -> Dict[str, Any]:
    sections = []
    for spec in REVIEW_SECTIONS:
        evidence = section_evidence.get(str(spec["section_id"]), [])
        paper_ids = _unique_strings(str(item.get("doc_id") or "") for item in evidence if item.get("doc_id"))
        section_warnings = [] if evidence else [f"missing_section_evidence:{spec['section_id']}"]
        sections.append(
            {
                "section_id": spec["section_id"],
                "title": spec["title"],
                "purpose": spec["purpose"],
                "paper_ids": paper_ids,
                "evidence": evidence,
                "evidence_count": len(evidence),
                "source_doc_count": len(set(paper_ids)),
                "warnings": section_warnings,
            }
        )
    return {
        "schema": "review_outline.v1",
        "status": "partial",
        "source": "rule",
        "topic": topic,
        "title": f"{topic}研究综述规划",
        "scope": _review_scope(topic, contexts),
        "sections": sections,
        "open_questions": _review_open_questions(sections),
        "warnings": _unique_strings([*warnings, "rule_based_review_plan"]),
        "created_at": time.time(),
    }


def _normalize_review_payload(
    payload: Dict[str, object],
    topic: str,
    contexts: List[Dict[str, Any]],
    section_evidence: Dict[str, List[Dict[str, Any]]],
    *,
    source: str,
    warnings: List[str],
) -> Dict[str, Any]:
    raw_sections = payload.get("sections") if isinstance(payload, dict) else []
    sections = []
    for spec in REVIEW_SECTIONS:
        section_id = str(spec["section_id"])
        raw_section = _find_by_id(raw_sections, section_id, key="section_id")
        fallback_evidence = section_evidence.get(section_id, [])
        evidence = _normalize_evidence_refs(raw_section.get("evidence") if raw_section else None, fallback_evidence)
        paper_ids = _string_list(raw_section.get("paper_ids") if raw_section else None)
        if not paper_ids:
            paper_ids = _unique_strings(str(item.get("doc_id") or "") for item in evidence if item.get("doc_id"))
        section_warnings = _string_list(raw_section.get("warnings") if raw_section else None)
        if not evidence:
            section_warnings.append(f"missing_section_evidence:{section_id}")
        sections.append(
            {
                "section_id": section_id,
                "title": _string_value(raw_section.get("title") if raw_section else "") or spec["title"],
                "purpose": _string_value(raw_section.get("purpose") if raw_section else "") or spec["purpose"],
                "paper_ids": paper_ids,
                "evidence": evidence,
                "evidence_count": len(evidence),
                "source_doc_count": len(set(paper_ids)),
                "warnings": _unique_strings(section_warnings),
            }
        )
    return {
        "schema": "review_outline.v1",
        "status": "extracted",
        "source": source,
        "topic": topic,
        "title": _string_value(payload.get("title")) or f"{topic}研究综述规划",
        "scope": _string_value(payload.get("scope")) or _review_scope(topic, contexts),
        "sections": sections,
        "open_questions": _string_list(payload.get("open_questions")),
        "warnings": _unique_strings([*warnings, *_string_list(payload.get("warnings"))]),
        "created_at": time.time(),
    }


def _format_contexts_for_prompt(
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> List[str]:
    lines: List[str] = []
    for context in contexts:
        lines.append(f"doc_id: {context['doc_id']}")
        lines.append(f"title: {context['title']}")
        lines.append(f"description: {_excerpt(context.get('description', ''), 600)}")
        lines.append(f"keywords: {context.get('keywords', [])}")
        lines.append(f"innovation_items: {_format_innovations(context.get('innovation', {}))}")
        lines.append(f"facts: {_format_fact_summary(context.get('facts', {}))}")
        for dimension in COMPARE_DIMENSIONS:
            dimension_id = str(dimension["id"])
            lines.append(f"dimension: {dimension_id}")
            for evidence in evidence_by_dimension.get(dimension_id, {}).get(context["doc_id"], [])[:3]:
                lines.append(_format_evidence_line(evidence))
        lines.append("")
    return lines


def _format_papers_for_prompt(contexts: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for context in contexts:
        lines.append(f"- doc_id: {context['doc_id']}")
        lines.append(f"  title: {context['title']}")
        lines.append(f"  description: {_excerpt(context.get('description', ''), 500)}")
        lines.append(f"  innovations: {_format_innovations(context.get('innovation', {}), limit=4)}")
        lines.append(f"  facts: {_format_fact_summary(context.get('facts', {}), limit=4)}")
    return lines


def _format_section_evidence_for_prompt(section_evidence: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = []
    for section_id, items in section_evidence.items():
        lines.append(f"section_id: {section_id}")
        for evidence in items[:8]:
            lines.append(_format_evidence_line(evidence))
        lines.append("")
    return lines


def _format_evidence_line(evidence: Dict[str, Any]) -> str:
    return (
        f"- node_id={evidence.get('node_id')} doc_id={evidence.get('doc_id')} "
        f"path={evidence.get('node_path')} page={evidence.get('page_range')} "
        f"excerpt={_excerpt(str(evidence.get('excerpt') or ''), 360)}"
    )


def _format_innovations(innovation: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    items = []
    for item in (innovation.get("items") or [])[:limit]:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "title": item.get("title") or "",
                "type": item.get("type") or "",
                "claim": _excerpt(str(item.get("claim") or ""), 260),
            }
        )
    return items


def _format_fact_summary(facts: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
    if not facts or not facts.get("available"):
        return {"available": False}
    return {
        "available": True,
        "claim_count": facts.get("claim_count", 0),
        "entity_count": facts.get("entity_count", 0),
        "relation_count": facts.get("relation_count", 0),
        "table_backed_fact_count": facts.get("table_backed_fact_count", 0),
        "table_entity_count": facts.get("table_entity_count", 0),
        "table_relation_count": facts.get("table_relation_count", 0),
        "top_claims": facts.get("top_claims", [])[:limit],
        "top_entities": facts.get("top_entities", [])[:limit],
        "top_table_entities": facts.get("top_table_entities", [])[:limit],
        "top_table_relations": facts.get("top_table_relations", [])[:limit],
    }


def _dimension_claim(context: Dict[str, Any], dimension_id: str, evidence: List[Dict[str, Any]]) -> str:
    facts = context.get("facts") or {}
    if dimension_id == "innovation_overlap":
        claims = [
            _excerpt(str(item.get("claim") or item.get("title") or ""), 180)
            for item in context.get("innovation", {}).get("items", [])
            if isinstance(item, dict) and (item.get("claim") or item.get("title"))
        ]
        if claims:
            return "；".join(claims[:2])
        fact_claims = [str(item.get("text") or "") for item in facts.get("top_claims", []) if isinstance(item, dict)]
        if fact_claims:
            return "；".join(_excerpt(item, 180) for item in fact_claims[:2])
    if dimension_id == "limitations":
        limitations = context.get("innovation", {}).get("limitations") or []
        if limitations:
            return "；".join(_excerpt(str(item), 160) for item in limitations[:2])
    if dimension_id == "evaluation_protocol":
        table_relations = [str(item.get("text") or "") for item in facts.get("top_table_relations", []) if isinstance(item, dict)]
        table_entities = [str(item.get("name") or "") for item in facts.get("top_table_entities", []) if isinstance(item, dict)]
        if table_relations:
            return "；".join(_excerpt(item, 180) for item in table_relations[:2])
        if table_entities:
            return f"表格事实包含：{'、'.join(_excerpt(item, 60) for item in table_entities[:4])}。"
    if dimension_id == "evidence_strength":
        quality = context.get("quality") or {}
        citation_count = len((context.get("citation_map") or {}).get("references") or [])
        table_fact_count = int(facts.get("table_backed_fact_count") or 0)
        return (
            f"章节数 {quality.get('section_count', 0)}，参考文献 {citation_count}，"
            f"表格事实 {table_fact_count} 条，质量告警 {len(quality.get('quality_warnings') or [])} 个。"
        )
    if evidence:
        return _excerpt(str(evidence[0].get("excerpt") or ""), 260)
    description = str(context.get("description") or context.get("abstract") or "")
    return _excerpt(description, 220) if description else "当前证据不足，无法稳定归纳。"


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


def _review_scope(topic: str, contexts: List[Dict[str, Any]]) -> str:
    titles = "、".join(str(context["title"]) for context in contexts[:5])
    if titles:
        return f"围绕“{topic}”，当前候选论文包括：{titles}。"
    return f"围绕“{topic}”，当前知识库没有检索到可用候选论文。"


def _comparison_open_questions(dimensions: List[Dict[str, Any]]) -> List[str]:
    questions = []
    for dimension in dimensions:
        if dimension.get("warnings"):
            questions.append(f"{dimension['name']}维度缺少部分论文证据，需要补充检索或人工确认。")
    return _unique_strings(questions)


def _review_open_questions(sections: List[Dict[str, Any]]) -> List[str]:
    questions = []
    for section in sections:
        if section.get("warnings"):
            questions.append(f"{section['title']}章节缺少足够证据，需要补充文献或重新解析。")
    return _unique_strings(questions)


def _fact_audit_warning_tags(audit: Dict[str, Any]) -> List[str]:
    warnings = []
    if audit.get("conflict_count", 0) > 0:
        warnings.append(f"fact_audit_conflicts:{audit.get('conflict_count')}")
    if audit.get("high_severity_conflict_count", 0) > 0:
        warnings.append(f"fact_audit_high_conflicts:{audit.get('high_severity_conflict_count')}")
    if audit.get("table_text_mismatch_count", 0) > 0:
        warnings.append(f"fact_audit_table_text_mismatch:{audit.get('table_text_mismatch_count')}")
    if audit.get("citation_gap_count", 0) > 0:
        warnings.append(f"fact_audit_citation_gaps:{audit.get('citation_gap_count')}")
    if audit.get("no_evidence_count", 0) > 0:
        warnings.append(f"fact_audit_no_evidence:{audit.get('no_evidence_count')}")
    return warnings


def _apply_fact_audit_to_comparison(matrix: Dict[str, Any], audit: Dict[str, Any]) -> None:
    matrix["fact_audit"] = audit
    audit_warnings = _fact_audit_warning_tags(audit)
    if not audit_warnings:
        return
    matrix["warnings"] = _unique_strings([*matrix.get("warnings", []), *audit_warnings])
    questions = [
        f"事实审计发现 {audit.get('conflict_count', 0)} 个冲突和 {audit.get('citation_gap_count', 0)} 个引用关系缺口，需要回到 evidence packet 人工确认。"
    ]
    matrix["open_questions"] = _unique_strings([*matrix.get("open_questions", []), *questions])
    for dimension in matrix.get("dimensions") or []:
        if dimension.get("id") in {"evidence_strength", "limitations"}:
            dimension["warnings"] = _unique_strings([*(dimension.get("warnings") or []), *audit_warnings])


def _apply_fact_audit_to_review(outline: Dict[str, Any], audit: Dict[str, Any]) -> None:
    outline["fact_audit"] = audit
    audit_warnings = _fact_audit_warning_tags(audit)
    if not audit_warnings:
        return
    outline["warnings"] = _unique_strings([*outline.get("warnings", []), *audit_warnings])
    questions = [
        f"事实层存在 {audit.get('conflict_count', 0)} 个冲突、{audit.get('table_text_mismatch_count', 0)} 个表格-正文不一致和 {audit.get('citation_gap_count', 0)} 个引用缺口，综述写作前需要复核。"
    ]
    outline["open_questions"] = _unique_strings([*outline.get("open_questions", []), *questions])
    for section in outline.get("sections") or []:
        if section.get("section_id") in {"evaluation_evidence", "limitations_future"}:
            section["warnings"] = _unique_strings([*(section.get("warnings") or []), *audit_warnings])


def _matrix_coverage(matrix: Dict[str, Any]) -> Dict[str, Any]:
    dimensions = matrix.get("dimensions") or []
    cell_count = 0
    cells_with_evidence = 0
    missing_cells = []
    doc_ids = set()
    for dimension in dimensions:
        for cell in dimension.get("cells") or []:
            cell_count += 1
            doc_id = str(cell.get("doc_id") or "")
            if doc_id:
                doc_ids.add(doc_id)
            if cell.get("evidence"):
                cells_with_evidence += 1
            else:
                missing_cells.append({"dimension_id": dimension.get("id"), "doc_id": doc_id})
    warnings = []
    if missing_cells:
        warnings.append("comparison_has_missing_evidence")
    return {
        "schema": "evidence_coverage.v1",
        "dimension_count": len(dimensions),
        "cell_count": cell_count,
        "cells_with_evidence": cells_with_evidence,
        "missing_cells": missing_cells,
        "source_doc_count": len(doc_ids),
        "warnings": warnings,
    }


def _outline_coverage(outline: Dict[str, Any], section_evidence: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    sections = outline.get("sections") or []
    missing_sections = []
    source_doc_ids = set()
    total_evidence = 0
    for section in sections:
        section_id = str(section.get("section_id") or "")
        evidence = section.get("evidence") or section_evidence.get(section_id, [])
        total_evidence += len(evidence)
        for item in evidence:
            if item.get("doc_id"):
                source_doc_ids.add(str(item["doc_id"]))
        if not evidence:
            missing_sections.append(section_id)
    warnings = []
    if missing_sections:
        warnings.append("review_has_missing_section_evidence")
    return {
        "schema": "evidence_coverage.v1",
        "section_count": len(sections),
        "total_evidence_count": total_evidence,
        "source_doc_count": len(source_doc_ids),
        "missing_sections": missing_sections,
        "warnings": warnings,
    }


def _selected_papers_artifact(
    task_id: str,
    task_type: str,
    query: str,
    contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    papers = []
    for context in contexts:
        citation_map = context.get("citation_map") or {}
        innovation = context.get("innovation") or {}
        facts = context.get("facts") or {}
        papers.append(
            {
                "doc_id": context["doc_id"],
                "title": context["title"],
                "path": context["path"],
                "description": context.get("description") or "",
                "abstract": context.get("abstract") or "",
                "keywords": context.get("keywords") or [],
                "quality_warnings": (context.get("quality") or {}).get("quality_warnings") or [],
                "innovation_status": innovation.get("status") or "",
                "innovation_count": len(innovation.get("items") or []),
                "citation_count": len(citation_map.get("references") or []),
                "fact_available": bool(facts.get("available")),
                "claim_count": facts.get("claim_count", 0),
                "entity_count": facts.get("entity_count", 0),
                "relation_count": facts.get("relation_count", 0),
                "table_backed_fact_count": facts.get("table_backed_fact_count", 0),
                "route_score": context.get("route_score"),
                "node_matches": context.get("node_matches"),
            }
        )
    return {
        "schema": "selected_papers.v1",
        "task_id": task_id,
        "task_type": task_type,
        "query": query,
        "papers": papers,
        "paper_count": len(papers),
        "created_at": time.time(),
    }


def _section_evidence_artifact(
    task_id: str,
    section_id: str,
    topic: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    doc_ids = _unique_strings(str(item.get("doc_id") or "") for item in evidence if item.get("doc_id"))
    return {
        "schema": "section_evidence.v1",
        "task_id": task_id,
        "section_id": section_id,
        "topic": topic,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "source_doc_count": len(doc_ids),
        "source_doc_ids": doc_ids,
        "created_at": time.time(),
    }


def _open_questions_artifact(
    task_id: str,
    questions: Any,
    coverage: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    items = _string_list(questions)
    if coverage.get("missing_cells"):
        items.append("部分比较单元缺少证据，需要补充检索或人工阅读。")
    if coverage.get("missing_sections"):
        items.append("部分综述章节缺少证据，需要补充论文或重新解析。")
    if not items and warnings:
        items.append("当前任务存在质量告警，需要确认是否影响结论可信度。")
    return {
        "schema": "open_questions.v1",
        "task_id": task_id,
        "items": _unique_strings(items),
        "created_at": time.time(),
    }


def _next_actions_artifact(
    task_id: str,
    task_type: str,
    coverage: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    actions = []
    if warnings:
        actions.append("查看 warnings，确认是否需要重新同步或重新抽取论文工件。")
    if coverage.get("missing_cells") or coverage.get("missing_sections"):
        actions.append("针对缺证据维度补充关键词检索，必要时人工指定 doc_id。")
    if coverage.get("source_doc_count", 0) < 2 and task_type in {"compare", "review"}:
        actions.append("补充更多同主题论文后重新运行任务。")
    if task_type == "review":
        actions.append("基于 section_evidence 逐节撰写综述正文，并做引用一致性检查。")
    else:
        actions.append("基于 comparison_matrix 复核差异点，再决定是否进入综述规划。")
    return {
        "schema": "next_actions.v1",
        "task_id": task_id,
        "items": _unique_strings(actions),
        "created_at": time.time(),
    }


def _manifest(task_id: str, task_type: str, query: str, status: str, warnings: List[str]) -> Dict[str, Any]:
    return {
        "schema": "task_manifest.v1",
        "task_id": task_id,
        "task_type": task_type,
        "query": query,
        "status": status,
        "warnings": _unique_strings(warnings),
        "created_at": time.time(),
    }


def _write_task_artifacts(
    db_path: Path,
    task_id: str,
    *,
    manifest: Dict[str, Any],
    selected_papers: Dict[str, Any],
    open_questions: Dict[str, Any],
    next_actions: Dict[str, Any],
    comparison_matrix: Optional[Dict[str, Any]] = None,
    review_outline: Optional[Dict[str, Any]] = None,
    section_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, str]:
    root = _task_state_root(db_path)
    task_dir = root / task_id
    paths = {
        "state_root": str(root),
        "task_dir": str(task_dir),
    }
    write_json(task_dir / "manifest.json", manifest)
    write_json(task_dir / "selected_papers.json", selected_papers)
    write_json(task_dir / "open_questions.json", open_questions)
    write_json(task_dir / "next_actions.json", next_actions)
    paths["manifest"] = str(task_dir / "manifest.json")
    paths["selected_papers"] = str(task_dir / "selected_papers.json")
    paths["open_questions"] = str(task_dir / "open_questions.json")
    paths["next_actions"] = str(task_dir / "next_actions.json")
    if comparison_matrix is not None:
        write_json(task_dir / "comparison_matrix.json", comparison_matrix)
        paths["comparison_matrix"] = str(task_dir / "comparison_matrix.json")
    if review_outline is not None:
        write_json(task_dir / "review_outline.json", review_outline)
        paths["review_outline"] = str(task_dir / "review_outline.json")
    if section_evidence:
        section_dir = task_dir / "section_evidence"
        for section_id, payload in section_evidence.items():
            path = section_dir / f"{section_id}.json"
            write_json(path, payload)
            paths[f"section_evidence/{section_id}.json"] = str(path)
    write_json(
        root / "current_task.json",
        {
            "schema": "current_task.v1",
            "task_id": task_id,
            "task_type": manifest["task_type"],
            "query": manifest["query"],
            "status": manifest["status"],
            "task_dir": str(task_dir),
            "updated_at": time.time(),
        },
    )
    paths["current_task"] = str(root / "current_task.json")
    return paths


def _log_task_query(
    db_path: Path,
    *,
    operation: str,
    query: str,
    search_mode: str,
    task_id: str,
    contexts: List[Dict[str, Any]],
    status: str,
    warnings: List[str],
    coverage: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    started: float,
    llm_error: str = "",
) -> None:
    doc_ids = _unique_strings(
        [
            *(str(context.get("doc_id") or "") for context in contexts),
            *(str(item.get("doc_id") or "") for item in evidence if isinstance(item, dict)),
        ]
    )
    node_ids = _unique_strings(
        str(item.get("node_id") or "")
        for item in evidence
        if isinstance(item, dict) and item.get("node_id")
    )
    log_warnings = list(warnings)
    if llm_error:
        log_warnings.append(f"llm_unavailable:{llm_error}")
    write_query_log(
        db_path,
        operation=operation,
        query=query,
        intent="compare" if operation == "compare" else "review",
        search_mode=search_mode,
        status=status or "ok",
        task_id=task_id,
        docs_used=doc_ids,
        nodes_used=node_ids,
        latency_ms=round((time.time() - started) * 1000, 3),
        warnings=log_warnings,
        metrics={
            "context_count": len(contexts),
            "evidence_count": len(evidence),
            "coverage": coverage,
            "llm_error": bool(llm_error),
        },
    )


def _flatten_dimension_evidence(evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for by_doc in evidence_by_dimension.values():
        for items in by_doc.values():
            evidence.extend(items)
    return _dedupe_evidence(evidence)


def _task_state_root(db_path: Path) -> Path:
    resolved = db_path.expanduser().resolve()
    if resolved == DEFAULT_DB_PATH.expanduser().resolve():
        return PROJECT_ROOT / ".kb_state"
    return resolved.parent / ".kb_state"


def _new_task_id(task_type: str, query: str, doc_ids: List[str]) -> str:
    return stable_id("task", task_type, query, ",".join(doc_ids), time.time(), length=12)


def _valid_task_artifact_name(name: str) -> bool:
    if name == "current_task.json":
        return True
    if name in TASK_ARTIFACT_WHITELIST:
        return True
    if name.startswith("section_evidence/") and name.endswith(".json"):
        parts = Path(name).parts
        return len(parts) == 2 and parts[0] == "section_evidence" and ".." not in parts
    if name.startswith("section_drafts/") and (name.endswith(".json") or name.endswith(".md")):
        parts = Path(name).parts
        return len(parts) == 2 and parts[0] == "section_drafts" and ".." not in parts
    return False


def _innovation_evidence_for_dimension(context: Dict[str, Any], dimension_id: str) -> List[Dict[str, Any]]:
    result = []
    for item in context.get("innovation", {}).get("items", []):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if dimension_id == "limitations" and item_type != "limitation":
            continue
        if dimension_id == "evaluation_protocol" and item_type != "result":
            continue
        for evidence in item.get("evidence") or []:
            if isinstance(evidence, dict):
                enriched = dict(evidence)
                enriched.setdefault("doc_id", context["doc_id"])
                enriched.setdefault("title", context["title"])
                enriched.setdefault("path", context["path"])
                result.append(enriched)
    return _dedupe_evidence(result)


def _normalize_evidence_refs(value: object, fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not value:
        return fallback[:4]
    by_node_id = {str(item.get("node_id") or ""): item for item in fallback}
    result = []
    if isinstance(value, list):
        for raw in value:
            node_id = ""
            if isinstance(raw, str):
                node_id = raw
            elif isinstance(raw, dict):
                node_id = str(raw.get("node_id") or raw.get("id") or "")
            if node_id and node_id in by_node_id:
                result.append(by_node_id[node_id])
    return _dedupe_evidence(result or fallback)[:4]


def _dedupe_evidence(evidence: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for item in evidence:
        node_id = str(item.get("node_id") or "")
        doc_id = str(item.get("doc_id") or "")
        key = (doc_id, node_id)
        if not node_id or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _find_by_id(raw_items: object, expected_id: str, key: str = "id") -> Dict[str, Any]:
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict) and str(item.get(key) or "") == expected_id:
                return item
    return {}


def _find_by_doc_id(raw_items: object, doc_id: str) -> Dict[str, Any]:
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict) and str(item.get("doc_id") or "") == doc_id:
                return item
    return {}


def _evidence_confidence(evidence: List[Dict[str, Any]]) -> float:
    if len(evidence) >= 3:
        return 0.75
    if len(evidence) >= 1:
        return 0.6
    return 0.25


def _confidence(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _string_value(value: object) -> str:
    return compact_whitespace(str(value)) if value is not None else ""


def _string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = _string_value(item)
        if text:
            result.append(text)
    return result


def _unique_strings(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _excerpt(text: str, max_chars: int) -> str:
    clean = compact_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + " ..."
