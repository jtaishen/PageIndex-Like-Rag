from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .answer_plan import (
    answer_plan_open_questions,
    answer_plan_summary,
    answer_plan_warning_tags,
    build_answer_plan_from_evidence,
)
from .claim_alignment import (
    build_claim_alignment,
    build_claim_relations,
    claim_alignment_summary,
    review_alignment_sections,
)
from .fact_audit import fact_audit_summary
from .knowledge_graph import graph_summary
from .llm import generate_json_object
from .llm_policies import structured_json_generator
from .query_log import write_query_log
from .task_compare import build_comparison_matrix
from .task_artifacts import (
    TASK_ARTIFACT_WHITELIST,
    TASK_ID_RE,
    get_task_artifact,
    new_task_id as _new_task_id,
    next_actions_artifact as _next_actions_artifact,
    open_questions_artifact as _open_questions_artifact,
    section_evidence_artifact as _section_evidence_artifact,
    selected_papers_artifact as _selected_papers_artifact,
    task_manifest as _manifest,
    task_state_root as _task_state_root,
    valid_task_artifact_name as _valid_task_artifact_name,
    write_task_artifacts as _write_task_artifacts,
)
from .task_contexts import (
    prepare_paper_contexts as _prepare_paper_contexts,
    select_papers as _select_papers,
)
from .task_evidence import (
    evidence_duplicate_summary as _evidence_duplicate_summary,
    flatten_dimension_evidence as _flatten_dimension_evidence,
    flatten_dimension_evidence_raw as _flatten_dimension_evidence_raw,
)
from .task_evidence_collection import (
    collect_dimension_evidence as _collect_dimension_evidence_core,
    collect_section_evidence as _collect_section_evidence_core,
    search_doc_evidence as _search_doc_evidence_core,
)
from .task_review_plan import build_review_outline
from .utils import unique_strings as _unique_strings


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
    evidence_quality = _evidence_duplicate_summary(_flatten_dimension_evidence_raw(evidence_by_dimension))
    warnings = [*prepare_warnings, *_fact_audit_warning_tags(audit)]
    if len(contexts) < 2:
        warnings.append("insufficient_papers_for_comparison")

    comparison = build_comparison_matrix(
        query,
        contexts,
        evidence_by_dimension,
        COMPARE_DIMENSIONS,
        warnings=warnings,
        use_llm=use_llm,
        require_llm=require_llm,
        json_generator=structured_json_generator(
            "compare",
            "legacy_dimensions",
            json_generator=generate_json_object,
        ),
    )
    matrix = comparison.matrix
    llm_error = comparison.llm_error

    coverage = _matrix_coverage(matrix)
    coverage["duplicate_evidence_removed"] = evidence_quality["duplicate_evidence_removed"]
    graph = graph_summary(db_path, doc_ids=[context["doc_id"] for context in contexts], include_conflicts=True)
    _apply_fact_audit_to_comparison(matrix, audit)
    _apply_graph_summary_to_comparison(matrix, graph)
    matrix["evidence_coverage"] = coverage
    matrix["evidence_quality"] = evidence_quality
    matrix["duplicate_evidence_removed"] = evidence_quality["duplicate_evidence_removed"]
    answer_plan = build_answer_plan_from_evidence(query, _flatten_dimension_evidence_raw(evidence_by_dimension))
    matrix["answer_plan_summary"] = answer_plan_summary(answer_plan)
    alignment = build_claim_alignment(query, contexts)
    relations = build_claim_relations(alignment)
    alignment_summary = claim_alignment_summary(alignment, relations)
    matrix["claim_alignment"] = alignment
    matrix["claim_relations"] = relations
    matrix["claim_alignment_summary"] = alignment_summary
    matrix["method_family_groups"] = alignment.get("method_family_groups") or []
    matrix["conflicting_claim_groups"] = alignment.get("conflicting_claim_groups") or []
    matrix["research_gap_candidates"] = alignment.get("research_gap_candidates") or []
    matrix["warnings"] = _unique_strings(
        [
            *matrix.get("warnings", []),
            *coverage["warnings"],
            *answer_plan_warning_tags(matrix["answer_plan_summary"]),
            *alignment_summary.get("warnings", []),
        ]
    )
    matrix["open_questions"] = _unique_strings(
        [
            *matrix.get("open_questions", []),
            *answer_plan_open_questions(matrix["answer_plan_summary"]),
            *_claim_alignment_open_questions(alignment_summary),
        ]
    )
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
    prefer_section_llm: bool = False,
) -> Dict[str, Any]:
    started = time.time()
    selected = _select_papers(db_path, topic, doc_ids, top_k_docs, search_mode)
    contexts, prepare_warnings = _prepare_paper_contexts(db_path, selected)
    audit = fact_audit_summary(db_path, doc_ids=[context["doc_id"] for context in contexts])
    section_evidence, evidence_quality = _collect_section_evidence(db_path, topic, contexts, search_mode)
    warnings = [*prepare_warnings, *_fact_audit_warning_tags(audit)]
    if not contexts:
        warnings.append("no_selected_papers")

    review_plan = build_review_outline(
        topic,
        contexts,
        section_evidence,
        REVIEW_SECTIONS,
        warnings=warnings,
        use_llm=use_llm,
        require_llm=require_llm,
        prefer_section_llm=prefer_section_llm,
        json_generator=structured_json_generator(
            "review_outline",
            "legacy_outline",
            json_generator=generate_json_object,
        ),
    )
    outline = review_plan.outline
    llm_error = review_plan.llm_error

    coverage = _outline_coverage(outline, section_evidence, evidence_quality)
    graph = graph_summary(db_path, doc_ids=[context["doc_id"] for context in contexts], include_conflicts=True)
    _apply_fact_audit_to_review(outline, audit)
    _apply_graph_summary_to_review(outline, graph)
    outline["evidence_coverage"] = coverage
    outline["evidence_quality"] = evidence_quality
    outline["duplicate_evidence_removed"] = evidence_quality["duplicate_evidence_removed"]
    answer_plan = build_answer_plan_from_evidence(topic, [item for items in section_evidence.values() for item in items])
    outline["answer_plan_summary"] = answer_plan_summary(answer_plan)
    alignment = build_claim_alignment(topic, contexts)
    relations = build_claim_relations(alignment)
    alignment_sections = review_alignment_sections(alignment, relations)
    outline.update(alignment_sections)
    outline["warnings"] = _unique_strings(
        [
            *outline.get("warnings", []),
            *coverage["warnings"],
            *answer_plan_warning_tags(outline["answer_plan_summary"]),
            *(alignment_sections.get("claim_alignment_summary") or {}).get("warnings", []),
        ]
    )
    outline["open_questions"] = _unique_strings(
        [
            *outline.get("open_questions", []),
            *answer_plan_open_questions(outline["answer_plan_summary"]),
            *_claim_alignment_open_questions(outline["claim_alignment_summary"]),
        ]
    )
    outline["review_partial_reasons"] = _review_partial_reasons(outline, contexts, coverage)
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


def _collect_dimension_evidence(
    db_path: Path,
    query: str,
    contexts: List[Dict[str, Any]],
    dimensions: List[Dict[str, Any]],
    search_mode: str,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    return _collect_dimension_evidence_core(
        db_path,
        query,
        contexts,
        dimensions,
        search_mode,
        search_evidence_fn=_search_doc_evidence,
    )


def _collect_section_evidence(
    db_path: Path,
    topic: str,
    contexts: List[Dict[str, Any]],
    search_mode: str,
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    return _collect_section_evidence_core(
        db_path,
        topic,
        contexts,
        REVIEW_SECTIONS,
        search_mode,
        search_evidence_fn=_search_doc_evidence,
    )


def _search_doc_evidence(db_path: Path, doc_id: str, query: str, top_k: int, search_mode: str = "hybrid") -> List[Dict[str, Any]]:
    return _search_doc_evidence_core(db_path, doc_id, query, top_k, search_mode=search_mode)


def _claim_alignment_open_questions(summary: Dict[str, Any]) -> List[str]:
    questions = []
    if int(summary.get("conflicting_group_count") or 0) > 0:
        questions.append("ClaimFrame 跨论文对齐发现可比冲突，需要核验证据链后再写成结论。")
    if int(summary.get("research_gap_count") or 0) > 0:
        questions.append("部分 ClaimFrame 对齐组证据不足，可作为综述中的研究空白候选。")
    return questions


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


def _graph_warning_tags(graph: Dict[str, Any]) -> List[str]:
    warnings = []
    if not graph.get("available"):
        warnings.append("claim_graph_unavailable")
        return warnings
    if graph.get("conflict_count", 0) > 0:
        warnings.append(f"claim_graph_conflicts:{graph.get('conflict_count')}")
    if graph.get("isolated_fact_count", 0) > 0:
        warnings.append(f"claim_graph_isolated_facts:{graph.get('isolated_fact_count')}")
    if float(graph.get("evidence_coverage_rate") or 0.0) < 1.0:
        warnings.append(f"claim_graph_evidence_coverage:{graph.get('evidence_coverage_rate')}")
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


def _apply_graph_summary_to_comparison(matrix: Dict[str, Any], graph: Dict[str, Any]) -> None:
    matrix["claim_graph"] = graph
    graph_warnings = _graph_warning_tags(graph)
    if not graph_warnings:
        return
    matrix["warnings"] = _unique_strings([*matrix.get("warnings", []), *graph_warnings])
    matrix["open_questions"] = _unique_strings(
        [
            *matrix.get("open_questions", []),
            (
                f"Claim Graph 中有 {graph.get('conflict_count', 0)} 个冲突、"
                f"{graph.get('isolated_fact_count', 0)} 个孤立事实，比较结论需要回到 evidence packet 核验。"
            ),
        ]
    )
    for dimension in matrix.get("dimensions") or []:
        if dimension.get("id") in {"innovation_overlap", "evidence_strength", "limitations"}:
            dimension["warnings"] = _unique_strings([*(dimension.get("warnings") or []), *graph_warnings])


def _apply_graph_summary_to_review(outline: Dict[str, Any], graph: Dict[str, Any]) -> None:
    outline["claim_graph"] = graph
    graph_warnings = _graph_warning_tags(graph)
    if not graph_warnings:
        return
    outline["warnings"] = _unique_strings([*outline.get("warnings", []), *graph_warnings])
    outline["open_questions"] = _unique_strings(
        [
            *outline.get("open_questions", []),
            (
                f"Claim Graph 提示 {graph.get('conflict_count', 0)} 个冲突和 "
                f"{graph.get('isolated_fact_count', 0)} 个孤立事实，综述写作前需要复核相关证据链。"
            ),
        ]
    )
    for section in outline.get("sections") or []:
        if section.get("section_id") in {"method_paradigms", "evaluation_evidence", "limitations_future"}:
            section["warnings"] = _unique_strings([*(section.get("warnings") or []), *graph_warnings])


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


def _outline_coverage(
    outline: Dict[str, Any],
    section_evidence: Dict[str, List[Dict[str, Any]]],
    evidence_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
        "pre_dedupe_count": (evidence_quality or {}).get("pre_dedupe_count", total_evidence),
        "post_dedupe_count": (evidence_quality or {}).get("post_dedupe_count", total_evidence),
        "duplicate_evidence_removed": (evidence_quality or {}).get("duplicate_evidence_removed", 0),
        "post_dedupe_duplicate_count": (evidence_quality or {}).get("post_dedupe_duplicate_count", 0),
        "duplicate_evidence_removed_by_section": (evidence_quality or {}).get("duplicate_evidence_removed_by_section", {}),
        "warnings": warnings,
    }


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


def _review_partial_reasons(outline: Dict[str, Any], contexts: List[Dict[str, Any]], coverage: Dict[str, Any]) -> List[str]:
    reasons = []
    if len(contexts) < 3:
        reasons.append("small_corpus")
    if coverage.get("missing_sections"):
        reasons.append("missing_section_evidence")
    if int(coverage.get("source_doc_count") or 0) < max(1, min(2, len(contexts))):
        reasons.append("low_source_doc_coverage")
    warnings = [str(item) for item in outline.get("warnings") or []]
    if any("citation_gap" in item or "引用缺口" in item for item in warnings):
        reasons.append("citation_relation_gaps")
    if any("局限" in item or "limitation" in item for item in warnings):
        reasons.append("missing_limitation_evidence")
    if int(coverage.get("post_dedupe_duplicate_count") or 0) > 0:
        reasons.append("duplicate_evidence")
    if outline.get("source") == "rule":
        reasons.append("rule_based_review_plan")
    if any("llm_unavailable" in item for item in warnings):
        reasons.append("llm_unavailable")
    if not reasons and (outline.get("status") == "partial" or warnings):
        reasons.append("quality_warnings")
    return _unique_strings(reasons)
