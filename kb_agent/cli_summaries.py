from __future__ import annotations

from typing import List


def extract_summary(result: dict) -> dict:
    innovation = result.get("innovation") or {}
    citation_map = result.get("citation_map") or {}
    return {
        "doc_id": result.get("doc_id"),
        "version_id": result.get("version_id"),
        "artifact_dir": result.get("artifact_dir"),
        "skipped": result.get("skipped", False),
        "innovation_path": result.get("innovation_path"),
        "citation_map_path": result.get("citation_map_path"),
        "innovation": {
            "schema": innovation.get("schema"),
            "status": innovation.get("status"),
            "item_count": len(innovation.get("items") or []),
            "warning_count": len(innovation.get("warnings") or []),
        },
        "citation_map": {
            "schema": citation_map.get("schema"),
            "status": citation_map.get("status"),
            "reference_count": len(citation_map.get("references") or []),
            "in_text_citation_count": len(citation_map.get("in_text_citations") or []),
            "relation_count": len(citation_map.get("relations") or []),
            "warning_count": len(citation_map.get("warnings") or []),
        },
        "llm_error": result.get("llm_error", ""),
    }


def fact_summary(result: dict) -> dict:
    report = result.get("fact_report") or {}
    claims = result.get("claims") or {}
    entities = result.get("entities") or {}
    relations = result.get("relations") or {}
    return {
        "schema": result.get("schema"),
        "doc_id": result.get("doc_id"),
        "version_id": result.get("version_id"),
        "artifact_dir": result.get("artifact_dir"),
        "skipped": result.get("skipped", False),
        "claims_path": result.get("claims_path"),
        "entities_path": result.get("entities_path"),
        "relations_path": result.get("relations_path"),
        "fact_graph_path": result.get("fact_graph_path"),
        "fact_report_path": result.get("fact_report_path"),
        "status": report.get("status") or claims.get("status"),
        "claim_count": report.get("claim_count", claims.get("count")),
        "entity_count": report.get("entity_count", entities.get("count")),
        "relation_count": report.get("relation_count", relations.get("count")),
        "low_confidence_count": report.get("low_confidence_count"),
        "no_evidence_count": report.get("no_evidence_count"),
        "evidence_unit_count": report.get("evidence_unit_count", 0),
        "source_kind_counts": report.get("source_kind_counts", {}),
        "claim_frame_count": report.get("claim_frame_count", 0),
        "verified_frame_rate": report.get("verified_frame_rate", 0.0),
        "unsupported_frame_count": report.get("unsupported_frame_count", 0),
        "trace_status_counts": report.get("trace_status_counts", {}),
        "support_status_counts": report.get("support_status_counts", {}),
        "missing_evidence_unit_count": report.get("missing_evidence_unit_count", 0),
        "missing_node_count": report.get("missing_node_count", 0),
        "missing_source_count": report.get("missing_source_count", 0),
        "low_quality_frame_count": report.get("low_quality_frame_count", 0),
        "noisy_frame_count": report.get("noisy_frame_count", 0),
        "ignored_noise_frame_count": report.get("ignored_noise_frame_count", 0),
        "top_frame_noise_reasons": report.get("top_frame_noise_reasons", []),
        "warnings": report.get("warnings") or [],
        "llm_error": result.get("llm_error", ""),
    }


def task_summary(result: dict) -> dict:
    coverage = {}
    if result.get("comparison_matrix"):
        coverage = result["comparison_matrix"].get("evidence_coverage", {})
    elif result.get("review_outline"):
        coverage = result["review_outline"].get("evidence_coverage", {})
    return {
        "task_id": result.get("task_id"),
        "task_type": result.get("task_type"),
        "status": result.get("status"),
        "query": result.get("query") or result.get("topic"),
        "selected_paper_count": (result.get("selected_papers") or {}).get("paper_count"),
        "artifact_paths": result.get("artifact_paths", {}),
        "evidence_coverage": coverage,
        "warning_count": len(_task_warnings(result)),
        "warnings": _task_warnings(result),
        "llm_error": result.get("llm_error", ""),
    }


def _task_warnings(result: dict) -> list:
    if result.get("comparison_matrix"):
        return result["comparison_matrix"].get("warnings", [])
    if result.get("review_outline"):
        return result["review_outline"].get("warnings", [])
    return []


def review_summary(result: dict) -> dict:
    report = result.get("review_report") or {}
    citation_check = result.get("citation_check") or {}
    return {
        "task_id": result.get("task_id"),
        "status": result.get("status") or report.get("status"),
        "drafted_section_count": result.get("drafted_section_count") or report.get("drafted_section_count"),
        "draft_quality_level": report.get("draft_quality_level", ""),
        "quality_reasons": report.get("quality_reasons") or [],
        "citation_coverage_score": report.get("citation_coverage_score") or citation_check.get("coverage_score"),
        "missing_ref_count": len(citation_check.get("missing_refs") or []),
        "unused_evidence_count": len(citation_check.get("unused_evidence") or []),
        "unsupported_paragraph_count": len(citation_check.get("unsupported_paragraphs") or []),
        "revision_actions": report.get("revision_actions") or report.get("next_actions") or [],
        "artifact_paths": result.get("artifact_paths", {}),
        "warnings": report.get("warnings", []),
        "llm_error": result.get("llm_error", ""),
    }


def eval_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "search_mode": result.get("search_mode"),
        "compare_modes": result.get("compare_modes") or [],
        "query_count": result.get("query_count"),
        "doc_recall_at_k": result.get("doc_recall_at_k"),
        "node_recall_at_k": result.get("node_recall_at_k"),
        "node_keyword_hit_rate": result.get("node_keyword_hit_rate"),
        "evidence_precision": result.get("evidence_precision"),
        "mrr": result.get("mrr"),
        "evidence_count": result.get("evidence_count"),
        "fallback_count": result.get("fallback_count"),
        "weak_parse_quality_count": result.get("weak_parse_quality_count"),
        "best_mode_by_node_recall": result.get("best_mode_by_node_recall"),
    }


def review_eval_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "task_id": result.get("task_id"),
        "status": result.get("status"),
        "section_count": result.get("section_count"),
        "drafted_section_count": result.get("drafted_section_count"),
        "citation_coverage_score": result.get("citation_coverage_score"),
        "missing_ref_count": result.get("missing_ref_count"),
        "unsupported_paragraph_count": result.get("unsupported_paragraph_count"),
        "warnings": result.get("warnings") or [],
    }


def memory_eval_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "status": result.get("status"),
        "memory_count": result.get("memory_count"),
        "expired_count": result.get("expired_count"),
        "duplicate_subject_count": result.get("duplicate_subject_count"),
        "suspected_pollution_count": result.get("suspected_pollution_count"),
        "resume_available": result.get("resume_available"),
        "warnings": result.get("warnings") or [],
    }


def fact_eval_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "status": result.get("status"),
        "doc_ids": result.get("doc_ids") or [],
        "total_fact_count": result.get("total_fact_count", 0),
        "claim_count": result.get("claim_count", 0),
        "entity_count": result.get("entity_count", 0),
        "relation_count": result.get("relation_count", 0),
        "evidence_coverage_rate": result.get("evidence_coverage_rate", 0.0),
        "low_confidence_rate": result.get("low_confidence_rate", 0.0),
        "duplicate_rate": result.get("duplicate_rate", 0.0),
        "table_backed_fact_count": result.get("table_backed_fact_count", 0),
        "table_backed_fact_rate": result.get("table_backed_fact_rate", 0.0),
        "warnings": result.get("warnings") or [],
    }


def fact_audit_cli_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "md_path": result.get("md_path"),
        "audit_id": result.get("audit_id"),
        "doc_ids": result.get("doc_ids") or [],
        "status": result.get("status"),
        "total_fact_count": result.get("total_fact_count", 0),
        "duplicate_group_count": result.get("duplicate_group_count", 0),
        "low_confidence_count": result.get("low_confidence_count", 0),
        "no_evidence_count": result.get("no_evidence_count", 0),
        "conflict_count": result.get("conflict_count", 0),
        "high_severity_conflict_count": result.get("high_severity_conflict_count", 0),
        "table_text_mismatch_count": result.get("table_text_mismatch_count", 0),
        "citation_gap_count": result.get("citation_gap_count", 0),
        "warnings": result.get("warnings") or [],
    }


def fact_conflicts_cli_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "audit_id": result.get("audit_id"),
        "audit_path": result.get("audit_path"),
        "doc_ids": result.get("doc_ids") or [],
        "severity": result.get("severity") or "",
        "count": result.get("count", 0),
        "high_severity_count": result.get("high_severity_count", 0),
        "conflicts": result.get("conflicts") or [],
        "warnings": result.get("warnings") or [],
    }


def graph_build_cli_summary(result: dict) -> dict:
    report = result.get("graph_report") or {}
    return {
        "schema": result.get("schema"),
        "graph_id": result.get("graph_id"),
        "graph_dir": result.get("graph_dir"),
        "knowledge_graph_path": result.get("knowledge_graph_path"),
        "graph_index_path": result.get("graph_index_path"),
        "graph_report_path": result.get("graph_report_path"),
        "doc_ids": report.get("doc_ids") or [],
        "node_count": report.get("node_count", 0),
        "edge_count": report.get("edge_count", 0),
        "conflict_count": report.get("conflict_count", 0),
        "isolated_fact_count": report.get("isolated_fact_count", 0),
        "evidence_coverage_rate": report.get("evidence_coverage_rate", 0.0),
        "warnings": result.get("warnings") or [],
    }


def quality_baseline_cli_summary(result: dict) -> dict:
    benchmark = result.get("benchmark") or {}
    embedding = result.get("embedding") or {}
    claim_frame_verification = result.get("claim_frame_verification") or {}
    memory_context = result.get("memory_context") or {}
    real_embedding = embedding.get("sentence_transformers") or {}
    llm_status = result.get("llm_status") or {}
    llm_baseline = result.get("llm_baseline") or {}
    stage_summary = llm_baseline.get("stage_summary") or {}
    llm_facts = llm_baseline.get("insights_and_facts") or {}
    llm_tasks = llm_baseline.get("tasks") or {}
    review_task = ((result.get("tasks") or {}).get("review") or {})
    review_draft = ((result.get("tasks") or {}).get("review_draft") or {})
    review_diagnostics = review_task.get("llm_diagnostics") or {}
    return {
        "schema": result.get("schema"),
        "code_version": result.get("code_version", ""),
        "git_commit": result.get("git_commit", ""),
        "feature_flags": result.get("feature_flags") or {},
        "is_current_code_baseline": bool(result.get("feature_flags", {}).get("review_draft_baseline")),
        "baseline_stale_reason": result.get("baseline_stale_reason", ""),
        "baseline_id": result.get("baseline_id"),
        "path": result.get("path"),
        "md_path": result.get("md_path"),
        "html_path": result.get("html_path"),
        "run_kind": result.get("run_kind", ""),
        "corpus_name": result.get("corpus_name", ""),
        "corpus_fingerprint": result.get("corpus_fingerprint", ""),
        "is_real_corpus": bool(result.get("is_real_corpus")),
        "corpus_path": result.get("corpus_path"),
        "doc_count": result.get("doc_count", 0),
        "pdf_count": result.get("pdf_count", 0),
        "best_search_mode": benchmark.get("best_mode_by_score"),
        "best_mode_by_node_recall": benchmark.get("best_mode_by_node_recall"),
        "real_embedding_provider": embedding.get("real_embedding_provider", ""),
        "real_embedding_status": embedding.get("real_embedding_status") or real_embedding.get("status", ""),
        "real_embedding_model": embedding.get("real_embedding_model", ""),
        "real_embedding_dim": embedding.get("real_embedding_dim", 0),
        "real_embedding_node_coverage": embedding.get("real_embedding_node_coverage", 0.0),
        "real_embedding_doc_coverage": embedding.get("real_embedding_doc_coverage", 0.0),
        "hybrid_embedding_provider": embedding.get("hybrid_embedding_provider", ""),
        "hybrid_embedding_model": embedding.get("hybrid_embedding_model", ""),
        "embedding_rebuild_needed": bool(embedding.get("embedding_rebuild_needed")),
        "llm_baseline_status": llm_baseline.get("status", ""),
        "llm_reachable": llm_status.get("reachable"),
        "llm_stage_status": {name: (stage.get("status") if isinstance(stage, dict) else "") for name, stage in stage_summary.items()},
        "llm_timeout_count": llm_baseline.get("timeout_count", 0),
        "llm_hard_timeout_count": llm_baseline.get("hard_timeout_count", 0),
        "llm_slow_call_count": llm_baseline.get("slow_call_count", 0),
        "llm_total_duration_ms": llm_baseline.get("total_llm_duration_ms", 0.0),
        "llm_budget_exhausted": bool(llm_baseline.get("budget_exhausted")),
        "llm_tree_used": ((llm_baseline.get("tree_search") or {}).get("llm_used_count")),
        "llm_tree_fallback": ((llm_baseline.get("tree_search") or {}).get("fallback_count")),
        "llm_fact_used": ((llm_baseline.get("insights_and_facts") or {}).get("llm_used_count")),
        "llm_facts_success_rate": llm_facts.get("llm_facts_success_rate", 0.0),
        "llm_facts_batch_timeout_count": llm_facts.get("llm_facts_batch_timeout_count", 0),
        "llm_compare_dimension_success_rate": llm_tasks.get("llm_compare_dimension_success_rate", 0.0),
        "llm_compare_dimension_timeout_count": llm_tasks.get("compare_dimension_timeout_count", 0),
        "review_llm_error": review_task.get("llm_error", ""),
        "review_fallback_mode": review_diagnostics.get("mode", ""),
        "review_retry_count": review_diagnostics.get("retry_count", 0),
        "review_partial_reasons": review_task.get("review_partial_reasons") or [],
        "review_draft_status": review_draft.get("status", ""),
        "review_draft_skip_reason": review_draft.get("review_draft_skip_reason", "") or review_draft.get("reason", ""),
        "review_draft_quality_level": review_draft.get("draft_quality_level", ""),
        "citation_coverage_score": review_draft.get("citation_coverage_score", 0.0),
        "missing_ref_count": review_draft.get("missing_ref_count", 0),
        "unsupported_paragraph_count": review_draft.get("unsupported_paragraph_count", 0),
        "optional_unused_evidence_count": review_draft.get("optional_unused_evidence_count", 0),
        "removed_paragraph_count": (review_draft.get("paragraph_support_report") or {}).get("removed_paragraph_count", 0),
        "drafted_section_count": review_draft.get("drafted_section_count", 0),
        "skipped_section_count": review_draft.get("skipped_section_count", 0),
        "review_draft_path": review_draft.get("review_draft_path", ""),
        "section_revision_actions": review_draft.get("section_revision_actions") or [],
        "top_review_blockers": result.get("top_review_blockers", []),
        "baseline_limitations": result.get("baseline_limitations", []),
        "llm_runtime_limitations": result.get("llm_runtime_limitations", []),
        "duplicate_evidence_removed": review_task.get("duplicate_evidence_removed", 0),
        "citation_gap_count_before": (result.get("fact_audit_delta") or {}).get("citation_gap_count_before", 0),
        "citation_gap_count_after": (result.get("fact_audit_delta") or {}).get("citation_gap_count_after", 0),
        "claim_frame_count": claim_frame_verification.get("frame_count", 0),
        "verified_frame_rate": claim_frame_verification.get("verified_frame_rate", 0.0),
        "unsupported_frame_count": claim_frame_verification.get("unsupported_frame_count", 0),
        "trace_status_counts": claim_frame_verification.get("trace_status_counts", {}),
        "support_status_counts": claim_frame_verification.get("support_status_counts", {}),
        "low_quality_frame_count": claim_frame_verification.get("low_quality_frame_count", 0),
        "noisy_frame_count": claim_frame_verification.get("noisy_frame_count", 0),
        "ignored_noise_frame_count": claim_frame_verification.get("ignored_noise_frame_count", 0),
        "missing_evidence_unit_count": claim_frame_verification.get("missing_evidence_unit_count", 0),
        "missing_node_count": claim_frame_verification.get("missing_node_count", 0),
        "missing_source_count": claim_frame_verification.get("missing_source_count", 0),
        "top_frame_noise_reasons": claim_frame_verification.get("top_frame_noise_reasons", []),
        "compiled_context_available": bool(memory_context.get("available")),
        "compiled_context_schema": memory_context.get("schema", ""),
        "selected_memory_count": memory_context.get("selected_memory_count", 0),
        "artifact_ref_count": memory_context.get("artifact_ref_count", 0),
        "filtered_memory_count": memory_context.get("filtered_memory_count", 0),
        "context_char_count": memory_context.get("context_char_count", 0),
        "memory_context_warnings": memory_context.get("warnings") or [],
        "tree_trace_completeness_before": ((result.get("tree_search") or {}).get("comparison_summary") or {}).get("rule_trace_completeness_avg", 0.0),
        "tree_trace_completeness_after": ((result.get("tree_search") or {}).get("comparison_summary") or {}).get("llm_trace_completeness_avg", 0.0),
        "compare_task_id": ((result.get("tasks") or {}).get("compare") or {}).get("task_id", ""),
        "review_task_id": review_task.get("task_id", ""),
        "claim_graph_id": (result.get("claim_graph") or {}).get("graph_id", ""),
        "warning_count": len(result.get("warnings") or []),
        "warnings": result.get("warnings") or [],
        "recommendations": result.get("recommendations") or [],
    }


def suite_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "suite_id": result.get("suite_id"),
        "name": result.get("name"),
        "query_count": result.get("query_count", 0),
        "sources": result.get("sources") or [],
        "warnings": result.get("warnings") or [],
    }


def benchmark_cli_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "md_path": result.get("md_path"),
        "benchmark_id": result.get("benchmark_id"),
        "suite_name": result.get("suite_name"),
        "compare_modes": result.get("compare_modes") or [],
        "query_count": result.get("query_count", 0),
        "best_mode_by_score": result.get("best_mode_by_score"),
        "best_mode_by_node_recall": result.get("best_mode_by_node_recall"),
        "summary": result.get("summary") or {},
        "warnings": result.get("warnings") or [],
    }


def failure_cli_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "next_actions_path": result.get("next_actions_path"),
        "benchmark_id": result.get("benchmark_id"),
        "status": result.get("status"),
        "failure_count": result.get("failure_count", 0),
        "reason_counts": result.get("reason_counts") or {},
        "next_actions": result.get("next_actions") or [],
    }


def case_cli_summary(result: dict) -> dict:
    graph = result.get("claim_graph") or {}
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "md_path": result.get("md_path"),
        "case_id": result.get("case_id"),
        "query": result.get("query"),
        "compare_modes": result.get("compare_modes") or [],
        "intent": (result.get("query_profile") or {}).get("intent"),
        "evidence_count": (result.get("evidence_summary") or {}).get("count", 0),
        "fact_match_count": (result.get("fact_matches") or {}).get("count", 0),
        "fact_conflict_count": (result.get("fact_conflicts") or {}).get("conflict_count", 0),
        "high_severity_fact_conflict_count": (result.get("fact_conflicts") or {}).get("high_severity_conflict_count", 0),
        "claim_graph_id": graph.get("graph_id", ""),
        "claim_graph_conflict_count": graph.get("conflict_count", 0),
        "claim_graph_isolated_fact_count": graph.get("isolated_fact_count", 0),
        "claim_graph_evidence_coverage_rate": graph.get("evidence_coverage_rate", 0.0),
        "warnings": result.get("warnings") or [],
    }


def comma_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_report_summary(report: dict) -> dict:
    return {
        "doc_id": report.get("doc_id"),
        "version_id": report.get("version_id"),
        "title": report.get("title"),
        "status": report.get("status"),
        "parser_name": report.get("parser_name"),
        "parser_version": report.get("parser_version"),
        "requested_pdf_parser": report.get("requested_pdf_parser"),
        "parser_chain": report.get("parser_chain") or [],
        "fallback_used": report.get("fallback_used", False),
        "external_parser_errors": report.get("external_parser_errors") or [],
        "adapter_statuses": report.get("adapter_statuses") or {},
        "warning_count": len(report.get("warnings") or []),
        "warnings": report.get("warnings") or [],
        "block_count": report.get("block_count"),
        "layout_block_count": report.get("layout_block_count"),
        "table_count": report.get("table_count"),
        "table_content_count": report.get("table_content_count"),
        "table_parse_score": report.get("table_parse_score"),
        "table_warning_count": report.get("table_warning_count"),
        "figure_count": report.get("figure_count"),
        "reference_section_count": report.get("reference_section_count"),
        "noise_removed_count": report.get("noise_removed_count"),
        "node_count": report.get("node_count"),
        "artifact_dir": report.get("artifact_dir"),
    }


def layout_summary(layout: dict) -> dict:
    blocks = layout.get("blocks") or []
    return {
        "schema": layout.get("schema"),
        "doc_id": layout.get("doc_id"),
        "version_id": layout.get("version_id"),
        "count": layout.get("count"),
        "type_counts": layout.get("type_counts") or {},
        "page_count": layout.get("page_count"),
        "sample": blocks[:10] if isinstance(blocks, list) else [],
    }
