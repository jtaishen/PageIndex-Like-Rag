from __future__ import annotations

from html import escape
from typing import Any, Dict, Iterable, List

from .answer_plan import answer_plan_rollup
from .claim_alignment import claim_alignment_rollup


def baseline_markdown(report: Dict[str, Any]) -> str:
    answer_plan = answer_plan_rollup(_answer_plan_summaries(report))
    alignment = claim_alignment_rollup(_claim_alignment_summaries(report))
    lines = [
        "# Quality Baseline",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- code_version: `{report.get('code_version')}`",
        f"- git_commit: `{report.get('git_commit')}`",
        f"- is_current_code_baseline: `{report.get('is_current_code_baseline')}`",
        f"- baseline_id: `{report.get('baseline_id')}`",
        f"- run_kind: `{report.get('run_kind')}`",
        f"- corpus_fingerprint: `{report.get('corpus_fingerprint')}`",
        f"- corpus_path: `{report.get('corpus_path')}`",
        f"- doc_count: `{report.get('doc_count')}`",
        f"- pdf_count: `{report.get('pdf_count')}`",
        f"- best_search_mode: `{(report.get('benchmark') or {}).get('best_mode_by_score', '')}`",
        f"- llm_baseline_status: `{(report.get('llm_baseline') or {}).get('status', '')}`",
        f"- llm_reachable: `{(report.get('llm_status') or {}).get('reachable', '')}`",
        f"- llm_timeout_count: `{(report.get('llm_baseline') or {}).get('timeout_count', 0)}`",
        f"- llm_hard_timeout_count: `{(report.get('llm_baseline') or {}).get('hard_timeout_count', 0)}`",
        f"- llm_slow_call_count: `{(report.get('llm_baseline') or {}).get('slow_call_count', 0)}`",
        f"- llm_budget_exhausted: `{(report.get('llm_baseline') or {}).get('budget_exhausted', False)}`",
        f"- llm_facts_success_rate: `{((report.get('llm_baseline') or {}).get('insights_and_facts') or {}).get('llm_facts_success_rate', 0.0)}`",
        f"- llm_compare_dimension_success_rate: `{((report.get('llm_baseline') or {}).get('tasks') or {}).get('llm_compare_dimension_success_rate', 0.0)}`",
        f"- review_draft_status: `{((report.get('tasks') or {}).get('review_draft') or {}).get('status', '')}`",
        f"- review_draft_skip_reason: `{((report.get('tasks') or {}).get('review_draft') or {}).get('review_draft_skip_reason', '')}`",
        f"- review_draft_quality_level: `{((report.get('tasks') or {}).get('review_draft') or {}).get('draft_quality_level', '')}`",
        f"- skipped_section_count: `{((report.get('tasks') or {}).get('review_draft') or {}).get('skipped_section_count', 0)}`",
        f"- removed_paragraph_count: `{(((report.get('tasks') or {}).get('review_draft') or {}).get('paragraph_support_report') or {}).get('removed_paragraph_count', 0)}`",
        f"- optional_unused_evidence_count: `{((report.get('tasks') or {}).get('review_draft') or {}).get('optional_unused_evidence_count', 0)}`",
        f"- citation_coverage_score: `{((report.get('tasks') or {}).get('review_draft') or {}).get('citation_coverage_score', 0.0)}`",
        f"- real_embedding_provider: `{(report.get('embedding') or {}).get('real_embedding_provider', '')}`",
        f"- real_embedding_status: `{(report.get('embedding') or {}).get('real_embedding_status') or (report.get('embedding') or {}).get('sentence_transformers', {}).get('status', '')}`",
        f"- real_embedding_model: `{(report.get('embedding') or {}).get('real_embedding_model', '')}`",
        f"- real_embedding_node_coverage: `{(report.get('embedding') or {}).get('real_embedding_node_coverage', 0.0)}`",
        f"- hybrid_embedding_provider: `{(report.get('embedding') or {}).get('hybrid_embedding_provider', '')}`",
        f"- embedding_rebuild_needed: `{(report.get('embedding') or {}).get('embedding_rebuild_needed', False)}`",
        f"- claim_frame_count: `{(report.get('claim_frame_verification') or {}).get('frame_count', 0)}`",
        f"- verified_frame_rate: `{(report.get('claim_frame_verification') or {}).get('verified_frame_rate', 0.0)}`",
        f"- unsupported_frame_count: `{(report.get('claim_frame_verification') or {}).get('unsupported_frame_count', 0)}`",
        f"- trace_status_counts: `{(report.get('claim_frame_verification') or {}).get('trace_status_counts', {})}`",
        f"- support_status_counts: `{(report.get('claim_frame_verification') or {}).get('support_status_counts', {})}`",
        f"- semantic_support_status_counts: `{(report.get('claim_frame_verification') or {}).get('semantic_support_status_counts', {})}`",
        f"- semantic_supported_frame_rate: `{(report.get('claim_frame_verification') or {}).get('semantic_supported_frame_rate', 0.0)}`",
        f"- semantic_verified_frame_count: `{(report.get('claim_frame_verification') or {}).get('semantic_verified_frame_count', 0)}`",
        f"- partial_supported_frame_count: `{(report.get('claim_frame_verification') or {}).get('partial_supported_frame_count', 0)}`",
        f"- related_only_frame_count: `{(report.get('claim_frame_verification') or {}).get('related_only_frame_count', 0)}`",
        f"- contradicted_frame_count: `{(report.get('claim_frame_verification') or {}).get('contradicted_frame_count', 0)}`",
        f"- insufficient_evidence_frame_count: `{(report.get('claim_frame_verification') or {}).get('insufficient_evidence_frame_count', 0)}`",
        f"- citation_risk_counts: `{(report.get('claim_frame_verification') or {}).get('citation_risk_counts', {})}`",
        f"- missing_evidence_unit_count: `{(report.get('claim_frame_verification') or {}).get('missing_evidence_unit_count', 0)}`",
        f"- missing_node_count: `{(report.get('claim_frame_verification') or {}).get('missing_node_count', 0)}`",
        f"- missing_source_count: `{(report.get('claim_frame_verification') or {}).get('missing_source_count', 0)}`",
        f"- low_quality_frame_count: `{(report.get('claim_frame_verification') or {}).get('low_quality_frame_count', 0)}`",
        f"- noisy_frame_count: `{(report.get('claim_frame_verification') or {}).get('noisy_frame_count', 0)}`",
        f"- ignored_noise_frame_count: `{(report.get('claim_frame_verification') or {}).get('ignored_noise_frame_count', 0)}`",
        f"- top_frame_noise_reasons: `{', '.join((report.get('claim_frame_verification') or {}).get('top_frame_noise_reasons') or [])}`",
        f"- compiled_context_available: `{(report.get('memory_context') or {}).get('available', False)}`",
        f"- compiled_context_schema: `{(report.get('memory_context') or {}).get('schema', '')}`",
        f"- selected_memory_count: `{(report.get('memory_context') or {}).get('selected_memory_count', 0)}`",
        f"- artifact_ref_count: `{(report.get('memory_context') or {}).get('artifact_ref_count', 0)}`",
        f"- filtered_memory_count: `{(report.get('memory_context') or {}).get('filtered_memory_count', 0)}`",
        f"- context_char_count: `{(report.get('memory_context') or {}).get('context_char_count', 0)}`",
        f"- review_partial_reasons: `{', '.join(((report.get('tasks') or {}).get('review') or {}).get('review_partial_reasons') or [])}`",
        f"- compare_answerability: `{(((report.get('tasks') or {}).get('compare') or {}).get('answer_plan_summary') or {}).get('answerability', '')}`",
        f"- review_answerability: `{(((report.get('tasks') or {}).get('review') or {}).get('answer_plan_summary') or {}).get('answerability', '')}`",
        f"- answer_plan_claim_counts: `strong={answer_plan['strong_claim_count']} qualified={answer_plan['qualified_claim_count']} conflicting={answer_plan['conflicting_claim_count']} insufficient={answer_plan['insufficient_claim_count']}`",
        f"- claim_alignment_counts: `groups={alignment['group_count']} relations={alignment['relation_count']} methods={alignment['method_family_group_count']} conflicts={alignment['conflicting_group_count']} incomparable={alignment['incomparable_pair_count']} gaps={alignment['research_gap_count']} avg_align={alignment['avg_claim_align_score']}`",
        f"- top_review_blockers: `{', '.join(report.get('top_review_blockers') or [])}`",
        f"- baseline_limitations: `{', '.join(report.get('baseline_limitations') or [])}`",
        f"- llm_runtime_limitations: `{', '.join(report.get('llm_runtime_limitations') or [])}`",
        f"- warning_count: `{len(report.get('warnings') or [])}`",
        "",
        "## Documents",
    ]
    for doc in report.get("documents") or []:
        lines.append(
            f"- `{doc.get('doc_id')}` quality=`{doc.get('quality_level')}` "
            f"sections=`{doc.get('section_count')}` tables=`{doc.get('table_count')}` warnings=`{doc.get('warning_count')}`"
        )
    lines.extend(["", "## Parser Comparison"])
    for provider in (report.get("parser_comparison") or {}).get("providers") or []:
        summary = provider.get("quality_summary") or {}
        lines.append(
            f"- `{provider.get('provider')}` status=`{provider.get('status')}` reason=`{provider.get('reason', '')}` "
            f"docs=`{summary.get('document_count', 0)}` metadata_avg=`{(summary.get('metadata') or {}).get('avg_score', 0.0)}` "
            f"sections=`{(summary.get('section') or {}).get('total_count', 0)}` "
            f"refs=`{(summary.get('reference') or {}).get('total_count', 0)}` "
            f"tables=`{(summary.get('table') or {}).get('total_count', 0)}` "
            f"figures=`{(summary.get('figure') or {}).get('total_count', 0)}` "
            f"warnings=`{(summary.get('warning') or {}).get('total_count', 0)}`"
        )
    lines.extend(["", "## LLM Runtime"])
    for name, stage in ((report.get("llm_baseline") or {}).get("stage_summary") or {}).items():
        if not isinstance(stage, dict):
            continue
        lines.append(
            f"- `{name}` status=`{stage.get('status')}` calls=`{stage.get('call_count', 0)}` "
            f"timeouts=`{stage.get('timeout_count', 0)}` fallback=`{stage.get('fallback_count', 0)}`"
        )
    lines.extend(["", "## Review Draft Actions"])
    for item in (((report.get("tasks") or {}).get("review_draft") or {}).get("section_revision_actions") or []):
        actions = "; ".join(item.get("actions") or [])
        lines.append(
            f"- section=`{item.get('section_id', '')}` quality=`{item.get('draft_quality_level', '')}` "
            f"actions=`{actions}`"
        )
    lines.extend(["", "## Recommendations"])
    for item in report.get("recommendations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def baseline_html(report: Dict[str, Any]) -> str:
    compare = ((report.get("tasks") or {}).get("compare") or {})
    review = ((report.get("tasks") or {}).get("review") or {})
    compare_answer_plan = compare.get("answer_plan_summary") or {}
    review_answer_plan = review.get("answer_plan_summary") or {}
    answer_plan = answer_plan_rollup([compare_answer_plan, review_answer_plan])
    alignment = claim_alignment_rollup(_claim_alignment_summaries(report))
    review_draft = ((report.get("tasks") or {}).get("review_draft") or {})
    fact_delta = report.get("fact_audit_delta") or {}
    tree_summary = (report.get("tree_search") or {}).get("comparison_summary") or {}
    llm_baseline = report.get("llm_baseline") or {}
    llm_facts = llm_baseline.get("insights_and_facts") or {}
    llm_tasks = llm_baseline.get("tasks") or {}
    embedding = report.get("embedding") or {}
    memory_context = report.get("memory_context") or {}
    claim_frame_verification = report.get("claim_frame_verification") or {}
    cards = [
        ("Docs", report.get("doc_count", 0)),
        ("PDFs", report.get("pdf_count", 0)),
        ("Code Version", report.get("code_version", "")),
        ("Current Code", report.get("is_current_code_baseline", "")),
        ("Run Kind", report.get("run_kind", "")),
        ("Warnings", len(report.get("warnings") or [])),
        ("Best Mode", (report.get("benchmark") or {}).get("best_mode_by_score", "")),
        ("LLM Baseline", (report.get("llm_baseline") or {}).get("status", "")),
        ("LLM Reachable", (report.get("llm_status") or {}).get("reachable", "")),
        ("LLM Calls", llm_baseline.get("total_llm_call_count", 0)),
        ("LLM Timeouts", llm_baseline.get("timeout_count", 0)),
        ("Hard Timeouts", llm_baseline.get("hard_timeout_count", 0)),
        ("Slow Calls", llm_baseline.get("slow_call_count", 0)),
        ("LLM Budget", "exhausted" if llm_baseline.get("budget_exhausted") else "ok"),
        ("Facts LLM Rate", llm_facts.get("llm_facts_success_rate", 0.0)),
        ("Compare LLM Rate", llm_tasks.get("llm_compare_dimension_success_rate", 0.0)),
        ("Compare Answerability", compare_answer_plan.get("answerability", "")),
        ("Review Answerability", review_answer_plan.get("answerability", "")),
        ("Strong Claims", answer_plan["strong_claim_count"]),
        ("Conflicting Claims", answer_plan["conflicting_claim_count"]),
        ("Alignment Groups", alignment["group_count"]),
        ("Claim Relations", alignment["relation_count"]),
        ("Avg Align Score", alignment["avg_claim_align_score"]),
        ("Incomparable Pairs", alignment["incomparable_pair_count"]),
        ("Research Gaps", alignment["research_gap_count"]),
        ("Draft Status", review_draft.get("status", "")),
        ("Draft Skip", review_draft.get("review_draft_skip_reason", "")),
        ("Draft Quality", review_draft.get("draft_quality_level", "")),
        ("Skipped Sections", review_draft.get("skipped_section_count", 0)),
        ("Removed Paragraphs", (review_draft.get("paragraph_support_report") or {}).get("removed_paragraph_count", 0)),
        ("Optional Unused", review_draft.get("optional_unused_evidence_count", 0)),
        ("Citation Coverage", review_draft.get("citation_coverage_score", 0.0)),
        ("Real Provider", embedding.get("real_embedding_provider", "")),
        ("Real Embedding", embedding.get("real_embedding_status") or (embedding.get("sentence_transformers") or {}).get("status", "")),
        ("Embedding Model", embedding.get("real_embedding_model", "")),
        ("Embedding Coverage", embedding.get("real_embedding_node_coverage", 0.0)),
        ("Hybrid Provider", embedding.get("hybrid_embedding_provider", "")),
        ("Tree Trace", tree_summary.get("llm_trace_completeness_avg") or tree_summary.get("rule_trace_completeness_avg") or 0.0),
        ("Evidence Dedupe", review.get("duplicate_evidence_removed", 0)),
        ("Limitations", ", ".join(report.get("baseline_limitations") or [])),
        ("Citation Gaps", f"{fact_delta.get('citation_gap_count_before', 0)}->{fact_delta.get('citation_gap_count_after', 0)}"),
        ("Memory", (report.get("memory") or {}).get("status", "")),
        ("Semantic Support", claim_frame_verification.get("semantic_supported_frame_rate", 0.0)),
        ("Semantic Frames", claim_frame_verification.get("semantic_verified_frame_count", 0)),
        ("Contradicted", claim_frame_verification.get("contradicted_frame_count", 0)),
        ("Citation Risk", ", ".join(f"{key}:{value}" for key, value in (claim_frame_verification.get("citation_risk_counts") or {}).items())),
        ("Context Available", memory_context.get("available", False)),
        ("Selected Memory", memory_context.get("selected_memory_count", 0)),
        ("Artifact Refs", memory_context.get("artifact_ref_count", 0)),
        ("Context Chars", memory_context.get("context_char_count", 0)),
        ("Graph Conflicts", (report.get("claim_graph") or {}).get("conflict_count", 0)),
        ("Graph Isolated", (report.get("claim_graph") or {}).get("isolated_fact_count", 0)),
    ]
    card_html = "\n".join(
        f"<section class='card'><div class='label'>{escape(str(label))}</div><div class='value'>{escape(str(value))}</div></section>"
        for label, value in cards
    )
    docs_html = html_table(
        "Documents",
        ["Doc", "Quality", "Sections", "Tables", "Warnings"],
        [
            [
                doc.get("doc_id", ""),
                doc.get("quality_level", ""),
                doc.get("section_count", 0),
                doc.get("table_count", 0),
                doc.get("warning_count", 0),
            ]
            for doc in report.get("documents") or []
        ],
    )
    parser_html = html_table(
        "Parser Comparison",
        ["Provider", "Status", "Reason", "Docs", "Metadata", "Sections", "Refs", "Tables", "Figures", "Warnings"],
        [
            [
                item.get("provider", ""),
                item.get("status", ""),
                item.get("reason", ""),
                (item.get("quality_summary") or {}).get("document_count", 0),
                ((item.get("quality_summary") or {}).get("metadata") or {}).get("avg_score", 0.0),
                ((item.get("quality_summary") or {}).get("section") or {}).get("total_count", 0),
                ((item.get("quality_summary") or {}).get("reference") or {}).get("total_count", 0),
                ((item.get("quality_summary") or {}).get("table") or {}).get("total_count", 0),
                ((item.get("quality_summary") or {}).get("figure") or {}).get("total_count", 0),
                ((item.get("quality_summary") or {}).get("warning") or {}).get("total_count", 0),
            ]
            for item in (report.get("parser_comparison") or {}).get("providers") or []
        ],
    )
    llm_runtime_html = html_table(
        "LLM Runtime",
        ["Stage", "Status", "Calls", "Timeouts", "Fallback", "Duration"],
        [
            [
                name,
                stage.get("status", ""),
                stage.get("call_count", 0),
                stage.get("timeout_count", 0),
                stage.get("fallback_count", 0),
                stage.get("duration_ms", 0),
            ]
            for name, stage in (llm_baseline.get("stage_summary") or {}).items()
            if isinstance(stage, dict)
        ],
    )
    links = html_list_section(
        "Artifacts",
        [
            f"eval_set: {(report.get('eval_set') or {}).get('path', '')}",
            f"benchmark: {(report.get('benchmark') or {}).get('path', '')}",
            f"compare_task: {((report.get('tasks') or {}).get('compare') or {}).get('task_id', '')}",
            f"review_task: {((report.get('tasks') or {}).get('review') or {}).get('task_id', '')}",
            f"review_draft: {review_draft.get('review_draft_path', '')}",
            f"case_study: {((report.get('tasks') or {}).get('case_study') or {}).get('path', '')}",
            f"claim_graph: {(report.get('claim_graph') or {}).get('graph_dir', '')}",
        ],
    )
    recs = html_list_section("Recommendations", report.get("recommendations") or [])
    partial_reasons = html_list_section("Review Partial Reasons", review.get("review_partial_reasons") or [])
    review_actions = html_table(
        "Review Draft Actions",
        ["Section", "Quality", "Actions"],
        [
            [
                item.get("section_id", ""),
                item.get("draft_quality_level", ""),
                "; ".join(item.get("actions") or []),
            ]
            for item in review_draft.get("section_revision_actions") or []
        ],
    )
    warnings = html_list_section("Warnings", report.get("warnings") or [])
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Quality Baseline</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; background: #f7f8fa; }}
    header {{ padding: 26px 32px; background: #fff; border-bottom: 1px solid #d9dee7; }}
    main {{ padding: 24px 32px 40px; }}
    h1 {{ margin: 0; font-size: 26px; letter-spacing: 0; }}
    h2 {{ font-size: 16px; margin: 0 0 10px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card, .panel {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
    .label {{ font-size: 12px; color: #64748b; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ border-top: 1px solid #edf1f5; padding: 8px; font-size: 13px; overflow-wrap: anywhere; text-align: left; }}
    ul {{ padding-left: 18px; margin: 0; }}
    li {{ margin: 6px 0; font-size: 13px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>Quality Baseline</h1>
    <p>{escape(str(report.get('baseline_id') or ''))}</p>
  </header>
  <main>
    <div class="grid">{card_html}</div>
    {docs_html}
    {parser_html}
    {llm_runtime_html}
    {links}
    {recs}
    {partial_reasons}
    {review_actions}
    {warnings}
  </main>
</body>
</html>
"""


def html_table(title: str, headers: List[str], rows: List[List[Any]]) -> str:
    head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<section class='panel'><h2>{escape(title)}</h2><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>"


def html_list_section(title: str, items: Iterable[Any]) -> str:
    rows = "\n".join(f"<li>{escape(str(item))}</li>" for item in items)
    return f"<section class='panel'><h2>{escape(title)}</h2><ul>{rows}</ul></section>"


def _answer_plan_summaries(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = report.get("tasks") or {}
    return [(tasks.get(name) or {}).get("answer_plan_summary") or {} for name in ("compare", "review")]


def _claim_alignment_summaries(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = report.get("tasks") or {}
    return [(tasks.get(name) or {}).get("claim_alignment_summary") or {} for name in ("compare", "review")]
