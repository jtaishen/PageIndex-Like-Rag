from __future__ import annotations

from html import escape
from typing import Any, Dict, Iterable, List


def baseline_markdown(report: Dict[str, Any]) -> str:
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
        f"- real_embedding_status: `{(report.get('embedding') or {}).get('sentence_transformers', {}).get('status', '')}`",
        f"- real_embedding_model: `{(report.get('embedding') or {}).get('real_embedding_model', '')}`",
        f"- real_embedding_node_coverage: `{(report.get('embedding') or {}).get('real_embedding_node_coverage', 0.0)}`",
        f"- hybrid_embedding_provider: `{(report.get('embedding') or {}).get('hybrid_embedding_provider', '')}`",
        f"- embedding_rebuild_needed: `{(report.get('embedding') or {}).get('embedding_rebuild_needed', False)}`",
        f"- review_partial_reasons: `{', '.join(((report.get('tasks') or {}).get('review') or {}).get('review_partial_reasons') or [])}`",
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
        lines.append(f"- `{provider.get('provider')}` status=`{provider.get('status')}` reason=`{provider.get('reason', '')}`")
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
    review = ((report.get("tasks") or {}).get("review") or {})
    review_draft = ((report.get("tasks") or {}).get("review_draft") or {})
    fact_delta = report.get("fact_audit_delta") or {}
    tree_summary = (report.get("tree_search") or {}).get("comparison_summary") or {}
    llm_baseline = report.get("llm_baseline") or {}
    llm_facts = llm_baseline.get("insights_and_facts") or {}
    llm_tasks = llm_baseline.get("tasks") or {}
    embedding = report.get("embedding") or {}
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
        ("Draft Status", review_draft.get("status", "")),
        ("Draft Skip", review_draft.get("review_draft_skip_reason", "")),
        ("Draft Quality", review_draft.get("draft_quality_level", "")),
        ("Skipped Sections", review_draft.get("skipped_section_count", 0)),
        ("Removed Paragraphs", (review_draft.get("paragraph_support_report") or {}).get("removed_paragraph_count", 0)),
        ("Optional Unused", review_draft.get("optional_unused_evidence_count", 0)),
        ("Citation Coverage", review_draft.get("citation_coverage_score", 0.0)),
        ("Real Embedding", (embedding.get("sentence_transformers") or {}).get("status", "")),
        ("Embedding Model", embedding.get("real_embedding_model", "")),
        ("Embedding Coverage", embedding.get("real_embedding_node_coverage", 0.0)),
        ("Hybrid Provider", embedding.get("hybrid_embedding_provider", "")),
        ("Tree Trace", tree_summary.get("llm_trace_completeness_avg") or tree_summary.get("rule_trace_completeness_avg") or 0.0),
        ("Evidence Dedupe", review.get("duplicate_evidence_removed", 0)),
        ("Limitations", ", ".join(report.get("baseline_limitations") or [])),
        ("Citation Gaps", f"{fact_delta.get('citation_gap_count_before', 0)}->{fact_delta.get('citation_gap_count_after', 0)}"),
        ("Memory", (report.get("memory") or {}).get("status", "")),
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
        ["Provider", "Status", "Reason"],
        [
            [item.get("provider", ""), item.get("status", ""), item.get("reason", "")]
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
