from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb", description="PageIndex-like knowledge base MVP")
    parser.add_argument("--db", default=None, help="SQLite database path. Defaults to data/kb.sqlite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Index a directory or supported file")
    sync_parser.add_argument("path")
    sync_parser.add_argument("--force", action="store_true")
    sync_parser.add_argument("--pdf-parser", choices=["auto", "pypdf", "docling", "grobid"], default=None)
    sync_parser.add_argument("--build-embeddings", action="store_true", help="Build semantic index after sync")

    subparsers.add_parser("list", help="List indexed documents")

    search_parser = subparsers.add_parser("search", help="Search evidence nodes")
    search_parser.add_argument("query")
    search_parser.add_argument("--doc-id", default=None)
    search_parser.add_argument("--top-k", type=int, default=8)
    search_parser.add_argument("--search-mode", choices=["hybrid", "fts", "tree", "auto"], default="hybrid")

    docs_parser = subparsers.add_parser("docs", help="Search candidate documents")
    docs_parser.add_argument("query")
    docs_parser.add_argument("--top-k", type=int, default=8)
    docs_parser.add_argument("--search-mode", choices=["hybrid", "fts", "tree", "auto"], default="hybrid")

    embed_parser = subparsers.add_parser("embed", help="Build semantic embeddings for ready documents")
    embed_parser.add_argument("--doc-id", action="append", default=[], help="Build only one document id; repeatable")
    embed_parser.add_argument("--force", action="store_true")
    embed_parser.add_argument("--provider", choices=["hash", "sentence-transformers", "openai-compatible"], default=None)
    embed_parser.add_argument("--model", default=None)
    embed_parser.add_argument("--batch-size", type=int, default=16)
    embed_parser.add_argument("--status", action="store_true", help="Only show semantic index status")

    llm_status_parser = subparsers.add_parser("llm-status", help="Show sanitized DeepSeek configuration and connectivity")
    llm_status_parser.add_argument("--probe", action="store_true", help="Call the configured chat endpoint with a short connectivity probe")

    report_parser = subparsers.add_parser("search-report", help="Explain hybrid search candidates and scores")
    report_parser.add_argument("query")
    report_parser.add_argument("--doc-id", default=None)
    report_parser.add_argument("--top-k", type=int, default=8)
    report_parser.add_argument("--search-mode", choices=["hybrid", "fts", "tree", "auto"], default="hybrid")

    eval_parser = subparsers.add_parser("eval-search", help="Run a JSON search evaluation set")
    eval_parser.add_argument("queries_json")
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--search-mode", choices=["hybrid", "fts", "tree"], default="hybrid")
    eval_parser.add_argument(
        "--compare-modes",
        default="",
        help="Comma-separated search modes to compare, for example hybrid,tree,fts",
    )

    eval_review_parser = subparsers.add_parser("eval-review", help="Evaluate review task evidence and citation coverage")
    eval_review_parser.add_argument("task_id")

    subparsers.add_parser("eval-memory", help="Evaluate long-term memory hygiene and resume readiness")

    eval_facts_parser = subparsers.add_parser("eval-facts", help="Evaluate grounded paper facts and table-backed coverage")
    eval_facts_parser.add_argument("--doc-id", action="append", default=[], help="Limit to one document id; repeatable")

    audit_facts_parser = subparsers.add_parser("audit-facts", help="Audit grounded facts for duplicates, conflicts, and evidence gaps")
    audit_facts_parser.add_argument("--doc-id", action="append", default=[], help="Limit to one document id; repeatable")
    audit_facts_parser.add_argument("--min-confidence", type=float, default=0.5)

    fact_conflicts_parser = subparsers.add_parser("fact-conflicts", help="Show fact conflicts from a fresh fact audit")
    fact_conflicts_parser.add_argument("--doc-id", action="append", default=[], help="Limit to one document id; repeatable")
    fact_conflicts_parser.add_argument("--severity", choices=["low", "medium", "high"], default=None)
    fact_conflicts_parser.add_argument("--min-confidence", type=float, default=0.5)

    graph_build_parser = subparsers.add_parser("graph-build", help="Build a lightweight claim graph from facts and audit risks")
    graph_build_parser.add_argument("--doc-id", action="append", default=[], help="Limit graph to one document id; repeatable")
    graph_build_parser.add_argument("--include-conflicts", action="store_true", help="Include fact audit conflict nodes")
    graph_build_parser.add_argument("--min-confidence", type=float, default=0.0)

    graph_neighborhood_parser = subparsers.add_parser("graph-neighborhood", help="Show a claim graph neighborhood")
    graph_neighborhood_parser.add_argument("node_or_fact_id")
    graph_neighborhood_parser.add_argument("--depth", type=int, default=1)
    graph_neighborhood_parser.add_argument("--graph-id", default=None)

    graph_export_parser = subparsers.add_parser("graph-export", help="Export a claim graph as json, mermaid, or html")
    graph_export_parser.add_argument("graph_id")
    graph_export_parser.add_argument("--format", choices=["json", "mermaid", "html"], default="json")

    graph_report_parser = subparsers.add_parser("graph-report", help="Show a claim graph quality report")
    graph_report_parser.add_argument("graph_id")

    baseline_parser = subparsers.add_parser("quality-baseline", help="Run a real-corpus quality baseline")
    baseline_parser.add_argument("corpus_path", nargs="?", default="articles")
    baseline_parser.add_argument("--no-force", action="store_true", help="Do not force rebuild corpus artifacts")
    baseline_parser.add_argument("--top-k", type=int, default=5)
    baseline_parser.add_argument("--with-llm", action="store_true", help="Allow optional DeepSeek calls during insight/task checks")
    baseline_parser.add_argument("--embedding-provider", choices=["hash", "sentence-transformers", "openai-compatible"], default=None)
    baseline_parser.add_argument("--embedding-model", default=None)
    baseline_parser.add_argument("--llm-timeout-seconds", type=int, default=None, help="Per DeepSeek call timeout used during the baseline")
    baseline_parser.add_argument("--llm-stage-timeout-seconds", type=int, default=None, help="Maximum runtime for each LLM baseline stage")
    baseline_parser.add_argument("--llm-max-docs", type=int, default=None, help="Maximum documents to process with LLM stages")
    baseline_parser.add_argument("--skip-llm-tasks", action="store_true", help="Skip LLM compare/review stages while keeping other LLM checks")

    latest_baseline_parser = subparsers.add_parser("latest-quality-baseline", help="Show latest quality baseline reports")
    latest_baseline_parser.add_argument("--limit", type=int, default=1)
    latest_baseline_parser.add_argument("--corpus", default=None, help="Filter by corpus path, for example articles")
    latest_baseline_parser.add_argument("--real-only", action="store_true", help="Only show real articles baselines")
    latest_baseline_parser.add_argument("--exclude-temp", action="store_true", help="Exclude temporary test fixture baselines")

    eval_suite_parser = subparsers.add_parser("eval-suite", help="Create, list, or show reusable evaluation suites")
    eval_suite_subparsers = eval_suite_parser.add_subparsers(dest="suite_command", required=True)
    eval_suite_create = eval_suite_subparsers.add_parser("create", help="Create an eval suite from JSON, feedback, query logs, or docs")
    eval_suite_create.add_argument("name")
    eval_suite_create.add_argument("--input-json", default=None)
    eval_suite_create.add_argument("--from-feedback", action="store_true")
    eval_suite_create.add_argument("--from-query-log", action="store_true")
    eval_suite_create.add_argument("--doc-id", action="append", default=[], help="Add document smoke cases; repeatable")
    eval_suite_create.add_argument("--limit", type=int, default=100)
    eval_suite_create.add_argument("--min-rating", type=int, default=4)
    eval_suite_subparsers.add_parser("list", help="List saved eval suites")
    eval_suite_show = eval_suite_subparsers.add_parser("show", help="Show one saved eval suite")
    eval_suite_show.add_argument("name")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a benchmark suite across search modes")
    benchmark_parser.add_argument("suite_name")
    benchmark_parser.add_argument("--compare-modes", default="fts,hybrid,tree,auto", help="Comma-separated modes, e.g. fts,hash-hybrid,bge-m3-hybrid,tree,auto")
    benchmark_parser.add_argument("--top-k", type=int, default=5)

    failure_parser = subparsers.add_parser("analyze-failures", help="Analyze benchmark failures and next actions")
    failure_parser.add_argument("benchmark_id")

    case_parser = subparsers.add_parser("case-study", help="Generate a retrieval case study without evidence text")
    case_parser.add_argument("query")
    case_parser.add_argument("--doc-id", action="append", default=[], help="Limit case study to one document id; repeatable")
    case_parser.add_argument("--compare-modes", default="hybrid,tree")
    case_parser.add_argument("--top-k", type=int, default=5)

    classify_parser = subparsers.add_parser("classify-query", help="Classify a query intent for tree search")
    classify_parser.add_argument("query")
    classify_parser.add_argument("--no-llm", action="store_true", help="Use rule-based classification only")
    classify_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")

    tree_search_parser = subparsers.add_parser("tree-search", help="Run explainable tree search inside one document")
    tree_search_parser.add_argument("doc_id")
    tree_search_parser.add_argument("query")
    tree_search_parser.add_argument("--budget", type=int, default=8)
    tree_search_parser.add_argument("--no-llm", action="store_true", help="Use value-function tree search only")
    tree_search_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")
    tree_search_parser.add_argument("--search-mode", choices=["hybrid", "fts"], default="hybrid")

    tree_parser = subparsers.add_parser("tree", help="Show a document tree")
    tree_parser.add_argument("doc_id")

    card_parser = subparsers.add_parser("card", help="Show a document card")
    card_parser.add_argument("doc_id")

    artifacts_parser = subparsers.add_parser("artifacts", help="List generated artifacts for a document")
    artifacts_parser.add_argument("doc_id")
    artifacts_parser.add_argument("--version-id", default=None)

    quality_parser = subparsers.add_parser("quality", help="Show parse quality for a document")
    quality_parser.add_argument("doc_id")

    parse_report_parser = subparsers.add_parser("parse-report", help="Show parser diagnostics for a document")
    parse_report_parser.add_argument("doc_id")
    parse_report_parser.add_argument("--version-id", default=None)

    layout_parser = subparsers.add_parser("layout", help="Show PDF layout block artifact summary")
    layout_parser.add_argument("doc_id")
    layout_parser.add_argument("--version-id", default=None)

    figures_parser = subparsers.add_parser("figures", help="Show parsed figure caption artifacts")
    figures_parser.add_argument("doc_id")
    figures_parser.add_argument("--version-id", default=None)

    tables_parser = subparsers.add_parser("tables", help="Show parsed table caption artifacts")
    tables_parser.add_argument("doc_id")
    tables_parser.add_argument("--version-id", default=None)

    table_content_parser = subparsers.add_parser("table-content", help="Show parsed table row and cell artifacts")
    table_content_parser.add_argument("doc_id")
    table_content_parser.add_argument("--version-id", default=None)

    table_summaries_parser = subparsers.add_parser("table-summaries", help="Show parsed table summary artifacts")
    table_summaries_parser.add_argument("doc_id")
    table_summaries_parser.add_argument("--version-id", default=None)

    extract_parser = subparsers.add_parser("extract", help="Extract paper insight artifacts")
    extract_parser.add_argument("doc_id")
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--no-llm", action="store_true", help="Use rule-based extraction only")
    extract_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")

    innovations_parser = subparsers.add_parser("innovations", help="Show extracted innovation artifact")
    innovations_parser.add_argument("doc_id")

    citations_parser = subparsers.add_parser("citations", help="Show extracted citation map artifact")
    citations_parser.add_argument("doc_id")

    facts_parser = subparsers.add_parser("extract-facts", help="Extract grounded claims, entities, and relations")
    facts_parser.add_argument("doc_id")
    facts_parser.add_argument("--force", action="store_true")
    facts_parser.add_argument("--no-llm", action="store_true", help="Use rule-based fact extraction only")
    facts_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")

    evidence_units_parser = subparsers.add_parser("extract-evidence-units", help="Extract normalized EvidenceUnit artifacts")
    evidence_units_parser.add_argument("doc_id")
    evidence_units_parser.add_argument("--force", action="store_true")

    evidence_units_show_parser = subparsers.add_parser("evidence-units", help="Show extracted EvidenceUnit artifact")
    evidence_units_show_parser.add_argument("doc_id")

    claim_frames_parser = subparsers.add_parser("extract-claim-frames", help="Extract ClaimFrame artifacts")
    claim_frames_parser.add_argument("doc_id")
    claim_frames_parser.add_argument("--force", action="store_true")
    claim_frames_parser.add_argument("--no-llm", action="store_true", help="Use rule-based ClaimFrame extraction only")
    claim_frames_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")

    claim_frames_show_parser = subparsers.add_parser("claim-frames", help="Show extracted ClaimFrame artifact")
    claim_frames_show_parser.add_argument("doc_id")

    verify_claim_frames_parser = subparsers.add_parser("verify-claim-frames", help="Verify ClaimFrames against EvidenceUnits and nodes")
    verify_claim_frames_parser.add_argument("--doc-id", action="append", default=[], help="Limit to one document id; repeatable")

    claims_parser = subparsers.add_parser("claims", help="Show extracted claims artifact")
    claims_parser.add_argument("doc_id")

    entities_parser = subparsers.add_parser("entities", help="Show extracted entities artifact")
    entities_parser.add_argument("doc_id")

    relations_parser = subparsers.add_parser("relations", help="Show extracted relations artifact")
    relations_parser.add_argument("doc_id")

    fact_graph_parser = subparsers.add_parser("fact-graph", help="Show extracted fact graph artifact")
    fact_graph_parser.add_argument("doc_id")

    fact_search_parser = subparsers.add_parser("fact-search", help="Search extracted claims/entities/relations")
    fact_search_parser.add_argument("query")
    fact_search_parser.add_argument("--doc-id", action="append", default=[], help="Limit to one document id; repeatable")
    fact_search_parser.add_argument("--type", choices=["claim", "entity", "relation"], default=None)
    fact_search_parser.add_argument("--source", choices=["text", "table", "all"], default="all")
    fact_search_parser.add_argument("--min-confidence", type=float, default=0.0)
    fact_search_parser.add_argument("--top-k", type=int, default=20)

    compare_parser = subparsers.add_parser("compare", help="Compare papers with grounded task artifacts")
    compare_parser.add_argument("query")
    compare_parser.add_argument("--doc-id", action="append", default=[], help="Limit comparison to a document id; repeatable")
    compare_parser.add_argument("--top-k-docs", type=int, default=5)
    compare_parser.add_argument("--no-llm", action="store_true", help="Use rule-based comparison only")
    compare_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")
    compare_parser.add_argument("--search-mode", choices=["hybrid", "fts", "tree", "auto"], default="hybrid")

    review_parser = subparsers.add_parser("generate-review", help="Generate review planning task artifacts")
    review_parser.add_argument("topic")
    review_parser.add_argument("--doc-id", action="append", default=[], help="Limit review planning to a document id; repeatable")
    review_parser.add_argument("--top-k-docs", type=int, default=8)
    review_parser.add_argument("--no-llm", action="store_true", help="Use rule-based planning only")
    review_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")
    review_parser.add_argument("--search-mode", choices=["hybrid", "fts", "tree", "auto"], default="hybrid")

    task_artifact_parser = subparsers.add_parser("task-artifact", help="Read a task artifact")
    task_artifact_parser.add_argument("task_id")
    task_artifact_parser.add_argument("name")

    draft_review_parser = subparsers.add_parser("draft-review", help="Draft review sections from review task evidence")
    draft_review_parser.add_argument("task_id")
    draft_review_parser.add_argument("--section-id", action="append", default=[], help="Draft only one section id; repeatable")
    draft_review_parser.add_argument("--no-llm", action="store_true", help="Use evidence bullet drafts only")
    draft_review_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")

    assemble_review_parser = subparsers.add_parser("assemble-review", help="Assemble section drafts into review_draft.md")
    assemble_review_parser.add_argument("task_id")

    check_review_parser = subparsers.add_parser("check-review", help="Check review draft citation consistency")
    check_review_parser.add_argument("task_id")

    evidence_parser = subparsers.add_parser("evidence", help="Get evidence packets")
    evidence_parser.add_argument("doc_id")
    evidence_parser.add_argument("node_ids", nargs="+")

    ask_parser = subparsers.add_parser("ask", help="Answer a question using grounded evidence")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--top-k", type=int, default=6)
    ask_parser.add_argument("--no-llm", action="store_true", help="Only print evidence; do not call DeepSeek")
    ask_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")
    ask_parser.add_argument("--json", action="store_true", help="Print full JSON result")
    ask_parser.add_argument("--search-mode", choices=["hybrid", "fts", "tree", "auto"], default="hybrid")

    mem_put_parser = subparsers.add_parser("memory-put", help="Store explicit long-term memory")
    mem_put_parser.add_argument("scope")
    mem_put_parser.add_argument("type")
    mem_put_parser.add_argument("subject_key")
    mem_put_parser.add_argument("content")
    mem_put_parser.add_argument("--importance", type=float, default=0.5)
    mem_put_parser.add_argument("--confidence", type=float, default=1.0)
    mem_put_parser.add_argument("--ttl-days", type=float, default=None)
    mem_put_parser.add_argument("--refs", default="")
    mem_put_parser.add_argument("--force", action="store_true")

    mem_search_parser = subparsers.add_parser("memory-search", help="Search memory items")
    mem_search_parser.add_argument("query")
    mem_search_parser.add_argument("--scope", default=None)
    mem_search_parser.add_argument("--top-k", type=int, default=8)

    mem_compile_parser = subparsers.add_parser("memory-compile", help="Compile artifact-first memory context")
    mem_compile_parser.add_argument("query")
    mem_compile_parser.add_argument("--intent", default="default")
    mem_compile_parser.add_argument("--task-id", default="")
    mem_compile_parser.add_argument("--skill-scope", default="default")
    mem_compile_parser.add_argument("--max-items", type=int, default=8)
    mem_compile_parser.add_argument("--max-chars", type=int, default=4000)

    remember_task_parser = subparsers.add_parser("remember-task", help="Store compressed progress for a task")
    remember_task_parser.add_argument("task_id")

    subparsers.add_parser("resume-task", help="Resume the latest task from task state and memory")

    compact_parser = subparsers.add_parser("memory-compact", help="Compact task progress memory")
    compact_parser.add_argument("--scope", default=None)

    query_log_parser = subparsers.add_parser("query-log", help="List recent query logs")
    query_log_parser.add_argument("--limit", type=int, default=20)
    query_log_parser.add_argument("--operation", default=None)
    query_log_parser.add_argument("--intent", default=None)
    query_log_parser.add_argument("--status", default=None)

    query_stats_parser = subparsers.add_parser("query-stats", help="Summarize query log metrics")
    query_stats_parser.add_argument("--since-days", type=float, default=None)

    feedback_put_parser = subparsers.add_parser("feedback-put", help="Record human feedback for a query result")
    feedback_put_parser.add_argument("query")
    feedback_put_parser.add_argument("--query-id", default="")
    feedback_put_parser.add_argument("--operation", default="")
    feedback_put_parser.add_argument("--rating", type=int, required=True)
    feedback_put_parser.add_argument("--label", default="")
    feedback_put_parser.add_argument("--comment", default="")
    feedback_put_parser.add_argument("--expected-doc-id", action="append", default=[])
    feedback_put_parser.add_argument("--expected-node-id", action="append", default=[])
    feedback_put_parser.add_argument("--expected-keyword", action="append", default=[])
    feedback_put_parser.add_argument("--preferred-search-mode", choices=["hybrid", "tree", "fts"], default="")

    feedback_list_parser = subparsers.add_parser("feedback-list", help="List recorded human feedback")
    feedback_list_parser.add_argument("--limit", type=int, default=20)
    feedback_list_parser.add_argument("--operation", default=None)
    feedback_list_parser.add_argument("--label", default=None)
    feedback_list_parser.add_argument("--rating", type=int, default=None)
    feedback_list_parser.add_argument("--min-rating", type=int, default=None)

    feedback_eval_parser = subparsers.add_parser("feedback-to-eval", help="Build a search eval set from feedback")
    feedback_eval_parser.add_argument("output_json", nargs="?", default=None)
    feedback_eval_parser.add_argument("--min-rating", type=int, default=4)
    feedback_eval_parser.add_argument("--label", default=None)
    feedback_eval_parser.add_argument("--operation", default=None)
    feedback_eval_parser.add_argument("--limit", type=int, default=200)

    eval_dashboard_parser = subparsers.add_parser("eval-dashboard", help="Write a static query/eval/feedback dashboard")
    eval_dashboard_parser.add_argument("--since-days", type=float, default=None)
    eval_dashboard_parser.add_argument("--format", choices=["json", "md", "html"], default="json")

    tune_parser = subparsers.add_parser("tune-search", help="Tune search mode preferences from an eval set")
    tune_parser.add_argument("queries_json")
    tune_parser.add_argument("--top-k", type=int, default=5)
    tune_parser.add_argument("--compare-modes", default="hybrid,tree,fts")
    tune_parser.add_argument("--save-profile", default=None)

    profile_parser = subparsers.add_parser("search-profile", help="Manage local search tuning profiles")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command", required=True)
    profile_subparsers.add_parser("list", help="List saved search profiles")
    profile_show = profile_subparsers.add_parser("show", help="Show a saved or active search profile")
    profile_show.add_argument("name", nargs="?", default="active")
    profile_apply = profile_subparsers.add_parser("apply", help="Apply a saved search profile for auto mode")
    profile_apply.add_argument("name")

    return parser
