from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from . import db
from .answer import answer_query, route_documents
from .artifacts import (
    get_citation_map,
    get_doc_card,
    get_figures,
    get_innovations,
    get_layout_blocks,
    get_parse_quality,
    get_parse_report,
    get_table_content,
    get_table_summaries,
    get_tables,
    list_artifacts,
)
from .benchmark import (
    analyze_failures,
    create_eval_suite,
    generate_case_study,
    get_eval_suite,
    list_eval_suites,
    run_benchmark,
)
from .config import resolve_db_path
from .embeddings import build_semantic_index, semantic_index_status
from .eval import eval_facts, eval_memory, eval_review, eval_search
from .fact_audit import audit_facts, get_fact_conflicts
from .facts import extract_facts, fact_search, get_claims, get_entities, get_fact_graph, get_relations
from .feedback import build_eval_set_from_feedback, eval_dashboard, list_feedback, put_feedback
from .ingest import sync_directory
from .insights import extract_doc_insights
from .llm import llm_status
from .knowledge_graph import build_knowledge_graph, export_knowledge_graph, get_graph_neighborhood, get_graph_report
from .memory import compact_memory, put_memory_gated, remember_task, resume_task, search_memory
from .quality_baseline import latest_quality_baseline, run_quality_baseline
from .query import classify_query
from .query_log import list_query_logs, query_stats, write_query_log
from .review import assemble_review, check_review_citations, draft_review
from .search import build_search_report, get_evidence, search_nodes
from .search_profile import apply_search_profile, get_search_profile, list_search_profiles, tune_search
from .tasks import compare_papers, generate_review_plan, get_task_artifact
from .tree_search import tree_search


def main(argv: Any = None) -> None:
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
    embed_parser.add_argument("--provider", choices=["hash", "sentence-transformers"], default=None)
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
    benchmark_parser.add_argument("--compare-modes", default="fts,hybrid,tree,auto")
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

    args = parser.parse_args(argv)
    db_path = resolve_db_path(args.db)

    if args.command == "sync":
        result = sync_directory(Path(args.path), db_path, force=args.force, pdf_parser=args.pdf_parser)
        if args.build_embeddings:
            result["semantic_index"] = build_semantic_index(db_path)
        _print_json(result)
    elif args.command == "list":
        _list_documents(db_path)
    elif args.command == "search":
        started = time.time()
        results = search_nodes(db_path, args.query, args.doc_id, args.top_k, search_mode=args.search_mode)
        _log_search_results(db_path, "search", args.query, args.search_mode, started, results)
        _print_json([result.__dict__ for result in results])
    elif args.command == "docs":
        started = time.time()
        docs = route_documents(db_path, args.query, args.top_k, search_mode=args.search_mode)
        _log_doc_results(db_path, args.query, args.search_mode, started, docs)
        _print_json(docs)
    elif args.command == "embed":
        if args.status:
            _print_json(semantic_index_status(db_path, provider=args.provider, model=args.model))
        else:
            _print_json(
                build_semantic_index(
                    db_path,
                    doc_ids=args.doc_id or None,
                    force=args.force,
                    provider=args.provider,
                    model=args.model,
                    batch_size=args.batch_size,
                )
            )
    elif args.command == "search-report":
        _print_json(build_search_report(db_path, args.query, doc_id=args.doc_id, top_k=args.top_k, search_mode=args.search_mode))
    elif args.command == "eval-search":
        compare_modes = _comma_list(args.compare_modes)
        _print_json(
            _eval_summary(
                eval_search(
                    db_path,
                    Path(args.queries_json),
                    search_mode=args.search_mode,
                    top_k=args.top_k,
                    compare_modes=compare_modes or None,
                )
            )
        )
    elif args.command == "eval-review":
        _print_json(_review_eval_summary(eval_review(db_path, args.task_id)))
    elif args.command == "eval-memory":
        _print_json(_memory_eval_summary(eval_memory(db_path)))
    elif args.command == "eval-facts":
        _print_json(_fact_eval_summary(eval_facts(db_path, doc_ids=args.doc_id or None)))
    elif args.command == "audit-facts":
        _print_json(_fact_audit_cli_summary(audit_facts(db_path, doc_ids=args.doc_id or None, min_confidence=args.min_confidence)))
    elif args.command == "fact-conflicts":
        _print_json(
            _fact_conflicts_cli_summary(
                get_fact_conflicts(
                    db_path,
                    doc_ids=args.doc_id or None,
                    severity=args.severity,
                    min_confidence=args.min_confidence,
                )
            )
        )
    elif args.command == "graph-build":
        _print_json(
            _graph_build_cli_summary(
                build_knowledge_graph(
                    db_path,
                    doc_ids=args.doc_id or None,
                    include_conflicts=args.include_conflicts,
                    min_confidence=args.min_confidence,
                )
            )
        )
    elif args.command == "graph-neighborhood":
        _print_json(
            get_graph_neighborhood(
                db_path,
                args.node_or_fact_id,
                depth=args.depth,
                graph_id=args.graph_id,
            )
        )
    elif args.command == "graph-export":
        _print_json(export_knowledge_graph(db_path, args.graph_id, format=args.format))
    elif args.command == "graph-report":
        _print_json(get_graph_report(db_path, args.graph_id))
    elif args.command == "quality-baseline":
        _print_json(
            _quality_baseline_cli_summary(
                run_quality_baseline(
                    db_path,
                    Path(args.corpus_path),
                    force=not args.no_force,
                    top_k=args.top_k,
                    use_llm=args.with_llm,
                    embedding_model=args.embedding_model,
                    llm_timeout_seconds=args.llm_timeout_seconds,
                    llm_stage_timeout_seconds=args.llm_stage_timeout_seconds,
                    llm_max_docs=args.llm_max_docs,
                    skip_llm_tasks=args.skip_llm_tasks,
                )
            )
        )
    elif args.command == "latest-quality-baseline":
        _print_json(
            latest_quality_baseline(
                limit=args.limit,
                corpus=args.corpus,
                real_only=args.real_only,
                exclude_temp=args.exclude_temp,
            )
        )
    elif args.command == "llm-status":
        _print_json(llm_status(probe=args.probe))
    elif args.command == "eval-suite":
        if args.suite_command == "create":
            _print_json(
                _suite_summary(
                    create_eval_suite(
                        db_path,
                        args.name,
                        input_json=Path(args.input_json) if args.input_json else None,
                        from_feedback=args.from_feedback,
                        from_query_log=args.from_query_log,
                        doc_ids=args.doc_id or None,
                        limit=args.limit,
                        min_rating=args.min_rating,
                    )
                )
            )
        elif args.suite_command == "list":
            _print_json(list_eval_suites())
        elif args.suite_command == "show":
            _print_json(get_eval_suite(args.name))
    elif args.command == "benchmark":
        _print_json(
            _benchmark_cli_summary(
                run_benchmark(
                    db_path,
                    args.suite_name,
                    compare_modes=_comma_list(args.compare_modes),
                    top_k=args.top_k,
                )
            )
        )
    elif args.command == "analyze-failures":
        _print_json(_failure_cli_summary(analyze_failures(db_path, args.benchmark_id)))
    elif args.command == "case-study":
        _print_json(
            _case_cli_summary(
                generate_case_study(
                    db_path,
                    args.query,
                    doc_ids=args.doc_id or None,
                    compare_modes=_comma_list(args.compare_modes),
                    top_k=args.top_k,
                )
            )
        )
    elif args.command == "classify-query":
        _print_json(classify_query(args.query, use_llm=not args.no_llm, require_llm=args.require_llm))
    elif args.command == "tree-search":
        _print_json(
            tree_search(
                db_path,
                args.doc_id,
                args.query,
                budget=args.budget,
                use_llm=not args.no_llm,
                require_llm=args.require_llm,
                search_mode=args.search_mode,
            )
        )
    elif args.command == "tree":
        _print_tree(db_path, args.doc_id)
    elif args.command == "card":
        _print_json(get_doc_card(db_path, args.doc_id))
    elif args.command == "artifacts":
        _print_json(list_artifacts(db_path, args.doc_id, args.version_id))
    elif args.command == "quality":
        _print_json(get_parse_quality(db_path, args.doc_id))
    elif args.command == "parse-report":
        _print_json(_parse_report_summary(get_parse_report(db_path, args.doc_id, args.version_id)))
    elif args.command == "layout":
        _print_json(_layout_summary(get_layout_blocks(db_path, args.doc_id, args.version_id)))
    elif args.command == "figures":
        _print_json(get_figures(db_path, args.doc_id, args.version_id))
    elif args.command == "tables":
        _print_json(get_tables(db_path, args.doc_id, args.version_id))
    elif args.command == "table-content":
        _print_json(get_table_content(db_path, args.doc_id, args.version_id))
    elif args.command == "table-summaries":
        _print_json(get_table_summaries(db_path, args.doc_id, args.version_id))
    elif args.command == "extract":
        result = extract_doc_insights(
            db_path,
            args.doc_id,
            force=args.force,
            use_llm=not args.no_llm,
            require_llm=args.require_llm,
        )
        _print_json(_extract_summary(result))
    elif args.command == "innovations":
        _print_json(get_innovations(db_path, args.doc_id))
    elif args.command == "citations":
        _print_json(get_citation_map(db_path, args.doc_id))
    elif args.command == "extract-facts":
        result = extract_facts(
            db_path,
            args.doc_id,
            force=args.force,
            use_llm=not args.no_llm,
            require_llm=args.require_llm,
        )
        _print_json(_fact_summary(result))
    elif args.command == "claims":
        _print_json(get_claims(db_path, args.doc_id))
    elif args.command == "entities":
        _print_json(get_entities(db_path, args.doc_id))
    elif args.command == "relations":
        _print_json(get_relations(db_path, args.doc_id))
    elif args.command == "fact-graph":
        _print_json(get_fact_graph(db_path, args.doc_id))
    elif args.command == "fact-search":
        _print_json(
            fact_search(
                db_path,
                args.query,
                doc_ids=args.doc_id or None,
                fact_type=args.type,
                source=args.source,
                min_confidence=args.min_confidence,
                top_k=args.top_k,
            )
        )
    elif args.command == "compare":
        result = compare_papers(
            db_path,
            args.query,
            doc_ids=args.doc_id,
            top_k_docs=args.top_k_docs,
            use_llm=not args.no_llm,
            require_llm=args.require_llm,
            search_mode=args.search_mode,
        )
        _print_json(_task_summary(result))
    elif args.command == "generate-review":
        result = generate_review_plan(
            db_path,
            args.topic,
            doc_ids=args.doc_id,
            top_k_docs=args.top_k_docs,
            use_llm=not args.no_llm,
            require_llm=args.require_llm,
            search_mode=args.search_mode,
        )
        _print_json(_task_summary(result))
    elif args.command == "task-artifact":
        _print_json(get_task_artifact(db_path, args.task_id, args.name))
    elif args.command == "draft-review":
        result = draft_review(
            db_path,
            args.task_id,
            section_ids=args.section_id,
            use_llm=not args.no_llm,
            require_llm=args.require_llm,
        )
        _print_json(_review_summary(result))
    elif args.command == "assemble-review":
        result = assemble_review(db_path, args.task_id)
        _print_json(_review_summary(result))
    elif args.command == "check-review":
        result = check_review_citations(db_path, args.task_id)
        _print_json(_review_summary(result))
    elif args.command == "evidence":
        _print_json([packet.to_dict() for packet in get_evidence(db_path, args.doc_id, args.node_ids)])
    elif args.command == "ask":
        result = answer_query(
            db_path,
            args.query,
            top_k=args.top_k,
            use_llm=not args.no_llm,
            require_llm=args.require_llm,
            search_mode=args.search_mode,
        )
        if args.json:
            _print_json(result)
        else:
            print(result["answer"])
            if result.get("llm_error"):
                print(f"\nDeepSeek 调用失败：{result['llm_error']}")
    elif args.command == "memory-put":
        _print_json(
            put_memory_gated(
                db_path,
                args.scope,
                args.type,
                args.subject_key,
                args.content,
                importance=args.importance,
                confidence=args.confidence,
                ttl_days=args.ttl_days,
                refs=args.refs,
                force=args.force,
            )
        )
    elif args.command == "memory-search":
        _print_json(search_memory(db_path, args.query, args.scope, args.top_k))
    elif args.command == "remember-task":
        _print_json(remember_task(db_path, args.task_id))
    elif args.command == "resume-task":
        _print_json(resume_task(db_path))
    elif args.command == "memory-compact":
        _print_json(compact_memory(db_path, scope=args.scope))
    elif args.command == "query-log":
        _print_json(
            list_query_logs(
                db_path,
                limit=args.limit,
                operation=args.operation,
                intent=args.intent,
                status=args.status,
            )
        )
    elif args.command == "query-stats":
        _print_json(query_stats(db_path, since_days=args.since_days))
    elif args.command == "feedback-put":
        _print_json(
            put_feedback(
                db_path,
                query=args.query,
                query_id=args.query_id,
                operation=args.operation,
                rating=args.rating,
                label=args.label,
                comment=args.comment,
                expected_doc_ids=args.expected_doc_id,
                expected_node_ids=args.expected_node_id,
                expected_keywords=args.expected_keyword,
                preferred_search_mode=args.preferred_search_mode,
            )
        )
    elif args.command == "feedback-list":
        _print_json(
            list_feedback(
                db_path,
                limit=args.limit,
                operation=args.operation,
                label=args.label,
                rating=args.rating,
                min_rating=args.min_rating,
            )
        )
    elif args.command == "feedback-to-eval":
        _print_json(
            build_eval_set_from_feedback(
                db_path,
                output_path=Path(args.output_json) if args.output_json else None,
                min_rating=args.min_rating,
                label=args.label,
                operation=args.operation,
                limit=args.limit,
            )
        )
    elif args.command == "eval-dashboard":
        _print_json(eval_dashboard(db_path, since_days=args.since_days, output_format=args.format))
    elif args.command == "tune-search":
        _print_json(
            tune_search(
                db_path,
                Path(args.queries_json),
                compare_modes=_comma_list(args.compare_modes),
                top_k=args.top_k,
                save_profile=args.save_profile,
            )
        )
    elif args.command == "search-profile":
        if args.profile_command == "list":
            _print_json(list_search_profiles())
        elif args.profile_command == "show":
            _print_json(get_search_profile(args.name))
        elif args.profile_command == "apply":
            _print_json(apply_search_profile(args.name))


def _list_documents(db_path: Path) -> None:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        rows = db.list_documents(conn)
        for row in rows:
            status = row["status"]
            print(f"{row['doc_id']}\t{status}\t{row['title']}\t{row['path']}")
            if status != "ready" and row["error"]:
                print(f"  error: {row['error']}")
    finally:
        conn.close()


def _print_tree(db_path: Path, doc_id: str) -> None:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        rows = db.get_doc_tree_rows(conn, doc_id)
        for row in rows:
            indent = "  " * int(row["level"])
            page = ""
            if row["page_start"]:
                page = f" [p.{row['page_start']}]"
            print(f"{indent}- {row['type']} {row['node_id']} {row['heading']}{page}")
    finally:
        conn.close()


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _log_search_results(
    db_path: Path,
    operation: str,
    query: str,
    search_mode: str,
    started: float,
    results: List[Any],
) -> None:
    warnings = []
    if not results:
        warnings.append("no_search_results")
    if any("fts_fallback" in str(getattr(result, "rank_reason", "")) for result in results):
        warnings.append("fts_fallback")
    write_query_log(
        db_path,
        operation=operation,
        query=query,
        intent=str(classify_query(query, use_llm=False).get("intent") or ""),
        search_mode=search_mode,
        status="ok" if results else "empty",
        docs_used=_unique_values(str(getattr(result, "doc_id", "")) for result in results),
        nodes_used=_unique_values(str(getattr(result, "node_id", "")) for result in results),
        latency_ms=round((time.time() - started) * 1000, 3),
        warnings=warnings,
        metrics={
            "result_count": len(results),
            "fallback_used": bool(warnings),
        },
    )


def _log_doc_results(
    db_path: Path,
    query: str,
    search_mode: str,
    started: float,
    docs: List[Dict[str, Any]],
) -> None:
    warnings = [] if docs else ["no_search_results"]
    write_query_log(
        db_path,
        operation="search-docs",
        query=query,
        intent=str(classify_query(query, use_llm=False).get("intent") or ""),
        search_mode=search_mode,
        status="ok" if docs else "empty",
        docs_used=_unique_values(str(item.get("doc_id") or "") for item in docs),
        nodes_used=_unique_values(str(item.get("best_node_id") or "") for item in docs if item.get("best_node_id")),
        latency_ms=round((time.time() - started) * 1000, 3),
        warnings=warnings,
        metrics={
            "result_count": len(docs),
            "fallback_used": any("fts_fallback" in str(item.get("rank_reason") or "") for item in docs),
        },
    )


def _unique_values(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _extract_summary(result: dict) -> dict:
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


def _fact_summary(result: dict) -> dict:
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
        "warnings": report.get("warnings") or [],
        "llm_error": result.get("llm_error", ""),
    }


def _task_summary(result: dict) -> dict:
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


def _review_summary(result: dict) -> dict:
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


def _eval_summary(result: dict) -> dict:
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


def _review_eval_summary(result: dict) -> dict:
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


def _memory_eval_summary(result: dict) -> dict:
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


def _fact_eval_summary(result: dict) -> dict:
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


def _fact_audit_cli_summary(result: dict) -> dict:
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


def _fact_conflicts_cli_summary(result: dict) -> dict:
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


def _graph_build_cli_summary(result: dict) -> dict:
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


def _quality_baseline_cli_summary(result: dict) -> dict:
    benchmark = result.get("benchmark") or {}
    embedding = result.get("embedding") or {}
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
        "real_embedding_status": real_embedding.get("status", ""),
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
        "drafted_section_count": review_draft.get("drafted_section_count", 0),
        "review_draft_path": review_draft.get("review_draft_path", ""),
        "section_revision_actions": review_draft.get("section_revision_actions") or [],
        "top_review_blockers": result.get("top_review_blockers", []),
        "duplicate_evidence_removed": review_task.get("duplicate_evidence_removed", 0),
        "citation_gap_count_before": (result.get("fact_audit_delta") or {}).get("citation_gap_count_before", 0),
        "citation_gap_count_after": (result.get("fact_audit_delta") or {}).get("citation_gap_count_after", 0),
        "tree_trace_completeness_before": ((result.get("tree_search") or {}).get("comparison_summary") or {}).get("rule_trace_completeness_avg", 0.0),
        "tree_trace_completeness_after": ((result.get("tree_search") or {}).get("comparison_summary") or {}).get("llm_trace_completeness_avg", 0.0),
        "compare_task_id": ((result.get("tasks") or {}).get("compare") or {}).get("task_id", ""),
        "review_task_id": review_task.get("task_id", ""),
        "claim_graph_id": (result.get("claim_graph") or {}).get("graph_id", ""),
        "warning_count": len(result.get("warnings") or []),
        "warnings": result.get("warnings") or [],
        "recommendations": result.get("recommendations") or [],
    }


def _suite_summary(result: dict) -> dict:
    return {
        "schema": result.get("schema"),
        "path": result.get("path"),
        "suite_id": result.get("suite_id"),
        "name": result.get("name"),
        "query_count": result.get("query_count", 0),
        "sources": result.get("sources") or [],
        "warnings": result.get("warnings") or [],
    }


def _benchmark_cli_summary(result: dict) -> dict:
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


def _failure_cli_summary(result: dict) -> dict:
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


def _case_cli_summary(result: dict) -> dict:
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


def _comma_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_report_summary(report: dict) -> dict:
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


def _layout_summary(layout: dict) -> dict:
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


if __name__ == "__main__":
    main()
