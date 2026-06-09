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
from .config import resolve_db_path
from .embeddings import build_semantic_index, semantic_index_status
from .eval import eval_facts, eval_memory, eval_review, eval_search
from .facts import extract_facts, fact_search, get_claims, get_entities, get_fact_graph, get_relations
from .feedback import build_eval_set_from_feedback, eval_dashboard, list_feedback, put_feedback
from .ingest import sync_directory
from .insights import extract_doc_insights
from .memory import compact_memory, put_memory_gated, remember_task, resume_task, search_memory
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
        "citation_coverage_score": report.get("citation_coverage_score") or citation_check.get("coverage_score"),
        "missing_ref_count": len(citation_check.get("missing_refs") or []),
        "unsupported_paragraph_count": len(citation_check.get("unsupported_paragraphs") or []),
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
