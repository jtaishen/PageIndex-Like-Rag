from __future__ import annotations

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
from .claim_frames import (
    extract_claim_frames,
    extract_evidence_units,
    get_claim_frames,
    get_evidence_units,
    verify_claim_frames,
)
from .cli_summaries import (
    benchmark_cli_summary,
    case_cli_summary,
    comma_list,
    eval_summary,
    extract_summary,
    fact_audit_cli_summary,
    fact_conflicts_cli_summary,
    fact_eval_summary,
    fact_summary,
    failure_cli_summary,
    graph_build_cli_summary,
    layout_summary,
    memory_eval_summary,
    parse_report_summary,
    quality_baseline_cli_summary,
    review_eval_summary,
    review_summary,
    suite_summary,
    task_summary,
)
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


def dispatch_command(args: Any, db_path: Path) -> None:
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
        compare_modes = comma_list(args.compare_modes)
        _print_json(
            eval_summary(
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
        _print_json(review_eval_summary(eval_review(db_path, args.task_id)))
    elif args.command == "eval-memory":
        _print_json(memory_eval_summary(eval_memory(db_path)))
    elif args.command == "eval-facts":
        _print_json(fact_eval_summary(eval_facts(db_path, doc_ids=args.doc_id or None)))
    elif args.command == "audit-facts":
        _print_json(fact_audit_cli_summary(audit_facts(db_path, doc_ids=args.doc_id or None, min_confidence=args.min_confidence)))
    elif args.command == "fact-conflicts":
        _print_json(
            fact_conflicts_cli_summary(
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
            graph_build_cli_summary(
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
            quality_baseline_cli_summary(
                run_quality_baseline(
                    db_path,
                    Path(args.corpus_path),
                    force=not args.no_force,
                    top_k=args.top_k,
                    use_llm=args.with_llm,
                    embedding_provider=args.embedding_provider,
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
                suite_summary(
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
            benchmark_cli_summary(
                run_benchmark(
                    db_path,
                    args.suite_name,
                    compare_modes=comma_list(args.compare_modes),
                    top_k=args.top_k,
                )
            )
        )
    elif args.command == "analyze-failures":
        _print_json(failure_cli_summary(analyze_failures(db_path, args.benchmark_id)))
    elif args.command == "case-study":
        _print_json(
            case_cli_summary(
                generate_case_study(
                    db_path,
                    args.query,
                    doc_ids=args.doc_id or None,
                    compare_modes=comma_list(args.compare_modes),
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
        _print_json(parse_report_summary(get_parse_report(db_path, args.doc_id, args.version_id)))
    elif args.command == "layout":
        _print_json(layout_summary(get_layout_blocks(db_path, args.doc_id, args.version_id)))
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
        _print_json(extract_summary(result))
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
        _print_json(fact_summary(result))
    elif args.command == "extract-evidence-units":
        _print_json(extract_evidence_units(db_path, args.doc_id, force=args.force))
    elif args.command == "evidence-units":
        _print_json(get_evidence_units(db_path, args.doc_id))
    elif args.command == "extract-claim-frames":
        _print_json(
            extract_claim_frames(
                db_path,
                args.doc_id,
                force=args.force,
                use_llm=not args.no_llm,
                require_llm=args.require_llm,
            )
        )
    elif args.command == "claim-frames":
        _print_json(get_claim_frames(db_path, args.doc_id))
    elif args.command == "verify-claim-frames":
        _print_json(verify_claim_frames(db_path, doc_ids=args.doc_id or None))
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
        _print_json(task_summary(result))
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
        _print_json(task_summary(result))
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
        _print_json(review_summary(result))
    elif args.command == "assemble-review":
        result = assemble_review(db_path, args.task_id)
        _print_json(review_summary(result))
    elif args.command == "check-review":
        result = check_review_citations(db_path, args.task_id)
        _print_json(review_summary(result))
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
                compare_modes=comma_list(args.compare_modes),
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
