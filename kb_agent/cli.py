from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import db
from .answer import answer_query, route_documents
from .artifacts import get_citation_map, get_doc_card, get_innovations, get_parse_quality, list_artifacts
from .config import resolve_db_path
from .ingest import sync_directory
from .insights import extract_doc_insights
from .memory import put_memory, search_memory
from .search import get_evidence, search_nodes
from .tasks import compare_papers, generate_review_plan, get_task_artifact


def main(argv: Any = None) -> None:
    parser = argparse.ArgumentParser(prog="kb", description="PageIndex-like knowledge base MVP")
    parser.add_argument("--db", default=None, help="SQLite database path. Defaults to data/kb.sqlite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Index a directory or supported file")
    sync_parser.add_argument("path")
    sync_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("list", help="List indexed documents")

    search_parser = subparsers.add_parser("search", help="Search evidence nodes")
    search_parser.add_argument("query")
    search_parser.add_argument("--doc-id", default=None)
    search_parser.add_argument("--top-k", type=int, default=8)

    docs_parser = subparsers.add_parser("docs", help="Search candidate documents")
    docs_parser.add_argument("query")
    docs_parser.add_argument("--top-k", type=int, default=8)

    tree_parser = subparsers.add_parser("tree", help="Show a document tree")
    tree_parser.add_argument("doc_id")

    card_parser = subparsers.add_parser("card", help="Show a document card")
    card_parser.add_argument("doc_id")

    artifacts_parser = subparsers.add_parser("artifacts", help="List generated artifacts for a document")
    artifacts_parser.add_argument("doc_id")
    artifacts_parser.add_argument("--version-id", default=None)

    quality_parser = subparsers.add_parser("quality", help="Show parse quality for a document")
    quality_parser.add_argument("doc_id")

    extract_parser = subparsers.add_parser("extract", help="Extract paper insight artifacts")
    extract_parser.add_argument("doc_id")
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--no-llm", action="store_true", help="Use rule-based extraction only")
    extract_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")

    innovations_parser = subparsers.add_parser("innovations", help="Show extracted innovation artifact")
    innovations_parser.add_argument("doc_id")

    citations_parser = subparsers.add_parser("citations", help="Show extracted citation map artifact")
    citations_parser.add_argument("doc_id")

    compare_parser = subparsers.add_parser("compare", help="Compare papers with grounded task artifacts")
    compare_parser.add_argument("query")
    compare_parser.add_argument("--doc-id", action="append", default=[], help="Limit comparison to a document id; repeatable")
    compare_parser.add_argument("--top-k-docs", type=int, default=5)
    compare_parser.add_argument("--no-llm", action="store_true", help="Use rule-based comparison only")
    compare_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")

    review_parser = subparsers.add_parser("generate-review", help="Generate review planning task artifacts")
    review_parser.add_argument("topic")
    review_parser.add_argument("--doc-id", action="append", default=[], help="Limit review planning to a document id; repeatable")
    review_parser.add_argument("--top-k-docs", type=int, default=8)
    review_parser.add_argument("--no-llm", action="store_true", help="Use rule-based planning only")
    review_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")

    task_artifact_parser = subparsers.add_parser("task-artifact", help="Read a v0.5 task artifact")
    task_artifact_parser.add_argument("task_id")
    task_artifact_parser.add_argument("name")

    evidence_parser = subparsers.add_parser("evidence", help="Get evidence packets")
    evidence_parser.add_argument("doc_id")
    evidence_parser.add_argument("node_ids", nargs="+")

    ask_parser = subparsers.add_parser("ask", help="Answer a question using grounded evidence")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--top-k", type=int, default=6)
    ask_parser.add_argument("--no-llm", action="store_true", help="Only print evidence; do not call DeepSeek")
    ask_parser.add_argument("--require-llm", action="store_true", help="Fail if DeepSeek cannot be called")
    ask_parser.add_argument("--json", action="store_true", help="Print full JSON result")

    mem_put_parser = subparsers.add_parser("memory-put", help="Store explicit long-term memory")
    mem_put_parser.add_argument("scope")
    mem_put_parser.add_argument("type")
    mem_put_parser.add_argument("subject_key")
    mem_put_parser.add_argument("content")
    mem_put_parser.add_argument("--importance", type=float, default=0.5)
    mem_put_parser.add_argument("--confidence", type=float, default=1.0)

    mem_search_parser = subparsers.add_parser("memory-search", help="Search memory items")
    mem_search_parser.add_argument("query")
    mem_search_parser.add_argument("--scope", default=None)
    mem_search_parser.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args(argv)
    db_path = resolve_db_path(args.db)

    if args.command == "sync":
        _print_json(sync_directory(Path(args.path), db_path, force=args.force))
    elif args.command == "list":
        _list_documents(db_path)
    elif args.command == "search":
        _print_json([result.__dict__ for result in search_nodes(db_path, args.query, args.doc_id, args.top_k)])
    elif args.command == "docs":
        _print_json(route_documents(db_path, args.query, args.top_k))
    elif args.command == "tree":
        _print_tree(db_path, args.doc_id)
    elif args.command == "card":
        _print_json(get_doc_card(db_path, args.doc_id))
    elif args.command == "artifacts":
        _print_json(list_artifacts(db_path, args.doc_id, args.version_id))
    elif args.command == "quality":
        _print_json(get_parse_quality(db_path, args.doc_id))
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
    elif args.command == "compare":
        result = compare_papers(
            db_path,
            args.query,
            doc_ids=args.doc_id,
            top_k_docs=args.top_k_docs,
            use_llm=not args.no_llm,
            require_llm=args.require_llm,
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
        )
        _print_json(_task_summary(result))
    elif args.command == "task-artifact":
        _print_json(get_task_artifact(db_path, args.task_id, args.name))
    elif args.command == "evidence":
        _print_json([packet.to_dict() for packet in get_evidence(db_path, args.doc_id, args.node_ids)])
    elif args.command == "ask":
        result = answer_query(
            db_path,
            args.query,
            top_k=args.top_k,
            use_llm=not args.no_llm,
            require_llm=args.require_llm,
        )
        if args.json:
            _print_json(result)
        else:
            print(result["answer"])
            if result.get("llm_error"):
                print(f"\nDeepSeek 调用失败：{result['llm_error']}")
    elif args.command == "memory-put":
        _print_json(
            put_memory(
                db_path,
                args.scope,
                args.type,
                args.subject_key,
                args.content,
                importance=args.importance,
                confidence=args.confidence,
            )
        )
    elif args.command == "memory-search":
        _print_json(search_memory(db_path, args.query, args.scope, args.top_k))


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


if __name__ == "__main__":
    main()
