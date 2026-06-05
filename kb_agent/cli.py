from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import db
from .answer import answer_query, route_documents
from .config import resolve_db_path
from .ingest import sync_directory
from .memory import put_memory, search_memory
from .search import get_evidence, search_nodes


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

    evidence_parser = subparsers.add_parser("evidence", help="Get evidence packets")
    evidence_parser.add_argument("doc_id")
    evidence_parser.add_argument("node_ids", nargs="+")

    ask_parser = subparsers.add_parser("ask", help="Return grounded evidence for a question")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--top-k", type=int, default=6)

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
    elif args.command == "evidence":
        _print_json([packet.to_dict() for packet in get_evidence(db_path, args.doc_id, args.node_ids)])
    elif args.command == "ask":
        result = answer_query(db_path, args.query, top_k=args.top_k)
        print(result["answer"])
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


if __name__ == "__main__":
    main()

