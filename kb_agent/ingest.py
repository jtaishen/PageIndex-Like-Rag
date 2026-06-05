from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from . import db
from .config import DATA_DIR, SUPPORTED_EXTENSIONS, ensure_data_dirs
from .models import DocumentRecord
from .parsers import ParseError, parse_document
from .tree import build_document_tree, tree_to_dict
from .utils import first_words, sha256_file, stable_id, write_json


def discover_files(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path.resolve())
    return sorted(files)


def sync_directory(path: Path, db_path: Path, force: bool = False) -> Dict[str, object]:
    ensure_data_dirs()
    root = path.expanduser().resolve()
    files = discover_files(root)
    report: Dict[str, object] = {
        "root": str(root),
        "discovered": len(files),
        "indexed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        for file_path in files:
            result = _sync_file(conn, file_path, force=force)
            report[result] = int(report[result]) + 1  # type: ignore[arg-type]
        conn.commit()
    finally:
        conn.close()
    return report


def _sync_file(conn, file_path: Path, force: bool = False) -> str:  # type: ignore[no-untyped-def]
    file_hash = sha256_file(file_path)
    stat = file_path.stat()
    existing = db.get_document_by_path(conn, str(file_path))
    if existing and existing["hash"] == file_hash and not force:
        return "skipped"

    doc_id = stable_id("doc", str(file_path))
    db.delete_document_by_path(conn, str(file_path))

    try:
        parsed = parse_document(file_path)
        nodes = build_document_tree(doc_id, parsed)
        record = DocumentRecord(
            doc_id=doc_id,
            path=str(file_path),
            hash=file_hash,
            title=parsed.title or file_path.stem,
            file_type=parsed.file_type,
            size=stat.st_size,
            mtime=stat.st_mtime,
            summary=first_words(parsed.raw_text, 80),
            status="ready",
            error="",
        )
        db.upsert_document(conn, record)
        db.insert_nodes(conn, nodes)
        _write_artifacts(doc_id, parsed, nodes)
        return "indexed"
    except Exception as exc:
        record = DocumentRecord(
            doc_id=doc_id,
            path=str(file_path),
            hash=file_hash,
            title=file_path.stem,
            file_type=file_path.suffix.lower().lstrip("."),
            size=stat.st_size,
            mtime=stat.st_mtime,
            summary="",
            status="failed",
            error=str(exc),
        )
        db.upsert_document(conn, record)
        return "failed"


def _write_artifacts(doc_id: str, parsed, nodes) -> None:  # type: ignore[no-untyped-def]
    base = DATA_DIR / "parsed" / doc_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "raw_text.txt").write_text(parsed.raw_text, encoding="utf-8")
    write_json(
        base / "parse_report.json",
        {
            "doc_id": doc_id,
            "title": parsed.title,
            "file_type": parsed.file_type,
            "metadata": parsed.metadata,
            "block_count": len(parsed.blocks),
            "node_count": len(nodes),
            "created_at": time.time(),
        },
    )
    write_json(base / "tree.json", tree_to_dict(nodes))
    write_json(base / "node_index.json", [asdict(node) for node in nodes])

