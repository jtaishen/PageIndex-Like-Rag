from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from . import db
from .config import DATA_DIR, SUPPORTED_EXTENSIONS, ensure_data_dirs
from .models import DocumentRecord
from .parsers import parser_identity_for_path, parse_document
from .tree import build_document_tree, tree_to_dict
from .utils import first_words, sha256_file, stable_id, write_json, write_jsonl


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
            result, error = _sync_file(conn, file_path, force=force)
            report[result] = int(report[result]) + 1  # type: ignore[arg-type]
            if error:
                report["errors"].append({"path": str(file_path), "error": error})  # type: ignore[union-attr]
        conn.commit()
    finally:
        conn.close()
    return report


def _sync_file(conn, file_path: Path, force: bool = False) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    file_hash = sha256_file(file_path)
    stat = file_path.stat()
    parser_name, parser_version = parser_identity_for_path(file_path)
    existing = db.get_document_by_path(conn, str(file_path))
    if (
        existing
        and existing["hash"] == file_hash
        and existing["parser_name"] == parser_name
        and existing["parser_version"] == parser_version
        and not force
    ):
        return "skipped", ""

    doc_id = stable_id("doc", str(file_path))
    version_id = stable_id("ver", doc_id, file_hash, parser_name, parser_version)
    artifact_dir = DATA_DIR / "parsed" / doc_id / version_id
    db.delete_document_by_path(conn, str(file_path))

    try:
        parsed = parse_document(file_path)
        nodes = build_document_tree(doc_id, parsed, doc_hash=file_hash)
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
            authors=list(parsed.metadata.get("authors") or []),
            year=parsed.metadata.get("year"),
            venue=str(parsed.metadata.get("venue") or ""),
            doi=str(parsed.metadata.get("doi") or ""),
            abstract=str(parsed.metadata.get("abstract") or ""),
            keywords=list(parsed.metadata.get("keywords") or []),
            parser_name=parsed.parser_name,
            parser_version=parsed.parser_version,
        )
        db.upsert_document(conn, record)
        db.insert_nodes(conn, nodes)
        artifacts = _write_artifacts(doc_id, version_id, artifact_dir, file_path, file_hash, parsed, nodes)
        db.insert_document_version(
            conn,
            version_id=version_id,
            doc_id=doc_id,
            file_hash=file_hash,
            parser_name=parsed.parser_name,
            parser_version=parsed.parser_version,
            artifact_dir=str(artifact_dir),
            parse_status="ready",
        )
        db.upsert_doc_card(conn, doc_id, version_id, artifacts["doc_card"])  # type: ignore[arg-type]
        return "indexed", ""
    except Exception as exc:
        error = str(exc)
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
            error=error,
            parser_name=parser_name,
            parser_version=parser_version,
        )
        db.upsert_document(conn, record)
        _write_failure_report(doc_id, version_id, artifact_dir, file_path, file_hash, parser_name, parser_version, error)
        db.insert_document_version(
            conn,
            version_id=version_id,
            doc_id=doc_id,
            file_hash=file_hash,
            parser_name=parser_name,
            parser_version=parser_version,
            artifact_dir=str(artifact_dir),
            parse_status="failed",
            error=error,
        )
        return "failed", error


def _write_artifacts(doc_id: str, version_id: str, base: Path, source_path: Path, file_hash: str, parsed, nodes) -> Dict[str, object]:  # type: ignore[no-untyped-def]
    base.mkdir(parents=True, exist_ok=True)
    (base / "raw_text.txt").write_text(parsed.raw_text, encoding="utf-8")
    (base / "body.md").write_text(parsed.body_md, encoding="utf-8")
    write_json(base / "structured.json", parsed.structured)
    write_json(base / "metadata.json", parsed.metadata)
    write_json(base / "references.json", parsed.references)
    write_json(
        base / "parse_report.json",
        {
            "schema": "parse_report.v0",
            "doc_id": doc_id,
            "version_id": version_id,
            "title": parsed.title,
            "file_type": parsed.file_type,
            "source_path": str(source_path),
            "file_hash": file_hash,
            "parser_name": parsed.parser_name,
            "parser_version": parsed.parser_version,
            "status": "ready",
            "error": "",
            "warnings": parsed.parse_warnings,
            "metadata": parsed.metadata,
            "block_count": len(parsed.blocks),
            "node_count": len(nodes),
            "artifact_dir": str(base),
            "created_at": time.time(),
        },
    )
    write_json(base / "tree.json", tree_to_dict(nodes))
    write_jsonl(base / "node_index.jsonl", [asdict(node) for node in nodes])
    doc_card = _build_doc_card(doc_id, version_id, base, source_path, file_hash, parsed, nodes)
    write_json(base / "doc_card.json", doc_card)
    write_json(base / "innovation.json", {"schema": "innovation.v0", "status": "not_extracted", "items": []})
    write_json(
        base / "citation_map.json",
        {"schema": "citation_map.v0", "status": "not_extracted", "citations": [], "relations": []},
    )
    return {"doc_card": doc_card}


def _write_failure_report(
    doc_id: str,
    version_id: str,
    base: Path,
    source_path: Path,
    file_hash: str,
    parser_name: str,
    parser_version: str,
    error: str,
) -> None:
    base.mkdir(parents=True, exist_ok=True)
    write_json(
        base / "parse_report.json",
        {
            "schema": "parse_report.v0",
            "doc_id": doc_id,
            "version_id": version_id,
            "title": source_path.stem,
            "file_type": source_path.suffix.lower().lstrip("."),
            "source_path": str(source_path),
            "file_hash": file_hash,
            "parser_name": parser_name,
            "parser_version": parser_version,
            "status": "failed",
            "error": error,
            "warnings": [],
            "metadata": {},
            "block_count": 0,
            "node_count": 0,
            "artifact_dir": str(base),
            "created_at": time.time(),
        },
    )


def _build_doc_card(doc_id: str, version_id: str, base: Path, source_path: Path, file_hash: str, parsed, nodes) -> Dict[str, object]:  # type: ignore[no-untyped-def]
    section_count = sum(1 for node in nodes if node.kind == "section")
    page_values = [node.page_start for node in nodes if node.page_start]
    return {
        "schema": "doc_card.v0",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": parsed.title,
        "path": str(source_path),
        "file_type": parsed.file_type,
        "file_hash": file_hash,
        "summary": first_words(parsed.raw_text, 80),
        "authors": parsed.metadata.get("authors") or [],
        "year": parsed.metadata.get("year"),
        "venue": parsed.metadata.get("venue") or "",
        "doi": parsed.metadata.get("doi") or "",
        "abstract": parsed.metadata.get("abstract") or "",
        "keywords": parsed.metadata.get("keywords") or [],
        "parser_name": parsed.parser_name,
        "parser_version": parsed.parser_version,
        "page_count": max(page_values) if page_values else parsed.metadata.get("pages"),
        "block_count": len(parsed.blocks),
        "node_count": len(nodes),
        "section_count": section_count,
        "artifact_dir": str(base),
        "artifacts": [
            "raw_text.txt",
            "body.md",
            "structured.json",
            "metadata.json",
            "references.json",
            "parse_report.json",
            "tree.json",
            "node_index.jsonl",
            "doc_card.json",
            "innovation.json",
            "citation_map.json",
        ],
        "created_at": time.time(),
    }
