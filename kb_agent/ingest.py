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
        summary = _doc_description(parsed)
        record = DocumentRecord(
            doc_id=doc_id,
            path=str(file_path),
            hash=file_hash,
            title=parsed.title or file_path.stem,
            file_type=parsed.file_type,
            size=stat.st_size,
            mtime=stat.st_mtime,
            summary=summary,
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
    section_nodes = [node for node in nodes if node.kind == "section"]
    section_count = len(section_nodes)
    page_values = [node.page_start for node in nodes if node.page_start]
    page_count = max(page_values) if page_values else parsed.metadata.get("pages")
    paragraph_count = sum(1 for node in nodes if node.kind == "paragraph")
    reference_count = sum(1 for node in nodes if node.kind == "reference" and node.text)
    figure_count = sum(1 for node in nodes if node.kind == "figure")
    table_count = sum(1 for node in nodes if node.kind == "table")
    page_only_tree = section_count == 0 and any(node.kind == "page" for node in nodes)
    missing_abstract = not bool(parsed.metadata.get("abstract"))
    quality_warnings = _quality_warnings(parsed, section_count, reference_count, page_only_tree, missing_abstract)
    parse_quality = {
        "schema": "parse_quality.v0",
        "doc_id": doc_id,
        "version_id": version_id,
        "page_count": page_count,
        "section_count": section_count,
        "paragraph_count": paragraph_count,
        "reference_count": reference_count,
        "figure_count": figure_count,
        "table_count": table_count,
        "missing_abstract": missing_abstract,
        "page_only_tree": page_only_tree,
        "quality_warnings": quality_warnings,
    }
    sections = [
        {
            "node_id": node.node_id,
            "title": node.heading,
            "node_path": node.node_path,
            "page_range": [node.page_start, node.page_end],
            "level": node.level,
            "type": node.kind,
        }
        for node in section_nodes[:80]
    ]
    description = _doc_description(parsed)
    return {
        "schema": "doc_card.v0",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": parsed.title,
        "path": str(source_path),
        "file_type": parsed.file_type,
        "file_hash": file_hash,
        "summary": description,
        "description": description,
        "authors": parsed.metadata.get("authors") or [],
        "year": parsed.metadata.get("year"),
        "venue": parsed.metadata.get("venue") or "",
        "doi": parsed.metadata.get("doi") or "",
        "abstract": parsed.metadata.get("abstract") or "",
        "keywords": parsed.metadata.get("keywords") or [],
        "parser_name": parsed.parser_name,
        "parser_version": parsed.parser_version,
        "page_count": page_count,
        "block_count": len(parsed.blocks),
        "node_count": len(nodes),
        "section_count": section_count,
        "sections": sections,
        "quality_warnings": quality_warnings,
        "parse_quality": parse_quality,
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


def _doc_description(parsed) -> str:  # type: ignore[no-untyped-def]
    abstract = str(parsed.metadata.get("abstract") or "").strip()
    if abstract:
        return _text_excerpt(abstract, 500)
    keywords = parsed.metadata.get("keywords") or []
    if keywords:
        return _text_excerpt(f"{parsed.title}。关键词：{'、'.join(str(item) for item in keywords)}", 500)
    return _text_excerpt(parsed.raw_text, 500)


def _text_excerpt(text: str, max_chars: int) -> str:
    compacted = " ".join(str(text).split())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[:max_chars].rstrip() + " ..."


def _quality_warnings(
    parsed,  # type: ignore[no-untyped-def]
    section_count: int,
    reference_count: int,
    page_only_tree: bool,
    missing_abstract: bool,
) -> List[str]:
    warnings = list(parsed.parse_warnings)
    if missing_abstract:
        warnings.append("missing_abstract")
    if page_only_tree:
        warnings.append("page_only_tree")
    if parsed.file_type == "pdf" and section_count < 2:
        warnings.append("low_section_count")
    references_status = (parsed.references or {}).get("status")
    if reference_count == 0 and references_status != "extracted":
        warnings.append("missing_references")
    return warnings
