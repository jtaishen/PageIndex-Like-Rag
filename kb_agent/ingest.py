from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db, doc_cards, ingest_artifacts, parse_quality
from .config import DATA_DIR, SUPPORTED_EXTENSIONS, ensure_data_dirs
from .llm import LLMError, generate_json_object, get_llm_settings
from .models import DocumentRecord
from .parser_artifacts import build_layout_blocks
from .parsers import parser_identity_for_path, parse_document
from .tree import build_document_tree, tree_to_dict
from .utils import sha256_file, stable_id, write_json, write_jsonl


def discover_files(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path.resolve())
    return sorted(files)


def sync_directory(
    path: Path,
    db_path: Path,
    force: bool = False,
    pdf_parser: Optional[str] = None,
    doc_card_use_llm: bool = True,
) -> Dict[str, object]:
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
            result, error = _sync_file(
                conn,
                file_path,
                force=force,
                pdf_parser=pdf_parser,
                doc_card_use_llm=doc_card_use_llm,
            )
            report[result] = int(report[result]) + 1  # type: ignore[arg-type]
            if error:
                report["errors"].append({"path": str(file_path), "error": error})  # type: ignore[union-attr]
            conn.commit()
    finally:
        conn.close()
    return report


def _sync_file(
    conn,
    file_path: Path,
    force: bool = False,
    pdf_parser: Optional[str] = None,
    doc_card_use_llm: bool = True,
) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    file_hash = sha256_file(file_path)
    stat = file_path.stat()
    parser_name, parser_version = parser_identity_for_path(file_path, pdf_parser=pdf_parser)
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
    try:
        parsed = parse_document(file_path, pdf_parser=pdf_parser)
        build_layout_blocks(parsed.blocks, parsed.parser_name or parsed.file_type)
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
        artifacts = _write_artifacts(
            doc_id,
            version_id,
            artifact_dir,
            file_path,
            file_hash,
            parsed,
            nodes,
            doc_card_use_llm=doc_card_use_llm,
        )
        db.delete_document_by_path(conn, str(file_path))
        db.upsert_document(conn, record)
        db.insert_nodes(conn, nodes)
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
        _write_failure_report(doc_id, version_id, artifact_dir, file_path, file_hash, parser_name, parser_version, error)
        db.delete_document_by_path(conn, str(file_path))
        db.upsert_document(conn, record)
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


def _write_artifacts(
    doc_id: str,
    version_id: str,
    base: Path,
    source_path: Path,
    file_hash: str,
    parsed: Any,
    nodes: List[Any],
    doc_card_use_llm: bool = True,
) -> Dict[str, object]:
    base.mkdir(parents=True, exist_ok=True)
    (base / "raw_text.txt").write_text(parsed.raw_text, encoding="utf-8")
    (base / "body.md").write_text(parsed.body_md, encoding="utf-8")
    artifacts = ingest_artifacts.build_ingest_artifacts(
        doc_id,
        version_id,
        base,
        source_path,
        file_hash,
        parsed,
        nodes,
    )
    write_json(base / "structured.json", artifacts["structured"])
    write_json(base / "metadata.json", artifacts["metadata"])
    write_json(base / "references.json", artifacts["references"])
    write_json(base / "layout_blocks.json", artifacts["layout_blocks"])
    write_json(base / "tables.json", artifacts["tables"])
    write_json(base / "table_content.json", artifacts["table_content"])
    write_json(base / "table_summaries.json", artifacts["table_summaries"])
    write_json(base / "figures.json", artifacts["figures"])
    write_json(base / "reference_sections.json", artifacts["reference_sections"])
    write_json(base / "parse_report.json", artifacts["parse_report"])
    write_json(base / "tree.json", tree_to_dict(nodes))
    write_jsonl(base / "node_index.jsonl", [asdict(node) for node in nodes])
    doc_card = _build_doc_card(
        doc_id,
        version_id,
        base,
        source_path,
        file_hash,
        parsed,
        nodes,
        doc_card_use_llm=doc_card_use_llm,
        components=artifacts["components"],
    )
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
        ingest_artifacts.build_failure_parse_report(
            doc_id,
            version_id,
            base,
            source_path,
            file_hash,
            parser_name,
            parser_version,
            error,
        ),
    )


def _build_doc_card(
    doc_id: str,
    version_id: str,
    base: Path,
    source_path: Path,
    file_hash: str,
    parsed: Any,
    nodes: List[Any],
    doc_card_use_llm: bool = True,
    *,
    components: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, object]:
    return doc_cards.build_doc_card(
        doc_id,
        version_id,
        base,
        source_path,
        file_hash,
        parsed,
        nodes,
        doc_card_use_llm=doc_card_use_llm,
        json_generator=generate_json_object,
        llm_settings_getter=get_llm_settings,
        llm_error_cls=LLMError,
        components=components,
    )


def _layout_blocks_from(parsed: Any) -> List[Dict[str, Any]]:
    return ingest_artifacts.layout_blocks_from(parsed)


def _tables_from(parsed: Any) -> List[Dict[str, Any]]:
    return ingest_artifacts.tables_from(parsed)


def _table_content_from(
    parsed: Any,
    tables: List[Dict[str, Any]],
    layout_blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return ingest_artifacts.table_content_from(parsed, tables, layout_blocks)


def _table_summaries_from(parsed: Any, table_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return ingest_artifacts.table_summaries_from(parsed, table_content)


def _figures_from(parsed: Any) -> List[Dict[str, Any]]:
    return ingest_artifacts.figures_from(parsed)


def _reference_sections_from(parsed: Any) -> List[Dict[str, Any]]:
    return ingest_artifacts.reference_sections_from(parsed)


def _layout_artifact(doc_id: str, version_id: str, layout_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return ingest_artifacts.layout_artifact(doc_id, version_id, layout_blocks)


def _visual_artifact(kind: str, doc_id: str, version_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return ingest_artifacts.visual_artifact(kind, doc_id, version_id, items)


def _table_content_artifact(doc_id: str, version_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return ingest_artifacts.table_content_artifact(doc_id, version_id, items)


def _table_summaries_artifact(doc_id: str, version_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return ingest_artifacts.table_summaries_artifact(doc_id, version_id, items)


def _reference_sections_artifact(
    doc_id: str,
    version_id: str,
    sections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return ingest_artifacts.reference_sections_artifact(doc_id, version_id, sections)


def _doc_description(parsed: Any) -> str:
    return doc_cards.doc_description(parsed)


def _doc_card_summaries(parsed: Any, *, use_llm: bool = True) -> Dict[str, object]:
    return doc_cards.doc_card_summaries(
        parsed,
        use_llm=use_llm,
        json_generator=generate_json_object,
        llm_settings_getter=get_llm_settings,
        llm_error_cls=LLMError,
    )


_DOC_CARD_SUMMARY_LIMITS = doc_cards.DOC_CARD_SUMMARY_LIMITS


def _rule_doc_card_summaries(parsed: Any) -> Dict[str, object]:
    return doc_cards.rule_doc_card_summaries(parsed)


def _doc_card_summary_prompt(parsed: Any, fallback: Dict[str, object]) -> str:
    return doc_cards.doc_card_summary_prompt(parsed, fallback)


def _section_signals(parsed: Any, limit: int) -> List[Dict[str, str]]:
    return doc_cards.section_signals(parsed, limit)


def _section_excerpt(parsed: Any, heading_terms: Iterable[str], max_chars: int) -> str:
    return doc_cards.section_excerpt(parsed, heading_terms, max_chars)


def _short_summary(value: object, max_chars: int) -> str:
    return doc_cards.short_summary(value, max_chars)


def _content_excerpt(text: str, max_chars: int) -> str:
    return doc_cards.content_excerpt(text, max_chars)


def _clean_doc_card_text(text: object) -> str:
    return doc_cards.clean_doc_card_text(text)


def _is_doc_card_noise_text(text: object, *, heading: str = "", page: Optional[int] = None) -> bool:
    return doc_cards.is_doc_card_noise_text(text, heading=heading, page=page)


def _text_excerpt(text: str, max_chars: int) -> str:
    return doc_cards.text_excerpt(text, max_chars)


def _metadata_score(parsed: Any) -> float:
    return parse_quality.metadata_score(parsed)


def _structure_score(section_count: int, paragraph_count: int, page_only_tree: bool) -> float:
    return parse_quality.structure_score(section_count, paragraph_count, page_only_tree)


def _reference_score(reference_count: int, parsed: Any) -> float:
    return parse_quality.reference_score(reference_count, parsed)


def _layout_score(layout_blocks: List[Dict[str, Any]], parsed: Any) -> float:
    return parse_quality.layout_score(layout_blocks, parsed)


def _caption_score(figures: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> float:
    return parse_quality.caption_score(figures, tables)


def _caption_link_rate(figures: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> float:
    return parse_quality.caption_link_rate(figures, tables)


def _has_caption(item: Dict[str, Any]) -> bool:
    return parse_quality.has_caption(item)


def _quality_level(
    metadata_score: float,
    structure_score: float,
    reference_score: float,
    warnings: List[str],
) -> str:
    return parse_quality.quality_level(metadata_score, structure_score, reference_score, warnings)


def _quality_warnings(
    parsed: Any,
    section_count: int,
    reference_count: int,
    page_only_tree: bool,
    missing_abstract: bool,
) -> List[str]:
    return parse_quality.quality_warnings(parsed, section_count, reference_count, page_only_tree, missing_abstract)
