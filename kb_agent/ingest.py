from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .config import DATA_DIR, SUPPORTED_EXTENSIONS, ensure_data_dirs
from .models import DocumentRecord
from .parsers import (
    build_layout_blocks,
    build_reference_sections,
    build_table_content,
    build_table_summaries,
    build_visual_items,
    enhance_table_items,
    parser_identity_for_path,
    parse_document,
    table_parse_score,
    table_warning_count,
)
from .tree import build_document_tree, tree_to_dict
from .llm import LLMError, generate_json_object, get_llm_settings
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
    db.delete_document_by_path(conn, str(file_path))

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
        db.upsert_document(conn, record)
        db.insert_nodes(conn, nodes)
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


def _write_artifacts(
    doc_id: str,
    version_id: str,
    base: Path,
    source_path: Path,
    file_hash: str,
    parsed,
    nodes,
    doc_card_use_llm: bool = True,
) -> Dict[str, object]:  # type: ignore[no-untyped-def]
    base.mkdir(parents=True, exist_ok=True)
    (base / "raw_text.txt").write_text(parsed.raw_text, encoding="utf-8")
    (base / "body.md").write_text(parsed.body_md, encoding="utf-8")
    layout_blocks = _layout_blocks_from(parsed)
    tables = _tables_from(parsed)
    table_content = _table_content_from(parsed, tables, layout_blocks)
    tables = enhance_table_items(tables, table_content)
    table_summaries = _table_summaries_from(parsed, table_content)
    figures = _figures_from(parsed)
    reference_sections = _reference_sections_from(parsed)
    structured_payload = dict(parsed.structured or {})
    structured_payload.setdefault("schema", "structured.v0")
    structured_payload["layout_schema"] = "layout_blocks.v1"
    structured_payload["layout_blocks"] = layout_blocks
    structured_payload["layout_blocks_count"] = len(layout_blocks)
    structured_payload["tables"] = tables
    structured_payload["table_content"] = table_content
    structured_payload["table_summaries"] = table_summaries
    structured_payload["figures"] = figures
    structured_payload["table_count"] = len(tables)
    structured_payload["table_content_count"] = len(table_content)
    structured_payload["table_parse_score"] = table_parse_score(table_content, tables)
    structured_payload["table_warning_count"] = table_warning_count(table_content)
    structured_payload["figure_count"] = len(figures)
    structured_payload["reference_sections"] = reference_sections
    structured_payload["reference_section_count"] = len(reference_sections)
    write_json(base / "structured.json", structured_payload)
    write_json(base / "metadata.json", parsed.metadata)
    write_json(base / "references.json", parsed.references)
    write_json(base / "layout_blocks.json", _layout_artifact(doc_id, version_id, layout_blocks))
    write_json(base / "tables.json", _visual_artifact("tables", doc_id, version_id, tables))
    write_json(base / "table_content.json", _table_content_artifact(doc_id, version_id, table_content))
    write_json(base / "table_summaries.json", _table_summaries_artifact(doc_id, version_id, table_summaries))
    write_json(base / "figures.json", _visual_artifact("figures", doc_id, version_id, figures))
    write_json(base / "reference_sections.json", _reference_sections_artifact(doc_id, version_id, reference_sections))
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
            "requested_pdf_parser": (parsed.metadata.get("_parse_diagnostics") or {}).get("requested_pdf_parser"),
            "parser_chain": (parsed.metadata.get("_parse_diagnostics") or {}).get("parser_chain", [parsed.parser_name]),
            "fallback_used": (parsed.metadata.get("_parse_diagnostics") or {}).get("fallback_used", False),
            "external_parser_errors": (parsed.metadata.get("_parse_diagnostics") or {}).get("external_parser_errors", []),
            "adapter_statuses": (parsed.metadata.get("_parse_diagnostics") or {}).get("adapter_statuses", {}),
            "status": "ready",
            "error": "",
            "warnings": parsed.parse_warnings,
            "metadata": parsed.metadata,
            "block_count": len(parsed.blocks),
            "layout_block_count": len(layout_blocks),
            "table_count": len(tables),
            "table_content_count": len(table_content),
            "table_parse_score": table_parse_score(table_content, tables),
            "table_warning_count": table_warning_count(table_content),
            "figure_count": len(figures),
            "reference_section_count": len(reference_sections),
            "noise_removed_count": int(parsed.metadata.get("noise_removed_count") or 0),
            "node_count": len(nodes),
            "artifact_dir": str(base),
            "created_at": time.time(),
        },
    )
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
            "requested_pdf_parser": parser_name.removeprefix("pdf_") if parser_name.startswith("pdf_") else "",
            "parser_chain": [parser_name],
            "fallback_used": False,
            "external_parser_errors": [error] if error else [],
            "adapter_statuses": {},
            "status": "failed",
            "error": error,
            "warnings": [],
            "metadata": {},
            "block_count": 0,
            "layout_block_count": 0,
            "table_count": 0,
            "table_content_count": 0,
            "table_parse_score": 0.0,
            "table_warning_count": 0,
            "figure_count": 0,
            "reference_section_count": 0,
            "noise_removed_count": 0,
            "node_count": 0,
            "artifact_dir": str(base),
            "created_at": time.time(),
        },
    )


def _build_doc_card(
    doc_id: str,
    version_id: str,
    base: Path,
    source_path: Path,
    file_hash: str,
    parsed,
    nodes,
    doc_card_use_llm: bool = True,
) -> Dict[str, object]:  # type: ignore[no-untyped-def]
    layout_blocks = _layout_blocks_from(parsed)
    tables = _tables_from(parsed)
    table_content = _table_content_from(parsed, tables, layout_blocks)
    tables = enhance_table_items(tables, table_content)
    figures = _figures_from(parsed)
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
    diagnostics = parsed.metadata.get("_parse_diagnostics") or {}
    metadata_score = _metadata_score(parsed)
    structure_score = _structure_score(section_count, paragraph_count, page_only_tree)
    reference_score = _reference_score(reference_count, parsed)
    layout_score = _layout_score(layout_blocks, parsed)
    caption_score = _caption_score(figures, tables)
    caption_link_rate = _caption_link_rate(figures, tables)
    current_table_parse_score = table_parse_score(table_content, tables)
    current_table_warning_count = table_warning_count(table_content)
    if parsed.file_type == "pdf" and layout_score < 0.45:
        quality_warnings.append("weak_layout_blocks")
    if (figures or tables) and caption_link_rate < 0.5:
        quality_warnings.append("low_caption_link_rate")
    if tables and not table_content:
        quality_warnings.append("missing_table_content")
    if current_table_parse_score < 0.5 and tables:
        quality_warnings.append("weak_table_parse")
    quality_level = _quality_level(metadata_score, structure_score, reference_score, quality_warnings)
    parse_quality = {
        "schema": "parse_quality.v0",
        "doc_id": doc_id,
        "version_id": version_id,
        "quality_level": quality_level,
        "page_count": page_count,
        "section_count": section_count,
        "paragraph_count": paragraph_count,
        "reference_count": reference_count,
        "figure_count": figure_count,
        "table_count": table_count,
        "table_content_count": len(table_content),
        "table_parse_score": current_table_parse_score,
        "table_warning_count": current_table_warning_count,
        "parser_chain": diagnostics.get("parser_chain", [parsed.parser_name]),
        "fallback_used": diagnostics.get("fallback_used", False),
        "metadata_score": metadata_score,
        "structure_score": structure_score,
        "reference_score": reference_score,
        "layout_score": layout_score,
        "caption_score": caption_score,
        "noise_removed_count": int(parsed.metadata.get("noise_removed_count") or 0),
        "layout_block_count": len(layout_blocks),
        "caption_link_rate": caption_link_rate,
        "warning_count": len(quality_warnings),
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
    summaries = _doc_card_summaries(parsed, use_llm=doc_card_use_llm)
    description = str(summaries.get("description") or _doc_description(parsed))
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
        "method_summary": summaries.get("method_summary", ""),
        "innovation_summary": summaries.get("innovation_summary", ""),
        "limitation_summary": summaries.get("limitation_summary", ""),
        "summary_source": summaries.get("summary_source", "rule"),
        "summary_warnings": summaries.get("summary_warnings", []),
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
            "layout_blocks.json",
            "tables.json",
            "table_content.json",
            "table_summaries.json",
            "figures.json",
            "reference_sections.json",
            "parse_report.json",
            "tree.json",
            "node_index.jsonl",
            "doc_card.json",
            "innovation.json",
            "citation_map.json",
        ],
        "created_at": time.time(),
    }


def _layout_blocks_from(parsed) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    blocks = parsed.structured.get("layout_blocks") if isinstance(parsed.structured, dict) else []
    if isinstance(blocks, list):
        cleaned = [item for item in blocks if isinstance(item, dict)]
        if cleaned:
            return cleaned
    return build_layout_blocks(parsed.blocks, parsed.parser_name or parsed.file_type)


def _tables_from(parsed) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    tables = parsed.structured.get("tables") if isinstance(parsed.structured, dict) else []
    if isinstance(tables, list):
        cleaned = [item for item in tables if isinstance(item, dict)]
        if cleaned and all(item.get("schema") == "table.v1" for item in cleaned):
            return cleaned
    return build_visual_items(_layout_blocks_from(parsed), "table")


def _table_content_from(parsed, tables: List[Dict[str, Any]], layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    content = parsed.structured.get("table_content") if isinstance(parsed.structured, dict) else []
    if isinstance(content, list):
        cleaned = [item for item in content if isinstance(item, dict) and item.get("schema") == "table_content.v1"]
        if cleaned:
            return cleaned
    raw_tables = parsed.structured.get("tables") if isinstance(parsed.structured, dict) else []
    raw_tables = raw_tables if isinstance(raw_tables, list) else []
    return build_table_content(parsed.blocks, layout_blocks, raw_tables=raw_tables)


def _table_summaries_from(parsed, table_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    summaries = parsed.structured.get("table_summaries") if isinstance(parsed.structured, dict) else []
    if isinstance(summaries, list):
        cleaned = [item for item in summaries if isinstance(item, dict) and item.get("schema") == "table_summary.v1"]
        if cleaned:
            return cleaned
    return build_table_summaries(table_content)


def _figures_from(parsed) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    figures = parsed.structured.get("figures") if isinstance(parsed.structured, dict) else []
    if isinstance(figures, list):
        cleaned = [item for item in figures if isinstance(item, dict)]
        if cleaned and all(item.get("schema") == "figure.v1" for item in cleaned):
            return cleaned
    return build_visual_items(_layout_blocks_from(parsed), "figure")


def _reference_sections_from(parsed) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    sections = parsed.structured.get("reference_sections") if isinstance(parsed.structured, dict) else []
    if isinstance(sections, list):
        cleaned = [item for item in sections if isinstance(item, dict)]
        if cleaned:
            return cleaned
    return build_reference_sections(_layout_blocks_from(parsed), parsed.references)


def _layout_artifact(doc_id: str, version_id: str, layout_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    pages = set()
    for block in layout_blocks:
        block_type = str(block.get("type") or "")
        if block_type:
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        page = block.get("page")
        if isinstance(page, int):
            pages.add(page)
    return {
        "schema": "layout_blocks.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(layout_blocks),
        "type_counts": type_counts,
        "page_count": len(pages),
        "blocks": layout_blocks,
    }


def _visual_artifact(kind: str, doc_id: str, version_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema": f"{kind}.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(items),
        kind: items,
    }


def _table_content_artifact(doc_id: str, version_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema": "table_content.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(items),
        "table_content": items,
    }


def _table_summaries_artifact(doc_id: str, version_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema": "table_summaries.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(items),
        "table_summaries": items,
    }


def _reference_sections_artifact(
    doc_id: str,
    version_id: str,
    sections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema": "reference_sections.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(sections),
        "reference_sections": sections,
    }


def _doc_description(parsed) -> str:  # type: ignore[no-untyped-def]
    abstract = str(parsed.metadata.get("abstract") or "").strip()
    if abstract:
        return _text_excerpt(abstract, 500)
    keywords = parsed.metadata.get("keywords") or []
    if keywords:
        return _text_excerpt(f"{parsed.title}。关键词：{'、'.join(str(item) for item in keywords)}", 500)
    return _text_excerpt(parsed.raw_text, 500)


def _doc_card_summaries(parsed, *, use_llm: bool = True) -> Dict[str, object]:  # type: ignore[no-untyped-def]
    fallback = _rule_doc_card_summaries(parsed)
    if not use_llm:
        return {
            **fallback,
            "summary_source": "rule",
            "summary_warnings": ["deepseek_summary_skipped:disabled"],
        }
    if get_llm_settings() is None:
        return {
            **fallback,
            "summary_source": "rule",
            "summary_warnings": ["deepseek_summary_skipped:not_configured"],
        }
    try:
        payload = generate_json_object(
            "你是论文知识库的文档卡片摘要器。只返回 JSON object，不要输出正文、证据长摘录或 prompt。",
            _doc_card_summary_prompt(parsed, fallback),
            timeout_seconds=20,
            retry_count=1,
            operation="doc_card_summary",
            stage="ingest",
        )
    except LLMError as exc:
        return {
            **fallback,
            "summary_source": "rule",
            "summary_warnings": [f"deepseek_summary_failed:{exc.error_type}"],
        }

    summaries: Dict[str, object] = dict(fallback)
    used_llm = False
    for field, max_chars in _DOC_CARD_SUMMARY_LIMITS.items():
        value = _short_summary(payload.get(field), max_chars)
        if value:
            summaries[field] = value
            used_llm = True
    if not used_llm:
        return {
            **fallback,
            "summary_source": "rule",
            "summary_warnings": ["deepseek_summary_failed:empty_summary"],
        }
    summaries["summary_source"] = "deepseek"
    summaries["summary_warnings"] = []
    return summaries


_DOC_CARD_SUMMARY_LIMITS = {
    "description": 180,
    "method_summary": 220,
    "innovation_summary": 220,
    "limitation_summary": 220,
}


def _rule_doc_card_summaries(parsed) -> Dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        "description": _text_excerpt(_doc_description(parsed), _DOC_CARD_SUMMARY_LIMITS["description"]),
        "method_summary": _section_excerpt(
            parsed,
            ("method", "approach", "方法", "方法设计", "框架", "模型", "算法", "系统设计"),
            _DOC_CARD_SUMMARY_LIMITS["method_summary"],
        ),
        "innovation_summary": _section_excerpt(
            parsed,
            ("contribution", "innovation", "novel", "贡献", "创新", "提出", "研究内容"),
            _DOC_CARD_SUMMARY_LIMITS["innovation_summary"],
        ),
        "limitation_summary": _section_excerpt(
            parsed,
            ("limitation", "future", "局限", "不足", "未来", "结论"),
            _DOC_CARD_SUMMARY_LIMITS["limitation_summary"],
        ),
        "summary_source": "rule",
        "summary_warnings": [],
    }


def _doc_card_summary_prompt(parsed, fallback: Dict[str, object]) -> str:  # type: ignore[no-untyped-def]
    payload = {
        "instruction": "生成短字段：description、method_summary、innovation_summary、limitation_summary。每项不超过 80 个中文字或 50 个英文词。",
        "title": parsed.title,
        "abstract": _text_excerpt(str(parsed.metadata.get("abstract") or ""), 700),
        "keywords": [str(item) for item in (parsed.metadata.get("keywords") or [])][:12],
        "section_signals": _section_signals(parsed, limit=10),
        "fallback": {field: fallback.get(field, "") for field in _DOC_CARD_SUMMARY_LIMITS},
    }
    return json.dumps(payload, ensure_ascii=False)


def _section_signals(parsed, limit: int) -> List[Dict[str, str]]:  # type: ignore[no-untyped-def]
    signals: List[Dict[str, str]] = []
    current_heading = ""
    for block in parsed.blocks:
        heading = str(getattr(block, "heading", "") or "")
        text = str(getattr(block, "text", "") or "")
        kind = str(getattr(block, "kind", "") or "")
        if kind == "heading" and heading:
            current_heading = heading
            continue
        if not text.strip():
            continue
        if current_heading or kind in {"abstract", "paragraph"}:
            signals.append(
                {
                    "heading": current_heading or kind,
                    "text": _text_excerpt(text, 240),
                }
            )
        if len(signals) >= limit:
            break
    return signals


def _section_excerpt(parsed, heading_terms: Iterable[str], max_chars: int) -> str:  # type: ignore[no-untyped-def]
    terms = [term.lower() for term in heading_terms]
    current_heading = ""
    matched: List[str] = []
    for block in parsed.blocks:
        heading = str(getattr(block, "heading", "") or "")
        text = str(getattr(block, "text", "") or "")
        kind = str(getattr(block, "kind", "") or "")
        if kind == "heading" and heading:
            current_heading = heading
            continue
        haystack = f"{current_heading} {text}".lower()
        if text.strip() and any(term and term in haystack for term in terms):
            matched.append(text)
        if matched and len(" ".join(matched)) >= max_chars:
            break
    return _text_excerpt(" ".join(matched), max_chars)


def _short_summary(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = _text_excerpt(value, max_chars)
    banned_markers = ("```", "prompt", "原文：", "正文：")
    if any(marker.lower() in cleaned.lower() for marker in banned_markers):
        return ""
    return cleaned


def _text_excerpt(text: str, max_chars: int) -> str:
    compacted = " ".join(str(text).split())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[:max_chars].rstrip() + " ..."


def _metadata_score(parsed) -> float:  # type: ignore[no-untyped-def]
    checks = [
        bool(parsed.title),
        bool(parsed.metadata.get("authors")),
        bool(parsed.metadata.get("year")),
        bool(parsed.metadata.get("doi") or parsed.metadata.get("venue")),
        bool(parsed.metadata.get("abstract")),
    ]
    return round(sum(1 for item in checks if item) / len(checks), 2)


def _structure_score(section_count: int, paragraph_count: int, page_only_tree: bool) -> float:
    if page_only_tree:
        return 0.25 if paragraph_count else 0.0
    if section_count >= 4:
        return 1.0
    if section_count >= 2:
        return 0.75
    if paragraph_count:
        return 0.45
    return 0.0


def _reference_score(reference_count: int, parsed) -> float:  # type: ignore[no-untyped-def]
    if reference_count >= 10:
        return 1.0
    if reference_count >= 3:
        return 0.75
    if reference_count >= 1:
        return 0.5
    if (parsed.references or {}).get("status") == "extracted":
        return 0.45
    return 0.0


def _layout_score(layout_blocks: List[Dict[str, Any]], parsed) -> float:  # type: ignore[no-untyped-def]
    if not layout_blocks:
        return 0.0 if parsed.file_type == "pdf" else 0.45
    types = {str(block.get("type") or "") for block in layout_blocks}
    score = 0.35
    if "heading" in types or "abstract" in types:
        score += 0.2
    if "paragraph" in types:
        score += 0.15
    if "reference" in types:
        score += 0.1
    if "figure" in types or "table" in types:
        score += 0.1
    if any(block.get("bbox") for block in layout_blocks):
        score += 0.1
    return round(min(score, 1.0), 2)


def _caption_score(figures: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> float:
    items = [*figures, *tables]
    if not items:
        return 1.0
    return round(sum(1 for item in items if _has_caption(item)) / len(items), 2)


def _caption_link_rate(figures: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> float:
    items = [*figures, *tables]
    if not items:
        return 1.0
    linked = [
        item
        for item in items
        if (item.get("caption_id") or item.get("layout_block_id")) and _has_caption(item)
    ]
    return round(len(linked) / len(items), 2)


def _has_caption(item: Dict[str, Any]) -> bool:
    return bool(str(item.get("caption") or item.get("text") or "").strip())


def _quality_level(
    metadata_score: float,
    structure_score: float,
    reference_score: float,
    warnings: List[str],
) -> str:
    if "parse_failed" in warnings:
        return "failed"
    score = (metadata_score * 0.35) + (structure_score * 0.45) + (reference_score * 0.20)
    if score >= 0.78 and "page_only_tree" not in warnings and "missing_abstract" not in warnings:
        return "good"
    if score >= 0.45:
        return "usable"
    return "weak"


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
