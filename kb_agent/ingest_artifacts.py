from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parser_artifacts import (
    build_layout_blocks,
    build_reference_sections,
    build_table_content,
    build_table_summaries,
    build_visual_items,
    enhance_table_items,
    table_parse_score,
    table_warning_count,
)


def build_ingest_artifacts(
    doc_id: str,
    version_id: str,
    base: Path,
    source_path: Path,
    file_hash: str,
    parsed: Any,
    nodes: List[Any],
) -> Dict[str, Any]:
    components = artifact_components(parsed)
    layout_blocks = components["layout_blocks"]
    tables = components["tables"]
    table_content = components["table_content"]
    table_summaries = components["table_summaries"]
    figures = components["figures"]
    reference_sections = components["reference_sections"]
    return {
        "structured": structured_payload(parsed, components),
        "metadata": parsed.metadata,
        "references": parsed.references,
        "layout_blocks": layout_artifact(doc_id, version_id, layout_blocks),
        "tables": visual_artifact("tables", doc_id, version_id, tables),
        "table_content": table_content_artifact(doc_id, version_id, table_content),
        "table_summaries": table_summaries_artifact(doc_id, version_id, table_summaries),
        "figures": visual_artifact("figures", doc_id, version_id, figures),
        "reference_sections": reference_sections_artifact(doc_id, version_id, reference_sections),
        "parse_report": parse_report(
            doc_id,
            version_id,
            base,
            source_path,
            file_hash,
            parsed,
            nodes,
            components,
        ),
        "components": components,
    }


def artifact_components(parsed: Any) -> Dict[str, List[Dict[str, Any]]]:
    layout_blocks = layout_blocks_from(parsed)
    tables = tables_from(parsed, layout_blocks=layout_blocks)
    table_content = table_content_from(parsed, tables, layout_blocks)
    tables = enhance_table_items(tables, table_content)
    return {
        "layout_blocks": layout_blocks,
        "tables": tables,
        "table_content": table_content,
        "table_summaries": table_summaries_from(parsed, table_content),
        "figures": figures_from(parsed, layout_blocks=layout_blocks),
        "reference_sections": reference_sections_from(parsed, layout_blocks=layout_blocks),
    }


def structured_payload(parsed: Any, components: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    layout_blocks = components["layout_blocks"]
    tables = components["tables"]
    table_content = components["table_content"]
    table_summaries = components["table_summaries"]
    figures = components["figures"]
    reference_sections = components["reference_sections"]
    payload = dict(parsed.structured or {})
    payload.setdefault("schema", "structured.v0")
    payload["layout_schema"] = "layout_blocks.v1"
    payload["layout_blocks"] = layout_blocks
    payload["layout_blocks_count"] = len(layout_blocks)
    payload["tables"] = tables
    payload["table_content"] = table_content
    payload["table_summaries"] = table_summaries
    payload["figures"] = figures
    payload["table_count"] = len(tables)
    payload["table_content_count"] = len(table_content)
    payload["table_parse_score"] = table_parse_score(table_content, tables)
    payload["table_warning_count"] = table_warning_count(table_content)
    payload["figure_count"] = len(figures)
    payload["reference_sections"] = reference_sections
    payload["reference_section_count"] = len(reference_sections)
    return payload


def parse_report(
    doc_id: str,
    version_id: str,
    base: Path,
    source_path: Path,
    file_hash: str,
    parsed: Any,
    nodes: List[Any],
    components: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    diagnostics = parsed.metadata.get("_parse_diagnostics") or {}
    layout_blocks = components["layout_blocks"]
    tables = components["tables"]
    table_content = components["table_content"]
    figures = components["figures"]
    reference_sections = components["reference_sections"]
    return {
        "schema": "parse_report.v0",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": parsed.title,
        "file_type": parsed.file_type,
        "source_path": str(source_path),
        "file_hash": file_hash,
        "parser_name": parsed.parser_name,
        "parser_version": parsed.parser_version,
        "requested_pdf_parser": diagnostics.get("requested_pdf_parser"),
        "parser_chain": diagnostics.get("parser_chain", [parsed.parser_name]),
        "fallback_used": diagnostics.get("fallback_used", False),
        "external_parser_errors": diagnostics.get("external_parser_errors", []),
        "adapter_statuses": diagnostics.get("adapter_statuses", {}),
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
    }


def build_failure_parse_report(
    doc_id: str,
    version_id: str,
    base: Path,
    source_path: Path,
    file_hash: str,
    parser_name: str,
    parser_version: str,
    error: str,
) -> Dict[str, Any]:
    return {
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
    }


def layout_blocks_from(parsed: Any) -> List[Dict[str, Any]]:
    blocks = parsed.structured.get("layout_blocks") if isinstance(parsed.structured, dict) else []
    if isinstance(blocks, list):
        cleaned = [item for item in blocks if isinstance(item, dict)]
        if cleaned:
            return cleaned
    return build_layout_blocks(parsed.blocks, parsed.parser_name or parsed.file_type)


def tables_from(parsed: Any, *, layout_blocks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    tables = parsed.structured.get("tables") if isinstance(parsed.structured, dict) else []
    if isinstance(tables, list):
        cleaned = [item for item in tables if isinstance(item, dict)]
        if cleaned and all(item.get("schema") == "table.v1" for item in cleaned):
            return cleaned
    return build_visual_items(layout_blocks or layout_blocks_from(parsed), "table")


def table_content_from(
    parsed: Any,
    tables: List[Dict[str, Any]],
    layout_blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    del tables
    content = parsed.structured.get("table_content") if isinstance(parsed.structured, dict) else []
    if isinstance(content, list):
        cleaned = [item for item in content if isinstance(item, dict) and item.get("schema") == "table_content.v1"]
        if cleaned:
            return cleaned
    raw_tables = parsed.structured.get("tables") if isinstance(parsed.structured, dict) else []
    raw_tables = raw_tables if isinstance(raw_tables, list) else []
    return build_table_content(parsed.blocks, layout_blocks, raw_tables=raw_tables)


def table_summaries_from(parsed: Any, table_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = parsed.structured.get("table_summaries") if isinstance(parsed.structured, dict) else []
    if isinstance(summaries, list):
        cleaned = [item for item in summaries if isinstance(item, dict) and item.get("schema") == "table_summary.v1"]
        if cleaned:
            return cleaned
    return build_table_summaries(table_content)


def figures_from(parsed: Any, *, layout_blocks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    figures = parsed.structured.get("figures") if isinstance(parsed.structured, dict) else []
    if isinstance(figures, list):
        cleaned = [item for item in figures if isinstance(item, dict)]
        if cleaned and all(item.get("schema") == "figure.v1" for item in cleaned):
            return cleaned
    return build_visual_items(layout_blocks or layout_blocks_from(parsed), "figure")


def reference_sections_from(parsed: Any, *, layout_blocks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    sections = parsed.structured.get("reference_sections") if isinstance(parsed.structured, dict) else []
    if isinstance(sections, list):
        cleaned = [item for item in sections if isinstance(item, dict)]
        if cleaned:
            return cleaned
    return build_reference_sections(layout_blocks or layout_blocks_from(parsed), parsed.references)


def layout_artifact(doc_id: str, version_id: str, layout_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def visual_artifact(kind: str, doc_id: str, version_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema": f"{kind}.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(items),
        kind: items,
    }


def table_content_artifact(doc_id: str, version_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema": "table_content.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(items),
        "table_content": items,
    }


def table_summaries_artifact(doc_id: str, version_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema": "table_summaries.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "count": len(items),
        "table_summaries": items,
    }


def reference_sections_artifact(
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
