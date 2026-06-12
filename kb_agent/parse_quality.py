from __future__ import annotations

from typing import Any, Dict, List

from .parser_artifacts import table_parse_score, table_warning_count


def build_parse_quality(
    doc_id: str,
    version_id: str,
    parsed: Any,
    nodes: List[Any],
    layout_blocks: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    table_content: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
) -> Dict[str, Any]:
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
    warnings = quality_warnings(parsed, section_count, reference_count, page_only_tree, missing_abstract)
    diagnostics = parsed.metadata.get("_parse_diagnostics") or {}
    metadata = metadata_score(parsed)
    structure = structure_score(section_count, paragraph_count, page_only_tree)
    reference = reference_score(reference_count, parsed)
    layout = layout_score(layout_blocks, parsed)
    caption = caption_score(figures, tables)
    caption_rate = caption_link_rate(figures, tables)
    current_table_parse_score = table_parse_score(table_content, tables)
    current_table_warning_count = table_warning_count(table_content)
    if parsed.file_type == "pdf" and layout < 0.45:
        warnings.append("weak_layout_blocks")
    if (figures or tables) and caption_rate < 0.5:
        warnings.append("low_caption_link_rate")
    if tables and not table_content:
        warnings.append("missing_table_content")
    if current_table_parse_score < 0.5 and tables:
        warnings.append("weak_table_parse")
    return {
        "schema": "parse_quality.v0",
        "doc_id": doc_id,
        "version_id": version_id,
        "quality_level": quality_level(metadata, structure, reference, warnings),
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
        "metadata_score": metadata,
        "structure_score": structure,
        "reference_score": reference,
        "layout_score": layout,
        "caption_score": caption,
        "noise_removed_count": int(parsed.metadata.get("noise_removed_count") or 0),
        "layout_block_count": len(layout_blocks),
        "caption_link_rate": caption_rate,
        "warning_count": len(warnings),
        "missing_abstract": missing_abstract,
        "page_only_tree": page_only_tree,
        "quality_warnings": warnings,
    }


def metadata_score(parsed: Any) -> float:
    checks = [
        bool(parsed.title),
        bool(parsed.metadata.get("authors")),
        bool(parsed.metadata.get("year")),
        bool(parsed.metadata.get("doi") or parsed.metadata.get("venue")),
        bool(parsed.metadata.get("abstract")),
    ]
    return round(sum(1 for item in checks if item) / len(checks), 2)


def structure_score(section_count: int, paragraph_count: int, page_only_tree: bool) -> float:
    if page_only_tree:
        return 0.25 if paragraph_count else 0.0
    if section_count >= 4:
        return 1.0
    if section_count >= 2:
        return 0.75
    if paragraph_count:
        return 0.45
    return 0.0


def reference_score(reference_count: int, parsed: Any) -> float:
    if reference_count >= 10:
        return 1.0
    if reference_count >= 3:
        return 0.75
    if reference_count >= 1:
        return 0.5
    if (parsed.references or {}).get("status") == "extracted":
        return 0.45
    return 0.0


def layout_score(layout_blocks: List[Dict[str, Any]], parsed: Any) -> float:
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


def caption_score(figures: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> float:
    items = [*figures, *tables]
    if not items:
        return 1.0
    return round(sum(1 for item in items if has_caption(item)) / len(items), 2)


def caption_link_rate(figures: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> float:
    items = [*figures, *tables]
    if not items:
        return 1.0
    linked = [
        item
        for item in items
        if (item.get("caption_id") or item.get("layout_block_id")) and has_caption(item)
    ]
    return round(len(linked) / len(items), 2)


def has_caption(item: Dict[str, Any]) -> bool:
    return bool(str(item.get("caption") or item.get("text") or "").strip())


def quality_level(
    metadata_score_value: float,
    structure_score_value: float,
    reference_score_value: float,
    warnings: List[str],
) -> str:
    if "parse_failed" in warnings:
        return "failed"
    score = (metadata_score_value * 0.35) + (structure_score_value * 0.45) + (reference_score_value * 0.20)
    if score >= 0.78 and "page_only_tree" not in warnings and "missing_abstract" not in warnings:
        return "good"
    if score >= 0.45:
        return "usable"
    return "weak"


def quality_warnings(
    parsed: Any,
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
