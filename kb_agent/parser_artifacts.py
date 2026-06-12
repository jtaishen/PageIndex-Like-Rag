from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .models import ParsedBlock
from .utils import compact_whitespace


def _ensure_layout_metadata(blocks: List[ParsedBlock], source_parser: str) -> None:
    counters = Counter()
    for index, block in enumerate(blocks, start=1):
        if not block.source_parser:
            block.source_parser = source_parser
        if not block.layout_block_id:
            block.layout_block_id = f"layout_{index:04d}"
        if block.kind in {"figure", "table"} and not block.caption_id:
            counters[block.kind] += 1
            prefix = "fig" if block.kind == "figure" else "table"
            block.caption_id = f"{prefix}_{counters[block.kind]:03d}"


def build_layout_blocks(blocks: List[ParsedBlock], source_parser: str = "") -> List[Dict[str, Any]]:
    current_section: List[str] = []
    layout_blocks: List[Dict[str, Any]] = []
    _ensure_layout_metadata(blocks, source_parser or "unknown")
    for block in blocks:
        text = compact_whitespace(block.heading or block.text)
        if block.kind == "heading" and text:
            level = max(1, block.level or 1)
            current_section = current_section[: level - 1]
            current_section.append(text)
        if not text:
            continue
        layout_blocks.append(
            {
                "schema": "layout_block.v1",
                "block_id": block.layout_block_id,
                "type": block.kind,
                "text": text,
                "page": block.page,
                "bbox": block.bbox,
                "section_path": list(current_section),
                "caption_id": block.caption_id,
                "confidence": round(float(block.confidence), 3),
                "source_parser": block.source_parser or source_parser or "unknown",
            }
        )
    return layout_blocks


def build_visual_items(layout_blocks: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for index, block in enumerate((item for item in layout_blocks if item.get("type") == kind), start=1):
        caption = compact_whitespace(str(block.get("text") or ""))
        if kind == "table" and not _is_table_line(caption):
            continue
        item_id = str(block.get("caption_id") or f"{kind}_{index:03d}")
        item: Dict[str, Any] = {
            "schema": f"{kind}.v1",
            "id": item_id,
            "caption_id": item_id,
            "layout_block_id": block.get("block_id"),
            "caption": caption,
            "text": caption,
            "page": block.get("page"),
            "bbox": block.get("bbox"),
            "section_path": block.get("section_path") or [],
            "confidence": block.get("confidence", 1.0),
            "source_parser": block.get("source_parser") or "",
        }
        if kind == "table":
            item["table_id"] = item_id
            item["row_count"] = 0
            item["column_count"] = 0
            item["cell_count"] = 0
            item["quality_warnings"] = ["table_content_not_extracted"]
        items.append(item)
    return items


def build_table_content(
    blocks: List[ParsedBlock],
    layout_blocks: List[Dict[str, Any]],
    *,
    raw_tables: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Build conservative table content artifacts from structured or line-based parser output."""
    items = _table_content_from_raw_tables(raw_tables or [], layout_blocks)
    existing_ids = {str(item.get("table_id") or "") for item in items}
    items.extend(
        item
        for item in _table_content_from_blocks(blocks, layout_blocks)
        if str(item.get("table_id") or "") not in existing_ids
    )
    existing_ids = {str(item.get("table_id") or "") for item in items}
    items.extend(
        item
        for item in _table_content_from_layout_blocks(layout_blocks)
        if str(item.get("table_id") or "") not in existing_ids
    )
    return items[:200]


def enhance_table_items(table_items: List[Dict[str, Any]], table_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(item.get("table_id") or ""): item for item in table_content if item.get("table_id")}
    enhanced = []
    for item in table_items:
        table_id = str(item.get("table_id") or item.get("id") or item.get("caption_id") or "")
        content = by_id.get(table_id)
        merged = dict(item)
        merged["table_id"] = table_id
        if content:
            merged["row_count"] = int(content.get("row_count") or 0)
            merged["column_count"] = int(content.get("column_count") or 0)
            merged["cell_count"] = int(content.get("cell_count") or 0)
            merged["source_parser"] = content.get("source_parser") or merged.get("source_parser") or ""
            merged["quality_warnings"] = list(content.get("quality_warnings") or [])
        else:
            merged.setdefault("row_count", 0)
            merged.setdefault("column_count", 0)
            merged.setdefault("cell_count", 0)
            merged.setdefault("quality_warnings", ["table_content_not_extracted"])
        enhanced.append(merged)
    return enhanced


def build_table_summaries(table_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for item in table_content:
        headers = [str(value) for value in item.get("headers") or [] if str(value).strip()]
        rows = item.get("rows") or []
        flat_values = [
            str(cell.get("text") or "")
            for row in rows
            if isinstance(row, dict)
            for cell in row.get("cells") or []
            if isinstance(cell, dict) and str(cell.get("text") or "").strip()
        ]
        metrics = _unique_preserve_order(
            [
                value
                for value in [*headers, *flat_values]
                if _looks_like_metric_cell(value)
            ]
        )[:12]
        methods = _unique_preserve_order(
            [
                value
                for value in flat_values
                if _looks_like_method_cell(value)
            ]
        )[:12]
        results = _unique_preserve_order(
            [
                value
                for value in flat_values
                if _looks_like_result_cell(value)
            ]
        )[:16]
        summaries.append(
            {
                "schema": "table_summary.v1",
                "table_id": item.get("table_id") or "",
                "caption": item.get("caption") or "",
                "page": item.get("page"),
                "headers": headers,
                "row_count": item.get("row_count", 0),
                "column_count": item.get("column_count", 0),
                "cell_count": item.get("cell_count", 0),
                "metrics": metrics,
                "methods": methods,
                "results": results,
                "source": item.get("source") or "",
                "source_parser": item.get("source_parser") or "",
                "quality_warnings": item.get("quality_warnings") or [],
            }
        )
    return summaries


def table_parse_score(table_content: List[Dict[str, Any]], table_items: Optional[List[Dict[str, Any]]] = None) -> float:
    if not table_items:
        return 1.0 if not table_content else 0.85
    if not table_content:
        return 0.0
    complete = sum(1 for item in table_content if int(item.get("row_count") or 0) > 0 and int(item.get("column_count") or 0) > 1)
    caption_count = max(1, len(table_items))
    score = min(1.0, complete / caption_count)
    if any(str(item.get("source") or "") == "docling_table" for item in table_content):
        score = min(1.0, score + 0.1)
    return round(score, 2)


def table_warning_count(table_content: List[Dict[str, Any]]) -> int:
    return sum(len(item.get("quality_warnings") or []) for item in table_content)


def _table_content_from_raw_tables(raw_tables: List[Any], layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for index, raw in enumerate(raw_tables, start=1):
        if not isinstance(raw, dict):
            continue
        rows = _rows_from_raw_table(raw)
        caption = compact_whitespace(str(raw.get("caption") or raw.get("text") or raw.get("name") or f"表 {index}"))
        block = _matching_table_caption(layout_blocks, caption, index)
        table_id = str(raw.get("table_id") or raw.get("id") or raw.get("caption_id") or (block or {}).get("caption_id") or f"table_{index:03d}")
        if not rows:
            continue
        result.append(
            _table_content_item(
                table_id=table_id,
                caption=caption,
                rows=rows,
                page=page_from_payload(raw) or (block or {}).get("page"),
                bbox=bbox_from_payload(raw) or (block or {}).get("bbox"),
                layout_block_id=str(raw.get("layout_block_id") or raw.get("block_id") or (block or {}).get("block_id") or ""),
                content_layout_block_ids=[],
                section_path=(block or {}).get("section_path") or [],
                source="docling_table",
                source_parser=str(raw.get("source_parser") or raw.get("source") or (block or {}).get("source_parser") or "docling"),
                confidence=confidence_from_payload(raw),
                warnings=[],
            )
        )
    return result


def _table_content_from_layout_blocks(layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for index, block in enumerate(layout_blocks):
        if block.get("type") != "table":
            continue
        caption = compact_whitespace(str(block.get("text") or ""))
        if not _is_table_line(caption):
            continue
        rows_text: List[str] = []
        content_block_ids: List[str] = []
        for follower in layout_blocks[index + 1 : index + 8]:
            follower_type = str(follower.get("type") or "")
            if follower_type in {"heading", "figure", "reference"}:
                break
            text = str(follower.get("text") or "")
            if follower_type == "table" and not _is_table_line(text):
                rows_text.append(text)
                if follower.get("block_id"):
                    content_block_ids.append(str(follower.get("block_id")))
                continue
            if follower_type == "paragraph":
                candidate_rows = _rows_from_text(text)
                if len(candidate_rows) >= 2:
                    rows_text.append(text)
                    if follower.get("block_id"):
                        content_block_ids.append(str(follower.get("block_id")))
                    continue
            if rows_text:
                break
        rows = _rows_from_text("\n".join(rows_text))
        warnings = []
        if not rows:
            warnings.append("table_content_not_detected")
        elif len(rows) < 2 or _max_columns(rows) < 2:
            warnings.append("weak_table_structure")
        if not rows:
            continue
        table_id = str(block.get("caption_id") or f"table_{len(result) + 1:03d}")
        result.append(
            _table_content_item(
                table_id=table_id,
                caption=caption,
                rows=rows,
                page=block.get("page"),
                bbox=block.get("bbox"),
                layout_block_id=str(block.get("block_id") or ""),
                content_layout_block_ids=content_block_ids,
                section_path=block.get("section_path") or [],
                source="table_rule",
                source_parser=str(block.get("source_parser") or ""),
                confidence=0.58 if warnings else 0.68,
                warnings=warnings,
            )
        )
    return result


def _table_content_from_blocks(blocks: List[ParsedBlock], layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    layout_by_id = {str(item.get("block_id") or ""): item for item in layout_blocks if item.get("block_id")}
    result = []
    for index, block in enumerate(blocks):
        if block.kind != "table":
            continue
        caption = compact_whitespace(block.text)
        if not _is_table_line(caption):
            continue
        rows_text: List[str] = []
        content_layout_block_ids: List[str] = []
        for follower in blocks[index + 1 : index + 8]:
            if follower.kind in {"heading", "figure", "reference"}:
                break
            if follower.kind == "table" and not _is_table_line(follower.text):
                rows_text.append(follower.text)
                if follower.layout_block_id:
                    content_layout_block_ids.append(follower.layout_block_id)
                continue
            if follower.kind == "paragraph" and len(_rows_from_text(follower.text)) >= 2:
                rows_text.append(follower.text)
                if follower.layout_block_id:
                    content_layout_block_ids.append(follower.layout_block_id)
                continue
            if rows_text:
                break
        rows = _rows_from_text("\n".join(rows_text))
        if not rows:
            continue
        warnings = []
        if len(rows) < 2 or _max_columns(rows) < 2:
            warnings.append("weak_table_structure")
        layout = layout_by_id.get(block.layout_block_id, {})
        table_id = str(block.caption_id or layout.get("caption_id") or f"table_{len(result) + 1:03d}")
        result.append(
            _table_content_item(
                table_id=table_id,
                caption=caption,
                rows=rows,
                page=block.page,
                bbox=block.bbox,
                layout_block_id=str(block.layout_block_id or layout.get("block_id") or ""),
                content_layout_block_ids=content_layout_block_ids,
                section_path=layout.get("section_path") or [],
                source="table_rule",
                source_parser=block.source_parser or str(layout.get("source_parser") or ""),
                confidence=0.58 if warnings else 0.68,
                warnings=warnings,
            )
        )
    return result


def _table_content_item(
    *,
    table_id: str,
    caption: str,
    rows: List[List[str]],
    page: Any,
    bbox: Any,
    layout_block_id: str,
    content_layout_block_ids: List[str],
    section_path: List[str],
    source: str,
    source_parser: str,
    confidence: float,
    warnings: List[str],
) -> Dict[str, Any]:
    headers, data_rows = _split_header_rows(rows)
    column_count = _max_columns(rows)
    row_payload = [
        {
            "row_index": row_index,
            "cells": [
                {"column_index": column_index, "text": compact_whitespace(cell)}
                for column_index, cell in enumerate(row)
            ],
        }
        for row_index, row in enumerate(data_rows)
    ]
    cell_count = sum(len(row) for row in rows)
    return {
        "schema": "table_content.v1",
        "table_id": table_id,
        "caption": caption,
        "page": page,
        "page_range": [page, page] if page else [None, None],
        "bbox": bbox,
        "layout_block_id": layout_block_id,
        "content_layout_block_ids": content_layout_block_ids,
        "section_path": section_path,
        "headers": headers,
        "rows": row_payload,
        "row_count": len(data_rows),
        "column_count": column_count,
        "cell_count": cell_count,
        "source": source,
        "source_parser": source_parser,
        "confidence": round(float(confidence), 3),
        "quality_warnings": _unique_preserve_order(warnings),
    }


def _rows_from_raw_table(raw: Dict[str, Any]) -> List[List[str]]:
    for key in ("rows", "data", "grid"):
        rows = _rows_from_matrix(raw.get(key))
        if rows:
            return rows
    data = raw.get("data")
    if isinstance(data, dict):
        for key in ("rows", "grid", "table_data"):
            rows = _rows_from_matrix(data.get(key))
            if rows:
                return rows
    rows = _rows_from_cells(raw.get("cells") or raw.get("table_cells"))
    if rows:
        return rows
    text = str(raw.get("text") or raw.get("markdown") or "")
    return _rows_from_text(text)


def _rows_from_matrix(value: Any) -> List[List[str]]:
    if not isinstance(value, list):
        return []
    rows: List[List[str]] = []
    for row in value:
        if isinstance(row, dict):
            cells = row.get("cells") or row.get("values") or row.get("row")
            parsed = _row_from_value(cells)
        else:
            parsed = _row_from_value(row)
        if parsed:
            rows.append(parsed)
    return rows


def _row_from_value(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [compact_whitespace(str(cell.get("text") if isinstance(cell, dict) else cell)) for cell in value if compact_whitespace(str(cell.get("text") if isinstance(cell, dict) else cell))]
    if isinstance(value, dict):
        return [compact_whitespace(str(cell)) for _, cell in sorted(value.items()) if compact_whitespace(str(cell))]
    if isinstance(value, str):
        return _split_table_row(value)
    return []


def _rows_from_cells(value: Any) -> List[List[str]]:
    if not isinstance(value, list):
        return []
    grid: Dict[int, Dict[int, str]] = {}
    for cell in value:
        if not isinstance(cell, dict):
            continue
        row_index = _int_from_any(cell.get("row_index"), cell.get("row"), cell.get("row_idx"), default=0)
        column_index = _int_from_any(cell.get("column_index"), cell.get("col"), cell.get("column"), cell.get("col_idx"), default=0)
        text = compact_whitespace(str(cell.get("text") or cell.get("content") or ""))
        if text:
            grid.setdefault(row_index, {})[column_index] = text
    rows = []
    for row_index in sorted(grid):
        row = [grid[row_index][column_index] for column_index in sorted(grid[row_index])]
        if row:
            rows.append(row)
    return rows


def _rows_from_text(text: str) -> List[List[str]]:
    rows = []
    for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = compact_whitespace(raw_line.strip().strip("|"))
        if not line or re.fullmatch(r"[:：|\-\s]+", line):
            continue
        row = _split_table_row(line)
        if row:
            rows.append(row)
    if len(rows) < 2 and "\n" not in str(text or ""):
        rows = _rows_from_flat_text(text)
    return rows


def _rows_from_flat_text(text: str) -> List[List[str]]:
    tokens = [token for token in compact_whitespace(text).split(" ") if token]
    if len(tokens) < 6:
        return []
    numeric_positions = [index for index, token in enumerate(tokens) if _looks_like_result_cell(token)]
    if len(numeric_positions) < 2:
        return []
    column_count = max(2, min(6, numeric_positions[0] + 1))
    rows = [tokens[index : index + column_count] for index in range(0, len(tokens), column_count)]
    return [row for row in rows if len(row) >= 2]


def _split_table_row(line: str) -> List[str]:
    if "\t" in line:
        parts = line.split("\t")
    elif "|" in line:
        parts = line.split("|")
    elif re.search(r"\s{2,}", line):
        parts = re.split(r"\s{2,}", line)
    else:
        parts = line.split()
    cells = [compact_whitespace(part) for part in parts if compact_whitespace(part)]
    if len(cells) < 2:
        return []
    if len(cells) >= 3:
        return cells
    if any(_looks_like_metric_cell(cell) or _looks_like_result_cell(cell) for cell in cells):
        return cells
    return []


def _split_header_rows(rows: List[List[str]]) -> tuple[List[str], List[List[str]]]:
    if len(rows) >= 2 and not any(_looks_like_result_cell(cell) for cell in rows[0]):
        return rows[0], rows[1:]
    return [], rows


def _max_columns(rows: List[List[str]]) -> int:
    return max((len(row) for row in rows), default=0)


def _matching_table_caption(layout_blocks: List[Dict[str, Any]], caption: str, index: int) -> Optional[Dict[str, Any]]:
    candidates = [block for block in layout_blocks if block.get("type") == "table" and _is_table_line(str(block.get("text") or ""))]
    normalized_caption = re.sub(r"\s+", "", caption)
    for block in candidates:
        block_text = re.sub(r"\s+", "", str(block.get("text") or ""))
        if normalized_caption and (normalized_caption in block_text or block_text in normalized_caption):
            return block
    if 0 <= index - 1 < len(candidates):
        return candidates[index - 1]
    return None


def _looks_like_metric_cell(value: str) -> bool:
    text = compact_whitespace(value)
    return bool(re.search(r"(率|时间|开销|准确|精度|召回|F1|AUC|BLEU|ROUGE|指标|性能|鲁棒|延迟|吞吐)", text, re.IGNORECASE))


def _looks_like_method_cell(value: str) -> bool:
    text = compact_whitespace(value)
    return bool(re.search(r"(方法|算法|模型|框架|基线|baseline|ours|本文)", text, re.IGNORECASE))


def _looks_like_result_cell(value: str) -> bool:
    return bool(re.search(r"[-+]?\d+(?:\.\d+)?\s*(?:%|ms|s|秒|分|x|倍)?", compact_whitespace(value), re.IGNORECASE))


def _int_from_any(*values: Any, default: int = 0) -> int:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _unique_preserve_order(values: List[Any]) -> List[Any]:
    result = []
    seen = set()
    for value in values:
        marker = str(value)
        if not marker or marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def page_from_payload(payload: Dict[str, Any]) -> Optional[int]:
    for key in ("page", "page_no", "page_number"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    prov = payload.get("prov")
    if isinstance(prov, list) and prov and isinstance(prov[0], dict):
        return page_from_payload(prov[0])
    return None


def bbox_from_payload(payload: Dict[str, Any]) -> Optional[List[float]]:
    for key in ("bbox", "box", "bounding_box"):
        bbox = payload.get(key)
        parsed = _parse_bbox(bbox)
        if parsed:
            return parsed
    prov = payload.get("prov")
    if isinstance(prov, list) and prov and isinstance(prov[0], dict):
        return bbox_from_payload(prov[0])
    return None


def _parse_bbox(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(item) for item in value[:4]]
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        keys = ("x0", "y0", "x1", "y1")
        if all(key in value for key in keys):
            try:
                return [float(value[key]) for key in keys]
            except (TypeError, ValueError):
                return None
        keys = ("l", "t", "r", "b")
        if all(key in value for key in keys):
            try:
                return [float(value[key]) for key in keys]
            except (TypeError, ValueError):
                return None
        if all(key in value for key in ("left", "top", "right", "bottom")):
            try:
                return [float(value[key]) for key in ("left", "top", "right", "bottom")]
            except (TypeError, ValueError):
                return None
    return None


def confidence_from_payload(payload: Dict[str, Any]) -> float:
    for key in ("confidence", "score", "probability"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return 0.95


_page_from_payload = page_from_payload
_bbox_from_payload = bbox_from_payload
_confidence_from_payload = confidence_from_payload


def _is_table_line(text: str) -> bool:
    return bool(re.match(r"^(表\s*\d+|Table\s+\d+)", compact_whitespace(text), re.IGNORECASE))


def _looks_like_table_data_line(text: str) -> bool:
    line = compact_whitespace(text)
    if not line or len(line) > 180:
        return False
    row = _split_table_row(line)
    return len(row) >= 2


def build_reference_sections(
    layout_blocks: List[Dict[str, Any]],
    references: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    reference_blocks = [block for block in layout_blocks if block.get("type") == "reference"]
    pages = [int(block["page"]) for block in reference_blocks if isinstance(block.get("page"), int)]
    if reference_blocks:
        return [
            {
                "schema": "reference_section.v1",
                "section_id": "references",
                "source": _dominant_source(reference_blocks),
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "block_ids": [str(block.get("block_id") or "") for block in reference_blocks if block.get("block_id")],
                "item_count": len(reference_blocks),
                "references_count": len((references or {}).get("references") or []),
            }
        ]
    ref_items = (references or {}).get("references") or []
    if ref_items:
        return [
            {
                "schema": "reference_section.v1",
                "section_id": "references",
                "source": str((references or {}).get("source") or "references"),
                "page_start": None,
                "page_end": None,
                "block_ids": [],
                "item_count": len(ref_items),
                "references_count": len(ref_items),
            }
        ]
    return []


def _dominant_source(blocks: List[Dict[str, Any]]) -> str:
    sources = [str(block.get("source_parser") or "") for block in blocks if block.get("source_parser")]
    if not sources:
        return ""
    return Counter(sources).most_common(1)[0][0]
