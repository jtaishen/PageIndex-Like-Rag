from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .utils import compact_whitespace, excerpt as _excerpt, stable_id, unique_strings as _unique_strings


MAX_UNIT_EXCERPT_CHARS = 360
MAX_UNIT_SUMMARY_CHARS = 180


def evidence_units_from_artifacts(
    doc_id: str,
    version_id: str,
    nodes: List[Dict[str, Any]],
    table_summaries: Any,
    figures: Any,
    reference_sections: Any,
    citation_map: Any,
) -> List[Dict[str, Any]]:
    return _dedupe_evidence_units(
        [
            *_evidence_units_from_nodes(doc_id, version_id, nodes),
            *_evidence_units_from_table_summaries(doc_id, version_id, table_summaries),
            *_evidence_units_from_figures(doc_id, version_id, figures),
            *_evidence_units_from_reference_sections(doc_id, version_id, reference_sections),
            *_evidence_units_from_citation_map(doc_id, version_id, citation_map),
        ]
    )


def unit_by_node_id(units: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for unit in units:
        node_id = str(unit.get("node_id") or "")
        if node_id:
            result.setdefault(node_id, []).append(unit)
    return result


def unit_by_id(units: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(unit.get("unit_id") or ""): unit for unit in units if isinstance(unit, dict) and unit.get("unit_id")}


def unit_by_source_id(units: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for unit in units:
        source_id = str(unit.get("source_id") or "")
        if source_id:
            result.setdefault(source_id, []).append(unit)
    return result


def unit_ids_for_node(node_id: str, unit_by_node: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    if not node_id:
        return []
    return [str(unit.get("unit_id") or "") for unit in unit_by_node.get(node_id, []) if unit.get("unit_id")]


def unit_ids_for_source(source_id: str, unit_by_source_id: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    if not source_id:
        return []
    return [str(unit.get("unit_id") or "") for unit in unit_by_source_id.get(source_id, []) if unit.get("unit_id")]


def extract_evidence_refs(claim: Dict[str, Any]) -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {
        "node_ids": [],
        "source_node_ids": [],
        "unit_ids": [],
        "source_ids": [],
        "ref_ids": [],
        "raw_refs": [],
    }

    def add(name: str, value: Any) -> None:
        text = str(value or "").strip()
        if text:
            refs[name].append(text)

    add("node_ids", claim.get("node_id"))
    add("source_node_ids", claim.get("source_node_id"))
    add("node_ids", claim.get("evidence_node_id"))
    add("unit_ids", claim.get("unit_id"))
    add("source_ids", claim.get("source_id"))
    add("ref_ids", claim.get("ref_id"))
    add("source_ids", claim.get("evidence_id"))

    def visit_evidence(evidence: Any) -> None:
        if isinstance(evidence, dict):
            add("node_ids", evidence.get("node_id"))
            add("source_node_ids", evidence.get("source_node_id"))
            add("unit_ids", evidence.get("unit_id"))
            add("ref_ids", evidence.get("ref_id") or evidence.get("reference_id"))
            add("source_ids", evidence.get("evidence_id"))
            add("source_ids", evidence.get("source_id"))
        elif isinstance(evidence, list):
            for item in evidence:
                visit_evidence(item)
        elif isinstance(evidence, str):
            add("raw_refs", evidence)

    if "evidence" in claim:
        visit_evidence(claim.get("evidence"))
    return {key: _unique_strings(value) for key, value in refs.items()}


def evidence_unit_ids_for_claim(
    claim: Dict[str, Any],
    units_by_node: Dict[str, List[Dict[str, Any]]],
    units_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    units_by_source_id: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> tuple[List[str], List[str]]:
    refs = extract_evidence_refs(claim)
    units_by_id = units_by_id or {}
    units_by_source_id = units_by_source_id or {}
    resolved: List[str] = []
    warnings: List[str] = []
    for unit_id in refs["unit_ids"]:
        if unit_id in units_by_id:
            resolved.append(unit_id)
        else:
            warnings.append("unresolved_evidence_ref")
    for node_id in [*refs["node_ids"], *refs["source_node_ids"]]:
        resolved.extend(unit_ids_for_node(node_id, units_by_node))
    for source_id in [*refs["source_ids"], *refs["ref_ids"]]:
        resolved.extend(unit_ids_for_source(source_id, units_by_source_id))
    if refs["raw_refs"]:
        warnings.append("unresolved_evidence_ref")
    if "evidence" in claim and not resolved:
        warnings.append("unresolved_evidence_ref")
    return _unique_strings(resolved), _unique_strings(warnings)


def source_artifact_ids(doc_id: str, artifact_lookup) -> set[str]:  # type: ignore[no-untyped-def]
    source_ids: set[str] = set()
    table_summaries = artifact_lookup(doc_id, "table_summaries.json", {})
    for index, table in enumerate((table_summaries or {}).get("table_summaries") or []):
        if isinstance(table, dict):
            source_ids.add(str(table.get("table_id") or table.get("id") or stable_id("table", doc_id, index, length=12)))
    figures = artifact_lookup(doc_id, "figures.json", {})
    for index, figure in enumerate((figures or {}).get("figures") or []):
        if isinstance(figure, dict):
            source_ids.add(str(figure.get("id") or figure.get("figure_id") or figure.get("caption_id") or stable_id("figure", doc_id, index, length=12)))
    reference_sections = artifact_lookup(doc_id, "reference_sections.json", {})
    for index, section in enumerate((reference_sections or {}).get("reference_sections") or []):
        if isinstance(section, dict):
            source_ids.add(str(section.get("section_id") or stable_id("reference_section", doc_id, index, length=12)))
    citation_map = artifact_lookup(doc_id, "citation_map.json", {})
    for index, ref in enumerate((citation_map or {}).get("references") or []):
        if isinstance(ref, dict):
            source_ids.add(str(ref.get("ref_id") or ref.get("reference_id") or ref.get("id") or stable_id("ref", doc_id, index, length=12)))
    citation_items = []
    for name in ("in_text_citations", "relations"):
        for item in (citation_map or {}).get(name) or []:
            if isinstance(item, dict):
                citation_items.append(item)
    for index, citation in enumerate(citation_items):
        ref_id = str(citation.get("ref_id") or citation.get("reference_id") or citation.get("id") or "")
        source_ids.add(str(citation.get("citation_id") or citation.get("source_id") or ref_id or stable_id("citation", doc_id, ref_id, citation.get("node_id"), index, length=12)))
        if ref_id:
            source_ids.add(ref_id)
    return {item for item in source_ids if item}


def _evidence_units_from_nodes(doc_id: str, version_id: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units = []
    seen = set()
    for node in sorted((item for item in nodes if isinstance(item, dict)), key=lambda item: int(item.get("order_index") or 0)):
        node_id = str(node.get("node_id") or "")
        if not node_id or node_id in seen:
            continue
        kind = str(node.get("kind") or node.get("type") or "paragraph")
        if kind == "document":
            continue
        text = compact_whitespace(str(node.get("text") or node.get("summary") or node.get("heading") or ""))
        if not text:
            continue
        seen.add(node_id)
        unit_type = _unit_type(kind, str(node.get("node_path") or ""), str(node.get("heading") or ""))
        source_kind = _source_kind(unit_type, str(node.get("node_path") or ""), str(node.get("heading") or ""))
        warnings = []
        if not node.get("text"):
            warnings.append("summary_only_unit")
        confidence = 0.78 if node.get("text") else 0.68
        if source_kind in {"table", "figure", "reference"}:
            confidence = max(0.62, confidence - 0.05)
        units.append(
            {
                "unit_id": stable_id("eu", version_id, node_id, unit_type, length=14),
                "doc_id": doc_id,
                "version_id": version_id,
                "node_id": node_id,
                "source_id": node_id,
                "unit_type": unit_type,
                "node_path": str(node.get("node_path") or ""),
                "page_range": _page_range(node),
                "source_kind": source_kind,
                "text_excerpt": _excerpt(text, MAX_UNIT_EXCERPT_CHARS),
                "summary": _excerpt(str(node.get("summary") or text), MAX_UNIT_SUMMARY_CHARS),
                "keywords": _short_keywords(node.get("keywords") or [], text),
                "confidence": round(confidence, 3),
                "warnings": warnings,
            }
        )
    return units


def _evidence_units_from_table_summaries(doc_id: str, version_id: str, payload: Any) -> List[Dict[str, Any]]:
    units = []
    for index, table in enumerate((payload or {}).get("table_summaries") or []):
        if not isinstance(table, dict):
            continue
        source_id = str(table.get("table_id") or table.get("id") or stable_id("table", doc_id, index, length=12))
        text = compact_whitespace(
            " ".join(
                [
                    str(table.get("caption") or ""),
                    str(table.get("summary") or ""),
                    " ".join(str(item) for item in table.get("headers") or []),
                    " ".join(str(item) for item in table.get("methods") or []),
                    " ".join(str(item) for item in table.get("results") or []),
                    " ".join(str(item) for item in table.get("metrics") or []),
                ]
            )
        )
        if text:
            units.append(
                _artifact_evidence_unit(
                    doc_id,
                    version_id,
                    source_kind="table",
                    unit_type="table",
                    source_id=source_id,
                    node_id=str(table.get("node_id") or table.get("source_node_id") or ""),
                    node_path=str(table.get("node_path") or table.get("section_path") or ""),
                    page_range=_artifact_page_range(table),
                    text=text,
                    summary=str(table.get("summary") or table.get("caption") or text),
                    keywords=table.get("headers") or [],
                    confidence=_confidence(table.get("confidence"), 0.66),
                    extra_warnings=list(table.get("quality_warnings") or []),
                )
            )
    return units


def _evidence_units_from_figures(doc_id: str, version_id: str, payload: Any) -> List[Dict[str, Any]]:
    units = []
    for index, figure in enumerate((payload or {}).get("figures") or []):
        if not isinstance(figure, dict):
            continue
        source_id = str(figure.get("id") or figure.get("figure_id") or figure.get("caption_id") or stable_id("figure", doc_id, index, length=12))
        text = compact_whitespace(f"{figure.get('caption') or ''} {figure.get('text') or ''}")
        if text:
            units.append(
                _artifact_evidence_unit(
                    doc_id,
                    version_id,
                    source_kind="figure",
                    unit_type="figure",
                    source_id=source_id,
                    node_id=str(figure.get("node_id") or figure.get("source_node_id") or ""),
                    node_path=" > ".join(str(item) for item in figure.get("section_path") or []) or str(figure.get("node_path") or ""),
                    page_range=_artifact_page_range(figure),
                    text=text,
                    summary=str(figure.get("caption") or figure.get("text") or text),
                    keywords=[],
                    confidence=_confidence(figure.get("confidence"), 0.62),
                    extra_warnings=[],
                )
            )
    return units


def _evidence_units_from_reference_sections(doc_id: str, version_id: str, payload: Any) -> List[Dict[str, Any]]:
    units = []
    for index, section in enumerate((payload or {}).get("reference_sections") or []):
        if not isinstance(section, dict):
            continue
        source_id = str(section.get("section_id") or stable_id("reference_section", doc_id, index, length=12))
        count = int(section.get("references_count") or section.get("item_count") or 0)
        text = compact_whitespace(str(section.get("text") or section.get("raw") or f"参考文献区段包含 {count} 条参考文献。"))
        units.append(
            _artifact_evidence_unit(
                doc_id,
                version_id,
                source_kind="reference",
                unit_type="reference",
                source_id=source_id,
                node_id=str(section.get("node_id") or section.get("source_node_id") or ""),
                node_path=str(section.get("node_path") or "参考文献"),
                page_range=_artifact_page_range(section),
                text=text,
                summary=f"参考文献区段，条目数 {count}。",
                keywords=["参考文献"],
                confidence=_confidence(section.get("confidence"), 0.58),
                extra_warnings=[],
            )
        )
    return units


def _evidence_units_from_citation_map(doc_id: str, version_id: str, payload: Any) -> List[Dict[str, Any]]:
    units = []
    for index, ref in enumerate((payload or {}).get("references") or []):
        if not isinstance(ref, dict):
            continue
        ref_id = str(ref.get("ref_id") or ref.get("reference_id") or ref.get("id") or stable_id("ref", doc_id, index, length=12))
        text = compact_whitespace(str(ref.get("raw") or ref.get("title") or ref_id))
        if text:
            units.append(
                _artifact_evidence_unit(
                    doc_id,
                    version_id,
                    source_kind="reference",
                    unit_type="reference",
                    source_id=ref_id,
                    node_id=str(ref.get("node_id") or ""),
                    node_path=str(ref.get("node_path") or "参考文献"),
                    page_range=_artifact_page_range(ref),
                    text=text,
                    summary=str(ref.get("title") or ref.get("raw") or ref_id),
                    keywords=[ref_id],
                    confidence=_confidence(ref.get("confidence"), 0.56),
                    extra_warnings=[],
                )
            )
    citation_items = []
    for name in ("in_text_citations", "relations"):
        for item in (payload or {}).get(name) or []:
            if isinstance(item, dict):
                citation_items.append(item)
    for index, citation in enumerate(citation_items):
        ref_id = str(citation.get("ref_id") or citation.get("reference_id") or citation.get("id") or "")
        source_id = str(citation.get("citation_id") or citation.get("source_id") or ref_id or stable_id("citation", doc_id, ref_id, citation.get("node_id"), index, length=12))
        text = compact_whitespace(str(citation.get("context") or citation.get("raw") or ref_id))
        if text:
            units.append(
                _artifact_evidence_unit(
                    doc_id,
                    version_id,
                    source_kind="citation",
                    unit_type="citation",
                    source_id=source_id,
                    node_id=str(citation.get("node_id") or citation.get("source_node_id") or ""),
                    node_path=str(citation.get("node_path") or ""),
                    page_range=_artifact_page_range(citation),
                    text=text,
                    summary=f"文内引用 {ref_id}：{_excerpt(text, 120)}" if ref_id else _excerpt(text, 140),
                    keywords=[ref_id] if ref_id else [],
                    confidence=_confidence(citation.get("confidence"), 0.6),
                    extra_warnings=[],
                )
            )
    return units


def _artifact_evidence_unit(
    doc_id: str,
    version_id: str,
    *,
    source_kind: str,
    unit_type: str,
    source_id: str,
    node_id: str,
    node_path: str,
    page_range: List[Optional[int]],
    text: str,
    summary: str,
    keywords: Any,
    confidence: float,
    extra_warnings: List[str],
) -> Dict[str, Any]:
    warnings = list(extra_warnings or [])
    if not node_id:
        warnings.append("source_without_node")
    return {
        "unit_id": stable_id("eu", version_id, source_kind, source_id or node_id, unit_type, length=14),
        "doc_id": doc_id,
        "version_id": version_id,
        "node_id": node_id,
        "source_id": source_id,
        "unit_type": unit_type,
        "source_kind": source_kind,
        "node_path": node_path,
        "page_range": page_range,
        "text_excerpt": _excerpt(text, MAX_UNIT_EXCERPT_CHARS),
        "summary": _excerpt(summary or text, MAX_UNIT_SUMMARY_CHARS),
        "keywords": _short_keywords(keywords, text),
        "confidence": round(confidence, 3),
        "warnings": _unique_strings(warnings),
    }


def _artifact_page_range(item: Dict[str, Any]) -> List[Optional[int]]:
    if item.get("page_range"):
        page_range = item.get("page_range")
        if isinstance(page_range, list) and len(page_range) >= 2:
            return [page_range[0], page_range[1]]
    page = item.get("page")
    if page is not None:
        return [page, page]
    return [item.get("page_start"), item.get("page_end")]


def _dedupe_evidence_units(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for unit in units:
        source_id = str(unit.get("source_id") or "")
        node_id = str(unit.get("node_id") or "")
        source_key = (str(unit.get("source_kind") or ""), str(unit.get("source_id") or ""), str(unit.get("node_id") or ""))
        text_key = (str(unit.get("source_kind") or ""), compact_whitespace(str(unit.get("text_excerpt") or "")))
        key = source_key if source_id or node_id else text_key
        if key in seen:
            continue
        seen.add(key)
        result.append(unit)
    return result


def _unit_type(kind: str, node_path: str, heading: str) -> str:
    raw = f"{kind} {node_path} {heading}".lower()
    if kind in {"abstract", "keywords", "reference", "figure", "table", "paragraph", "section", "subsection"}:
        return kind
    if any(token in raw for token in ("reference", "参考文献", "引用")):
        return "reference"
    if any(token in raw for token in ("figure", "fig.", "图 ")):
        return "figure"
    if any(token in raw for token in ("table", "表 ")):
        return "table"
    if int(_safe_int_from_level(raw)) >= 2:
        return "subsection"
    if kind in {"page"}:
        return "paragraph"
    return "section" if kind == "section" else "paragraph"


def _safe_int_from_level(text: str) -> int:
    match = re.search(r"\blevel[:=](\d+)", text)
    return int(match.group(1)) if match else 0


def _source_kind(unit_type: str, node_path: str, heading: str) -> str:
    text = f"{unit_type} {node_path} {heading}".lower()
    if unit_type in {"table", "figure", "reference"}:
        return unit_type
    if any(token in text for token in ("table", "表 ")):
        return "table"
    if any(token in text for token in ("figure", "fig.", "图 ")):
        return "figure"
    if any(token in text for token in ("reference", "参考文献", "引用")):
        return "reference"
    return "node"


def _page_range(node: Dict[str, Any]) -> List[Optional[int]]:
    return [node.get("page_start"), node.get("page_end")]


def _short_keywords(value: Any, text: str) -> List[str]:
    items = []
    if isinstance(value, list):
        items.extend(str(item) for item in value)
    elif isinstance(value, str):
        items.extend(re.split(r"[,，;；\s]+", value))
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text):
        items.append(token)
    return _unique_strings(items)[:12]


def _confidence(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
