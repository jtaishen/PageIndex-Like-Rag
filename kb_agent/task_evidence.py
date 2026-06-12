from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .utils import compact_whitespace, stable_id, unique_strings


def flatten_dimension_evidence(evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for by_doc in evidence_by_dimension.values():
        for items in by_doc.values():
            evidence.extend(items)
    return dedupe_evidence(evidence)


def flatten_dimension_evidence_raw(evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for by_doc in evidence_by_dimension.values():
        for items in by_doc.values():
            evidence.extend(items)
    return evidence


def normalize_evidence_refs(value: object, fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not value:
        return fallback[:4]
    by_node_id = {str(item.get("node_id") or ""): item for item in fallback}
    result = []
    if isinstance(value, list):
        for raw in value:
            node_id = ""
            if isinstance(raw, str):
                node_id = raw
            elif isinstance(raw, dict):
                node_id = str(raw.get("node_id") or raw.get("id") or "")
            if node_id and node_id in by_node_id:
                result.append(by_node_id[node_id])
    return dedupe_evidence(result or fallback)[:4]


def dedupe_evidence(evidence: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _dedupe_evidence_with_stats(evidence)[0]


def compact_section_evidence(
    evidence: Iterable[Dict[str, Any]],
    *,
    max_items: int = 8,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    deduped, dedupe_stats = _dedupe_evidence_with_stats(evidence)
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for item in deduped:
        doc_id = str(item.get("doc_id") or "")
        by_doc.setdefault(doc_id, []).append(item)
    for items in by_doc.values():
        items.sort(key=_evidence_priority, reverse=True)
    compacted: List[Dict[str, Any]] = []
    doc_ids = sorted(by_doc)
    cursor = 0
    while len(compacted) < max_items and any(by_doc.values()):
        doc_id = doc_ids[cursor % len(doc_ids)]
        cursor += 1
        if not by_doc.get(doc_id):
            continue
        compacted.append(_compact_evidence_item(by_doc[doc_id].pop(0)))
    warnings = []
    if dedupe_stats.get("duplicate_evidence_removed"):
        warnings.append("duplicate_evidence_compacted")
    if int(dedupe_stats.get("unique_evidence_count") or 0) > len(compacted):
        warnings.append("section_evidence_truncated")
    source_doc_ids = unique_strings(str(item.get("doc_id") or "") for item in compacted if item.get("doc_id"))
    return compacted, {
        "schema": "section_evidence_compaction.v1",
        "raw_evidence_count": dedupe_stats.get("raw_evidence_count", 0),
        "unique_evidence_count": dedupe_stats.get("unique_evidence_count", 0),
        "duplicate_evidence_removed": dedupe_stats.get("duplicate_evidence_removed", 0),
        "kept_evidence_count": len(compacted),
        "max_evidence_count": max_items,
        "source_doc_count": len(source_doc_ids),
        "source_doc_ids": source_doc_ids,
        "warnings": warnings,
    }


def section_evidence_quality(
    compaction_by_section: Dict[str, Dict[str, Any]],
    section_evidence: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    removed_by_section = {
        section_id: int(report.get("duplicate_evidence_removed") or 0)
        for section_id, report in compaction_by_section.items()
    }
    post_duplicate_count = 0
    for items in section_evidence.values():
        post_stats = _dedupe_evidence_with_stats(items)[1]
        post_duplicate_count += int(post_stats.get("duplicate_evidence_removed") or 0)
    pre_count = sum(int(report.get("raw_evidence_count") or 0) for report in compaction_by_section.values())
    post_count = sum(len(items) for items in section_evidence.values())
    warnings = []
    if post_duplicate_count:
        warnings.append("post_dedupe_duplicate_evidence")
    return {
        "schema": "section_evidence_quality.v1",
        "pre_dedupe_count": pre_count,
        "post_dedupe_count": post_count,
        "duplicate_evidence_removed": sum(removed_by_section.values()),
        "duplicate_evidence_removed_by_section": removed_by_section,
        "post_dedupe_duplicate_count": post_duplicate_count,
        "warnings": warnings,
    }


def evidence_duplicate_summary(evidence: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    _, stats = _dedupe_evidence_with_stats(evidence)
    return stats


def evidence_confidence(evidence: List[Dict[str, Any]]) -> float:
    if len(evidence) >= 3:
        return 0.75
    if len(evidence) >= 1:
        return 0.6
    return 0.25


def _compact_evidence_item(item: Dict[str, Any]) -> Dict[str, Any]:
    compacted = dict(item)
    summary = compact_whitespace(
        str(item.get("summary") or item.get("claim") or item.get("excerpt") or item.get("snippet") or "")
    )
    compacted["evidence_summary"] = summary[:240]
    return compacted


def _dedupe_evidence_with_stats(evidence: Iterable[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = [item for item in evidence if isinstance(item, dict)]
    result = []
    seen: Dict[tuple[str, str], Dict[str, Any]] = {}
    duplicate_count = 0
    for index, item in enumerate(raw):
        keys = _evidence_keys(item)
        if not keys:
            continue
        existing_key = next((key for key in keys if key in seen), None)
        if existing_key is None:
            kept = dict(item)
            kept.setdefault("dedupe_reason", "kept:unique")
            kept["_dedupe_index"] = index
            for key in keys:
                seen[key] = kept
            result.append(kept)
            continue
        duplicate_count += 1
        existing = seen[existing_key]
        if _evidence_priority(item) > _evidence_priority(existing):
            replacement = dict(item)
            replacement["dedupe_reason"] = "kept:higher_score"
            replacement["_dedupe_index"] = existing.get("_dedupe_index", index)
            result = [replacement if current is existing else current for current in result]
            for key in keys:
                seen[key] = replacement
    for item in result:
        item.pop("_dedupe_index", None)
    stats = {
        "schema": "evidence_dedupe.v1",
        "raw_evidence_count": len(raw),
        "unique_evidence_count": len(result),
        "duplicate_evidence_removed": duplicate_count,
    }
    return result, stats


def _evidence_keys(item: Dict[str, Any]) -> List[tuple[str, str]]:
    doc_id = str(item.get("doc_id") or "")
    node_id = str(item.get("node_id") or "")
    keys = []
    if doc_id and node_id:
        keys.append(("node", f"{doc_id}:{node_id}"))
    node_path = compact_whitespace(str(item.get("node_path") or ""))
    text = compact_whitespace(
        str(
            item.get("evidence_summary")
            or item.get("summary")
            or item.get("claim")
            or item.get("excerpt")
            or item.get("snippet")
            or ""
        )
    )
    if doc_id and node_path and text:
        keys.append(("path_text", stable_id("evidence_text", doc_id, node_path, text[:260], length=18)))
    elif doc_id and text:
        keys.append(("text", stable_id("evidence_text", doc_id, text[:260], length=18)))
    return keys


def _evidence_priority(item: Dict[str, Any]) -> tuple[float, float]:
    score = item.get("tree_score")
    if score is None:
        score = item.get("confidence")
    try:
        parsed = float(score or 0.0)
    except (TypeError, ValueError):
        parsed = 0.0
    excerpt_len = len(compact_whitespace(str(item.get("excerpt") or "")))
    return (parsed, min(240.0, float(excerpt_len)))
