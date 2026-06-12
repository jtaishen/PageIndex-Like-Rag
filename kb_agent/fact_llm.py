from __future__ import annotations

from typing import Any, Callable, Dict, List

from .config import llm_fact_batch_size, llm_fact_max_nodes
from .fact_records import (
    apply_fact_quality_filters,
    claim_record,
    claim_type,
    clean_llm_claim_text,
    clean_llm_entity_name,
    clean_llm_relation_endpoint,
    dedupe_facts,
    entity_record,
    evidence_node,
    node_text,
    quality_warnings,
    relation_record,
)
from .fact_utils import confidence as _confidence, excerpt as _excerpt
from .llm import LLMError
from .utils import string_list as _string_list, unique_strings as _unique_strings


JsonGenerator = Callable[[str, str], Dict[str, object]]


def extract_facts_with_llm_batches(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    quality: Dict[str, Any],
    innovation: Dict[str, Any],
    citation_map: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
    table_summaries: Any,
    node_by_id: Dict[str, Dict[str, Any]],
    warnings: List[str],
    *,
    json_generator: JsonGenerator,
) -> Dict[str, Any]:
    max_nodes = max(1, llm_fact_max_nodes())
    batch_size = max(1, llm_fact_batch_size())
    llm_nodes = selected_nodes[:max_nodes]
    batches = node_batches(llm_nodes, batch_size)
    parts: List[Dict[str, Any]] = []
    batch_warnings: List[str] = []
    timeout_count = 0
    fallback_count = 0
    for batch_index, batch_nodes in enumerate(batches, start=1):
        try:
            payload = extract_facts_batch_with_llm(
                card,
                quality,
                innovation,
                citation_map,
                batch_nodes,
                table_summaries,
                batch_index=batch_index,
                batch_count=len(batches),
                json_generator=json_generator,
            )
            normalized = normalize_fact_payload(
                payload,
                doc_id=doc_id,
                version_id=version_id,
                card=card,
                quality=quality,
                node_by_id=node_by_id,
                selected_nodes=batch_nodes,
                source="llm",
                status="extracted",
                warnings=[],
            )
            parts.append(normalized)
            batch_warnings.extend(f"batch_{batch_index}:{warning}" for warning in normalized.get("warnings") or [])
        except LLMError as exc:
            fallback_count += 1
            if exc.error_type == "request_timeout":
                timeout_count += 1
            batch_warnings.append(f"batch_{batch_index}:{exc.error_type}")
    if not parts:
        raise LLMError(
            "DeepSeek fact batch extraction failed.",
            error_type="all_fact_batches_failed",
            metadata={
                "batch_count": len(batches),
                "batch_success_count": 0,
                "batch_timeout_count": timeout_count,
                "batch_fallback_count": fallback_count,
            },
        )
    merged = merge_fact_parts(parts)
    batch_report = {
        "schema": "llm_fact_batch_report.v1",
        "llm_mode": "batch_json",
        "batch_size": batch_size,
        "max_nodes": max_nodes,
        "selected_node_count": len(llm_nodes),
        "batch_count": len(batches),
        "batch_success_count": len(parts),
        "batch_timeout_count": timeout_count,
        "batch_fallback_count": fallback_count,
        "llm_batch_warnings": _unique_strings(batch_warnings),
        "success_rate": round(len(parts) / max(1, len(batches)), 4),
    }
    status = "extracted" if fallback_count == 0 and merged.get("claims") else "partial"
    if fallback_count:
        warnings.append("llm_fact_batch_partial")
    merged["status"] = status
    merged["source"] = "llm"
    merged["llm_batch_report"] = batch_report
    merged["warnings"] = _unique_strings([*warnings, *merged.get("warnings", []), *batch_report["llm_batch_warnings"]])
    return merged


def node_batches(nodes: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    if not nodes:
        return []
    return [nodes[index : index + batch_size] for index in range(0, len(nodes), batch_size)]


def merge_fact_parts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    quality_stats: Dict[str, int] = {}
    dedupe_stats = {"dedupe_input_count": 0, "dedupe_merged_count": 0}
    warnings: List[str] = []
    merged = {
        "claims": [],
        "entities": [],
        "relations": [],
    }
    for part in parts:
        merged["claims"].extend(part.get("claims") or [])
        merged["entities"].extend(part.get("entities") or [])
        merged["relations"].extend(part.get("relations") or [])
        warnings.extend(part.get("warnings") or [])
        for key, value in (part.get("quality_stats") or {}).items():
            quality_stats[key] = quality_stats.get(key, 0) + int(value or 0)
        part_dedupe = part.get("dedupe_stats") or {}
        dedupe_stats["dedupe_input_count"] += int(part_dedupe.get("dedupe_input_count") or 0)
        dedupe_stats["dedupe_merged_count"] += int(part_dedupe.get("dedupe_merged_count") or 0)
    merged["dedupe_stats"] = dedupe_stats
    normalized = dedupe_facts(merged)
    apply_fact_quality_filters(normalized, quality_stats)
    normalized["quality_stats"] = quality_stats
    normalized["warnings"] = _unique_strings(warnings)
    return normalized


def extract_facts_batch_with_llm(
    card: Dict[str, Any],
    quality: Dict[str, Any],
    innovation: Dict[str, Any],
    citation_map: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
    table_summaries: Any,
    *,
    batch_index: int,
    batch_count: int,
    json_generator: JsonGenerator,
) -> Dict[str, object]:
    system_prompt = (
        "你是严谨的论文事实层抽取助手。只能基于给定短证据节点抽取短 claims、entities、relations，"
        "每条必须引用已有 evidence 编号或 node_id，不能编造。输出要短。只返回 JSON object，不要 Markdown。"
    )
    user_prompt = "\n".join(
        [
            f"fact_batch: {batch_index}/{batch_count}",
            "请抽取短 facts；每批最多 4 条 claims、6 个 entities、4 条 relations。返回格式：",
            '{"claims":[{"type":"","text":"","evidence":[],"confidence":0.0}],'
            '"entities":[{"type":"","name":"","aliases":[],"evidence":[],"confidence":0.0}],'
            '"relations":[{"type":"","subject":"","object":"","evidence":[],"confidence":0.0}],'
            '"warnings":[]}',
            "",
            f"title: {card.get('title')}",
            f"abstract: {_excerpt(str(card.get('abstract') or card.get('description') or ''), 160)}",
            f"quality_level: {quality.get('quality_level')}",
            f"innovation_items: {short_innovation_items(innovation)}",
            f"citation_relation_count: {len(citation_map.get('relations') or [])}",
            f"table_summaries: {short_table_summaries(table_summaries)}",
            "",
            "候选证据节点：",
            *format_nodes_for_prompt(selected_nodes),
        ]
    )
    return json_generator(system_prompt, user_prompt)


def short_innovation_items(innovation: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for item in (innovation.get("items") or [])[:4]:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "title": _excerpt(str(item.get("title") or ""), 80),
                "type": item.get("type") or "",
                "claim": _excerpt(str(item.get("claim") or ""), 80),
            }
        )
    return items


def short_table_summaries(table_summaries: Any) -> List[Dict[str, Any]]:
    if not isinstance(table_summaries, dict):
        return []
    result = []
    for item in (table_summaries.get("table_summaries") or [])[:3]:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "table_id": item.get("table_id") or "",
                "caption": _excerpt(str(item.get("caption") or ""), 80),
                "summary": _excerpt(str(item.get("summary") or ""), 120),
            }
        )
    return result


def format_nodes_for_prompt(nodes: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for index, node in enumerate(nodes, start=1):
        lines.append(f"[N{index}] node_id: {node.get('node_id')}")
        lines.append(f"node_path: {node.get('node_path')}")
        lines.append(f"page_range: {[node.get('page_start'), node.get('page_end')]}")
        lines.append(f"summary: {_excerpt(node_text(node), 180)}")
        lines.append("")
    return lines


def normalize_fact_payload(
    payload: Dict[str, object],
    *,
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    quality: Dict[str, Any],
    node_by_id: Dict[str, Dict[str, Any]],
    selected_nodes: List[Dict[str, Any]],
    source: str,
    status: str,
    warnings: List[str],
) -> Dict[str, Any]:
    del card
    selected_by_ref = {f"N{index}": node for index, node in enumerate(selected_nodes, start=1)}
    quality_stats = {"noise_filtered_count": 0, "long_claim_trimmed_count": 0}
    claims = []
    raw_claims = (payload.get("claims") if isinstance(payload, dict) else []) or []
    for index, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            continue
        item_evidence_node = evidence_node(item.get("evidence"), node_by_id, selected_by_ref)
        if not item_evidence_node:
            warnings.append("claim_without_evidence_skipped")
            continue
        item_claim_type = claim_type(str(item.get("type") or item.get("claim_type") or ""))
        text = clean_llm_claim_text(str(item.get("text") or item.get("claim") or ""), quality_stats)
        if not text:
            warnings.append("noisy_llm_claim_skipped")
            continue
        claim = claim_record(
            doc_id,
            version_id,
            item_claim_type,
            text,
            item_evidence_node,
            source,
            _confidence(item.get("confidence"), 0.75),
            index,
        )
        if claim:
            claims.append(claim)
    entities = []
    raw_entities = (payload.get("entities") if isinstance(payload, dict) else []) or []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        item_evidence_node = evidence_node(item.get("evidence"), node_by_id, selected_by_ref)
        if not item_evidence_node:
            warnings.append("entity_without_evidence_skipped")
            continue
        name = clean_llm_entity_name(str(item.get("name") or ""), quality_stats)
        if not name:
            warnings.append("noisy_llm_entity_skipped")
            continue
        aliases = [
            alias
            for alias in (clean_llm_entity_name(value, quality_stats, count_noise=False) for value in _string_list(item.get("aliases")))
            if alias
        ]
        entity = entity_record(
            doc_id,
            version_id,
            str(item.get("type") or item.get("entity_type") or "term"),
            name,
            item_evidence_node,
            source,
            _confidence(item.get("confidence"), 0.7),
            aliases=aliases,
        )
        if entity:
            entities.append(entity)
    relation_rows = []
    raw_relations = (payload.get("relations") if isinstance(payload, dict) else []) or []
    for index, item in enumerate(raw_relations):
        if not isinstance(item, dict):
            continue
        item_evidence_node = evidence_node(item.get("evidence"), node_by_id, selected_by_ref)
        if not item_evidence_node:
            warnings.append("relation_without_evidence_skipped")
            continue
        subject = clean_llm_relation_endpoint(str(item.get("subject") or item.get("subject_name") or ""), quality_stats)
        obj = clean_llm_relation_endpoint(str(item.get("object") or item.get("object_name") or ""), quality_stats)
        if not subject or not obj:
            warnings.append("noisy_llm_relation_skipped")
            continue
        relation = relation_record(
            doc_id,
            version_id,
            str(item.get("type") or item.get("relation_type") or "related_to"),
            subject,
            obj,
            item_evidence_node,
            source,
            _confidence(item.get("confidence"), 0.7),
            index=index,
        )
        if relation:
            relation_rows.append(relation)
    if not claims:
        status = "partial"
        warnings.append("empty_llm_claims")
    if quality_stats["noise_filtered_count"]:
        warnings.append("llm_noise_filtered")
    if quality_stats["long_claim_trimmed_count"]:
        warnings.append("long_llm_claim_trimmed")
    normalized = dedupe_facts({"claims": claims, "entities": entities, "relations": relation_rows})
    apply_fact_quality_filters(normalized, quality_stats)
    normalized["status"] = "extracted" if status == "extracted" and normalized["claims"] else "partial"
    normalized["source"] = source
    normalized["quality_stats"] = quality_stats
    normalized["warnings"] = _unique_strings([*warnings, *quality_warnings(quality)])
    return normalized
