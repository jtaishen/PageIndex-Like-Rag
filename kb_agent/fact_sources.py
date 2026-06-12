from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .fact_records import (
    CLAIM_TOKENS,
    apply_fact_quality_filters,
    claim_record,
    claim_sentences,
    claim_type,
    classify_claim,
    dedupe_facts,
    entity_record,
    evidence_node,
    first_node_by_kind,
    node_text,
    page_range_from_node,
    quality_warnings,
    relation_record,
    source_offsets_dict,
)
from .fact_utils import (
    confidence as _confidence,
    excerpt as _excerpt,
    normalize_key as _normalize_key,
)
from .utils import unique_strings as _unique_strings


ENTITY_PATTERNS = {
    "method": re.compile(r"[\u4e00-\u9fffA-Za-z0-9_-]{2,24}(?:方法|算法|模型|框架|机制)"),
    "system": re.compile(r"[\u4e00-\u9fffA-Za-z0-9_-]{2,24}(?:系统|平台|模块)"),
}
KNOWN_ENTITY_TERMS = {
    "task": ("任务规划", "任务分配", "任务分解", "协同调度", "工具调用", "路径规划"),
    "scenario": ("多智能体", "服务机器人", "动态环境", "开放环境", "真实场景"),
    "metric": ("任务完成率", "响应时间", "负载均衡", "成功率", "通信开销", "计算开销", "鲁棒性"),
    "model": ("大语言模型", "动作编码器", "技能库"),
}


def node_map(nodes: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(nodes, list):
        return {}
    return {str(node.get("node_id") or ""): node for node in nodes if isinstance(node, dict) and node.get("node_id")}


def select_fact_nodes(nodes: Any, innovation: Dict[str, Any], citation_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    node_by_id = node_map(nodes)
    selected: List[Dict[str, Any]] = []
    seen = set()
    for item in innovation.get("items") or []:
        if not isinstance(item, dict):
            continue
        for evidence in item.get("evidence") or []:
            node_id = str((evidence or {}).get("node_id") if isinstance(evidence, dict) else evidence)
            if node_id in node_by_id and node_id not in seen:
                selected.append(node_by_id[node_id])
                seen.add(node_id)
    for citation in citation_map.get("in_text_citations") or []:
        if not isinstance(citation, dict):
            continue
        node_id = str(citation.get("node_id") or "")
        if node_id in node_by_id and node_id not in seen:
            selected.append(node_by_id[node_id])
            seen.add(node_id)
    if isinstance(nodes, list):
        scored = []
        for node in nodes:
            if not isinstance(node, dict) or not node.get("text"):
                continue
            if str(node.get("kind") or "") == "reference":
                continue
            text = node_text(node)
            score = sum(2 for tokens in CLAIM_TOKENS.values() for token in tokens if token in text)
            if str(node.get("kind") or "") in {"abstract", "section"}:
                score += 2
            if score > 0:
                scored.append((score, int(node.get("order_index") or 0), node))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for _, _, node in scored:
            node_id = str(node.get("node_id") or "")
            if node_id and node_id not in seen:
                selected.append(node)
                seen.add(node_id)
            if len(selected) >= 24:
                break
    return selected[:24]


def rule_based_facts(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    quality: Dict[str, Any],
    innovation: Dict[str, Any],
    citation_map: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
    node_by_id: Dict[str, Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []
    abstract_node = first_node_by_kind(selected_nodes, "abstract") or (selected_nodes[0] if selected_nodes else None)

    for index, item in enumerate(innovation.get("items") or []):
        if not isinstance(item, dict):
            continue
        item_evidence_node = evidence_node(item.get("evidence"), node_by_id, {})
        text = _excerpt(str(item.get("claim") or item.get("title") or ""), 420)
        item_claim_type = claim_type(str(item.get("type") or "contribution"))
        claim = claim_record(doc_id, version_id, item_claim_type, text, item_evidence_node, "rule", 0.68, index)
        if claim:
            claims.append(claim)

    for node in selected_nodes:
        text = node_text(node)
        for sentence in claim_sentences(text):
            sentence_claim_type = classify_claim(sentence)
            claim = claim_record(doc_id, version_id, sentence_claim_type, sentence, node, "rule", 0.58, len(claims))
            if claim:
                claims.append(claim)

    keyword_node = abstract_node or (selected_nodes[0] if selected_nodes else None)
    for keyword in card.get("keywords") or []:
        entity = entity_record(doc_id, version_id, "topic", str(keyword), keyword_node, "rule", 0.58)
        if entity:
            entities.append(entity)
    for node in selected_nodes:
        entities.extend(_entities_from_text(doc_id, version_id, node, node_text(node)))

    citation_entities = _citation_entities(doc_id, version_id, citation_map, node_by_id)
    entities.extend(citation_entities)
    relation_source_entities = {entity["normalized_name"]: entity for entity in entities}
    for claim in claims:
        relations.append(
            relation_record(
                doc_id,
                version_id,
                "supports",
                str(claim.get("claim_id") or ""),
                str(claim.get("node_id") or ""),
                node_by_id.get(str(claim.get("node_id") or "")),
                "rule",
                0.62,
                subject_id=str(claim.get("claim_id") or ""),
                object_id=str(claim.get("node_id") or ""),
                text=f"claim supported by {claim.get('node_id')}",
            )
        )
        for entity in entities:
            if entity.get("node_id") == claim.get("node_id") and entity.get("name") and str(entity["name"]) in str(claim.get("text") or ""):
                relation_type = "limits" if claim.get("type") == "limitation" else "uses"
                relations.append(
                    relation_record(
                        doc_id,
                        version_id,
                        relation_type,
                        str(claim.get("claim_id") or ""),
                        str(entity.get("name") or ""),
                        node_by_id.get(str(claim.get("node_id") or "")),
                        "rule",
                        0.56,
                        subject_id=str(claim.get("claim_id") or ""),
                        object_id=str(entity.get("entity_id") or ""),
                    )
                )
    for citation in citation_map.get("relations") or []:
        if not isinstance(citation, dict):
            continue
        node = node_by_id.get(str(citation.get("node_id") or ""))
        if not node:
            continue
        ref_id = str(citation.get("ref_id") or "")
        ref_entity = relation_source_entities.get(_normalize_key(ref_id))
        relations.append(
            relation_record(
                doc_id,
                version_id,
                "cites",
                str(card.get("title") or doc_id),
                ref_id,
                node,
                "rule",
                0.72,
                object_id=str(ref_entity.get("entity_id") if ref_entity else ""),
                text=f"{card.get('title') or doc_id} cites {ref_id}",
            )
        )

    quality_stats = {"noise_filtered_count": 0, "long_claim_trimmed_count": 0, "entity_noise_filtered_count": 0}
    normalized = dedupe_facts({"claims": claims, "entities": entities, "relations": [item for item in relations if item]})
    apply_fact_quality_filters(normalized, quality_stats)
    normalized["status"] = "partial"
    normalized["source"] = "rule"
    normalized["quality_stats"] = quality_stats
    normalized["warnings"] = _unique_strings([*warnings, "rule_based_fact_extraction", *quality_warnings(quality)])
    return normalized


def merge_citation_relations(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    facts: Dict[str, Any],
    citation_map: Dict[str, Any],
    node_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    citation_relations = facts_from_citations(doc_id, version_id, card, citation_map, node_by_id)
    if not citation_relations:
        return facts
    merged = {
        "claims": facts.get("claims") or [],
        "entities": facts.get("entities") or [],
        "relations": [*citation_relations, *(facts.get("relations") or [])],
        "dedupe_stats": facts.get("dedupe_stats") or {},
    }
    normalized = dedupe_facts(merged)
    quality_stats = dict(facts.get("quality_stats") or {})
    apply_fact_quality_filters(normalized, quality_stats)
    normalized["status"] = facts.get("status") or "partial"
    normalized["source"] = facts.get("source") or "rule"
    normalized["quality_stats"] = quality_stats
    if facts.get("llm_batch_report"):
        normalized["llm_batch_report"] = facts.get("llm_batch_report")
    normalized["warnings"] = _unique_strings([*(facts.get("warnings") or []), "citation_fact_relations_added"])
    return normalized


def facts_from_citations(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    citation_map: Dict[str, Any],
    node_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_items = []
    for name in ("relations", "in_text_citations"):
        for item in citation_map.get(name) or []:
            if isinstance(item, dict):
                raw_items.append(item)
    relations: List[Dict[str, Any]] = []
    seen = set()
    title = str(card.get("title") or doc_id)
    for index, item in enumerate(raw_items):
        ref_id = str(item.get("ref_id") or item.get("reference_id") or "")
        node_id = str(item.get("node_id") or "")
        if not ref_id or not node_id:
            continue
        key = (ref_id, node_id)
        if key in seen:
            continue
        seen.add(key)
        node = node_by_id.get(node_id)
        if not node:
            continue
        relation = relation_record(
            doc_id,
            version_id,
            "cites",
            title,
            ref_id,
            node,
            "citation_rule",
            _confidence(item.get("confidence"), 0.72),
            object_id=ref_id,
            text=f"{title} cites {ref_id}",
            index=index,
            extra_evidence={
                "ref_id": ref_id,
                "page_range": item.get("page_range") or page_range_from_node(node),
                "citation_marker": item.get("marker") or item.get("raw") or "",
            },
        )
        if relation:
            relations.append(relation)
    return relations


def merge_table_facts(
    doc_id: str,
    version_id: str,
    facts: Dict[str, Any],
    table_content: Any,
    table_summaries: Any,
    node_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    del table_summaries
    table_facts = facts_from_tables(doc_id, version_id, table_content, node_by_id)
    if not table_facts["claims"] and not table_facts["entities"] and not table_facts["relations"]:
        return facts
    merged = {
        "claims": [*(facts.get("claims") or []), *table_facts["claims"]],
        "entities": [*(facts.get("entities") or []), *table_facts["entities"]],
        "relations": [*(facts.get("relations") or []), *table_facts["relations"]],
        "dedupe_stats": facts.get("dedupe_stats") or {},
    }
    normalized = dedupe_facts(merged)
    quality_stats = dict(facts.get("quality_stats") or {})
    apply_fact_quality_filters(normalized, quality_stats)
    normalized["status"] = facts.get("status") or "partial"
    normalized["source"] = facts.get("source") or "rule"
    normalized["quality_stats"] = quality_stats
    if facts.get("llm_batch_report"):
        normalized["llm_batch_report"] = facts.get("llm_batch_report")
    normalized["warnings"] = _unique_strings([*(facts.get("warnings") or []), *table_facts["warnings"]])
    return normalized


def facts_from_tables(
    doc_id: str,
    version_id: str,
    table_content: Any,
    node_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    tables = table_content.get("table_content") if isinstance(table_content, dict) else table_content
    if not isinstance(tables, list):
        return {"claims": [], "entities": [], "relations": [], "warnings": []}
    claims: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for table_index, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        node = _node_for_table(table, node_by_id)
        if not node:
            warnings.append("table_fact_without_node_skipped")
            continue
        source = str(table.get("source") or "table_rule")
        confidence = _confidence(table.get("confidence"), 0.62)
        evidence = _table_evidence(table)
        caption = str(table.get("caption") or "")
        headers = [str(item) for item in table.get("headers") or [] if str(item).strip()]
        rows = [row for row in table.get("rows") or [] if isinstance(row, dict)]
        metric_headers = [header for header in headers if _looks_like_metric(header)]
        if caption and rows:
            claim = claim_record(
                doc_id,
                version_id,
                "result",
                f"{caption} 汇总了 {len(rows)} 行实验或对比结果。",
                node,
                source,
                max(0.5, confidence - 0.02),
                table_index,
                extra_evidence=evidence,
            )
            if claim:
                claims.append(claim)
        for metric in metric_headers:
            entity = entity_record(doc_id, version_id, "metric", metric, node, source, confidence, extra_evidence=evidence)
            if entity:
                entities.append(entity)
        baseline_names: List[str] = []
        method_names: List[str] = []
        for row_index, row in enumerate(rows):
            cells = [
                str(cell.get("text") or "")
                for cell in row.get("cells") or []
                if isinstance(cell, dict) and str(cell.get("text") or "").strip()
            ]
            if not cells:
                continue
            method_name = _table_method_name(cells, headers)
            if method_name:
                method_type = "baseline" if _looks_like_baseline(method_name) else "method"
                method_entity = entity_record(
                    doc_id,
                    version_id,
                    method_type,
                    method_name,
                    node,
                    source,
                    confidence,
                    extra_evidence=evidence,
                )
                if method_entity:
                    entities.append(method_entity)
                method_names.append(method_name)
                if method_type == "baseline":
                    baseline_names.append(method_name)
            for cell_index, value in enumerate(cells):
                header = headers[cell_index] if cell_index < len(headers) else ""
                if _looks_like_metric(header):
                    metric_entity = entity_record(
                        doc_id,
                        version_id,
                        "metric",
                        header,
                        node,
                        source,
                        confidence,
                        extra_evidence=evidence,
                    )
                    if metric_entity:
                        entities.append(metric_entity)
                    result_name = f"{header}: {value}" if value else header
                    if _looks_like_result(value):
                        result_entity = entity_record(
                            doc_id,
                            version_id,
                            "result",
                            result_name,
                            node,
                            source,
                            max(0.5, confidence - 0.04),
                            extra_evidence=evidence,
                        )
                        if result_entity:
                            entities.append(result_entity)
                    if method_name and header and value:
                        relation = relation_record(
                            doc_id,
                            version_id,
                            "reports_metric",
                            method_name,
                            result_name,
                            node,
                            source,
                            max(0.5, confidence - 0.03),
                            text=f"{method_name} reports {result_name}",
                            index=table_index * 100 + row_index * 10 + cell_index,
                            extra_evidence=evidence,
                        )
                        if relation:
                            relations.append(relation)
                if _looks_like_dataset(header) or _looks_like_dataset(value):
                    dataset_name = value if not _looks_like_dataset(header) else value or header
                    dataset = entity_record(
                        doc_id,
                        version_id,
                        "dataset",
                        dataset_name,
                        node,
                        source,
                        confidence,
                        extra_evidence=evidence,
                    )
                    if dataset:
                        entities.append(dataset)
                    if method_name and dataset_name:
                        relation = relation_record(
                            doc_id,
                            version_id,
                            "evaluates_on",
                            method_name,
                            dataset_name,
                            node,
                            source,
                            max(0.5, confidence - 0.03),
                            index=table_index * 100 + row_index * 10 + cell_index,
                            extra_evidence=evidence,
                        )
                        if relation:
                            relations.append(relation)
        if baseline_names:
            for method_name in _unique_strings(method_names):
                if _looks_like_baseline(method_name):
                    continue
                for baseline in baseline_names[:3]:
                    relation_type = "improves" if _looks_like_ours(method_name) else "compares_with"
                    relation = relation_record(
                        doc_id,
                        version_id,
                        relation_type,
                        method_name,
                        baseline,
                        node,
                        source,
                        max(0.48, confidence - 0.08),
                        text=f"{method_name} {relation_type} {baseline} in {caption}",
                        index=table_index,
                        extra_evidence=evidence,
                    )
                    if relation:
                        relations.append(relation)
    return {
        "claims": claims,
        "entities": entities,
        "relations": relations,
        "warnings": _unique_strings([*warnings, "table_fact_extraction"]),
    }


def _table_evidence(table: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "table_id": table.get("table_id") or "",
        "caption_id": table.get("table_id") or "",
        "layout_block_id": table.get("layout_block_id") or "",
        "content_layout_block_ids": table.get("content_layout_block_ids") or [],
        "page_range": table.get("page_range") or [table.get("page"), table.get("page")],
        "source": table.get("source") or "",
        "source_parser": table.get("source_parser") or "",
    }


def _node_for_table(table: Dict[str, Any], node_by_id: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    layout_ids = [
        *(str(item) for item in table.get("content_layout_block_ids") or [] if str(item)),
        str(table.get("layout_block_id") or ""),
    ]
    caption_ids = {str(table.get("table_id") or ""), str(table.get("caption_id") or "")}
    for wanted in layout_ids:
        if not wanted:
            continue
        for node in node_by_id.values():
            offsets = source_offsets_dict(node)
            if str(offsets.get("layout_block_id") or "") == wanted:
                return node
    for node in node_by_id.values():
        offsets = source_offsets_dict(node)
        if str(offsets.get("caption_id") or "") in caption_ids:
            return node
    return None


def _table_method_name(cells: List[str], headers: List[str]) -> str:
    for index, cell in enumerate(cells):
        header = headers[index] if index < len(headers) else ""
        if _looks_like_result(cell):
            continue
        if _looks_like_method_header(header) or _looks_like_method(cell) or index == 0:
            return _excerpt(cell, 120)
    return ""


def _looks_like_metric(value: str) -> bool:
    return bool(re.search(r"(率|时间|开销|准确|精度|召回|F1|AUC|BLEU|ROUGE|指标|性能|鲁棒|延迟|吞吐)", value, re.IGNORECASE))


def _looks_like_method_header(value: str) -> bool:
    return bool(re.search(r"(方法|算法|模型|框架|method|model|baseline)", value, re.IGNORECASE))


def _looks_like_method(value: str) -> bool:
    return bool(re.search(r"(方法|算法|模型|框架|基线|baseline|ours|本文)", value, re.IGNORECASE))


def _looks_like_baseline(value: str) -> bool:
    return bool(re.search(r"(基线|baseline|对比|传统|规则)", value, re.IGNORECASE))


def _looks_like_ours(value: str) -> bool:
    return bool(re.search(r"(本文|ours|提出|本方法|所提)", value, re.IGNORECASE))


def _looks_like_result(value: str) -> bool:
    return bool(re.search(r"[-+]?\d+(?:\.\d+)?\s*(?:%|ms|s|秒|分|x|倍)?", value, re.IGNORECASE))


def _looks_like_dataset(value: str) -> bool:
    return bool(re.search(r"(数据集|dataset|benchmark|场景|语料|任务集)", value, re.IGNORECASE))


def _entities_from_text(doc_id: str, version_id: str, node: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    result = []
    for entity_type, pattern in ENTITY_PATTERNS.items():
        for match in pattern.findall(text):
            entity = entity_record(doc_id, version_id, entity_type, match, node, "rule", 0.56)
            if entity:
                result.append(entity)
    for entity_type, terms in KNOWN_ENTITY_TERMS.items():
        for term in terms:
            if term in text:
                entity = entity_record(doc_id, version_id, entity_type, term, node, "rule", 0.62)
                if entity:
                    result.append(entity)
    return result


def _citation_entities(
    doc_id: str,
    version_id: str,
    citation_map: Dict[str, Any],
    node_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    refs = {str(item.get("ref_id") or ""): item for item in citation_map.get("references") or [] if isinstance(item, dict)}
    result = []
    seen = set()
    for citation in citation_map.get("in_text_citations") or []:
        if not isinstance(citation, dict):
            continue
        ref_id = str(citation.get("ref_id") or "")
        node = node_by_id.get(str(citation.get("node_id") or ""))
        if not ref_id or not node or ref_id in seen:
            continue
        ref = refs.get(ref_id) or {}
        name = str(ref.get("title") or ref.get("raw") or ref_id)
        entity = entity_record(doc_id, version_id, "citation", name, node, "rule", 0.6, aliases=[ref_id])
        if entity:
            result.append(entity)
            seen.add(ref_id)
    return result
