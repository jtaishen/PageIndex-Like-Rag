from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .artifacts import get_artifact, get_doc_card, get_parse_quality, list_artifacts
from .config import llm_fact_batch_size, llm_fact_max_nodes
from .fact_queries import (
    fact_coverage_summary,
    fact_search,
    fact_summary_for_doc,
    get_claims,
    get_entities,
    get_fact_graph,
    get_relations,
)
from .fact_utils import (
    confidence as _confidence,
    excerpt as _excerpt,
    is_table_source as _is_table_source,
    normalize_key as _normalize_key,
)
from .insights import extract_doc_insights
from .llm import LLMError, generate_json_object
from .utils import compact_whitespace, stable_id, string_list as _string_list, unique_strings as _unique_strings, write_json


FACT_ARTIFACTS = {
    "claims.json",
    "entities.json",
    "relations.json",
    "fact_graph.json",
    "fact_report.json",
}
CLAIM_TOKENS = {
    "problem": ("问题", "挑战", "不足", "难以", "瓶颈"),
    "method": ("方法", "算法", "模型", "框架", "机制", "设计", "构建"),
    "contribution": ("提出", "贡献", "创新", "研究内容", "主要贡献"),
    "result": ("实验", "结果", "优于", "提升", "降低", "验证"),
    "limitation": ("局限", "不足", "限制", "未来工作", "展望"),
}
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


def extract_facts(
    db_path: Path,
    doc_id: str,
    *,
    force: bool = False,
    use_llm: bool = True,
    require_llm: bool = False,
) -> Dict[str, Any]:
    listing = list_artifacts(db_path, doc_id)
    artifact_dir = Path(str(listing["artifact_dir"]))
    version_id = str(listing["version_id"])
    existing = _read_existing_facts(db_path, doc_id)
    if existing and not force:
        return {
            "schema": "fact_extraction_result.v1",
            "doc_id": doc_id,
            "version_id": version_id,
            "artifact_dir": str(artifact_dir),
            "skipped": True,
            **existing,
        }

    card = get_doc_card(db_path, doc_id)
    quality = get_parse_quality(db_path, doc_id)
    nodes = _artifact_content(db_path, doc_id, "node_index.jsonl", [])
    layout = _artifact_content(db_path, doc_id, "layout_blocks.json", {})
    tables = _artifact_content(db_path, doc_id, "tables.json", {})
    table_content = _artifact_content(db_path, doc_id, "table_content.json", {})
    table_summaries = _artifact_content(db_path, doc_id, "table_summaries.json", {})
    figures = _artifact_content(db_path, doc_id, "figures.json", {})
    innovation, citation_map, insight_warnings = _read_or_extract_insight_artifacts(db_path, doc_id)
    node_by_id = _node_map(nodes)
    selected_nodes = _select_fact_nodes(nodes, innovation, citation_map)
    warnings = [*insight_warnings]
    llm_error = ""

    if use_llm:
        batch_report: Dict[str, Any] = {}
        try:
            facts = _extract_facts_with_llm_batches(
                doc_id,
                version_id,
                card,
                quality,
                innovation,
                citation_map,
                selected_nodes,
                table_summaries,
                node_by_id,
                warnings,
            )
            batch_report = dict(facts.get("llm_batch_report") or {})
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            warnings.append(f"llm_unavailable:{llm_error}")
            facts = _rule_based_facts(doc_id, version_id, card, quality, innovation, citation_map, selected_nodes, node_by_id, warnings)
            batch_report = {
                "schema": "llm_fact_batch_report.v1",
                "llm_mode": "batch_json",
                "batch_count": int(exc.metadata.get("batch_count") or 0),
                "batch_success_count": int(exc.metadata.get("batch_success_count") or 0),
                "batch_timeout_count": int(exc.metadata.get("batch_timeout_count") or 0),
                "batch_fallback_count": int(exc.metadata.get("batch_fallback_count") or 0),
                "llm_batch_warnings": [exc.error_type],
                "success_rate": 0.0,
            }
            facts["llm_batch_report"] = batch_report
    else:
        warnings.append("llm_disabled")
        facts = _rule_based_facts(doc_id, version_id, card, quality, innovation, citation_map, selected_nodes, node_by_id, warnings)

    facts = _merge_citation_relations(doc_id, version_id, card, facts, citation_map, node_by_id)
    facts = _merge_table_facts(doc_id, version_id, facts, table_content, table_summaries, node_by_id)
    facts = _dedupe_facts(facts)
    artifacts = _build_fact_artifacts(doc_id, version_id, card, quality, facts, llm_error)
    _write_fact_artifacts(artifact_dir, artifacts)
    _replace_fact_rows(db_path, doc_id, version_id, facts)
    return {
        "schema": "fact_extraction_result.v1",
        "doc_id": doc_id,
        "version_id": version_id,
        "artifact_dir": str(artifact_dir),
        "skipped": False,
        "claims_path": str(artifact_dir / "claims.json"),
        "entities_path": str(artifact_dir / "entities.json"),
        "relations_path": str(artifact_dir / "relations.json"),
        "fact_graph_path": str(artifact_dir / "fact_graph.json"),
        "fact_report_path": str(artifact_dir / "fact_report.json"),
        "claims": artifacts["claims"],
        "entities": artifacts["entities"],
        "relations": artifacts["relations"],
        "fact_graph": artifacts["fact_graph"],
        "fact_report": artifacts["fact_report"],
        "llm_error": llm_error,
    }


def _read_existing_facts(db_path: Path, doc_id: str) -> Optional[Dict[str, Any]]:
    try:
        claims = get_claims(db_path, doc_id)
        entities = get_entities(db_path, doc_id)
        relations = get_relations(db_path, doc_id)
        fact_graph = get_fact_graph(db_path, doc_id)
        fact_report = get_artifact(db_path, doc_id, "fact_report.json")["content"]
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if fact_report.get("schema") != "fact_report.v1":
        return None
    return {
        "claims": claims,
        "entities": entities,
        "relations": relations,
        "fact_graph": fact_graph,
        "fact_report": fact_report,
    }


def _read_or_extract_insight_artifacts(db_path: Path, doc_id: str) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    warnings: List[str] = []
    try:
        innovation = get_artifact(db_path, doc_id, "innovation.json")["content"]
        citation_map = get_artifact(db_path, doc_id, "citation_map.json")["content"]
    except (FileNotFoundError, KeyError, ValueError):
        innovation = {}
        citation_map = {}
    if innovation.get("schema") == "innovation.v1" and citation_map.get("schema") == "citation_map.v1":
        return innovation, citation_map, warnings
    result = extract_doc_insights(db_path, doc_id, force=True, use_llm=False)
    warnings.append(f"insights_rule_refreshed:{doc_id}")
    return result["innovation"], result["citation_map"], warnings


def _artifact_content(db_path: Path, doc_id: str, name: str, default: Any) -> Any:
    try:
        return get_artifact(db_path, doc_id, name)["content"]
    except (FileNotFoundError, KeyError, ValueError):
        return default


def _node_map(nodes: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(nodes, list):
        return {}
    return {str(node.get("node_id") or ""): node for node in nodes if isinstance(node, dict) and node.get("node_id")}


def _select_fact_nodes(nodes: Any, innovation: Dict[str, Any], citation_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    node_by_id = _node_map(nodes)
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
            text = _node_text(node)
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


def _extract_facts_with_llm_batches(
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
) -> Dict[str, Any]:
    max_nodes = max(1, llm_fact_max_nodes())
    batch_size = max(1, llm_fact_batch_size())
    llm_nodes = selected_nodes[:max_nodes]
    batches = _node_batches(llm_nodes, batch_size)
    parts: List[Dict[str, Any]] = []
    batch_warnings: List[str] = []
    timeout_count = 0
    fallback_count = 0
    for batch_index, batch_nodes in enumerate(batches, start=1):
        try:
            payload = _extract_facts_batch_with_llm(
                card,
                quality,
                innovation,
                citation_map,
                batch_nodes,
                table_summaries,
                batch_index=batch_index,
                batch_count=len(batches),
            )
            normalized = _normalize_fact_payload(
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
    merged = _merge_fact_parts(parts)
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


def _node_batches(nodes: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    if not nodes:
        return []
    return [nodes[index : index + batch_size] for index in range(0, len(nodes), batch_size)]


def _merge_fact_parts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    normalized = _dedupe_facts(merged)
    _apply_fact_quality_filters(normalized, quality_stats)
    normalized["quality_stats"] = quality_stats
    normalized["warnings"] = _unique_strings(warnings)
    return normalized


def _extract_facts_with_llm(
    card: Dict[str, Any],
    quality: Dict[str, Any],
    innovation: Dict[str, Any],
    citation_map: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
    layout: Any,
    tables: Any,
    table_summaries: Any,
    figures: Any,
) -> Dict[str, object]:
    return _extract_facts_batch_with_llm(
        card,
        quality,
        innovation,
        citation_map,
        selected_nodes,
        table_summaries,
        batch_index=1,
        batch_count=1,
    )


def _extract_facts_batch_with_llm(
    card: Dict[str, Any],
    quality: Dict[str, Any],
    innovation: Dict[str, Any],
    citation_map: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
    table_summaries: Any,
    *,
    batch_index: int,
    batch_count: int,
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
            f"innovation_items: {_short_innovation_items(innovation)}",
            f"citation_relation_count: {len(citation_map.get('relations') or [])}",
            f"table_summaries: {_short_table_summaries(table_summaries)}",
            "",
            "候选证据节点：",
            *_format_nodes_for_prompt(selected_nodes),
        ]
    )
    return generate_json_object(system_prompt, user_prompt)


def _short_innovation_items(innovation: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def _short_table_summaries(table_summaries: Any) -> List[Dict[str, Any]]:
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


def _format_nodes_for_prompt(nodes: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for index, node in enumerate(nodes, start=1):
        lines.append(f"[N{index}] node_id: {node.get('node_id')}")
        lines.append(f"node_path: {node.get('node_path')}")
        lines.append(f"page_range: {[node.get('page_start'), node.get('page_end')]}")
        lines.append(f"summary: {_excerpt(_node_text(node), 180)}")
        lines.append("")
    return lines


def _normalize_fact_payload(
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
        evidence_node = _evidence_node(item.get("evidence"), node_by_id, selected_by_ref)
        if not evidence_node:
            warnings.append("claim_without_evidence_skipped")
            continue
        claim_type = _claim_type(str(item.get("type") or item.get("claim_type") or ""))
        text = _clean_llm_claim_text(str(item.get("text") or item.get("claim") or ""), quality_stats)
        if not text:
            warnings.append("noisy_llm_claim_skipped")
            continue
        claim = _claim_record(doc_id, version_id, claim_type, text, evidence_node, source, _confidence(item.get("confidence"), 0.75), index)
        if claim:
            claims.append(claim)
    entities = []
    raw_entities = (payload.get("entities") if isinstance(payload, dict) else []) or []
    for index, item in enumerate(raw_entities):
        if not isinstance(item, dict):
            continue
        evidence_node = _evidence_node(item.get("evidence"), node_by_id, selected_by_ref)
        if not evidence_node:
            warnings.append("entity_without_evidence_skipped")
            continue
        name = _clean_llm_entity_name(str(item.get("name") or ""), quality_stats)
        if not name:
            warnings.append("noisy_llm_entity_skipped")
            continue
        aliases = [
            alias
            for alias in (_clean_llm_entity_name(value, quality_stats, count_noise=False) for value in _string_list(item.get("aliases")))
            if alias
        ]
        entity = _entity_record(
            doc_id,
            version_id,
            str(item.get("type") or item.get("entity_type") or "term"),
            name,
            evidence_node,
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
        evidence_node = _evidence_node(item.get("evidence"), node_by_id, selected_by_ref)
        if not evidence_node:
            warnings.append("relation_without_evidence_skipped")
            continue
        subject = _clean_llm_relation_endpoint(str(item.get("subject") or item.get("subject_name") or ""), quality_stats)
        obj = _clean_llm_relation_endpoint(str(item.get("object") or item.get("object_name") or ""), quality_stats)
        if not subject or not obj:
            warnings.append("noisy_llm_relation_skipped")
            continue
        relation = _relation_record(
            doc_id,
            version_id,
            str(item.get("type") or item.get("relation_type") or "related_to"),
            subject,
            obj,
            evidence_node,
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
    normalized = _dedupe_facts({"claims": claims, "entities": entities, "relations": relation_rows})
    _apply_fact_quality_filters(normalized, quality_stats)
    normalized["status"] = "extracted" if status == "extracted" and normalized["claims"] else "partial"
    normalized["source"] = source
    normalized["quality_stats"] = quality_stats
    normalized["warnings"] = _unique_strings([*warnings, *_quality_warnings(quality)])
    return normalized


def _rule_based_facts(
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
    abstract_node = _first_node_by_kind(selected_nodes, "abstract") or (selected_nodes[0] if selected_nodes else None)

    for index, item in enumerate(innovation.get("items") or []):
        if not isinstance(item, dict):
            continue
        evidence_node = _evidence_node(item.get("evidence"), node_by_id, {})
        text = _excerpt(str(item.get("claim") or item.get("title") or ""), 420)
        claim_type = _claim_type(str(item.get("type") or "contribution"))
        claim = _claim_record(doc_id, version_id, claim_type, text, evidence_node, "rule", 0.68, index)
        if claim:
            claims.append(claim)

    for node in selected_nodes:
        text = _node_text(node)
        for sentence in _claim_sentences(text):
            claim_type = _classify_claim(sentence)
            claim = _claim_record(doc_id, version_id, claim_type, sentence, node, "rule", 0.58, len(claims))
            if claim:
                claims.append(claim)

    keyword_node = abstract_node or (selected_nodes[0] if selected_nodes else None)
    for keyword in card.get("keywords") or []:
        entity = _entity_record(doc_id, version_id, "topic", str(keyword), keyword_node, "rule", 0.58)
        if entity:
            entities.append(entity)
    for node in selected_nodes:
        entities.extend(_entities_from_text(doc_id, version_id, node, _node_text(node)))

    citation_entities = _citation_entities(doc_id, version_id, citation_map, node_by_id)
    entities.extend(citation_entities)
    relation_source_entities = {entity["normalized_name"]: entity for entity in entities}
    for claim in claims:
        relations.append(
            _relation_record(
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
                    _relation_record(
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
            _relation_record(
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
    normalized = _dedupe_facts({"claims": claims, "entities": entities, "relations": [item for item in relations if item]})
    _apply_fact_quality_filters(normalized, quality_stats)
    normalized["status"] = "partial"
    normalized["source"] = "rule"
    normalized["quality_stats"] = quality_stats
    normalized["warnings"] = _unique_strings([*warnings, "rule_based_fact_extraction", *_quality_warnings(quality)])
    return normalized


def _merge_citation_relations(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    facts: Dict[str, Any],
    citation_map: Dict[str, Any],
    node_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    citation_relations = _facts_from_citations(doc_id, version_id, card, citation_map, node_by_id)
    if not citation_relations:
        return facts
    merged = {
        "claims": facts.get("claims") or [],
        "entities": facts.get("entities") or [],
        "relations": [*citation_relations, *(facts.get("relations") or [])],
        "dedupe_stats": facts.get("dedupe_stats") or {},
    }
    normalized = _dedupe_facts(merged)
    quality_stats = dict(facts.get("quality_stats") or {})
    _apply_fact_quality_filters(normalized, quality_stats)
    normalized["status"] = facts.get("status") or "partial"
    normalized["source"] = facts.get("source") or "rule"
    normalized["quality_stats"] = quality_stats
    if facts.get("llm_batch_report"):
        normalized["llm_batch_report"] = facts.get("llm_batch_report")
    normalized["warnings"] = _unique_strings([*(facts.get("warnings") or []), "citation_fact_relations_added"])
    return normalized


def _facts_from_citations(
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
        relation = _relation_record(
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
                "page_range": item.get("page_range") or _page_range_from_node(node),
                "citation_marker": item.get("marker") or item.get("raw") or "",
            },
        )
        if relation:
            relations.append(relation)
    return relations


def _merge_table_facts(
    doc_id: str,
    version_id: str,
    facts: Dict[str, Any],
    table_content: Any,
    table_summaries: Any,
    node_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    del table_summaries
    table_facts = _facts_from_tables(doc_id, version_id, table_content, node_by_id)
    if not table_facts["claims"] and not table_facts["entities"] and not table_facts["relations"]:
        return facts
    merged = {
        "claims": [*(facts.get("claims") or []), *table_facts["claims"]],
        "entities": [*(facts.get("entities") or []), *table_facts["entities"]],
        "relations": [*(facts.get("relations") or []), *table_facts["relations"]],
        "dedupe_stats": facts.get("dedupe_stats") or {},
    }
    normalized = _dedupe_facts(merged)
    quality_stats = dict(facts.get("quality_stats") or {})
    _apply_fact_quality_filters(normalized, quality_stats)
    normalized["status"] = facts.get("status") or "partial"
    normalized["source"] = facts.get("source") or "rule"
    normalized["quality_stats"] = quality_stats
    if facts.get("llm_batch_report"):
        normalized["llm_batch_report"] = facts.get("llm_batch_report")
    normalized["warnings"] = _unique_strings([*(facts.get("warnings") or []), *table_facts["warnings"]])
    return normalized


def _facts_from_tables(
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
            claim = _claim_record(
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
            entity = _entity_record(doc_id, version_id, "metric", metric, node, source, confidence, extra_evidence=evidence)
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
                method_entity = _entity_record(
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
                    metric_entity = _entity_record(
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
                        result_entity = _entity_record(
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
                        relation = _relation_record(
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
                    dataset = _entity_record(
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
                        relation = _relation_record(
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
                    relation = _relation_record(
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


def _build_fact_artifacts(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    quality: Dict[str, Any],
    facts: Dict[str, Any],
    llm_error: str,
) -> Dict[str, Any]:
    created_at = time.time()
    claims = facts.get("claims") or []
    entities = facts.get("entities") or []
    relations = facts.get("relations") or []
    warnings = _unique_strings(facts.get("warnings") or [])
    quality_stats = facts.get("quality_stats") or {}
    batch_report = facts.get("llm_batch_report") or {}
    dedupe_stats = facts.get("dedupe_stats") or {}
    low_confidence = sum(1 for item in [*claims, *entities, *relations] if float(item.get("confidence") or 0.0) < 0.5)
    no_evidence = sum(1 for item in [*claims, *entities, *relations] if not item.get("node_id"))
    table_backed = sum(1 for item in [*claims, *entities, *relations] if _is_table_source(str(item.get("source") or "")))
    claims_artifact = {
        "schema": "claims.v1",
        "status": facts.get("status") or "partial",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": card.get("title") or "",
        "count": len(claims),
        "claims": claims,
        "warnings": warnings,
        "created_at": created_at,
    }
    entities_artifact = {
        "schema": "entities.v1",
        "status": facts.get("status") or "partial",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": card.get("title") or "",
        "count": len(entities),
        "entities": entities,
        "warnings": warnings,
        "created_at": created_at,
    }
    relations_artifact = {
        "schema": "relations.v1",
        "status": facts.get("status") or "partial",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": card.get("title") or "",
        "count": len(relations),
        "relations": relations,
        "warnings": warnings,
        "created_at": created_at,
    }
    fact_graph = {
        "schema": "fact_graph.v1",
        "status": facts.get("status") or "partial",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": card.get("title") or "",
        "nodes": _graph_nodes(claims, entities),
        "edges": _graph_edges(relations),
        "warnings": warnings,
        "created_at": created_at,
    }
    fact_report = {
        "schema": "fact_report.v1",
        "status": facts.get("status") or "partial",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": card.get("title") or "",
        "source": facts.get("source") or "rule",
        "llm_used": (facts.get("source") == "llm" and not llm_error),
        "llm_mode": batch_report.get("llm_mode") or ("batch_json" if facts.get("source") == "llm" else ""),
        "batch_count": int(batch_report.get("batch_count") or 0),
        "batch_success_count": int(batch_report.get("batch_success_count") or 0),
        "batch_timeout_count": int(batch_report.get("batch_timeout_count") or 0),
        "batch_fallback_count": int(batch_report.get("batch_fallback_count") or 0),
        "llm_batch_warnings": batch_report.get("llm_batch_warnings") or [],
        "llm_batch_success_rate": float(batch_report.get("success_rate") or 0.0),
        "noise_filtered_count": int(quality_stats.get("noise_filtered_count") or 0),
        "entity_noise_filtered_count": int(quality_stats.get("entity_noise_filtered_count") or 0),
        "long_claim_trimmed_count": int(quality_stats.get("long_claim_trimmed_count") or 0),
        "dedupe_input_count": int(dedupe_stats.get("dedupe_input_count") or len(claims) + len(entities) + len(relations)),
        "dedupe_merged_count": int(dedupe_stats.get("dedupe_merged_count") or 0),
        "post_dedupe_duplicate_count": int(dedupe_stats.get("post_dedupe_duplicate_count") or 0),
        "fact_dedupe": dedupe_stats,
        "claim_count": len(claims),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "low_confidence_count": low_confidence,
        "no_evidence_count": no_evidence,
        "table_backed_fact_count": table_backed,
        "table_backed_fact_rate": round(table_backed / max(1, len(claims) + len(entities) + len(relations)), 4),
        "quality_level": quality.get("quality_level"),
        "quality_warnings": quality.get("quality_warnings") or [],
        "warnings": warnings,
        "llm_error": llm_error,
        "created_at": created_at,
    }
    return {
        "claims": claims_artifact,
        "entities": entities_artifact,
        "relations": relations_artifact,
        "fact_graph": fact_graph,
        "fact_report": fact_report,
    }


def _write_fact_artifacts(artifact_dir: Path, artifacts: Dict[str, Any]) -> None:
    write_json(artifact_dir / "claims.json", artifacts["claims"])
    write_json(artifact_dir / "entities.json", artifacts["entities"])
    write_json(artifact_dir / "relations.json", artifacts["relations"])
    write_json(artifact_dir / "fact_graph.json", artifacts["fact_graph"])
    write_json(artifact_dir / "fact_report.json", artifacts["fact_report"])


def _replace_fact_rows(db_path: Path, doc_id: str, version_id: str, facts: Dict[str, Any]) -> None:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        # Facts are queried by document, not by historical parser version. Keeping
        # older version rows makes repeated sync/extract runs look like duplicate
        # facts, so refresh the document's fact layer as a single current snapshot.
        db.delete_paper_facts(conn, doc_id)
        db.insert_paper_claims(conn, facts.get("claims") or [])
        db.insert_paper_entities(conn, facts.get("entities") or [])
        db.insert_paper_relations(conn, facts.get("relations") or [])
        conn.commit()
    finally:
        conn.close()


def _claim_record(
    doc_id: str,
    version_id: str,
    claim_type: str,
    text: str,
    node: Optional[Dict[str, Any]],
    source: str,
    confidence: float,
    index: int,
    extra_evidence: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    clean = _excerpt(text, 420)
    if not clean or not node:
        return None
    node_id = str(node.get("node_id") or "")
    if not node_id:
        return None
    page_range = _page_range_from_node(node)
    claim_id = stable_id("claim", doc_id, version_id, claim_type, clean, node_id, index, length=14)
    return {
        "claim_id": claim_id,
        "doc_id": doc_id,
        "version_id": version_id,
        "node_id": node_id,
        "type": claim_type,
        "claim_type": claim_type,
        "text": clean,
        "normalized_text": _normalize_key(clean),
        "page_range": page_range,
        "confidence": confidence,
        "source": source,
        "evidence": _merge_evidence(_evidence_ref(node), extra_evidence),
        "created_at": time.time(),
    }


def _entity_record(
    doc_id: str,
    version_id: str,
    entity_type: str,
    name: str,
    node: Optional[Dict[str, Any]],
    source: str,
    confidence: float,
    aliases: Optional[List[str]] = None,
    extra_evidence: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    clean = _excerpt(name, 120)
    if not clean or not node:
        return None
    node_id = str(node.get("node_id") or "")
    if not node_id:
        return None
    normalized = _normalize_key(clean)
    entity_id = stable_id("entity", doc_id, version_id, entity_type, normalized, length=14)
    return {
        "entity_id": entity_id,
        "doc_id": doc_id,
        "version_id": version_id,
        "node_id": node_id,
        "type": entity_type,
        "entity_type": entity_type,
        "name": clean,
        "normalized_name": normalized,
        "aliases": aliases or [],
        "page_range": _page_range_from_node(node),
        "confidence": confidence,
        "source": source,
        "evidence": _merge_evidence(_evidence_ref(node), extra_evidence),
        "created_at": time.time(),
    }


def _relation_record(
    doc_id: str,
    version_id: str,
    relation_type: str,
    subject_name: str,
    object_name: str,
    node: Optional[Dict[str, Any]],
    source: str,
    confidence: float,
    *,
    subject_id: str = "",
    object_id: str = "",
    text: str = "",
    index: int = 0,
    extra_evidence: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not node:
        return None
    node_id = str(node.get("node_id") or "")
    if not node_id:
        return None
    subject = _excerpt(subject_name, 160)
    obj = _excerpt(object_name, 160)
    if not subject or not obj:
        return None
    clean_text = _excerpt(text or f"{subject} {relation_type} {obj}", 260)
    relation_id = stable_id("rel", doc_id, version_id, relation_type, subject_id or subject, object_id or obj, node_id, index, length=14)
    return {
        "relation_id": relation_id,
        "doc_id": doc_id,
        "version_id": version_id,
        "node_id": node_id,
        "type": relation_type,
        "relation_type": relation_type,
        "subject_id": subject_id,
        "subject_name": subject,
        "object_id": object_id,
        "object_name": obj,
        "text": clean_text,
        "page_range": _page_range_from_node(node),
        "confidence": confidence,
        "source": source,
        "evidence": _merge_evidence(_evidence_ref(node), extra_evidence),
        "created_at": time.time(),
    }


def _merge_evidence(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base or {})
    if extra:
        for key, value in extra.items():
            if value not in (None, "", [], {}):
                merged[key] = value
    return merged


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
            offsets = _source_offsets_dict(node)
            if str(offsets.get("layout_block_id") or "") == wanted:
                return node
    for node in node_by_id.values():
        offsets = _source_offsets_dict(node)
        if str(offsets.get("caption_id") or "") in caption_ids:
            return node
    return None


def _source_offsets_dict(node: Dict[str, Any]) -> Dict[str, Any]:
    offsets = node.get("source_offsets") or {}
    if isinstance(offsets, str):
        try:
            parsed = json.loads(offsets)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return offsets if isinstance(offsets, dict) else {}


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
            entity = _entity_record(doc_id, version_id, entity_type, match, node, "rule", 0.56)
            if entity:
                result.append(entity)
    for entity_type, terms in KNOWN_ENTITY_TERMS.items():
        for term in terms:
            if term in text:
                entity = _entity_record(doc_id, version_id, entity_type, term, node, "rule", 0.62)
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
        entity = _entity_record(doc_id, version_id, "citation", name, node, "rule", 0.6, aliases=[ref_id])
        if entity:
            result.append(entity)
            seen.add(ref_id)
    return result


def _dedupe_facts(facts: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    claims, claim_stats = _dedupe_by_key(facts.get("claims") or [], "normalized_text", "claim")
    entities, entity_stats = _dedupe_by_key(facts.get("entities") or [], "normalized_name", "entity")
    relations, relation_stats = _dedupe_relations(facts.get("relations") or [])
    prior_stats = facts.get("dedupe_stats") or {}
    input_count = claim_stats["input_count"] + entity_stats["input_count"] + relation_stats["input_count"]
    output_count = len(claims) + len(entities) + len(relations)
    current_merged = max(0, input_count - output_count)
    prior_input = int(prior_stats.get("dedupe_input_count") or 0)
    prior_merged = int(prior_stats.get("dedupe_merged_count") or 0)
    dedupe_stats = {
        "schema": "fact_dedupe.v1",
        "dedupe_input_count": max(input_count, prior_input),
        "dedupe_output_count": output_count,
        "dedupe_merged_count": prior_merged + current_merged,
        "post_dedupe_duplicate_count": _post_dedupe_duplicate_count(claims, entities, relations),
        "by_type": {
            "claims": claim_stats,
            "entities": entity_stats,
            "relations": relation_stats,
        },
    }
    return {
        "claims": claims,
        "entities": entities,
        "relations": relations,
        "dedupe_stats": dedupe_stats,
        "status": facts.get("status"),
        "source": facts.get("source"),
        "quality_stats": facts.get("quality_stats") or {},
        "llm_batch_report": facts.get("llm_batch_report") or {},
        "warnings": facts.get("warnings") or [],
    }


def _apply_fact_quality_filters(facts: Dict[str, Any], quality_stats: Dict[str, int]) -> None:
    entities = facts.get("entities") or []
    filtered = []
    removed = 0
    for entity in entities:
        if _looks_like_noisy_entity_name(str(entity.get("name") or ""), str(entity.get("type") or entity.get("entity_type") or "")):
            removed += 1
            continue
        filtered.append(entity)
    if removed:
        quality_stats["entity_noise_filtered_count"] = quality_stats.get("entity_noise_filtered_count", 0) + removed
    facts["entities"] = filtered


def _looks_like_noisy_entity_name(value: str, entity_type: str = "") -> bool:
    text = compact_whitespace(value).strip(" ,，.。;；:：()（）[]【】")
    lowered = text.lower()
    if not text:
        return True
    if lowered in {"no", "no.", "ra", "rb", "rc", "rd"}:
        return True
    if re.fullmatch(r"(?:[A-Za-z]{1,2}\.?|\d+)", text):
        return True
    if entity_type in {"metric", "result", "dataset", "citation"}:
        return False
    if len(text) < 2:
        return True
    if re.search(r"[。！？!?；;\n]", text):
        return True
    if "、" in text and len(text) > 12:
        return True
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    allowed_suffix = ("方法", "算法", "模型", "框架", "系统", "平台", "数据集", "指标", "任务", "场景", "机制", "模块")
    if chinese_count > 24 and not text.endswith(allowed_suffix):
        return True
    if len(text) > 18 and any(token in text for token in ("则", "并", "以及", "进行", "涵盖", "包括", "通过", "用于")) and not text.endswith(allowed_suffix):
        return True
    return False


def _dedupe_by_key(items: List[Dict[str, Any]], key: str, fact_type: str) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    result = []
    seen: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for item in items:
        fallback_key = "text" if fact_type == "claim" else "name"
        normalized_value = _normalize_key(str(item.get(key) or item.get(fallback_key) or ""))
        if fact_type == "claim" and len(normalized_value) > 120:
            normalized_value = normalized_value[:120]
        marker = (
            str(item.get("doc_id") or ""),
            str(item.get("type") or ""),
            normalized_value,
        )
        if not marker[2]:
            continue
        existing = seen.get(marker)
        if existing is None:
            kept = dict(item)
            seen[marker] = kept
            result.append(kept)
            continue
        _merge_duplicate_fact(existing, item)
    stats = {
        "input_count": len(items),
        "output_count": len(result[:300]),
        "merged_count": max(0, len(items) - len(result[:300])),
    }
    del fact_type
    return result[:300], stats


def _dedupe_relations(items: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    result = []
    seen: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for item in items:
        relation_type = str(item.get("relation_type") or item.get("type") or "")
        marker = (
            str(item.get("doc_id") or ""),
            relation_type,
            _relation_endpoint_key(item, "subject"),
            _relation_endpoint_key(item, "object"),
        )
        if relation_type in {"cites", "citation"}:
            marker = (*marker, str(item.get("node_id") or ""))
        if not marker[1] or not marker[2] or not marker[3]:
            continue
        existing = seen.get(marker)
        if existing is None:
            kept = dict(item)
            seen[marker] = kept
            result.append(kept)
            continue
        _merge_duplicate_fact(existing, item)
    stats = {
        "input_count": len(items),
        "output_count": len(result[:500]),
        "merged_count": max(0, len(items) - len(result[:500])),
    }
    return result[:500], stats


def _relation_endpoint_key(item: Dict[str, Any], side: str) -> str:
    name = _normalize_key(str(item.get(f"{side}_name") or ""))
    if name:
        return name
    return _normalize_key(str(item.get(f"{side}_id") or ""))


def _merge_duplicate_fact(existing: Dict[str, Any], candidate: Dict[str, Any]) -> None:
    existing_aliases = list(existing.get("aliases") or [])
    if float(candidate.get("confidence") or 0.0) > float(existing.get("confidence") or 0.0):
        keep_keys = {"claim_id", "entity_id", "relation_id", "created_at"}
        preserved = {key: existing.get(key) for key in keep_keys if key in existing}
        existing.clear()
        existing.update(candidate)
        existing.update({key: value for key, value in preserved.items() if value})
    existing["confidence"] = max(float(existing.get("confidence") or 0.0), float(candidate.get("confidence") or 0.0))
    existing["evidence"] = _merge_evidence(existing.get("evidence") or {}, candidate.get("evidence") or {})
    aliases = _unique_strings([*existing_aliases, *(existing.get("aliases") or []), *(candidate.get("aliases") or [])])
    if aliases:
        existing["aliases"] = aliases


def _post_dedupe_duplicate_count(
    claims: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> int:
    markers = []
    markers.extend(("claim", item.get("doc_id"), item.get("type"), item.get("normalized_text")) for item in claims)
    markers.extend(("entity", item.get("doc_id"), item.get("type"), item.get("normalized_name")) for item in entities)
    for item in relations:
        relation_type = item.get("relation_type") or item.get("type")
        marker = (
            "relation",
            item.get("doc_id"),
            relation_type,
            _relation_endpoint_key(item, "subject"),
            _relation_endpoint_key(item, "object"),
        )
        if relation_type in {"cites", "citation"}:
            marker = (*marker, item.get("node_id"))
        markers.append(marker)
    seen = set()
    duplicates = 0
    for marker in markers:
        if marker in seen:
            duplicates += 1
            continue
        seen.add(marker)
    return duplicates


def _graph_nodes(claims: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes = []
    for claim in claims:
        nodes.append(
            {
                "id": claim.get("claim_id"),
                "kind": "claim",
                "type": claim.get("type"),
                "label": _excerpt(str(claim.get("text") or ""), 120),
                "doc_id": claim.get("doc_id"),
                "node_id": claim.get("node_id"),
                "page_range": claim.get("page_range"),
                "confidence": claim.get("confidence"),
            }
        )
    for entity in entities:
        nodes.append(
            {
                "id": entity.get("entity_id"),
                "kind": "entity",
                "type": entity.get("type"),
                "label": entity.get("name"),
                "doc_id": entity.get("doc_id"),
                "node_id": entity.get("node_id"),
                "page_range": entity.get("page_range"),
                "confidence": entity.get("confidence"),
            }
        )
    return nodes


def _graph_edges(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": relation.get("relation_id"),
            "type": relation.get("type"),
            "source": relation.get("subject_id") or relation.get("subject_name"),
            "target": relation.get("object_id") or relation.get("object_name"),
            "doc_id": relation.get("doc_id"),
            "node_id": relation.get("node_id"),
            "page_range": relation.get("page_range"),
            "confidence": relation.get("confidence"),
        }
        for relation in relations
    ]


def _replace_long_text(value: str) -> str:
    return _excerpt(compact_whitespace(value), 420)


def _clean_llm_claim_text(value: str, stats: Dict[str, int]) -> str:
    text = compact_whitespace(value)
    if len(text) < 8:
        stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    if re.search(r"(node_id|page_range|evidence packet|证据包)\s*[:：]", text, re.IGNORECASE):
        stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    if len(text) > 220:
        stats["long_claim_trimmed_count"] = stats.get("long_claim_trimmed_count", 0) + 1
        text = _excerpt(text, 220)
    return text


def _clean_llm_entity_name(value: str, stats: Dict[str, int], *, count_noise: bool = True) -> str:
    text = compact_whitespace(value).strip(" ,，.。;；:：()（）[]【】")
    if len(text) < 2:
        if count_noise:
            stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    if len(text) > 48 or re.search(r"[。！？!?；;\n]", text):
        if count_noise:
            stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    if re.fullmatch(r"[A-Za-z]", text):
        if count_noise:
            stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    if len(text) > 28 and not re.search(r"(方法|算法|模型|框架|系统|平台|数据集|指标|任务|场景|机制)$", text):
        if count_noise:
            stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    if _looks_like_noisy_entity_name(text):
        if count_noise:
            stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    return text


def _clean_llm_relation_endpoint(value: str, stats: Dict[str, int]) -> str:
    text = compact_whitespace(value).strip(" ,，.。;；:：()（）[]【】")
    if len(text) < 2:
        stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    if len(text) > 80:
        stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    return text


def _claim_sentences(text: str) -> List[str]:
    sentences = [compact_whitespace(item) for item in re.split(r"[。！？!?；;]\s*", text) if compact_whitespace(item)]
    result = []
    for sentence in sentences:
        if len(sentence) < 12:
            continue
        if any(token in sentence for tokens in CLAIM_TOKENS.values() for token in tokens):
            result.append(_replace_long_text(sentence))
        if len(result) >= 5:
            break
    return result


def _classify_claim(text: str) -> str:
    for claim_type, tokens in CLAIM_TOKENS.items():
        if any(token in text for token in tokens):
            return claim_type
    return "claim"


def _claim_type(value: str) -> str:
    text = compact_whitespace(value).lower()
    mapping = {
        "method": "method",
        "approach": "method",
        "contribution": "contribution",
        "innovation": "contribution",
        "result": "result",
        "experiment": "result",
        "limitation": "limitation",
        "future": "future_work",
        "problem": "problem",
    }
    for key, target in mapping.items():
        if key in text:
            return target
    if any(token in value for token in CLAIM_TOKENS["limitation"]):
        return "limitation"
    if any(token in value for token in CLAIM_TOKENS["result"]):
        return "result"
    if any(token in value for token in CLAIM_TOKENS["method"]):
        return "method"
    return "claim"


def _quality_warnings(quality: Dict[str, Any]) -> List[str]:
    warnings = []
    quality_level = str(quality.get("quality_level") or "")
    if quality_level == "weak":
        warnings.append("weak_parse_quality")
    for warning in quality.get("quality_warnings") or []:
        if warning in {"page_only_tree", "weak_layout_blocks", "missing_abstract", "missing_references"}:
            warnings.append(str(warning))
    return warnings


def _evidence_node(value: Any, node_by_id: Dict[str, Dict[str, Any]], selected_by_ref: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    refs = value if isinstance(value, list) else [value]
    for ref in refs:
        node_id = ""
        if isinstance(ref, dict):
            node_id = str(ref.get("node_id") or ref.get("id") or "")
        else:
            node_id = str(ref or "")
        if node_id in selected_by_ref:
            return selected_by_ref[node_id]
        if node_id in node_by_id:
            return node_by_id[node_id]
    return None


def _evidence_ref(node: Dict[str, Any]) -> Dict[str, Any]:
    source_offsets = node.get("source_offsets")
    if isinstance(source_offsets, str):
        try:
            source_offsets = json.loads(source_offsets)
        except json.JSONDecodeError:
            source_offsets = {}
    return {
        "doc_id": node.get("doc_id") or "",
        "node_id": node.get("node_id") or "",
        "node_path": node.get("node_path") or "",
        "page_range": _page_range_from_node(node),
        "layout_block_id": (source_offsets or {}).get("layout_block_id") if isinstance(source_offsets, dict) else "",
        "caption_id": (source_offsets or {}).get("caption_id") if isinstance(source_offsets, dict) else "",
    }


def _page_range_from_node(node: Dict[str, Any]) -> List[Any]:
    return [node.get("page_start"), node.get("page_end")]


def _first_node_by_kind(nodes: List[Dict[str, Any]], kind: str) -> Optional[Dict[str, Any]]:
    for node in nodes:
        if str(node.get("kind") or node.get("type") or "") == kind:
            return node
    return None


def _node_text(node: Dict[str, Any]) -> str:
    return compact_whitespace(" ".join(str(node.get(key) or "") for key in ("heading", "summary", "text", "node_path")))
