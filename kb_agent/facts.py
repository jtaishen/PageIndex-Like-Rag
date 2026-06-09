from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .artifacts import get_artifact, get_doc_card, get_parse_quality, list_artifacts
from .insights import extract_doc_insights
from .llm import LLMError, generate_json_object
from .utils import compact_whitespace, stable_id, write_json


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
        try:
            payload = _extract_facts_with_llm(
                card,
                quality,
                innovation,
                citation_map,
                selected_nodes,
                layout,
                tables,
                table_summaries,
                figures,
            )
            facts = _normalize_fact_payload(
                payload,
                doc_id=doc_id,
                version_id=version_id,
                card=card,
                quality=quality,
                node_by_id=node_by_id,
                selected_nodes=selected_nodes,
                source="llm",
                status="extracted",
                warnings=warnings,
            )
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            warnings.append(f"llm_unavailable:{llm_error}")
            facts = _rule_based_facts(doc_id, version_id, card, quality, innovation, citation_map, selected_nodes, node_by_id, warnings)
    else:
        warnings.append("llm_disabled")
        facts = _rule_based_facts(doc_id, version_id, card, quality, innovation, citation_map, selected_nodes, node_by_id, warnings)

    facts = _merge_table_facts(doc_id, version_id, facts, table_content, table_summaries, node_by_id)
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


def get_claims(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, "claims.json", version_id=version_id)["content"]


def get_entities(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, "entities.json", version_id=version_id)["content"]


def get_relations(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, "relations.json", version_id=version_id)["content"]


def get_fact_graph(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    return get_artifact(db_path, doc_id, "fact_graph.json", version_id=version_id)["content"]


def fact_search(
    db_path: Path,
    query: str,
    *,
    doc_ids: Optional[List[str]] = None,
    fact_type: Optional[str] = None,
    source: str = "all",
    min_confidence: float = 0.0,
    top_k: int = 20,
) -> Dict[str, Any]:
    fact_kind = (fact_type or "").strip().lower()
    if fact_kind and fact_kind not in {"claim", "entity", "relation"}:
        raise ValueError("fact type must be one of: claim, entity, relation")
    source_filter = (source or "all").strip().lower()
    if source_filter not in {"all", "text", "table"}:
        raise ValueError("source must be one of: all, text, table")
    terms = _query_terms(query)
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        items: List[Dict[str, Any]] = []
        if fact_kind in {"", "claim"}:
            items.extend(_search_claim_rows(conn, terms, doc_ids, source_filter, min_confidence))
        if fact_kind in {"", "entity"}:
            items.extend(_search_entity_rows(conn, terms, doc_ids, source_filter, min_confidence))
        if fact_kind in {"", "relation"}:
            items.extend(_search_relation_rows(conn, terms, doc_ids, source_filter, min_confidence))
    finally:
        conn.close()
    ranked = sorted(items, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("fact_id") or "")))[: max(1, top_k)]
    return {
        "schema": "fact_search.v1",
        "query": query,
        "type": fact_kind or "all",
        "source": source_filter,
        "min_confidence": min_confidence,
        "doc_ids": doc_ids or [],
        "top_k": top_k,
        "count": len(ranked),
        "items": ranked,
    }


def fact_coverage_summary(db_path: Path, *, doc_id: Optional[str] = None) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        counts = db.paper_fact_counts(conn, doc_id=doc_id)
        source_counts = _fact_source_counts(conn, doc_id=doc_id)
    finally:
        conn.close()
    total = counts["claim_count"] + counts["entity_count"] + counts["relation_count"]
    return {
        "schema": "fact_coverage.v1",
        "doc_id": doc_id or "",
        "total_fact_count": total,
        **counts,
        **source_counts,
        "table_backed_fact_rate": round(source_counts["table_backed_fact_count"] / max(1, total), 4),
    }


def fact_summary_for_doc(db_path: Path, doc_id: str) -> Dict[str, Any]:
    try:
        claims = get_claims(db_path, doc_id)
        entities = get_entities(db_path, doc_id)
        relations = get_relations(db_path, doc_id)
    except (FileNotFoundError, KeyError, ValueError):
        return {"schema": "fact_summary.v1", "doc_id": doc_id, "available": False}
    table_claims = [
        item
        for item in (claims.get("claims") or [])
        if isinstance(item, dict) and _is_table_source(str(item.get("source") or ""))
    ]
    table_entities = [
        item
        for item in (entities.get("entities") or [])
        if isinstance(item, dict) and _is_table_source(str(item.get("source") or ""))
    ]
    table_relations = [
        item
        for item in (relations.get("relations") or [])
        if isinstance(item, dict) and _is_table_source(str(item.get("source") or ""))
    ]
    return {
        "schema": "fact_summary.v1",
        "doc_id": doc_id,
        "available": True,
        "claim_count": int(claims.get("count") or 0),
        "entity_count": int(entities.get("count") or 0),
        "relation_count": int(relations.get("count") or 0),
        "table_backed_fact_count": len(table_claims) + len(table_entities) + len(table_relations),
        "table_claim_count": len(table_claims),
        "table_entity_count": len(table_entities),
        "table_relation_count": len(table_relations),
        "top_claims": [
            {
                "claim_id": item.get("claim_id"),
                "type": item.get("type"),
                "text": _excerpt(str(item.get("text") or ""), 180),
            }
            for item in (claims.get("claims") or [])[:5]
            if isinstance(item, dict)
        ],
        "top_entities": [
            {
                "entity_id": item.get("entity_id"),
                "type": item.get("type"),
                "name": item.get("name"),
            }
            for item in (entities.get("entities") or [])[:8]
            if isinstance(item, dict)
        ],
        "top_table_entities": [
            {
                "entity_id": item.get("entity_id"),
                "type": item.get("type"),
                "name": item.get("name"),
                "confidence": item.get("confidence"),
            }
            for item in table_entities[:8]
        ],
        "top_table_relations": [
            {
                "relation_id": item.get("relation_id"),
                "type": item.get("type"),
                "text": _excerpt(str(item.get("text") or ""), 180),
                "confidence": item.get("confidence"),
            }
            for item in table_relations[:8]
        ],
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
    system_prompt = (
        "你是严谨的论文事实层抽取助手。只能基于给定节点、创新点、引用和版面工件抽取，"
        "每个事实必须绑定 evidence 节点，不能编造。必须返回 JSON object，不要返回 Markdown。"
    )
    user_prompt = "\n".join(
        [
            "请抽取 claims、entities、relations。返回格式：",
            '{"claims":[{"type":"","text":"","evidence":[],"confidence":0.0}],'
            '"entities":[{"type":"","name":"","aliases":[],"evidence":[],"confidence":0.0}],'
            '"relations":[{"type":"","subject":"","object":"","evidence":[],"confidence":0.0}],'
            '"warnings":[]}',
            "",
            f"title: {card.get('title')}",
            f"abstract: {_excerpt(str(card.get('abstract') or card.get('description') or ''), 800)}",
            f"parse_quality: {quality}",
            f"innovation_items: {innovation.get('items') or []}",
            f"citation_relation_count: {len(citation_map.get('relations') or [])}",
            f"layout_block_count: {(layout or {}).get('count') if isinstance(layout, dict) else 0}",
            f"table_count: {(tables or {}).get('count') if isinstance(tables, dict) else 0}",
            f"table_summaries: {(table_summaries or {}).get('table_summaries') if isinstance(table_summaries, dict) else []}",
            f"figure_count: {(figures or {}).get('count') if isinstance(figures, dict) else 0}",
            "",
            "候选证据节点：",
            *_format_nodes_for_prompt(selected_nodes),
        ]
    )
    return generate_json_object(system_prompt, user_prompt)


def _format_nodes_for_prompt(nodes: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for index, node in enumerate(nodes, start=1):
        lines.append(f"[N{index}] node_id: {node.get('node_id')}")
        lines.append(f"node_path: {node.get('node_path')}")
        lines.append(f"page_range: {[node.get('page_start'), node.get('page_end')]}")
        lines.append(f"text: {_excerpt(_node_text(node), 900)}")
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
        text = _excerpt(str(item.get("text") or item.get("claim") or ""), 420)
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
        entity = _entity_record(
            doc_id,
            version_id,
            str(item.get("type") or item.get("entity_type") or "term"),
            str(item.get("name") or ""),
            evidence_node,
            source,
            _confidence(item.get("confidence"), 0.7),
            aliases=_string_list(item.get("aliases")),
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
        relation = _relation_record(
            doc_id,
            version_id,
            str(item.get("type") or item.get("relation_type") or "related_to"),
            str(item.get("subject") or item.get("subject_name") or ""),
            str(item.get("object") or item.get("object_name") or ""),
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
    normalized = _dedupe_facts({"claims": claims, "entities": entities, "relations": relation_rows})
    normalized["status"] = "extracted" if status == "extracted" and normalized["claims"] else "partial"
    normalized["source"] = source
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

    normalized = _dedupe_facts({"claims": claims, "entities": entities, "relations": [item for item in relations if item]})
    normalized["status"] = "partial"
    normalized["source"] = "rule"
    normalized["warnings"] = _unique_strings([*warnings, "rule_based_fact_extraction", *_quality_warnings(quality)])
    return normalized


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
    }
    normalized = _dedupe_facts(merged)
    normalized["status"] = facts.get("status") or "partial"
    normalized["source"] = facts.get("source") or "rule"
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
        db.delete_paper_facts(conn, doc_id, version_id)
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
    return {
        "claims": _dedupe_by_key(facts.get("claims") or [], "normalized_text"),
        "entities": _dedupe_by_key(facts.get("entities") or [], "normalized_name"),
        "relations": _dedupe_relations(facts.get("relations") or []),
    }


def _dedupe_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for item in items:
        marker = (item.get(key), item.get("node_id"), item.get("type"))
        if not marker[0] or marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result[:300]


def _dedupe_relations(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for item in items:
        marker = (item.get("relation_type") or item.get("type"), item.get("subject_id") or item.get("subject_name"), item.get("object_id") or item.get("object_name"), item.get("node_id"))
        if not marker[0] or marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result[:500]


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


def _query_terms(query: str) -> List[str]:
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{5,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    return _unique_strings(terms)[:12] or [query]


def _search_claim_rows(
    conn,
    terms: List[str],
    doc_ids: Optional[List[str]],
    source: str,
    min_confidence: float,
) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    conditions, params = _like_conditions("text", terms)
    doc_filter = _doc_filter(doc_ids, params)
    source_filter = _source_filter(source, params)
    confidence_filter = _confidence_filter(min_confidence, params)
    rows = conn.execute(
        f"""
        SELECT claim_id AS fact_id, 'claim' AS fact_type, doc_id, version_id, node_id,
               claim_type AS type, text, page_range, confidence, source, evidence_json
        FROM paper_claims
        WHERE ({conditions}) {doc_filter} {source_filter} {confidence_filter}
        LIMIT 200
        """,
        params,
    ).fetchall()
    return [_fact_row(dict(row), terms) for row in rows]


def _search_entity_rows(
    conn,
    terms: List[str],
    doc_ids: Optional[List[str]],
    source: str,
    min_confidence: float,
) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    conditions, params = _like_conditions("name", terms)
    doc_filter = _doc_filter(doc_ids, params)
    source_filter = _source_filter(source, params)
    confidence_filter = _confidence_filter(min_confidence, params)
    rows = conn.execute(
        f"""
        SELECT entity_id AS fact_id, 'entity' AS fact_type, doc_id, version_id, node_id,
               entity_type AS type, name AS text, page_range, confidence, source, evidence_json
        FROM paper_entities
        WHERE ({conditions}) {doc_filter} {source_filter} {confidence_filter}
        LIMIT 200
        """,
        params,
    ).fetchall()
    return [_fact_row(dict(row), terms) for row in rows]


def _search_relation_rows(
    conn,
    terms: List[str],
    doc_ids: Optional[List[str]],
    source: str,
    min_confidence: float,
) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    conditions, params = _like_conditions("text || ' ' || subject_name || ' ' || object_name", terms)
    doc_filter = _doc_filter(doc_ids, params)
    source_filter = _source_filter(source, params)
    confidence_filter = _confidence_filter(min_confidence, params)
    rows = conn.execute(
        f"""
        SELECT relation_id AS fact_id, 'relation' AS fact_type, doc_id, version_id, node_id,
               relation_type AS type, text, subject_name, object_name, page_range,
               confidence, source, evidence_json
        FROM paper_relations
        WHERE ({conditions}) {doc_filter} {source_filter} {confidence_filter}
        LIMIT 200
        """,
        params,
    ).fetchall()
    return [_fact_row(dict(row), terms) for row in rows]


def _like_conditions(column_sql: str, terms: List[str]) -> tuple[str, List[Any]]:
    conditions = []
    params: List[Any] = []
    for term in terms:
        conditions.append(f"{column_sql} LIKE ?")
        params.append(f"%{term}%")
    return " OR ".join(conditions) if conditions else "1 = 0", params


def _doc_filter(doc_ids: Optional[List[str]], params: List[Any]) -> str:
    clean = _unique_strings(doc_ids or [])
    if not clean:
        return ""
    placeholders = ",".join("?" for _ in clean)
    params.extend(clean)
    return f"AND doc_id IN ({placeholders})"


def _source_filter(source: str, params: List[Any]) -> str:
    if source == "table":
        params.append("%table%")
        return "AND source LIKE ?"
    if source == "text":
        params.append("%table%")
        return "AND source NOT LIKE ?"
    return ""


def _confidence_filter(min_confidence: float, params: List[Any]) -> str:
    try:
        threshold = float(min_confidence)
    except (TypeError, ValueError):
        threshold = 0.0
    if threshold <= 0:
        return ""
    params.append(threshold)
    return "AND confidence >= ?"


def _fact_row(row: Dict[str, Any], terms: List[str]) -> Dict[str, Any]:
    text = compact_whitespace(str(row.get("text") or ""))
    haystack = text + " " + str(row.get("subject_name") or "") + " " + str(row.get("object_name") or "")
    score = sum(1 for term in terms if term and term in haystack)
    return {
        "fact_id": row.get("fact_id") or "",
        "fact_type": row.get("fact_type") or "",
        "doc_id": row.get("doc_id") or "",
        "version_id": row.get("version_id") or "",
        "node_id": row.get("node_id") or "",
        "type": row.get("type") or "",
        "text": _excerpt(text, 240),
        "subject_name": row.get("subject_name") or "",
        "object_name": row.get("object_name") or "",
        "page_range": _json_value(row.get("page_range"), []),
        "confidence": float(row.get("confidence") or 0.0),
        "source": row.get("source") or "",
        "source_kind": "table" if _is_table_source(str(row.get("source") or "")) else "text",
        "evidence": _json_value(row.get("evidence_json"), {}),
        "score": score,
    }


def _fact_source_counts(conn, *, doc_id: Optional[str] = None) -> Dict[str, int]:  # type: ignore[no-untyped-def]
    params: List[Any] = []
    where = ""
    if doc_id:
        where = "WHERE doc_id = ?"
        params.append(doc_id)
    table_count = 0
    text_count = 0
    for table in ("paper_claims", "paper_entities", "paper_relations"):
        prefix = f"{where} AND" if where else "WHERE"
        table_row = conn.execute(
            f"SELECT COUNT(*) AS count FROM {table} {prefix} source LIKE ?",
            [*params, "%table%"],
        ).fetchone()
        text_row = conn.execute(
            f"SELECT COUNT(*) AS count FROM {table} {prefix} source NOT LIKE ?",
            [*params, "%table%"],
        ).fetchone()
        table_count += int(table_row["count"] or 0)
        text_count += int(text_row["count"] or 0)
    return {
        "table_backed_fact_count": table_count,
        "text_backed_fact_count": text_count,
    }


def _is_table_source(source: str) -> bool:
    return "table" in source.lower()


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value or ""))
    except json.JSONDecodeError:
        return default


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", "", compact_whitespace(text).lower())


def _string_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [compact_whitespace(str(item)) for item in value if compact_whitespace(str(item))]
    if isinstance(value, str) and value.strip():
        return [compact_whitespace(value)]
    return []


def _confidence(value: object, default: float) -> float:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        score = default
    return round(max(0.0, min(1.0, score)), 3)


def _excerpt(text: str, max_chars: int) -> str:
    cleaned = compact_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + " ..."


def _unique_strings(values: Iterable[Any]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = compact_whitespace(str(value))
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
