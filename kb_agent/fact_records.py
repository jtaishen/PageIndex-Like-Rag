from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from .fact_utils import excerpt as _excerpt, normalize_key as _normalize_key
from .utils import compact_whitespace, stable_id, unique_strings as _unique_strings


CLAIM_TOKENS = {
    "problem": ("问题", "挑战", "不足", "难以", "瓶颈"),
    "method": ("方法", "算法", "模型", "框架", "机制", "设计", "构建"),
    "contribution": ("提出", "贡献", "创新", "研究内容", "主要贡献"),
    "result": ("实验", "结果", "优于", "提升", "降低", "验证"),
    "limitation": ("局限", "不足", "限制", "未来工作", "展望"),
}


def claim_record(
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
    page_range = page_range_from_node(node)
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
        "evidence": merge_evidence(evidence_ref(node), extra_evidence),
        "created_at": time.time(),
    }


def entity_record(
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
        "page_range": page_range_from_node(node),
        "confidence": confidence,
        "source": source,
        "evidence": merge_evidence(evidence_ref(node), extra_evidence),
        "created_at": time.time(),
    }


def relation_record(
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
    relation_id = stable_id(
        "rel",
        doc_id,
        version_id,
        relation_type,
        subject_id or subject,
        object_id or obj,
        node_id,
        index,
        length=14,
    )
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
        "page_range": page_range_from_node(node),
        "confidence": confidence,
        "source": source,
        "evidence": merge_evidence(evidence_ref(node), extra_evidence),
        "created_at": time.time(),
    }


def merge_evidence(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base or {})
    if extra:
        for key, value in extra.items():
            if value not in (None, "", [], {}):
                merged[key] = value
    return merged


def source_offsets_dict(node: Dict[str, Any]) -> Dict[str, Any]:
    offsets = node.get("source_offsets") or {}
    if isinstance(offsets, str):
        try:
            parsed = json.loads(offsets)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return offsets if isinstance(offsets, dict) else {}


def evidence_ref(node: Dict[str, Any]) -> Dict[str, Any]:
    source_offsets = source_offsets_dict(node)
    return {
        "doc_id": node.get("doc_id") or "",
        "node_id": node.get("node_id") or "",
        "node_path": node.get("node_path") or "",
        "page_range": page_range_from_node(node),
        "layout_block_id": source_offsets.get("layout_block_id") or "",
        "caption_id": source_offsets.get("caption_id") or "",
    }


def page_range_from_node(node: Dict[str, Any]) -> List[Any]:
    return [node.get("page_start"), node.get("page_end")]


def first_node_by_kind(nodes: List[Dict[str, Any]], kind: str) -> Optional[Dict[str, Any]]:
    for node in nodes:
        if str(node.get("kind") or node.get("type") or "") == kind:
            return node
    return None


def node_text(node: Dict[str, Any]) -> str:
    return compact_whitespace(" ".join(str(node.get(key) or "") for key in ("heading", "summary", "text", "node_path")))


def evidence_node(
    value: Any,
    node_by_id: Dict[str, Dict[str, Any]],
    selected_by_ref: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
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


def dedupe_facts(facts: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
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


def apply_fact_quality_filters(facts: Dict[str, Any], quality_stats: Dict[str, int]) -> None:
    entities = facts.get("entities") or []
    filtered = []
    removed = 0
    for entity in entities:
        if looks_like_noisy_entity_name(str(entity.get("name") or ""), str(entity.get("type") or entity.get("entity_type") or "")):
            removed += 1
            continue
        filtered.append(entity)
    if removed:
        quality_stats["entity_noise_filtered_count"] = quality_stats.get("entity_noise_filtered_count", 0) + removed
    facts["entities"] = filtered


def looks_like_noisy_entity_name(value: str, entity_type: str = "") -> bool:
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


def graph_nodes(claims: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def graph_edges(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def clean_llm_claim_text(value: str, stats: Dict[str, int]) -> str:
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


def clean_llm_entity_name(value: str, stats: Dict[str, int], *, count_noise: bool = True) -> str:
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
    if looks_like_noisy_entity_name(text):
        if count_noise:
            stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    return text


def clean_llm_relation_endpoint(value: str, stats: Dict[str, int]) -> str:
    text = compact_whitespace(value).strip(" ,，.。;；:：()（）[]【】")
    if len(text) < 2:
        stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    if len(text) > 80:
        stats["noise_filtered_count"] = stats.get("noise_filtered_count", 0) + 1
        return ""
    return text


def claim_sentences(text: str) -> List[str]:
    sentences = [compact_whitespace(item) for item in re.split(r"[。！？!?；;]\s*", text) if compact_whitespace(item)]
    result = []
    for sentence in sentences:
        if len(sentence) < 12:
            continue
        if any(token in sentence for tokens in CLAIM_TOKENS.values() for token in tokens):
            result.append(replace_long_text(sentence))
        if len(result) >= 5:
            break
    return result


def classify_claim(text: str) -> str:
    for claim_type, tokens in CLAIM_TOKENS.items():
        if any(token in text for token in tokens):
            return claim_type
    return "claim"


def claim_type(value: str) -> str:
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


def quality_warnings(quality: Dict[str, Any]) -> List[str]:
    warnings = []
    quality_level = str(quality.get("quality_level") or "")
    if quality_level == "weak":
        warnings.append("weak_parse_quality")
    for warning in quality.get("quality_warnings") or []:
        if warning in {"page_only_tree", "weak_layout_blocks", "missing_abstract", "missing_references"}:
            warnings.append(str(warning))
    return warnings


def replace_long_text(value: str) -> str:
    return _excerpt(compact_whitespace(value), 420)


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
    existing["evidence"] = merge_evidence(existing.get("evidence") or {}, candidate.get("evidence") or {})
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
