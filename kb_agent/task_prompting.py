from __future__ import annotations

import json
from typing import Any, Dict, List

from .config import llm_compare_evidence_per_doc
from .utils import compact_whitespace


def format_dimension_contexts_for_prompt(
    contexts: List[Dict[str, Any]],
    dimension_id: str,
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> List[str]:
    limit = max(1, llm_compare_evidence_per_doc())
    lines: List[str] = []
    for context in contexts:
        lines.append(f"doc_id: {context['doc_id']}")
        lines.append(f"title: {excerpt(str(context['title']), 120)}")
        lines.append(f"description: {excerpt(context.get('description', ''), 180)}")
        lines.append(f"innovations: {format_innovations(context.get('innovation', {}), limit=2)}")
        facts = format_fact_summary(context.get("facts", {}), limit=2)
        lines.append(
            "facts: "
            + json.dumps(
                {
                    "available": facts.get("available", False),
                    "claim_count": facts.get("claim_count", 0),
                    "entity_count": facts.get("entity_count", 0),
                    "top_claims": facts.get("top_claims", [])[:2],
                    "top_table_entities": facts.get("top_table_entities", [])[:2],
                },
                ensure_ascii=False,
            )
        )
        lines.append("evidence:")
        for evidence in evidence_by_dimension.get(dimension_id, {}).get(context["doc_id"], [])[:limit]:
            lines.append(format_short_evidence_line(evidence))
        lines.append("")
    return lines


def format_contexts_for_prompt(
    contexts: List[Dict[str, Any]],
    evidence_by_dimension: Dict[str, Dict[str, List[Dict[str, Any]]]],
    dimensions: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    for context in contexts:
        lines.append(f"doc_id: {context['doc_id']}")
        lines.append(f"title: {context['title']}")
        lines.append(f"description: {excerpt(context.get('description', ''), 600)}")
        lines.append(f"keywords: {context.get('keywords', [])}")
        lines.append(f"innovation_items: {format_innovations(context.get('innovation', {}))}")
        lines.append(f"facts: {format_fact_summary(context.get('facts', {}))}")
        for dimension in dimensions:
            dimension_id = str(dimension["id"])
            lines.append(f"dimension: {dimension_id}")
            for evidence in evidence_by_dimension.get(dimension_id, {}).get(context["doc_id"], [])[:3]:
                lines.append(format_evidence_line(evidence))
        lines.append("")
    return lines


def format_papers_for_prompt(contexts: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for context in contexts:
        lines.append(f"- doc_id: {context['doc_id']}")
        lines.append(f"  title: {context['title']}")
        lines.append(f"  description: {excerpt(context.get('description', ''), 500)}")
        lines.append(f"  innovations: {format_innovations(context.get('innovation', {}), limit=4)}")
        lines.append(f"  facts: {format_fact_summary(context.get('facts', {}), limit=4)}")
    return lines


def format_papers_for_review_prompt(contexts: List[Dict[str, Any]], limit: int = 4) -> List[str]:
    lines = []
    for context in contexts[:limit]:
        lines.append(f"- doc_id: {context['doc_id']}")
        lines.append(f"  title: {context['title']}")
        lines.append(f"  description: {excerpt(context.get('description', ''), 240)}")
        lines.append(f"  innovations: {format_innovations(context.get('innovation', {}), limit=2)}")
        facts = format_fact_summary(context.get("facts", {}), limit=2)
        lines.append(
            "  facts: "
            + json.dumps(
                {
                    "available": facts.get("available", False),
                    "claim_count": facts.get("claim_count", 0),
                    "entity_count": facts.get("entity_count", 0),
                    "table_backed_fact_count": facts.get("table_backed_fact_count", 0),
                    "top_claims": facts.get("top_claims", [])[:2],
                    "top_table_entities": facts.get("top_table_entities", [])[:2],
                },
                ensure_ascii=False,
            )
        )
    return lines


def format_section_evidence_for_prompt(section_evidence: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = []
    for section_id, items in section_evidence.items():
        lines.append(f"section_id: {section_id}")
        for evidence in items[:8]:
            lines.append(format_evidence_line(evidence))
        lines.append("")
    return lines


def format_section_evidence_for_review_prompt(section_evidence: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = []
    for section_id, items in section_evidence.items():
        lines.append(f"section_id: {section_id}")
        for evidence in items[:4]:
            lines.append(format_review_evidence_line(evidence))
        lines.append("")
    return lines


def format_evidence_line(evidence: Dict[str, Any]) -> str:
    return (
        f"- node_id={evidence.get('node_id')} doc_id={evidence.get('doc_id')} "
        f"path={evidence.get('node_path')} page={evidence.get('page_range')} "
        f"excerpt={excerpt(str(evidence.get('excerpt') or ''), 360)}"
    )


def format_short_evidence_line(evidence: Dict[str, Any]) -> str:
    return (
        f"- node_id={evidence.get('node_id')} doc_id={evidence.get('doc_id')} "
        f"path={excerpt(str(evidence.get('node_path') or ''), 120)} "
        f"page={evidence.get('page_range')} "
        f"summary={excerpt(str(evidence.get('excerpt') or evidence.get('summary') or ''), 160)} "
        f"confidence={evidence.get('confidence', '')}"
    )


def format_review_evidence_line(evidence: Dict[str, Any]) -> str:
    return (
        f"- node_id={evidence.get('node_id')} doc_id={evidence.get('doc_id')} "
        f"page={evidence.get('page_range')} summary={excerpt(str(evidence.get('excerpt') or ''), 160)}"
    )


def format_innovations(innovation: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    items = []
    for item in (innovation.get("items") or [])[:limit]:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "title": item.get("title") or "",
                "type": item.get("type") or "",
                "claim": excerpt(str(item.get("claim") or ""), 260),
            }
        )
    return items


def format_fact_summary(facts: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
    if not facts or not facts.get("available"):
        return {"available": False}
    return {
        "available": True,
        "claim_count": facts.get("claim_count", 0),
        "entity_count": facts.get("entity_count", 0),
        "relation_count": facts.get("relation_count", 0),
        "table_backed_fact_count": facts.get("table_backed_fact_count", 0),
        "table_entity_count": facts.get("table_entity_count", 0),
        "table_relation_count": facts.get("table_relation_count", 0),
        "top_claims": facts.get("top_claims", [])[:limit],
        "top_entities": facts.get("top_entities", [])[:limit],
        "top_table_entities": facts.get("top_table_entities", [])[:limit],
        "top_table_relations": facts.get("top_table_relations", [])[:limit],
    }


def excerpt(text: str, max_chars: int) -> str:
    clean = compact_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + " ..."
