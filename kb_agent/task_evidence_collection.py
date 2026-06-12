from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .answer_plan import task_semantic_score_adjustment
from .task_evidence import compact_section_evidence, dedupe_evidence, section_evidence_quality
from .utils import unique_strings


class SearchEvidenceFn(Protocol):
    def __call__(
        self,
        db_path: Path,
        doc_id: str,
        query: str,
        top_k: int,
        search_mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        ...


DIMENSION_FRAME_TYPES = {
    "problem_setting": {"problem", "claim"},
    "method_paradigm": {"method", "claim"},
    "evaluation_protocol": {"result", "citation"},
    "innovation_overlap": {"method", "claim", "result"},
    "limitations": {"limitation"},
    "evidence_strength": {"result", "citation"},
}

SECTION_FRAME_TYPES = {
    "background_problem": {"problem", "claim"},
    "method_paradigms": {"method", "claim"},
    "coordination_mechanisms": {"method", "claim"},
    "evaluation_evidence": {"result", "citation"},
    "limitations_future": {"limitation"},
}


def collect_dimension_evidence(
    db_path: Path,
    query: str,
    contexts: List[Dict[str, Any]],
    dimensions: List[Dict[str, Any]],
    search_mode: str,
    *,
    search_evidence_fn: Optional[SearchEvidenceFn] = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    searcher = search_evidence_fn or search_doc_evidence
    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for dimension in dimensions:
        dimension_id = str(dimension["id"])
        result[dimension_id] = {}
        terms = " ".join(str(term) for term in dimension["search_terms"])
        search_query = f"{query} {terms}"
        for context in contexts:
            frame_evidence = claim_frame_evidence_for_dimension(context, dimension_id, search_query, limit=3)
            searched = searcher(db_path, context["doc_id"], search_query, top_k=4, search_mode=search_mode)
            evidence = dedupe_evidence([*frame_evidence, *searched])[:4]
            if not evidence:
                evidence = innovation_evidence_for_dimension(context, dimension_id)[:3]
            result[dimension_id][context["doc_id"]] = evidence[:4]
    return result


def collect_section_evidence(
    db_path: Path,
    topic: str,
    contexts: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    search_mode: str,
    *,
    search_evidence_fn: Optional[SearchEvidenceFn] = None,
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    searcher = search_evidence_fn or search_doc_evidence
    result: Dict[str, List[Dict[str, Any]]] = {}
    by_section: Dict[str, Dict[str, Any]] = {}
    for section in sections:
        section_id = str(section["section_id"])
        terms = " ".join(str(term) for term in section["search_terms"])
        search_query = f"{topic} {terms}"
        evidence: List[Dict[str, Any]] = []
        for context in contexts:
            evidence.extend(claim_frame_evidence_for_section(context, section_id, search_query, limit=3))
            evidence.extend(searcher(db_path, context["doc_id"], search_query, top_k=3, search_mode=search_mode))
        compacted, report = compact_section_evidence(evidence, max_items=12)
        result[section_id] = compacted
        by_section[section_id] = report
    quality = section_evidence_quality(by_section, result)
    return result, quality


def search_doc_evidence(db_path: Path, doc_id: str, query: str, top_k: int, search_mode: str = "hybrid") -> List[Dict[str, Any]]:
    if search_mode == "tree":
        tree_search_module = importlib.import_module("kb_agent.tree_search")
        trace = tree_search_module.tree_search(db_path, doc_id, query, budget=top_k, use_llm=False, search_mode="hybrid")
        return dedupe_evidence(list(trace.get("evidence") or []))

    search_module = importlib.import_module("kb_agent.search")
    results = search_module.search_nodes(db_path, query, doc_id=doc_id, top_k=top_k, search_mode=search_mode)
    packets = []
    for result in results:
        packets.extend(packet.to_dict() for packet in search_module.get_evidence(db_path, result.doc_id, [result.node_id]))
    return dedupe_evidence(packets)


def claim_frame_evidence_for_dimension(
    context: Dict[str, Any],
    dimension_id: str,
    query: str,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    wanted = DIMENSION_FRAME_TYPES.get(dimension_id, {"claim"})
    return claim_frame_evidence(context, query, wanted_types=wanted, limit=limit)


def claim_frame_evidence_for_section(
    context: Dict[str, Any],
    section_id: str,
    query: str,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    wanted = SECTION_FRAME_TYPES.get(section_id, {"claim", "method", "result"})
    return claim_frame_evidence(context, query, wanted_types=wanted, limit=limit)


def claim_frame_evidence(
    context: Dict[str, Any],
    query: str,
    *,
    wanted_types: set[str],
    limit: int,
) -> List[Dict[str, Any]]:
    frames = (context.get("claim_frames") or {}).get("frames") or []
    units = (context.get("evidence_units") or {}).get("units") or []
    unit_by_id = {str(unit.get("unit_id") or ""): unit for unit in units if isinstance(unit, dict)}
    terms = _task_query_terms(query)
    candidates = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        claim_type = str(frame.get("claim_type") or "")
        if claim_type not in wanted_types:
            continue
        score = _task_frame_score(frame, terms)
        if score <= 0 and terms:
            continue
        candidates.append((score, frame))
    candidates.sort(key=lambda item: (-float(item[0]), -float(item[1].get("confidence") or 0.0), str(item[1].get("frame_id") or "")))
    evidence = []
    for _, frame in candidates[:limit]:
        evidence.append(_frame_to_evidence_item(context, frame, unit_by_id))
    return dedupe_evidence(evidence)


def innovation_evidence_for_dimension(context: Dict[str, Any], dimension_id: str) -> List[Dict[str, Any]]:
    result = []
    for item in context.get("innovation", {}).get("items", []):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if dimension_id == "limitations" and item_type != "limitation":
            continue
        if dimension_id == "evaluation_protocol" and item_type != "result":
            continue
        for evidence in item.get("evidence") or []:
            if isinstance(evidence, dict):
                enriched = dict(evidence)
                enriched.setdefault("doc_id", context["doc_id"])
                enriched.setdefault("title", context["title"])
                enriched.setdefault("path", context["path"])
                result.append(enriched)
    return dedupe_evidence(result)


def _frame_to_evidence_item(
    context: Dict[str, Any],
    frame: Dict[str, Any],
    unit_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    unit_ids = [str(item) for item in frame.get("evidence_unit_ids") or [] if str(item)]
    unit = next((unit_by_id[unit_id] for unit_id in unit_ids if unit_id in unit_by_id), {})
    return {
        "doc_id": context["doc_id"],
        "title": context.get("title") or context["doc_id"],
        "path": context.get("path") or "",
        "node_id": unit.get("node_id") or "",
        "node_path": unit.get("node_path") or "",
        "page_range": unit.get("page_range") or [],
        "excerpt": frame.get("short_claim") or unit.get("summary") or "",
        "summary": frame.get("short_claim") or unit.get("summary") or "",
        "evidence_type": frame.get("claim_type") or unit.get("unit_type") or "claim_frame",
        "confidence": frame.get("confidence", 0.0),
        "claim_frame_id": frame.get("frame_id") or "",
        "evidence_unit_ids": unit_ids,
        "semantic_support_status": frame.get("semantic_support_status") or "not_checked",
        "semantic_support_score": frame.get("semantic_support_score", 0.0),
        "semantic_support_reason": frame.get("semantic_support_reason") or "",
        "citation_risk": frame.get("citation_risk") or "not_checked",
        "primary_evidence_unit_ids": frame.get("primary_evidence_unit_ids") or [],
        "weak_evidence_unit_ids": frame.get("weak_evidence_unit_ids") or [],
        "contradictory_evidence_unit_ids": frame.get("contradictory_evidence_unit_ids") or [],
        "source": "claim_frame",
    }


def _task_query_terms(query: str) -> List[str]:
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    return unique_strings(terms)[:16]


def _task_frame_score(frame: Dict[str, Any], terms: List[str]) -> float:
    quality_score = _confidence(frame.get("quality_score"), 0.6)
    support_status = str(frame.get("support_status") or "")
    if support_status == "supported":
        support_status = "structurally_supported"
    if quality_score < 0.35 or support_status == "unsupported":
        return 0.0
    semantic_status = str(frame.get("semantic_support_status") or "not_checked")
    if semantic_status == "contradicted":
        return 0.0
    haystack = " ".join(
        str(frame.get(field) or "")
        for field in (
            "short_claim",
            "problem",
            "method",
            "dataset_or_setting",
            "metric_or_signal",
            "result_or_gain",
            "limitation",
        )
    ).lower()
    term_score = sum(1.0 for term in terms if term.lower() in haystack)
    support_score = 0.35 if support_status == "structurally_supported" else (0.15 if frame.get("evidence_unit_ids") else 0.0)
    semantic_score = task_semantic_score_adjustment(semantic_status)
    confidence_score = 0.2 * _confidence(frame.get("confidence"), 0.0)
    quality_boost = 0.25 * quality_score
    return max(0.0, term_score + support_score + semantic_score + confidence_score + quality_boost)


def _confidence(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))
