from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .artifacts import get_citation_map, get_doc_card, get_innovations, get_parse_quality
from .claim_frames import claim_frame_summary_for_doc, extract_claim_frames, get_claim_frames, get_evidence_units
from .facts import fact_summary_for_doc
from .insights import extract_doc_insights
from .search import search_documents
from .utils import unique_strings


def select_papers(
    db_path: Path,
    query: str,
    doc_ids: Optional[List[str]],
    top_k_docs: int,
    search_mode: str,
) -> List[Dict[str, Any]]:
    if doc_ids:
        return [{"doc_id": doc_id, "score": None, "node_matches": None} for doc_id in unique_strings(doc_ids)]
    route_mode = "hybrid" if search_mode == "tree" else search_mode
    return search_documents(db_path, query, top_k=max(1, top_k_docs), search_mode=route_mode)


def prepare_paper_contexts(db_path: Path, selected: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    contexts = []
    warnings: List[str] = []
    for item in selected:
        doc_id = str(item.get("doc_id") or "")
        if not doc_id:
            continue
        try:
            card = get_doc_card(db_path, doc_id)
            quality = get_parse_quality(db_path, doc_id)
            innovation, insight_warnings = read_or_extract_insights(db_path, doc_id)
            citation_map = get_citation_map(db_path, doc_id)
            facts = fact_summary_for_doc(db_path, doc_id)
            claim_frames, evidence_units, claim_frame_warnings = read_or_extract_claim_frame_context(db_path, doc_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            warnings.append(f"paper_prepare_failed:{doc_id}:{exc}")
            continue
        warnings.extend([*insight_warnings, *claim_frame_warnings])
        contexts.append(
            {
                "doc_id": doc_id,
                "title": card.get("title") or doc_id,
                "path": card.get("path") or "",
                "abstract": card.get("abstract") or "",
                "description": card.get("description") or card.get("summary") or "",
                "keywords": card.get("keywords") or [],
                "quality": quality,
                "innovation": innovation,
                "citation_map": citation_map,
                "facts": facts,
                "claim_frames": claim_frames,
                "evidence_units": evidence_units,
                "route_score": item.get("score"),
                "node_matches": item.get("node_matches"),
            }
        )
    return contexts, unique_strings(warnings)


def read_or_extract_insights(db_path: Path, doc_id: str) -> tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    try:
        innovation = get_innovations(db_path, doc_id)
        citation_map = get_citation_map(db_path, doc_id)
    except (FileNotFoundError, KeyError, ValueError):
        innovation = {}
        citation_map = {}
    if innovation.get("schema") == "innovation.v1" and citation_map.get("schema") == "citation_map.v1":
        return innovation, warnings
    result = extract_doc_insights(db_path, doc_id, force=True, use_llm=False)
    warnings.append(f"insights_rule_refreshed:{doc_id}")
    return result["innovation"], warnings


def read_or_extract_claim_frame_context(db_path: Path, doc_id: str) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    warnings: List[str] = []
    try:
        frames = get_claim_frames(db_path, doc_id)
        units = get_evidence_units(db_path, doc_id)
        summary = claim_frame_summary_for_doc(db_path, doc_id)
        if frames.get("schema") == "claim_frames.v1" and units.get("schema") == "evidence_units.v1":
            return {**frames, "summary": summary}, units, warnings
    except (FileNotFoundError, KeyError, ValueError):
        pass
    try:
        result = extract_claim_frames(db_path, doc_id, force=True, use_llm=False, require_llm=False)
        warnings.append(f"claim_frames_rule_refreshed:{doc_id}")
        return result["claim_frames"], get_evidence_units(db_path, doc_id), warnings
    except Exception as exc:
        warnings.append(f"claim_frames_unavailable:{doc_id}:{exc}")
        return {"schema": "claim_frames.v1", "status": "skipped", "frames": [], "summary": {"available": False}}, {"units": []}, warnings
