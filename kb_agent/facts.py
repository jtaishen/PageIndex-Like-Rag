from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .artifacts import get_artifact, get_doc_card, get_parse_quality, list_artifacts
from .fact_artifacts import (
    build_fact_artifacts,
    read_existing_facts,
    replace_fact_rows,
    write_fact_artifacts,
)
from .fact_llm import extract_facts_with_llm_batches
from .fact_queries import (
    fact_coverage_summary,
    fact_search,
    fact_summary_for_doc,
    get_claims,
    get_entities,
    get_fact_graph,
    get_relations,
)
from .fact_records import dedupe_facts
from .fact_sources import (
    merge_citation_relations,
    merge_table_facts,
    node_map,
    rule_based_facts,
    select_fact_nodes,
)
from .insights import extract_doc_insights
from .llm import LLMError, generate_json_object
from .utils import unique_strings as _unique_strings, write_json


FACT_ARTIFACTS = {
    "claims.json",
    "entities.json",
    "relations.json",
    "fact_graph.json",
    "fact_report.json",
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
    existing = read_existing_facts(db_path, doc_id)
    if existing and not force:
        return {
            "schema": "fact_extraction_result.v1",
            "doc_id": doc_id,
            "version_id": version_id,
            "artifact_dir": str(artifact_dir),
            "skipped": True,
            **existing,
        }

    inputs = load_fact_extraction_inputs(db_path, doc_id, listing=listing)
    card = inputs["card"]
    quality = inputs["quality"]
    table_summaries = inputs["table_summaries"]
    innovation = inputs["innovation"]
    citation_map = inputs["citation_map"]
    node_by_id = inputs["node_by_id"]
    selected_nodes = inputs["selected_nodes"]
    warnings = [*inputs["warnings"]]
    llm_error = ""

    if use_llm:
        try:
            facts = extract_facts_with_llm_batches(
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
                json_generator=generate_json_object,
            )
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            warnings.append(f"llm_unavailable:{llm_error}")
            facts = rule_based_facts(doc_id, version_id, card, quality, innovation, citation_map, selected_nodes, node_by_id, warnings)
            facts["llm_batch_report"] = {
                "schema": "llm_fact_batch_report.v1",
                "llm_mode": "batch_json",
                "batch_count": int(exc.metadata.get("batch_count") or 0),
                "batch_success_count": int(exc.metadata.get("batch_success_count") or 0),
                "batch_timeout_count": int(exc.metadata.get("batch_timeout_count") or 0),
                "batch_fallback_count": int(exc.metadata.get("batch_fallback_count") or 0),
                "llm_batch_warnings": [exc.error_type],
                "success_rate": 0.0,
            }
    else:
        warnings.append("llm_disabled")
        facts = rule_based_facts(doc_id, version_id, card, quality, innovation, citation_map, selected_nodes, node_by_id, warnings)

    return persist_fact_result(db_path, inputs, facts, llm_error=llm_error)


def load_fact_extraction_inputs(
    db_path: Path,
    doc_id: str,
    *,
    listing: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_listing = listing or list_artifacts(db_path, doc_id)
    nodes = _artifact_content(db_path, doc_id, "node_index.jsonl", [])
    innovation, citation_map, insight_warnings = _read_or_extract_insight_artifacts(db_path, doc_id)
    return {
        "doc_id": doc_id,
        "version_id": str(resolved_listing["version_id"]),
        "artifact_dir": Path(str(resolved_listing["artifact_dir"])),
        "card": get_doc_card(db_path, doc_id),
        "quality": get_parse_quality(db_path, doc_id),
        "nodes": nodes,
        "table_content": _artifact_content(db_path, doc_id, "table_content.json", {}),
        "table_summaries": _artifact_content(db_path, doc_id, "table_summaries.json", {}),
        "innovation": innovation,
        "citation_map": citation_map,
        "node_by_id": node_map(nodes),
        "selected_nodes": select_fact_nodes(nodes, innovation, citation_map),
        "warnings": [*insight_warnings],
    }


def persist_fact_result(
    db_path: Path,
    inputs: Dict[str, Any],
    facts: Dict[str, Any],
    *,
    llm_error: str = "",
) -> Dict[str, Any]:
    doc_id = str(inputs["doc_id"])
    version_id = str(inputs["version_id"])
    artifact_dir = Path(inputs["artifact_dir"])
    card = inputs["card"]
    quality = inputs["quality"]
    citation_map = inputs["citation_map"]
    node_by_id = inputs["node_by_id"]
    facts = merge_citation_relations(doc_id, version_id, card, facts, citation_map, node_by_id)
    facts = merge_table_facts(
        doc_id,
        version_id,
        facts,
        inputs["table_content"],
        inputs["table_summaries"],
        node_by_id,
    )
    facts = dedupe_facts(facts)
    artifacts = build_fact_artifacts(doc_id, version_id, card, quality, facts, llm_error)
    write_fact_artifacts(artifact_dir, artifacts)
    replace_fact_rows(db_path, doc_id, version_id, facts)
    claim_frame_result: Dict[str, Any] = {}
    try:
        from .claim_frames import extract_claim_frames, extract_evidence_units

        evidence_unit_result = extract_evidence_units(db_path, doc_id, force=True)
        claim_frame_result = extract_claim_frames(db_path, doc_id, force=True, use_llm=False, require_llm=False)
        evidence_units = evidence_unit_result.get("evidence_units") or {}
        artifacts["fact_report"]["evidence_unit_count"] = evidence_units.get("count", 0)
        artifacts["fact_report"]["source_kind_counts"] = evidence_units.get("source_kind_counts") or {}
        artifacts["fact_report"]["claim_frame_count"] = (claim_frame_result.get("claim_frames") or {}).get("count", 0)
        verifier = claim_frame_result.get("verifier") or {}
        artifacts["fact_report"]["verified_frame_rate"] = verifier.get("verified_frame_rate", 0.0)
        artifacts["fact_report"]["unsupported_frame_count"] = verifier.get("unsupported_frame_count", 0)
        artifacts["fact_report"]["trace_status_counts"] = verifier.get("trace_status_counts") or {}
        artifacts["fact_report"]["support_status_counts"] = verifier.get("support_status_counts") or {}
        artifacts["fact_report"]["semantic_support_status_counts"] = verifier.get("semantic_support_status_counts") or {}
        artifacts["fact_report"]["semantic_verified_frame_count"] = verifier.get("semantic_verified_frame_count", 0)
        artifacts["fact_report"]["semantic_supported_frame_rate"] = verifier.get("semantic_supported_frame_rate", 0.0)
        artifacts["fact_report"]["partial_supported_frame_count"] = verifier.get("partial_supported_frame_count", 0)
        artifacts["fact_report"]["related_only_frame_count"] = verifier.get("related_only_frame_count", 0)
        artifacts["fact_report"]["contradicted_frame_count"] = verifier.get("contradicted_frame_count", 0)
        artifacts["fact_report"]["insufficient_evidence_frame_count"] = verifier.get("insufficient_evidence_frame_count", 0)
        artifacts["fact_report"]["citation_risk_counts"] = verifier.get("citation_risk_counts") or {}
        artifacts["fact_report"]["missing_evidence_unit_count"] = verifier.get("missing_evidence_unit_count", 0)
        artifacts["fact_report"]["missing_node_count"] = verifier.get("missing_node_count", 0)
        artifacts["fact_report"]["missing_source_count"] = verifier.get("missing_source_count", 0)
        artifacts["fact_report"]["low_quality_frame_count"] = verifier.get("low_quality_frame_count", 0)
        artifacts["fact_report"]["noisy_frame_count"] = verifier.get("noisy_frame_count", 0)
        artifacts["fact_report"]["ignored_noise_frame_count"] = verifier.get("ignored_noise_frame_count", 0)
        artifacts["fact_report"]["top_frame_noise_reasons"] = verifier.get("top_frame_noise_reasons", [])
        write_json(artifact_dir / "fact_report.json", artifacts["fact_report"])
    except Exception as exc:
        artifacts["fact_report"]["warnings"] = _unique_strings(
            [*(artifacts["fact_report"].get("warnings") or []), f"claim_frame_artifacts_failed:{exc}"]
        )
        write_json(artifact_dir / "fact_report.json", artifacts["fact_report"])
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
        "claim_frame_result": claim_frame_result,
        "llm_error": llm_error,
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
