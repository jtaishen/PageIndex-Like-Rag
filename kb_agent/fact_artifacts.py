from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import db
from .artifacts import get_artifact
from .fact_queries import get_claims, get_entities, get_fact_graph, get_relations
from .fact_records import graph_edges, graph_nodes
from .fact_utils import is_table_source as _is_table_source
from .utils import unique_strings as _unique_strings, write_json


def read_existing_facts(db_path: Path, doc_id: str) -> Optional[Dict[str, Any]]:
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


def build_fact_artifacts(
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
        "nodes": graph_nodes(claims, entities),
        "edges": graph_edges(relations),
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


def write_fact_artifacts(artifact_dir: Path, artifacts: Dict[str, Any]) -> None:
    write_json(artifact_dir / "claims.json", artifacts["claims"])
    write_json(artifact_dir / "entities.json", artifacts["entities"])
    write_json(artifact_dir / "relations.json", artifacts["relations"])
    write_json(artifact_dir / "fact_graph.json", artifacts["fact_graph"])
    write_json(artifact_dir / "fact_report.json", artifacts["fact_report"])


def replace_fact_rows(db_path: Path, doc_id: str, version_id: str, facts: Dict[str, Any]) -> None:
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
