from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import llm_fact_batch_size, llm_fact_max_nodes
from .fact_artifacts import read_existing_facts
from .fact_llm import extract_facts_batch_with_llm, merge_fact_parts, node_batches, normalize_fact_payload
from .facts import load_fact_extraction_inputs, persist_fact_result
from .llm import LLMError
from .task_artifacts import (
    get_task_artifact,
    new_task_id,
    task_manifest,
    task_state_root,
    update_task_status,
)
from .utils import unique_strings, write_json
from .workflow_state import (
    create_workflow_state,
    finish_workflow_step,
    get_workflow_state,
    set_workflow_phase,
    single_request_json_generator,
    start_workflow_step,
    workflow_step,
    workflow_steps_for_phase,
)


JsonGenerator = Callable[[str, str], Dict[str, object]]


def prepare_fact_extraction_workflow(
    db_path: Path,
    doc_id: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    existing = read_existing_facts(db_path, doc_id)
    if existing and not force:
        return {
            "schema": "staged_fact_prepare_result.v1",
            "task_id": "",
            "doc_id": doc_id,
            "status": "already_available",
            "fact_report": existing.get("fact_report") or {},
        }

    inputs = load_fact_extraction_inputs(db_path, doc_id)
    max_nodes = max(1, llm_fact_max_nodes())
    batch_size = max(1, llm_fact_batch_size())
    selected_nodes = inputs["selected_nodes"][:max_nodes]
    batches = node_batches(selected_nodes, batch_size)
    if not batches:
        return {
            "schema": "staged_fact_prepare_result.v1",
            "task_id": "",
            "doc_id": doc_id,
            "status": "no_candidate_nodes",
            "warnings": ["no_candidate_fact_nodes"],
        }

    task_id = new_task_id("fact_extraction", doc_id, [doc_id])
    task_dir = task_state_root(db_path) / task_id
    plan_batches = []
    steps = []
    for index, batch in enumerate(batches, start=1):
        batch_id = f"batch_{index:03d}"
        node_ids = [str(node.get("node_id") or "") for node in batch if node.get("node_id")]
        plan_batches.append({"batch_id": batch_id, "node_ids": node_ids})
        steps.append(
            {
                "step_id": f"fact:{batch_id}",
                "phase": "fact_batches",
                "artifact": f"fact_batches/{batch_id}.json",
                "metadata": {"batch_id": batch_id, "node_ids": node_ids},
            }
        )
    plan = {
        "schema": "fact_extraction_plan.v1",
        "task_id": task_id,
        "doc_id": doc_id,
        "version_id": inputs["version_id"],
        "batch_size": batch_size,
        "max_nodes": max_nodes,
        "selected_node_count": len(selected_nodes),
        "batch_count": len(plan_batches),
        "batches": plan_batches,
        "created_at": time.time(),
    }
    manifest = task_manifest(task_id, "fact_extraction", doc_id, "prepared", inputs["warnings"])
    write_json(task_dir / "manifest.json", manifest)
    write_json(task_dir / "fact_extraction_plan.json", plan)
    workflow = create_workflow_state(
        db_path,
        task_id,
        "fact_extraction",
        steps,
        phase="fact_batches",
        metadata={"doc_id": doc_id, "version_id": inputs["version_id"], "force": force},
    )
    update_task_status(db_path, task_id, "prepared", warnings=inputs["warnings"])
    return {
        "schema": "staged_fact_prepare_result.v1",
        "task_id": task_id,
        "doc_id": doc_id,
        "version_id": inputs["version_id"],
        "status": "prepared",
        "plan": plan,
        "workflow": workflow,
        "artifact_paths": {
            "manifest": str(task_dir / "manifest.json"),
            "plan": str(task_dir / "fact_extraction_plan.json"),
            "workflow_state": str(task_dir / "workflow_state.json"),
        },
    }


def extract_fact_batch_workflow(
    db_path: Path,
    task_id: str,
    batch_id: str,
    *,
    json_generator: Optional[JsonGenerator] = None,
) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    _require_task_type(state, "fact_extraction")
    step_id = f"fact:{batch_id}"
    workflow_step(state, step_id)
    plan = _task_json(db_path, task_id, "fact_extraction_plan.json")
    batch = _find_batch(plan, batch_id)
    start_workflow_step(db_path, task_id, step_id)
    try:
        inputs = load_fact_extraction_inputs(db_path, str(plan["doc_id"]))
        if str(inputs["version_id"]) != str(plan["version_id"]):
            workflow = finish_workflow_step(
                db_path,
                task_id,
                step_id,
                status="failed",
                error_type="document_version_changed",
            )
            return _failed_result(task_id, batch_id, "document_version_changed", workflow)
        batch_nodes = [
            inputs["node_by_id"][node_id]
            for node_id in batch.get("node_ids") or []
            if node_id in inputs["node_by_id"]
        ]
        if not batch_nodes:
            workflow = finish_workflow_step(
                db_path,
                task_id,
                step_id,
                status="failed",
                error_type="batch_nodes_missing",
            )
            return _failed_result(task_id, batch_id, "batch_nodes_missing", workflow)
        generator = json_generator or single_request_json_generator("fact_extraction", batch_id)
        payload = extract_facts_batch_with_llm(
            inputs["card"],
            inputs["quality"],
            inputs["innovation"],
            inputs["citation_map"],
            batch_nodes,
            inputs["table_summaries"],
            batch_index=_batch_index(plan, batch_id),
            batch_count=int(plan["batch_count"]),
            json_generator=generator,
        )
        normalized = normalize_fact_payload(
            payload,
            doc_id=str(plan["doc_id"]),
            version_id=str(plan["version_id"]),
            card=inputs["card"],
            quality=inputs["quality"],
            node_by_id=inputs["node_by_id"],
            selected_nodes=batch_nodes,
            source="llm",
            status="extracted",
            warnings=[],
        )
        artifact = {
            "schema": "fact_batch_result.v1",
            "task_id": task_id,
            "batch_id": batch_id,
            "doc_id": plan["doc_id"],
            "version_id": plan["version_id"],
            "status": "completed",
            "facts": normalized,
            "created_at": time.time(),
        }
        task_dir = task_state_root(db_path) / task_id
        artifact_path = task_dir / "fact_batches" / f"{batch_id}.json"
        write_json(artifact_path, artifact)
        workflow = finish_workflow_step(db_path, task_id, step_id, status="completed")
        return {
            "schema": "staged_fact_batch_result.v1",
            "task_id": task_id,
            "batch_id": batch_id,
            "status": "completed",
            "claim_count": len(normalized.get("claims") or []),
            "entity_count": len(normalized.get("entities") or []),
            "relation_count": len(normalized.get("relations") or []),
            "warnings": normalized.get("warnings") or [],
            "workflow": workflow,
            "artifact_path": str(artifact_path),
        }
    except LLMError as exc:
        workflow = finish_workflow_step(db_path, task_id, step_id, status="failed", error_type=exc.error_type)
        return _failed_result(task_id, batch_id, exc.error_type, workflow)


def finalize_fact_extraction_workflow(db_path: Path, task_id: str) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    _require_task_type(state, "fact_extraction")
    batch_steps = workflow_steps_for_phase(state, "fact_batches")
    incomplete = [str(step.get("step_id") or "") for step in batch_steps if step.get("status") != "completed"]
    if incomplete:
        return {
            "schema": "staged_fact_finalize_result.v1",
            "task_id": task_id,
            "status": "incomplete",
            "pending_steps": incomplete,
            "workflow": state,
        }

    plan = _task_json(db_path, task_id, "fact_extraction_plan.json")
    parts = []
    for batch in plan.get("batches") or []:
        batch_id = str(batch.get("batch_id") or "")
        artifact = _task_json(db_path, task_id, f"fact_batches/{batch_id}.json")
        parts.append(artifact.get("facts") or {})
    facts = merge_fact_parts(parts)
    facts["status"] = "extracted"
    facts["source"] = "llm"
    facts["warnings"] = unique_strings(facts.get("warnings") or [])
    facts["llm_batch_report"] = {
        "schema": "llm_fact_batch_report.v1",
        "llm_mode": "staged_batch_json",
        "batch_size": int(plan.get("batch_size") or 0),
        "max_nodes": int(plan.get("max_nodes") or 0),
        "selected_node_count": int(plan.get("selected_node_count") or 0),
        "batch_count": len(parts),
        "batch_success_count": len(parts),
        "batch_timeout_count": 0,
        "batch_fallback_count": 0,
        "llm_batch_warnings": [],
        "success_rate": 1.0,
    }
    inputs = load_fact_extraction_inputs(db_path, str(plan["doc_id"]))
    facts["warnings"] = unique_strings([*inputs["warnings"], *facts["warnings"]])
    result = persist_fact_result(db_path, inputs, facts)
    update_task_status(
        db_path,
        task_id,
        str((result.get("fact_report") or {}).get("status") or "completed"),
        warnings=(result.get("fact_report") or {}).get("warnings") or [],
    )
    workflow = set_workflow_phase(db_path, task_id, "completed", status="completed")
    return {
        "schema": "staged_fact_finalize_result.v1",
        "task_id": task_id,
        "doc_id": plan["doc_id"],
        "status": (result.get("fact_report") or {}).get("status") or "completed",
        "fact_report": result.get("fact_report") or {},
        "claim_frame_result": result.get("claim_frame_result") or {},
        "artifact_paths": {
            "claims": result.get("claims_path") or "",
            "entities": result.get("entities_path") or "",
            "relations": result.get("relations_path") or "",
            "fact_graph": result.get("fact_graph_path") or "",
            "fact_report": result.get("fact_report_path") or "",
        },
        "workflow": workflow,
    }


def _task_json(db_path: Path, task_id: str, name: str) -> Dict[str, Any]:
    content = get_task_artifact(db_path, task_id, name)["content"]
    if not isinstance(content, dict):
        raise ValueError(f"Task artifact is not a JSON object: {name}")
    return content


def _find_batch(plan: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    for batch in plan.get("batches") or []:
        if str(batch.get("batch_id") or "") == batch_id:
            return batch
    raise ValueError(f"Unknown fact batch: {batch_id}")


def _batch_index(plan: Dict[str, Any], batch_id: str) -> int:
    for index, batch in enumerate(plan.get("batches") or [], start=1):
        if str(batch.get("batch_id") or "") == batch_id:
            return index
    raise ValueError(f"Unknown fact batch: {batch_id}")


def _require_task_type(state: Dict[str, Any], task_type: str) -> None:
    if state.get("task_type") != task_type:
        raise ValueError(f"Workflow task type must be {task_type}.")


def _failed_result(task_id: str, batch_id: str, error_type: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "staged_fact_batch_result.v1",
        "task_id": task_id,
        "batch_id": batch_id,
        "status": "failed",
        "error_type": error_type,
        "workflow": workflow,
    }
