from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .insights import (
    build_citation_map,
    extract_innovation_with_llm,
    load_insight_extraction_inputs,
    normalize_innovation_payload,
    read_existing_insights,
)
from .llm import LLMError, llm_payload_metadata
from .task_artifacts import (
    get_task_artifact,
    new_task_id,
    task_manifest,
    task_state_root,
    update_task_status,
)
from .utils import compact_whitespace, unique_strings, write_json
from .workflow_state import (
    create_workflow_state,
    finish_workflow_step,
    get_workflow_state,
    set_workflow_phase,
    staged_json_generator,
    start_workflow_step,
    workflow_step,
    workflow_steps_for_phase,
)


JsonGenerator = Callable[[str, str], Dict[str, object]]
INSIGHT_BATCH_SIZE = 6
INSIGHT_MAX_NODES = 18


def prepare_insight_extraction_workflow(
    db_path: Path,
    doc_id: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    existing = read_existing_insights(db_path, doc_id)
    if existing and not force:
        return {
            "schema": "staged_insight_prepare_result.v1",
            "task_id": "",
            "doc_id": doc_id,
            "status": "already_available",
            "innovation": existing["innovation"],
            "citation_map": existing["citation_map"],
        }

    inputs = load_insight_extraction_inputs(db_path, doc_id)
    selected_nodes = inputs["selected_nodes"][:INSIGHT_MAX_NODES]
    batches = [
        selected_nodes[index : index + INSIGHT_BATCH_SIZE]
        for index in range(0, len(selected_nodes), INSIGHT_BATCH_SIZE)
    ]
    if not batches:
        return {
            "schema": "staged_insight_prepare_result.v1",
            "task_id": "",
            "doc_id": doc_id,
            "status": "no_candidate_nodes",
            "warnings": ["no_candidate_insight_nodes"],
        }

    task_id = new_task_id("insight_extraction", doc_id, [doc_id])
    task_dir = task_state_root(db_path) / task_id
    plan_batches = []
    steps = []
    for index, batch in enumerate(batches, start=1):
        batch_id = f"batch_{index:03d}"
        node_ids = [str(node.get("node_id") or "") for node in batch if node.get("node_id")]
        plan_batches.append({"batch_id": batch_id, "node_ids": node_ids})
        steps.append(
            {
                "step_id": f"insight:{batch_id}",
                "phase": "insight_batches",
                "artifact": f"insight_batches/{batch_id}.json",
                "metadata": {"batch_id": batch_id, "node_ids": node_ids},
            }
        )

    plan = {
        "schema": "insight_extraction_plan.v1",
        "task_id": task_id,
        "doc_id": doc_id,
        "version_id": inputs["version_id"],
        "batch_size": INSIGHT_BATCH_SIZE,
        "max_nodes": INSIGHT_MAX_NODES,
        "selected_node_count": len(selected_nodes),
        "batch_count": len(plan_batches),
        "batches": plan_batches,
        "created_at": time.time(),
    }
    quality_warnings = list(inputs["quality"].get("quality_warnings") or [])
    write_json(task_dir / "manifest.json", task_manifest(task_id, "insight_extraction", doc_id, "prepared", quality_warnings))
    write_json(task_dir / "insight_extraction_plan.json", plan)
    workflow = create_workflow_state(
        db_path,
        task_id,
        "insight_extraction",
        steps,
        phase="insight_batches",
        metadata={"doc_id": doc_id, "version_id": inputs["version_id"], "force": force},
    )
    update_task_status(db_path, task_id, "prepared", warnings=quality_warnings)
    return {
        "schema": "staged_insight_prepare_result.v1",
        "task_id": task_id,
        "doc_id": doc_id,
        "version_id": inputs["version_id"],
        "status": "prepared",
        "plan": plan,
        "workflow": workflow,
        "artifact_paths": {
            "manifest": str(task_dir / "manifest.json"),
            "plan": str(task_dir / "insight_extraction_plan.json"),
            "workflow_state": str(task_dir / "workflow_state.json"),
        },
    }


def extract_insight_batch_workflow(
    db_path: Path,
    task_id: str,
    batch_id: str,
    *,
    json_generator: Optional[JsonGenerator] = None,
) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    _require_task_type(state)
    step_id = f"insight:{batch_id}"
    workflow_step(state, step_id)
    plan = _task_json(db_path, task_id, "insight_extraction_plan.json")
    batch = _find_batch(plan, batch_id)
    start_workflow_step(db_path, task_id, step_id)
    try:
        inputs = load_insight_extraction_inputs(db_path, str(plan["doc_id"]))
        if str(inputs["version_id"]) != str(plan["version_id"]):
            workflow = finish_workflow_step(
                db_path,
                task_id,
                step_id,
                status="failed",
                error_type="document_version_changed",
            )
            return _failed_result(task_id, batch_id, "document_version_changed", workflow)
        node_by_id = {str(node.get("node_id") or ""): node for node in inputs["selected_nodes"]}
        batch_nodes = [node_by_id[node_id] for node_id in batch.get("node_ids") or [] if node_id in node_by_id]
        if not batch_nodes:
            workflow = finish_workflow_step(
                db_path,
                task_id,
                step_id,
                status="failed",
                error_type="batch_nodes_missing",
            )
            return _failed_result(task_id, batch_id, "batch_nodes_missing", workflow)

        generator = json_generator or staged_json_generator("insight_extraction", batch_id)
        payload = extract_innovation_with_llm(
            inputs["card"],
            inputs["quality"],
            batch_nodes,
            json_generator=generator,
            stage=batch_id,
        )
        diagnostics = llm_payload_metadata(payload)
        innovation = normalize_innovation_payload(
            payload,
            doc_id=str(plan["doc_id"]),
            version_id=str(plan["version_id"]),
            card=inputs["card"],
            quality=inputs["quality"],
            selected_nodes=batch_nodes,
            status="extracted",
            warnings=[],
        )
        if not innovation.get("items"):
            workflow = finish_workflow_step(
                db_path,
                task_id,
                step_id,
                status="failed",
                error_type="empty_llm_items",
                diagnostics=diagnostics,
            )
            return _failed_result(task_id, batch_id, "empty_llm_items", workflow)

        artifact = {
            "schema": "insight_batch_result.v1",
            "task_id": task_id,
            "batch_id": batch_id,
            "doc_id": plan["doc_id"],
            "version_id": plan["version_id"],
            "status": "completed",
            "innovation": innovation,
            "created_at": time.time(),
        }
        artifact_path = task_state_root(db_path) / task_id / "insight_batches" / f"{batch_id}.json"
        write_json(artifact_path, artifact)
        workflow = finish_workflow_step(
            db_path,
            task_id,
            step_id,
            status="completed",
            diagnostics=diagnostics,
        )
        return {
            "schema": "staged_insight_batch_result.v1",
            "task_id": task_id,
            "batch_id": batch_id,
            "status": "completed",
            "item_count": len(innovation["items"]),
            "warnings": innovation.get("warnings") or [],
            "workflow": workflow,
            "artifact_path": str(artifact_path),
        }
    except LLMError as exc:
        workflow = finish_workflow_step(
            db_path,
            task_id,
            step_id,
            status="failed",
            error_type=exc.error_type,
            diagnostics=exc.metadata,
        )
        return _failed_result(task_id, batch_id, exc.error_type, workflow)


def finalize_insight_extraction_workflow(db_path: Path, task_id: str) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    _require_task_type(state)
    batch_steps = workflow_steps_for_phase(state, "insight_batches")
    incomplete = [str(step.get("step_id") or "") for step in batch_steps if step.get("status") != "completed"]
    if incomplete:
        return {
            "schema": "staged_insight_finalize_result.v1",
            "task_id": task_id,
            "status": "incomplete",
            "pending_steps": incomplete,
            "workflow": state,
        }

    plan = _task_json(db_path, task_id, "insight_extraction_plan.json")
    parts = []
    for batch in plan.get("batches") or []:
        artifact = _task_json(db_path, task_id, f"insight_batches/{batch['batch_id']}.json")
        parts.append(artifact.get("innovation") or {})
    innovation = _merge_innovations(parts, plan)
    inputs = load_insight_extraction_inputs(db_path, str(plan["doc_id"]))
    citation_map = build_citation_map(
        str(plan["doc_id"]),
        str(plan["version_id"]),
        inputs["card"],
        inputs["references"],
        inputs["nodes"],
    )
    artifact_dir = inputs["artifact_dir"]
    write_json(artifact_dir / "innovation.json", innovation)
    write_json(artifact_dir / "citation_map.json", citation_map)
    update_task_status(db_path, task_id, innovation["status"], warnings=innovation["warnings"])
    workflow = set_workflow_phase(db_path, task_id, "completed", status="completed")
    return {
        "schema": "staged_insight_finalize_result.v1",
        "task_id": task_id,
        "doc_id": plan["doc_id"],
        "status": innovation["status"],
        "innovation": innovation,
        "citation_map": citation_map,
        "workflow": workflow,
        "artifact_paths": {
            "innovation": str(artifact_dir / "innovation.json"),
            "citation_map": str(artifact_dir / "citation_map.json"),
        },
    }


def _merge_innovations(parts: List[Dict[str, Any]], plan: Dict[str, Any]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    seen = set()
    for part in parts:
        for item in part.get("items") or []:
            if not isinstance(item, dict):
                continue
            key = compact_whitespace(str(item.get("claim") or item.get("title") or "")).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(item)
    warnings = unique_strings(warning for part in parts for warning in part.get("warnings") or [])
    return {
        "schema": "innovation.v1",
        "status": "extracted" if items else "partial",
        "doc_id": plan["doc_id"],
        "version_id": plan["version_id"],
        "title": next((str(part.get("title") or "") for part in parts if part.get("title")), ""),
        "source": "llm",
        "llm_mode": "staged_batch_json",
        "items": items[:16],
        "limitations": unique_strings(item for part in parts for item in part.get("limitations") or []),
        "open_questions": unique_strings(item for part in parts for item in part.get("open_questions") or []),
        "warnings": warnings,
        "batch_count": len(parts),
        "created_at": time.time(),
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
    raise ValueError(f"Unknown insight batch: {batch_id}")


def _require_task_type(state: Dict[str, Any]) -> None:
    if state.get("task_type") != "insight_extraction":
        raise ValueError("Workflow task type must be insight_extraction.")


def _failed_result(task_id: str, batch_id: str, error_type: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
    retryable = error_type not in {"document_version_changed", "batch_nodes_missing"}
    return {
        "schema": "staged_insight_batch_result.v1",
        "task_id": task_id,
        "batch_id": batch_id,
        "status": "failed",
        "error_type": error_type,
        "retryable": retryable,
        "recovery": {
            "action": "retry_same_step" if retryable else "prepare_new_workflow",
            "completed_steps_preserved": True,
        },
        "workflow": workflow,
    }
