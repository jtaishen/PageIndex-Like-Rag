from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .llm import LLMError
from .task_artifacts import get_task_artifact, task_state_root, update_task_status
from .task_compare import build_comparison_dimension
from .tasks import compare_papers
from .utils import unique_strings, write_json
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


def prepare_compare_workflow(
    db_path: Path,
    query: str,
    *,
    doc_ids: Optional[List[str]] = None,
    top_k_docs: int = 5,
    search_mode: str = "hybrid",
) -> Dict[str, Any]:
    prepared = compare_papers(
        db_path,
        query,
        doc_ids=doc_ids,
        top_k_docs=top_k_docs,
        use_llm=False,
        require_llm=False,
        search_mode=search_mode,
    )
    task_id = str(prepared["task_id"])
    dimensions = (prepared.get("comparison_matrix") or {}).get("dimensions") or []
    steps = [
        {
            "step_id": f"dimension:{dimension['id']}",
            "phase": "dimensions",
            "artifact": f"comparison_dimensions/{dimension['id']}.json",
            "metadata": {"dimension_id": str(dimension["id"])},
        }
        for dimension in dimensions
        if dimension.get("id")
    ]
    workflow = create_workflow_state(
        db_path,
        task_id,
        "compare",
        steps,
        phase="dimensions",
        metadata={
            "query": query,
            "doc_ids": [
                str(item.get("doc_id") or "")
                for item in (prepared.get("selected_papers") or {}).get("papers") or []
            ],
            "search_mode": search_mode,
        },
    )
    update_task_status(
        db_path,
        task_id,
        "prepared",
        warnings=(prepared.get("comparison_matrix") or {}).get("warnings") or [],
    )
    return {
        "schema": "staged_compare_prepare_result.v1",
        "task_id": task_id,
        "query": query,
        "status": "prepared",
        "selected_papers": prepared.get("selected_papers") or {},
        "dimension_ids": [str(item.get("id") or "") for item in dimensions if item.get("id")],
        "workflow": workflow,
        "artifact_paths": prepared.get("artifact_paths") or {},
    }


def generate_compare_dimension(
    db_path: Path,
    task_id: str,
    dimension_id: str,
    *,
    json_generator: Optional[JsonGenerator] = None,
) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    _require_compare(state)
    step_id = f"dimension:{dimension_id}"
    workflow_step(state, step_id)
    start_workflow_step(db_path, task_id, step_id)
    try:
        matrix = _task_json(db_path, task_id, "comparison_matrix.json")
        selected = _task_json(db_path, task_id, "selected_papers.json")
        dimension = _find_dimension(matrix, dimension_id)
        evidence_by_doc = {
            str(cell.get("doc_id") or ""): cell.get("evidence") or []
            for cell in dimension.get("cells") or []
            if cell.get("doc_id")
        }
        generator = json_generator or staged_json_generator("compare", dimension_id)
        built = build_comparison_dimension(
            str(matrix.get("query") or ""),
            dimension,
            selected.get("papers") or [],
            evidence_by_doc,
            json_generator=generator,
        )
        artifact = {
            "schema": "comparison_dimension.v1",
            "task_id": task_id,
            "dimension_id": dimension_id,
            "status": "completed",
            "dimension": built.dimension,
            "llm_diagnostics": built.llm_diagnostics,
            "created_at": time.time(),
        }
        task_dir = task_state_root(db_path) / task_id
        artifact_path = task_dir / "comparison_dimensions" / f"{dimension_id}.json"
        write_json(artifact_path, artifact)
        matrix["dimensions"] = [
            built.dimension if str(item.get("id") or "") == dimension_id else item
            for item in matrix.get("dimensions") or []
        ]
        matrix["source"] = "llm_staged_partial"
        matrix["status"] = "partial"
        write_json(task_dir / "comparison_matrix.json", matrix)
        workflow = finish_workflow_step(
            db_path,
            task_id,
            step_id,
            status="completed",
            diagnostics=built.llm_diagnostics,
        )
        return {
            "schema": "staged_compare_dimension_result.v1",
            "task_id": task_id,
            "dimension_id": dimension_id,
            "status": "completed",
            "dimension": built.dimension,
            "llm_diagnostics": built.llm_diagnostics,
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
        return {
            "schema": "staged_compare_dimension_result.v1",
            "task_id": task_id,
            "dimension_id": dimension_id,
            "status": "failed",
            "error_type": exc.error_type,
            "retryable": True,
            "recovery": {"action": "retry_same_step", "completed_steps_preserved": True},
            "workflow": workflow,
        }


def finalize_compare_workflow(db_path: Path, task_id: str) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    _require_compare(state)
    dimension_steps = workflow_steps_for_phase(state, "dimensions")
    incomplete = [str(step.get("step_id") or "") for step in dimension_steps if step.get("status") != "completed"]
    if incomplete:
        return {
            "schema": "staged_compare_finalize_result.v1",
            "task_id": task_id,
            "status": "incomplete",
            "pending_steps": incomplete,
            "workflow": state,
        }

    task_dir = task_state_root(db_path) / task_id
    matrix = _task_json(db_path, task_id, "comparison_matrix.json")
    dimensions = {}
    for step in dimension_steps:
        dimension_id = str((step.get("metadata") or {}).get("dimension_id") or "")
        artifact = _task_json(db_path, task_id, f"comparison_dimensions/{dimension_id}.json")
        dimensions[dimension_id] = artifact["dimension"]
    matrix["dimensions"] = [dimensions.get(str(item.get("id") or ""), item) for item in matrix.get("dimensions") or []]
    dimension_warnings = [warning for item in matrix["dimensions"] for warning in item.get("warnings") or []]
    matrix_warnings = [
        warning
        for warning in matrix.get("warnings") or []
        if warning not in {"llm_disabled", "rule_based_comparison"} and not str(warning).startswith("llm_unavailable:")
    ]
    matrix["warnings"] = unique_strings([*matrix_warnings, *dimension_warnings])
    matrix["source"] = "llm_staged"
    matrix["status"] = "partial" if matrix["warnings"] else "extracted"
    matrix["llm_diagnostics"] = {
        "schema": "llm_diagnostics.v1",
        "mode": "staged_dimension_json",
        "retry_count": 0,
        "repair_used": False,
        "fallback_dimensions": [],
        "error_type": "",
        "dimension_count": len(dimension_steps),
        "dimension_success_count": len(dimension_steps),
    }
    write_json(task_dir / "comparison_matrix.json", matrix)
    update_task_status(db_path, task_id, matrix["status"], warnings=matrix["warnings"])
    workflow = set_workflow_phase(db_path, task_id, "completed", status="completed")
    return {
        "schema": "staged_compare_finalize_result.v1",
        "task_id": task_id,
        "status": matrix["status"],
        "comparison_matrix": matrix,
        "workflow": workflow,
        "artifact_path": str(task_dir / "comparison_matrix.json"),
    }


def _task_json(db_path: Path, task_id: str, name: str) -> Dict[str, Any]:
    content = get_task_artifact(db_path, task_id, name)["content"]
    if not isinstance(content, dict):
        raise ValueError(f"Task artifact is not a JSON object: {name}")
    return content


def _find_dimension(matrix: Dict[str, Any], dimension_id: str) -> Dict[str, Any]:
    for dimension in matrix.get("dimensions") or []:
        if str(dimension.get("id") or "") == dimension_id:
            return dimension
    raise ValueError(f"Unknown comparison dimension: {dimension_id}")


def _require_compare(state: Dict[str, Any]) -> None:
    if state.get("task_type") != "compare":
        raise ValueError("Workflow task type must be compare.")
