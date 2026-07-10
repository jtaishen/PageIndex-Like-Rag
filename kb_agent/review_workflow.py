from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .llm import LLMError
from .review import draft_review
from .task_artifacts import get_task_artifact, task_state_root, update_task_status
from .task_review_plan import build_review_section
from .tasks import generate_review_plan
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


def prepare_review_workflow(
    db_path: Path,
    topic: str,
    *,
    doc_ids: Optional[List[str]] = None,
    top_k_docs: int = 5,
    search_mode: str = "hybrid",
) -> Dict[str, Any]:
    prepared = generate_review_plan(
        db_path,
        topic,
        doc_ids=doc_ids,
        top_k_docs=top_k_docs,
        use_llm=False,
        require_llm=False,
        search_mode=search_mode,
    )
    task_id = str(prepared["task_id"])
    sections = (prepared.get("review_outline") or {}).get("sections") or []
    steps = []
    for section in sections:
        section_id = str(section.get("section_id") or "")
        if not section_id:
            continue
        steps.extend(
            [
                {
                    "step_id": f"outline:{section_id}",
                    "phase": "outline",
                    "artifact": f"review_sections/{section_id}.json",
                    "metadata": {"section_id": section_id},
                },
                {
                    "step_id": f"draft:{section_id}",
                    "phase": "draft",
                    "artifact": f"section_drafts/{section_id}.json",
                    "metadata": {"section_id": section_id},
                },
            ]
        )
    workflow = create_workflow_state(
        db_path,
        task_id,
        "review",
        steps,
        phase="outline",
        metadata={
            "topic": topic,
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
        warnings=(prepared.get("review_outline") or {}).get("warnings") or [],
    )
    return {
        "schema": "staged_review_prepare_result.v1",
        "task_id": task_id,
        "status": "prepared",
        "topic": topic,
        "selected_papers": prepared.get("selected_papers") or {},
        "section_ids": [str(section.get("section_id") or "") for section in sections if section.get("section_id")],
        "workflow": workflow,
        "artifact_paths": prepared.get("artifact_paths") or {},
    }


def generate_review_outline_section(
    db_path: Path,
    task_id: str,
    section_id: str,
    *,
    json_generator: Optional[JsonGenerator] = None,
) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    _require_task_type(state, "review")
    step_id = f"outline:{section_id}"
    workflow_step(state, step_id)
    start_workflow_step(db_path, task_id, step_id)
    try:
        outline = _task_json(db_path, task_id, "review_outline.json")
        selected = _task_json(db_path, task_id, "selected_papers.json")
        evidence_artifact = _task_json(db_path, task_id, f"section_evidence/{section_id}.json")
        spec = _find_section(outline, section_id)
        evidence = evidence_artifact.get("evidence") or []
        generator = json_generator or single_request_json_generator("review_outline", section_id)
        built = build_review_section(
            str(outline.get("topic") or ""),
            spec,
            selected.get("papers") or [],
            evidence,
            json_generator=generator,
        )
        artifact = {
            "schema": "review_outline_section.v1",
            "task_id": task_id,
            "section_id": section_id,
            "status": "completed",
            "section": built.section,
            "llm_diagnostics": built.llm_diagnostics,
            "created_at": time.time(),
        }
        task_dir = task_state_root(db_path) / task_id
        write_json(task_dir / "review_sections" / f"{section_id}.json", artifact)
        outline["sections"] = [
            built.section if str(item.get("section_id") or "") == section_id else item
            for item in outline.get("sections") or []
        ]
        outline["source"] = "llm_staged_partial"
        outline["status"] = "partial"
        write_json(task_dir / "review_outline.json", outline)
        workflow = finish_workflow_step(db_path, task_id, step_id, status="completed")
        return {
            "schema": "staged_review_section_result.v1",
            "task_id": task_id,
            "section_id": section_id,
            "status": "completed",
            "section": built.section,
            "llm_diagnostics": built.llm_diagnostics,
            "workflow": workflow,
            "artifact_path": str(task_dir / "review_sections" / f"{section_id}.json"),
        }
    except LLMError as exc:
        workflow = finish_workflow_step(db_path, task_id, step_id, status="failed", error_type=exc.error_type)
        return _failed_step_result("staged_review_section_result.v1", task_id, step_id, exc.error_type, workflow)


def finalize_review_outline(db_path: Path, task_id: str) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    _require_task_type(state, "review")
    outline_steps = workflow_steps_for_phase(state, "outline")
    incomplete = [str(step.get("step_id") or "") for step in outline_steps if step.get("status") != "completed"]
    if incomplete:
        return {
            "schema": "staged_review_finalize_result.v1",
            "task_id": task_id,
            "status": "incomplete",
            "pending_steps": incomplete,
            "workflow": state,
        }

    task_dir = task_state_root(db_path) / task_id
    outline = _task_json(db_path, task_id, "review_outline.json")
    section_by_id = {}
    for step in outline_steps:
        section_id = str((step.get("metadata") or {}).get("section_id") or "")
        artifact = _task_json(db_path, task_id, f"review_sections/{section_id}.json")
        section_by_id[section_id] = artifact["section"]
    outline["sections"] = [
        section_by_id.get(str(item.get("section_id") or ""), item)
        for item in outline.get("sections") or []
    ]
    section_warnings = [warning for section in outline["sections"] for warning in section.get("warnings") or []]
    outline_warnings = [
        warning
        for warning in outline.get("warnings") or []
        if warning not in {"llm_disabled", "rule_based_review_plan"} and not str(warning).startswith("llm_unavailable:")
    ]
    outline["warnings"] = unique_strings([*outline_warnings, *section_warnings])
    outline["source"] = "llm_staged"
    outline["status"] = "partial" if outline["warnings"] else "extracted"
    outline["llm_diagnostics"] = {
        "schema": "llm_diagnostics.v1",
        "mode": "staged_section_json",
        "retry_count": 0,
        "repair_used": False,
        "fallback_sections": [],
        "error_type": "",
        "section_count": len(outline_steps),
        "section_success_count": len(outline_steps),
    }
    write_json(task_dir / "review_outline.json", outline)
    update_task_status(db_path, task_id, outline["status"], warnings=outline["warnings"])
    workflow = set_workflow_phase(db_path, task_id, "draft", status="in_progress")
    return {
        "schema": "staged_review_finalize_result.v1",
        "task_id": task_id,
        "status": outline["status"],
        "review_outline": outline,
        "workflow": workflow,
        "artifact_path": str(task_dir / "review_outline.json"),
    }


def draft_review_section(
    db_path: Path,
    task_id: str,
    section_id: str,
    *,
    require_llm: bool = True,
    json_generator: Optional[JsonGenerator] = None,
) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    _require_task_type(state, "review")
    step_id = f"draft:{section_id}"
    workflow_step(state, step_id)
    start_workflow_step(db_path, task_id, step_id)
    generator = json_generator or single_request_json_generator("review_draft", section_id)
    try:
        result = draft_review(
            db_path,
            task_id,
            section_ids=[section_id],
            use_llm=True,
            require_llm=require_llm,
            json_generator=generator,
        )
        workflow = finish_workflow_step(db_path, task_id, step_id, status="completed")
        if not workflow_steps_for_phase(workflow, "draft") or all(
            step.get("status") == "completed" for step in workflow_steps_for_phase(workflow, "draft")
        ):
            workflow = set_workflow_phase(db_path, task_id, "completed", status="completed")
            report = result.get("review_report") or {}
            update_task_status(
                db_path,
                task_id,
                result.get("status") or "drafted",
                warnings=report.get("warnings") or [],
            )
        return {**result, "workflow": workflow}
    except LLMError as exc:
        workflow = finish_workflow_step(db_path, task_id, step_id, status="failed", error_type=exc.error_type)
        return _failed_step_result("review_draft_result.v1", task_id, step_id, exc.error_type, workflow)


def _task_json(db_path: Path, task_id: str, name: str) -> Dict[str, Any]:
    content = get_task_artifact(db_path, task_id, name)["content"]
    if not isinstance(content, dict):
        raise ValueError(f"Task artifact is not a JSON object: {name}")
    return content


def _find_section(outline: Dict[str, Any], section_id: str) -> Dict[str, Any]:
    for section in outline.get("sections") or []:
        if str(section.get("section_id") or "") == section_id:
            return section
    raise ValueError(f"Unknown review section: {section_id}")


def _require_task_type(state: Dict[str, Any], task_type: str) -> None:
    if state.get("task_type") != task_type:
        raise ValueError(f"Workflow task type must be {task_type}.")


def _failed_step_result(
    schema: str,
    task_id: str,
    step_id: str,
    error_type: str,
    workflow: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": schema,
        "task_id": task_id,
        "step_id": step_id,
        "status": "failed",
        "error_type": error_type,
        "workflow": workflow,
    }
