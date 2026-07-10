from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .config import mcp_llm_step_timeout_seconds
from .llm import generate_json_object, llm_status
from .task_artifacts import TASK_ID_RE, task_state_root
from .utils import read_json, unique_strings, write_json


WORKFLOW_SCHEMA = "staged_workflow.v1"
JsonGenerator = Callable[[str, str], Dict[str, object]]


def create_workflow_state(
    db_path: Path,
    task_id: str,
    task_type: str,
    steps: Iterable[Dict[str, Any]],
    *,
    phase: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    state = {
        "schema": WORKFLOW_SCHEMA,
        "task_id": task_id,
        "task_type": task_type,
        "status": "prepared",
        "phase": phase,
        "llm": _llm_identity(),
        "steps": [_normalize_step(step) for step in steps],
        "metadata": metadata or {},
        "created_at": now,
        "updated_at": now,
    }
    return save_workflow_state(db_path, state)


def get_workflow_state(db_path: Path, task_id: str) -> Dict[str, Any]:
    path = _workflow_path(db_path, task_id)
    state = read_json(path, None)
    if not isinstance(state, dict) or state.get("schema") != WORKFLOW_SCHEMA:
        raise FileNotFoundError(f"Workflow state not found: {path}")
    return _with_summary(state)


def save_workflow_state(db_path: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(state.get("task_id") or "")
    path = _workflow_path(db_path, task_id)
    state["updated_at"] = time.time()
    state["summary"] = _summary(state)
    write_json(path, state)
    return _with_summary(state)


def start_workflow_step(db_path: Path, task_id: str, step_id: str) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    step = _find_step(state, step_id)
    llm_identity = _llm_identity()
    step["status"] = "running"
    step["attempt_count"] = int(step.get("attempt_count") or 0) + 1
    step["error_type"] = ""
    step["started_at"] = time.time()
    step["finished_at"] = None
    step["llm"] = llm_identity
    state["llm"] = llm_identity
    state["status"] = "in_progress"
    return save_workflow_state(db_path, state)


def finish_workflow_step(
    db_path: Path,
    task_id: str,
    step_id: str,
    *,
    status: str,
    error_type: str = "",
) -> Dict[str, Any]:
    if status not in {"completed", "failed", "skipped"}:
        raise ValueError(f"Unsupported workflow step status: {status}")
    state = get_workflow_state(db_path, task_id)
    step = _find_step(state, step_id)
    step["status"] = status
    step["error_type"] = error_type
    step["finished_at"] = time.time()
    summary = _summary(state)
    state["status"] = (
        "ready_to_finalize"
        if summary["remaining_step_count"] == 0 and not summary["failed_steps"]
        else "in_progress"
    )
    return save_workflow_state(db_path, state)


def set_workflow_phase(db_path: Path, task_id: str, phase: str, *, status: Optional[str] = None) -> Dict[str, Any]:
    state = get_workflow_state(db_path, task_id)
    state["phase"] = phase
    if status:
        state["status"] = status
    return save_workflow_state(db_path, state)


def workflow_step(state: Dict[str, Any], step_id: str) -> Dict[str, Any]:
    return _find_step(state, step_id)


def workflow_steps_for_phase(state: Dict[str, Any], phase: str) -> List[Dict[str, Any]]:
    return [step for step in state.get("steps") or [] if str(step.get("phase") or "") == phase]


def single_request_json_generator(
    operation: str,
    stage: str,
    *,
    max_tokens: Optional[int] = None,
    thinking: Optional[bool] = None,
) -> JsonGenerator:
    """Create a JSON generator that can issue only one bounded LLM request."""

    def generate(system_prompt: str, user_prompt: str) -> Dict[str, object]:
        options: Dict[str, Any] = {
            "timeout_seconds": mcp_llm_step_timeout_seconds(),
            "retry_count": 0,
            "operation": operation,
            "stage": stage,
        }
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if thinking is not None:
            options["thinking"] = thinking
        return generate_json_object(
            system_prompt,
            user_prompt,
            **options,
        )

    return generate


def _workflow_path(db_path: Path, task_id: str) -> Path:
    if not TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"Unsupported task id: {task_id}")
    return task_state_root(db_path) / task_id / "workflow_state.json"


def _normalize_step(step: Dict[str, Any]) -> Dict[str, Any]:
    step_id = str(step.get("step_id") or "")
    if not step_id:
        raise ValueError("Workflow step_id is required.")
    return {
        "step_id": step_id,
        "phase": str(step.get("phase") or ""),
        "status": str(step.get("status") or "pending"),
        "artifact": str(step.get("artifact") or ""),
        "attempt_count": int(step.get("attempt_count") or 0),
        "error_type": str(step.get("error_type") or ""),
        "started_at": step.get("started_at"),
        "finished_at": step.get("finished_at"),
        "metadata": step.get("metadata") if isinstance(step.get("metadata"), dict) else {},
    }


def _find_step(state: Dict[str, Any], step_id: str) -> Dict[str, Any]:
    for step in state.get("steps") or []:
        if str(step.get("step_id") or "") == step_id:
            return step
    raise ValueError(f"Unknown workflow step: {step_id}")


def _with_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(state)
    result["summary"] = _summary(state)
    return result


def _summary(state: Dict[str, Any]) -> Dict[str, Any]:
    steps = state.get("steps") or []
    completed = [str(step.get("step_id")) for step in steps if step.get("status") == "completed"]
    failed = [str(step.get("step_id")) for step in steps if step.get("status") == "failed"]
    pending = [str(step.get("step_id")) for step in steps if step.get("status") in {"pending", "running", "failed"}]
    return {
        "step_count": len(steps),
        "completed_step_count": len(completed),
        "remaining_step_count": len(pending),
        "completed_steps": completed,
        "failed_steps": failed,
        "pending_steps": unique_strings(pending),
    }


def _llm_identity() -> Dict[str, Any]:
    status = llm_status(probe=False)
    return {
        "provider": status.get("provider") or "deepseek",
        "profile": status.get("profile") or "default",
        "model": status.get("model") or "",
        "configured": bool(status.get("configured")),
        "insecure_http": bool(status.get("insecure_http")),
        "step_timeout_seconds": mcp_llm_step_timeout_seconds(),
    }
