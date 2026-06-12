from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .task_artifacts import TASK_ID_RE, task_state_root
from .utils import compact_whitespace, stable_id, unique_strings as _unique_strings


MEMORY_CONTEXT_SCHEMA = "memory_context.v1"
MEMORY_CONTEXT_PREVIEW_CHARS = 900

ALLOWED_MEMORY_TYPES = {
    "preference",
    "project_rule",
    "setting",
    "task_progress",
    "next_action",
    "failure",
    "default",
}
PAPER_ASSET_PATTERNS = (
    re.compile(r"\bnode_id\b", re.IGNORECASE),
    re.compile(r"\bpage_range\b", re.IGNORECASE),
    re.compile(r"\bevidence(_packet)?\b", re.IGNORECASE),
    re.compile(r"\bexcerpt\b", re.IGNORECASE),
    re.compile(r"\bprompt\s*[:=]", re.IGNORECASE),
    re.compile(r"(完整|原始)\s*prompt", re.IGNORECASE),
    re.compile(r"模型原文"),
    re.compile(r"\braw_text\b", re.IGNORECASE),
    re.compile(r"\bnode_path\b", re.IGNORECASE),
    re.compile(r"\[\s*E\d+\s*\]"),
)
PAPER_TEXT_TOKENS = (
    "摘要：",
    "关键词：",
    "参考文献",
    "本文提出",
    "实验结果表明",
    "研究内容包括",
)
SKILL_MEMORY_TYPES = {
    "paper_qa": {"preference", "project_rule", "setting", "default"},
    "compare": {"preference", "project_rule", "setting", "task_progress", "next_action", "failure", "default"},
    "review": {"preference", "project_rule", "setting", "task_progress", "next_action", "failure", "default"},
    "memory_curator": set(ALLOWED_MEMORY_TYPES),
    "default": set(ALLOWED_MEMORY_TYPES),
}
TASK_ARTIFACT_ORDER = (
    "manifest.json",
    "selected_papers.json",
    "open_questions.json",
    "next_actions.json",
    "comparison_matrix.json",
    "review_outline.json",
    "review_report.json",
    "citation_check.json",
)


def put_memory(
    db_path: Path,
    scope: str,
    type_: str,
    subject_key: str,
    content: str,
    importance: float = 0.5,
    confidence: float = 1.0,
    ttl: Optional[float] = None,
    refs: str = "",
) -> Dict[str, object]:
    conn = db.connect(db_path)
    db.init_db(conn)
    now = time.time()
    memory_id = stable_id("mem", scope, type_, subject_key)
    try:
        conn.execute(
            """
            INSERT INTO memory_items(
                memory_id, scope, type, subject_key, content, refs, ttl,
                importance, confidence, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                content = excluded.content,
                refs = excluded.refs,
                ttl = excluded.ttl,
                importance = excluded.importance,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (
                memory_id,
                scope,
                type_,
                subject_key,
                content,
                refs,
                ttl,
                importance,
                confidence,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"memory_id": memory_id, "scope": scope, "subject_key": subject_key}


def put_memory_gated(
    db_path: Path,
    scope: str,
    type_: str,
    subject_key: str,
    content: str,
    *,
    importance: float = 0.5,
    confidence: float = 1.0,
    ttl_days: Optional[float] = None,
    refs: str = "",
    force: bool = False,
) -> Dict[str, object]:
    gate = evaluate_memory_write(scope, type_, subject_key, content, confidence=confidence, refs=refs, force=force)
    ttl = time.time() + ttl_days * 86400 if ttl_days is not None else None
    existing = _get_memory(db_path, scope, type_, subject_key)
    if not gate["accepted"]:
        return {
            "schema": "memory_write.v1",
            "action": "rejected",
            "accepted": False,
            "reason": gate["reason"],
            "gate_decision": "rejected",
            "gate_reason": gate["reason"],
            "memory_kind": gate["memory_kind"],
            "safety_warnings": gate["safety_warnings"],
            "scope": scope,
            "type": type_,
            "subject_key": subject_key,
            "confidence": confidence,
            "ttl": ttl,
        }

    merged_content = content
    if existing and not force:
        merged_content = _merge_memory_content(str(existing["content"]), content)
        importance = max(float(existing["importance"] or 0.0), importance)
        confidence = max(float(existing["confidence"] or 0.0), confidence)
        if not refs:
            refs = str(existing["refs"] or "")

    saved = put_memory(
        db_path,
        scope,
        type_,
        subject_key,
        merged_content,
        importance=importance,
        confidence=confidence,
        ttl=ttl,
        refs=refs,
    )
    action = "merged" if existing and not force else "accepted"
    return {
        "schema": "memory_write.v1",
        "action": action,
        "accepted": True,
        "reason": gate["reason"],
        "gate_decision": action,
        "gate_reason": gate["reason"],
        "memory_kind": gate["memory_kind"],
        "safety_warnings": gate["safety_warnings"],
        "memory_id": saved["memory_id"],
        "scope": scope,
        "type": type_,
        "subject_key": subject_key,
        "confidence": confidence,
        "ttl": ttl,
        "refs": refs,
    }


def evaluate_memory_write(
    scope: str,
    type_: str,
    subject_key: str,
    content: str,
    *,
    confidence: float = 1.0,
    refs: str = "",
    force: bool = False,
) -> Dict[str, object]:
    memory_kind = _memory_kind(type_)
    if force:
        return {
            "accepted": True,
            "reason": "force",
            "memory_kind": memory_kind,
            "safety_warnings": ["forced_write_bypasses_gate"],
        }
    text = compact_whitespace(f"{scope} {type_} {subject_key} {content}")
    warnings = _memory_safety_warnings(type_, content, refs=refs)
    if confidence < 0.45:
        return {"accepted": False, "reason": "low_confidence", "memory_kind": memory_kind, "safety_warnings": warnings}
    if type_ not in ALLOWED_MEMORY_TYPES:
        return {"accepted": False, "reason": "unsupported_memory_type", "memory_kind": memory_kind, "safety_warnings": warnings}
    if _looks_like_paper_asset(text):
        return {"accepted": False, "reason": "paper_asset_boundary", "memory_kind": memory_kind, "safety_warnings": ["paper_asset_boundary"]}
    if type_ == "failure" and len(content) > 600:
        return {"accepted": False, "reason": "failure_memory_too_long", "memory_kind": memory_kind, "safety_warnings": warnings}
    if len(content) > 1200 and type_ != "project_rule":
        return {"accepted": False, "reason": "content_too_long", "memory_kind": memory_kind, "safety_warnings": warnings}
    return {"accepted": True, "reason": "allowed_memory", "memory_kind": memory_kind, "safety_warnings": warnings}


def search_memory(db_path: Path, query: str, scope: Optional[str] = None, top_k: int = 8) -> List[Dict[str, object]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    like = f"%{query}%"
    params: List[object] = [like, like, like]
    scope_filter = ""
    if scope:
        scope_filter = "AND scope = ?"
        params.append(scope)
    params.append(top_k)
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM memory_items
            WHERE (content LIKE ? OR subject_key LIKE ? OR type LIKE ?)
              {scope_filter}
              AND (ttl IS NULL OR ttl > ?)
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            params[:-1] + [time.time(), params[-1]],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def remember_task(db_path: Path, task_id: str) -> Dict[str, object]:
    task_dir = _task_dir(db_path, task_id)
    manifest = _read_task_json(task_dir / "manifest.json")
    next_actions = _read_task_json(task_dir / "next_actions.json")
    review_report = _read_task_json(task_dir / "review_report.json", optional=True)
    citation_check = _read_task_json(task_dir / "citation_check.json", optional=True)
    content = _task_memory_content(task_id, manifest, next_actions, review_report, citation_check)
    result = put_memory_gated(
        db_path,
        "project",
        "task_progress",
        f"task:{task_id}",
        content,
        importance=0.8,
        confidence=0.9,
        refs=str(task_dir),
        force=True,
    )
    return {
        "schema": "remember_task.v1",
        "task_id": task_id,
        "memory": result,
        "summary": content,
    }


def resume_task(db_path: Path) -> Dict[str, object]:
    state_root = task_state_root(db_path)
    current = _read_task_json(state_root / "current_task.json", optional=True)
    current_task_id = str(current.get("task_id") or "") if current else ""
    current_task = _current_task_status(db_path, current_task_id) if current_task_id else {}
    task_memories = _recent_task_memories(db_path, limit=5)
    context = compile_memory_context(
        db_path,
        intent=str(current_task.get("task_type") or current.get("task_type") or "resume") if current else "resume",
        query=str(current_task.get("query") or current.get("query") or "resume task") if current else "resume task",
        task_id=current_task_id,
        skill_scope=str(current_task.get("task_type") or current.get("task_type") or "default") if current else "default",
        max_items=5,
        max_chars=MEMORY_CONTEXT_PREVIEW_CHARS,
    )
    resume_plan = _resume_plan(current_task_id, current_task, context)
    return {
        "schema": "resume_task.v1",
        "current_task": current_task,
        "remembered_tasks": task_memories,
        "resume_plan": resume_plan,
        "next_actions": current_task.get("next_actions", []),
        "suggested_commands": resume_plan["suggested_commands"],
        "compiled_context_preview": {
            "schema": context.get("schema"),
            "available": bool(context.get("compiled_context")),
            "selected_memory_count": context.get("selected_memory_count", 0),
            "negative_memory_count": context.get("negative_memory_count", 0),
            "artifact_ref_count": context.get("artifact_ref_count", 0),
            "context_char_count": context.get("context_char_count", 0),
            "compiled_context": context.get("compiled_context", ""),
            "warnings": context.get("warnings") or [],
        },
    }


def compile_memory_context(
    db_path: Path,
    intent: str,
    query: str,
    *,
    task_id: str = "",
    skill_scope: str = "default",
    max_items: int = 8,
    max_chars: int = 4000,
) -> Dict[str, object]:
    resolved_task_id = _resolve_task_id(db_path, task_id)
    task_snapshot: Dict[str, object] = {}
    artifact_refs: List[Dict[str, object]] = []
    artifact_warnings: List[str] = []
    if resolved_task_id:
        try:
            task_snapshot = _current_task_status(db_path, resolved_task_id)
            artifact_refs, artifact_warnings = _task_artifact_refs(db_path, resolved_task_id)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            artifact_warnings.append("task_artifacts_unavailable")
    memory_candidates = _memory_rows(db_path, scope=None, type_=None, limit=100)
    selected_memories, filtered_memories = _select_memories(
        memory_candidates,
        intent=intent,
        query=query,
        task_id=resolved_task_id,
        skill_scope=skill_scope,
        max_items=max(0, max_items),
    )
    memory_groups = _memory_groups(selected_memories)
    warnings = list(artifact_warnings)
    if not resolved_task_id:
        warnings.append("no_task_context")
    if not selected_memories:
        warnings.append("no_selected_memories")
    if filtered_memories:
        warnings.append("filtered_memory_items")
    if memory_groups["negative_memories"]:
        warnings.append("negative_memory_available")
    memory_plan = _memory_plan(artifact_refs, selected_memories, filtered_memories, memory_groups)
    compiled, truncated = _compiled_context_text(
        intent=intent,
        query=query,
        skill_scope=skill_scope,
        task_snapshot=task_snapshot,
        artifact_refs=artifact_refs,
        memory_groups=memory_groups,
        max_chars=max(0, max_chars),
    )
    if truncated:
        warnings.append("context_truncated")
    return {
        "schema": MEMORY_CONTEXT_SCHEMA,
        "intent": intent,
        "query": query,
        "task_id": resolved_task_id,
        "skill_scope": _normalize_skill_scope(skill_scope),
        "read_policy": _read_policy(skill_scope, max_items, max_chars),
        "task_snapshot": task_snapshot,
        "artifact_refs": artifact_refs,
        "artifact_ref_count": len(artifact_refs),
        "selected_memories": selected_memories,
        "selected_memory_count": len(selected_memories),
        "preferences": memory_groups["preferences"],
        "project_rules": memory_groups["project_rules"],
        "task_progress": memory_groups["task_progress"],
        "negative_memories": memory_groups["negative_memories"],
        "negative_memory_count": len(memory_groups["negative_memories"]),
        "filtered_memories": filtered_memories,
        "filtered_memory_count": len(filtered_memories),
        "memory_plan": memory_plan,
        "compiled_context": compiled,
        "context_char_count": len(compiled),
        "token_or_char_budget": {"max_chars": max_chars, "truncated": truncated},
        "warnings": _unique_strings(warnings),
        "created_at": time.time(),
    }


def compact_memory(db_path: Path, scope: Optional[str] = None) -> Dict[str, object]:
    rows = _memory_rows(db_path, scope=scope, type_="task_progress", limit=50)
    if not rows:
        return {
            "schema": "memory_compact.v1",
            "status": "empty",
            "scope": scope,
            "compacted_count": 0,
            "memory": None,
        }
    compacted = _compact_task_progress(rows)
    resolved_scope = scope or str(rows[0]["scope"])
    result = put_memory(
        db_path,
        resolved_scope,
        "task_progress",
        "task_progress_summary",
        compacted["content"],
        importance=compacted["importance"],
        confidence=compacted["confidence"],
        refs=compacted["refs"],
    )
    return {
        "schema": "memory_compact.v1",
        "status": "compacted",
        "scope": resolved_scope,
        "compacted_count": len(rows),
        "memory": result,
        "content": compacted["content"],
    }


def _resolve_task_id(db_path: Path, task_id: str) -> str:
    if task_id:
        return task_id
    current = _read_task_json(task_state_root(db_path) / "current_task.json", optional=True)
    return str(current.get("task_id") or "")


def _task_artifact_refs(db_path: Path, task_id: str) -> tuple[List[Dict[str, object]], List[str]]:
    task_dir = _task_dir(db_path, task_id)
    refs: List[Dict[str, object]] = []
    warnings: List[str] = []
    for name in TASK_ARTIFACT_ORDER:
        path = task_dir / name
        if not path.exists():
            continue
        try:
            payload = _read_task_json(path, optional=True)
        except (OSError, json.JSONDecodeError):
            warnings.append(f"artifact_unreadable:{name}")
            continue
        refs.append(_artifact_ref(name, path, payload))
    section_dir = task_dir / "section_evidence"
    if section_dir.exists():
        section_refs = []
        for path in sorted(section_dir.glob("*.json"))[:20]:
            try:
                payload = _read_task_json(path, optional=True)
            except (OSError, json.JSONDecodeError):
                warnings.append(f"artifact_unreadable:section_evidence/{path.name}")
                continue
            section_refs.append(_artifact_ref(f"section_evidence/{path.name}", path, payload))
        if section_refs:
            refs.append(
                {
                    "name": "section_evidence/*",
                    "path": str(section_dir),
                    "schema": "section_evidence_collection.v1",
                    "summary": f"{len(section_refs)} section evidence artifacts",
                    "counts": {
                        "section_count": len(section_refs),
                        "total_evidence_count": sum(int((item.get("counts") or {}).get("evidence_count") or 0) for item in section_refs),
                    },
                    "items": section_refs[:8],
                }
            )
    return refs, warnings


def _artifact_ref(name: str, path: Path, payload: Dict[str, Any]) -> Dict[str, object]:
    return {
        "name": name,
        "path": str(path),
        "schema": payload.get("schema", ""),
        "summary": _artifact_summary(name, payload),
        "counts": _artifact_counts(name, payload),
    }


def _artifact_summary(name: str, payload: Dict[str, Any]) -> str:
    if name == "manifest.json":
        return compact_whitespace(f"{payload.get('task_type') or ''}: {payload.get('query') or ''} status={payload.get('status') or ''}")[:260]
    if name == "selected_papers.json":
        titles = [str(item.get("title") or item.get("doc_id") or "") for item in (payload.get("papers") or [])[:5] if isinstance(item, dict)]
        return compact_whitespace("selected papers: " + "；".join(titles))[:300]
    if name == "next_actions.json":
        return compact_whitespace("next actions: " + "；".join(str(item) for item in (payload.get("items") or [])[:5]))[:300]
    if name == "open_questions.json":
        return compact_whitespace("open questions: " + "；".join(str(item) for item in (payload.get("items") or [])[:5]))[:300]
    if name == "comparison_matrix.json":
        dimensions = payload.get("dimensions") or []
        names = [str(item.get("name") or item.get("id") or "") for item in dimensions[:6] if isinstance(item, dict)]
        return compact_whitespace("comparison dimensions: " + "；".join(names))[:300]
    if name == "review_outline.json":
        sections = payload.get("sections") or []
        titles = [str(item.get("title") or item.get("section_id") or "") for item in sections[:8] if isinstance(item, dict)]
        return compact_whitespace("review sections: " + "；".join(titles))[:300]
    if name == "review_report.json":
        return compact_whitespace(f"review status={payload.get('status') or ''} quality={payload.get('draft_quality_level') or ''}")[:240]
    if name == "citation_check.json":
        return compact_whitespace(f"citation coverage={payload.get('coverage_score', '')} missing_refs={len(payload.get('missing_refs') or [])}")[:240]
    if name.startswith("section_evidence/"):
        return compact_whitespace(f"{payload.get('section_id') or name}: evidence_count={payload.get('evidence_count', 0)} source_doc_count={payload.get('source_doc_count', 0)}")[:240]
    return compact_whitespace(str(payload.get("schema") or name))[:240]


def _artifact_counts(name: str, payload: Dict[str, Any]) -> Dict[str, int]:
    if name == "selected_papers.json":
        return {"paper_count": int(payload.get("paper_count") or len(payload.get("papers") or []))}
    if name == "comparison_matrix.json":
        dimensions = payload.get("dimensions") or []
        cells = [cell for dimension in dimensions if isinstance(dimension, dict) for cell in dimension.get("cells") or []]
        return {"dimension_count": len(dimensions), "cell_count": len(cells)}
    if name == "review_outline.json":
        return {"section_count": len(payload.get("sections") or [])}
    if name.startswith("section_evidence/"):
        return {
            "evidence_count": int(payload.get("evidence_count") or len(payload.get("evidence") or [])),
            "source_doc_count": int(payload.get("source_doc_count") or 0),
        }
    if name == "next_actions.json" or name == "open_questions.json":
        return {"item_count": len(payload.get("items") or [])}
    return {}


def _select_memories(
    rows: List[Dict[str, object]],
    *,
    intent: str,
    query: str,
    task_id: str,
    skill_scope: str,
    max_items: int,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    allowed_types = SKILL_MEMORY_TYPES.get(_normalize_skill_scope(skill_scope), SKILL_MEMORY_TYPES["default"])
    scored: List[Dict[str, object]] = []
    filtered: List[Dict[str, object]] = []
    for row in rows:
        memory_type = str(row.get("type") or "")
        filter_reason = _memory_filter_reason(row, allowed_types)
        if filter_reason:
            filtered.append(_filtered_memory(row, filter_reason))
            continue
        item = _memory_context_item(row, intent=intent, query=query, task_id=task_id)
        scored.append(item)
    scored.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("memory_id") or "")))
    return scored[:max_items], filtered[: max(20, max_items * 2)]


def _memory_groups(selected: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    groups = {
        "preferences": [],
        "project_rules": [],
        "task_progress": [],
        "negative_memories": [],
    }
    for item in selected:
        memory_type = str(item.get("type") or "")
        if memory_type == "failure":
            groups["negative_memories"].append(item)
        elif memory_type == "project_rule":
            groups["project_rules"].append(item)
        elif memory_type in {"task_progress", "next_action"}:
            groups["task_progress"].append(item)
        elif memory_type in {"preference", "setting", "default"}:
            groups["preferences"].append(item)
    return groups


def _memory_plan(
    artifact_refs: List[Dict[str, object]],
    selected_memories: List[Dict[str, object]],
    filtered_memories: List[Dict[str, object]],
    memory_groups: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    return {
        "schema": "memory_plan.v1",
        "artifact_first": True,
        "artifact_sources": [str(item.get("name") or "") for item in artifact_refs if item.get("name")],
        "selected_memory_ids": [str(item.get("memory_id") or "") for item in selected_memories if item.get("memory_id")],
        "filtered_memory_ids": [str(item.get("memory_id") or "") for item in filtered_memories if item.get("memory_id")],
        "negative_memory_ids": [str(item.get("memory_id") or "") for item in memory_groups["negative_memories"] if item.get("memory_id")],
        "counts": {
            "artifact_ref_count": len(artifact_refs),
            "selected_memory_count": len(selected_memories),
            "filtered_memory_count": len(filtered_memories),
            "preference_count": len(memory_groups["preferences"]),
            "project_rule_count": len(memory_groups["project_rules"]),
            "task_progress_count": len(memory_groups["task_progress"]),
            "negative_memory_count": len(memory_groups["negative_memories"]),
        },
        "warnings": ["negative_memory_available"] if memory_groups["negative_memories"] else [],
    }


def _memory_filter_reason(row: Dict[str, object], allowed_types: set[str]) -> str:
    memory_type = str(row.get("type") or "")
    confidence = _float(row.get("confidence"), 1.0)
    if memory_type not in allowed_types:
        return "skill_scope_mismatch"
    gate = evaluate_memory_write(
        str(row.get("scope") or ""),
        memory_type,
        str(row.get("subject_key") or ""),
        str(row.get("content") or ""),
        confidence=confidence,
    )
    if not gate.get("accepted"):
        return str(gate.get("reason") or "memory_gate_rejected")
    return ""


def _filtered_memory(row: Dict[str, object], reason: str) -> Dict[str, object]:
    return {
        "memory_id": row.get("memory_id"),
        "scope": row.get("scope"),
        "type": row.get("type"),
        "subject_key": row.get("subject_key"),
        "reason": reason,
    }


def _memory_context_item(row: Dict[str, object], *, intent: str, query: str, task_id: str) -> Dict[str, object]:
    text = compact_whitespace(f"{row.get('scope') or ''} {row.get('type') or ''} {row.get('subject_key') or ''} {row.get('content') or ''}")
    terms = _query_terms(f"{intent} {query}")
    hits = [term for term in terms if term.lower() in text.lower()]
    reasons = []
    score = 2.0 * _float(row.get("importance"), 0.5) + _float(row.get("confidence"), 1.0)
    if hits:
        score += min(2.0, 0.35 * len(hits))
        reasons.append("query_match")
    if task_id and (task_id in str(row.get("refs") or "") or task_id in str(row.get("content") or "") or task_id in str(row.get("subject_key") or "")):
        score += 1.0
        reasons.append("task_match")
    if row.get("type") in {"project_rule", "preference"}:
        score += 0.35
        reasons.append("stable_context")
    if row.get("type") == "task_progress":
        score += 0.25
        reasons.append("task_progress")
    if row.get("type") == "failure":
        score += 0.2
        reasons.append("negative_memory")
    return {
        "memory_id": row.get("memory_id"),
        "scope": row.get("scope"),
        "type": row.get("type"),
        "subject_key": row.get("subject_key"),
        "content": _short_text(str(row.get("content") or ""), 360),
        "refs": _short_refs(str(row.get("refs") or "")),
        "importance": row.get("importance"),
        "confidence": row.get("confidence"),
        "score": round(score, 4),
        "matched_terms": hits[:8],
        "selection_reasons": _unique_strings(reasons or ["high_priority"]),
    }


def _compiled_context_text(
    *,
    intent: str,
    query: str,
    skill_scope: str,
    task_snapshot: Dict[str, object],
    artifact_refs: List[Dict[str, object]],
    memory_groups: Dict[str, List[Dict[str, object]]],
    max_chars: int,
) -> tuple[str, bool]:
    lines = [
        f"intent: {intent}",
        f"skill_scope: {_normalize_skill_scope(skill_scope)}",
        f"query: {compact_whitespace(query)}",
    ]
    if task_snapshot:
        lines.extend(
            [
                f"task_id: {task_snapshot.get('task_id') or ''}",
                f"task_type: {task_snapshot.get('task_type') or ''}",
                f"task_status: {task_snapshot.get('status') or ''}",
                f"task_query: {task_snapshot.get('query') or ''}",
            ]
        )
        actions = task_snapshot.get("next_actions") or []
        if actions:
            lines.append("next_actions: " + "；".join(str(item) for item in actions[:5]))
    if artifact_refs:
        lines.append("artifact_refs:")
        for item in artifact_refs[:10]:
            lines.append(f"- {item.get('name')}: {item.get('summary')} ({item.get('path')})")
    if any(memory_groups.get(key) for key in ("preferences", "project_rules", "task_progress", "negative_memories")):
        lines.append("selected_memories:")
    for label, title in (
        ("preferences", "preferences:"),
        ("project_rules", "project_rules:"),
        ("task_progress", "task_progress:"),
        ("negative_memories", "negative_memories:"),
    ):
        items = memory_groups.get(label) or []
        if not items:
            continue
        lines.append(title)
        for item in items:
            refs = item.get("refs") or []
            ref_text = f" refs={','.join(str(ref) for ref in refs[:3])}" if refs else ""
            lines.append(f"- [{item.get('type')}] {item.get('subject_key')}: {item.get('content')}{ref_text}")
    text = "\n".join(line for line in lines if str(line).strip())
    if max_chars and len(text) > max_chars:
        return text[: max(0, max_chars - 3)].rstrip() + "...", True
    return text, False


def _read_policy(skill_scope: str, max_items: int, max_chars: int) -> Dict[str, object]:
    normalized = _normalize_skill_scope(skill_scope)
    return {
        "schema": "memory_read_policy.v1",
        "skill_scope": normalized,
        "allowed_memory_types": sorted(SKILL_MEMORY_TYPES.get(normalized, SKILL_MEMORY_TYPES["default"])),
        "artifact_first": True,
        "artifact_order": list(TASK_ARTIFACT_ORDER),
        "max_items": max_items,
        "max_chars": max_chars,
    }


def _normalize_skill_scope(skill_scope: str) -> str:
    value = compact_whitespace(skill_scope).lower().replace("-", "_")
    if value in {"qa", "paperqa", "paper_qa"}:
        return "paper_qa"
    if value in {"compare", "comparison"}:
        return "compare"
    if value in {"review", "survey", "writer"}:
        return "review"
    if value in {"memory", "memory_curator", "curator"}:
        return "memory_curator"
    return value or "default"


def _query_terms(text: str) -> List[str]:
    terms: List[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    return _unique_strings(terms)[:16]


def _short_text(text: str, limit: int) -> str:
    return compact_whitespace(text)[:limit]


def _short_refs(refs: str) -> List[str]:
    return [item for item in _unique_strings(line.strip() for line in refs.splitlines()) if item][:8]


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _memory_kind(type_: str) -> str:
    if type_ == "failure":
        return "negative_memory"
    if type_ in {"task_progress", "next_action"}:
        return "task_state"
    if type_ in {"preference", "setting", "default"}:
        return "preference"
    if type_ == "project_rule":
        return "project_rule"
    return "unknown"


def _memory_safety_warnings(type_: str, content: str, refs: str) -> List[str]:
    warnings = []
    if type_ == "failure" and len(compact_whitespace(content)) > 480:
        warnings.append("failure_memory_near_length_limit")
    if refs and _looks_like_paper_asset(refs):
        warnings.append("refs_may_point_to_paper_asset")
    return warnings


def _get_memory(db_path: Path, scope: str, type_: str, subject_key: str) -> Optional[Dict[str, object]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    memory_id = stable_id("mem", scope, type_, subject_key)
    try:
        row = conn.execute("SELECT * FROM memory_items WHERE memory_id = ?", (memory_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _looks_like_paper_asset(text: str) -> bool:
    if any(pattern.search(text) for pattern in PAPER_ASSET_PATTERNS):
        return True
    token_hits = sum(1 for token in PAPER_TEXT_TOKENS if token in text)
    return token_hits >= 2


def _merge_memory_content(old: str, new: str) -> str:
    old_clean = compact_whitespace(old)
    new_clean = compact_whitespace(new)
    if not old_clean:
        return new_clean
    if not new_clean or new_clean in old_clean:
        return old_clean
    if old_clean in new_clean:
        return new_clean
    return f"{old_clean}\n{new_clean}"


def _task_dir(db_path: Path, task_id: str) -> Path:
    if not TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"Unsupported task id: {task_id}")
    path = task_state_root(db_path) / task_id
    if not path.exists():
        raise FileNotFoundError(f"Task directory not found: {path}")
    return path


def _read_task_json(path: Path, optional: bool = False) -> Dict[str, Any]:
    if not path.exists():
        if optional:
            return {}
        raise FileNotFoundError(f"Task artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _task_memory_content(
    task_id: str,
    manifest: Dict[str, Any],
    next_actions: Dict[str, Any],
    review_report: Dict[str, Any],
    citation_check: Dict[str, Any],
) -> str:
    actions = next_actions.get("items") or review_report.get("next_actions") or []
    warnings = manifest.get("warnings") or review_report.get("warnings") or citation_check.get("warnings") or []
    lines = [
        f"task_id: {task_id}",
        f"task_type: {manifest.get('task_type') or 'unknown'}",
        f"query: {manifest.get('query') or ''}",
        f"status: {review_report.get('status') or manifest.get('status') or 'unknown'}",
        f"citation_coverage_score: {review_report.get('citation_coverage_score', citation_check.get('coverage_score', ''))}",
    ]
    if actions:
        lines.append("next_actions: " + "；".join(str(item) for item in actions[:5]))
    if warnings:
        lines.append("warnings: " + "；".join(str(item) for item in warnings[:5]))
    return "\n".join(line for line in lines if line.strip())


def _current_task_status(db_path: Path, task_id: str) -> Dict[str, object]:
    task_dir = task_state_root(db_path) / task_id
    manifest = _read_task_json(task_dir / "manifest.json", optional=True)
    next_actions = _read_task_json(task_dir / "next_actions.json", optional=True)
    review_report = _read_task_json(task_dir / "review_report.json", optional=True)
    citation_check = _read_task_json(task_dir / "citation_check.json", optional=True)
    section_dir = task_dir / "section_evidence"
    return {
        "task_id": task_id,
        "task_type": manifest.get("task_type") or "",
        "query": manifest.get("query") or "",
        "status": review_report.get("status") or manifest.get("status") or "",
        "task_dir": str(task_dir),
        "has_manifest": (task_dir / "manifest.json").exists(),
        "has_comparison_matrix": (task_dir / "comparison_matrix.json").exists(),
        "has_review_outline": (task_dir / "review_outline.json").exists(),
        "has_review_draft": (task_dir / "review_draft.md").exists(),
        "has_review_report": bool(review_report),
        "has_citation_check": bool(citation_check),
        "has_section_evidence": section_dir.exists() and any(section_dir.glob("*.json")),
        "drafted_section_count": review_report.get("drafted_section_count"),
        "citation_coverage_score": review_report.get("citation_coverage_score", citation_check.get("coverage_score")),
        "next_actions": next_actions.get("items") or review_report.get("next_actions") or [],
        "warnings": review_report.get("warnings") or manifest.get("warnings") or [],
    }


def _recent_task_memories(db_path: Path, limit: int) -> List[Dict[str, object]]:
    return _memory_rows(db_path, scope=None, type_="task_progress", limit=limit)


def _memory_rows(
    db_path: Path,
    *,
    scope: Optional[str],
    type_: Optional[str],
    limit: int,
) -> List[Dict[str, object]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    params: List[object] = [time.time()]
    filters = ["(ttl IS NULL OR ttl > ?)"]
    if scope:
        filters.append("scope = ?")
        params.append(scope)
    if type_:
        filters.append("type = ?")
        params.append(type_)
    params.append(limit)
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM memory_items
            WHERE {' AND '.join(filters)}
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _suggest_commands(task_id: str, current_task: Dict[str, object]) -> List[str]:
    if not task_id:
        return ["uv run python -m kb_agent.cli generate-review \"<topic>\""]
    commands = []
    if current_task.get("has_review_outline") and not current_task.get("has_review_draft"):
        commands.append(f"uv run python -m kb_agent.cli draft-review {task_id}")
    if current_task.get("has_review_draft"):
        commands.append(f"uv run python -m kb_agent.cli check-review {task_id}")
        commands.append(f"uv run python -m kb_agent.cli assemble-review {task_id}")
    commands.append(f"uv run python -m kb_agent.cli remember-task {task_id}")
    return commands


def _resume_plan(task_id: str, current_task: Dict[str, object], context: Dict[str, object]) -> Dict[str, object]:
    if not task_id:
        return {
            "schema": "resume_plan.v1",
            "status": "blocked",
            "current_stage": "no_current_task",
            "blocking_gaps": ["no_current_task"],
            "next_actions": ["provide_task_id_or_create_task"],
            "suggested_commands": _suggest_commands("", {}),
            "relevant_negative_memory_ids": [],
        }
    task_type = str(current_task.get("task_type") or "task")
    if task_type == "review":
        stage, gaps = _review_resume_stage(current_task)
    elif task_type == "compare":
        stage, gaps = _compare_resume_stage(current_task)
    else:
        stage, gaps = ("task_ready", [] if current_task.get("has_manifest") else ["missing_manifest"])
    commands = _resume_commands(task_id, task_type, stage, gaps, current_task)
    actions = [str(item) for item in current_task.get("next_actions") or [] if str(item)] or _default_resume_actions(task_type, stage, gaps)
    return {
        "schema": "resume_plan.v1",
        "status": "blocked" if gaps else "ready",
        "current_stage": stage,
        "blocking_gaps": gaps,
        "next_actions": actions,
        "suggested_commands": commands,
        "relevant_negative_memory_ids": [str(item.get("memory_id") or "") for item in context.get("negative_memories") or [] if item.get("memory_id")],
    }


def _review_resume_stage(current_task: Dict[str, object]) -> tuple[str, List[str]]:
    gaps: List[str] = []
    if not current_task.get("has_manifest"):
        gaps.append("missing_manifest")
    if not current_task.get("has_review_outline"):
        gaps.append("missing_review_outline")
        return "needs_review_outline", gaps
    if not current_task.get("has_section_evidence"):
        gaps.append("missing_section_evidence")
        return "needs_section_evidence", gaps
    if not current_task.get("has_review_draft"):
        return "ready_to_draft", gaps
    if not current_task.get("has_review_report") or not current_task.get("has_citation_check"):
        gaps.append("missing_citation_check")
        return "needs_citation_check", gaps
    coverage = _float(current_task.get("citation_coverage_score"), 1.0)
    if coverage < 0.6:
        gaps.append("low_citation_coverage")
        return "needs_citation_repair", gaps
    return "ready_to_assemble", gaps


def _compare_resume_stage(current_task: Dict[str, object]) -> tuple[str, List[str]]:
    gaps = []
    if not current_task.get("has_manifest"):
        gaps.append("missing_manifest")
    if not current_task.get("has_comparison_matrix"):
        gaps.append("missing_comparison_matrix")
        return "needs_comparison_matrix", gaps
    return "comparison_ready", gaps


def _resume_commands(task_id: str, task_type: str, stage: str, gaps: List[str], current_task: Dict[str, object]) -> List[str]:
    if "missing_review_outline" in gaps:
        return [f"uv run python -m kb_agent.cli generate-review \"{current_task.get('query') or '<topic>'}\" --no-llm"]
    if "missing_comparison_matrix" in gaps:
        return [f"uv run python -m kb_agent.cli compare \"{current_task.get('query') or '<topic>'}\" --no-llm"]
    if stage == "ready_to_draft":
        return [f"uv run python -m kb_agent.cli draft-review {task_id}", f"uv run python -m kb_agent.cli remember-task {task_id}"]
    if stage in {"needs_citation_check", "needs_citation_repair"}:
        return [f"uv run python -m kb_agent.cli check-review {task_id}", f"uv run python -m kb_agent.cli remember-task {task_id}"]
    if stage == "ready_to_assemble":
        return [f"uv run python -m kb_agent.cli assemble-review {task_id}", f"uv run python -m kb_agent.cli remember-task {task_id}"]
    if task_type == "compare":
        return [f"uv run python -m kb_agent.cli task-artifact {task_id} comparison_matrix.json", f"uv run python -m kb_agent.cli remember-task {task_id}"]
    return _suggest_commands(task_id, current_task)


def _default_resume_actions(task_type: str, stage: str, gaps: List[str]) -> List[str]:
    if gaps:
        return [f"resolve:{gap}" for gap in gaps]
    if task_type == "review":
        return {
            "ready_to_draft": ["draft_review_sections"],
            "ready_to_assemble": ["assemble_review"],
        }.get(stage, ["inspect_review_artifacts"])
    if task_type == "compare":
        return ["inspect_comparison_matrix"]
    return ["inspect_task_artifacts"]


def _compact_task_progress(rows: List[Dict[str, object]]) -> Dict[str, object]:
    lines = ["recent task progress:"]
    refs = []
    importance = 0.0
    confidence = 0.0
    seen = set()
    for row in rows[:12]:
        content = compact_whitespace(str(row.get("content") or ""))
        if content in seen:
            continue
        seen.add(content)
        lines.append(f"- {content[:360]}")
        if row.get("refs"):
            refs.append(str(row["refs"]))
        importance = max(importance, float(row.get("importance") or 0.0))
        confidence = max(confidence, float(row.get("confidence") or 0.0))
    return {
        "content": "\n".join(lines),
        "refs": "\n".join(_unique_strings(refs)),
        "importance": importance or 0.7,
        "confidence": confidence or 0.8,
    }
