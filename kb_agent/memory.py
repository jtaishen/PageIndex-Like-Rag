from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .tasks import TASK_ID_RE, _task_state_root
from .utils import compact_whitespace, stable_id


ALLOWED_MEMORY_TYPES = {
    "preference",
    "project_rule",
    "setting",
    "task_progress",
    "next_action",
    "default",
}
PAPER_ASSET_PATTERNS = (
    re.compile(r"\bnode_id\b", re.IGNORECASE),
    re.compile(r"\bpage_range\b", re.IGNORECASE),
    re.compile(r"\bevidence(_packet)?\b", re.IGNORECASE),
    re.compile(r"\bexcerpt\b", re.IGNORECASE),
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
    gate = evaluate_memory_write(scope, type_, subject_key, content, confidence=confidence, force=force)
    ttl = time.time() + ttl_days * 86400 if ttl_days is not None else None
    existing = _get_memory(db_path, scope, type_, subject_key)
    if not gate["accepted"]:
        return {
            "schema": "memory_write.v1",
            "action": "rejected",
            "accepted": False,
            "reason": gate["reason"],
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
    force: bool = False,
) -> Dict[str, object]:
    if force:
        return {"accepted": True, "reason": "force"}
    text = compact_whitespace(f"{scope} {type_} {subject_key} {content}")
    if confidence < 0.45:
        return {"accepted": False, "reason": "low_confidence"}
    if type_ not in ALLOWED_MEMORY_TYPES:
        return {"accepted": False, "reason": "unsupported_memory_type"}
    if _looks_like_paper_asset(text):
        return {"accepted": False, "reason": "paper_asset_boundary"}
    if len(content) > 1200 and type_ != "project_rule":
        return {"accepted": False, "reason": "content_too_long"}
    return {"accepted": True, "reason": "allowed_memory"}


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
    state_root = _task_state_root(db_path)
    current = _read_task_json(state_root / "current_task.json", optional=True)
    current_task_id = str(current.get("task_id") or "") if current else ""
    current_task = _current_task_status(db_path, current_task_id) if current_task_id else {}
    task_memories = _recent_task_memories(db_path, limit=5)
    suggested = _suggest_commands(current_task_id, current_task)
    return {
        "schema": "resume_task.v1",
        "current_task": current_task,
        "remembered_tasks": task_memories,
        "next_actions": current_task.get("next_actions", []),
        "suggested_commands": suggested,
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
    path = _task_state_root(db_path) / task_id
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
    task_dir = _task_state_root(db_path) / task_id
    manifest = _read_task_json(task_dir / "manifest.json", optional=True)
    next_actions = _read_task_json(task_dir / "next_actions.json", optional=True)
    review_report = _read_task_json(task_dir / "review_report.json", optional=True)
    citation_check = _read_task_json(task_dir / "citation_check.json", optional=True)
    return {
        "task_id": task_id,
        "task_type": manifest.get("task_type") or "",
        "query": manifest.get("query") or "",
        "status": review_report.get("status") or manifest.get("status") or "",
        "task_dir": str(task_dir),
        "has_review_outline": (task_dir / "review_outline.json").exists(),
        "has_review_draft": (task_dir / "review_draft.md").exists(),
        "has_citation_check": bool(citation_check),
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


def _unique_strings(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
