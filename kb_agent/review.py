from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .llm import generate_json_object
from .llm_policies import structured_json_generator
from .review_quality import assemble_review_markdown, build_citation_check, build_review_report
from .review_section_draft import (
    build_section_draft,
    build_skipped_section_draft,
    prepare_numbered_draft_evidence,
)
from .task_artifacts import TASK_ID_RE, task_state_root
from .utils import write_json


def draft_review(
    db_path: Path,
    task_id: str,
    *,
    section_ids: Optional[List[str]] = None,
    use_llm: bool = True,
    require_llm: bool = False,
    should_continue: Optional[Callable[[], bool]] = None,
    skip_reason: str = "review_draft_budget_exhausted",
    budget_fallback_to_rule: bool = False,
    json_generator: Optional[Callable[[str, str], Dict[str, object]]] = None,
) -> Dict[str, Any]:
    task_dir = _review_task_dir(db_path, task_id)
    outline = _read_json(task_dir / "review_outline.json")
    sections = _selected_sections(outline, section_ids)
    if not sections:
        raise ValueError("No review sections were selected.")

    drafted = []
    warnings: List[str] = []
    llm_error = ""
    paths: Dict[str, str] = {}
    generator = json_generator or structured_json_generator(
        "review_draft",
        "legacy_sections",
        json_generator=generate_json_object,
    )
    for section in sections:
        evidence_artifact = _read_section_evidence(task_dir, str(section["section_id"]))
        numbered_evidence, compaction = prepare_numbered_draft_evidence(
            evidence_artifact.get("evidence") or [],
            artifact_compaction=evidence_artifact.get("compaction_report")
            if isinstance(evidence_artifact.get("compaction_report"), dict)
            else {},
        )
        if should_continue is not None and not should_continue():
            if budget_fallback_to_rule and not require_llm:
                draft_result = build_section_draft(
                    task_id,
                    outline,
                    section,
                    numbered_evidence,
                    compaction,
                    use_llm=False,
                    rule_warnings=[skip_reason, "llm_budget_exhausted"],
                    fallback_reason=skip_reason,
                    json_generator=generator,
                )
                draft = draft_result.draft
            else:
                draft = build_skipped_section_draft(
                    task_id,
                    section,
                    numbered_evidence,
                    reason=skip_reason,
                    compaction=compaction,
                )
            drafted.append(draft)
            section_paths = _write_section_draft(task_dir, draft)
            paths.update(section_paths)
            warnings.extend(draft.get("warnings") or [])
            continue
        draft_result = build_section_draft(
            task_id,
            outline,
            section,
            numbered_evidence,
            compaction,
            use_llm=use_llm,
            require_llm=require_llm,
            json_generator=generator,
        )
        draft = draft_result.draft
        if draft_result.llm_error:
            llm_error = draft_result.llm_error
        drafted.append(draft)
        section_paths = _write_section_draft(task_dir, draft)
        paths.update(section_paths)
        warnings.extend(draft.get("warnings") or [])

    assembled = assemble_review(db_path, task_id)
    paths.update(assembled.get("artifact_paths") or {})
    report = assembled["review_report"]
    status = "partial" if warnings or report.get("status") == "partial" else "drafted"
    return {
        "schema": "review_draft_result.v1",
        "task_id": task_id,
        "status": status,
        "drafted_section_count": len(drafted),
        "drafted_sections": [draft["section_id"] for draft in drafted],
        "skipped_section_count": sum(1 for draft in drafted if draft.get("status") == "skipped"),
        "section_drafts": drafted,
        "citation_check": assembled["citation_check"],
        "review_report": report,
        "artifact_paths": paths,
        "llm_error": llm_error,
    }


def assemble_review(db_path: Path, task_id: str) -> Dict[str, Any]:
    task_dir = _review_task_dir(db_path, task_id)
    outline = _read_json(task_dir / "review_outline.json")
    drafts = _read_section_drafts(task_dir, outline)
    markdown = assemble_review_markdown(outline, drafts)
    review_path = task_dir / "review_draft.md"
    review_path.write_text(markdown, encoding="utf-8")
    check = check_review_citations(db_path, task_id)
    paths = {
        "review_draft": str(review_path),
        "citation_check": str(task_dir / "citation_check.json"),
        "review_report": str(task_dir / "review_report.json"),
    }
    return {
        "schema": "review_assemble_result.v1",
        "task_id": task_id,
        "status": check["review_report"]["status"],
        "review_draft_path": str(review_path),
        "citation_check": check["citation_check"],
        "review_report": check["review_report"],
        "artifact_paths": paths,
    }


def check_review_citations(db_path: Path, task_id: str) -> Dict[str, Any]:
    task_dir = _review_task_dir(db_path, task_id)
    outline = _read_json(task_dir / "review_outline.json")
    drafts = _read_section_drafts(task_dir, outline)
    citation_check = build_citation_check(task_id, drafts)
    report = build_review_report(task_id, outline, drafts, citation_check)
    write_json(task_dir / "citation_check.json", citation_check)
    write_json(task_dir / "review_report.json", report)
    return {
        "schema": "review_check_result.v1",
        "task_id": task_id,
        "status": report["status"],
        "citation_check": citation_check,
        "review_report": report,
        "artifact_paths": {
            "citation_check": str(task_dir / "citation_check.json"),
            "review_report": str(task_dir / "review_report.json"),
        },
    }


def _review_task_dir(db_path: Path, task_id: str) -> Path:
    if not TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"Unsupported task id: {task_id}")
    task_dir = task_state_root(db_path) / task_id
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    if not (task_dir / "review_outline.json").exists():
        raise FileNotFoundError(f"Review outline not found for task: {task_id}")
    return task_dir


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact is not a JSON object: {path}")
    return payload


def _selected_sections(outline: Dict[str, Any], section_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    raw_sections = outline.get("sections") or []
    requested = set(section_ids or [])
    sections = []
    for item in raw_sections:
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("section_id") or "")
        if requested and section_id not in requested:
            continue
        if section_id:
            sections.append(item)
    return sections


def _read_section_evidence(task_dir: Path, section_id: str) -> Dict[str, Any]:
    path = task_dir / "section_evidence" / f"{section_id}.json"
    if path.exists():
        return _read_json(path)
    return {
        "schema": "section_evidence.v1",
        "section_id": section_id,
        "evidence": [],
        "warnings": ["missing_section_evidence_artifact"],
    }


def _write_section_draft(task_dir: Path, draft: Dict[str, Any]) -> Dict[str, str]:
    section_id = str(draft["section_id"])
    draft_dir = task_dir / "section_drafts"
    json_path = draft_dir / f"{section_id}.json"
    md_path = draft_dir / f"{section_id}.md"
    write_json(json_path, draft)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(str(draft["body_markdown"]), encoding="utf-8")
    return {
        f"section_drafts/{section_id}.json": str(json_path),
        f"section_drafts/{section_id}.md": str(md_path),
    }


def _read_section_drafts(task_dir: Path, outline: Dict[str, Any]) -> List[Dict[str, Any]]:
    draft_dir = task_dir / "section_drafts"
    drafts_by_id: Dict[str, Dict[str, Any]] = {}
    if draft_dir.exists():
        for path in sorted(draft_dir.glob("*.json")):
            try:
                draft = _read_json(path)
            except (json.JSONDecodeError, ValueError):
                continue
            section_id = str(draft.get("section_id") or path.stem)
            drafts_by_id[section_id] = draft
    ordered = []
    for section in outline.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("section_id") or "")
        if section_id in drafts_by_id:
            ordered.append(drafts_by_id[section_id])
    for section_id, draft in drafts_by_id.items():
        if all(item.get("section_id") != section_id for item in ordered):
            ordered.append(draft)
    return ordered
