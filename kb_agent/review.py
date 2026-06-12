from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .llm import LLMError, generate_json_object, llm_payload_metadata
from .task_artifacts import TASK_ID_RE, task_state_root
from .utils import compact_whitespace, stable_id, unique_strings as _unique_strings, write_json


EVIDENCE_REF_RE = re.compile(r"\[E(\d+)\]")
MAX_PROMPT_EVIDENCE = 5
MAX_PROMPT_SUMMARY_CHARS = 240


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
    for section in sections:
        evidence_artifact = _read_section_evidence(task_dir, str(section["section_id"]))
        prepared_evidence, draft_compaction = _prepare_draft_evidence(evidence_artifact.get("evidence") or [])
        numbered_evidence = _number_evidence(prepared_evidence)
        compaction = _merge_compaction_reports(
            evidence_artifact.get("compaction_report") if isinstance(evidence_artifact.get("compaction_report"), dict) else {},
            draft_compaction,
        )
        if should_continue is not None and not should_continue():
            if budget_fallback_to_rule and not require_llm:
                draft = _rule_based_section_draft(
                    task_id,
                    section,
                    numbered_evidence,
                    warnings=[skip_reason, "llm_budget_exhausted"],
                    llm_diagnostics=_llm_diagnostics(
                        used=False,
                        fallback_reason=skip_reason,
                        evidence_count=len(numbered_evidence),
                        compaction=compaction,
                    ),
                )
            else:
                draft = _skipped_section_draft(
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
        if use_llm:
            try:
                payload = _draft_section_with_llm(outline, section, numbered_evidence)
                draft = _normalize_llm_section_draft(
                    task_id,
                    section,
                    numbered_evidence,
                    payload,
                    llm_diagnostics=_llm_diagnostics(
                        used=True,
                        metadata=llm_payload_metadata(payload),
                        evidence_count=len(numbered_evidence),
                        compaction=compaction,
                    ),
                )
            except LLMError as exc:
                if require_llm:
                    raise
                llm_error = str(exc)
                draft = _rule_based_section_draft(
                    task_id,
                    section,
                    numbered_evidence,
                    warnings=[f"llm_unavailable:{llm_error}"],
                    llm_diagnostics=_llm_diagnostics(
                        used=False,
                        error=exc,
                        fallback_reason=getattr(exc, "error_type", "") or "llm_error",
                        evidence_count=len(numbered_evidence),
                        compaction=compaction,
                    ),
                )
        else:
            draft = _rule_based_section_draft(
                task_id,
                section,
                numbered_evidence,
                warnings=["llm_disabled"],
                llm_diagnostics=_llm_diagnostics(
                    used=False,
                    fallback_reason="llm_disabled",
                    evidence_count=len(numbered_evidence),
                    compaction=compaction,
                ),
            )
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
    markdown = _assemble_markdown(outline, drafts)
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
    citation_check = _build_citation_check(task_id, drafts)
    report = _build_review_report(task_id, outline, drafts, citation_check)
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


def _number_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numbered = []
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            continue
        enriched = dict(item)
        enriched["ref_id"] = f"E{index}"
        numbered.append(enriched)
    return numbered


def _prepare_draft_evidence(
    evidence: Iterable[Dict[str, Any]],
    *,
    max_items: int = MAX_PROMPT_EVIDENCE,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = [item for item in evidence if isinstance(item, dict)]
    unique, dedupe_stats = _dedupe_draft_evidence(raw)
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for item in unique:
        doc_id = str(item.get("doc_id") or "_unknown")
        by_doc.setdefault(doc_id, []).append(item)
    for items in by_doc.values():
        items.sort(key=_draft_evidence_priority, reverse=True)

    balanced: List[Dict[str, Any]] = []
    doc_ids = sorted(by_doc)
    cursor = 0
    while len(balanced) < max_items and any(by_doc.values()) and doc_ids:
        doc_id = doc_ids[cursor % len(doc_ids)]
        cursor += 1
        if not by_doc.get(doc_id):
            continue
        balanced.append(by_doc[doc_id].pop(0))

    source_doc_ids = _unique_strings(str(item.get("doc_id") or "") for item in balanced if item.get("doc_id"))
    warnings = []
    if int(dedupe_stats.get("duplicate_evidence_removed") or 0) > 0:
        warnings.append("draft_duplicate_evidence_compacted")
    if int(dedupe_stats.get("unique_evidence_count") or 0) > len(balanced):
        warnings.append("draft_evidence_truncated")
    return balanced, {
        "schema": "draft_evidence_compaction.v1",
        "raw_evidence_count": len(raw),
        "unique_evidence_count": dedupe_stats.get("unique_evidence_count", 0),
        "duplicate_evidence_removed": dedupe_stats.get("duplicate_evidence_removed", 0),
        "kept_evidence_count": len(balanced),
        "max_evidence_count": max_items,
        "source_doc_count": len(source_doc_ids),
        "source_doc_ids": source_doc_ids,
        "warnings": warnings,
    }


def _dedupe_draft_evidence(evidence: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: Dict[tuple[str, str], Dict[str, Any]] = {}
    duplicate_count = 0
    for index, item in enumerate(evidence):
        keys = _draft_evidence_keys(item)
        if not keys:
            continue
        existing_key = next((key for key in keys if key in seen), None)
        if existing_key is None:
            kept = dict(item)
            kept.setdefault("dedupe_reason", "kept:unique")
            kept["_dedupe_index"] = index
            for key in keys:
                seen[key] = kept
            result.append(kept)
            continue
        duplicate_count += 1
        existing = seen[existing_key]
        if _draft_evidence_priority(item) > _draft_evidence_priority(existing):
            replacement = dict(item)
            replacement["dedupe_reason"] = "kept:higher_score"
            replacement["_dedupe_index"] = existing.get("_dedupe_index", index)
            result = [replacement if current is existing else current for current in result]
            for key in keys:
                seen[key] = replacement
    for item in result:
        item.pop("_dedupe_index", None)
    return result, {
        "schema": "draft_evidence_dedupe.v1",
        "raw_evidence_count": len(evidence),
        "unique_evidence_count": len(result),
        "duplicate_evidence_removed": duplicate_count,
    }


def _draft_evidence_keys(item: Dict[str, Any]) -> List[tuple[str, str]]:
    doc_id = str(item.get("doc_id") or "")
    node_id = str(item.get("node_id") or "")
    keys: List[tuple[str, str]] = []
    if doc_id and node_id:
        keys.append(("node", f"{doc_id}:{node_id}"))
    node_path = compact_whitespace(str(item.get("node_path") or ""))
    text = compact_whitespace(
        str(
            item.get("evidence_summary")
            or item.get("summary")
            or item.get("claim")
            or item.get("excerpt")
            or item.get("snippet")
            or ""
        )
    )
    if doc_id and node_path and text:
        keys.append(("path_text", stable_id("draft_evidence", doc_id, node_path, text[:260], length=18)))
    elif doc_id and text:
        keys.append(("text", stable_id("draft_evidence", doc_id, text[:260], length=18)))
    return keys


def _draft_evidence_priority(item: Dict[str, Any]) -> tuple[float, float]:
    score = item.get("tree_score")
    if score is None:
        score = item.get("confidence")
    try:
        parsed = float(score or 0.0)
    except (TypeError, ValueError):
        parsed = 0.0
    text_len = len(_evidence_summary(item))
    return (parsed, min(240.0, float(text_len)))


def _merge_compaction_reports(artifact: Dict[str, Any], draft: Dict[str, Any]) -> Dict[str, Any]:
    warnings = _unique_strings([*(artifact.get("warnings") or []), *(draft.get("warnings") or [])])
    return {
        "schema": "review_draft_compaction.v1",
        "artifact": artifact,
        "draft": draft,
        "raw_evidence_count": draft.get("raw_evidence_count", artifact.get("raw_evidence_count", 0)),
        "unique_evidence_count": draft.get("unique_evidence_count", artifact.get("unique_evidence_count", 0)),
        "duplicate_evidence_removed": int(artifact.get("duplicate_evidence_removed") or 0)
        + int(draft.get("duplicate_evidence_removed") or 0),
        "kept_evidence_count": draft.get("kept_evidence_count", artifact.get("kept_evidence_count", 0)),
        "source_doc_count": draft.get("source_doc_count", artifact.get("source_doc_count", 0)),
        "source_doc_ids": draft.get("source_doc_ids") or artifact.get("source_doc_ids") or [],
        "warnings": warnings,
    }


def _draft_section_with_llm(
    outline: Dict[str, Any],
    section: Dict[str, Any],
    evidence: List[Dict[str, Any]],
) -> Dict[str, object]:
    system_prompt = (
        "你是严谨的中文论文综述写作助手。只能基于给定 evidence 写章节草稿。"
        "正文中每个自然段至少必须包含一个 [E1]、[E2] 这样的证据标记。"
        "不能被证据支持的内容必须放入 unsupported_claims，不允许写入正文。"
        "不要编造论文内容。必须返回 JSON object，不要返回 Markdown 代码块。"
    )
    user_prompt = "\n".join(
        [
            f"综述标题：{outline.get('title') or outline.get('topic') or ''}",
            f"章节标题：{section.get('title') or section.get('section_id')}",
            f"章节目标：{section.get('purpose') or ''}",
            "返回格式：",
            '{"claim_plan":[{"claim":"","evidence":["E1"]}],'
            '"body_markdown":"","unsupported_claims":[],"warnings":[]}',
            "",
            "可用证据：",
            *_format_evidence_for_prompt(evidence),
        ]
    )
    return generate_json_object(system_prompt, user_prompt)


def _format_evidence_for_prompt(evidence: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for item in evidence[:MAX_PROMPT_EVIDENCE]:
        lines.append(f"[{item.get('ref_id')}]")
        lines.append(f"title: {item.get('title') or item.get('doc_id') or ''}")
        lines.append(f"node_path: {item.get('node_path') or ''}")
        lines.append(f"page_range: {item.get('page_range') or ''}")
        lines.append(f"summary: {_evidence_summary(item)}")
        lines.append("")
    if not lines:
        lines.append("无可用证据。")
    return lines


def _evidence_summary(item: Dict[str, Any]) -> str:
    text = compact_whitespace(
        str(item.get("evidence_summary") or item.get("summary") or item.get("claim") or item.get("excerpt") or "")
    )
    if not text:
        text = compact_whitespace(f"{item.get('heading') or ''} {item.get('node_path') or ''}")
    return _excerpt(text, MAX_PROMPT_SUMMARY_CHARS)


def _llm_diagnostics(
    *,
    used: bool,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[LLMError] = None,
    fallback_reason: str = "",
    evidence_count: int = 0,
    compaction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    compaction = compaction or {}
    return {
        "schema": "section_draft_llm_diagnostics.v1",
        "used": used,
        "retry_count": int(metadata.get("retry_count") or 0),
        "repair_used": bool(metadata.get("repair_used")),
        "error_type": (getattr(error, "error_type", "") if error else str(metadata.get("error_type") or "")),
        "fallback_reason": fallback_reason,
        "evidence_count": evidence_count,
        "compact_count": int(compaction.get("kept_evidence_count") or evidence_count),
        "duplicate_evidence_removed": int(compaction.get("duplicate_evidence_removed") or 0),
        "compaction_warnings": _string_list(compaction.get("warnings")),
        "compaction": compaction,
    }


def _normalize_llm_section_draft(
    task_id: str,
    section: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    payload: Dict[str, object],
    *,
    llm_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    body = _body_value(payload.get("body_markdown") or payload.get("body") or payload.get("draft"))
    warnings = _string_list(payload.get("warnings"))
    unsupported_claims = _string_list(payload.get("unsupported_claims"))
    cleanup_report = {"removed_paragraph_count": 0, "removed_paragraphs": []}
    if not body:
        warnings.append("empty_llm_body")
        body = _fallback_body(section, evidence)
    else:
        body, cleanup_report = _clean_unsupported_body(body)
        if cleanup_report.get("removed_paragraph_count"):
            warnings.append("unsupported_paragraphs_removed")
            unsupported_claims.extend(cleanup_report.get("removed_paragraphs") or [])
    status = "partial" if "empty_llm_body" in warnings or "unsupported_paragraphs_removed" in warnings else "drafted"
    return _section_draft(
        task_id,
        section,
        evidence,
        body,
        source="llm",
        status=status,
        claim_plan=_normalize_claim_plan(payload.get("claim_plan")),
        unsupported_claims=unsupported_claims,
        warnings=warnings,
        llm_diagnostics=llm_diagnostics,
        paragraph_cleanup=cleanup_report,
    )


def _rule_based_section_draft(
    task_id: str,
    section: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    warnings: List[str],
    llm_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    body = _fallback_body(section, evidence)
    claim_plan = [
        {
            "claim": _excerpt(str(item.get("excerpt") or ""), 180),
            "evidence": [item["ref_id"]],
        }
        for item in evidence[:6]
    ]
    return _section_draft(
        task_id,
        section,
        evidence,
        body,
        source="rule",
        status="partial",
        claim_plan=claim_plan,
        unsupported_claims=[],
        warnings=[*warnings, "rule_based_section_draft"],
        llm_diagnostics=llm_diagnostics,
    )


def _skipped_section_draft(
    task_id: str,
    section: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    *,
    reason: str,
    compaction: Dict[str, Any],
) -> Dict[str, Any]:
    return _section_draft(
        task_id,
        section,
        evidence,
        "",
        source="skipped",
        status="skipped",
        claim_plan=[],
        unsupported_claims=[],
        warnings=[reason, "section_draft_skipped"],
        llm_diagnostics=_llm_diagnostics(
            used=False,
            fallback_reason=reason,
            evidence_count=len(evidence),
            compaction=compaction,
        ),
    )


def _section_draft(
    task_id: str,
    section: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    body: str,
    *,
    source: str,
    status: str,
    claim_plan: List[Dict[str, Any]],
    unsupported_claims: List[str],
    warnings: List[str],
    llm_diagnostics: Dict[str, Any],
    paragraph_cleanup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    refs = _body_refs(body)
    used_evidence = [item for item in evidence if item.get("ref_id") in refs]
    paragraph_support = _paragraph_support_report(body, evidence, cleanup=paragraph_cleanup)
    return {
        "schema": "section_draft.v1",
        "task_id": task_id,
        "section_id": section.get("section_id") or "",
        "title": section.get("title") or section.get("section_id") or "",
        "purpose": section.get("purpose") or "",
        "status": status,
        "source": source,
        "body_markdown": body,
        "claim_plan": claim_plan,
        "evidence": evidence,
        "used_evidence": used_evidence,
        "unsupported_claims": unsupported_claims,
        "paragraph_support_report": paragraph_support,
        "llm_diagnostics": llm_diagnostics,
        "evidence_compaction": llm_diagnostics.get("compaction") or {},
        "warnings": _unique_strings(warnings),
        "created_at": time.time(),
    }


def _fallback_body(section: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
    title = str(section.get("title") or section.get("section_id") or "章节草稿")
    purpose = str(section.get("purpose") or "整理本节证据。")
    lines = [
        f"### {title}",
        "",
        f"- 本节目标是{purpose}。",
    ]
    if not evidence:
        lines.append("- 当前没有足够证据支撑本节正文，需要补充检索或重新生成综述规划。")
        return "\n".join(lines)
    for item in evidence[:6]:
        excerpt = _excerpt(str(item.get("excerpt") or ""), 180)
        source = item.get("title") or item.get("doc_id") or "未知文档"
        path = item.get("node_path") or ""
        lines.append(f"- 《{source}》在“{path}”中提供了相关证据：{excerpt} [{item['ref_id']}]")
    return "\n".join(lines)


def _clean_unsupported_body(body: str) -> tuple[str, Dict[str, Any]]:
    kept: List[str] = []
    removed: List[str] = []
    for raw in re.split(r"\n\s*\n+", body.strip()):
        paragraph = raw.strip()
        if not paragraph:
            continue
        if _paragraph_requires_evidence(paragraph) and not EVIDENCE_REF_RE.search(paragraph):
            removed.append(_excerpt(paragraph, 180))
            continue
        kept.append(paragraph)
    return "\n\n".join(kept).strip(), {
        "schema": "paragraph_support_cleanup.v1",
        "removed_paragraph_count": len(removed),
        "removed_paragraphs": removed,
    }


def _paragraph_support_report(
    body: str,
    evidence: List[Dict[str, Any]],
    *,
    cleanup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cleanup = cleanup or {}
    paragraphs = _checkable_paragraphs(body)
    supported = [paragraph for paragraph in paragraphs if EVIDENCE_REF_RE.search(paragraph)]
    evidence_refs = {str(item.get("ref_id") or "") for item in evidence if item.get("ref_id")}
    body_refs = set(_body_refs(body))
    unused_refs = sorted(ref for ref in evidence_refs if ref not in body_refs)
    return {
        "schema": "paragraph_support_report.v1",
        "paragraph_count": len(paragraphs),
        "supported_paragraph_count": len(supported),
        "unsupported_paragraph_count": max(0, len(paragraphs) - len(supported)),
        "removed_paragraph_count": int(cleanup.get("removed_paragraph_count") or 0),
        "removed_paragraphs": cleanup.get("removed_paragraphs") or [],
        "unused_evidence_count": len(unused_refs),
        "unused_evidence_refs": unused_refs,
        "coverage_score": _coverage_score(len(supported), len(paragraphs), 0),
    }


def _paragraph_requires_evidence(paragraph: str) -> bool:
    text = compact_whitespace(paragraph)
    if not text:
        return False
    if text.startswith("#") or text.startswith("- "):
        return False
    if len(text) < 25:
        return False
    return True


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


def _assemble_markdown(outline: Dict[str, Any], drafts: List[Dict[str, Any]]) -> str:
    title = str(outline.get("title") or outline.get("topic") or "综述草稿")
    scope = str(outline.get("scope") or "")
    lines = [f"# {title}", ""]
    if scope:
        lines.extend([scope, ""])
    if not drafts:
        lines.append("当前还没有章节草稿。")
        return "\n".join(lines).strip() + "\n"
    for draft in drafts:
        title = str(draft.get("title") or draft.get("section_id") or "未命名章节")
        body = str(draft.get("body_markdown") or "").strip()
        lines.extend([f"## {title}", ""])
        lines.extend([body or "本节尚未生成正文。", ""])
    lines.extend(
        [
            "## 证据说明",
            "",
            "本文为可追溯综述草稿，正文中的 [E1]、[E2] 等证据编号在各章节内独立编号，具体来源见对应 section_drafts JSON 工件。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _build_citation_check(task_id: str, drafts: List[Dict[str, Any]]) -> Dict[str, Any]:
    sections = []
    missing_refs = []
    unused_evidence = []
    optional_unused_evidence = []
    unsupported_paragraphs = []
    scores = []
    skipped_sections = []
    for draft in drafts:
        section_id = str(draft.get("section_id") or "")
        body = str(draft.get("body_markdown") or "")
        evidence_refs = {str(item.get("ref_id") or "") for item in draft.get("evidence") or [] if item.get("ref_id")}
        body_refs = _body_refs(body)
        section_missing = sorted(ref for ref in body_refs if ref not in evidence_refs)
        section_unused = sorted(ref for ref in evidence_refs if ref not in body_refs)
        section_unsupported = _unsupported_paragraphs(section_id, body)
        hard_unused_count = 0
        optional_unused_count = 0
        missing_refs.extend({"section_id": section_id, "ref_id": ref} for ref in section_missing)
        if draft.get("status") == "skipped":
            skipped_sections.append(section_id)
            section_unused = []
        else:
            is_optional_unused = not section_unsupported and not section_missing
            target = optional_unused_evidence if is_optional_unused else unused_evidence
            target.extend({"section_id": section_id, "ref_id": ref} for ref in section_unused)
            if is_optional_unused:
                optional_unused_count = len(section_unused)
            else:
                hard_unused_count = len(section_unused)
        unsupported_paragraphs.extend(section_unsupported)
        paragraphs = _checkable_paragraphs(body)
        supported_count = max(0, len(paragraphs) - len(section_unsupported))
        score = 0.0 if draft.get("status") == "skipped" else _coverage_score(supported_count, len(paragraphs), len(section_missing))
        scores.append(score)
        sections.append(
            {
                "section_id": section_id,
                "title": draft.get("title") or section_id,
                "status": draft.get("status") or "",
                "evidence_count": len(evidence_refs),
                "body_ref_count": len(body_refs),
                "missing_ref_count": len(section_missing),
                "unused_evidence_count": hard_unused_count,
                "optional_unused_evidence_count": optional_unused_count,
                "unsupported_paragraph_count": len(section_unsupported),
                "coverage_score": score,
            }
        )
    overall = round(sum(scores) / len(scores), 3) if scores else 0.0
    warnings = []
    if missing_refs:
        warnings.append("missing_evidence_refs")
    if unused_evidence:
        warnings.append("unused_evidence")
    if unsupported_paragraphs:
        warnings.append("unsupported_paragraphs")
    if skipped_sections:
        warnings.append("section_draft_skipped")
    if scores and overall < 0.8:
        warnings.append("low_evidence_coverage")
    return {
        "schema": "citation_check.v1",
        "task_id": task_id,
        "status": "partial" if warnings else "passed",
        "coverage_score": overall,
        "sections": sections,
        "missing_refs": missing_refs,
        "unused_evidence": unused_evidence,
        "optional_unused_evidence": optional_unused_evidence,
        "unsupported_paragraphs": unsupported_paragraphs,
        "skipped_sections": skipped_sections,
        "missing_ref_count": len(missing_refs),
        "unused_evidence_count": len(unused_evidence),
        "optional_unused_evidence_count": len(optional_unused_evidence),
        "unsupported_paragraph_count": len(unsupported_paragraphs),
        "skipped_section_count": len(skipped_sections),
        "warnings": warnings,
        "created_at": time.time(),
    }


def _build_review_report(
    task_id: str,
    outline: Dict[str, Any],
    drafts: List[Dict[str, Any]],
    citation_check: Dict[str, Any],
) -> Dict[str, Any]:
    outline_sections = [item for item in outline.get("sections") or [] if isinstance(item, dict)]
    drafted_ids = {str(draft.get("section_id") or "") for draft in drafts}
    missing_sections = [
        str(section.get("section_id") or "")
        for section in outline_sections
        if str(section.get("section_id") or "") not in drafted_ids
    ]
    skipped_sections = [
        str(draft.get("section_id") or "")
        for draft in drafts
        if draft.get("status") == "skipped" and draft.get("section_id")
    ]
    warnings = list(citation_check.get("warnings") or [])
    for draft in drafts:
        warnings.extend(draft.get("warnings") or [])
    if missing_sections:
        warnings.append("missing_section_drafts")
    quality_reasons = _draft_quality_reasons(outline_sections, drafts, citation_check, missing_sections, warnings)
    draft_quality_level = _draft_quality_level(citation_check, missing_sections, quality_reasons)
    section_revision_actions = _section_revision_actions(outline_sections, drafts, citation_check)
    next_actions = []
    if citation_check.get("missing_refs"):
        next_actions.append("修正文中无法映射的证据编号。")
    if citation_check.get("unused_evidence"):
        next_actions.append("删除未使用证据，或把关键证据补充到正文对应观点。")
    if citation_check.get("optional_unused_evidence") and not citation_check.get("unused_evidence"):
        next_actions.append("存在可选未使用证据；可删除冗余证据，或保留为人工复核参考。")
    if citation_check.get("unsupported_paragraphs"):
        next_actions.append("为缺少证据标记的段落补充引用，或删除无证据观点。")
    if missing_sections:
        next_actions.append("为尚未生成的章节运行 draft-review。")
    if citation_check.get("coverage_score", 0.0) < 0.8:
        next_actions.append("优先重写低覆盖章节，确保每个关键段落至少有一个 [E#] 证据标记。")
    if not next_actions:
        next_actions.append("人工通读综述草稿，检查章节衔接和引用表达。")
    return {
        "schema": "review_report.v1",
        "task_id": task_id,
        "status": "partial" if warnings or missing_sections else "drafted",
        "draft_quality_level": draft_quality_level,
        "quality_reasons": quality_reasons,
        "title": outline.get("title") or outline.get("topic") or "",
        "section_count": len(outline_sections),
        "drafted_section_count": len(drafts),
        "skipped_section_count": len(skipped_sections),
        "missing_sections": missing_sections,
        "skipped_sections": skipped_sections,
        "citation_coverage_score": citation_check.get("coverage_score", 0.0),
        "paragraph_support_report": _review_paragraph_support_report(drafts),
        "optional_unused_evidence_count": len(citation_check.get("optional_unused_evidence") or []),
        "section_statuses": [
            {
                "section_id": draft.get("section_id") or "",
                "title": draft.get("title") or "",
                "status": draft.get("status") or "",
                "source": draft.get("source") or "",
                "draft_quality_level": _section_quality_level(str(draft.get("section_id") or ""), citation_check, draft),
                "warning_count": len(draft.get("warnings") or []),
                "warnings": draft.get("warnings") or [],
            }
            for draft in drafts
        ],
        "warnings": _unique_strings(warnings),
        "next_actions": next_actions,
        "revision_actions": next_actions,
        "section_revision_actions": section_revision_actions,
        "created_at": time.time(),
    }


def _section_revision_actions(
    outline_sections: List[Dict[str, Any]],
    drafts: List[Dict[str, Any]],
    citation_check: Dict[str, Any],
) -> List[Dict[str, Any]]:
    draft_map = {str(draft.get("section_id") or ""): draft for draft in drafts}
    check_map = {str(item.get("section_id") or ""): item for item in citation_check.get("sections") or [] if isinstance(item, dict)}
    result = []
    for section in outline_sections:
        section_id = str(section.get("section_id") or "")
        if not section_id:
            continue
        draft = draft_map.get(section_id, {})
        check = check_map.get(section_id, {})
        actions = []
        reasons = []
        if not draft:
            actions.append("为本节运行 draft-review。")
            reasons.append("section_draft_missing")
        if draft.get("status") == "skipped":
            actions.append("本节草稿阶段被跳过；需要释放 LLM 阶段预算后重跑，或用规则草稿补齐。")
            reasons.append("section_draft_skipped")
        if int(check.get("missing_ref_count") or 0) > 0:
            actions.append("修正文中无法映射的证据编号。")
            reasons.append("missing_refs")
        if int(check.get("unsupported_paragraph_count") or 0) > 0:
            actions.append("为无证据段落补充 [E#]，或删除无证据观点。")
            reasons.append("unsupported_paragraphs")
        if int(check.get("unused_evidence_count") or 0) > 0:
            actions.append("删除未使用证据，或把关键证据写入正文。")
            reasons.append("unused_evidence")
        optional_unused = _optional_unused_count(section_id, citation_check)
        if optional_unused and int(check.get("unused_evidence_count") or 0) == 0:
            actions.append("本节有可选未使用证据；可删除冗余证据或保留给人工复核。")
            reasons.append("optional_unused_evidence")
        if float(check.get("coverage_score") or 0.0) < 0.8:
            actions.append("重写低覆盖段落，提升证据覆盖。")
            reasons.append("low_evidence_coverage")
        draft_warnings = draft.get("warnings") or []
        if any(str(warning).startswith("llm_unavailable") for warning in draft_warnings):
            actions.append("检查 DeepSeek 回退原因，必要时重跑本节草稿。")
            reasons.append("llm_unavailable")
        if "rule_based_section_draft" in [str(warning) for warning in draft_warnings]:
            actions.append("人工润色规则版草稿，补足章节衔接。")
            reasons.append("rule_based_section_draft")
        if not actions:
            actions.append("人工通读本节，检查表达、衔接和引用格式。")
        result.append(
            {
                "section_id": section_id,
                "title": section.get("title") or section_id,
                "status": draft.get("status") or ("missing" if not draft else ""),
                "draft_quality_level": _section_quality_level(section_id, citation_check, draft),
                "quality_reasons": _unique_strings(reasons),
                "actions": _unique_strings(actions),
            }
        )
    return result


def _optional_unused_count(section_id: str, citation_check: Dict[str, Any]) -> int:
    count = 0
    for item in citation_check.get("optional_unused_evidence") or []:
        if isinstance(item, dict) and str(item.get("section_id") or "") == section_id:
            count += 1
    return count


def _review_paragraph_support_report(drafts: List[Dict[str, Any]]) -> Dict[str, Any]:
    section_reports = []
    paragraph_count = 0
    supported_count = 0
    unsupported_count = 0
    removed_count = 0
    unused_count = 0
    for draft in drafts:
        report = draft.get("paragraph_support_report") or {}
        if not isinstance(report, dict):
            report = _paragraph_support_report(str(draft.get("body_markdown") or ""), draft.get("evidence") or [])
        paragraph_count += int(report.get("paragraph_count") or 0)
        supported_count += int(report.get("supported_paragraph_count") or 0)
        unsupported_count += int(report.get("unsupported_paragraph_count") or 0)
        removed_count += int(report.get("removed_paragraph_count") or 0)
        unused_count += int(report.get("unused_evidence_count") or 0)
        section_reports.append(
            {
                "section_id": draft.get("section_id") or "",
                "paragraph_count": int(report.get("paragraph_count") or 0),
                "supported_paragraph_count": int(report.get("supported_paragraph_count") or 0),
                "unsupported_paragraph_count": int(report.get("unsupported_paragraph_count") or 0),
                "removed_paragraph_count": int(report.get("removed_paragraph_count") or 0),
                "unused_evidence_count": int(report.get("unused_evidence_count") or 0),
                "coverage_score": float(report.get("coverage_score") or 0.0),
            }
        )
    return {
        "schema": "review_paragraph_support_report.v1",
        "paragraph_count": paragraph_count,
        "supported_paragraph_count": supported_count,
        "unsupported_paragraph_count": unsupported_count,
        "removed_paragraph_count": removed_count,
        "unused_evidence_count": unused_count,
        "coverage_score": _coverage_score(supported_count, paragraph_count, 0),
        "sections": section_reports,
    }


def _section_quality_level(section_id: str, citation_check: Dict[str, Any], draft: Dict[str, Any]) -> str:
    if not draft:
        return "failed"
    if draft.get("status") == "skipped":
        return "failed"
    section_check = {}
    for item in citation_check.get("sections") or []:
        if isinstance(item, dict) and str(item.get("section_id") or "") == section_id:
            section_check = item
            break
    coverage = float(section_check.get("coverage_score") or 0.0)
    if int(section_check.get("missing_ref_count") or 0) > 0:
        return "weak"
    if int(section_check.get("unsupported_paragraph_count") or 0) > 0 or coverage < 0.5:
        return "weak"
    if draft.get("warnings") or int(section_check.get("unused_evidence_count") or 0) > 0 or coverage < 0.9:
        return "usable"
    return "good"


def _body_refs(body: str) -> List[str]:
    refs = [f"E{match.group(1)}" for match in EVIDENCE_REF_RE.finditer(body)]
    return _unique_strings(refs)


def _draft_quality_reasons(
    outline_sections: List[Dict[str, Any]],
    drafts: List[Dict[str, Any]],
    citation_check: Dict[str, Any],
    missing_sections: List[str],
    warnings: List[str],
) -> List[str]:
    reasons = []
    if outline_sections and not drafts:
        reasons.append("section_draft_missing")
    if missing_sections:
        reasons.append("section_draft_missing")
    if citation_check.get("skipped_sections"):
        reasons.append("section_draft_skipped")
    if citation_check.get("missing_refs"):
        reasons.append("missing_refs")
    if citation_check.get("unused_evidence"):
        reasons.append("unused_evidence")
    if citation_check.get("unsupported_paragraphs"):
        reasons.append("unsupported_paragraphs")
    if float(citation_check.get("coverage_score") or 0.0) < 0.8:
        reasons.append("low_evidence_coverage")
    if any(str(warning).startswith("llm_unavailable") for warning in warnings):
        reasons.append("llm_unavailable")
    if any(str(warning) == "rule_based_section_draft" for warning in warnings):
        reasons.append("rule_based_section_draft")
    return _unique_strings(reasons)


def _draft_quality_level(
    citation_check: Dict[str, Any],
    missing_sections: List[str],
    quality_reasons: List[str],
) -> str:
    coverage = float(citation_check.get("coverage_score") or 0.0)
    structural_reasons = [
        reason
        for reason in quality_reasons
        if reason
        not in {
            "llm_unavailable",
            "rule_based_section_draft",
            "llm_budget_exhausted",
        }
        and not str(reason).startswith("llm_review_draft_stage_budget_exhausted")
    ]
    if "section_draft_missing" in quality_reasons and coverage <= 0:
        return "failed"
    if "section_draft_skipped" in quality_reasons and coverage <= 0:
        return "failed"
    if missing_sections or coverage < 0.5 or "missing_refs" in quality_reasons or "section_draft_skipped" in quality_reasons:
        return "weak"
    if structural_reasons or coverage < 0.9:
        return "usable"
    return "good"


def _unsupported_paragraphs(section_id: str, body: str) -> List[Dict[str, Any]]:
    result = []
    for index, paragraph in enumerate(_checkable_paragraphs(body), start=1):
        if EVIDENCE_REF_RE.search(paragraph):
            continue
        result.append(
            {
                "section_id": section_id,
                "paragraph_index": index,
                "excerpt": _excerpt(paragraph, 180),
            }
        )
    return result


def _checkable_paragraphs(body: str) -> List[str]:
    paragraphs = []
    for raw in re.split(r"\n\s*\n+", body):
        paragraph = compact_whitespace(raw)
        if not paragraph:
            continue
        if paragraph.startswith("#"):
            continue
        if paragraph.startswith("本文为可追溯综述草稿"):
            continue
        if len(paragraph) < 25:
            continue
        paragraphs.append(paragraph)
    return paragraphs


def _coverage_score(supported_count: int, paragraph_count: int, missing_ref_count: int) -> float:
    if paragraph_count <= 0:
        base = 1.0
    else:
        base = supported_count / paragraph_count
    penalty = min(0.5, missing_ref_count * 0.1)
    return round(max(0.0, base - penalty), 3)


def _normalize_claim_plan(value: object) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            result.append({"claim": compact_whitespace(item), "evidence": []})
        elif isinstance(item, dict):
            result.append(
                {
                    "claim": _string_value(item.get("claim")),
                    "evidence": _string_list(item.get("evidence")),
                }
            )
    return result


def _string_value(value: object) -> str:
    return compact_whitespace(str(value)) if value is not None else ""


def _body_value(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = _string_value(item)
        if text:
            result.append(text)
    return result


def _excerpt(text: str, max_chars: int) -> str:
    clean = compact_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + " ..."
