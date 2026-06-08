from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .llm import LLMError, generate_json_object
from .tasks import TASK_ID_RE, _task_state_root
from .utils import compact_whitespace, write_json


EVIDENCE_REF_RE = re.compile(r"\[E(\d+)\]")


def draft_review(
    db_path: Path,
    task_id: str,
    *,
    section_ids: Optional[List[str]] = None,
    use_llm: bool = True,
    require_llm: bool = False,
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
        numbered_evidence = _number_evidence(evidence_artifact.get("evidence") or [])
        if use_llm:
            try:
                payload = _draft_section_with_llm(outline, section, numbered_evidence)
                draft = _normalize_llm_section_draft(task_id, section, numbered_evidence, payload)
            except LLMError as exc:
                if require_llm:
                    raise
                llm_error = str(exc)
                draft = _rule_based_section_draft(
                    task_id,
                    section,
                    numbered_evidence,
                    warnings=[f"llm_unavailable:{llm_error}"],
                )
        else:
            draft = _rule_based_section_draft(
                task_id,
                section,
                numbered_evidence,
                warnings=["llm_disabled"],
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
    task_dir = _task_state_root(db_path) / task_id
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


def _draft_section_with_llm(
    outline: Dict[str, Any],
    section: Dict[str, Any],
    evidence: List[Dict[str, Any]],
) -> Dict[str, object]:
    system_prompt = (
        "你是严谨的中文论文综述写作助手。只能基于给定 evidence 写章节草稿。"
        "正文中每个关键观点必须使用 [E1]、[E2] 这样的证据标记。"
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
    for item in evidence[:14]:
        lines.append(f"[{item.get('ref_id')}]")
        lines.append(f"title: {item.get('title') or item.get('doc_id') or ''}")
        lines.append(f"node_path: {item.get('node_path') or ''}")
        lines.append(f"page_range: {item.get('page_range') or ''}")
        lines.append(f"excerpt: {_excerpt(str(item.get('excerpt') or ''), 700)}")
        lines.append("")
    if not lines:
        lines.append("无可用证据。")
    return lines


def _normalize_llm_section_draft(
    task_id: str,
    section: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    payload: Dict[str, object],
) -> Dict[str, Any]:
    body = _body_value(payload.get("body_markdown") or payload.get("body") or payload.get("draft"))
    warnings = _string_list(payload.get("warnings"))
    if not body:
        warnings.append("empty_llm_body")
        body = _fallback_body(section, evidence)
    return _section_draft(
        task_id,
        section,
        evidence,
        body,
        source="llm",
        status="partial" if "empty_llm_body" in warnings else "drafted",
        claim_plan=_normalize_claim_plan(payload.get("claim_plan")),
        unsupported_claims=_string_list(payload.get("unsupported_claims")),
        warnings=warnings,
    )


def _rule_based_section_draft(
    task_id: str,
    section: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    warnings: List[str],
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
) -> Dict[str, Any]:
    refs = _body_refs(body)
    used_evidence = [item for item in evidence if item.get("ref_id") in refs]
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
    unsupported_paragraphs = []
    scores = []
    for draft in drafts:
        section_id = str(draft.get("section_id") or "")
        body = str(draft.get("body_markdown") or "")
        evidence_refs = {str(item.get("ref_id") or "") for item in draft.get("evidence") or [] if item.get("ref_id")}
        body_refs = _body_refs(body)
        section_missing = sorted(ref for ref in body_refs if ref not in evidence_refs)
        section_unused = sorted(ref for ref in evidence_refs if ref not in body_refs)
        section_unsupported = _unsupported_paragraphs(section_id, body)
        missing_refs.extend({"section_id": section_id, "ref_id": ref} for ref in section_missing)
        unused_evidence.extend({"section_id": section_id, "ref_id": ref} for ref in section_unused)
        unsupported_paragraphs.extend(section_unsupported)
        paragraphs = _checkable_paragraphs(body)
        supported_count = max(0, len(paragraphs) - len(section_unsupported))
        score = _coverage_score(supported_count, len(paragraphs), len(section_missing))
        scores.append(score)
        sections.append(
            {
                "section_id": section_id,
                "title": draft.get("title") or section_id,
                "evidence_count": len(evidence_refs),
                "body_ref_count": len(body_refs),
                "missing_ref_count": len(section_missing),
                "unused_evidence_count": len(section_unused),
                "unsupported_paragraph_count": len(section_unsupported),
                "coverage_score": score,
            }
        )
    overall = round(sum(scores) / len(scores), 3) if scores else 0.0
    warnings = []
    if missing_refs:
        warnings.append("missing_evidence_refs")
    if unsupported_paragraphs:
        warnings.append("unsupported_paragraphs")
    return {
        "schema": "citation_check.v1",
        "task_id": task_id,
        "status": "partial" if warnings else "passed",
        "coverage_score": overall,
        "sections": sections,
        "missing_refs": missing_refs,
        "unused_evidence": unused_evidence,
        "unsupported_paragraphs": unsupported_paragraphs,
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
    warnings = list(citation_check.get("warnings") or [])
    for draft in drafts:
        warnings.extend(draft.get("warnings") or [])
    if missing_sections:
        warnings.append("missing_section_drafts")
    next_actions = []
    if citation_check.get("missing_refs"):
        next_actions.append("修正文中无法映射的证据编号。")
    if citation_check.get("unsupported_paragraphs"):
        next_actions.append("为缺少证据标记的段落补充引用，或删除无证据观点。")
    if missing_sections:
        next_actions.append("为尚未生成的章节运行 draft-review。")
    if not next_actions:
        next_actions.append("人工通读综述草稿，检查章节衔接和引用表达。")
    return {
        "schema": "review_report.v1",
        "task_id": task_id,
        "status": "partial" if warnings or missing_sections else "drafted",
        "title": outline.get("title") or outline.get("topic") or "",
        "section_count": len(outline_sections),
        "drafted_section_count": len(drafts),
        "missing_sections": missing_sections,
        "citation_coverage_score": citation_check.get("coverage_score", 0.0),
        "section_statuses": [
            {
                "section_id": draft.get("section_id") or "",
                "title": draft.get("title") or "",
                "status": draft.get("status") or "",
                "source": draft.get("source") or "",
                "warning_count": len(draft.get("warnings") or []),
            }
            for draft in drafts
        ],
        "warnings": _unique_strings(warnings),
        "next_actions": next_actions,
        "created_at": time.time(),
    }


def _body_refs(body: str) -> List[str]:
    refs = [f"E{match.group(1)}" for match in EVIDENCE_REF_RE.finditer(body)]
    return _unique_strings(refs)


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


def _excerpt(text: str, max_chars: int) -> str:
    clean = compact_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + " ..."
