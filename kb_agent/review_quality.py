from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from .utils import compact_whitespace, unique_strings as _unique_strings


EVIDENCE_REF_RE = re.compile(r"\[E(\d+)\]")


def assemble_review_markdown(outline: Dict[str, Any], drafts: List[Dict[str, Any]]) -> str:
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


def build_citation_check(task_id: str, drafts: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        body_refs = body_refs_from_markdown(body)
        section_missing = sorted(ref for ref in body_refs if ref not in evidence_refs)
        section_unused = sorted(ref for ref in evidence_refs if ref not in body_refs)
        section_unsupported = unsupported_paragraphs_for_section(section_id, body)
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
        paragraphs = checkable_paragraphs(body)
        supported_count = max(0, len(paragraphs) - len(section_unsupported))
        score = 0.0 if draft.get("status") == "skipped" else coverage_score(supported_count, len(paragraphs), len(section_missing))
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


def build_review_report(
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
    quality_reasons = draft_quality_reasons(outline_sections, drafts, citation_check, missing_sections, warnings)
    draft_quality_level = draft_quality_level_from_reasons(citation_check, missing_sections, quality_reasons)
    revision_actions_by_section = section_revision_actions(outline_sections, drafts, citation_check)
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
        "paragraph_support_report": review_paragraph_support_report(drafts),
        "optional_unused_evidence_count": len(citation_check.get("optional_unused_evidence") or []),
        "section_statuses": [
            {
                "section_id": draft.get("section_id") or "",
                "title": draft.get("title") or "",
                "status": draft.get("status") or "",
                "source": draft.get("source") or "",
                "draft_quality_level": section_quality_level(str(draft.get("section_id") or ""), citation_check, draft),
                "warning_count": len(draft.get("warnings") or []),
                "warnings": draft.get("warnings") or [],
            }
            for draft in drafts
        ],
        "warnings": _unique_strings(warnings),
        "next_actions": next_actions,
        "revision_actions": next_actions,
        "section_revision_actions": revision_actions_by_section,
        "created_at": time.time(),
    }


def clean_unsupported_body(body: str) -> tuple[str, Dict[str, Any]]:
    kept: List[str] = []
    removed: List[str] = []
    for raw in re.split(r"\n\s*\n+", body.strip()):
        paragraph = raw.strip()
        if not paragraph:
            continue
        if paragraph_requires_evidence(paragraph) and not EVIDENCE_REF_RE.search(paragraph):
            removed.append(excerpt(paragraph, 180))
            continue
        kept.append(paragraph)
    return "\n\n".join(kept).strip(), {
        "schema": "paragraph_support_cleanup.v1",
        "removed_paragraph_count": len(removed),
        "removed_paragraphs": removed,
    }


def paragraph_support_report(
    body: str,
    evidence: List[Dict[str, Any]],
    *,
    cleanup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cleanup = cleanup or {}
    paragraphs = checkable_paragraphs(body)
    supported = [paragraph for paragraph in paragraphs if EVIDENCE_REF_RE.search(paragraph)]
    evidence_refs = {str(item.get("ref_id") or "") for item in evidence if item.get("ref_id")}
    body_refs = set(body_refs_from_markdown(body))
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
        "coverage_score": coverage_score(len(supported), len(paragraphs), 0),
    }


def paragraph_requires_evidence(paragraph: str) -> bool:
    text = compact_whitespace(paragraph)
    if not text:
        return False
    if text.startswith("#") or text.startswith("- "):
        return False
    if len(text) < 25:
        return False
    return True


def body_refs_from_markdown(body: str) -> List[str]:
    refs = [f"E{match.group(1)}" for match in EVIDENCE_REF_RE.finditer(body)]
    return _unique_strings(refs)


def unsupported_paragraphs_for_section(section_id: str, body: str) -> List[Dict[str, Any]]:
    result = []
    for index, paragraph in enumerate(checkable_paragraphs(body), start=1):
        if EVIDENCE_REF_RE.search(paragraph):
            continue
        result.append(
            {
                "section_id": section_id,
                "paragraph_index": index,
                "excerpt": excerpt(paragraph, 180),
            }
        )
    return result


def checkable_paragraphs(body: str) -> List[str]:
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


def coverage_score(supported_count: int, paragraph_count: int, missing_ref_count: int) -> float:
    if paragraph_count <= 0:
        base = 1.0
    else:
        base = supported_count / paragraph_count
    penalty = min(0.5, missing_ref_count * 0.1)
    return round(max(0.0, base - penalty), 3)


def section_revision_actions(
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
        optional_unused = optional_unused_count(section_id, citation_check)
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
                "draft_quality_level": section_quality_level(section_id, citation_check, draft),
                "quality_reasons": _unique_strings(reasons),
                "actions": _unique_strings(actions),
            }
        )
    return result


def optional_unused_count(section_id: str, citation_check: Dict[str, Any]) -> int:
    count = 0
    for item in citation_check.get("optional_unused_evidence") or []:
        if isinstance(item, dict) and str(item.get("section_id") or "") == section_id:
            count += 1
    return count


def review_paragraph_support_report(drafts: List[Dict[str, Any]]) -> Dict[str, Any]:
    section_reports = []
    paragraph_count = 0
    supported_count = 0
    unsupported_count = 0
    removed_count = 0
    unused_count = 0
    for draft in drafts:
        report = draft.get("paragraph_support_report") or {}
        if not isinstance(report, dict):
            report = paragraph_support_report(str(draft.get("body_markdown") or ""), draft.get("evidence") or [])
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
        "coverage_score": coverage_score(supported_count, paragraph_count, 0),
        "sections": section_reports,
    }


def section_quality_level(section_id: str, citation_check: Dict[str, Any], draft: Dict[str, Any]) -> str:
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


def draft_quality_reasons(
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


def draft_quality_level_from_reasons(
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


def excerpt(text: str, max_chars: int) -> str:
    clean = compact_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + " ..."
