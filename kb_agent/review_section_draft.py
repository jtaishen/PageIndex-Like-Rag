from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from .llm import LLMError, generate_json_object, llm_payload_metadata
from .review_quality import body_refs_from_markdown, clean_unsupported_body, excerpt, paragraph_support_report
from .utils import compact_whitespace, stable_id, unique_strings as _unique_strings


MAX_PROMPT_EVIDENCE = 5
MAX_PROMPT_SUMMARY_CHARS = 240

JsonGenerator = Callable[[str, str], Dict[str, object]]


@dataclass(frozen=True)
class SectionDraftBuildResult:
    draft: Dict[str, Any]
    llm_error: str = ""


def prepare_numbered_draft_evidence(
    evidence: Iterable[Dict[str, Any]],
    *,
    artifact_compaction: Optional[Dict[str, Any]] = None,
    max_items: int = MAX_PROMPT_EVIDENCE,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prepared_evidence, draft_compaction = _prepare_draft_evidence(evidence, max_items=max_items)
    numbered_evidence = _number_evidence(prepared_evidence)
    compaction = _merge_compaction_reports(artifact_compaction or {}, draft_compaction)
    return numbered_evidence, compaction


def build_section_draft(
    task_id: str,
    outline: Dict[str, Any],
    section: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    compaction: Dict[str, Any],
    *,
    use_llm: bool = True,
    require_llm: bool = False,
    rule_warnings: Optional[List[str]] = None,
    fallback_reason: str = "llm_disabled",
    json_generator: JsonGenerator | None = None,
) -> SectionDraftBuildResult:
    json_generator = json_generator or generate_json_object
    if use_llm:
        try:
            payload = _draft_section_with_llm(outline, section, evidence, json_generator)
            draft = _normalize_llm_section_draft(
                task_id,
                section,
                evidence,
                payload,
                llm_diagnostics=section_draft_llm_diagnostics(
                    used=True,
                    metadata=llm_payload_metadata(payload),
                    evidence_count=len(evidence),
                    compaction=compaction,
                ),
            )
            return SectionDraftBuildResult(draft=draft)
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            draft = _rule_based_section_draft(
                task_id,
                section,
                evidence,
                warnings=[f"llm_unavailable:{llm_error}"],
                llm_diagnostics=section_draft_llm_diagnostics(
                    used=False,
                    error=exc,
                    fallback_reason=getattr(exc, "error_type", "") or "llm_error",
                    evidence_count=len(evidence),
                    compaction=compaction,
                ),
            )
            return SectionDraftBuildResult(draft=draft, llm_error=llm_error)

    draft = _rule_based_section_draft(
        task_id,
        section,
        evidence,
        warnings=rule_warnings if rule_warnings is not None else ["llm_disabled"],
        llm_diagnostics=section_draft_llm_diagnostics(
            used=False,
            fallback_reason=fallback_reason,
            evidence_count=len(evidence),
            compaction=compaction,
        ),
    )
    return SectionDraftBuildResult(draft=draft)


def build_skipped_section_draft(
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
        llm_diagnostics=section_draft_llm_diagnostics(
            used=False,
            fallback_reason=reason,
            evidence_count=len(evidence),
            compaction=compaction,
        ),
    )


def section_draft_llm_diagnostics(
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
    diagnostics = {
        "schema": "section_draft_llm_diagnostics.v1",
        "used": used,
        "retry_count": int(metadata.get("retry_count") or 0),
        "repair_used": bool(metadata.get("repair_used")),
        "max_tokens": int(metadata.get("max_tokens") or 0),
        "thinking_mode": str(metadata.get("thinking_mode") or ""),
        "error_type": (getattr(error, "error_type", "") if error else str(metadata.get("error_type") or "")),
        "fallback_reason": fallback_reason,
        "evidence_count": evidence_count,
        "compact_count": int(compaction.get("kept_evidence_count") or evidence_count),
        "duplicate_evidence_removed": int(compaction.get("duplicate_evidence_removed") or 0),
        "compaction_warnings": _string_list(compaction.get("warnings")),
        "compaction": compaction,
    }
    for key in (
        "duration_ms",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_content_present",
        "operation",
        "stage",
    ):
        if key in metadata:
            diagnostics[key] = metadata[key]
    return diagnostics


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
    json_generator: JsonGenerator,
) -> Dict[str, object]:
    system_prompt = (
        "你是严谨的中文论文综述写作助手。只能基于给定 evidence 写章节草稿。"
        "正文中每个自然段至少必须包含一个 [E1]、[E2] 这样的证据标记。"
        "不能被证据支持的内容必须放入 unsupported_claims，不允许写入正文。"
        "正文写成 2 至 3 个简洁自然段，总长度不超过 500 个中文字符；claim_plan 最多 4 项。"
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
    return json_generator(system_prompt, user_prompt)


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
    return excerpt(text, MAX_PROMPT_SUMMARY_CHARS)


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
        body, cleanup_report = clean_unsupported_body(body)
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
            "claim": excerpt(str(item.get("excerpt") or ""), 180),
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
    refs = body_refs_from_markdown(body)
    used_evidence = [item for item in evidence if item.get("ref_id") in refs]
    support_report = paragraph_support_report(body, evidence, cleanup=paragraph_cleanup)
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
        "paragraph_support_report": support_report,
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
        evidence_excerpt = excerpt(str(item.get("excerpt") or ""), 180)
        source = item.get("title") or item.get("doc_id") or "未知文档"
        path = item.get("node_path") or ""
        lines.append(f"- 《{source}》在“{path}”中提供了相关证据：{evidence_excerpt} [{item['ref_id']}]")
    return "\n".join(lines)


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
