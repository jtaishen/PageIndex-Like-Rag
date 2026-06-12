from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .llm import LLMError, generate_json_object, llm_payload_metadata
from .task_evidence import normalize_evidence_refs
from .task_payloads import find_by_id, llm_diagnostics, string_list, string_value
from .task_prompting import format_papers_for_review_prompt, format_review_evidence_line, format_section_evidence_for_review_prompt
from .utils import unique_strings


JsonGenerator = Callable[[str, str], Dict[str, object]]


@dataclass(frozen=True)
class ReviewPlanBuildResult:
    outline: Dict[str, Any]
    llm_error: str
    llm_diagnostics: Dict[str, Any]


def build_review_outline(
    topic: str,
    contexts: List[Dict[str, Any]],
    section_evidence: Dict[str, List[Dict[str, Any]]],
    sections: List[Dict[str, Any]],
    *,
    warnings: List[str],
    use_llm: bool = True,
    require_llm: bool = False,
    prefer_section_llm: bool = False,
    json_generator: JsonGenerator | None = None,
) -> ReviewPlanBuildResult:
    json_generator = json_generator or generate_json_object
    llm_error = ""
    diagnostics = llm_diagnostics("disabled" if not use_llm else "fallback_rule")
    if use_llm:
        try:
            if prefer_section_llm:
                outline = _review_plan_with_section_llm(
                    topic,
                    contexts,
                    section_evidence,
                    sections,
                    warnings=warnings,
                    full_error=LLMError("Baseline fast path uses section JSON.", error_type="section_json_fast_path"),
                    recovery_warning="",
                    json_generator=json_generator,
                )
                diagnostics = outline.get("llm_diagnostics") or llm_diagnostics("section_json")
            else:
                payload = _review_plan_with_llm(topic, contexts, section_evidence, sections, json_generator)
                outline = _normalize_review_payload(
                    payload,
                    topic,
                    contexts,
                    section_evidence,
                    sections,
                    source="llm",
                    warnings=warnings,
                )
                diagnostics = llm_diagnostics("full_json", metadata=llm_payload_metadata(payload))
        except LLMError as exc:
            full_error = exc
            try:
                outline = _review_plan_with_section_llm(
                    topic,
                    contexts,
                    section_evidence,
                    sections,
                    warnings=warnings,
                    full_error=full_error,
                    json_generator=json_generator,
                )
                diagnostics = outline.get("llm_diagnostics") or llm_diagnostics("section_json", error=full_error)
                llm_error = ""
            except LLMError as section_exc:
                if require_llm:
                    raise section_exc
                llm_error = str(section_exc)
                warnings.append(f"llm_unavailable:{llm_error}")
                outline = _rule_based_review_plan(topic, contexts, section_evidence, sections, warnings)
                diagnostics = llm_diagnostics("fallback_rule", error=section_exc, first_error=full_error)
    else:
        warnings.append("llm_disabled")
        outline = _rule_based_review_plan(topic, contexts, section_evidence, sections, warnings)
    outline["llm_diagnostics"] = diagnostics
    return ReviewPlanBuildResult(outline=outline, llm_error=llm_error, llm_diagnostics=diagnostics)


def _review_plan_with_llm(
    topic: str,
    contexts: List[Dict[str, Any]],
    section_evidence: Dict[str, List[Dict[str, Any]]],
    sections: List[Dict[str, Any]],
    json_generator: JsonGenerator,
) -> Dict[str, object]:
    system_prompt = (
        "你是严谨的论文综述规划助手。只能基于给定论文和证据节点生成综述大纲，"
        "不要写完整正文，不要编造。必须返回 JSON object，不要返回 Markdown。"
    )
    user_prompt = "\n".join(
        [
            f"综述主题：{topic}",
            "请生成综述规划。返回格式：",
            '{"title":"","scope":"","sections":[{"section_id":"","title":"","purpose":"",'
            '"paper_ids":[],"evidence":[],"warnings":[]}],"open_questions":[],"warnings":[]}',
            "",
            "建议章节：",
            *[f"- {item['section_id']}: {item['title']}，{item['purpose']}" for item in sections],
            "",
            "论文：",
            *format_papers_for_review_prompt(contexts),
            "",
            "章节证据：",
            *format_section_evidence_for_review_prompt(section_evidence),
        ]
    )
    return json_generator(system_prompt, user_prompt)


def _review_plan_with_section_llm(
    topic: str,
    contexts: List[Dict[str, Any]],
    section_evidence: Dict[str, List[Dict[str, Any]]],
    sections: List[Dict[str, Any]],
    *,
    warnings: List[str],
    full_error: LLMError,
    json_generator: JsonGenerator,
    recovery_warning: str = "section_json_recovery",
) -> Dict[str, Any]:
    normalized_sections = []
    fallback_sections: List[str] = []
    total_retry = int(full_error.metadata.get("retry_count") or 0)
    repair_used = bool(full_error.metadata.get("repair_used"))
    last_error: LLMError | None = None
    success_count = 0
    for spec in sections:
        section_id = str(spec["section_id"])
        evidence = section_evidence.get(section_id, [])
        try:
            payload = _review_section_with_llm(topic, spec, contexts, evidence, json_generator)
            metadata = llm_payload_metadata(payload)
            total_retry += int(metadata.get("retry_count") or 0)
            repair_used = repair_used or bool(metadata.get("repair_used"))
            normalized_sections.append(_normalize_review_section_payload(payload, spec, evidence))
            success_count += 1
        except LLMError as exc:
            last_error = exc
            total_retry += int(exc.metadata.get("retry_count") or 0)
            repair_used = repair_used or bool(exc.metadata.get("repair_used"))
            fallback_sections.append(section_id)
            section = _rule_review_section(spec, evidence)
            section["warnings"] = unique_strings([*section.get("warnings", []), f"llm_section_unavailable:{exc.error_type}"])
            normalized_sections.append(section)
    if success_count == 0:
        error = last_error or full_error
        raise LLMError(
            f"DeepSeek section review failed: {error.error_type}",
            error_type=error.error_type,
            metadata={"retry_count": total_retry, "repair_used": repair_used, "first_error_type": full_error.error_type},
        ) from error
    outline_warnings = [*warnings]
    if recovery_warning:
        outline_warnings.append(recovery_warning)
    if fallback_sections:
        outline_warnings.append("section_json_partial")
    diagnostics = llm_diagnostics(
        "section_json",
        metadata={
            "retry_count": total_retry,
            "repair_used": repair_used,
            "error_type": full_error.error_type,
            "first_error_type": full_error.error_type,
        },
        fallback_sections=fallback_sections,
    )
    return {
        "schema": "review_outline.v1",
        "status": "partial" if fallback_sections else "extracted",
        "source": "llm_section",
        "topic": topic,
        "title": f"{topic}研究综述规划",
        "scope": _review_scope(topic, contexts),
        "sections": normalized_sections,
        "open_questions": _review_open_questions(normalized_sections),
        "warnings": unique_strings(outline_warnings),
        "llm_diagnostics": diagnostics,
        "created_at": time.time(),
    }


def _review_section_with_llm(
    topic: str,
    spec: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    json_generator: JsonGenerator,
) -> Dict[str, object]:
    section_id = str(spec["section_id"])
    system_prompt = (
        "你是严谨的论文综述章节规划助手。只能基于给定论文和证据节点规划一个章节，"
        "不要写正文，不要编造证据。必须返回 JSON object，不要返回 Markdown。"
    )
    user_prompt = "\n".join(
        [
            f"综述主题：{topic}",
            f"section_id: {section_id}",
            f"title: {spec['title']}",
            f"purpose: {spec['purpose']}",
            "返回格式：",
            '{"section_id":"","title":"","purpose":"","paper_ids":[],"evidence":[],"warnings":[]}',
            "证据只能引用下面已有的 node_id 或 evidence id；不要生成长正文。",
            "",
            "论文：",
            *format_papers_for_review_prompt(contexts, limit=2),
            "",
            "本章节候选证据：",
            *[format_review_evidence_line(item) for item in evidence[:6]],
        ]
    )
    return json_generator(system_prompt, user_prompt)


def _rule_based_review_plan(
    topic: str,
    contexts: List[Dict[str, Any]],
    section_evidence: Dict[str, List[Dict[str, Any]]],
    sections: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    normalized_sections = [_rule_review_section(spec, section_evidence.get(str(spec["section_id"]), [])) for spec in sections]
    return {
        "schema": "review_outline.v1",
        "status": "partial",
        "source": "rule",
        "topic": topic,
        "title": f"{topic}研究综述规划",
        "scope": _review_scope(topic, contexts),
        "sections": normalized_sections,
        "open_questions": _review_open_questions(normalized_sections),
        "warnings": unique_strings([*warnings, "rule_based_review_plan"]),
        "created_at": time.time(),
    }


def _rule_review_section(spec: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    paper_ids = unique_strings(str(item.get("doc_id") or "") for item in evidence if item.get("doc_id"))
    section_warnings = [] if evidence else [f"missing_section_evidence:{spec['section_id']}"]
    return {
        "section_id": spec["section_id"],
        "title": spec["title"],
        "purpose": spec["purpose"],
        "paper_ids": paper_ids,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "source_doc_count": len(set(paper_ids)),
        "warnings": section_warnings,
    }


def _normalize_review_payload(
    payload: Dict[str, object],
    topic: str,
    contexts: List[Dict[str, Any]],
    section_evidence: Dict[str, List[Dict[str, Any]]],
    sections: List[Dict[str, Any]],
    *,
    source: str,
    warnings: List[str],
) -> Dict[str, Any]:
    raw_sections = payload.get("sections") if isinstance(payload, dict) else []
    normalized_sections = []
    for spec in sections:
        section_id = str(spec["section_id"])
        raw_section = find_by_id(raw_sections, section_id, key="section_id")
        fallback_evidence = section_evidence.get(section_id, [])
        evidence = normalize_evidence_refs(raw_section.get("evidence") if raw_section else None, fallback_evidence)
        paper_ids = string_list(raw_section.get("paper_ids") if raw_section else None)
        if not paper_ids:
            paper_ids = unique_strings(str(item.get("doc_id") or "") for item in evidence if item.get("doc_id"))
        section_warnings = string_list(raw_section.get("warnings") if raw_section else None)
        if not evidence:
            section_warnings.append(f"missing_section_evidence:{section_id}")
        normalized_sections.append(
            {
                "section_id": section_id,
                "title": string_value(raw_section.get("title") if raw_section else "") or spec["title"],
                "purpose": string_value(raw_section.get("purpose") if raw_section else "") or spec["purpose"],
                "paper_ids": paper_ids,
                "evidence": evidence,
                "evidence_count": len(evidence),
                "source_doc_count": len(set(paper_ids)),
                "warnings": unique_strings(section_warnings),
            }
        )
    return {
        "schema": "review_outline.v1",
        "status": "extracted",
        "source": source,
        "topic": topic,
        "title": string_value(payload.get("title")) or f"{topic}研究综述规划",
        "scope": string_value(payload.get("scope")) or _review_scope(topic, contexts),
        "sections": normalized_sections,
        "open_questions": string_list(payload.get("open_questions")),
        "warnings": unique_strings([*warnings, *string_list(payload.get("warnings"))]),
        "created_at": time.time(),
    }


def _normalize_review_section_payload(
    payload: Dict[str, object],
    spec: Dict[str, Any],
    fallback_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    raw_section = payload.get("section") if isinstance(payload.get("section"), dict) else payload
    if not isinstance(raw_section, dict):
        raw_section = {}
    evidence = normalize_evidence_refs(raw_section.get("evidence"), fallback_evidence)
    paper_ids = string_list(raw_section.get("paper_ids"))
    if not paper_ids:
        paper_ids = unique_strings(str(item.get("doc_id") or "") for item in evidence if item.get("doc_id"))
    warnings = string_list(raw_section.get("warnings"))
    if not evidence:
        warnings.append(f"missing_section_evidence:{spec['section_id']}")
    return {
        "section_id": str(spec["section_id"]),
        "title": string_value(raw_section.get("title")) or str(spec["title"]),
        "purpose": string_value(raw_section.get("purpose")) or str(spec["purpose"]),
        "paper_ids": paper_ids,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "source_doc_count": len(set(paper_ids)),
        "warnings": unique_strings(warnings),
    }


def _review_scope(topic: str, contexts: List[Dict[str, Any]]) -> str:
    titles = "、".join(str(context["title"]) for context in contexts[:5])
    if titles:
        return f"围绕“{topic}”，当前候选论文包括：{titles}。"
    return f"围绕“{topic}”，当前知识库没有检索到可用候选论文。"


def _review_open_questions(sections: List[Dict[str, Any]]) -> List[str]:
    questions = []
    for section in sections:
        if section.get("warnings"):
            questions.append(f"{section['title']}章节缺少足够证据，需要补充文献或重新解析。")
    return unique_strings(questions)
