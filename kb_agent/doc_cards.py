from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import ingest_artifacts
from .parse_quality import build_parse_quality
from .text_quality import clean_research_text, is_research_noise_text


DOC_CARD_ARTIFACTS = [
    "raw_text.txt",
    "body.md",
    "structured.json",
    "metadata.json",
    "references.json",
    "layout_blocks.json",
    "tables.json",
    "table_content.json",
    "table_summaries.json",
    "figures.json",
    "reference_sections.json",
    "parse_report.json",
    "tree.json",
    "node_index.jsonl",
    "doc_card.json",
    "innovation.json",
    "citation_map.json",
]

DOC_CARD_SUMMARY_LIMITS = {
    "description": 180,
    "method_summary": 220,
    "innovation_summary": 220,
    "limitation_summary": 220,
}


def build_doc_card(
    doc_id: str,
    version_id: str,
    base: Path,
    source_path: Path,
    file_hash: str,
    parsed: Any,
    nodes: List[Any],
    *,
    doc_card_use_llm: bool = True,
    json_generator: Callable[..., Dict[str, Any]],
    llm_settings_getter: Callable[[], object],
    llm_error_cls: type[Exception],
    components: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    resolved_components = components or ingest_artifacts.artifact_components(parsed)
    layout_blocks = resolved_components["layout_blocks"]
    tables = resolved_components["tables"]
    table_content = resolved_components["table_content"]
    figures = resolved_components["figures"]
    parse_quality = build_parse_quality(
        doc_id,
        version_id,
        parsed,
        nodes,
        layout_blocks,
        tables,
        table_content,
        figures,
    )
    section_nodes = [node for node in nodes if node.kind == "section"]
    sections = [
        {
            "node_id": node.node_id,
            "title": node.heading,
            "node_path": node.node_path,
            "page_range": [node.page_start, node.page_end],
            "level": node.level,
            "type": node.kind,
        }
        for node in section_nodes[:80]
    ]
    summaries = doc_card_summaries(
        parsed,
        use_llm=doc_card_use_llm,
        json_generator=json_generator,
        llm_settings_getter=llm_settings_getter,
        llm_error_cls=llm_error_cls,
    )
    description = str(summaries.get("description") or doc_description(parsed))
    return {
        "schema": "doc_card.v0",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": parsed.title,
        "path": str(source_path),
        "file_type": parsed.file_type,
        "file_hash": file_hash,
        "summary": description,
        "description": description,
        "method_summary": summaries.get("method_summary", ""),
        "innovation_summary": summaries.get("innovation_summary", ""),
        "limitation_summary": summaries.get("limitation_summary", ""),
        "summary_source": summaries.get("summary_source", "rule"),
        "summary_warnings": summaries.get("summary_warnings", []),
        "authors": parsed.metadata.get("authors") or [],
        "year": parsed.metadata.get("year"),
        "venue": parsed.metadata.get("venue") or "",
        "doi": parsed.metadata.get("doi") or "",
        "abstract": parsed.metadata.get("abstract") or "",
        "keywords": parsed.metadata.get("keywords") or [],
        "parser_name": parsed.parser_name,
        "parser_version": parsed.parser_version,
        "page_count": parse_quality["page_count"],
        "block_count": len(parsed.blocks),
        "node_count": len(nodes),
        "section_count": parse_quality["section_count"],
        "sections": sections,
        "quality_warnings": parse_quality["quality_warnings"],
        "parse_quality": parse_quality,
        "artifact_dir": str(base),
        "artifacts": DOC_CARD_ARTIFACTS,
        "created_at": time.time(),
    }


def doc_description(parsed: Any) -> str:
    abstract = str(parsed.metadata.get("abstract") or "").strip()
    if abstract:
        return content_excerpt(abstract, 500)
    keywords = parsed.metadata.get("keywords") or []
    if keywords:
        return text_excerpt(f"{parsed.title}。关键词：{'、'.join(str(item) for item in keywords)}", 500)
    return content_excerpt(parsed.raw_text, 500)


def doc_card_summaries(
    parsed: Any,
    *,
    use_llm: bool = True,
    json_generator: Callable[..., Dict[str, Any]],
    llm_settings_getter: Callable[[], object],
    llm_error_cls: type[Exception],
) -> Dict[str, object]:
    fallback = rule_doc_card_summaries(parsed)
    if not use_llm:
        return {
            **fallback,
            "summary_source": "rule",
            "summary_warnings": ["deepseek_summary_skipped:disabled"],
        }
    if llm_settings_getter() is None:
        return {
            **fallback,
            "summary_source": "rule",
            "summary_warnings": ["deepseek_summary_skipped:not_configured"],
        }
    try:
        payload = json_generator(
            "你是论文知识库的文档卡片摘要器。只返回 JSON object，不要输出正文、证据长摘录或 prompt。",
            doc_card_summary_prompt(parsed, fallback),
            timeout_seconds=20,
            retry_count=1,
            operation="doc_card_summary",
            stage="ingest",
        )
    except llm_error_cls as exc:
        return {
            **fallback,
            "summary_source": "rule",
            "summary_warnings": [f"deepseek_summary_failed:{getattr(exc, 'error_type', 'llm_error')}"],
        }

    summaries: Dict[str, object] = dict(fallback)
    used_llm = False
    for field, max_chars in DOC_CARD_SUMMARY_LIMITS.items():
        value = short_summary(payload.get(field), max_chars)
        if value:
            summaries[field] = value
            used_llm = True
    if not used_llm:
        return {
            **fallback,
            "summary_source": "rule",
            "summary_warnings": ["deepseek_summary_failed:empty_summary"],
        }
    summaries["summary_source"] = "deepseek"
    summaries["summary_warnings"] = []
    return summaries


def rule_doc_card_summaries(parsed: Any) -> Dict[str, object]:
    return {
        "description": text_excerpt(doc_description(parsed), DOC_CARD_SUMMARY_LIMITS["description"]),
        "method_summary": section_excerpt(
            parsed,
            ("method", "approach", "方法", "方法设计", "框架", "模型", "算法", "系统设计"),
            DOC_CARD_SUMMARY_LIMITS["method_summary"],
        ),
        "innovation_summary": section_excerpt(
            parsed,
            ("contribution", "innovation", "novel", "贡献", "创新", "提出", "研究内容"),
            DOC_CARD_SUMMARY_LIMITS["innovation_summary"],
        ),
        "limitation_summary": section_excerpt(
            parsed,
            ("limitation", "future", "局限", "不足", "未来", "结论"),
            DOC_CARD_SUMMARY_LIMITS["limitation_summary"],
        ),
        "summary_source": "rule",
        "summary_warnings": [],
    }


def doc_card_summary_prompt(parsed: Any, fallback: Dict[str, object]) -> str:
    payload = {
        "instruction": "生成短字段：description、method_summary、innovation_summary、limitation_summary。每项不超过 80 个中文字或 50 个英文词。",
        "title": parsed.title,
        "abstract": text_excerpt(str(parsed.metadata.get("abstract") or ""), 700),
        "keywords": [str(item) for item in (parsed.metadata.get("keywords") or [])][:12],
        "section_signals": section_signals(parsed, limit=10),
        "fallback": {field: fallback.get(field, "") for field in DOC_CARD_SUMMARY_LIMITS},
    }
    return json.dumps(payload, ensure_ascii=False)


def section_signals(parsed: Any, limit: int) -> List[Dict[str, str]]:
    signals: List[Dict[str, str]] = []
    current_heading = ""
    for block in parsed.blocks:
        heading = str(getattr(block, "heading", "") or "")
        text = str(getattr(block, "text", "") or "")
        kind = str(getattr(block, "kind", "") or "")
        if kind == "heading" and heading:
            current_heading = heading
            continue
        if not text.strip() or is_doc_card_noise_text(text, heading=current_heading, page=getattr(block, "page", None)):
            continue
        if current_heading or kind in {"abstract", "paragraph"}:
            cleaned = clean_doc_card_text(text)
            if not cleaned:
                continue
            signals.append(
                {
                    "heading": current_heading or kind,
                    "text": text_excerpt(cleaned, 240),
                }
            )
        if len(signals) >= limit:
            break
    return signals


def section_excerpt(parsed: Any, heading_terms: Iterable[str], max_chars: int) -> str:
    terms = [term.lower() for term in heading_terms]
    current_heading = ""
    matched: List[str] = []
    for block in parsed.blocks:
        heading = str(getattr(block, "heading", "") or "")
        text = str(getattr(block, "text", "") or "")
        kind = str(getattr(block, "kind", "") or "")
        if kind == "heading" and heading:
            current_heading = heading
            continue
        if is_doc_card_noise_text(text, heading=current_heading, page=getattr(block, "page", None)):
            continue
        text = clean_doc_card_text(text)
        haystack = f"{current_heading} {text}".lower()
        if text.strip() and any(term and term in haystack for term in terms):
            matched.append(text)
        if matched and len(" ".join(matched)) >= max_chars:
            break
    return text_excerpt(" ".join(matched), max_chars)


def short_summary(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    compacted = " ".join(value.split())
    source = clean_research_text(compacted) if is_doc_card_noise_text(compacted) else compacted
    cleaned = text_excerpt(source, max_chars)
    banned_markers = ("```", "prompt", "原文：", "正文：")
    if any(marker.lower() in cleaned.lower() for marker in banned_markers):
        return ""
    if is_doc_card_noise_text(cleaned):
        return ""
    return cleaned


def content_excerpt(text: str, max_chars: int) -> str:
    cleaned = clean_doc_card_text(text)
    return text_excerpt(cleaned, max_chars) if cleaned else ""


def clean_doc_card_text(text: object) -> str:
    return clean_research_text(text)


def is_doc_card_noise_text(text: object, *, heading: str = "", page: Optional[int] = None) -> bool:
    return is_research_noise_text(text, heading=heading, page=page)


def text_excerpt(text: str, max_chars: int) -> str:
    compacted = " ".join(str(text).split())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[:max_chars].rstrip() + " ..."
