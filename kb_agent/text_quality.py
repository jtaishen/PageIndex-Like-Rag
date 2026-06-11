from __future__ import annotations

import re
from typing import List, Optional

from .utils import compact_whitespace, excerpt as _excerpt, unique_strings as _unique_strings


FRONT_MATTER_MARKERS = (
    "网络首发",
    "引用格式",
    "issn",
    "cn ",
    "journal of",
    "中图分类号",
    "文献标志码",
    "收稿日期",
    "基金项目",
    "资助",
    "版权所有",
    "copyright",
    "http://",
    "https://",
    "www.",
    "doi",
    "分类号",
    "学号",
    "密级",
    "作者简介",
    "通信作者",
)

RESEARCH_SIGNAL_TERMS = (
    "提出",
    "方法",
    "算法",
    "模型",
    "框架",
    "机制",
    "系统",
    "实验",
    "结果",
    "验证",
    "提升",
    "降低",
    "优于",
    "局限",
    "不足",
    "任务规划",
    "多智能体",
    "服务机器人",
    "dataset",
    "method",
    "model",
    "framework",
    "experiment",
    "result",
    "limitation",
)


def clean_research_text(text: object) -> str:
    raw = str(text or "")
    raw = _dedupe_repeated_captions(raw)
    pieces = re.split(r"[\n\r。；;]+", raw)
    kept: List[str] = []
    for piece in pieces:
        compacted = compact_whitespace(piece)
        if not compacted:
            continue
        if is_research_noise_text(compacted):
            continue
        if compacted not in kept:
            kept.append(compacted)
    return compact_whitespace(" ".join(kept))


def short_research_text(text: object, max_chars: int) -> str:
    cleaned = clean_research_text(text)
    source = cleaned or compact_whitespace(str(text or ""))
    source = _dedupe_repeated_captions(source)
    return _excerpt(source, max_chars)


def research_noise_reasons(text: object, *, heading: str = "", page: Optional[int] = None, source: str = "") -> List[str]:
    compacted = compact_whitespace(str(text or ""))
    if not compacted:
        return []
    lowered = f"{heading} {source} {compacted}".lower()
    marker_count = sum(1 for marker in FRONT_MATTER_MARKERS if marker in lowered)
    reasons: List[str] = []
    early_page = page in {None, 0, 1, 2}
    if marker_count >= 1 and (early_page or len(compacted) < 300 or marker_count >= 2):
        reasons.append("front_matter")
    if re.search(r"第\s*\d+\s*页|page\s+\d+", lowered):
        reasons.append("page_marker")
    if re.search(r"^(题目|作者|引用格式|基金项目|收稿日期|网络首发|分类号|学号|中图分类号)[:：]", compacted):
        reasons.append("metadata_prefix")
    if _has_repeated_caption(compacted):
        reasons.append("repeated_caption")
    if _looks_like_path_fragment(compacted):
        reasons.append("path_fragment")
    if _looks_like_reference_entry(compacted) and "citation" not in source:
        reasons.append("reference_entry")
    return _unique_strings(reasons)


def is_research_noise_text(text: object, *, heading: str = "", page: Optional[int] = None, source: str = "") -> bool:
    return bool(research_noise_reasons(text, heading=heading, page=page, source=source))


def has_research_signal(text: object) -> bool:
    lowered = compact_whitespace(str(text or "")).lower()
    return any(term.lower() in lowered for term in RESEARCH_SIGNAL_TERMS)


def _dedupe_repeated_captions(text: str) -> str:
    compacted = compact_whitespace(text)
    pattern = re.compile(r"((?:图|表)\s*\d+[\s\S]{2,80}?)(?:\s+\1)+")
    previous = ""
    while previous != compacted:
        previous = compacted
        compacted = pattern.sub(r"\1", compacted)
    return compacted


def _has_repeated_caption(text: str) -> bool:
    compacted = compact_whitespace(text)
    captions = re.findall(r"(图\s*\d+|表\s*\d+|fig(?:ure)?\.?\s*\d+|table\s*\d+)", compacted, flags=re.IGNORECASE)
    return len(captions) >= 2 and len(set(item.lower().replace(" ", "") for item in captions)) < len(captions)


def _looks_like_path_fragment(text: str) -> bool:
    compacted = compact_whitespace(text)
    if compacted.count(">") + compacted.count("/") < 2:
        return False
    return not has_research_signal(compacted)


def _looks_like_reference_entry(text: str) -> bool:
    compacted = compact_whitespace(text)
    if re.match(r"^\[\d+\]", compacted):
        return True
    lowered = compacted.lower()
    return bool(re.search(r"\b(doi|vol\.|no\.|pp\.|journal|proceedings)\b", lowered)) and not has_research_signal(compacted)
