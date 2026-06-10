from __future__ import annotations

import json
import re
from typing import Any, List

from .utils import compact_whitespace, unique_strings


def query_terms(query: str) -> List[str]:
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{5,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    return unique_strings(terms)[:12] or [query]


def is_table_source(source: str) -> bool:
    return "table" in source.lower()


def json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value or ""))
    except json.JSONDecodeError:
        return default


def normalize_key(text: str) -> str:
    value = compact_whitespace(text).lower()
    value = value.strip(" \t\r\n.,;:!?，。；：！？…")
    value = re.sub(r"^(?:\.\.\.|…)+", "", value)
    return re.sub(r"\s+", "", value)


def confidence(value: object, default: float) -> float:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        score = default
    return round(max(0.0, min(1.0, score)), 3)


def excerpt(text: str, max_chars: int) -> str:
    cleaned = compact_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + " ..."
