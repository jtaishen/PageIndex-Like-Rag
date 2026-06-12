from __future__ import annotations

from typing import Any, Dict, List, Optional

from .llm import LLMError
from .utils import compact_whitespace


def find_by_id(raw_items: object, expected_id: str, key: str = "id") -> Dict[str, Any]:
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict) and str(item.get(key) or "") == expected_id:
                return item
    return {}


def find_by_doc_id(raw_items: object, doc_id: str) -> Dict[str, Any]:
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict) and str(item.get("doc_id") or "") == doc_id:
                return item
    return {}


def confidence(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def string_value(value: object) -> str:
    return compact_whitespace(str(value)) if value is not None else ""


def string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = string_value(item)
        if text:
            result.append(text)
    return result


def llm_diagnostics(
    mode: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[LLMError] = None,
    first_error: Optional[LLMError] = None,
    fallback_sections: Optional[List[str]] = None,
    fallback_dimensions: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = metadata or {}
    error_type = str(meta.get("error_type") or "")
    if error is not None:
        error_type = error.error_type
        meta = {**error.metadata, **meta}
    result = {
        "schema": "llm_diagnostics.v1",
        "mode": mode,
        "retry_count": int(meta.get("retry_count") or 0),
        "repair_used": bool(meta.get("repair_used")),
        "fallback_sections": fallback_sections or [],
        "fallback_dimensions": fallback_dimensions or [],
        "error_type": error_type,
    }
    if extra:
        result.update(extra)
    first_type = str(meta.get("first_error_type") or "")
    if first_error is not None:
        first_type = first_error.error_type
    if first_type:
        result["first_error_type"] = first_type
    return result
