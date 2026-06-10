from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .config import (
    deepseek_api_key,
    deepseek_base_url,
    deepseek_max_tokens,
    deepseek_model,
    deepseek_temperature,
)
from .utils import compact_whitespace, first_words


class LLMError(RuntimeError):
    def __init__(self, message: str, *, error_type: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.error_type = error_type or "llm_error"
        self.metadata = metadata or {}


LLM_METADATA_KEY = "_llm_metadata"


@dataclass
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int


def get_llm_settings() -> Optional[LLMSettings]:
    api_key = deepseek_api_key()
    if not api_key:
        return None
    return LLMSettings(
        api_key=api_key,
        base_url=deepseek_base_url(),
        model=deepseek_model(),
        temperature=deepseek_temperature(),
        max_tokens=deepseek_max_tokens(),
    )


def llm_status(*, probe: bool = False) -> Dict[str, Any]:
    """Return sanitized DeepSeek configuration and optional connectivity state."""
    started = time.time()
    resolved = get_llm_settings()
    if resolved is None:
        return {
            "schema": "llm_status.v1",
            "provider": "deepseek",
            "configured": False,
            "reachable": False if probe else None,
            "probe": probe,
            "base_url": deepseek_base_url(),
            "model": deepseek_model(),
            "temperature": deepseek_temperature(),
            "max_tokens": deepseek_max_tokens(),
            "insecure_http": deepseek_base_url().startswith("http://"),
            "error": "DEEPSEEK_API_KEY is not configured." if probe else "",
            "response_sample": "",
            "latency_ms": round((time.time() - started) * 1000, 3),
        }

    result: Dict[str, Any] = {
        "schema": "llm_status.v1",
        "provider": "deepseek",
        "configured": True,
        "reachable": None,
        "probe": probe,
        "base_url": resolved.base_url,
        "model": resolved.model,
        "temperature": resolved.temperature,
        "max_tokens": resolved.max_tokens,
        "insecure_http": resolved.base_url.startswith("http://"),
        "error": "",
        "response_sample": "",
    }
    if not probe:
        result["latency_ms"] = round((time.time() - started) * 1000, 3)
        return result

    try:
        body = _chat_body(
            resolved,
            [
                {"role": "system", "content": "你是一个连接探针。只回复连接状态。"},
                {"role": "user", "content": "你好，请回复连接正常"},
            ],
        )
        body["temperature"] = 0
        body["max_tokens"] = min(resolved.max_tokens, 300)
        content = _chat_completion_content(body, resolved, timeout=30)
        result["reachable"] = True
        result["response_sample"] = first_words(compact_whitespace(content), 30)
    except LLMError as exc:
        result["reachable"] = False
        result["error"] = str(exc)
    result["latency_ms"] = round((time.time() - started) * 1000, 3)
    return result


def generate_grounded_answer(
    query: str,
    evidence: List[Dict[str, object]],
    settings: Optional[LLMSettings] = None,
) -> str:
    resolved = settings or get_llm_settings()
    if resolved is None:
        raise LLMError("DEEPSEEK_API_KEY is not configured.")
    if not evidence:
        raise LLMError("No evidence was provided to the LLM.")

    body = _chat_body(
        resolved,
        [
            {
                "role": "system",
                "content": (
                    "你是一个严谨的论文知识库问答助手。只允许基于给定证据回答，"
                    "不要编造来源。回答使用中文，关键结论后用 [E1]、[E2] 这样的证据编号标注。"
                    "如果证据不足，明确说明不足，并给出还需要检索什么。"
                ),
            },
            {"role": "user", "content": build_grounded_prompt(query, evidence)},
        ],
    )
    return _chat_completion_content(body, resolved)


def generate_json_object(
    system_prompt: str,
    user_prompt: str,
    settings: Optional[LLMSettings] = None,
) -> Dict[str, object]:
    resolved = settings or get_llm_settings()
    if resolved is None:
        raise LLMError("DEEPSEEK_API_KEY is not configured.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    body = _chat_body(resolved, messages)
    content = _chat_completion_content(body, resolved)
    try:
        payload, metadata = _parse_json_object(content)
        payload[LLM_METADATA_KEY] = {**metadata, "retry_count": 0, "error_type": ""}
        return payload
    except LLMError as first_error:
        retry_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "上一次输出不是可解析的完整 JSON object。请重新生成，只返回完整 JSON object，"
                    "不要解释、不要 Markdown、不要截断，不要输出 JSON 之外的任何文本。"
                ),
            },
        ]
        retry_body = _chat_body(resolved, retry_messages)
        retry_content = _chat_completion_content(retry_body, resolved)
        try:
            payload, metadata = _parse_json_object(retry_content)
        except LLMError as retry_error:
            raise LLMError(
                f"DeepSeek JSON parse failed: {retry_error.error_type}",
                error_type=retry_error.error_type,
                metadata={
                    "retry_count": 1,
                    "repair_used": False,
                    "first_error_type": first_error.error_type,
                    "error_type": retry_error.error_type,
                },
            ) from retry_error
        payload[LLM_METADATA_KEY] = {
            **metadata,
            "retry_count": 1,
            "first_error_type": first_error.error_type,
            "error_type": "",
        }
        return payload


def _chat_body(resolved: LLMSettings, messages: List[Dict[str, str]]) -> Dict[str, object]:
    body: Dict[str, object] = {
        "model": resolved.model,
        "messages": messages,
        "temperature": resolved.temperature,
        "max_tokens": resolved.max_tokens,
        "stream": False,
    }
    thinking = os.environ.get("DEEPSEEK_THINKING", "").strip().lower()
    if thinking in {"enabled", "true", "1", "yes"}:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = os.environ.get("DEEPSEEK_REASONING_EFFORT", "medium")
    return body


def _chat_completion_content(body: Dict[str, object], resolved: LLMSettings, *, timeout: int = 90) -> str:
    request = urllib.request.Request(
        url=f"{resolved.base_url}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved.api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"DeepSeek API request failed: {exc}", error_type="request_failed") from exc
    except TimeoutError as exc:
        raise LLMError("DeepSeek API request timed out.", error_type="request_timeout") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected DeepSeek API response: {payload}") from exc
    if not content:
        raise LLMError("DeepSeek API returned an empty answer.")
    return str(content).strip()


def llm_payload_metadata(payload: Dict[str, object]) -> Dict[str, Any]:
    metadata = payload.get(LLM_METADATA_KEY) if isinstance(payload, dict) else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def _parse_json_object(content: str) -> tuple[Dict[str, object], Dict[str, Any]]:
    text, clean_repair = _clean_json_text(content)
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LLMError("DeepSeek returned invalid JSON.", error_type="invalid_json") from exc
        raise LLMError("DeepSeek JSON response is not an object.", error_type="non_object_json")
    text, slice_repair = _balanced_json_object_text(text)
    repair_used = clean_repair or slice_repair
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError("DeepSeek returned invalid JSON.", error_type="invalid_json") from exc
    if not isinstance(payload, dict):
        raise LLMError("DeepSeek JSON response is not an object.", error_type="non_object_json")
    return payload, {"repair_used": repair_used}


def _clean_json_text(content: str) -> tuple[str, bool]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip(), True
    return text, False


def _balanced_json_object_text(text: str) -> tuple[str, bool]:
    start = text.find("{")
    if start < 0:
        raise LLMError("DeepSeek did not return a JSON object.", error_type="no_json_object")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                return text[start:end], start > 0 or end < len(text.strip())
    raise LLMError("DeepSeek returned truncated JSON.", error_type="truncated_json")


def build_grounded_prompt(query: str, evidence: Iterable[Dict[str, object]]) -> str:
    lines = [
        f"用户问题：{query}",
        "",
        "证据包：",
    ]
    for index, item in enumerate(evidence, start=1):
        title = str(item.get("title") or item.get("doc_id") or "unknown")
        node_path = str(item.get("node_path") or "")
        page_range = item.get("page_range")
        excerpt = first_words(compact_whitespace(str(item.get("excerpt") or "")), 120)
        lines.append(f"[E{index}]")
        lines.append(f"title: {title}")
        lines.append(f"node_path: {node_path}")
        if page_range:
            lines.append(f"page_range: {page_range}")
        lines.append(f"excerpt: {excerpt}")
        lines.append("")
    lines.append("请基于以上证据回答用户问题。")
    return "\n".join(lines)
