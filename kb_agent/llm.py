from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from .config import (
    deepseek_api_key,
    deepseek_base_url,
    deepseek_json_retry_count,
    deepseek_max_tokens,
    deepseek_model,
    deepseek_probe_timeout_seconds,
    deepseek_profile,
    deepseek_temperature,
    deepseek_timeout_seconds,
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


@dataclass(frozen=True)
class LLMRuntimeOptions:
    timeout_seconds: Optional[int] = None
    retry_count: Optional[int] = None
    operation: str = ""
    stage: str = ""


LLMEventCollector = Callable[[Dict[str, Any]], None]


_RUNTIME_OPTIONS: ContextVar[LLMRuntimeOptions] = ContextVar("kb_llm_runtime_options", default=LLMRuntimeOptions())
_EVENT_COLLECTOR: ContextVar[Optional[LLMEventCollector]] = ContextVar("kb_llm_event_collector", default=None)


@contextmanager
def llm_runtime_options(
    *,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    operation: str = "",
    stage: str = "",
    event_collector: Optional[LLMEventCollector] = None,
) -> Iterator[None]:
    """Temporarily annotate LLM calls with runtime limits and sanitized telemetry."""
    previous = _RUNTIME_OPTIONS.get()
    options = LLMRuntimeOptions(
        timeout_seconds=timeout_seconds if timeout_seconds is not None else previous.timeout_seconds,
        retry_count=retry_count if retry_count is not None else previous.retry_count,
        operation=operation or previous.operation,
        stage=stage or previous.stage,
    )
    options_token = _RUNTIME_OPTIONS.set(options)
    collector_token = _EVENT_COLLECTOR.set(event_collector if event_collector is not None else _EVENT_COLLECTOR.get())
    try:
        yield
    finally:
        _EVENT_COLLECTOR.reset(collector_token)
        _RUNTIME_OPTIONS.reset(options_token)


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


def llm_status(*, probe: bool = False, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
    """Return sanitized DeepSeek configuration and optional connectivity state."""
    started = time.time()
    resolved = get_llm_settings()
    if resolved is None:
        return {
            "schema": "llm_status.v1",
            "provider": "deepseek",
            "profile": deepseek_profile(),
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
        "profile": deepseek_profile(),
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
        content = _chat_completion_content(
            body,
            resolved,
            timeout=timeout_seconds or deepseek_probe_timeout_seconds(),
            operation="llm_status",
            stage="probe",
        )
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
    *,
    answer_plan: Optional[Dict[str, object]] = None,
    timeout_seconds: Optional[int] = None,
    operation: str = "",
    stage: str = "",
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
            {"role": "user", "content": build_grounded_prompt(query, evidence, answer_plan=answer_plan)},
        ],
    )
    options = _resolve_runtime_options(timeout_seconds=timeout_seconds, operation=operation, stage=stage)
    return _chat_completion_content(
        body,
        resolved,
        timeout=options.timeout_seconds or deepseek_timeout_seconds(),
        operation=options.operation,
        stage=options.stage,
    )


def generate_json_object(
    system_prompt: str,
    user_prompt: str,
    settings: Optional[LLMSettings] = None,
    *,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    max_tokens: Optional[int] = None,
    thinking: Optional[bool] = None,
    operation: str = "",
    stage: str = "",
) -> Dict[str, object]:
    resolved = settings or get_llm_settings()
    if resolved is None:
        raise LLMError("DEEPSEEK_API_KEY is not configured.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    options = _resolve_runtime_options(
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        operation=operation,
        stage=stage,
    )
    timeout = options.timeout_seconds or deepseek_timeout_seconds()
    retries = max(0, options.retry_count if options.retry_count is not None else deepseek_json_retry_count())
    request_max_tokens = (
        min(resolved.max_tokens, max_tokens)
        if max_tokens is not None and max_tokens > 0
        else resolved.max_tokens
    )
    started = time.time()
    body = _chat_body(resolved, messages, max_tokens=max_tokens, thinking=thinking)
    try:
        content = _chat_completion_content(body, resolved, timeout=timeout, operation=options.operation, stage=options.stage)
    except LLMError as exc:
        raise LLMError(
            str(exc),
            error_type=exc.error_type,
            metadata=_llm_error_metadata(
                exc,
                retry_count=0,
                operation=options.operation,
                stage=options.stage,
                started=started,
            ),
        ) from exc
    try:
        payload, metadata = _parse_json_object(content)
        payload[LLM_METADATA_KEY] = {
            **metadata,
            "retry_count": 0,
            "error_type": "",
            "operation": options.operation,
            "stage": options.stage,
            "max_tokens": request_max_tokens,
            "thinking_mode": _thinking_mode(thinking),
            "duration_ms": round((time.time() - started) * 1000, 3),
        }
        return payload
    except LLMError as first_error:
        if retries <= 0:
            raise LLMError(
                f"DeepSeek JSON parse failed: {first_error.error_type}",
                error_type=first_error.error_type,
                metadata=_llm_error_metadata(
                    first_error,
                    retry_count=0,
                    operation=options.operation,
                    stage=options.stage,
                    started=started,
                ),
            ) from first_error
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
        retry_error: Optional[LLMError] = None
        for retry_index in range(1, retries + 1):
            retry_body = _chat_body(resolved, retry_messages, max_tokens=max_tokens, thinking=thinking)
            try:
                retry_content = _chat_completion_content(
                    retry_body,
                    resolved,
                    timeout=timeout,
                    operation=options.operation,
                    stage=options.stage,
                )
                payload, metadata = _parse_json_object(retry_content)
                break
            except LLMError as exc:
                retry_error = exc
        else:
            error = retry_error or first_error
            raise LLMError(
                f"DeepSeek JSON parse failed: {error.error_type}",
                error_type=error.error_type,
                metadata=_llm_error_metadata(
                    error,
                    retry_count=retries,
                    operation=options.operation,
                    stage=options.stage,
                    started=started,
                    first_error_type=first_error.error_type,
                ),
            ) from error
        payload[LLM_METADATA_KEY] = {
            **metadata,
            "retry_count": retry_index,
            "first_error_type": first_error.error_type,
            "error_type": "",
            "operation": options.operation,
            "stage": options.stage,
            "max_tokens": request_max_tokens,
            "thinking_mode": _thinking_mode(thinking),
            "duration_ms": round((time.time() - started) * 1000, 3),
        }
        return payload


def _chat_body(
    resolved: LLMSettings,
    messages: List[Dict[str, str]],
    *,
    max_tokens: Optional[int] = None,
    thinking: Optional[bool] = None,
) -> Dict[str, object]:
    output_tokens = resolved.max_tokens
    if max_tokens is not None and max_tokens > 0:
        output_tokens = min(output_tokens, max_tokens)
    body: Dict[str, object] = {
        "model": resolved.model,
        "messages": messages,
        "temperature": resolved.temperature,
        "max_tokens": output_tokens,
        "stream": False,
    }
    env_thinking = os.environ.get("DEEPSEEK_THINKING", "").strip().lower()
    if thinking is True or (thinking is None and env_thinking in {"enabled", "true", "1", "yes"}):
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = os.environ.get("DEEPSEEK_REASONING_EFFORT", "medium")
    elif thinking is False:
        body["thinking"] = {"type": "disabled"}
    return body


def _chat_completion_content(
    body: Dict[str, object],
    resolved: LLMSettings,
    *,
    timeout: int,
    operation: str = "",
    stage: str = "",
) -> str:
    started = time.time()
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
        _record_llm_event(
            operation=operation,
            stage=stage,
            status="failed",
            error_type="http_error",
            started=started,
            timeout=timeout,
        )
        raise LLMError(
            f"DeepSeek API HTTP {exc.code}.",
            error_type="http_error",
            metadata={"http_status": exc.code, "operation": operation, "stage": stage},
        ) from exc
    except urllib.error.URLError as exc:
        error_type = "request_timeout" if _url_error_is_timeout(exc) else "request_failed"
        _record_llm_event(
            operation=operation,
            stage=stage,
            status="timeout" if error_type == "request_timeout" else "failed",
            error_type=error_type,
            started=started,
            timeout=timeout,
        )
        message = "DeepSeek API request timed out." if error_type == "request_timeout" else "DeepSeek API request failed."
        raise LLMError(message, error_type=error_type, metadata={"operation": operation, "stage": stage}) from exc
    except (TimeoutError, socket.timeout) as exc:
        _record_llm_event(
            operation=operation,
            stage=stage,
            status="timeout",
            error_type="request_timeout",
            started=started,
            timeout=timeout,
        )
        raise LLMError(
            "DeepSeek API request timed out.",
            error_type="request_timeout",
            metadata={"operation": operation, "stage": stage},
        ) from exc
    except json.JSONDecodeError as exc:
        _record_llm_event(
            operation=operation,
            stage=stage,
            status="failed",
            error_type="invalid_response_json",
            started=started,
            timeout=timeout,
        )
        raise LLMError(
            "DeepSeek API returned invalid response JSON.",
            error_type="invalid_response_json",
            metadata={"operation": operation, "stage": stage},
        ) from exc

    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        _record_llm_event(
            operation=operation,
            stage=stage,
            status="failed",
            error_type="unexpected_response",
            started=started,
            timeout=timeout,
        )
        raise LLMError(
            "Unexpected DeepSeek API response.",
            error_type="unexpected_response",
            metadata={"operation": operation, "stage": stage},
        ) from exc
    if not isinstance(message, dict):
        raise LLMError(
            "Unexpected DeepSeek API response.",
            error_type="unexpected_response",
            metadata={"operation": operation, "stage": stage},
        )
    content = message.get("content")
    if not content:
        finish_reason = str(choice.get("finish_reason") or "") if isinstance(choice, dict) else ""
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        output_limit_reached = finish_reason == "length"
        error_type = "output_token_limit" if output_limit_reached else "empty_response"
        _record_llm_event(
            operation=operation,
            stage=stage,
            status="failed",
            error_type=error_type,
            started=started,
            timeout=timeout,
        )
        raise LLMError(
            "DeepSeek output token budget was exhausted before content was produced."
            if output_limit_reached
            else "DeepSeek API returned an empty answer.",
            error_type=error_type,
            metadata={
                "operation": operation,
                "stage": stage,
                "finish_reason": finish_reason,
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_content_present": bool(message.get("reasoning_content")),
            },
        )
    _record_llm_event(
        operation=operation,
        stage=stage,
        status="completed",
        error_type="",
        started=started,
        timeout=timeout,
    )
    return str(content).strip()


def llm_payload_metadata(payload: Dict[str, object]) -> Dict[str, Any]:
    metadata = payload.get(LLM_METADATA_KEY) if isinstance(payload, dict) else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def _resolve_runtime_options(
    *,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    operation: str = "",
    stage: str = "",
) -> LLMRuntimeOptions:
    current = _RUNTIME_OPTIONS.get()
    return LLMRuntimeOptions(
        timeout_seconds=timeout_seconds if timeout_seconds is not None else current.timeout_seconds,
        retry_count=retry_count if retry_count is not None else current.retry_count,
        operation=operation or current.operation,
        stage=stage or current.stage,
    )


def _llm_error_metadata(
    error: LLMError,
    *,
    retry_count: int,
    operation: str,
    stage: str,
    started: float,
    first_error_type: str = "",
) -> Dict[str, Any]:
    metadata = {
        "retry_count": retry_count,
        "repair_used": False,
        "first_error_type": first_error_type,
        "error_type": error.error_type,
        "operation": operation,
        "stage": stage,
        "duration_ms": round((time.time() - started) * 1000, 3),
    }
    for key in ("finish_reason", "completion_tokens", "reasoning_content_present"):
        if key in error.metadata:
            metadata[key] = error.metadata[key]
    return metadata


def _thinking_mode(thinking: Optional[bool]) -> str:
    if thinking is True:
        return "enabled"
    if thinking is False:
        return "disabled"
    return "configured_default"


def _record_llm_event(
    *,
    operation: str,
    stage: str,
    status: str,
    error_type: str,
    started: float,
    timeout: int,
) -> None:
    collector = _EVENT_COLLECTOR.get()
    if collector is None:
        return
    collector(
        {
            "operation": operation,
            "stage": stage,
            "status": status,
            "error_type": error_type,
            "duration_ms": round((time.time() - started) * 1000, 3),
            "timeout_seconds": timeout,
        }
    )


def _url_error_is_timeout(exc: urllib.error.URLError) -> bool:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return True
    return "timed out" in str(reason).lower()


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


def build_grounded_prompt(
    query: str,
    evidence: Iterable[Dict[str, object]],
    *,
    answer_plan: Optional[Dict[str, object]] = None,
) -> str:
    lines = [
        f"用户问题：{query}",
        "",
    ]
    if answer_plan:
        lines.extend(_format_answer_plan_for_prompt(answer_plan))
        lines.append("")
    lines.append("证据包：")
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


def _format_answer_plan_for_prompt(answer_plan: Dict[str, object]) -> List[str]:
    lines = [
        "回答规划：",
        f"answerability: {answer_plan.get('answerability') or ''}",
        f"policy: {answer_plan.get('answer_policy') or ''}",
    ]
    for bucket, label in (
        ("strong_claims", "可作为正式结论"),
        ("qualified_claims", "必须限定表达"),
        ("conflicting_claims", "冲突证据，禁止确定表达"),
        ("insufficient_claims", "证据不足"),
    ):
        items = answer_plan.get(bucket) or []
        if not isinstance(items, list) or not items:
            continue
        lines.append(f"{bucket} ({label}):")
        for item in items[:4]:
            if not isinstance(item, dict):
                continue
            claim = first_words(compact_whitespace(str(item.get("short_claim") or "")), 48)
            evidence_ids = item.get("primary_evidence_unit_ids") or item.get("evidence_unit_ids") or []
            lines.append(
                f"- frame_id={item.get('frame_id') or ''} "
                f"status={item.get('semantic_support_status') or ''} "
                f"risk={item.get('citation_risk') or ''} "
                f"evidence_unit_ids={evidence_ids} claim={claim}"
            )
    return lines
