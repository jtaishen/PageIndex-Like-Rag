from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .config import (
    deepseek_api_key,
    deepseek_base_url,
    deepseek_max_tokens,
    deepseek_model,
    deepseek_temperature,
)
from .utils import compact_whitespace, first_words


class LLMError(RuntimeError):
    pass


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

    body: Dict[str, object] = {
        "model": resolved.model,
        "messages": [
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
        "temperature": resolved.temperature,
        "max_tokens": resolved.max_tokens,
        "stream": False,
    }

    thinking = os.environ.get("DEEPSEEK_THINKING", "").strip().lower()
    if thinking in {"enabled", "true", "1", "yes"}:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = os.environ.get("DEEPSEEK_REASONING_EFFORT", "medium")

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
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"DeepSeek API request failed: {exc}") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected DeepSeek API response: {payload}") from exc
    if not content:
        raise LLMError("DeepSeek API returned an empty answer.")
    return str(content).strip()


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

