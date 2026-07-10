from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .config import mcp_llm_step_timeout_seconds, mcp_review_draft_max_tokens
from .llm import generate_json_object


JsonGenerator = Callable[[str, str], Dict[str, object]]
JsonBackend = Callable[..., Dict[str, object]]


@dataclass(frozen=True)
class StructuredLLMPolicy:
    timeout_seconds: int
    retry_count: int
    max_tokens: int
    thinking: bool


_FAST_TIMEOUT_SECONDS = 25
_POLICIES = {
    "doc_card_summary": StructuredLLMPolicy(15, 1, 700, False),
    "query_classification": StructuredLLMPolicy(15, 1, 500, False),
    "tree_search": StructuredLLMPolicy(20, 1, 700, False),
    "insight_extraction": StructuredLLMPolicy(_FAST_TIMEOUT_SECONDS, 1, 1200, False),
    "fact_extraction": StructuredLLMPolicy(_FAST_TIMEOUT_SECONDS, 1, 1200, False),
    "claim_frame": StructuredLLMPolicy(_FAST_TIMEOUT_SECONDS, 1, 1200, False),
    "compare": StructuredLLMPolicy(_FAST_TIMEOUT_SECONDS, 1, 1600, False),
    "review_outline": StructuredLLMPolicy(_FAST_TIMEOUT_SECONDS, 1, 900, False),
    "review_draft": StructuredLLMPolicy(_FAST_TIMEOUT_SECONDS, 1, 900, False),
}


def structured_llm_policy(operation: str) -> StructuredLLMPolicy:
    policy = _POLICIES.get(operation)
    if policy is None:
        raise ValueError(f"Unknown structured LLM operation: {operation}")
    max_tokens = mcp_review_draft_max_tokens() if operation == "review_draft" else policy.max_tokens
    return StructuredLLMPolicy(
        timeout_seconds=min(policy.timeout_seconds, mcp_llm_step_timeout_seconds()),
        retry_count=policy.retry_count,
        max_tokens=max_tokens,
        thinking=policy.thinking,
    )


def structured_json_generator(
    operation: str,
    stage: str,
    *,
    json_generator: Optional[JsonBackend] = None,
) -> JsonGenerator:
    policy = structured_llm_policy(operation)
    backend = json_generator or generate_json_object

    def generate(system_prompt: str, user_prompt: str) -> Dict[str, object]:
        return backend(
            system_prompt,
            user_prompt,
            timeout_seconds=policy.timeout_seconds,
            retry_count=policy.retry_count,
            max_tokens=policy.max_tokens,
            thinking=policy.thinking,
            operation=operation,
            stage=stage,
        )

    return generate
