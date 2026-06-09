from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .llm import LLMError, generate_grounded_answer
from .search import get_evidence, search_documents, search_nodes
from .utils import compact_whitespace, first_words


def answer_query(
    db_path: Path,
    query: str,
    top_k: int = 6,
    use_llm: bool = True,
    require_llm: bool = False,
    search_mode: str = "hybrid",
) -> Dict[str, object]:
    results = search_nodes(db_path, query, top_k=top_k, search_mode=search_mode)
    evidence = []
    for result in results:
        packets = get_evidence(db_path, result.doc_id, [result.node_id])
        evidence.extend(packet.to_dict() for packet in packets)

    lines: List[str] = []
    llm_error: Optional[str] = None
    if not evidence:
        lines.append("没有找到足够证据支持回答。可以先运行 `kb sync <目录>`，或换一个更具体的关键词。")
    elif use_llm:
        try:
            lines.append(generate_grounded_answer(query, evidence))
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            lines.extend(_format_evidence_fallback(evidence))
    else:
        lines.extend(_format_evidence_fallback(evidence))

    return {
        "query": query,
        "search_mode": search_mode,
        "answer": "\n".join(lines),
        "evidence": evidence,
        "llm_error": llm_error,
    }


def route_documents(db_path: Path, query: str, top_k: int = 8, search_mode: str = "hybrid") -> List[Dict[str, object]]:
    return search_documents(db_path, query, top_k=top_k, search_mode=search_mode)


def _format_evidence_fallback(evidence: List[Dict[str, object]]) -> List[str]:
    lines = ["基于当前知识库，找到以下相关证据："]
    for index, item in enumerate(evidence, start=1):
        title = item.get("title") or item["doc_id"]
        path = item.get("node_path") or ""
        excerpt = first_words(compact_whitespace(str(item.get("excerpt", ""))), 52)
        page_range = item.get("page_range")
        page_text = ""
        if page_range and any(page_range):
            page_text = f" 页码: {page_range}"
        lines.append(f"{index}. {title} | {path}{page_text}\n   {excerpt}")
    lines.append("说明：当前未成功调用 DeepSeek，因此只返回可追溯证据。")
    return lines
