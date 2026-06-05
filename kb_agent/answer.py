from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .search import get_evidence, search_documents, search_nodes
from .utils import compact_whitespace, first_words


def answer_query(db_path: Path, query: str, top_k: int = 6) -> Dict[str, object]:
    results = search_nodes(db_path, query, top_k=top_k)
    evidence = []
    for result in results:
        packets = get_evidence(db_path, result.doc_id, [result.node_id])
        evidence.extend(packet.to_dict() for packet in packets)

    lines: List[str] = []
    if not evidence:
        lines.append("没有找到足够证据支持回答。可以先运行 `kb sync <目录>`，或换一个更具体的关键词。")
    else:
        lines.append("基于当前知识库，找到以下相关证据：")
        for index, item in enumerate(evidence, start=1):
            title = item.get("title") or item["doc_id"]
            path = item.get("node_path") or ""
            excerpt = first_words(compact_whitespace(str(item.get("excerpt", ""))), 52)
            page_range = item.get("page_range")
            page_text = ""
            if page_range and any(page_range):
                page_text = f" 页码: {page_range}"
            lines.append(f"{index}. {title} | {path}{page_text}\n   {excerpt}")
        lines.append("说明：当前 MVP 不直接调用大模型生成最终综述式答案，只返回可追溯证据；接入 OpenCode provider 后可由 agent 基于这些证据综合回答。")

    return {
        "query": query,
        "answer": "\n".join(lines),
        "evidence": evidence,
    }


def route_documents(db_path: Path, query: str, top_k: int = 8) -> List[Dict[str, object]]:
    return search_documents(db_path, query, top_k=top_k)

