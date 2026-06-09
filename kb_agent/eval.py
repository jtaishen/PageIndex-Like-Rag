from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .artifacts import get_parse_quality
from .config import DATA_DIR
from .search import search_documents, search_nodes
from .utils import compact_whitespace, write_json


def eval_search(
    db_path: Path,
    queries_path: Path,
    search_mode: str = "hybrid",
    top_k: int = 5,
) -> Dict[str, Any]:
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries, list):
        raise ValueError("Search eval file must be a JSON list.")

    items = []
    doc_recall_values: List[float] = []
    mrr_values: List[float] = []
    keyword_hits = 0
    keyword_total = 0
    evidence_count = 0
    fallback_count = 0
    weak_parse_quality_count = 0

    for raw_item in queries:
        if not isinstance(raw_item, dict):
            continue
        query = str(raw_item.get("query") or "").strip()
        if not query:
            continue
        expected_docs = [str(item) for item in raw_item.get("expected_doc_ids") or []]
        expected_keywords = [str(item) for item in raw_item.get("expected_node_keywords") or []]

        docs = search_documents(db_path, query, top_k=top_k, search_mode=search_mode)
        nodes = search_nodes(db_path, query, top_k=top_k, search_mode=search_mode)
        result_doc_ids = [str(item.get("doc_id")) for item in docs]
        node_text = "\n".join(
            compact_whitespace(f"{node.title} {node.node_path} {node.heading} {node.snippet}")
            for node in nodes
        )

        doc_recall = _recall(result_doc_ids, expected_docs)
        mrr = _mrr(result_doc_ids, expected_docs)
        doc_recall_values.append(doc_recall)
        mrr_values.append(mrr)
        keyword_hit = all(keyword in node_text for keyword in expected_keywords) if expected_keywords else True
        if expected_keywords:
            keyword_total += 1
            if keyword_hit:
                keyword_hits += 1
        evidence_count += len(nodes)
        if any("fts_fallback" in node.rank_reason for node in nodes):
            fallback_count += 1
        for doc_id in set(result_doc_ids):
            try:
                quality = get_parse_quality(db_path, doc_id)
            except (KeyError, FileNotFoundError, ValueError):
                continue
            if quality.get("quality_level") == "weak":
                weak_parse_quality_count += 1

        items.append(
            {
                "query": query,
                "intent": raw_item.get("intent") or "",
                "expected_doc_ids": expected_docs,
                "expected_node_keywords": expected_keywords,
                "result_doc_ids": result_doc_ids,
                "result_node_ids": [node.node_id for node in nodes],
                "doc_recall_at_k": doc_recall,
                "mrr": mrr,
                "node_keyword_hit": keyword_hit,
                "fallback_used": any("fts_fallback" in node.rank_reason for node in nodes),
            }
        )

    report = {
        "schema": "search_eval.v1",
        "queries_path": str(queries_path),
        "search_mode": search_mode,
        "top_k": top_k,
        "query_count": len(items),
        "doc_recall_at_k": _average(doc_recall_values),
        "node_keyword_hit_rate": (keyword_hits / keyword_total) if keyword_total else 1.0,
        "mrr": _average(mrr_values),
        "evidence_count": evidence_count,
        "fallback_count": fallback_count,
        "weak_parse_quality_count": weak_parse_quality_count,
        "items": items,
        "created_at": time.time(),
    }
    out_dir = DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"search_eval_{int(report['created_at'])}.json"
    write_json(out_path, report)
    return {**report, "path": str(out_path)}


def _recall(results: List[str], expected: List[str]) -> float:
    if not expected:
        return 1.0
    hits = len(set(results) & set(expected))
    return hits / len(set(expected))


def _mrr(results: List[str], expected: List[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    for index, doc_id in enumerate(results, start=1):
        if doc_id in expected_set:
            return 1.0 / index
    return 0.0


def _average(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
