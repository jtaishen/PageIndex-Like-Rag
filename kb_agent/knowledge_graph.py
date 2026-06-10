from __future__ import annotations

import json
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .config import DEFAULT_DB_PATH, PROJECT_ROOT
from .fact_audit import audit_facts
from .utils import compact_whitespace, stable_id, write_json


GRAPH_SCHEMA = "knowledge_graph.v1"
GRAPH_INDEX_SCHEMA = "knowledge_graph_index.v1"
GRAPH_REPORT_SCHEMA = "knowledge_graph_report.v1"


def build_knowledge_graph(
    db_path: Path,
    *,
    doc_ids: Optional[List[str]] = None,
    include_conflicts: bool = False,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    clean_doc_ids = _unique_strings(doc_ids or [])
    facts = _load_fact_rows(db_path, clean_doc_ids or None, min_confidence)
    if doc_ids is not None and not clean_doc_ids:
        facts = []
    if not clean_doc_ids:
        clean_doc_ids = _unique_strings(row["doc_id"] for row in facts if row.get("doc_id"))

    created_at = time.time()
    graph_id = stable_id(
        "claim_graph",
        ",".join(clean_doc_ids),
        len(facts),
        include_conflicts,
        min_confidence,
        created_at,
        length=12,
    )
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    for doc in _load_documents(db_path, clean_doc_ids or None):
        _add_node(
            nodes,
            {
                "id": _doc_node_id(doc["doc_id"]),
                "type": "document",
                "label": _short_label(doc.get("title") or doc["doc_id"], 120),
                "doc_id": doc["doc_id"],
                "version_id": "",
                "node_id": "",
                "page_range": [],
                "confidence": 1.0,
                "source": "documents",
                "metadata": {
                    "file_type": doc.get("file_type") or "",
                    "path": doc.get("path") or "",
                },
            },
        )

    entity_by_name: Dict[tuple[str, str], str] = {}
    for fact in facts:
        fact_node = _fact_node(fact)
        _add_node(nodes, fact_node)
        doc_node = _doc_node_id(fact["doc_id"])
        if doc_node in nodes:
            edge_type = {
                "claim": "has_claim",
                "entity": "mentions_entity",
                "relation": "has_relation",
            }.get(fact["fact_type"], "has_fact")
            _add_edge(edges, _edge(doc_node, fact_node["id"], edge_type, fact))
        evidence_node = _evidence_node(fact)
        if evidence_node:
            _add_node(nodes, evidence_node)
            _add_edge(edges, _edge(fact_node["id"], evidence_node["id"], "backed_by", fact))
        if fact["fact_type"] == "entity":
            entity_by_name[(fact["doc_id"], _normalize_key(fact.get("text") or fact.get("normalized_key")))] = fact_node["id"]
            entity_by_name[("", _normalize_key(fact.get("text") or fact.get("normalized_key")))] = fact_node["id"]

    relation_facts = [fact for fact in facts if fact.get("fact_type") == "relation"]
    fact_by_id = {fact["fact_id"]: _graph_fact_id(fact) for fact in facts}
    for fact in relation_facts:
        relation_node_id = _graph_fact_id(fact)
        subject_id = fact_by_id.get(str(fact.get("subject_id") or ""))
        object_id = fact_by_id.get(str(fact.get("object_id") or ""))
        if not subject_id:
            subject_id = entity_by_name.get((fact["doc_id"], _normalize_key(fact.get("subject_name"))))
            subject_id = subject_id or entity_by_name.get(("", _normalize_key(fact.get("subject_name"))))
        if not object_id:
            object_id = entity_by_name.get((fact["doc_id"], _normalize_key(fact.get("object_name"))))
            object_id = object_id or entity_by_name.get(("", _normalize_key(fact.get("object_name"))))
        for linked_id, edge_type in ((subject_id, _relation_edge_type(fact.get("type"))), (object_id, _relation_edge_type(fact.get("type")))):
            if linked_id and linked_id in nodes:
                _add_edge(edges, _edge(relation_node_id, linked_id, edge_type, fact))

    conflict_count = 0
    if include_conflicts:
        audit = audit_facts(db_path, doc_ids=clean_doc_ids or None, min_confidence=min_confidence, write_report=False)
        for conflict in audit.get("conflicts") or []:
            conflict_node = _conflict_node(conflict)
            _add_node(nodes, conflict_node)
            conflict_count += 1
            for side in ("left", "right"):
                ref = conflict.get(side) or {}
                linked_id = _graph_fact_id(ref)
                if linked_id in nodes:
                    _add_edge(edges, _conflict_edge(conflict_node["id"], linked_id, conflict, ref))
        warnings.extend(audit.get("warnings") or [])

    isolated = _isolated_fact_nodes(nodes, edges)
    low_confidence = [node for node in nodes.values() if node["type"] in {"claim", "entity", "relation"} and float(node["confidence"]) < min_confidence]
    evidence_linked = sum(1 for node in nodes.values() if node["type"] in {"claim", "entity", "relation"} and _has_backing_edge(node["id"], edges))
    fact_node_count = sum(1 for node in nodes.values() if node["type"] in {"claim", "entity", "relation"})
    warnings.extend(_graph_warnings(facts, isolated, conflict_count))

    graph = {
        "schema": GRAPH_SCHEMA,
        "graph_id": graph_id,
        "doc_ids": clean_doc_ids,
        "include_conflicts": include_conflicts,
        "min_confidence": min_confidence,
        "nodes": sorted(nodes.values(), key=lambda item: (item["type"], item["id"])),
        "edges": sorted(edges.values(), key=lambda item: (item["type"], item["source"], item["target"])),
        "warnings": _unique_strings(warnings),
        "created_at": created_at,
    }
    index = _graph_index(graph)
    report = _graph_report(graph, isolated, low_confidence, evidence_linked, fact_node_count)
    graph_dir = _graph_dir(db_path, graph_id)
    write_json(graph_dir / "knowledge_graph.json", graph)
    write_json(graph_dir / "graph_index.json", index)
    write_json(graph_dir / "graph_report.json", report)
    return {
        "schema": "knowledge_graph_build_result.v1",
        "graph_id": graph_id,
        "graph_dir": str(graph_dir),
        "knowledge_graph_path": str(graph_dir / "knowledge_graph.json"),
        "graph_index_path": str(graph_dir / "graph_index.json"),
        "graph_report_path": str(graph_dir / "graph_report.json"),
        "knowledge_graph": graph,
        "graph_index": index,
        "graph_report": report,
        "warnings": graph["warnings"],
    }


def get_knowledge_graph(db_path: Path, graph_id: str) -> Dict[str, Any]:
    return _read_graph_artifact(db_path, graph_id, "knowledge_graph.json", GRAPH_SCHEMA)


def get_graph_report(db_path: Path, graph_id: str) -> Dict[str, Any]:
    return _read_graph_artifact(db_path, graph_id, "graph_report.json", GRAPH_REPORT_SCHEMA)


def get_graph_neighborhood(db_path: Path, node_or_fact_id: str, *, depth: int = 1, graph_id: Optional[str] = None) -> Dict[str, Any]:
    graph = get_knowledge_graph(db_path, graph_id or _latest_graph_id(db_path))
    seed = _resolve_seed(graph, node_or_fact_id)
    if not seed:
        raise KeyError(f"Graph node not found: {node_or_fact_id}")
    max_depth = max(0, min(int(depth), 4))
    adjacency: Dict[str, List[Dict[str, Any]]] = {}
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        adjacency.setdefault(str(edge.get("source") or ""), []).append(edge)
        adjacency.setdefault(str(edge.get("target") or ""), []).append(edge)
    visited = {seed}
    frontier = {seed}
    edge_ids = set()
    for _ in range(max_depth):
        next_frontier = set()
        for node_id in frontier:
            for edge in adjacency.get(node_id, []):
                edge_ids.add(str(edge.get("id") or ""))
                other = str(edge.get("target") if edge.get("source") == node_id else edge.get("source"))
                if other and other not in visited:
                    visited.add(other)
                    next_frontier.add(other)
        frontier = next_frontier
        if not frontier:
            break
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict) and node.get("id") in visited]
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict) and edge.get("id") in edge_ids]
    return {
        "schema": "knowledge_graph_neighborhood.v1",
        "graph_id": graph.get("graph_id") or "",
        "seed": node_or_fact_id,
        "resolved_seed": seed,
        "depth": max_depth,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "warnings": graph.get("warnings") or [],
    }


def export_knowledge_graph(db_path: Path, graph_id: str, *, format: str = "json") -> Dict[str, Any]:
    graph = get_knowledge_graph(db_path, graph_id)
    fmt = compact_whitespace(format or "json").lower()
    if fmt not in {"json", "mermaid", "html"}:
        raise ValueError("format must be one of: json, mermaid, html")
    graph_dir = _graph_dir(db_path, graph_id)
    if fmt == "json":
        path = graph_dir / "knowledge_graph.json"
        content = json.dumps(graph, ensure_ascii=False, indent=2)
    elif fmt == "mermaid":
        content = _mermaid_graph(graph)
        path = graph_dir / "knowledge_graph.mmd"
        path.write_text(content, encoding="utf-8")
    else:
        content = _html_graph(graph)
        path = graph_dir / "knowledge_graph.html"
        path.write_text(content, encoding="utf-8")
    return {
        "schema": "knowledge_graph_export.v1",
        "graph_id": graph_id,
        "format": fmt,
        "path": str(path),
        "node_count": len(graph.get("nodes") or []),
        "edge_count": len(graph.get("edges") or []),
        "content": content if fmt == "mermaid" else "",
    }


def graph_summary(
    db_path: Path,
    *,
    doc_ids: Optional[List[str]] = None,
    include_conflicts: bool = True,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    try:
        result = build_knowledge_graph(
            db_path,
            doc_ids=doc_ids,
            include_conflicts=include_conflicts,
            min_confidence=min_confidence,
        )
    except Exception as exc:
        return {
            "schema": "knowledge_graph_summary.v1",
            "available": False,
            "error": str(exc),
            "warnings": ["knowledge_graph_summary_failed"],
        }
    report = result.get("graph_report") or {}
    return {
        "schema": "knowledge_graph_summary.v1",
        "available": True,
        "graph_id": result.get("graph_id") or "",
        "graph_dir": result.get("graph_dir") or "",
        "doc_ids": report.get("doc_ids") or [],
        "node_count": report.get("node_count", 0),
        "edge_count": report.get("edge_count", 0),
        "claim_count": report.get("type_counts", {}).get("claim", 0),
        "entity_count": report.get("type_counts", {}).get("entity", 0),
        "relation_count": report.get("type_counts", {}).get("relation", 0),
        "conflict_count": report.get("conflict_count", 0),
        "isolated_fact_count": report.get("isolated_fact_count", 0),
        "evidence_coverage_rate": report.get("evidence_coverage_rate", 0.0),
        "noisy_entity_count": report.get("noisy_entity_count", 0),
        "top_entities": report.get("top_entities") or [],
        "warnings": report.get("warnings") or [],
    }


def latest_graph_reports(db_path: Optional[Path] = None, limit: int = 5) -> List[Dict[str, Any]]:
    roots = []
    if db_path is not None:
        roots.append(_graph_root(db_path))
    roots.append(PROJECT_ROOT / ".kb_state" / "graphs")
    seen_paths = set()
    items: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/graph_report.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            payload = _read_json(path, {})
            if payload.get("schema") != GRAPH_REPORT_SCHEMA:
                continue
            items.append(
                {
                    "path": str(path),
                    "graph_id": payload.get("graph_id") or path.parent.name,
                    "doc_count": payload.get("doc_count", 0),
                    "node_count": payload.get("node_count", 0),
                    "edge_count": payload.get("edge_count", 0),
                    "conflict_count": payload.get("conflict_count", 0),
                    "isolated_fact_count": payload.get("isolated_fact_count", 0),
                    "evidence_coverage_rate": payload.get("evidence_coverage_rate", 0.0),
                    "warnings": payload.get("warnings") or [],
                    "created_at": payload.get("created_at"),
                }
            )
            if len(items) >= limit:
                return items
    return items


def _load_fact_rows(db_path: Path, doc_ids: Optional[List[str]], min_confidence: float) -> List[Dict[str, Any]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        filters: List[str] = []
        params: List[Any] = []
        if doc_ids:
            filters.append(f"doc_id IN ({','.join('?' for _ in doc_ids)})")
            params.extend(doc_ids)
        if min_confidence > 0:
            filters.append("confidence >= ?")
            params.append(float(min_confidence))
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        claims = [
            _normalize_fact_row(dict(row), "claim")
            for row in conn.execute(
                f"""
                SELECT claim_id AS fact_id, 'claim' AS fact_type, doc_id, version_id, node_id,
                       claim_type AS type, text, normalized_text AS normalized_key,
                       page_range, confidence, source, evidence_json,
                       '' AS subject_id, '' AS subject_name, '' AS object_id, '' AS object_name
                FROM paper_claims {where}
                """,
                params,
            ).fetchall()
        ]
        entities = [
            _normalize_fact_row(dict(row), "entity")
            for row in conn.execute(
                f"""
                SELECT entity_id AS fact_id, 'entity' AS fact_type, doc_id, version_id, node_id,
                       entity_type AS type, name AS text, normalized_name AS normalized_key,
                       page_range, confidence, source, evidence_json,
                       '' AS subject_id, '' AS subject_name, '' AS object_id, '' AS object_name
                FROM paper_entities {where}
                """,
                params,
            ).fetchall()
        ]
        relations = [
            _normalize_fact_row(dict(row), "relation")
            for row in conn.execute(
                f"""
                SELECT relation_id AS fact_id, 'relation' AS fact_type, doc_id, version_id, node_id,
                       relation_type AS type, text, '' AS normalized_key,
                       page_range, confidence, source, evidence_json,
                       subject_id, subject_name, object_id, object_name
                FROM paper_relations {where}
                """,
                params,
            ).fetchall()
        ]
    finally:
        conn.close()
    return [*claims, *entities, *relations]


def _load_documents(db_path: Path, doc_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        if doc_ids:
            rows = conn.execute(
                f"SELECT doc_id, title, path, file_type FROM documents WHERE doc_id IN ({','.join('?' for _ in doc_ids)})",
                doc_ids,
            ).fetchall()
        else:
            rows = conn.execute("SELECT doc_id, title, path, file_type FROM documents").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _normalize_fact_row(row: Dict[str, Any], fact_type: str) -> Dict[str, Any]:
    text = compact_whitespace(str(row.get("text") or ""))
    subject = compact_whitespace(str(row.get("subject_name") or ""))
    obj = compact_whitespace(str(row.get("object_name") or ""))
    source = compact_whitespace(str(row.get("source") or ""))
    evidence = _json_value(row.get("evidence_json"), {})
    return {
        "fact_id": str(row.get("fact_id") or ""),
        "fact_type": fact_type,
        "doc_id": str(row.get("doc_id") or ""),
        "version_id": str(row.get("version_id") or ""),
        "node_id": str(row.get("node_id") or ""),
        "type": str(row.get("type") or ""),
        "text": _short_label(text or subject or obj, 220),
        "normalized_key": _normalize_key(row.get("normalized_key") or text or subject or obj),
        "page_range": _json_value(row.get("page_range"), []),
        "confidence": float(row.get("confidence") or 0.0),
        "source": source,
        "source_kind": "table" if "table" in source.lower() else "text",
        "evidence": evidence if isinstance(evidence, dict) else {},
        "subject_id": str(row.get("subject_id") or ""),
        "subject_name": subject,
        "object_id": str(row.get("object_id") or ""),
        "object_name": obj,
    }


def _fact_node(fact: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "fact_id": fact["fact_id"],
        "fact_type": fact["fact_type"],
        "kind": fact.get("type") or "",
        "source_kind": fact.get("source_kind") or "",
    }
    if fact.get("subject_name") or fact.get("object_name"):
        metadata["subject_name"] = _short_label(str(fact.get("subject_name") or ""), 100)
        metadata["object_name"] = _short_label(str(fact.get("object_name") or ""), 100)
    evidence = fact.get("evidence") or {}
    for key in ("table_id", "layout_block_id", "caption_id", "ref_id"):
        if evidence.get(key):
            metadata[key] = str(evidence[key])
    return {
        "id": _graph_fact_id(fact),
        "type": fact["fact_type"],
        "label": _short_label(_fact_label(fact), 180),
        "doc_id": fact["doc_id"],
        "version_id": fact["version_id"],
        "node_id": fact["node_id"],
        "page_range": fact.get("page_range") or [],
        "confidence": fact["confidence"],
        "source": fact["source"],
        "metadata": metadata,
    }


def _fact_label(fact: Dict[str, Any]) -> str:
    if fact["fact_type"] == "relation":
        subject = fact.get("subject_name") or fact.get("subject_id") or "subject"
        obj = fact.get("object_name") or fact.get("object_id") or "object"
        return f"{subject} {fact.get('type') or 'related_to'} {obj}"
    return str(fact.get("text") or fact.get("fact_id") or "")


def _evidence_node(fact: Dict[str, Any]) -> Dict[str, Any]:
    node_id = str(fact.get("node_id") or "")
    if not node_id:
        return {}
    return {
        "id": _evidence_graph_id(fact["doc_id"], node_id),
        "type": "evidence",
        "label": _short_label(node_id, 120),
        "doc_id": fact["doc_id"],
        "version_id": fact["version_id"],
        "node_id": node_id,
        "page_range": fact.get("page_range") or [],
        "confidence": 1.0,
        "source": "doc_nodes",
        "metadata": {
            "table_id": str((fact.get("evidence") or {}).get("table_id") or ""),
            "layout_block_id": str((fact.get("evidence") or {}).get("layout_block_id") or ""),
        },
    }


def _conflict_node(conflict: Dict[str, Any]) -> Dict[str, Any]:
    left = conflict.get("left") or {}
    right = conflict.get("right") or {}
    return {
        "id": f"conflict:{conflict.get('conflict_id')}",
        "type": "conflict",
        "label": _short_label(str(conflict.get("reason") or conflict.get("anchor") or conflict.get("conflict_id")), 180),
        "doc_id": "",
        "version_id": "",
        "node_id": "",
        "page_range": [],
        "confidence": min(float(left.get("confidence") or 0.0), float(right.get("confidence") or 0.0)),
        "source": "fact_audit",
        "metadata": {
            "conflict_id": conflict.get("conflict_id") or "",
            "kind": conflict.get("kind") or "",
            "severity": conflict.get("severity") or "",
            "anchor": conflict.get("anchor") or "",
        },
    }


def _edge(source: str, target: str, edge_type: str, fact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": stable_id("kg_edge", source, target, edge_type, fact.get("fact_id"), length=14),
        "type": edge_type,
        "source": source,
        "target": target,
        "doc_id": fact.get("doc_id") or "",
        "version_id": fact.get("version_id") or "",
        "node_id": fact.get("node_id") or "",
        "page_range": fact.get("page_range") or [],
        "confidence": fact.get("confidence", 0.0),
        "source_label": fact.get("source") or "",
        "metadata": {
            "fact_id": fact.get("fact_id") or "",
            "fact_type": fact.get("fact_type") or "",
        },
    }


def _conflict_edge(source: str, target: str, conflict: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": stable_id("kg_edge", source, target, "conflicts_with", conflict.get("conflict_id"), length=14),
        "type": "conflicts_with",
        "source": source,
        "target": target,
        "doc_id": ref.get("doc_id") or "",
        "version_id": ref.get("version_id") or "",
        "node_id": ref.get("node_id") or "",
        "page_range": ref.get("page_range") or [],
        "confidence": ref.get("confidence", 0.0),
        "source_label": "fact_audit",
        "metadata": {
            "conflict_id": conflict.get("conflict_id") or "",
            "severity": conflict.get("severity") or "",
            "reason": conflict.get("reason") or "",
        },
    }


def _graph_index(graph: Dict[str, Any]) -> Dict[str, Any]:
    by_type: Dict[str, List[str]] = {}
    by_doc: Dict[str, List[str]] = {}
    by_fact_id: Dict[str, str] = {}
    by_node_id: Dict[str, List[str]] = {}
    for node in graph.get("nodes") or []:
        node_id = str(node.get("id") or "")
        by_type.setdefault(str(node.get("type") or ""), []).append(node_id)
        if node.get("doc_id"):
            by_doc.setdefault(str(node["doc_id"]), []).append(node_id)
        metadata = node.get("metadata") or {}
        if metadata.get("fact_id"):
            by_fact_id[str(metadata["fact_id"])] = node_id
        if node.get("node_id"):
            by_node_id.setdefault(str(node["node_id"]), []).append(node_id)
    return {
        "schema": GRAPH_INDEX_SCHEMA,
        "graph_id": graph.get("graph_id") or "",
        "by_type": by_type,
        "by_doc": by_doc,
        "by_fact_id": by_fact_id,
        "by_node_id": by_node_id,
        "created_at": time.time(),
    }


def _graph_report(
    graph: Dict[str, Any],
    isolated: List[Dict[str, Any]],
    low_confidence: List[Dict[str, Any]],
    evidence_linked: int,
    fact_node_count: int,
) -> Dict[str, Any]:
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    type_counts: Dict[str, int] = {}
    edge_type_counts: Dict[str, int] = {}
    entity_counts: Dict[str, int] = {}
    noisy_entity_count = 0
    for node in nodes:
        node_type = str(node.get("type") or "")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
        if node_type == "entity":
            label = str(node.get("label") or "")
            if _looks_like_noisy_entity_label(label):
                noisy_entity_count += 1
            else:
                entity_counts[label] = entity_counts.get(label, 0) + 1
    for edge in edges:
        edge_type = str(edge.get("type") or "")
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    conflict_count = type_counts.get("conflict", 0)
    report = {
        "schema": GRAPH_REPORT_SCHEMA,
        "graph_id": graph.get("graph_id") or "",
        "doc_ids": graph.get("doc_ids") or [],
        "doc_count": len(graph.get("doc_ids") or []),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "type_counts": type_counts,
        "edge_type_counts": edge_type_counts,
        "conflict_count": conflict_count,
        "isolated_fact_count": len(isolated),
        "isolated_facts": [_node_ref(item) for item in isolated[:50]],
        "low_confidence_count": len(low_confidence),
        "low_confidence_nodes": [_node_ref(item) for item in low_confidence[:50]],
        "noisy_entity_count": noisy_entity_count,
        "evidence_coverage_rate": round(evidence_linked / max(1, fact_node_count), 4),
        "top_entities": [
            {"label": key, "count": value}
            for key, value in sorted(entity_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
        "warnings": graph.get("warnings") or [],
        "created_at": graph.get("created_at"),
    }
    return report


def _looks_like_noisy_entity_label(value: str) -> bool:
    text = compact_whitespace(value).strip(" ,，.。;；:：()（）[]【】")
    lowered = text.lower()
    if not text:
        return True
    if lowered in {"no", "no.", "ra", "rb", "rc", "rd"}:
        return True
    if len(text) <= 2 and re_ascii_token(text):
        return True
    if re_contains_sentence_punctuation(text):
        return True
    chinese_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    allowed_suffix = ("方法", "算法", "模型", "框架", "系统", "平台", "数据集", "指标", "任务", "场景", "机制", "模块")
    if chinese_count > 24 and not text.endswith(allowed_suffix):
        return True
    if len(text) > 18 and any(token in text for token in ("则", "并", "以及", "进行", "涵盖", "包括", "通过", "用于")) and not text.endswith(allowed_suffix):
        return True
    return False


def re_ascii_token(text: str) -> bool:
    return all(char.isascii() and (char.isalnum() or char == ".") for char in text)


def re_contains_sentence_punctuation(text: str) -> bool:
    return any(char in text for char in "。！？!?；;\n")


def _isolated_fact_nodes(nodes: Dict[str, Dict[str, Any]], edges: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    linked = set()
    for edge in edges.values():
        linked.add(str(edge.get("source") or ""))
        linked.add(str(edge.get("target") or ""))
    return [
        node
        for node in nodes.values()
        if node.get("type") in {"claim", "entity", "relation"} and node.get("id") not in linked
    ]


def _has_backing_edge(node_id: str, edges: Dict[str, Dict[str, Any]]) -> bool:
    return any(edge.get("source") == node_id and edge.get("type") == "backed_by" for edge in edges.values())


def _node_ref(node: Dict[str, Any]) -> Dict[str, Any]:
    metadata = node.get("metadata") or {}
    return {
        "id": node.get("id") or "",
        "type": node.get("type") or "",
        "label": node.get("label") or "",
        "doc_id": node.get("doc_id") or "",
        "node_id": node.get("node_id") or "",
        "page_range": node.get("page_range") or [],
        "confidence": node.get("confidence", 0.0),
        "source": node.get("source") or "",
        "fact_id": metadata.get("fact_id") or "",
    }


def _graph_warnings(facts: List[Dict[str, Any]], isolated: List[Dict[str, Any]], conflict_count: int) -> List[str]:
    warnings = []
    if not facts:
        warnings.append("no_facts_for_graph")
    if isolated:
        warnings.append("isolated_fact_nodes")
    if conflict_count:
        warnings.append("graph_contains_fact_conflicts")
    if any(not fact.get("node_id") for fact in facts):
        warnings.append("graph_contains_unbacked_facts")
    return warnings


def _mermaid_graph(graph: Dict[str, Any]) -> str:
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    allowed_nodes = {str(node.get("id") or "") for node in nodes[:80]}
    lines = ["graph TD"]
    for node in nodes[:80]:
        mermaid_id = _mermaid_id(str(node.get("id") or ""))
        label = _short_label(f"{node.get('type')}:{node.get('label')}", 80).replace('"', "'")
        lines.append(f'  {mermaid_id}["{label}"]')
    for edge in edges[:160]:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in allowed_nodes and target in allowed_nodes:
            lines.append(f"  {_mermaid_id(source)} -->|{edge.get('type')}| {_mermaid_id(target)}")
    return "\n".join(lines) + "\n"


def _html_graph(graph: Dict[str, Any]) -> str:
    report = _graph_report(
        graph,
        _isolated_fact_nodes({str(node.get("id")): node for node in graph.get("nodes") or []}, {str(edge.get("id")): edge for edge in graph.get("edges") or []}),
        [],
        0,
        0,
    )
    node_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(node.get('type') or ''))}</td>"
        f"<td>{escape(str(node.get('label') or ''))}</td>"
        f"<td>{escape(str(node.get('doc_id') or ''))}</td>"
        f"<td>{escape(str(node.get('node_id') or ''))}</td>"
        f"<td>{escape(str(node.get('confidence') or ''))}</td>"
        "</tr>"
        for node in (graph.get("nodes") or [])[:300]
        if isinstance(node, dict)
    )
    edge_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(edge.get('type') or ''))}</td>"
        f"<td>{escape(str(edge.get('source') or ''))}</td>"
        f"<td>{escape(str(edge.get('target') or ''))}</td>"
        f"<td>{escape(str(edge.get('confidence') or ''))}</td>"
        "</tr>"
        for edge in (graph.get("edges") or [])[:500]
        if isinstance(edge, dict)
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Claim Graph {escape(str(graph.get('graph_id') or ''))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f7f8fa; color: #1f2933; }}
    header {{ background: #fff; border-bottom: 1px solid #d9dee7; padding: 24px 32px; }}
    main {{ padding: 24px 32px 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card, section {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ border-top: 1px solid #edf1f5; padding: 8px; font-size: 12px; overflow-wrap: anywhere; text-align: left; }}
    h1 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
    h2 {{ font-size: 16px; }}
  </style>
</head>
<body>
  <header>
    <h1>Claim Graph</h1>
    <p>{escape(str(graph.get('graph_id') or ''))}</p>
  </header>
  <main>
    <div class="grid">
      <div class="card"><div>Nodes</div><div class="value">{report.get('node_count')}</div></div>
      <div class="card"><div>Edges</div><div class="value">{report.get('edge_count')}</div></div>
      <div class="card"><div>Conflicts</div><div class="value">{report.get('conflict_count')}</div></div>
      <div class="card"><div>Isolated Facts</div><div class="value">{report.get('isolated_fact_count')}</div></div>
    </div>
    <section><h2>Nodes</h2><table><thead><tr><th>Type</th><th>Label</th><th>Doc</th><th>Evidence Node</th><th>Confidence</th></tr></thead><tbody>{node_rows}</tbody></table></section>
    <section><h2>Edges</h2><table><thead><tr><th>Type</th><th>Source</th><th>Target</th><th>Confidence</th></tr></thead><tbody>{edge_rows}</tbody></table></section>
  </main>
</body>
</html>
"""


def _read_graph_artifact(db_path: Path, graph_id: str, name: str, schema: str) -> Dict[str, Any]:
    graph = compact_whitespace(graph_id)
    if not graph:
        graph = _latest_graph_id(db_path)
    path = _graph_dir(db_path, graph) / name
    if not path.exists():
        raise FileNotFoundError(f"Graph artifact not found: {path}")
    payload = _read_json(path, {})
    if payload.get("schema") != schema:
        raise ValueError(f"Unsupported graph artifact schema in {path}")
    return {**payload, "path": str(path)}


def _latest_graph_id(db_path: Path) -> str:
    root = _graph_root(db_path)
    candidates = sorted(root.glob("*/graph_report.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No knowledge graph has been built")
    return candidates[0].parent.name


def _resolve_seed(graph: Dict[str, Any], seed: str) -> str:
    raw = compact_whitespace(seed)
    node_ids = {str(node.get("id") or "") for node in graph.get("nodes") or [] if isinstance(node, dict)}
    if raw in node_ids:
        return raw
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata") or {}
        if raw and raw in {str(metadata.get("fact_id") or ""), str(metadata.get("conflict_id") or ""), str(node.get("node_id") or "")}:
            return str(node.get("id") or "")
    return ""


def _add_node(nodes: Dict[str, Dict[str, Any]], node: Dict[str, Any]) -> None:
    node_id = str(node.get("id") or "")
    if node_id and node_id not in nodes:
        nodes[node_id] = node


def _add_edge(edges: Dict[str, Dict[str, Any]], edge: Dict[str, Any]) -> None:
    edge_id = str(edge.get("id") or "")
    if edge_id and edge.get("source") and edge.get("target"):
        edges[edge_id] = edge


def _doc_node_id(doc_id: str) -> str:
    return f"doc:{doc_id}"


def _graph_fact_id(fact: Dict[str, Any]) -> str:
    return f"{fact.get('fact_type') or 'fact'}:{fact.get('fact_id') or ''}"


def _evidence_graph_id(doc_id: str, node_id: str) -> str:
    return f"evidence:{doc_id}:{node_id}"


def _relation_edge_type(value: Any) -> str:
    text = compact_whitespace(str(value or "")).lower()
    if text in {"uses", "evaluates_on", "reports_metric", "cites", "compares_with", "improves"}:
        return text
    return "relates_to"


def _graph_root(db_path: Path) -> Path:
    resolved = db_path.expanduser().resolve()
    if resolved == DEFAULT_DB_PATH.expanduser().resolve():
        return PROJECT_ROOT / ".kb_state" / "graphs"
    return resolved.parent / ".kb_state" / "graphs"


def _graph_dir(db_path: Path, graph_id: str) -> Path:
    return _graph_root(db_path) / compact_whitespace(graph_id)


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _short_label(value: Any, limit: int = 160) -> str:
    text = compact_whitespace(str(value or ""))
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_key(value: Any) -> str:
    text = compact_whitespace(str(value or "")).lower()
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")[:120]


def _mermaid_id(value: str) -> str:
    return "n_" + "".join(ch if ch.isalnum() else "_" for ch in value)[:80]


def _unique_strings(values: Iterable[Any]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = compact_whitespace(str(value))
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
