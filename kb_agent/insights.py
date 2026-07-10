from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .artifacts import get_artifact, get_doc_card, get_parse_quality, list_artifacts
from .llm import LLMError, generate_json_object
from .llm_policies import structured_json_generator
from .utils import compact_whitespace, write_json


INSIGHT_NODE_TOKENS = (
    "摘要",
    "研究内容",
    "主要贡献",
    "研究贡献",
    "创新",
    "方法",
    "算法",
    "模型",
    "框架",
    "实验",
    "结果",
    "结论",
    "局限",
    "不足",
    "展望",
)
LIMITATION_TOKENS = ("局限", "不足", "限制", "展望", "未来工作", "讨论")
REFERENCE_MARK_RE = re.compile(r"\[(\d+(?:\s*(?:[-,，、]\s*)\d+)*)\]")
JsonGenerator = Callable[[str, str], Dict[str, object]]


def extract_doc_insights(
    db_path: Path,
    doc_id: str,
    *,
    force: bool = False,
    use_llm: bool = True,
    require_llm: bool = False,
) -> Dict[str, Any]:
    listing = list_artifacts(db_path, doc_id)
    artifact_dir = Path(str(listing["artifact_dir"]))
    existing = read_existing_insights(db_path, doc_id)
    if existing and not force:
        return {
            "doc_id": doc_id,
            "version_id": listing["version_id"],
            "artifact_dir": str(artifact_dir),
            "skipped": True,
            "innovation": existing["innovation"],
            "citation_map": existing["citation_map"],
        }

    inputs = load_insight_extraction_inputs(db_path, doc_id)
    card = inputs["card"]
    quality = inputs["quality"]
    nodes = inputs["nodes"]
    references = inputs["references"]
    selected_nodes = inputs["selected_nodes"]
    warnings: List[str] = []
    llm_error = ""

    if use_llm:
        try:
            llm_payload = extract_innovation_with_llm(card, quality, selected_nodes)
            innovation = normalize_innovation_payload(
                llm_payload,
                doc_id=doc_id,
                version_id=str(listing["version_id"]),
                card=card,
                quality=quality,
                selected_nodes=selected_nodes,
                status="extracted",
                warnings=[],
            )
        except LLMError as exc:
            if require_llm:
                raise
            llm_error = str(exc)
            warnings.append(f"llm_unavailable:{llm_error}")
            innovation = _rule_based_innovation(
                doc_id,
                str(listing["version_id"]),
                card,
                quality,
                selected_nodes,
                warnings,
            )
    else:
        warnings.append("llm_disabled")
        innovation = _rule_based_innovation(
            doc_id,
            str(listing["version_id"]),
            card,
            quality,
            selected_nodes,
            warnings,
        )

    citation_map = build_citation_map(
        doc_id,
        str(listing["version_id"]),
        card,
        references,
        nodes,
    )
    write_json(artifact_dir / "innovation.json", innovation)
    write_json(artifact_dir / "citation_map.json", citation_map)
    return {
        "doc_id": doc_id,
        "version_id": listing["version_id"],
        "artifact_dir": str(artifact_dir),
        "skipped": False,
        "innovation_path": str(artifact_dir / "innovation.json"),
        "citation_map_path": str(artifact_dir / "citation_map.json"),
        "innovation": innovation,
        "citation_map": citation_map,
        "llm_error": llm_error,
    }


def load_insight_extraction_inputs(db_path: Path, doc_id: str) -> Dict[str, Any]:
    listing = list_artifacts(db_path, doc_id)
    nodes = get_artifact(db_path, doc_id, "node_index.jsonl")["content"]
    return {
        "listing": listing,
        "doc_id": doc_id,
        "version_id": str(listing["version_id"]),
        "artifact_dir": Path(str(listing["artifact_dir"])),
        "card": get_doc_card(db_path, doc_id),
        "quality": get_parse_quality(db_path, doc_id),
        "nodes": nodes,
        "references": get_artifact(db_path, doc_id, "references.json")["content"],
        "selected_nodes": _select_insight_nodes(nodes),
    }


def read_existing_insights(db_path: Path, doc_id: str) -> Optional[Dict[str, Any]]:
    try:
        innovation = get_artifact(db_path, doc_id, "innovation.json")["content"]
        citation_map = get_artifact(db_path, doc_id, "citation_map.json")["content"]
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if (
        isinstance(innovation, dict)
        and isinstance(citation_map, dict)
        and innovation.get("schema") == "innovation.v1"
        and citation_map.get("schema") == "citation_map.v1"
        and innovation.get("status") in {"extracted", "partial"}
    ):
        return {"innovation": innovation, "citation_map": citation_map}
    return None


def _select_insight_nodes(nodes: Any) -> List[Dict[str, Any]]:
    if not isinstance(nodes, list):
        return []
    scored = []
    for node in nodes:
        if not isinstance(node, dict) or not node.get("text"):
            continue
        kind = str(node.get("kind") or "")
        if kind == "reference":
            continue
        text = _node_text(node)
        score = _insight_score(kind, text)
        if score <= 0:
            continue
        scored.append((score, int(node.get("order_index") or 0), _evidence_from_node(node)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: List[Dict[str, Any]] = []
    seen = set()
    for _, _, evidence in scored:
        node_id = evidence["node_id"]
        if node_id in seen:
            continue
        selected.append(evidence)
        seen.add(node_id)
        if len(selected) >= 18:
            break
    return selected


def _insight_score(kind: str, text: str) -> int:
    score = 0
    if kind == "abstract":
        score += 5
    for token in INSIGHT_NODE_TOKENS:
        if token in text:
            score += 3
    if "提出" in text or "设计" in text or "构建" in text:
        score += 2
    if len(text) < 40:
        score -= 2
    if re.search(r"(\.{4,}|…{2,})", text):
        score -= 5
    return score


def extract_innovation_with_llm(
    card: Dict[str, Any],
    quality: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
    *,
    json_generator: Optional[JsonGenerator] = None,
    stage: str = "legacy",
) -> Dict[str, object]:
    system_prompt = (
        "你是一个严谨的论文结构化抽取助手。只能基于给定节点抽取信息，"
        "不要编造证据。必须返回 JSON object，不要返回 Markdown。"
    )
    user_prompt = "\n".join(
        [
            "请抽取论文创新点、方法贡献、局限和开放问题。",
            "返回格式：",
            '{"items":[{"title":"","type":"","claim":"","problem":"","approach":"","evidence":[],"confidence":0.0}],'
            '"limitations":[],"open_questions":[],"warnings":[]}',
            "",
            f"title: {card.get('title')}",
            f"abstract: {card.get('abstract') or card.get('description')}",
            f"parse_quality: {quality}",
            "",
            "候选证据节点：",
            *_format_evidence_for_prompt(selected_nodes),
        ]
    )
    generator = json_generator or structured_json_generator(
        "insight_extraction",
        stage,
        json_generator=generate_json_object,
    )
    return generator(system_prompt, user_prompt)


def _format_evidence_for_prompt(nodes: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for index, node in enumerate(nodes, start=1):
        lines.append(f"[N{index}] node_id: {node['node_id']}")
        lines.append(f"node_path: {node['node_path']}")
        lines.append(f"page_range: {node['page_range']}")
        lines.append(f"excerpt: {_excerpt(node['excerpt'], 360)}")
        lines.append("")
    return lines


def normalize_innovation_payload(
    payload: Dict[str, object],
    *,
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    quality: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
    status: str,
    warnings: List[str],
) -> Dict[str, Any]:
    raw_items = payload.get("items")
    items = []
    if isinstance(raw_items, list):
        for index, raw_item in enumerate(raw_items[:8]):
            if not isinstance(raw_item, dict):
                continue
            evidence = _normalize_item_evidence(raw_item.get("evidence"), selected_nodes, index)
            items.append(
                {
                    "title": _excerpt(_string_value(raw_item.get("title")), 80) or f"创新点 {index + 1}",
                    "type": _string_value(raw_item.get("type")) or "contribution",
                    "claim": _excerpt(_string_value(raw_item.get("claim")), 220),
                    "problem": _excerpt(_string_value(raw_item.get("problem")), 180),
                    "approach": _excerpt(_string_value(raw_item.get("approach")), 180),
                    "evidence": evidence,
                    "confidence": _confidence(raw_item.get("confidence"), default=0.75),
                }
            )
    if not items:
        status = "partial"
        warnings = [*warnings, "empty_llm_items"]
    return {
        "schema": "innovation.v1",
        "status": status,
        "doc_id": doc_id,
        "version_id": version_id,
        "title": card.get("title") or "",
        "source": "llm" if status == "extracted" else "rule",
        "items": items,
        "limitations": _string_list(payload.get("limitations")),
        "open_questions": _string_list(payload.get("open_questions")),
        "warnings": [*warnings, *_string_list(payload.get("warnings")), *_quality_warnings(quality)],
        "created_at": time.time(),
    }


def _rule_based_innovation(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    quality: Dict[str, Any],
    selected_nodes: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    items = []
    for index, node in enumerate(selected_nodes[:6]):
        excerpt = node["excerpt"]
        items.append(
            {
                "title": _title_from_node(node, index),
                "type": _innovation_type(node),
                "claim": _excerpt(excerpt, 420),
                "problem": _sentence_with_tokens(excerpt, ("问题", "挑战", "不足", "局限")),
                "approach": _sentence_with_tokens(excerpt, ("提出", "设计", "构建", "方法", "算法", "模型")),
                "evidence": [node],
                "confidence": 0.55,
            }
        )
    limitations = [
        _excerpt(node["excerpt"], 260)
        for node in selected_nodes
        if any(token in _node_search_text(node) for token in LIMITATION_TOKENS)
    ][:5]
    return {
        "schema": "innovation.v1",
        "status": "partial",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": card.get("title") or "",
        "source": "rule",
        "items": items,
        "limitations": limitations,
        "open_questions": [],
        "warnings": [*warnings, "rule_based_extraction", *_quality_warnings(quality)],
        "created_at": time.time(),
    }


def build_citation_map(
    doc_id: str,
    version_id: str,
    card: Dict[str, Any],
    references_artifact: Any,
    nodes: Any,
) -> Dict[str, Any]:
    references = _parse_reference_nodes(nodes)
    if not references:
        references = _parse_references(references_artifact)
    ref_ids = {item["ref_id"] for item in references}
    in_text_citations = []
    relations = []
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict) or not node.get("text") or node.get("kind") == "reference":
                continue
            for ref_id, context in _citations_in_text(str(node.get("text") or "")):
                citation = {
                    "ref_id": ref_id,
                    "node_id": node.get("node_id") or "",
                    "node_path": node.get("node_path") or "",
                    "page_range": [node.get("page_start"), node.get("page_end")],
                    "context": context,
                    "matched_reference": ref_id in ref_ids,
                }
                in_text_citations.append(citation)
                relations.append({"relation_type": "cites", **citation})
                if len(in_text_citations) >= 500:
                    break
            if len(in_text_citations) >= 500:
                break
    return {
        "schema": "citation_map.v1",
        "status": "extracted" if references or in_text_citations else "partial",
        "doc_id": doc_id,
        "version_id": version_id,
        "title": card.get("title") or "",
        "references": references,
        "in_text_citations": in_text_citations,
        "relations": relations,
        "warnings": [] if references else ["missing_reference_list"],
        "created_at": time.time(),
    }


def _parse_references(references_artifact: Any) -> List[Dict[str, Any]]:
    raw_refs = []
    if isinstance(references_artifact, dict):
        raw_refs = references_artifact.get("references") or []
    lines = [str(item.get("raw") if isinstance(item, dict) else item) for item in raw_refs[:600]]
    return _parse_combined_reference_lines(lines)


def _parse_reference_nodes(nodes: Any) -> List[Dict[str, Any]]:
    if not isinstance(nodes, list):
        return []
    lines = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("kind") != "reference":
            continue
        node_path = str(node.get("node_path") or "")
        raw = compact_whitespace(str(node.get("text") or ""))
        if "参考文献" not in node_path and "References" not in node_path:
            continue
        if raw:
            lines.append(raw)
        if len(lines) >= 600:
            break
    return _parse_combined_reference_lines(lines)


def _parse_combined_reference_lines(lines: List[str]) -> List[Dict[str, Any]]:
    references = []
    current: List[str] = []
    for raw in lines:
        for text in _split_embedded_reference_lines(raw):
            if not text or re.search(r"(\.{4,}|…{2,})", text):
                continue
            if _reference_number(text):
                if current:
                    references.append(_parse_reference_line(" ".join(current), len(references) + 1))
                current = [text]
                continue
            if current and _looks_like_reference_continuation(text):
                current.append(text)
                continue
            if not current and _looks_like_reference_line(text):
                current = [text]
    if current:
        references.append(_parse_reference_line(" ".join(current), len(references) + 1))
    return references[:300]


def _split_embedded_reference_lines(raw: str) -> List[str]:
    text = compact_whitespace(raw)
    if not text:
        return []
    parts = re.split(r"\s+(?=\[\d{1,3}\]\s*)", text)
    return [part.strip() for part in parts if part.strip()]


def _parse_reference_line(raw: str, index: int) -> Dict[str, Any]:
    text = compact_whitespace(raw)
    match = re.match(r"^\[?(\d+)\]?[.)、]?\s*(.+)$", text)
    number = int(match.group(1)) if match else index
    body = match.group(2) if match else text
    year_match = re.search(r"(19|20)\d{2}", body)
    parts = [part.strip() for part in re.split(r"[.。]", body, maxsplit=2) if part.strip()]
    authors = parts[0] if parts else ""
    title = parts[1] if len(parts) > 1 else ""
    return {
        "ref_id": f"ref_{number}",
        "raw": text,
        "authors": authors,
        "title": title,
        "year": int(year_match.group(0)) if year_match else None,
    }


def _looks_like_reference_line(raw: str) -> bool:
    text = compact_whitespace(raw)
    if len(text) < 8:
        return False
    if re.search(r"(\.{4,}|…{2,})", text):
        return False
    if _reference_number(text):
        return True
    return bool(re.search(r"(19|20)\d{2}", text))


def _looks_like_reference_continuation(raw: str) -> bool:
    text = compact_whitespace(raw)
    if len(text) < 8 or text in {"摘要", "关键词", "Abstract", "Keywords"}:
        return False
    if re.search(r"(\.{4,}|…{2,})", text):
        return False
    return True


def _reference_number(raw: str) -> Optional[int]:
    match = re.match(r"^(?:\[(\d{1,3})\]|(\d{1,3})[.)、])\s*.+", compact_whitespace(raw))
    if not match:
        return None
    value = int(match.group(1) or match.group(2))
    return value if value > 0 else None


def _citations_in_text(text: str) -> Iterable[tuple[str, str]]:
    for match in REFERENCE_MARK_RE.finditer(text):
        raw = match.group(1).replace(" ", "")
        if raw in {"0,1", "0，1"}:
            continue
        context = _citation_context(text, match.start(), match.end())
        for number in _expand_citation_numbers(match.group(1)):
            yield f"ref_{number}", context


def _expand_citation_numbers(raw: str) -> List[int]:
    values: List[int] = []
    for part in re.split(r"[,，、]", raw):
        if "-" in part:
            left, right = [piece.strip() for piece in part.split("-", 1)]
            if left.isdigit() and right.isdigit():
                start, end = int(left), int(right)
                if 0 < start <= end <= start + 50:
                    values.extend(range(start, end + 1))
            continue
        stripped = part.strip()
        if stripped.isdigit() and int(stripped) > 0:
            values.append(int(stripped))
    return values


def _citation_context(text: str, start: int, end: int) -> str:
    return compact_whitespace(text[max(0, start - 80) : min(len(text), end + 80)])


def _normalize_item_evidence(value: object, selected_nodes: List[Dict[str, Any]], index: int) -> List[Dict[str, Any]]:
    by_id = {node["node_id"]: node for node in selected_nodes}
    result: List[Dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            node_id = ""
            if isinstance(item, str):
                node_id = item
            elif isinstance(item, dict):
                node_id = str(item.get("node_id") or item.get("id") or "")
            if node_id in by_id:
                result.append(by_id[node_id])
    if not result and selected_nodes:
        result.append(selected_nodes[min(index, len(selected_nodes) - 1)])
    return result[:3]


def _evidence_from_node(node: Dict[str, Any]) -> Dict[str, Any]:
    text = compact_whitespace(str(node.get("text") or node.get("summary") or ""))
    return {
        "node_id": str(node.get("node_id") or ""),
        "node_path": str(node.get("node_path") or ""),
        "page_range": [node.get("page_start"), node.get("page_end")],
        "evidence_type": str(node.get("kind") or "paragraph"),
        "excerpt": _excerpt(text, 900),
    }


def _node_text(node: Dict[str, Any]) -> str:
    return compact_whitespace(
        " ".join(
            str(node.get(key) or "")
            for key in ("kind", "heading", "summary", "text", "node_path")
        )
    )


def _node_search_text(node: Dict[str, Any]) -> str:
    return compact_whitespace(f"{node.get('node_path', '')} {node.get('excerpt', '')}")


def _title_from_node(node: Dict[str, Any], index: int) -> str:
    if node.get("evidence_type") == "abstract":
        return "摘要中的研究内容"
    path = str(node.get("node_path") or "")
    parts = [part.strip() for part in path.split(">") if part.strip()]
    for part in reversed(parts):
        if len(part) > 120:
            continue
        if any(token in part for token in ("研究内容", "贡献", "创新", "方法", "算法", "模型", "框架")):
            return part
    return f"候选创新点 {index + 1}"


def _innovation_type(node: Dict[str, Any]) -> str:
    text = _node_search_text(node)
    if any(token in text for token in ("实验", "结果", "评估")):
        return "result"
    if any(token in text for token in ("局限", "不足", "展望")):
        return "limitation"
    if any(token in text for token in ("方法", "算法", "模型", "框架")):
        return "method"
    return "contribution"


def _sentence_with_tokens(text: str, tokens: tuple[str, ...]) -> str:
    sentences = re.split(r"(?<=[。！？!?])", text)
    for sentence in sentences:
        if any(token in sentence for token in tokens):
            return _excerpt(sentence, 260)
    return ""


def _string_value(value: object) -> str:
    return compact_whitespace(str(value)) if value is not None else ""


def _string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_string_value(item) for item in value if _string_value(item)]


def _quality_warnings(quality: Dict[str, Any]) -> List[str]:
    raw = quality.get("quality_warnings")
    return _string_list(raw)


def _confidence(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _excerpt(text: str, max_chars: int) -> str:
    clean = compact_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + " ..."
