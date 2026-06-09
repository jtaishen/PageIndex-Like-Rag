from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from .models import NodeRecord, ParsedBlock, ParsedDocument
from .utils import chunk_text, first_words, stable_id


SEMANTIC_LEAF_KINDS = {"abstract", "keywords", "figure", "table", "reference"}


def build_document_tree(doc_id: str, parsed: ParsedDocument, doc_hash: str = "") -> List[NodeRecord]:
    root = NodeRecord(
        node_id=stable_id("node", doc_id, "root"),
        doc_id=doc_id,
        parent_id=None,
        kind="document",
        heading=parsed.title,
        summary=first_words(parsed.raw_text, 60),
        text="",
        level=0,
        node_path=parsed.title,
        page_start=None,
        page_end=None,
        order_index=0,
        keywords=_keywords(parsed.title, parsed.raw_text),
        source_offsets={},
        doc_hash=doc_hash,
    )
    nodes: List[NodeRecord] = [root]
    heading_stack: Dict[int, NodeRecord] = {0: root}
    current_parent = root
    order = 1

    has_headings = any(block.kind == "heading" for block in parsed.blocks)
    if has_headings:
        first_heading = True
        for block in parsed.blocks:
            if block.kind == "heading":
                if first_heading and block.level == 1 and block.heading == parsed.title:
                    first_heading = False
                    current_parent = root
                    continue
                first_heading = False
                parent = _nearest_parent(heading_stack, block.level)
                node_path = f"{parent.node_path} > {block.heading}"
                heading_node = NodeRecord(
                    node_id=stable_id("node", doc_id, order, block.heading),
                    doc_id=doc_id,
                    parent_id=parent.node_id,
                    kind=_heading_kind(block.heading),
                    heading=block.heading,
                    summary=block.heading,
                    text="",
                    level=parent.level + 1,
                    node_path=node_path,
                    page_start=block.page,
                    page_end=block.page,
                    order_index=order,
                    keywords=_keywords(block.heading),
                    source_offsets=_source_offsets(block),
                    doc_hash=doc_hash,
                )
                nodes.append(heading_node)
                heading_stack[block.level] = heading_node
                for stale_level in list(heading_stack.keys()):
                    if stale_level > block.level:
                        del heading_stack[stale_level]
                current_parent = heading_node
                order += 1
                continue
            first_heading = False

            for chunk in chunk_text(block.text):
                leaf = _leaf_node(doc_id, current_parent, chunk, order, block, doc_hash)
                nodes.append(leaf)
                order += 1
        _fill_section_summaries(nodes)
        return nodes

    page_groups = _group_blocks_by_page(parsed.blocks)
    for group_key, blocks in page_groups.items():
        if group_key is None:
            parent = root
        else:
            parent = NodeRecord(
                node_id=stable_id("node", doc_id, "page", group_key),
                doc_id=doc_id,
                parent_id=root.node_id,
                kind="page",
                heading=f"Page {group_key}",
                summary=f"Page {group_key}",
                text="",
                level=1,
                node_path=f"{root.node_path} > Page {group_key}",
                page_start=group_key,
                page_end=group_key,
                order_index=order,
                keywords=_keywords(f"Page {group_key}"),
                source_offsets={},
                doc_hash=doc_hash,
            )
            nodes.append(parent)
            order += 1
        for block in blocks:
            for chunk in chunk_text(block.text):
                leaf = _leaf_node(doc_id, parent, chunk, order, block, doc_hash)
                nodes.append(leaf)
                order += 1
    _fill_section_summaries(nodes)
    return nodes


def tree_to_dict(nodes: List[NodeRecord]) -> Dict[str, object]:
    by_id = {
        node.node_id: {
            "node_id": node.node_id,
            "type": node.kind,
            "heading": node.heading,
            "summary": node.summary,
            "node_path": node.node_path,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "page_range": [node.page_start, node.page_end],
            "keywords": node.keywords,
            "source_offsets": node.source_offsets,
            "doc_hash": node.doc_hash,
            "children": [],
        }
        for node in nodes
    }
    root: Optional[Dict[str, object]] = None
    for node in nodes:
        item = by_id[node.node_id]
        if not node.parent_id:
            root = item
            continue
        parent = by_id.get(node.parent_id)
        if parent is not None:
            parent["children"].append(item)  # type: ignore[index]
    return root or {}


def _nearest_parent(stack: Dict[int, NodeRecord], level: int) -> NodeRecord:
    for candidate in range(level - 1, -1, -1):
        if candidate in stack:
            return stack[candidate]
    return stack[0]


def _leaf_node(
    doc_id: str,
    parent: NodeRecord,
    text: str,
    order: int,
    block: ParsedBlock,
    doc_hash: str = "",
) -> NodeRecord:
    heading = first_words(text, 10)
    kind = block.kind if block.kind in SEMANTIC_LEAF_KINDS else "paragraph"
    return NodeRecord(
        node_id=stable_id("node", doc_id, order, heading, block.page or ""),
        doc_id=doc_id,
        parent_id=parent.node_id,
        kind=kind,
        heading=heading,
        summary=first_words(text, 32),
        text=text,
        level=parent.level + 1,
        node_path=f"{parent.node_path} > {heading}",
        page_start=block.page,
        page_end=block.page,
        order_index=order,
        char_start=block.char_start,
        char_end=block.char_end,
        keywords=_keywords(parent.heading, text),
        source_offsets=_source_offsets(block),
        doc_hash=doc_hash,
    )


def _group_blocks_by_page(blocks: List[ParsedBlock]) -> Dict[Optional[int], List[ParsedBlock]]:
    groups: Dict[Optional[int], List[ParsedBlock]] = {}
    for block in blocks:
        groups.setdefault(block.page, []).append(block)
    return groups


def _heading_kind(heading: str) -> str:
    normalized = re.sub(r"\s+", "", heading).lower()
    if normalized in {"摘要", "abstract"}:
        return "abstract"
    if normalized in {"关键词", "关键字", "keywords", "keyword"}:
        return "keywords"
    if normalized in {"参考文献", "references", "bibliography"}:
        return "reference"
    return "section"


def _fill_section_summaries(nodes: List[NodeRecord]) -> None:
    children: Dict[str, List[NodeRecord]] = {}
    for node in nodes:
        if node.parent_id:
            children.setdefault(node.parent_id, []).append(node)
    for node in nodes:
        if node.text:
            continue
        child_text = " ".join(child.summary or child.text for child in children.get(node.node_id, []))
        if child_text:
            node.summary = first_words(child_text, 50)
            if not node.keywords:
                node.keywords = _keywords(node.heading, child_text)


def _source_offsets(block: ParsedBlock) -> Dict[str, Any]:
    offsets: Dict[str, Any] = {
        "char_start": block.char_start,
        "char_end": block.char_end,
        "page_start": block.page,
        "page_end": block.page,
    }
    if block.layout_block_id:
        offsets["layout_block_id"] = block.layout_block_id
    if block.caption_id:
        offsets["caption_id"] = block.caption_id
    if block.bbox is not None:
        offsets["bbox"] = block.bbox
    if block.source_parser:
        offsets["source_parser"] = block.source_parser
    if block.confidence != 1.0:
        offsets["confidence"] = round(block.confidence, 3)
    return offsets


def _keywords(*values: str, limit: int = 10) -> List[str]:
    text = " ".join(value for value in values if value)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)
    seen = set()
    result = []
    stopwords = {
        "the", "and", "for", "with", "from", "this", "that", "are", "was",
        "into", "about", "研究", "系统", "方法", "问题", "分析",
    }
    for token in tokens:
        normalized = token.lower() if token.isascii() else token
        if normalized in stopwords or normalized in seen:
            continue
        seen.add(normalized)
        result.append(token)
        if len(result) >= limit:
            break
    return result
