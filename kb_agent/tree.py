from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .models import NodeRecord, ParsedBlock, ParsedDocument
from .utils import chunk_text, first_words, stable_id


def build_document_tree(doc_id: str, parsed: ParsedDocument) -> List[NodeRecord]:
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
                    kind="section",
                    heading=block.heading,
                    summary=block.heading,
                    text="",
                    level=parent.level + 1,
                    node_path=node_path,
                    page_start=block.page,
                    page_end=block.page,
                    order_index=order,
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
                leaf = _leaf_node(doc_id, current_parent, chunk, order, block)
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
            )
            nodes.append(parent)
            order += 1
        for block in blocks:
            for chunk in chunk_text(block.text):
                leaf = _leaf_node(doc_id, parent, chunk, order, block)
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
) -> NodeRecord:
    heading = first_words(text, 10)
    return NodeRecord(
        node_id=stable_id("node", doc_id, order, heading, block.page or ""),
        doc_id=doc_id,
        parent_id=parent.node_id,
        kind="paragraph",
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
    )


def _group_blocks_by_page(blocks: List[ParsedBlock]) -> Dict[Optional[int], List[ParsedBlock]]:
    groups: Dict[Optional[int], List[ParsedBlock]] = {}
    for block in blocks:
        groups.setdefault(block.page, []).append(block)
    return groups


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
