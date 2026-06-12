from __future__ import annotations

import unittest

from kb_agent.fact_sources import (
    facts_from_tables,
    merge_citation_relations,
    node_map,
    rule_based_facts,
    select_fact_nodes,
)


class FactSourcesTest(unittest.TestCase):
    def test_select_fact_nodes_uses_insight_evidence_then_scored_nodes(self) -> None:
        nodes = [
            _node("node-a", "reference", "参考文献"),
            _node("node-b", "section", "提出动态角色发现机制。"),
            _node("node-c", "section", "实验结果表明任务完成率提升。"),
        ]
        innovation = {"items": [{"evidence": [{"node_id": "node-c"}]}]}

        selected = select_fact_nodes(nodes, innovation, {"in_text_citations": []})

        self.assertEqual(selected[0]["node_id"], "node-c")
        self.assertIn("node-b", {item["node_id"] for item in selected})
        self.assertNotIn("node-a", {item["node_id"] for item in selected})

    def test_rule_based_facts_returns_existing_schema_fields_and_warnings(self) -> None:
        nodes = [
            _node("node-1", "abstract", "提出动态角色发现机制。"),
            _node("node-2", "section", "该方法提升任务完成率，并降低响应时间。"),
        ]
        by_id = node_map(nodes)
        citation_map = {
            "relations": [{"ref_id": "R1", "node_id": "node-2"}],
            "in_text_citations": [{"ref_id": "R1", "node_id": "node-2"}],
            "references": [{"ref_id": "R1", "title": "Prior Work"}],
        }

        result = rule_based_facts(
            "doc-1",
            "v1",
            {"title": "Doc A", "keywords": ["动态角色"]},
            {"quality_level": "usable", "quality_warnings": []},
            {"items": [{"type": "contribution", "claim": "提出动态角色发现机制。", "evidence": [{"node_id": "node-1"}]}]},
            citation_map,
            nodes,
            by_id,
            [],
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["source"], "rule")
        self.assertIn("rule_based_fact_extraction", result["warnings"])
        self.assertTrue(result["claims"])
        self.assertTrue(result["entities"])
        self.assertIn("supports", {item["type"] for item in result["relations"]})
        self.assertIn("cites", {item["type"] for item in result["relations"]})

    def test_merge_citation_relations_preserves_existing_facts_and_adds_citation_warning(self) -> None:
        node = _node("node-1", "section", "引用 R1 的相关工作。")
        facts = {"claims": [], "entities": [], "relations": [], "warnings": ["existing"], "quality_stats": {}}

        result = merge_citation_relations(
            "doc-1",
            "v1",
            {"title": "Doc A"},
            facts,
            {"relations": [{"ref_id": "R1", "node_id": "node-1"}], "in_text_citations": []},
            {"node-1": node},
        )

        self.assertEqual(result["relations"][0]["type"], "cites")
        self.assertEqual(result["relations"][0]["source"], "citation_rule")
        self.assertIn("existing", result["warnings"])
        self.assertIn("citation_fact_relations_added", result["warnings"])

    def test_table_facts_include_table_backed_entities_relations_and_warnings(self) -> None:
        node = _node("node-table", "section", "表 1 展示实验结果。", source_offsets={"layout_block_id": "tbl-1"})
        table_content = {
            "table_content": [
                {
                    "table_id": "table_001",
                    "layout_block_id": "tbl-1",
                    "caption": "表 1 实验结果",
                    "headers": ["方法", "任务完成率", "数据集"],
                    "rows": [
                        {"cells": [{"text": "基线方法"}, {"text": "80%"}, {"text": "测试数据集"}]},
                        {"cells": [{"text": "本文方法"}, {"text": "92%"}, {"text": "测试数据集"}]},
                    ],
                    "source": "table_rule",
                    "confidence": 0.7,
                }
            ]
        }

        result = facts_from_tables("doc-1", "v1", table_content, {"node-table": node})

        self.assertTrue(result["claims"])
        self.assertIn("table_fact_extraction", result["warnings"])
        self.assertIn("metric", {item["type"] for item in result["entities"]})
        relation_types = {item["type"] for item in result["relations"]}
        self.assertIn("reports_metric", relation_types)
        self.assertIn("improves", relation_types)
        self.assertTrue(all(item["evidence"]["table_id"] == "table_001" for item in [*result["entities"], *result["relations"]]))


def _node(node_id: str, kind: str, text: str, *, source_offsets: dict | None = None) -> dict:
    return {
        "doc_id": "doc-1",
        "node_id": node_id,
        "kind": kind,
        "heading": "Method",
        "summary": "",
        "text": text,
        "node_path": "1 Method",
        "page_start": 1,
        "page_end": 1,
        "order_index": 1,
        "source_offsets": source_offsets or {},
    }


if __name__ == "__main__":
    unittest.main()
