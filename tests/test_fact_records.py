from __future__ import annotations

import unittest

from kb_agent.fact_records import (
    apply_fact_quality_filters,
    claim_record,
    dedupe_facts,
    entity_record,
    graph_edges,
    graph_nodes,
    relation_record,
)


class FactRecordsTest(unittest.TestCase):
    def test_record_builders_preserve_existing_field_shape(self) -> None:
        node = _node()

        claim = claim_record("doc-1", "v1", "method", "提出动态角色发现机制。", node, "rule", 0.7, 0)
        entity = entity_record("doc-1", "v1", "method", "动态角色发现机制", node, "rule", 0.72, aliases=["角色发现"])
        relation = relation_record("doc-1", "v1", "uses", "动态角色发现机制", "任务分解", node, "rule", 0.66)

        self.assertIsNotNone(claim)
        self.assertIsNotNone(entity)
        self.assertIsNotNone(relation)
        assert claim is not None and entity is not None and relation is not None
        self.assertEqual(claim["claim_type"], "method")
        self.assertEqual(claim["page_range"], [2, 3])
        self.assertEqual(claim["evidence"]["layout_block_id"], "block-1")
        self.assertEqual(entity["entity_type"], "method")
        self.assertEqual(entity["aliases"], ["角色发现"])
        self.assertEqual(relation["relation_type"], "uses")
        self.assertEqual(relation["subject_name"], "动态角色发现机制")

        nodes = graph_nodes([claim], [entity])
        edges = graph_edges([relation])
        self.assertEqual(nodes[0]["kind"], "claim")
        self.assertEqual(nodes[1]["kind"], "entity")
        self.assertEqual(edges[0]["source"], "动态角色发现机制")

    def test_dedupe_keeps_stronger_duplicate_fields_and_stats(self) -> None:
        node = _node()
        low = claim_record("doc-1", "v1", "method", "提出动态角色发现机制。", node, "rule", 0.5, 0)
        high = claim_record("doc-1", "v1", "method", "提出动态角色发现机制。", node, "llm", 0.9, 1)
        assert low is not None and high is not None

        result = dedupe_facts({"claims": [low, high], "entities": [], "relations": []})

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["confidence"], 0.9)
        self.assertEqual(result["claims"][0]["source"], "llm")
        self.assertEqual(result["dedupe_stats"]["schema"], "fact_dedupe.v1")
        self.assertEqual(result["dedupe_stats"]["dedupe_merged_count"], 1)
        self.assertEqual(result["dedupe_stats"]["post_dedupe_duplicate_count"], 0)

    def test_quality_filter_removes_noisy_entities(self) -> None:
        node = _node()
        good = entity_record("doc-1", "v1", "method", "动态角色发现机制", node, "llm", 0.8)
        noisy = entity_record("doc-1", "v1", "term", "No.", node, "llm", 0.4)
        assert good is not None and noisy is not None
        facts = {"entities": [good, noisy]}
        stats: dict[str, int] = {}

        apply_fact_quality_filters(facts, stats)

        self.assertEqual([item["name"] for item in facts["entities"]], ["动态角色发现机制"])
        self.assertEqual(stats["entity_noise_filtered_count"], 1)


def _node() -> dict:
    return {
        "doc_id": "doc-1",
        "node_id": "node-1",
        "node_path": "1 Introduction",
        "page_start": 2,
        "page_end": 3,
        "source_offsets": {"layout_block_id": "block-1", "caption_id": "cap-1"},
    }


if __name__ == "__main__":
    unittest.main()
