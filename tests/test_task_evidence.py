from __future__ import annotations

import unittest

from kb_agent.task_evidence import (
    compact_section_evidence,
    dedupe_evidence,
    evidence_confidence,
    evidence_duplicate_summary,
    flatten_dimension_evidence,
    flatten_dimension_evidence_raw,
    normalize_evidence_refs,
    section_evidence_quality,
)


class TaskEvidenceTest(unittest.TestCase):
    def test_dedupe_keeps_higher_priority_duplicate_and_reports_schema(self) -> None:
        evidence = [
            _evidence("doc-1", "n1", "low", tree_score=0.1, excerpt="short"),
            _evidence("doc-1", "n1", "high", tree_score=0.9, excerpt="longer evidence excerpt"),
            _evidence("doc-1", "n2", "other", tree_score=0.2),
        ]

        deduped = dedupe_evidence(evidence)
        summary = evidence_duplicate_summary(evidence)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["summary"], "high")
        self.assertEqual(deduped[0]["dedupe_reason"], "kept:higher_score")
        self.assertEqual(
            summary,
            {
                "schema": "evidence_dedupe.v1",
                "raw_evidence_count": 3,
                "unique_evidence_count": 2,
                "duplicate_evidence_removed": 1,
            },
        )

    def test_compact_section_evidence_reports_truncation_and_source_docs(self) -> None:
        evidence = [
            _evidence("doc-1", "n1", "doc one low", tree_score=0.1),
            _evidence("doc-1", "n1", "doc one high", tree_score=0.8),
            _evidence("doc-1", "n2", "doc one extra", tree_score=0.7),
            _evidence("doc-2", "n3", "doc two", tree_score=0.6),
        ]

        compacted, report = compact_section_evidence(evidence, max_items=2)

        self.assertEqual(len(compacted), 2)
        self.assertEqual(report["schema"], "section_evidence_compaction.v1")
        self.assertEqual(report["raw_evidence_count"], 4)
        self.assertEqual(report["unique_evidence_count"], 3)
        self.assertEqual(report["duplicate_evidence_removed"], 1)
        self.assertEqual(report["kept_evidence_count"], 2)
        self.assertEqual(report["max_evidence_count"], 2)
        self.assertEqual(report["source_doc_count"], 2)
        self.assertEqual(report["source_doc_ids"], ["doc-1", "doc-2"])
        self.assertEqual(report["warnings"], ["duplicate_evidence_compacted", "section_evidence_truncated"])
        self.assertTrue(all("evidence_summary" in item for item in compacted))

    def test_section_evidence_quality_summarizes_dedupe_before_and_after_compaction(self) -> None:
        section_evidence = {
            "background": [_evidence("doc-1", "n1", "kept")],
            "methods": [_evidence("doc-2", "n2", "first"), _evidence("doc-2", "n2", "duplicate")],
        }
        compaction = {
            "background": {"raw_evidence_count": 2, "duplicate_evidence_removed": 1},
            "methods": {"raw_evidence_count": 3, "duplicate_evidence_removed": 0},
        }

        quality = section_evidence_quality(compaction, section_evidence)

        self.assertEqual(quality["schema"], "section_evidence_quality.v1")
        self.assertEqual(quality["pre_dedupe_count"], 5)
        self.assertEqual(quality["post_dedupe_count"], 3)
        self.assertEqual(quality["duplicate_evidence_removed"], 1)
        self.assertEqual(quality["duplicate_evidence_removed_by_section"], {"background": 1, "methods": 0})
        self.assertEqual(quality["post_dedupe_duplicate_count"], 1)
        self.assertEqual(quality["warnings"], ["post_dedupe_duplicate_evidence"])

    def test_dimension_flatten_keeps_raw_or_deduped_shape(self) -> None:
        low = _evidence("doc-1", "n1", "low", tree_score=0.1)
        high = _evidence("doc-1", "n1", "high", tree_score=0.9)
        unique = _evidence("doc-2", "n2", "unique", tree_score=0.5)
        by_dimension = {
            "problem": {"doc-1": [low], "doc-2": [unique]},
            "method": {"doc-1": [high]},
        }

        raw = flatten_dimension_evidence_raw(by_dimension)
        deduped = flatten_dimension_evidence(by_dimension)

        self.assertEqual([item["summary"] for item in raw], ["low", "unique", "high"])
        self.assertEqual(len(deduped), 2)
        self.assertEqual([item["summary"] for item in deduped], ["high", "unique"])

    def test_normalize_evidence_refs_uses_node_ids_and_fallback(self) -> None:
        fallback = [
            _evidence("doc-1", "n1", "first"),
            _evidence("doc-1", "n2", "second"),
            _evidence("doc-1", "n3", "third"),
        ]

        matched = normalize_evidence_refs([{"id": "n2"}, "missing"], fallback)
        fallback_slice = normalize_evidence_refs(None, fallback)

        self.assertEqual([item["node_id"] for item in matched], ["n2"])
        self.assertEqual([item["node_id"] for item in fallback_slice], ["n1", "n2", "n3"])

    def test_evidence_confidence_keeps_existing_thresholds(self) -> None:
        self.assertEqual(evidence_confidence([]), 0.25)
        self.assertEqual(evidence_confidence([_evidence("doc-1", "n1", "one")]), 0.6)
        self.assertEqual(
            evidence_confidence(
                [
                    _evidence("doc-1", "n1", "one"),
                    _evidence("doc-1", "n2", "two"),
                    _evidence("doc-1", "n3", "three"),
                ]
            ),
            0.75,
        )


def _evidence(
    doc_id: str,
    node_id: str,
    summary: str,
    *,
    tree_score: float | None = None,
    excerpt: str = "evidence excerpt",
) -> dict:
    item = {
        "doc_id": doc_id,
        "node_id": node_id,
        "node_path": "1",
        "summary": summary,
        "excerpt": excerpt,
    }
    if tree_score is not None:
        item["tree_score"] = tree_score
    return item


if __name__ == "__main__":
    unittest.main()
