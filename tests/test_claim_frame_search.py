from __future__ import annotations

import unittest

from kb_agent.claim_frame_search import claim_frame_search_item, query_terms, rank_claim_frame_items


class ClaimFrameSearchTest(unittest.TestCase):
    def test_supported_frame_search_item_preserves_semantic_and_citation_fields(self) -> None:
        terms = query_terms("动态任务规划方法")
        item = claim_frame_search_item(
            _frame(
                "supported",
                support_status="structurally_supported",
                semantic_support_status="semantically_supported",
                citation_risk="safe",
                quality_score=0.9,
            ),
            {
                "frame_id": "supported",
                "trace_status": "verified",
                "support_status": "structurally_supported",
                "support_reason": "evidence_units_verified",
                "semantic_support_status": "semantically_supported",
                "semantic_support_score": 0.92,
                "semantic_support_reason": "primary_evidence_overlap",
                "primary_evidence_unit_ids": ["eu-1"],
                "weak_evidence_unit_ids": [],
                "contradictory_evidence_unit_ids": [],
                "citation_risk": "safe",
                "warnings": [],
            },
            "动态任务规划方法",
            terms,
        )

        assert item is not None
        self.assertEqual(item["semantic_support_status"], "semantically_supported")
        self.assertEqual(item["citation_risk"], "safe")
        self.assertEqual(item["primary_evidence_unit_ids"], ["eu-1"])
        self.assertIn("semantic:semantically_supported", item["selection_reasons"])
        self.assertIn("citation_risk:safe", item["selection_reasons"])
        self.assertEqual(item["fallback_reason"], "")
        self.assertIn("short_claim", item["matched_fields"])

    def test_unsupported_frame_is_filtered_for_regular_query(self) -> None:
        item = claim_frame_search_item(
            _frame(
                "unsupported",
                support_status="unsupported",
                semantic_support_status="insufficient_evidence",
                citation_risk="needs_more_evidence",
                quality_score=0.8,
            ),
            {},
            "动态任务规划方法",
            query_terms("动态任务规划方法"),
        )

        self.assertIsNone(item)

    def test_weak_query_allows_unsupported_evidence_frames(self) -> None:
        item = claim_frame_search_item(
            _frame(
                "unsupported",
                support_status="unsupported",
                semantic_support_status="insufficient_evidence",
                citation_risk="needs_more_evidence",
                quality_score=0.8,
            ),
            {},
            "unsupported 证据 动态任务规划",
            query_terms("unsupported 证据 动态任务规划"),
        )

        assert item is not None
        self.assertEqual(item["support_status"], "unsupported")
        self.assertEqual(item["citation_risk"], "needs_more_evidence")
        self.assertIn(item["fallback_reason"], {"weak_frame_allowed_by_query", "partial_evidence", "missing_evidence_unit"})

    def test_rank_claim_frame_items_orders_by_score_then_frame_id(self) -> None:
        ranked = rank_claim_frame_items(
            [
                {"frame_id": "b", "score": 1.0},
                {"frame_id": "a", "score": 1.0},
                {"frame_id": "c", "score": 2.0},
            ],
            top_k=2,
        )

        self.assertEqual([item["frame_id"] for item in ranked], ["c", "a"])


def _frame(
    frame_id: str,
    *,
    support_status: str,
    semantic_support_status: str,
    citation_risk: str,
    quality_score: float,
) -> dict:
    return {
        "frame_id": frame_id,
        "doc_id": "doc-1",
        "claim_type": "method",
        "short_claim": "本文提出动态任务规划方法。",
        "method": "动态任务规划方法",
        "dataset_or_setting": "",
        "metric_or_signal": "",
        "result_or_gain": "",
        "limitation": "",
        "trace_status": "verified" if support_status == "structurally_supported" else "missing",
        "support_status": support_status,
        "support_reason": "evidence_units_verified" if support_status == "structurally_supported" else "no_evidence_unit_found",
        "semantic_support_status": semantic_support_status,
        "semantic_support_score": 0.8,
        "semantic_support_reason": "test",
        "citation_risk": citation_risk,
        "primary_evidence_unit_ids": ["eu-1"] if support_status == "structurally_supported" else [],
        "weak_evidence_unit_ids": [],
        "contradictory_evidence_unit_ids": [],
        "evidence_unit_ids": ["eu-1"] if support_status == "structurally_supported" else [],
        "source_claim_ids": ["claim-1"],
        "confidence": 0.8,
        "quality_score": quality_score,
        "frame_quality": "high",
        "noise_reasons": [],
        "warnings": [],
    }


if __name__ == "__main__":
    unittest.main()
