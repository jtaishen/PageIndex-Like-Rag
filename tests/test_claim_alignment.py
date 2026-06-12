from __future__ import annotations

import unittest

from kb_agent.claim_alignment import (
    build_claim_alignment,
    build_claim_relations,
    claim_alignment_rollup,
    claim_alignment_summary,
    review_alignment_sections,
)


class ClaimAlignmentTest(unittest.TestCase):
    def test_alignment_builds_typed_relations_and_review_sections(self) -> None:
        alignment = build_claim_alignment("任务规划方法", _contexts())
        relations = build_claim_relations(alignment)
        summary = claim_alignment_summary(alignment, relations)

        self.assertEqual(alignment["schema"], "claim_alignment.v1")
        self.assertEqual(relations["schema"], "claim_relations.v1")
        self.assertGreaterEqual(summary["group_count"], 2)
        self.assertGreaterEqual(summary["method_family_group_count"], 1)

        relation_types = {item["relation_type"] for item in relations["relations"]}
        self.assertIn("same_method_family", relation_types)
        self.assertIn("contradicts", relation_types)
        self.assertIn("incomparable", relation_types)

        review = review_alignment_sections(alignment, relations)
        self.assertEqual(review["method_lineage"]["schema"], "method_lineage.v1")
        self.assertGreaterEqual(review["method_lineage"]["relation_count"], 1)
        self.assertIn("research_gap_candidates", review)

    def test_alignment_rollup_sums_public_counts(self) -> None:
        rollup = claim_alignment_rollup(
            [
                {
                    "available": True,
                    "group_count": 2,
                    "method_family_group_count": 1,
                    "conflicting_group_count": 1,
                    "research_gap_count": 0,
                    "relation_count": 3,
                    "relation_type_counts": {"contradicts": 1},
                    "warnings": ["claim_alignment_conflicts"],
                },
                {
                    "available": True,
                    "group_count": 1,
                    "method_family_group_count": 0,
                    "conflicting_group_count": 0,
                    "research_gap_count": 1,
                    "relation_count": 1,
                    "relation_type_counts": {"incomparable": 1},
                    "warnings": ["claim_alignment_conflicts"],
                },
            ]
        )

        self.assertTrue(rollup["available"])
        self.assertEqual(rollup["group_count"], 3)
        self.assertEqual(rollup["relation_count"], 4)
        self.assertEqual(rollup["relation_type_counts"], {"contradicts": 1, "incomparable": 1})
        self.assertEqual(rollup["warnings"], ["claim_alignment_conflicts"])


def _contexts() -> list[dict]:
    return [
        {
            "doc_id": "doc-a",
            "title": "服务机器人任务规划",
            "claim_frames": {
                "frames": [
                    _frame(
                        "a-method",
                        "method",
                        "提出任务规划协同框架",
                        method="任务规划协同框架",
                        status="semantically_supported",
                        risk="safe",
                    ),
                    _frame(
                        "a-result",
                        "result",
                        "任务规划性能提升任务完成率",
                        metric="任务完成率",
                        result="提升任务完成率",
                        status="semantically_supported",
                        risk="safe",
                    ),
                    _frame(
                        "a-limitation",
                        "limitation",
                        "真实环境验证不足",
                        limitation="真实环境验证不足",
                        status="insufficient_evidence",
                        risk="needs_more_evidence",
                    ),
                ]
            },
        },
        {
            "doc_id": "doc-b",
            "title": "多智能体任务规划",
            "claim_frames": {
                "frames": [
                    _frame(
                        "b-method",
                        "method",
                        "提出动态任务规划协同框架",
                        method="动态任务规划协同框架",
                        status="semantically_supported",
                        risk="safe",
                    ),
                    _frame(
                        "b-result",
                        "result",
                        "任务规划性能在任务完成率上失败",
                        metric="任务完成率",
                        result="任务完成率下降且未验证",
                        status="contradicted",
                        risk="conflicting_evidence",
                    ),
                    _frame(
                        "b-result-time",
                        "result",
                        "任务规划性能改善响应时间",
                        metric="响应时间",
                        result="响应时间降低",
                        status="semantically_supported",
                        risk="safe",
                    ),
                    _frame(
                        "b-limitation",
                        "limitation",
                        "复杂通信约束验证不足",
                        limitation="复杂通信约束验证不足",
                        status="insufficient_evidence",
                        risk="needs_more_evidence",
                    ),
                ]
            },
        },
    ]


def _frame(
    frame_id: str,
    claim_type: str,
    short_claim: str,
    *,
    method: str = "",
    metric: str = "",
    result: str = "",
    limitation: str = "",
    status: str,
    risk: str,
) -> dict:
    return {
        "frame_id": frame_id,
        "claim_type": claim_type,
        "short_claim": short_claim,
        "method": method,
        "metric_or_signal": metric,
        "result_or_gain": result,
        "limitation": limitation,
        "semantic_support_status": status,
        "citation_risk": risk,
        "evidence_unit_ids": [f"eu-{frame_id}"],
        "primary_evidence_unit_ids": [f"eu-{frame_id}"],
    }


if __name__ == "__main__":
    unittest.main()
