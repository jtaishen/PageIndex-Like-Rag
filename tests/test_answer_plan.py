from __future__ import annotations

import unittest

from kb_agent.answer_plan import build_answer_plan


class AnswerPlanTest(unittest.TestCase):
    def test_buckets_claims_by_semantic_support_and_citation_risk(self) -> None:
        plan = build_answer_plan(
            "任务规划方法",
            {
                "items": [
                    _claim("strong", "semantically_supported", "safe"),
                    _claim("partial", "partially_supported", "needs_qualification"),
                    _claim("related", "related_only", "needs_more_evidence"),
                    _claim("conflict", "contradicted", "conflicting_evidence"),
                    _claim("insufficient", "insufficient_evidence", "needs_more_evidence"),
                    _claim("unchecked", "not_checked", "not_checked"),
                ]
            },
        )

        self.assertEqual(plan["answerability"], "conflicting")
        self.assertEqual(plan["strong_claims"][0]["frame_id"], "strong")
        self.assertEqual(plan["qualified_claims"][0]["frame_id"], "partial")
        self.assertEqual(plan["related_claims"][0]["frame_id"], "related")
        self.assertEqual(plan["conflicting_claims"][0]["frame_id"], "conflict")
        self.assertEqual(plan["insufficient_claims"][0]["frame_id"], "insufficient")
        self.assertEqual(plan["unchecked_claims"][0]["frame_id"], "unchecked")
        self.assertIn("answer_plan_conflicting_claims", plan["warnings"])

    def test_insufficient_when_no_strong_or_qualified_claims(self) -> None:
        plan = build_answer_plan("任务规划方法", {"items": [_claim("insufficient", "insufficient_evidence", "needs_more_evidence")]})

        self.assertEqual(plan["answerability"], "insufficient_evidence")
        self.assertEqual(plan["insufficient_claim_count"], 1)
        self.assertIn("answer_plan_insufficient_evidence", plan["warnings"])


def _claim(frame_id: str, status: str, risk: str) -> dict:
    return {
        "doc_id": "doc-1",
        "title": "Paper",
        "frame_id": frame_id,
        "short_claim": f"{frame_id} claim",
        "claim_type": "method",
        "semantic_support_status": status,
        "citation_risk": risk,
        "evidence_unit_ids": [f"eu-{frame_id}"],
        "primary_evidence_unit_ids": [f"eu-{frame_id}"] if status == "semantically_supported" else [],
        "score": 1.0,
    }
