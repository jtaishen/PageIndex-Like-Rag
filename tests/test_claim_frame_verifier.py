from __future__ import annotations

import unittest

from kb_agent.claim_frame_verifier import sync_claim_frames_with_verifier, verifier_totals, verify_claim_frames_payload


class ClaimFrameVerifierTest(unittest.TestCase):
    def test_trace_support_semantic_statuses_and_citation_risk(self) -> None:
        claim_frames = {
            "schema": "claim_frames.v1",
            "version_id": "v1",
            "frames": [
                {
                    "frame_id": "verified",
                    "claim_type": "method",
                    "short_claim": "本文提出动态任务规划算法提升任务分配效率。",
                    "method": "动态任务规划算法",
                    "evidence_unit_ids": ["eu-supported"],
                    "confidence": 0.9,
                },
                {
                    "frame_id": "partial",
                    "claim_type": "method",
                    "short_claim": "方法依赖缺失节点。",
                    "evidence_unit_ids": ["eu-missing-node"],
                    "confidence": 0.8,
                },
                {
                    "frame_id": "missing",
                    "claim_type": "result",
                    "short_claim": "结果有引用但 unit 缺失。",
                    "evidence_unit_ids": ["eu-absent"],
                    "confidence": 0.8,
                },
                {
                    "frame_id": "unsupported",
                    "claim_type": "result",
                    "short_claim": "实验验证该方法提升任务成功率。",
                    "result_or_gain": "提升任务成功率",
                    "evidence_unit_ids": ["eu-contradicted"],
                    "confidence": 0.8,
                },
                {
                    "frame_id": "no-evidence",
                    "claim_type": "result",
                    "short_claim": "完全没有证据。",
                    "evidence_unit_ids": [],
                    "confidence": 0.8,
                },
            ],
        }
        evidence_units = {
            "schema": "evidence_units.v1",
            "version_id": "v1",
            "units": [
                {
                    "unit_id": "eu-supported",
                    "node_id": "node-1",
                    "source_kind": "node",
                    "source_id": "node-1",
                    "summary": "动态任务规划算法用于任务分配效率提升。",
                    "text_excerpt": "本文提出动态任务规划算法，并用于提升任务分配效率。",
                    "keywords": ["动态任务规划", "任务分配效率"],
                    "confidence": 0.9,
                },
                {
                    "unit_id": "eu-missing-node",
                    "node_id": "node-missing",
                    "source_kind": "node",
                    "source_id": "node-missing",
                    "text_excerpt": "缺失节点。",
                    "confidence": 0.8,
                },
                {
                    "unit_id": "eu-contradicted",
                    "node_id": "node-2",
                    "source_kind": "node",
                    "source_id": "node-2",
                    "summary": "实验未验证任务成功率提升。",
                    "text_excerpt": "结果显示该方法未验证任务成功率提升，证据不足。",
                    "keywords": ["任务成功率", "未验证"],
                    "confidence": 0.8,
                },
            ],
        }

        verifier = verify_claim_frames_payload(
            "doc-1",
            claim_frames,
            evidence_units,
            node_ids={"node-1", "node-2"},
            source_ids={"node-1", "node-2"},
            citation_map={"references": [], "relations": []},
        )

        by_id = {item["frame_id"]: item for item in verifier["items"]}
        self.assertEqual(by_id["verified"]["trace_status"], "verified")
        self.assertEqual(by_id["verified"]["support_status"], "structurally_supported")
        self.assertEqual(by_id["verified"]["semantic_support_status"], "semantically_supported")
        self.assertEqual(by_id["verified"]["citation_risk"], "safe")
        self.assertEqual(by_id["partial"]["trace_status"], "partial")
        self.assertEqual(by_id["partial"]["support_status"], "unchecked")
        self.assertEqual(by_id["missing"]["trace_status"], "partial")
        self.assertEqual(by_id["missing"]["semantic_support_status"], "insufficient_evidence")
        self.assertEqual(by_id["unsupported"]["semantic_support_status"], "contradicted")
        self.assertEqual(by_id["unsupported"]["citation_risk"], "conflicting_evidence")
        self.assertEqual(by_id["no-evidence"]["trace_status"], "missing")
        self.assertEqual(verifier["schema"], "claim_frame_verifier.v1")
        self.assertEqual(verifier["verified_frame_count"], 2)
        self.assertEqual(verifier["unsupported_frame_count"], 1)
        self.assertEqual(verifier["missing_evidence_unit_count"], 1)
        self.assertEqual(verifier["missing_node_count"], 1)
        self.assertEqual(verifier["contradicted_frame_count"], 1)

    def test_sync_verifier_updates_payload_counts_and_warnings(self) -> None:
        payload = {
            "schema": "claim_frames.v1",
            "frames": [
                {
                    "frame_id": "frame-1",
                    "claim_type": "method",
                    "short_claim": "本文提出动态任务规划算法。",
                    "evidence_unit_ids": ["eu-1"],
                    "warnings": [],
                }
            ],
            "warnings": [],
        }
        verifier = {
            "items": [
                {
                    "frame_id": "frame-1",
                    "trace_status": "verified",
                    "support_status": "structurally_supported",
                    "semantic_support_status": "semantically_supported",
                    "semantic_support_score": 0.9,
                    "semantic_support_reason": "primary_evidence_overlap",
                    "primary_evidence_unit_ids": ["eu-1"],
                    "weak_evidence_unit_ids": [],
                    "contradictory_evidence_unit_ids": [],
                    "citation_risk": "safe",
                    "quality_score": 0.9,
                    "frame_quality": "high",
                    "noise_reasons": [],
                    "warnings": ["verified_warning"],
                }
            ]
        }

        sync_claim_frames_with_verifier(payload, verifier)

        frame = payload["frames"][0]
        self.assertEqual(frame["trace_status"], "verified")
        self.assertEqual(payload["trace_status_counts"], {"verified": 1})
        self.assertEqual(payload["semantic_verified_frame_count"], 1)
        self.assertEqual(payload["citation_risk_counts"], {"safe": 1})
        self.assertIn("verified_warning", payload["warnings"])

    def test_verifier_totals_aggregate_document_fields(self) -> None:
        totals = verifier_totals(
            [
                {
                    "frame_count": 2,
                    "verified_frame_count": 1,
                    "unsupported_frame_count": 1,
                    "trace_status_counts": {"verified": 1, "missing": 1},
                    "support_status_counts": {"structurally_supported": 1, "unsupported": 1},
                    "semantic_support_status_counts": {"semantically_supported": 1, "insufficient_evidence": 1},
                    "semantic_verified_frame_count": 1,
                    "citation_risk_counts": {"safe": 1, "needs_more_evidence": 1},
                    "low_quality_frame_count": 1,
                    "items": [{"noise_reasons": ["front_matter"]}],
                },
                {
                    "frame_count": 1,
                    "verified_frame_count": 1,
                    "unsupported_frame_count": 0,
                    "trace_status_counts": {"verified": 1},
                    "support_status_counts": {"structurally_supported": 1},
                    "semantic_support_status_counts": {"semantically_supported": 1},
                    "semantic_verified_frame_count": 1,
                    "citation_risk_counts": {"safe": 1},
                    "items": [],
                },
            ]
        )

        self.assertEqual(totals["frame_count"], 3)
        self.assertEqual(totals["verified_frame_count"], 2)
        self.assertEqual(totals["trace_status_counts"], {"verified": 2, "missing": 1})
        self.assertEqual(totals["citation_risk_counts"], {"safe": 2, "needs_more_evidence": 1})
        self.assertEqual(totals["top_frame_noise_reasons"], ["front_matter"])


if __name__ == "__main__":
    unittest.main()
