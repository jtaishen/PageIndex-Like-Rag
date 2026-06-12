from __future__ import annotations

import unittest

from kb_agent.claim_frame_builder import (
    LLM_ENHANCE_FRAME_LIMIT,
    LLM_ENHANCE_UNIT_LIMIT,
    build_claim_frames,
    claim_frames_payload,
    enhance_frames_with_llm,
    frame_record,
)
from kb_agent.claim_frame_evidence import unit_by_id, unit_by_node_id, unit_by_source_id


class ClaimFrameBuilderTest(unittest.TestCase):
    def test_builds_frames_from_claim_innovation_table_and_citation_sources(self) -> None:
        units = _units()
        frames = build_claim_frames(
            "doc-1",
            "v1",
            {"title": "Doc A"},
            {
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "claim_type": "method",
                        "text": "本文提出动态任务规划方法。",
                        "node_id": "node-1",
                        "confidence": 0.8,
                    }
                ]
            },
            {"items": [{"type": "contribution", "claim": "提出动态角色发现机制。", "evidence": [{"node_id": "node-2"}]}]},
            {"table_summaries": [{"table_id": "table-1", "caption": "表 1", "summary": "任务成功率提升。"}]},
            {"relations": [{"ref_id": "R1", "node_id": "node-3"}]},
            unit_by_node_id(units),
            unit_by_id(units),
            unit_by_source_id(units),
        )

        sources = {frame["source"] for frame in frames}
        self.assertIn("claim", sources)
        self.assertIn("innovation", sources)
        self.assertIn("table_summary", sources)
        self.assertIn("citation_map", sources)
        self.assertTrue(all(frame["frame_id"] and "quality_score" in frame for frame in frames))
        self.assertTrue(any(frame["evidence_unit_ids"] for frame in frames))

    def test_low_quality_table_frame_without_evidence_is_marked(self) -> None:
        frame = frame_record(
            "doc-1",
            "v1",
            "result",
            "目录",
            [],
            source="table_summary",
            source_claim_ids=["table-1"],
            confidence=0.2,
            index=0,
        )

        self.assertEqual(frame["trace_status"], "missing")
        self.assertLess(frame["quality_score"], 0.5)
        self.assertIn("low_quality_frame", frame["warnings"])
        self.assertIn("missing_evidence_unit", frame["warnings"])

    def test_payload_preserves_counts_quality_summary_and_warnings(self) -> None:
        frame = frame_record(
            "doc-1",
            "v1",
            "method",
            "本文提出动态任务规划方法。",
            ["eu-node-1"],
            source="claim",
            source_claim_ids=["claim-1"],
            confidence=0.8,
            index=0,
        )

        payload = claim_frames_payload(
            "doc-1",
            "v1",
            [frame],
            evidence_unit_count=2,
            warnings=["manual_warning"],
            llm_used=False,
            llm_error="",
            llm_metadata={},
        )

        self.assertEqual(payload["schema"], "claim_frames.v1")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["evidence_unit_count"], 2)
        self.assertEqual(payload["claim_type_counts"], {"method": 1})
        self.assertEqual(payload["quality_summary"]["schema"], "claim_frame_quality_summary.v1")
        self.assertIn("manual_warning", payload["warnings"])

    def test_llm_enhancement_uses_injected_generator_and_records_truncation(self) -> None:
        frames = [
            frame_record(
                "doc-1",
                "v1",
                "method",
                f"本文提出第 {index} 个动态任务规划方法。",
                ["eu-1"],
                source="claim",
                source_claim_ids=[f"claim-{index}"],
                confidence=0.8,
                index=index,
            )
            for index in range(LLM_ENHANCE_FRAME_LIMIT + 1)
        ]
        units = [{"unit_id": f"eu-{index}", "unit_type": "paragraph", "summary": "任务规划方法", "keywords": ["任务规划"]} for index in range(LLM_ENHANCE_UNIT_LIMIT + 1)]

        def fake_generate(system_prompt: str, user_prompt: str, **kwargs: object) -> dict:
            self.assertEqual(kwargs["operation"], "claim_frames")
            self.assertEqual(kwargs["stage"], "enhance")
            self.assertIn("claim_frames:", user_prompt)
            del system_prompt
            return {
                "frames": [{"frame_id": frames[0]["frame_id"], "method": "LLM 补全的方法摘要", "confidence": 0.91}],
                "_llm_metadata": {"retry_count": 0},
            }

        enhanced, metadata = enhance_frames_with_llm({"title": "测试论文"}, frames, units, json_generator=fake_generate)

        self.assertEqual(enhanced[0]["method"], "LLM 补全的方法摘要")
        self.assertEqual(enhanced[0]["confidence"], 0.91)
        self.assertTrue(metadata["llm_enhancement"]["truncated"])
        self.assertIn("llm_frame_enhancement_truncated", metadata["enhancement_warnings"])
        self.assertIn("llm_unit_context_truncated", metadata["enhancement_warnings"])


def _units() -> list[dict]:
    return [
        {"unit_id": "eu-node-1", "node_id": "node-1", "source_id": "node-1", "source_kind": "node", "text_excerpt": "本文提出动态任务规划方法。"},
        {"unit_id": "eu-node-2", "node_id": "node-2", "source_id": "node-2", "source_kind": "node", "text_excerpt": "提出动态角色发现机制。"},
        {"unit_id": "eu-node-3", "node_id": "node-3", "source_id": "node-3", "source_kind": "node", "text_excerpt": "相关研究 [R1]。"},
        {"unit_id": "eu-table", "node_id": "", "source_id": "table-1", "source_kind": "table", "text_excerpt": "任务成功率提升。"},
        {"unit_id": "eu-ref", "node_id": "", "source_id": "R1", "source_kind": "citation", "text_excerpt": "R1 任务规划研究。"},
    ]


if __name__ == "__main__":
    unittest.main()
