from __future__ import annotations

import unittest

from kb_agent.claim_frame_normalization import normalize_claim_frame_fields
from kb_agent.claim_frames import _frame_record


class ClaimFrameNormalizationTest(unittest.TestCase):
    def test_normalizes_technical_plan_fields_from_short_claim(self) -> None:
        frame = _frame_record(
            "doc-1",
            "v1",
            "result",
            "本文提出基于大语言模型的任务规划框架，在服务机器人真实环境中提升任务成功率并降低响应时间。",
            ["eu-1"],
            source="claim",
            source_claim_ids=["claim-1"],
            confidence=0.82,
            index=0,
        )

        self.assertEqual(frame["method_family"], "llm_planning")
        self.assertEqual(frame["metric"], "任务成功率、响应时间")
        self.assertEqual(frame["claimed_gain"], frame["result_or_gain"])
        self.assertIn("服务机器人", frame["condition"])
        self.assertEqual(frame["polarity"], "positive")
        self.assertTrue(frame["normalized_subject"])

    def test_normalization_reports_missing_metric_for_result_claim(self) -> None:
        normalized = normalize_claim_frame_fields(
            {
                "claim_type": "result",
                "short_claim": "本文实验结果显示方法整体有效。",
                "result_or_gain": "方法整体有效",
            }
        )

        self.assertIn("missing_metric", normalized["normalization_warnings"])
        self.assertEqual(normalized["polarity"], "neutral")


if __name__ == "__main__":
    unittest.main()
