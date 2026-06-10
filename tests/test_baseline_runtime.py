from __future__ import annotations

import time
import unittest

from kb_agent.baseline_runtime import LLMBaselineRuntime, null_stage


class BaselineRuntimeTest(unittest.TestCase):
    def test_stage_summary_counts_fallback_timeout_and_slow_calls(self) -> None:
        runtime = LLMBaselineRuntime(
            enabled=True,
            timeout_seconds=1,
            total_timeout_seconds=30,
            stage_timeout_seconds=30,
            max_docs=2,
            skip_tasks=False,
        )
        stage = runtime.stage("llm_facts")
        with stage:
            stage.record_call(1500)
            stage.record_timeout(hard=True)
            stage.record_fallback()

        summary = runtime.summary()

        self.assertEqual(summary["stage_summary"]["llm_facts"]["call_count"], 1)
        self.assertEqual(summary["stage_summary"]["llm_facts"]["timeout_count"], 1)
        self.assertEqual(summary["stage_summary"]["llm_facts"]["hard_timeout_count"], 1)
        self.assertEqual(summary["stage_summary"]["llm_facts"]["slow_call_count"], 1)
        self.assertEqual(summary["stage_summary"]["llm_facts"]["fallback_count"], 1)
        self.assertEqual(summary["timeout_count"], 1)
        self.assertEqual(summary["hard_timeout_count"], 1)
        self.assertEqual(summary["slow_call_count"], 1)
        self.assertEqual(summary["fallback_count"], 1)

    def test_budget_remaining_marks_exhausted_after_total_timeout(self) -> None:
        runtime = LLMBaselineRuntime(
            enabled=True,
            timeout_seconds=1,
            total_timeout_seconds=1,
            stage_timeout_seconds=30,
            max_docs=0,
            skip_tasks=False,
        )
        runtime.started = time.time() - 2

        self.assertFalse(runtime.budget_remaining())
        self.assertTrue(runtime.summary()["budget_exhausted"])

    def test_null_stage_is_noop_context_manager(self) -> None:
        with null_stage() as stage:
            stage.record_call(100)
            stage.record_timeout()
            stage.record_fallback()

        self.assertEqual(stage.summary()["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
