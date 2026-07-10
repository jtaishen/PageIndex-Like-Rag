from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kb_agent.workflow_state import (
    create_workflow_state,
    finish_workflow_step,
    get_workflow_state,
    single_request_json_generator,
    start_workflow_step,
)


class WorkflowStateTest(unittest.TestCase):
    def test_state_persists_retryable_failed_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"
            task_id = "task_abcdef123456"
            create_workflow_state(
                db_path,
                task_id,
                "review",
                [{"step_id": "outline:background", "phase": "outline"}],
                phase="outline",
            )
            start_workflow_step(db_path, task_id, "outline:background")
            failed = finish_workflow_step(
                db_path,
                task_id,
                "outline:background",
                status="failed",
                error_type="request_timeout",
            )

            self.assertEqual(failed["summary"]["failed_steps"], ["outline:background"])
            self.assertEqual(failed["summary"]["pending_steps"], ["outline:background"])
            persisted = get_workflow_state(db_path, task_id)
            self.assertEqual(persisted["steps"][0]["attempt_count"], 1)
            self.assertEqual(persisted["steps"][0]["error_type"], "request_timeout")

    def test_single_request_generator_disables_json_retry(self) -> None:
        with mock.patch.dict(os.environ, {"KB_MCP_LLM_STEP_TIMEOUT_SECONDS": "31"}, clear=False), mock.patch(
            "kb_agent.workflow_state.generate_json_object",
            return_value={"ok": True},
        ) as generate:
            result = single_request_json_generator("review", "background")("system", "user")

        self.assertEqual(result, {"ok": True})
        generate.assert_called_once_with(
            "system",
            "user",
            timeout_seconds=31,
            retry_count=0,
            operation="review",
            stage="background",
        )

    def test_single_request_generator_can_bound_output_tokens(self) -> None:
        with mock.patch(
            "kb_agent.workflow_state.generate_json_object",
            return_value={"ok": True},
        ) as generate:
            result = single_request_json_generator(
                "review_draft",
                "method_paradigms",
                max_tokens=900,
            )("system", "user")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(generate.call_args.kwargs["max_tokens"], 900)
        self.assertEqual(generate.call_args.kwargs["retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
