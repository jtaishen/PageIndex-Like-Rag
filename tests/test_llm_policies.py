from __future__ import annotations

import os
import unittest
from unittest import mock

from kb_agent.llm_policies import structured_json_generator, structured_llm_policy


class StructuredLLMPolicyTest(unittest.TestCase):
    def test_structured_operations_disable_reasoning_and_bound_runtime(self) -> None:
        operations = {
            "doc_card_summary": 700,
            "query_classification": 500,
            "tree_search": 700,
            "insight_extraction": 1200,
            "fact_extraction": 1200,
            "claim_frame": 1200,
            "compare": 1600,
            "review_outline": 900,
            "review_draft": 900,
        }
        with mock.patch.dict(
            os.environ,
            {"KB_MCP_LLM_STEP_TIMEOUT_SECONDS": "45", "KB_MCP_REVIEW_DRAFT_MAX_TOKENS": "900"},
            clear=False,
        ):
            for operation, max_tokens in operations.items():
                policy = structured_llm_policy(operation)
                self.assertFalse(policy.thinking)
                self.assertLessEqual(policy.timeout_seconds, 25)
                self.assertEqual(policy.retry_count, 1)
                self.assertEqual(policy.max_tokens, max_tokens)

    def test_generator_passes_policy_without_prompt_or_secret_metadata(self) -> None:
        with mock.patch("kb_agent.llm_policies.generate_json_object", return_value={"ok": True}) as generate:
            result = structured_json_generator("fact_extraction", "batch_001")("system", "user")

        self.assertEqual(result, {"ok": True})
        generate.assert_called_once_with(
            "system",
            "user",
            timeout_seconds=25,
            retry_count=1,
            max_tokens=1200,
            thinking=False,
            operation="fact_extraction",
            stage="batch_001",
        )


if __name__ == "__main__":
    unittest.main()
