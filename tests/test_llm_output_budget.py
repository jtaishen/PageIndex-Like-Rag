from __future__ import annotations

import unittest
from unittest import mock

from kb_agent.llm import LLMSettings, generate_json_object, llm_payload_metadata


class LLMOutputBudgetTest(unittest.TestCase):
    def test_json_request_caps_tokens_without_raising_global_limit(self) -> None:
        settings = LLMSettings(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            temperature=0,
            max_tokens=3000,
        )
        captured = {}

        def complete(body, _settings, **_kwargs):
            captured.update(body)
            return '{"ok": true}'

        with mock.patch("kb_agent.llm._chat_completion_content", side_effect=complete):
            payload = generate_json_object(
                "system",
                "user",
                settings=settings,
                max_tokens=900,
                retry_count=0,
            )

        self.assertEqual(captured["max_tokens"], 900)
        self.assertEqual(llm_payload_metadata(payload)["max_tokens"], 900)

    def test_json_request_does_not_exceed_smaller_global_limit(self) -> None:
        settings = LLMSettings(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            temperature=0,
            max_tokens=600,
        )
        captured = {}

        def complete(body, _settings, **_kwargs):
            captured.update(body)
            return '{"ok": true}'

        with mock.patch("kb_agent.llm._chat_completion_content", side_effect=complete):
            payload = generate_json_object(
                "system",
                "user",
                settings=settings,
                max_tokens=900,
                retry_count=0,
            )

        self.assertEqual(captured["max_tokens"], 600)
        self.assertEqual(llm_payload_metadata(payload)["max_tokens"], 600)


if __name__ == "__main__":
    unittest.main()
