from __future__ import annotations

import json
import unittest
from unittest import mock

from kb_agent.llm import LLMError, LLMSettings, generate_json_object, llm_payload_metadata


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
                thinking=False,
                retry_count=0,
            )

        self.assertEqual(captured["max_tokens"], 900)
        self.assertEqual(captured["thinking"], {"type": "disabled"})
        self.assertEqual(llm_payload_metadata(payload)["max_tokens"], 900)
        self.assertEqual(llm_payload_metadata(payload)["thinking_mode"], "disabled")

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

    def test_reasoning_only_length_response_reports_output_token_limit(self) -> None:
        settings = LLMSettings(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            temperature=0,
            max_tokens=3000,
        )
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "", "reasoning_content": "reasoning"},
                    }
                ],
                "usage": {"completion_tokens": 900},
            }
        ).encode("utf-8")

        with mock.patch("kb_agent.llm.urllib.request.urlopen", return_value=response):
            with self.assertRaises(LLMError) as raised:
                generate_json_object(
                    "system",
                    "user",
                    settings=settings,
                    max_tokens=900,
                    retry_count=0,
                )

        self.assertEqual(raised.exception.error_type, "output_token_limit")
        self.assertEqual(raised.exception.metadata["finish_reason"], "length")
        self.assertEqual(raised.exception.metadata["completion_tokens"], 900)
        self.assertTrue(raised.exception.metadata["reasoning_content_present"])
        self.assertEqual(raised.exception.metadata["max_tokens"], 900)

    def test_truncated_content_preserves_safe_response_diagnostics(self) -> None:
        settings = LLMSettings(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            temperature=0,
            max_tokens=1200,
        )
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": '{"items":[', "reasoning_content": ""},
                    }
                ],
                "usage": {"prompt_tokens": 240, "completion_tokens": 1200, "total_tokens": 1440},
            }
        ).encode("utf-8")

        with mock.patch("kb_agent.llm.urllib.request.urlopen", return_value=response):
            with self.assertRaises(LLMError) as raised:
                generate_json_object(
                    "system",
                    "user",
                    settings=settings,
                    max_tokens=1200,
                    thinking=False,
                    retry_count=0,
                )

        self.assertEqual(raised.exception.error_type, "truncated_json")
        self.assertEqual(raised.exception.metadata["finish_reason"], "length")
        self.assertEqual(raised.exception.metadata["completion_tokens"], 1200)
        self.assertEqual(raised.exception.metadata["thinking_mode"], "disabled")


if __name__ == "__main__":
    unittest.main()
