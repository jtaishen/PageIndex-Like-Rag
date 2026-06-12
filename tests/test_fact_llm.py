from __future__ import annotations

import os
import unittest
from unittest import mock

from kb_agent.fact_llm import extract_facts_with_llm_batches, normalize_fact_payload
from kb_agent.llm import LLMError


class FactLlmTest(unittest.TestCase):
    def test_normalize_fact_payload_preserves_fact_shapes_and_warnings(self) -> None:
        node = _node("node-1")
        payload = {
            "claims": [{"type": "method", "text": "提出动态角色发现机制以提升任务分解效率。", "evidence": ["N1"], "confidence": 0.84}],
            "entities": [{"type": "method", "name": "动态角色发现机制", "aliases": ["角色发现"], "evidence": ["N1"], "confidence": 0.8}],
            "relations": [{"type": "uses", "subject": "动态角色发现机制", "object": "任务分解", "evidence": ["N1"], "confidence": 0.78}],
        }

        result = normalize_fact_payload(
            payload,
            doc_id="doc-1",
            version_id="v1",
            card={},
            quality={"quality_level": "weak", "quality_warnings": ["missing_abstract"]},
            node_by_id={"node-1": node},
            selected_nodes=[node],
            source="llm",
            status="extracted",
            warnings=[],
        )

        self.assertEqual(result["status"], "extracted")
        self.assertEqual(result["source"], "llm")
        self.assertEqual(result["claims"][0]["type"], "method")
        self.assertEqual(result["entities"][0]["aliases"], ["角色发现"])
        self.assertEqual(result["relations"][0]["relation_type"], "uses")
        self.assertIn("weak_parse_quality", result["warnings"])
        self.assertIn("missing_abstract", result["warnings"])

    def test_batch_extraction_reports_partial_failure(self) -> None:
        nodes = [_node("node-1"), _node("node-2")]
        payload = {
            "claims": [{"type": "method", "text": "提出动态角色发现机制以提升任务分解效率。", "evidence": ["N1"], "confidence": 0.84}],
            "entities": [],
            "relations": [],
        }
        responses: list[object] = [payload, LLMError("timeout", error_type="request_timeout")]

        def fake_generate(system_prompt: str, user_prompt: str) -> dict:
            self.assertIn("fact_batch:", user_prompt)
            del system_prompt
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        with mock.patch.dict(os.environ, {"KB_LLM_FACT_BATCH_SIZE": "1", "KB_LLM_FACT_MAX_NODES": "2"}, clear=False):
            result = extract_facts_with_llm_batches(
                "doc-1",
                "v1",
                _card(),
                {"quality_level": "usable", "quality_warnings": []},
                {},
                {},
                nodes,
                {},
                {node["node_id"]: node for node in nodes},
                [],
                json_generator=fake_generate,
            )

        report = result["llm_batch_report"]
        self.assertEqual(result["status"], "partial")
        self.assertEqual(report["schema"], "llm_fact_batch_report.v1")
        self.assertEqual(report["batch_count"], 2)
        self.assertEqual(report["batch_success_count"], 1)
        self.assertEqual(report["batch_timeout_count"], 1)
        self.assertIn("llm_fact_batch_partial", result["warnings"])

    def test_all_batch_failures_raise_llm_error_with_metadata(self) -> None:
        node = _node("node-1")

        def fail_generate(system_prompt: str, user_prompt: str) -> dict:
            del system_prompt, user_prompt
            raise LLMError("boom", error_type="request_timeout")

        with mock.patch.dict(os.environ, {"KB_LLM_FACT_BATCH_SIZE": "1", "KB_LLM_FACT_MAX_NODES": "1"}, clear=False):
            with self.assertRaises(LLMError) as raised:
                extract_facts_with_llm_batches(
                    "doc-1",
                    "v1",
                    _card(),
                    {"quality_level": "usable", "quality_warnings": []},
                    {},
                    {},
                    [node],
                    {},
                    {"node-1": node},
                    [],
                    json_generator=fail_generate,
                )

        self.assertEqual(raised.exception.error_type, "all_fact_batches_failed")
        self.assertEqual(raised.exception.metadata["batch_count"], 1)
        self.assertEqual(raised.exception.metadata["batch_timeout_count"], 1)


def _card() -> dict:
    return {"title": "Doc A", "abstract": "动态角色任务规划。"}


def _node(node_id: str) -> dict:
    return {
        "doc_id": "doc-1",
        "node_id": node_id,
        "node_path": "1 Method",
        "page_start": 1,
        "page_end": 1,
        "kind": "section",
        "text": "提出动态角色发现机制以提升任务分解效率。",
    }


if __name__ == "__main__":
    unittest.main()
