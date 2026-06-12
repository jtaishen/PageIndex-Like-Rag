from __future__ import annotations

import unittest

from kb_agent.llm import LLMError
from kb_agent.task_compare import build_comparison_matrix


class TaskCompareTest(unittest.TestCase):
    def test_rule_fallback_outputs_existing_matrix_structure(self) -> None:
        result = build_comparison_matrix(
            "task planning",
            [_context("doc-1"), _context("doc-2")],
            {"problem_setting": {"doc-1": [_evidence("doc-1", "n1")], "doc-2": []}},
            [_dimension("problem_setting", "问题设定")],
            warnings=["prepare_warning"],
            use_llm=False,
        )

        matrix = result.matrix
        self.assertEqual(matrix["schema"], "comparison_matrix.v1")
        self.assertEqual(matrix["source"], "rule")
        self.assertEqual(matrix["llm_diagnostics"]["mode"], "disabled")
        self.assertEqual(matrix["dimensions"][0]["id"], "problem_setting")
        self.assertEqual(len(matrix["dimensions"][0]["cells"]), 2)
        self.assertEqual(matrix["dimensions"][0]["cells"][0]["evidence"][0]["node_id"], "n1")
        self.assertIn("missing_evidence:problem_setting:doc-2", matrix["dimensions"][0]["cells"][1]["warnings"])
        self.assertIn("llm_disabled", matrix["warnings"])
        self.assertIn("rule_based_comparison", matrix["warnings"])

    def test_dimension_llm_normalizes_claims_evidence_and_confidence(self) -> None:
        payload = {
            "id": "problem_setting",
            "synthesis": "结构化比较。",
            "overlaps": ["都研究任务规划。"],
            "differences": ["一个偏机器人，一个偏多智能体。"],
            "cells": [
                {
                    "doc_id": "doc-1",
                    "claim": "论文一关注机器人任务规划。",
                    "evidence": [{"id": "n1"}],
                    "confidence": "0.82",
                    "warnings": [],
                }
            ],
            "warnings": [],
        }

        result = build_comparison_matrix(
            "task planning",
            [_context("doc-1")],
            {"problem_setting": {"doc-1": [_evidence("doc-1", "n1")]}},
            [_dimension("problem_setting", "问题设定")],
            warnings=[],
            use_llm=True,
            json_generator=lambda _system, _user: payload,
        )

        matrix = result.matrix
        cell = matrix["dimensions"][0]["cells"][0]
        self.assertEqual(matrix["source"], "llm_dimension")
        self.assertEqual(matrix["llm_diagnostics"]["mode"], "dimension_json")
        self.assertEqual(matrix["llm_diagnostics"]["dimension_success_count"], 1)
        self.assertEqual(cell["claim"], "论文一关注机器人任务规划。")
        self.assertEqual(cell["evidence"][0]["node_id"], "n1")
        self.assertEqual(cell["confidence"], 0.82)

    def test_llm_failure_falls_back_to_rule_with_visible_diagnostics(self) -> None:
        def fail(_system: str, _user: str) -> dict:
            raise LLMError("boom", error_type="request_timeout")

        result = build_comparison_matrix(
            "task planning",
            [_context("doc-1")],
            {"problem_setting": {"doc-1": [_evidence("doc-1", "n1")]}},
            [_dimension("problem_setting", "问题设定")],
            warnings=[],
            use_llm=True,
            json_generator=fail,
        )

        self.assertEqual(result.matrix["source"], "rule")
        self.assertEqual(result.matrix["llm_diagnostics"]["mode"], "fallback_rule")
        self.assertEqual(result.matrix["llm_diagnostics"]["error_type"], "request_timeout")
        self.assertTrue(result.llm_error)
        self.assertTrue(any(str(item).startswith("llm_unavailable:") for item in result.matrix["warnings"]))


def _dimension(dimension_id: str, name: str) -> dict:
    return {"id": dimension_id, "name": name, "search_terms": ["task"]}


def _context(doc_id: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"Paper {doc_id}",
        "description": "This paper studies task planning.",
        "innovation": {"items": [{"title": "method", "claim": "improves planning"}]},
        "facts": {"available": False},
        "quality": {"section_count": 1, "quality_warnings": []},
        "citation_map": {"references": []},
    }


def _evidence(doc_id: str, node_id: str) -> dict:
    return {
        "doc_id": doc_id,
        "node_id": node_id,
        "node_path": "1",
        "summary": "task planning evidence",
        "excerpt": "task planning evidence excerpt",
    }


if __name__ == "__main__":
    unittest.main()
