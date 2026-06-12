from __future__ import annotations

import unittest

from kb_agent.llm import LLMError
from kb_agent.task_review_plan import build_review_outline


class TaskReviewPlanTest(unittest.TestCase):
    def test_rule_fallback_outputs_existing_outline_structure(self) -> None:
        result = build_review_outline(
            "task planning",
            [_context("doc-1")],
            {"background_problem": [_evidence("doc-1", "n1")]},
            [_section("background_problem", "研究背景与问题定义")],
            warnings=["prepare_warning"],
            use_llm=False,
        )

        outline = result.outline
        self.assertEqual(outline["schema"], "review_outline.v1")
        self.assertEqual(outline["source"], "rule")
        self.assertEqual(outline["llm_diagnostics"]["mode"], "disabled")
        self.assertEqual(outline["sections"][0]["section_id"], "background_problem")
        self.assertEqual(outline["sections"][0]["paper_ids"], ["doc-1"])
        self.assertEqual(outline["sections"][0]["evidence"][0]["node_id"], "n1")
        self.assertIn("llm_disabled", outline["warnings"])
        self.assertIn("rule_based_review_plan", outline["warnings"])

    def test_full_llm_normalizes_sections_and_evidence_refs(self) -> None:
        payload = {
            "title": "任务规划综述",
            "scope": "聚焦任务规划。",
            "sections": [
                {
                    "section_id": "background_problem",
                    "title": "背景",
                    "purpose": "界定问题。",
                    "paper_ids": ["doc-1"],
                    "evidence": [{"id": "n1"}],
                    "warnings": [],
                }
            ],
            "open_questions": ["需要补充评测。"],
            "warnings": [],
        }

        result = build_review_outline(
            "task planning",
            [_context("doc-1")],
            {"background_problem": [_evidence("doc-1", "n1")]},
            [_section("background_problem", "研究背景与问题定义")],
            warnings=[],
            use_llm=True,
            json_generator=lambda _system, _user: payload,
        )

        outline = result.outline
        section = outline["sections"][0]
        self.assertEqual(outline["source"], "llm")
        self.assertEqual(outline["llm_diagnostics"]["mode"], "full_json")
        self.assertEqual(outline["title"], "任务规划综述")
        self.assertEqual(section["title"], "背景")
        self.assertEqual(section["evidence"][0]["node_id"], "n1")
        self.assertEqual(section["source_doc_count"], 1)

    def test_full_llm_failure_recovers_with_section_json(self) -> None:
        calls = []

        def generate(_system: str, user: str) -> dict:
            calls.append(user)
            if len(calls) == 1:
                raise LLMError("truncated", error_type="truncated_json", metadata={"retry_count": 1})
            return {
                "section_id": "background_problem",
                "title": "背景",
                "purpose": "界定问题。",
                "paper_ids": ["doc-1"],
                "evidence": [{"id": "n1"}],
                "warnings": [],
            }

        result = build_review_outline(
            "task planning",
            [_context("doc-1")],
            {"background_problem": [_evidence("doc-1", "n1")]},
            [_section("background_problem", "研究背景与问题定义")],
            warnings=[],
            use_llm=True,
            json_generator=generate,
        )

        outline = result.outline
        self.assertEqual(outline["source"], "llm_section")
        self.assertEqual(outline["llm_diagnostics"]["mode"], "section_json")
        self.assertEqual(outline["llm_diagnostics"]["retry_count"], 1)
        self.assertEqual(outline["llm_diagnostics"]["fallback_sections"], [])
        self.assertIn("section_json_recovery", outline["warnings"])
        self.assertEqual(outline["sections"][0]["evidence"][0]["node_id"], "n1")


def _section(section_id: str, title: str) -> dict:
    return {
        "section_id": section_id,
        "title": title,
        "purpose": "界定主题范围。",
        "search_terms": ["task"],
    }


def _context(doc_id: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"Paper {doc_id}",
        "description": "This paper studies task planning.",
        "innovation": {"items": [{"title": "method", "claim": "improves planning"}]},
        "facts": {"available": False},
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
