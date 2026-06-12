from __future__ import annotations

import unittest

from kb_agent.llm import LLMError
from kb_agent.review_section_draft import (
    build_section_draft,
    build_skipped_section_draft,
    prepare_numbered_draft_evidence,
)


class ReviewSectionDraftTest(unittest.TestCase):
    def test_prepare_numbered_evidence_dedupes_and_preserves_cross_doc_coverage(self) -> None:
        evidence = [
            _evidence("doc-1", "n1", "low", tree_score=0.1),
            _evidence("doc-1", "n1", "high", tree_score=0.9),
            _evidence("doc-2", "n2", "other", tree_score=0.8),
        ]

        numbered, compaction = prepare_numbered_draft_evidence(evidence, max_items=2)

        self.assertEqual([item["ref_id"] for item in numbered], ["E1", "E2"])
        self.assertEqual([item["summary"] for item in numbered], ["high", "other"])
        self.assertEqual(compaction["schema"], "review_draft_compaction.v1")
        self.assertEqual(compaction["duplicate_evidence_removed"], 1)
        self.assertEqual(compaction["kept_evidence_count"], 2)
        self.assertEqual(compaction["source_doc_ids"], ["doc-1", "doc-2"])
        self.assertIn("draft_duplicate_evidence_compacted", compaction["warnings"])

    def test_llm_draft_normalizes_body_claim_plan_and_used_evidence(self) -> None:
        evidence, compaction = prepare_numbered_draft_evidence([_evidence("doc-1", "n1", "one")])
        payload = {
            "claim_plan": [{"claim": "任务规划需要动态约束证据。", "evidence": ["E1"]}],
            "body_markdown": "任务规划章节需要结合动态约束和执行证据进行讨论。[E1]",
            "unsupported_claims": [],
            "warnings": [],
        }

        result = build_section_draft(
            "review-test",
            _outline(),
            _section(),
            evidence,
            compaction,
            use_llm=True,
            json_generator=lambda _system, _user: payload,
        )

        draft = result.draft
        self.assertEqual(draft["schema"], "section_draft.v1")
        self.assertEqual(draft["source"], "llm")
        self.assertEqual(draft["status"], "drafted")
        self.assertEqual(draft["claim_plan"][0]["evidence"], ["E1"])
        self.assertEqual([item["ref_id"] for item in draft["used_evidence"]], ["E1"])
        self.assertTrue(draft["llm_diagnostics"]["used"])
        self.assertEqual(draft["paragraph_support_report"]["supported_paragraph_count"], 1)

    def test_llm_draft_removes_unsupported_paragraphs(self) -> None:
        evidence, compaction = prepare_numbered_draft_evidence([_evidence("doc-1", "n1", "one")])
        payload = {
            "body_markdown": (
                "这个段落声称系统已经解决所有长期开放问题，但没有任何证据引用。\n\n"
                "这个段落只讨论证据中出现的动态约束处理能力，并带有引用。[E1]"
            ),
            "unsupported_claims": [],
            "warnings": [],
        }

        result = build_section_draft(
            "review-test",
            _outline(),
            _section(),
            evidence,
            compaction,
            use_llm=True,
            json_generator=lambda _system, _user: payload,
        )

        draft = result.draft
        self.assertIn("unsupported_paragraphs_removed", draft["warnings"])
        self.assertNotIn("长期开放问题", draft["body_markdown"])
        self.assertEqual(draft["paragraph_support_report"]["removed_paragraph_count"], 1)
        self.assertEqual(draft["paragraph_support_report"]["unsupported_paragraph_count"], 0)
        self.assertTrue(draft["unsupported_claims"])

    def test_llm_error_falls_back_to_rule_with_visible_diagnostics(self) -> None:
        evidence, compaction = prepare_numbered_draft_evidence([_evidence("doc-1", "n1", "one")])

        def fail(_system: str, _user: str) -> dict:
            raise LLMError("boom", error_type="request_timeout")

        result = build_section_draft(
            "review-test",
            _outline(),
            _section(),
            evidence,
            compaction,
            use_llm=True,
            json_generator=fail,
        )

        draft = result.draft
        self.assertEqual(draft["source"], "rule")
        self.assertTrue(result.llm_error)
        self.assertIn("rule_based_section_draft", draft["warnings"])
        self.assertTrue(any(str(item).startswith("llm_unavailable:") for item in draft["warnings"]))
        self.assertEqual(draft["llm_diagnostics"]["error_type"], "request_timeout")

    def test_skipped_section_keeps_section_draft_schema(self) -> None:
        evidence, compaction = prepare_numbered_draft_evidence([_evidence("doc-1", "n1", "one")])

        draft = build_skipped_section_draft(
            "review-test",
            _section(),
            evidence,
            reason="budget_exhausted",
            compaction=compaction,
        )

        self.assertEqual(draft["schema"], "section_draft.v1")
        self.assertEqual(draft["status"], "skipped")
        self.assertEqual(draft["source"], "skipped")
        self.assertIn("section_draft_skipped", draft["warnings"])
        self.assertEqual(draft["llm_diagnostics"]["fallback_reason"], "budget_exhausted")


def _outline() -> dict:
    return {"title": "任务规划综述", "topic": "task planning"}


def _section() -> dict:
    return {
        "section_id": "background_problem",
        "title": "研究背景与问题定义",
        "purpose": "界定任务规划问题。",
    }


def _evidence(doc_id: str, node_id: str, summary: str, *, tree_score: float = 0.5) -> dict:
    return {
        "doc_id": doc_id,
        "node_id": node_id,
        "node_path": "1",
        "title": f"Paper {doc_id}",
        "summary": summary,
        "excerpt": f"{summary} evidence excerpt for task planning.",
        "tree_score": tree_score,
    }


if __name__ == "__main__":
    unittest.main()
