from __future__ import annotations

import unittest

from kb_agent.review_quality import assemble_review_markdown, build_citation_check, build_review_report, paragraph_support_report


class ReviewQualityTest(unittest.TestCase):
    def test_citation_check_detects_missing_refs_unsupported_and_unused_evidence(self) -> None:
        draft = _draft(
            body=(
                "这一段引用了不存在的证据编号，因此应该被标记为 missing ref。[E9]\n\n"
                "这一段提出了一个没有任何证据标记的长观点，应该被识别为 unsupported paragraph。"
            ),
            evidence=[_evidence("E1")],
        )

        check = build_citation_check("review-test", [draft])

        self.assertEqual(check["schema"], "citation_check.v1")
        self.assertEqual(check["status"], "partial")
        self.assertEqual(check["missing_refs"], [{"section_id": "background_problem", "ref_id": "E9"}])
        self.assertEqual(check["unused_evidence"], [{"section_id": "background_problem", "ref_id": "E1"}])
        self.assertEqual(check["unsupported_paragraph_count"], 1)
        self.assertIn("missing_evidence_refs", check["warnings"])
        self.assertIn("unsupported_paragraphs", check["warnings"])

    def test_optional_unused_evidence_does_not_create_hard_unused_warning(self) -> None:
        draft = _draft(
            body="这一段有足够的证据引用来支持核心观点，因此未使用证据可以作为可选冗余处理。[E1]",
            evidence=[_evidence("E1"), _evidence("E2")],
        )

        check = build_citation_check("review-test", [draft])

        self.assertEqual(check["unused_evidence"], [])
        self.assertEqual(check["optional_unused_evidence"], [{"section_id": "background_problem", "ref_id": "E2"}])
        self.assertNotIn("unused_evidence", check["warnings"])

    def test_review_report_builds_quality_reasons_and_section_actions(self) -> None:
        outline = {
            "title": "任务规划综述",
            "sections": [
                {
                    "section_id": "background_problem",
                    "title": "研究背景与问题定义",
                }
            ],
        }
        draft = _draft(
            body="这一段有证据引用，详细说明任务规划综述草稿来自规则降级路径，因此后续仍然需要人工润色和章节衔接检查。[E1]",
            evidence=[_evidence("E1")],
            warnings=["rule_based_section_draft"],
        )
        check = build_citation_check("review-test", [draft])

        report = build_review_report("review-test", outline, [draft], check)

        self.assertEqual(report["schema"], "review_report.v1")
        self.assertEqual(report["draft_quality_level"], "good")
        self.assertEqual(report["section_statuses"][0]["draft_quality_level"], "usable")
        self.assertIn("rule_based_section_draft", report["quality_reasons"])
        self.assertEqual(report["section_revision_actions"][0]["section_id"], "background_problem")
        self.assertIn("人工润色规则版草稿，补足章节衔接。", report["section_revision_actions"][0]["actions"])
        self.assertEqual(report["paragraph_support_report"]["supported_paragraph_count"], 1)

    def test_assemble_review_markdown_keeps_title_sections_and_evidence_note(self) -> None:
        markdown = assemble_review_markdown(
            {"title": "任务规划综述", "scope": "聚焦动态任务规划。"},
            [_draft(body="章节正文包含证据引用。[E1]", evidence=[_evidence("E1")])],
        )

        self.assertIn("# 任务规划综述", markdown)
        self.assertIn("聚焦动态任务规划。", markdown)
        self.assertIn("## 研究背景与问题定义", markdown)
        self.assertIn("章节正文包含证据引用。[E1]", markdown)
        self.assertIn("## 证据说明", markdown)


def _draft(*, body: str, evidence: list[dict], warnings: list[str] | None = None) -> dict:
    return {
        "schema": "section_draft.v1",
        "section_id": "background_problem",
        "title": "研究背景与问题定义",
        "status": "partial" if warnings else "drafted",
        "source": "rule" if warnings else "llm",
        "body_markdown": body,
        "evidence": evidence,
        "paragraph_support_report": paragraph_support_report(body, evidence),
        "warnings": warnings or [],
    }


def _evidence(ref_id: str) -> dict:
    return {
        "ref_id": ref_id,
        "doc_id": "doc-1",
        "node_id": ref_id.lower(),
        "summary": f"Evidence {ref_id}",
    }


if __name__ == "__main__":
    unittest.main()
