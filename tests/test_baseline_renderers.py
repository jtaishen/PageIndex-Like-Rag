from __future__ import annotations

import unittest

from kb_agent.baseline_renderers import baseline_html, baseline_markdown


class BaselineRenderersTest(unittest.TestCase):
    def test_baseline_markdown_includes_core_summary_fields(self) -> None:
        report = {
            "schema": "quality_baseline.v1",
            "code_version": "v0.33",
            "git_commit": "abc123",
            "is_current_code_baseline": True,
            "baseline_id": "baseline-1",
            "run_kind": "test_fixture",
            "corpus_fingerprint": "fingerprint",
            "corpus_path": "/tmp/articles",
            "doc_count": 1,
            "pdf_count": 0,
            "benchmark": {"best_mode_by_score": "tree"},
            "llm_baseline": {"status": "completed", "stage_summary": {}},
            "llm_status": {"reachable": True},
            "tasks": {"review_draft": {"status": "completed"}},
            "embedding": {"sentence_transformers": {"status": "skipped"}},
            "documents": [{"doc_id": "doc-1", "quality_level": "good", "section_count": 3, "table_count": 0, "warning_count": 0}],
            "parser_comparison": {"providers": []},
            "recommendations": ["keep monitoring baseline quality"],
            "warnings": [],
        }

        markdown = baseline_markdown(report)

        self.assertIn("# Quality Baseline", markdown)
        self.assertIn("code_version: `v0.33`", markdown)
        self.assertIn("best_search_mode: `tree`", markdown)
        self.assertIn("`doc-1`", markdown)

    def test_baseline_html_escapes_report_values(self) -> None:
        html = baseline_html(
            {
                "baseline_id": "<baseline>",
                "code_version": "v0.33",
                "documents": [{"doc_id": "<doc>", "quality_level": "good"}],
                "parser_comparison": {"providers": []},
                "recommendations": ["review <unsafe>"],
                "warnings": ["warn <unsafe>"],
            }
        )

        self.assertIn("&lt;baseline&gt;", html)
        self.assertIn("&lt;doc&gt;", html)
        self.assertIn("review &lt;unsafe&gt;", html)
        self.assertNotIn("<doc>", html)


if __name__ == "__main__":
    unittest.main()
