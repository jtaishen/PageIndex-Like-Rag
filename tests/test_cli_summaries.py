from __future__ import annotations

import unittest

from kb_agent.cli_summaries import (
    fact_eval_summary,
    fact_summary,
    memory_eval_summary,
    quality_baseline_cli_summary,
    review_eval_summary,
)


class CliSummariesTest(unittest.TestCase):
    def test_fact_summary_reports_counts_and_warning_defaults(self) -> None:
        result = fact_summary(
            {
                "schema": "paper_facts.v1",
                "doc_id": "doc-1",
                "version_id": "v1",
                "artifact_dir": "artifacts/doc-1/v1",
                "claims_path": "claims.json",
                "entities_path": "entities.json",
                "relations_path": "relations.json",
                "fact_graph_path": "fact_graph.json",
                "fact_report": {
                    "status": "ok",
                    "claim_count": 3,
                    "entity_count": 2,
                    "relation_count": 1,
                    "low_confidence_count": 0,
                    "no_evidence_count": 0,
                },
            }
        )

        self.assertEqual(result["doc_id"], "doc-1")
        self.assertEqual(result["claim_count"], 3)
        self.assertEqual(result["entity_count"], 2)
        self.assertEqual(result["relation_count"], 1)
        self.assertEqual(result["warnings"], [])

    def test_quality_baseline_summary_exposes_runtime_limits(self) -> None:
        result = quality_baseline_cli_summary(
            {
                "schema": "quality_baseline.v1",
                "code_version": "v0.32",
                "baseline_id": "baseline-1",
                "benchmark": {"best_mode_by_score": "tree"},
                "llm_baseline": {
                    "status": "partial",
                    "timeout_count": 2,
                    "hard_timeout_count": 1,
                    "budget_exhausted": True,
                },
                "warnings": ["llm_timeout"],
            }
        )

        self.assertEqual(result["code_version"], "v0.32")
        self.assertEqual(result["best_search_mode"], "tree")
        self.assertEqual(result["llm_baseline_status"], "partial")
        self.assertEqual(result["llm_timeout_count"], 2)
        self.assertEqual(result["llm_hard_timeout_count"], 1)
        self.assertTrue(result["llm_budget_exhausted"])
        self.assertEqual(result["warning_count"], 1)

    def test_eval_summary_names_are_public_and_consistent(self) -> None:
        self.assertTrue(callable(review_eval_summary))
        self.assertTrue(callable(memory_eval_summary))
        self.assertTrue(callable(fact_eval_summary))


if __name__ == "__main__":
    unittest.main()
