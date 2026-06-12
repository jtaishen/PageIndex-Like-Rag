from __future__ import annotations

import unittest

from kb_agent.cli_summaries import (
    fact_eval_summary,
    fact_summary,
    memory_eval_summary,
    quality_baseline_cli_summary,
    review_eval_summary,
    task_summary,
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

    def test_task_summary_separates_relation_types_from_conflict_classes(self) -> None:
        result = task_summary(
            {
                "task_id": "task-1",
                "task_type": "compare",
                "status": "partial",
                "comparison_matrix": {
                    "evidence_coverage": {"schema": "evidence_coverage.v1"},
                    "claim_alignment_summary": {
                        "available": True,
                        "group_count": 3,
                        "relation_count": 2,
                        "relation_type_counts": {"supports": 1, "contradicts": 1},
                        "conflict_classification_counts": {"supports": 1, "contradicts": 1, "incomparable": 1},
                        "incomparable_pair_count": 1,
                        "avg_claim_align_score": 0.42,
                        "max_claim_align_score": 0.68,
                    },
                    "warnings": [],
                },
            }
        )

        self.assertEqual(result["claim_relation_type_counts"], {"supports": 1, "contradicts": 1})
        self.assertEqual(result["claim_conflict_classification_counts"], {"supports": 1, "contradicts": 1, "incomparable": 1})
        self.assertEqual(result["claim_incomparable_pair_count"], 1)
        self.assertEqual(result["claim_avg_align_score"], 0.42)
        self.assertEqual(result["claim_max_align_score"], 0.68)

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
                "tasks": {
                    "compare": {
                        "answer_plan_summary": {
                            "available": True,
                            "answerability": "answerable",
                            "strong_claim_count": 2,
                            "qualified_claim_count": 1,
                            "conflicting_claim_count": 0,
                            "insufficient_claim_count": 0,
                            "warnings": [],
                        },
                        "claim_alignment_summary": {
                            "available": True,
                            "group_count": 2,
                            "method_family_group_count": 1,
                            "conflicting_group_count": 0,
                            "research_gap_count": 1,
                            "relation_count": 2,
                            "relation_type_counts": {"supports": 1, "same_metric": 1},
                            "conflict_classification_counts": {"supports": 2, "incomparable": 1},
                            "incomparable_pair_count": 1,
                            "avg_claim_align_score": 0.5,
                            "max_claim_align_score": 0.8,
                            "warnings": ["claim_alignment_insufficient_evidence"],
                        },
                    },
                    "review": {
                        "answer_plan_summary": {
                            "available": True,
                            "answerability": "conflicting",
                            "strong_claim_count": 1,
                            "qualified_claim_count": 0,
                            "conflicting_claim_count": 1,
                            "insufficient_claim_count": 2,
                            "warnings": ["answer_plan_conflicting_claims"],
                        },
                        "claim_alignment_summary": {
                            "available": True,
                            "group_count": 1,
                            "method_family_group_count": 0,
                            "conflicting_group_count": 1,
                            "research_gap_count": 0,
                            "relation_count": 1,
                            "relation_type_counts": {"contradicts": 1},
                            "conflict_classification_counts": {"contradicts": 1},
                            "incomparable_pair_count": 0,
                            "avg_claim_align_score": 0.3,
                            "max_claim_align_score": 0.6,
                            "warnings": ["claim_alignment_conflicts"],
                        },
                    },
                },
                "warnings": ["llm_timeout"],
                "claim_frame_verification": {
                    "frame_count": 4,
                    "verified_frame_rate": 0.75,
                    "unsupported_frame_count": 1,
                    "low_quality_frame_count": 2,
                    "noisy_frame_count": 1,
                    "ignored_noise_frame_count": 1,
                    "top_frame_noise_reasons": ["front_matter"],
                    "semantic_support_status_counts": {"semantically_supported": 2, "related_only": 1},
                    "semantic_verified_frame_count": 2,
                    "semantic_supported_frame_rate": 0.5,
                    "partial_supported_frame_count": 1,
                    "related_only_frame_count": 1,
                    "contradicted_frame_count": 0,
                    "insufficient_evidence_frame_count": 1,
                    "citation_risk_counts": {"safe": 2, "needs_more_evidence": 2},
                },
                "memory_context": {
                    "schema": "memory_context.v1",
                    "available": True,
                    "selected_memory_count": 2,
                    "artifact_ref_count": 4,
                    "filtered_memory_count": 1,
                    "context_char_count": 780,
                    "warnings": ["filtered_memory_items"],
                },
            }
        )

        self.assertEqual(result["code_version"], "v0.32")
        self.assertEqual(result["best_search_mode"], "tree")
        self.assertEqual(result["llm_baseline_status"], "partial")
        self.assertEqual(result["llm_timeout_count"], 2)
        self.assertEqual(result["llm_hard_timeout_count"], 1)
        self.assertTrue(result["llm_budget_exhausted"])
        self.assertEqual(result["low_quality_frame_count"], 2)
        self.assertEqual(result["top_frame_noise_reasons"], ["front_matter"])
        self.assertEqual(result["semantic_support_status_counts"], {"semantically_supported": 2, "related_only": 1})
        self.assertEqual(result["semantic_verified_frame_count"], 2)
        self.assertEqual(result["semantic_supported_frame_rate"], 0.5)
        self.assertEqual(result["partial_supported_frame_count"], 1)
        self.assertEqual(result["related_only_frame_count"], 1)
        self.assertEqual(result["contradicted_frame_count"], 0)
        self.assertEqual(result["insufficient_evidence_frame_count"], 1)
        self.assertEqual(result["citation_risk_counts"], {"safe": 2, "needs_more_evidence": 2})
        self.assertTrue(result["answer_plan_available"])
        self.assertEqual(result["answerability_counts"], {"answerable": 1, "conflicting": 1})
        self.assertEqual(result["strong_claim_count"], 3)
        self.assertEqual(result["qualified_claim_count"], 1)
        self.assertEqual(result["conflicting_claim_count"], 1)
        self.assertEqual(result["insufficient_claim_count"], 2)
        self.assertEqual(result["answer_plan_warning_counts"], {"answer_plan_conflicting_claims": 1})
        self.assertTrue(result["claim_alignment_available"])
        self.assertEqual(result["claim_alignment_group_count"], 3)
        self.assertEqual(result["claim_relation_count"], 3)
        self.assertEqual(result["claim_relation_type_counts"], {"supports": 1, "same_metric": 1, "contradicts": 1})
        self.assertEqual(result["claim_conflict_classification_counts"], {"supports": 2, "incomparable": 1, "contradicts": 1})
        self.assertEqual(result["claim_incomparable_pair_count"], 1)
        self.assertEqual(result["claim_avg_align_score"], 0.4)
        self.assertEqual(result["claim_max_align_score"], 0.8)
        self.assertEqual(result["method_family_group_count"], 1)
        self.assertEqual(result["conflicting_group_count"], 1)
        self.assertEqual(result["research_gap_count"], 1)
        self.assertEqual(result["claim_alignment_warnings"], ["claim_alignment_insufficient_evidence", "claim_alignment_conflicts"])
        self.assertTrue(result["compiled_context_available"])
        self.assertEqual(result["compiled_context_schema"], "memory_context.v1")
        self.assertEqual(result["selected_memory_count"], 2)
        self.assertEqual(result["artifact_ref_count"], 4)
        self.assertEqual(result["filtered_memory_count"], 1)
        self.assertEqual(result["context_char_count"], 780)
        self.assertEqual(result["memory_context_warnings"], ["filtered_memory_items"])
        self.assertEqual(result["warning_count"], 1)

    def test_eval_summary_names_are_public_and_consistent(self) -> None:
        self.assertTrue(callable(review_eval_summary))
        self.assertTrue(callable(memory_eval_summary))
        self.assertTrue(callable(fact_eval_summary))


if __name__ == "__main__":
    unittest.main()
