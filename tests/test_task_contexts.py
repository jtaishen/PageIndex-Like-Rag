from __future__ import annotations

from pathlib import Path
from unittest import mock
import unittest

from kb_agent.task_contexts import prepare_paper_contexts, select_papers


class TaskContextsTest(unittest.TestCase):
    def test_select_papers_keeps_explicit_doc_ids_deduped_and_ordered(self) -> None:
        selected = select_papers(Path("kb.sqlite"), "任务规划", ["doc-1", "doc-2", "doc-1"], 5, "hybrid")

        self.assertEqual(
            selected,
            [
                {"doc_id": "doc-1", "score": None, "node_matches": None},
                {"doc_id": "doc-2", "score": None, "node_matches": None},
            ],
        )

    def test_select_papers_routes_tree_mode_to_hybrid_search_documents(self) -> None:
        with mock.patch(
            "kb_agent.task_contexts.search_documents",
            return_value=[{"doc_id": "doc-1", "score": 0.9, "node_matches": ["n1"]}],
        ) as search_documents:
            selected = select_papers(Path("kb.sqlite"), "任务规划", None, 3, "tree")

        search_documents.assert_called_once_with(Path("kb.sqlite"), "任务规划", top_k=3, search_mode="hybrid")
        self.assertEqual(selected[0]["doc_id"], "doc-1")

    def test_prepare_paper_contexts_aggregates_artifacts_and_claim_frames(self) -> None:
        selected = [{"doc_id": "doc-1", "score": 0.8, "node_matches": ["n1"]}]

        with _patched_context_dependencies():
            contexts, warnings = prepare_paper_contexts(Path("kb.sqlite"), selected)

        self.assertEqual(warnings, [])
        self.assertEqual(len(contexts), 1)
        context = contexts[0]
        self.assertEqual(context["doc_id"], "doc-1")
        self.assertEqual(context["title"], "Paper 1")
        self.assertEqual(context["description"], "description")
        self.assertEqual(context["quality"]["schema"], "parse_quality.v1")
        self.assertEqual(context["innovation"]["schema"], "innovation.v1")
        self.assertEqual(context["citation_map"]["schema"], "citation_map.v1")
        self.assertEqual(context["facts"]["claim_count"], 2)
        self.assertEqual(context["claim_frames"]["summary"]["frame_count"], 3)
        self.assertEqual(context["evidence_units"]["schema"], "evidence_units.v1")
        self.assertEqual(context["route_score"], 0.8)
        self.assertEqual(context["node_matches"], ["n1"])

    def test_prepare_paper_contexts_reports_failed_docs_as_warnings(self) -> None:
        with mock.patch("kb_agent.task_contexts.get_doc_card", side_effect=FileNotFoundError("missing")):
            contexts, warnings = prepare_paper_contexts(Path("kb.sqlite"), [{"doc_id": "missing-doc"}])

        self.assertEqual(contexts, [])
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0].startswith("paper_prepare_failed:missing-doc:"))

    def test_prepare_paper_contexts_refreshes_missing_claim_frames_with_visible_warning(self) -> None:
        patches = _patched_context_dependencies(
            claim_frames_side_effect=FileNotFoundError("missing frames"),
            extracted_claim_frames={"schema": "claim_frames.v1", "frames": [{"frame_id": "f1"}]},
        )
        with patches:
            contexts, warnings = prepare_paper_contexts(Path("kb.sqlite"), [{"doc_id": "doc-1"}])

        self.assertEqual(contexts[0]["claim_frames"]["frames"], [{"frame_id": "f1"}])
        self.assertEqual(warnings, ["claim_frames_rule_refreshed:doc-1"])


def _patched_context_dependencies(
    *,
    claim_frames_side_effect: Exception | None = None,
    extracted_claim_frames: dict | None = None,
):
    return mock.patch.multiple(
        "kb_agent.task_contexts",
        get_doc_card=mock.Mock(
            return_value={
                "title": "Paper 1",
                "path": "/tmp/paper1.txt",
                "abstract": "abstract",
                "description": "description",
                "keywords": ["task"],
            }
        ),
        get_parse_quality=mock.Mock(return_value={"schema": "parse_quality.v1"}),
        get_innovations=mock.Mock(return_value={"schema": "innovation.v1", "items": []}),
        get_citation_map=mock.Mock(return_value={"schema": "citation_map.v1", "references": []}),
        fact_summary_for_doc=mock.Mock(return_value={"available": True, "claim_count": 2}),
        get_claim_frames=mock.Mock(
            side_effect=claim_frames_side_effect,
            return_value={"schema": "claim_frames.v1", "frames": [{"frame_id": "f1"}]},
        ),
        get_evidence_units=mock.Mock(return_value={"schema": "evidence_units.v1", "units": [{"unit_id": "u1"}]}),
        claim_frame_summary_for_doc=mock.Mock(return_value={"available": True, "frame_count": 3}),
        extract_claim_frames=mock.Mock(
            return_value={
                "claim_frames": extracted_claim_frames or {"schema": "claim_frames.v1", "frames": [{"frame_id": "f1"}]}
            }
        ),
        extract_doc_insights=mock.Mock(return_value={"innovation": {"schema": "innovation.v1", "items": []}}),
    )


if __name__ == "__main__":
    unittest.main()
