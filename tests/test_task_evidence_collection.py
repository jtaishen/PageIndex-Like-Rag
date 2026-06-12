from __future__ import annotations

from pathlib import Path
from unittest import mock
import unittest

from kb_agent.task_evidence_collection import (
    claim_frame_evidence_for_dimension,
    claim_frame_evidence_for_section,
    collect_dimension_evidence,
    collect_section_evidence,
    search_doc_evidence,
)


class TaskEvidenceCollectionTest(unittest.TestCase):
    def test_claim_frame_evidence_filters_by_dimension_and_section_type(self) -> None:
        context = _context(
            frames=[
                _frame("method-1", "method", "task planning method improves scheduling", unit_ids=["u1"]),
                _frame("claim-1", "claim", "task planning claim explains constraints", unit_ids=["u2"]),
                _frame("result-1", "result", "evaluation result improves success rate", unit_ids=["u3"]),
                _frame("limitation-1", "limitation", "task planning has limitation", unit_ids=["u4"]),
            ]
        )

        dimension_evidence = claim_frame_evidence_for_dimension(context, "method_paradigm", "task planning method", limit=10)
        section_evidence = claim_frame_evidence_for_section(context, "evaluation_evidence", "evaluation result", limit=10)

        self.assertEqual([item["claim_frame_id"] for item in dimension_evidence], ["method-1", "claim-1"])
        self.assertEqual([item["claim_frame_id"] for item in section_evidence], ["result-1"])

    def test_claim_frame_evidence_filters_contradicted_unsupported_and_low_quality_frames(self) -> None:
        context = _context(
            frames=[
                _frame("bad-support", "claim", "task planning claim", support_status="unsupported"),
                _frame("bad-quality", "claim", "task planning claim", quality_score=0.2),
                _frame("bad-semantic", "claim", "task planning claim", semantic_status="contradicted"),
            ]
        )

        evidence = claim_frame_evidence_for_dimension(context, "problem_setting", "task planning claim", limit=10)

        self.assertEqual(evidence, [])

    def test_claim_frame_evidence_preserves_semantic_and_citation_fields(self) -> None:
        context = _context(
            frames=[
                _frame(
                    "method-1",
                    "method",
                    "task planning method improves scheduling",
                    citation_risk="needs_qualification",
                    unit_ids=["u1"],
                )
            ]
        )

        evidence = claim_frame_evidence_for_dimension(context, "method_paradigm", "task planning method", limit=1)

        self.assertEqual(len(evidence), 1)
        item = evidence[0]
        self.assertEqual(item["doc_id"], "doc-1")
        self.assertEqual(item["node_id"], "n1")
        self.assertEqual(item["claim_frame_id"], "method-1")
        self.assertEqual(item["evidence_unit_ids"], ["u1"])
        self.assertEqual(item["semantic_support_status"], "semantically_supported")
        self.assertEqual(item["citation_risk"], "needs_qualification")
        self.assertEqual(item["source"], "claim_frame")

    def test_collect_dimension_evidence_merges_search_and_uses_innovation_fallback(self) -> None:
        context_with_frame = _context(
            frames=[_frame("method-1", "method", "task planning method improves scheduling", unit_ids=["u1"])]
        )
        context_with_fallback = _context(
            doc_id="doc-2",
            frames=[],
            units=[],
            innovation={
                "items": [
                    {
                        "type": "limitation",
                        "evidence": [
                            {
                                "node_id": "fallback-1",
                                "node_path": "局限",
                                "summary": "方法仍存在动态约束处理不足。",
                                "excerpt": "方法仍存在动态约束处理不足。",
                            }
                        ],
                    }
                ]
            },
        )
        dimensions = [
            {"id": "method_paradigm", "search_terms": ["method"]},
            {"id": "limitations", "search_terms": ["limitation"]},
        ]

        def fake_search(_db_path: Path, doc_id: str, _query: str, top_k: int, search_mode: str = "hybrid") -> list[dict]:
            del top_k, search_mode
            if doc_id == "doc-1":
                return [
                    _search_item("doc-1", "n1", "searched replacement", tree_score=0.9),
                    _search_item("doc-1", "search-2", "searched extra", tree_score=0.7),
                ]
            return []

        evidence = collect_dimension_evidence(
            Path("db.sqlite"),
            "task planning",
            [context_with_frame, context_with_fallback],
            dimensions,
            "hybrid",
            search_evidence_fn=fake_search,
        )

        method_doc = evidence["method_paradigm"]["doc-1"]
        self.assertEqual([item["node_id"] for item in method_doc], ["n1", "search-2"])
        self.assertEqual(method_doc[0]["summary"], "searched replacement")
        fallback_doc = evidence["limitations"]["doc-2"]
        self.assertEqual([item["node_id"] for item in fallback_doc], ["fallback-1"])
        self.assertEqual(fallback_doc[0]["doc_id"], "doc-2")

    def test_collect_section_evidence_uses_injected_search_and_reports_quality(self) -> None:
        context = _context(frames=[])
        sections = [{"section_id": "background_problem", "search_terms": ["problem"]}]
        calls = []

        def fake_search(_db_path: Path, doc_id: str, query: str, top_k: int, search_mode: str = "hybrid") -> list[dict]:
            calls.append((doc_id, query, top_k, search_mode))
            return [
                _search_item(doc_id, "dup", "duplicate", tree_score=0.8),
                _search_item(doc_id, "dup", "duplicate", tree_score=0.7),
            ]

        section_evidence, quality = collect_section_evidence(
            Path("db.sqlite"),
            "task planning",
            [context],
            sections,
            "tree",
            search_evidence_fn=fake_search,
        )

        self.assertEqual(calls, [("doc-1", "task planning problem", 3, "tree")])
        self.assertEqual([item["node_id"] for item in section_evidence["background_problem"]], ["dup"])
        self.assertEqual(quality["schema"], "section_evidence_quality.v1")
        self.assertEqual(quality["duplicate_evidence_removed"], 1)
        self.assertEqual(quality["post_dedupe_duplicate_count"], 0)

    def test_search_doc_evidence_tree_mode_returns_deduped_tree_trace_evidence(self) -> None:
        duplicate = _search_item("doc-1", "n1", "tree evidence", tree_score=0.9)
        with mock.patch(
            "kb_agent.tree_search.tree_search",
            return_value={"evidence": [duplicate, dict(duplicate)]},
        ) as tree_search:
            evidence = search_doc_evidence(Path("db.sqlite"), "doc-1", "task planning", top_k=2, search_mode="tree")

        tree_search.assert_called_once_with(
            Path("db.sqlite"),
            "doc-1",
            "task planning",
            budget=2,
            use_llm=False,
            search_mode="hybrid",
        )
        self.assertEqual([item["node_id"] for item in evidence], ["n1"])


def _context(
    *,
    doc_id: str = "doc-1",
    frames: list[dict],
    units: list[dict] | None = None,
    innovation: dict | None = None,
) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"Paper {doc_id}",
        "path": f"/tmp/{doc_id}.pdf",
        "claim_frames": {"frames": frames},
        "evidence_units": {"units": units if units is not None else _units()},
        "innovation": innovation or {"items": []},
    }


def _units() -> list[dict]:
    return [
        {
            "unit_id": "u1",
            "node_id": "n1",
            "node_path": "1",
            "page_range": [1],
            "summary": "unit one",
            "unit_type": "method",
        },
        {
            "unit_id": "u2",
            "node_id": "n2",
            "node_path": "2",
            "page_range": [2],
            "summary": "unit two",
            "unit_type": "claim",
        },
        {
            "unit_id": "u3",
            "node_id": "n3",
            "node_path": "3",
            "page_range": [3],
            "summary": "unit three",
            "unit_type": "result",
        },
        {
            "unit_id": "u4",
            "node_id": "n4",
            "node_path": "4",
            "page_range": [4],
            "summary": "unit four",
            "unit_type": "limitation",
        },
    ]


def _frame(
    frame_id: str,
    claim_type: str,
    short_claim: str,
    *,
    support_status: str = "structurally_supported",
    semantic_status: str = "semantically_supported",
    citation_risk: str = "safe",
    quality_score: float = 0.8,
    confidence: float = 0.7,
    unit_ids: list[str] | None = None,
) -> dict:
    return {
        "frame_id": frame_id,
        "claim_type": claim_type,
        "short_claim": short_claim,
        "support_status": support_status,
        "semantic_support_status": semantic_status,
        "semantic_support_score": 0.8,
        "semantic_support_reason": "matched",
        "citation_risk": citation_risk,
        "quality_score": quality_score,
        "confidence": confidence,
        "evidence_unit_ids": unit_ids or ["u1"],
        "primary_evidence_unit_ids": unit_ids or ["u1"],
        "weak_evidence_unit_ids": [],
        "contradictory_evidence_unit_ids": [],
    }


def _search_item(doc_id: str, node_id: str, summary: str, *, tree_score: float) -> dict:
    return {
        "doc_id": doc_id,
        "node_id": node_id,
        "node_path": "1",
        "title": f"Paper {doc_id}",
        "summary": summary,
        "excerpt": summary,
        "tree_score": tree_score,
    }


if __name__ == "__main__":
    unittest.main()
