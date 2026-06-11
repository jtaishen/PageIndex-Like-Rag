from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kb_agent.claim_frames import (
    _claim_frames_payload,
    _enhance_frames_with_llm,
    _frame_record,
    _verify_claim_frames_payload,
    claim_frame_summary_for_doc,
    extract_claim_frames,
    extract_evidence_units,
    get_claim_frames,
    get_evidence_units,
    search_claim_frames,
    verify_claim_frames,
)
from kb_agent.claim_frame_evidence import (
    evidence_unit_ids_for_claim,
    unit_by_id,
    unit_by_node_id,
    unit_by_source_id,
)
from kb_agent.cli import main as cli_main
from kb_agent.facts import extract_facts
from kb_agent.ingest import sync_directory
from kb_agent.llm import LLMError
from kb_agent.search import build_search_report, search_documents
from kb_agent.tasks import compare_papers, generate_review_plan


class ClaimFrameTest(unittest.TestCase):
    def test_extract_facts_builds_evidence_units_claim_frames_and_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_claim_sample(Path(tmp), name="robot")

            fact = extract_facts(db_path, doc_id, force=True, use_llm=False)
            report = fact["fact_report"]
            self.assertGreater(report["evidence_unit_count"], 0)
            self.assertIn("node", report["source_kind_counts"])
            self.assertGreater(report["claim_frame_count"], 0)
            self.assertGreaterEqual(report["verified_frame_rate"], 0.0)
            self.assertIn("trace_status_counts", report)
            self.assertIn("support_status_counts", report)

            units = get_evidence_units(db_path, doc_id)
            self.assertEqual(units["schema"], "evidence_units.v1")
            self.assertTrue(all(len(item["text_excerpt"]) <= 363 for item in units["units"]))
            self.assertTrue(all("unit_id" in item and "node_id" in item for item in units["units"]))

            frames = get_claim_frames(db_path, doc_id)
            self.assertEqual(frames["schema"], "claim_frames.v1")
            self.assertTrue(any(frame["evidence_unit_ids"] for frame in frames["frames"]))
            self.assertTrue(all(len(frame["short_claim"]) <= 243 for frame in frames["frames"]))
            self.assertIn("quality_summary", frames)
            self.assertTrue(all("quality_score" in frame and "frame_quality" in frame for frame in frames["frames"]))

            verifier = verify_claim_frames(db_path, doc_ids=[doc_id])
            self.assertEqual(verifier["schema"], "claim_frame_verifier_result.v1")
            self.assertGreater(verifier["frame_count"], 0)
            self.assertGreater(verifier["verified_frame_rate"], 0.0)
            self.assertIn("low_quality_frame_count", verifier)
            self.assertIn("top_frame_noise_reasons", verifier)
            self.assertEqual(frames["trace_status_counts"], verifier["documents"][0]["trace_status_counts"])
            self.assertEqual(frames["support_status_counts"], verifier["documents"][0]["support_status_counts"])

    def test_evidence_units_include_secondary_artifacts_without_hard_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)

            def fake_artifact(_db_path: Path, _doc_id: str, name: str, default: object) -> object:
                if name == "node_index.jsonl":
                    return [
                        {
                            "node_id": "node-1",
                            "kind": "paragraph",
                            "text": "本文提出任务规划方法并验证任务成功率。",
                            "node_path": "摘要",
                            "page_start": 1,
                            "page_end": 1,
                        }
                    ]
                if name == "table_summaries.json":
                    return {
                        "table_summaries": [
                            {
                                "table_id": "table-1",
                                "caption": "表 1 任务成功率结果",
                                "headers": ["方法", "任务成功率"],
                                "results": ["本文方法更高"],
                                "page": 3,
                            }
                        ]
                    }
                if name == "figures.json":
                    return {"figures": [{"id": "fig-1", "caption": "图 1 方法框架", "text": "任务分解流程", "page": 2}]}
                if name == "reference_sections.json":
                    return {"reference_sections": [{"section_id": "refs", "references_count": 2, "page_start": 8, "page_end": 8}]}
                if name == "citation_map.json":
                    return {
                        "references": [{"ref_id": "R1", "title": "任务规划研究", "raw": "R1. 任务规划研究."}],
                        "in_text_citations": [{"ref_id": "R1", "node_id": "node-1", "context": "相关研究 [R1]"}],
                    }
                return default

            with mock.patch(
                "kb_agent.claim_frames.list_artifacts",
                return_value={"artifact_dir": str(artifact_dir), "version_id": "v1"},
            ), mock.patch("kb_agent.claim_frames._artifact_content", side_effect=fake_artifact):
                result = extract_evidence_units(Path("unused.sqlite"), "doc-1", force=True)

            payload = result["evidence_units"]
            kinds = payload["source_kind_counts"]
            self.assertGreaterEqual(payload["count"], 5)
            self.assertEqual(kinds["node"], 1)
            self.assertEqual(kinds["table"], 1)
            self.assertEqual(kinds["figure"], 1)
            self.assertGreaterEqual(kinds["reference"], 1)
            self.assertGreaterEqual(kinds["citation"], 1)
            artifact_units = [unit for unit in payload["units"] if unit["source_kind"] in {"table", "figure", "reference"}]
            self.assertTrue(any("source_without_node" in unit["warnings"] for unit in artifact_units))
            self.assertTrue(all(len(unit["summary"]) <= 183 for unit in payload["units"]))

    def test_evidence_ref_binding_variants_report_unresolved_refs(self) -> None:
        units = [
            {"unit_id": "eu-node", "node_id": "node-1", "source_id": "node-1"},
            {"unit_id": "eu-source", "node_id": "node-2", "source_id": "source-1"},
            {"unit_id": "eu-ref", "node_id": "", "source_id": "R1"},
        ]
        claim = {
            "node_id": "node-1",
            "source_node_id": "node-2",
            "evidence_node_id": "missing-node",
            "unit_id": "eu-node",
            "source_id": "source-1",
            "ref_id": "R1",
            "evidence": [
                {"unit_id": "missing-unit"},
                {"evidence_id": "source-1"},
                {"source_id": "R1"},
                {"ref_id": "R1"},
                "loose citation text",
            ],
        }

        ids, warnings = evidence_unit_ids_for_claim(
            claim,
            unit_by_node_id(units),
            unit_by_id(units),
            unit_by_source_id(units),
        )

        self.assertEqual(ids, ["eu-node", "eu-source", "eu-ref"])
        self.assertEqual(warnings, ["unresolved_evidence_ref"])

    def test_claim_frame_verifier_separates_trace_and_support_status(self) -> None:
        claim_frames = {
            "schema": "claim_frames.v1",
            "version_id": "v1",
            "frames": [
                {"frame_id": "verified", "claim_type": "method", "short_claim": "提出任务规划方法。", "evidence_unit_ids": ["eu-node"], "confidence": 0.8},
                {"frame_id": "missing-node", "claim_type": "method", "short_claim": "方法依赖额外节点。", "evidence_unit_ids": ["eu-missing-node"], "confidence": 0.8},
                {"frame_id": "missing-unit", "claim_type": "result", "short_claim": "结果有引用但 unit 缺失。", "evidence_unit_ids": ["eu-absent"], "confidence": 0.8},
                {"frame_id": "unsupported", "claim_type": "result", "short_claim": "完全没有证据。", "evidence_unit_ids": [], "confidence": 0.8},
            ],
        }
        evidence_units = {
            "schema": "evidence_units.v1",
            "version_id": "v1",
            "units": [
                {"unit_id": "eu-node", "node_id": "node-1", "source_kind": "node", "source_id": "node-1", "text_excerpt": "任务规划方法。"},
                {
                    "unit_id": "eu-missing-node",
                    "node_id": "node-missing",
                    "source_kind": "node",
                    "source_id": "node-missing",
                    "text_excerpt": "额外节点。",
                },
            ],
        }

        def fake_artifact(_db_path: Path, _doc_id: str, name: str, default: object) -> object:
            if name == "node_index.jsonl":
                return [{"node_id": "node-1"}]
            if name == "citation_map.json":
                return {"references": [], "relations": []}
            return default

        with mock.patch("kb_agent.claim_frames._artifact_content", side_effect=fake_artifact):
            verifier = _verify_claim_frames_payload(Path("unused.sqlite"), "doc-1", claim_frames, evidence_units)

        by_id = {item["frame_id"]: item for item in verifier["items"]}
        self.assertEqual(by_id["verified"]["trace_status"], "verified")
        self.assertEqual(by_id["verified"]["support_status"], "structurally_supported")
        self.assertEqual(by_id["missing-node"]["trace_status"], "partial")
        self.assertEqual(by_id["missing-node"]["support_status"], "unchecked")
        self.assertEqual(by_id["missing-unit"]["trace_status"], "partial")
        self.assertEqual(by_id["missing-unit"]["support_status"], "unchecked")
        self.assertEqual(by_id["unsupported"]["trace_status"], "missing")
        self.assertEqual(by_id["unsupported"]["support_status"], "unsupported")
        self.assertEqual(verifier["verified_frame_count"], 1)
        self.assertEqual(verifier["unsupported_frame_count"], 1)
        self.assertEqual(verifier["trace_status_counts"], {"verified": 1, "partial": 2, "missing": 1})
        self.assertEqual(verifier["support_status_counts"], {"structurally_supported": 1, "unchecked": 2, "unsupported": 1})
        self.assertEqual(verifier["missing_evidence_unit_count"], 1)
        self.assertEqual(verifier["missing_node_count"], 1)

    def test_llm_enhancement_records_truncation_metadata(self) -> None:
        frames = [
            {
                "frame_id": f"frame-{index}",
                "claim_type": "method",
                "short_claim": f"本文提出第 {index} 个任务规划方法。",
                "evidence_unit_ids": ["eu-1"],
                "trace_status": "verified",
                "support_status": "structurally_supported",
                "warnings": [],
                "quality_score": 0.8,
            }
            for index in range(25)
        ]
        units = [{"unit_id": f"eu-{index}", "unit_type": "paragraph", "summary": "任务规划方法", "keywords": ["任务规划"]} for index in range(61)]

        with mock.patch("kb_agent.claim_frames.generate_json_object", return_value={"frames": [], "_llm_metadata": {"retry_count": 0}}):
            enhanced, metadata = _enhance_frames_with_llm({"title": "测试论文"}, frames, units)

        payload = _claim_frames_payload(
            "doc-1",
            "v1",
            enhanced,
            evidence_unit_count=len(units),
            warnings=metadata["enhancement_warnings"],
            llm_used=True,
            llm_error="",
            llm_metadata=metadata,
        )
        self.assertTrue(payload["llm_enhancement"]["used"])
        self.assertTrue(payload["llm_enhancement"]["truncated"])
        self.assertEqual(payload["llm_enhancement"]["enhanced_frame_limit"], 24)
        self.assertEqual(payload["llm_enhancement"]["context_unit_limit"], 60)
        self.assertIn("llm_frame_enhancement_truncated", payload["warnings"])
        self.assertIn("llm_unit_context_truncated", payload["warnings"])

    def test_claim_frame_summary_exposes_demo_metrics(self) -> None:
        frames = {
            "schema": "claim_frames.v1",
            "frames": [
                {
                    "frame_id": "frame-1",
                    "claim_type": "method",
                    "short_claim": "本文提出任务规划方法。",
                    "trace_status": "verified",
                    "support_status": "structurally_supported",
                    "support_reason": "evidence_units_verified",
                    "source": "claim",
                    "quality_score": 0.92,
                    "frame_quality": "high",
                    "noise_reasons": [],
                    "evidence_unit_ids": ["eu-1"],
                    "confidence": 0.85,
                    "warnings": [],
                }
            ],
            "claim_type_counts": {"method": 1},
            "warnings": [],
        }
        verifier = {
            "verified_frame_rate": 1.0,
            "unsupported_frame_count": 0,
            "trace_status_counts": {"verified": 1},
            "support_status_counts": {"structurally_supported": 1},
            "missing_evidence_unit_count": 0,
            "missing_node_count": 0,
            "missing_source_count": 0,
            "citation_gap_count": 0,
            "low_quality_frame_count": 0,
            "noisy_frame_count": 0,
            "ignored_noise_frame_count": 0,
            "top_frame_noise_reasons": [],
            "warnings": [],
        }
        evidence_units = {
            "schema": "evidence_units.v1",
            "count": 1,
            "source_kind_counts": {"node": 1},
            "units": [{"unit_id": "eu-1", "source_kind": "node"}],
        }

        with mock.patch("kb_agent.claim_frames.get_claim_frames", return_value=frames), mock.patch(
            "kb_agent.claim_frames._artifact_content",
            return_value=verifier,
        ), mock.patch("kb_agent.claim_frames._ensure_evidence_units", return_value=evidence_units):
            summary = claim_frame_summary_for_doc(Path("unused.sqlite"), "doc-1")

        self.assertTrue(summary["available"])
        self.assertEqual(summary["evidence_unit_count"], 1)
        self.assertEqual(summary["source_kind_counts"], {"node": 1})
        self.assertEqual(summary["trace_status_counts"], {"verified": 1})
        self.assertEqual(summary["support_status_counts"], {"structurally_supported": 1})
        self.assertEqual(summary["top_frames"][0]["trace_status"], "verified")

    def test_claim_frame_llm_success_and_failure_keep_rule_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_claim_sample(Path(tmp), name="llm")
            extract_facts(db_path, doc_id, force=True, use_llm=False)
            first_frame = get_claim_frames(db_path, doc_id)["frames"][0]

            with mock.patch(
                "kb_agent.claim_frames.generate_json_object",
                return_value={
                    "frames": [
                        {
                            "frame_id": first_frame["frame_id"],
                            "method": "LLM 补全的方法摘要",
                            "metric_or_signal": "任务成功率",
                            "confidence": 0.91,
                            "warnings": [],
                        }
                    ],
                    "_llm_metadata": {"retry_count": 0},
                },
            ):
                result = extract_claim_frames(db_path, doc_id, force=True, use_llm=True)
            enhanced = result["claim_frames"]["frames"][0]
            self.assertEqual(enhanced["method"], "LLM 补全的方法摘要")
            self.assertEqual(enhanced["metric_or_signal"], "任务成功率")
            self.assertTrue(result["claim_frames"]["llm_used"])

            with mock.patch(
                "kb_agent.claim_frames.generate_json_object",
                side_effect=LLMError("timeout", error_type="request_timeout"),
            ):
                fallback = extract_claim_frames(db_path, doc_id, force=True, use_llm=True, require_llm=False)
            self.assertGreater(fallback["claim_frames"]["count"], 0)
            self.assertIn("llm_unavailable:request_timeout", fallback["claim_frames"]["warnings"])

    def test_search_report_compare_and_review_include_claim_frame_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, doc_a = _sync_claim_sample(root, name="robot")
            _, doc_b = _sync_claim_sample(root, name="agent", append=True)
            extract_facts(db_path, doc_a, force=True, use_llm=False)
            extract_facts(db_path, doc_b, force=True, use_llm=False)

            report = build_search_report(db_path, "任务规划方法", search_mode="hybrid", top_k=3)
            frame_matches = report["fact_matches"]["claim_frame_matches"]
            self.assertEqual(frame_matches["schema"], "claim_frame_search.v1")
            self.assertGreaterEqual(frame_matches["count"], 1)
            self.assertIn("evidence_unit_ids", frame_matches["items"][0])
            self.assertIn("quality_score", frame_matches["items"][0])
            self.assertIn("selection_reasons", frame_matches["items"][0])

            compare = compare_papers(db_path, "任务规划方法比较", top_k_docs=2, use_llm=False)
            evidence = [
                item
                for dimension in compare["comparison_matrix"]["dimensions"]
                for cell in dimension["cells"]
                for item in cell.get("evidence") or []
            ]
            self.assertTrue(any(item.get("claim_frame_id") for item in evidence))

            review = generate_review_plan(db_path, "任务规划方法", top_k_docs=2, use_llm=False)
            section_evidence = [
                item
                for artifact in review["section_evidence"].values()
                for item in artifact.get("evidence") or []
            ]
            self.assertTrue(any(item.get("claim_frame_id") for item in section_evidence))

    def test_claim_frame_quality_filters_front_matter_and_ranks_supported_frames(self) -> None:
        supported = _frame_record(
            "doc-1",
            "v1",
            "method",
            "本文提出任务规划方法，结合技能库完成复杂任务分解。",
            ["eu-1"],
            source="claim",
            source_claim_ids=["claim-1"],
            confidence=0.82,
            index=0,
        )
        noisy = _frame_record(
            "doc-1",
            "v1",
            "result",
            "网络首发 ISSN 1000-0000 引用格式 图 3 图书馆接待服务场景 图 3 图书馆接待服务场景",
            [],
            source="table_summary",
            source_claim_ids=["table-1"],
            confidence=0.6,
            index=1,
        )

        self.assertTrue(supported)
        self.assertGreaterEqual(supported["quality_score"], 0.75)
        self.assertEqual(noisy, {})

        weak = dict(supported)
        weak.update(
            {
                "frame_id": "weak",
                "short_claim": "任务规划方法 图 3 图书馆接待服务场景 图 3 图书馆接待服务场景",
                "evidence_unit_ids": [],
                "support_status": "unsupported",
                "quality_score": 0.2,
                "frame_quality": "low",
                "noise_reasons": ["repeated_caption"],
            }
        )
        with mock.patch("kb_agent.claim_frames._ready_doc_ids", return_value=["doc-1"]), mock.patch(
            "kb_agent.claim_frames.get_claim_frames",
            return_value={"schema": "claim_frames.v1", "frames": [weak, supported]},
        ), mock.patch(
            "kb_agent.claim_frames._artifact_content",
            return_value={
                "items": [
                    {"frame_id": "weak", "support_status": "unsupported", "warnings": ["unsupported_frame"]},
                    {"frame_id": supported["frame_id"], "support_status": "supported", "warnings": []},
                ]
            },
        ):
            result = search_claim_frames(Path("unused.sqlite"), "任务规划方法", top_k=3)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["frame_id"], supported["frame_id"])
        self.assertEqual(result["items"][0]["support_status"], "structurally_supported")

    def test_cli_claim_frame_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_claim_sample(Path(tmp), name="cli")
            extract_facts(db_path, doc_id, force=True, use_llm=False)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "evidence-units", doc_id])
            self.assertIn("evidence_units.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "extract-claim-frames", doc_id, "--force", "--no-llm"])
            self.assertIn("claim_frames.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "verify-claim-frames", "--doc-id", doc_id])
            self.assertIn("claim_frame_verifier_result.v1", stdout.getvalue())


def _sync_claim_sample(root: Path, *, name: str, append: bool = False) -> tuple[Path, str]:
    papers = root / "papers"
    papers.mkdir(exist_ok=True)
    db_path = root / "kb.sqlite"
    suffix = "服务机器人" if name != "agent" else "多智能体"
    (papers / f"{name}.txt").write_text(
        f"摘要：本文研究{suffix}任务规划方法，解决任务分解和工具调用问题。\n"
        "关键词：任务规划；方法；实验\n\n"
        "1 方法设计\n"
        "本文提出结合大语言模型和技能库的任务规划框架。\n\n"
        "2 实验结果\n"
        "实验结果表明，该方法提升任务成功率和响应时间表现。\n\n"
        "3 局限性\n"
        "本文仍存在真实环境验证不足的局限。\n\n"
        "参考文献\n"
        "[1] 张三. 任务规划研究. 2025.\n",
        encoding="utf-8",
    )
    sync_directory(papers, db_path, force=True)
    query = "多智能体" if name == "agent" else "服务机器人"
    doc_id = str(search_documents(db_path, query, top_k=1)[0]["doc_id"])
    if append:
        return db_path, doc_id
    return db_path, doc_id
