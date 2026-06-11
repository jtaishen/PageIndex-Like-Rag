from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kb_agent.claim_frames import (
    extract_claim_frames,
    get_claim_frames,
    get_evidence_units,
    verify_claim_frames,
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
            self.assertGreater(report["claim_frame_count"], 0)
            self.assertGreaterEqual(report["verified_frame_rate"], 0.0)

            units = get_evidence_units(db_path, doc_id)
            self.assertEqual(units["schema"], "evidence_units.v1")
            self.assertTrue(all(len(item["text_excerpt"]) <= 363 for item in units["units"]))
            self.assertTrue(all("unit_id" in item and "node_id" in item for item in units["units"]))

            frames = get_claim_frames(db_path, doc_id)
            self.assertEqual(frames["schema"], "claim_frames.v1")
            self.assertTrue(any(frame["evidence_unit_ids"] for frame in frames["frames"]))
            self.assertTrue(all(len(frame["short_claim"]) <= 243 for frame in frames["frames"]))

            verifier = verify_claim_frames(db_path, doc_ids=[doc_id])
            self.assertEqual(verifier["schema"], "claim_frame_verifier_result.v1")
            self.assertGreater(verifier["frame_count"], 0)
            self.assertGreater(verifier["verified_frame_rate"], 0.0)

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
