from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kb_agent.artifacts import get_citation_map, get_innovations
from kb_agent.ingest import sync_directory
from kb_agent.insight_workflow import (
    extract_insight_batch_workflow,
    finalize_insight_extraction_workflow,
    prepare_insight_extraction_workflow,
)
from kb_agent.search import search_documents


class InsightWorkflowTest(unittest.TestCase):
    def test_staged_insights_persist_batches_and_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            prepared = prepare_insight_extraction_workflow(db_path, doc_id, force=True)

            self.assertEqual(prepared["status"], "prepared")
            calls = 0

            def generate(_system: str, _user: str) -> dict:
                nonlocal calls
                calls += 1
                return {
                    "items": [
                        {
                            "title": "动态任务重分配",
                            "type": "method",
                            "claim": f"提出动态任务重分配机制 {calls}。",
                            "problem": "执行环境持续变化。",
                            "approach": "结合反馈更新任务分配。",
                            "evidence": ["N1"],
                            "confidence": 0.82,
                        }
                    ],
                    "limitations": ["真实环境验证不足。"],
                    "open_questions": [],
                    "warnings": [],
                }

            last = {}
            for batch in prepared["plan"]["batches"]:
                last = extract_insight_batch_workflow(
                    db_path,
                    prepared["task_id"],
                    batch["batch_id"],
                    json_generator=generate,
                )
                self.assertEqual(last["status"], "completed")

            self.assertEqual(calls, prepared["plan"]["batch_count"])
            self.assertEqual(last["workflow"]["status"], "ready_to_finalize")
            finalized = finalize_insight_extraction_workflow(db_path, prepared["task_id"])

            self.assertEqual(finalized["status"], "extracted")
            self.assertEqual(finalized["innovation"]["llm_mode"], "staged_batch_json")
            self.assertEqual(finalized["workflow"]["status"], "completed")
            self.assertEqual(get_innovations(db_path, doc_id)["schema"], "innovation.v1")
            self.assertEqual(get_citation_map(db_path, doc_id)["schema"], "citation_map.v1")

    def test_finalize_refuses_incomplete_insight_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            prepared = prepare_insight_extraction_workflow(db_path, doc_id, force=True)

            result = finalize_insight_extraction_workflow(db_path, prepared["task_id"])

            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["pending_steps"])


def _sync_insight_sample(root: Path) -> tuple[Path, str]:
    papers = root / "papers"
    papers.mkdir()
    db_path = root / "kb.sqlite"
    (papers / "insight.md").write_text(
        "# 摘要\n本文研究动态多智能体任务规划。\n\n"
        "# 主要贡献\n本文提出动态角色发现与任务重分配机制。\n\n"
        "# 方法\n系统依据执行反馈更新任务分配。\n\n"
        "# 实验\n实验表明任务完成率得到提升。\n\n"
        "# 局限\n真实环境验证仍然不足。\n",
        encoding="utf-8",
    )
    report = sync_directory(papers, db_path, doc_card_use_llm=False)
    if report["failed"]:
        raise AssertionError(report["errors"])
    doc_id = str(search_documents(db_path, "动态任务规划", top_k=1)[0]["doc_id"])
    return db_path, doc_id


if __name__ == "__main__":
    unittest.main()
