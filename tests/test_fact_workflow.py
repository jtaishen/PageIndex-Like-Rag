from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kb_agent.fact_workflow import (
    extract_fact_batch_workflow,
    finalize_fact_extraction_workflow,
    prepare_fact_extraction_workflow,
)
from kb_agent.facts import get_claims
from kb_agent.ingest import sync_directory
from kb_agent.search import search_documents


class FactWorkflowTest(unittest.TestCase):
    def test_staged_fact_batch_calls_llm_once_then_persists_canonical_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"KB_LLM_FACT_BATCH_SIZE": "50", "KB_LLM_FACT_MAX_NODES": "50"},
            clear=False,
        ):
            db_path, doc_id = _sync_fact_sample(Path(tmp))
            prepared = prepare_fact_extraction_workflow(db_path, doc_id, force=True)
            self.assertEqual(prepared["status"], "prepared")
            self.assertEqual(prepared["plan"]["batch_count"], 1)
            batch_id = prepared["plan"]["batches"][0]["batch_id"]
            calls = 0

            def generate(_system: str, _user: str) -> dict:
                nonlocal calls
                calls += 1
                return {
                    "claims": [
                        {
                            "type": "method",
                            "text": "提出动态角色发现机制以改进任务规划。",
                            "evidence": ["N1"],
                            "confidence": 0.84,
                        }
                    ],
                    "entities": [
                        {
                            "type": "method",
                            "name": "动态角色发现机制",
                            "evidence": ["N1"],
                            "confidence": 0.8,
                        }
                    ],
                    "relations": [],
                    "warnings": [],
                }

            batch = extract_fact_batch_workflow(
                db_path,
                prepared["task_id"],
                batch_id,
                json_generator=generate,
            )
            self.assertEqual(calls, 1)
            self.assertEqual(batch["status"], "completed")
            self.assertEqual(batch["workflow"]["status"], "ready_to_finalize")

            finalized = finalize_fact_extraction_workflow(db_path, prepared["task_id"])
            self.assertEqual(finalized["status"], "extracted")
            self.assertEqual(finalized["fact_report"]["source"], "llm")
            self.assertEqual(finalized["fact_report"]["llm_mode"], "staged_batch_json")
            self.assertEqual(finalized["workflow"]["status"], "completed")
            claims = get_claims(db_path, doc_id)
            self.assertTrue(any(item["type"] == "method" for item in claims["claims"]))

    def test_finalize_refuses_incomplete_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"KB_LLM_FACT_BATCH_SIZE": "50", "KB_LLM_FACT_MAX_NODES": "50"},
            clear=False,
        ):
            db_path, doc_id = _sync_fact_sample(Path(tmp))
            prepared = prepare_fact_extraction_workflow(db_path, doc_id, force=True)
            result = finalize_fact_extraction_workflow(db_path, prepared["task_id"])

            self.assertEqual(result["status"], "incomplete")
            with self.assertRaises(FileNotFoundError):
                get_claims(db_path, doc_id)


def _sync_fact_sample(root: Path) -> tuple[Path, str]:
    papers = root / "papers"
    papers.mkdir()
    db_path = root / "kb.sqlite"
    (papers / "facts.md").write_text(
        "# 摘要\n本文研究多智能体动态任务规划。\n\n"
        "# 主要贡献\n本文提出动态角色发现机制与任务重分配算法。\n\n"
        "# 方法\n该方法使用聚类模型识别角色，并据此完成任务分配。\n\n"
        "# 实验\n实验结果表明任务完成率得到提升。\n\n"
        "# 局限\n真实场景验证不足。\n",
        encoding="utf-8",
    )
    report = sync_directory(papers, db_path)
    if report["failed"]:
        raise AssertionError(report["errors"])
    doc_id = str(search_documents(db_path, "动态角色", top_k=1)[0]["doc_id"])
    return db_path, doc_id


if __name__ == "__main__":
    unittest.main()
