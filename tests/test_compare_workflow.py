from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from kb_agent.compare_workflow import (
    finalize_compare_workflow,
    generate_compare_dimension,
    prepare_compare_workflow,
)
from kb_agent.ingest import sync_directory
from kb_agent.search import search_documents


class CompareWorkflowTest(unittest.TestCase):
    def test_staged_compare_calls_llm_once_per_dimension_and_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            prepared = prepare_compare_workflow(
                db_path,
                "任务规划方法对比",
                doc_ids=doc_ids,
                top_k_docs=2,
                search_mode="fts",
            )
            self.assertEqual(prepared["status"], "prepared")
            calls: list[str] = []

            def generate(_system: str, user: str) -> dict:
                calls.append(user)
                dimension_id = re.search(r"dimension_id: ([^\n]+)", user).group(1)  # type: ignore[union-attr]
                return {
                    "id": dimension_id,
                    "synthesis": "基于现有证据完成该维度比较。",
                    "overlaps": [],
                    "differences": [],
                    "cells": [
                        {
                            "doc_id": doc_id,
                            "claim": "该论文在此维度具有可追踪证据。",
                            "evidence": [],
                            "confidence": 0.7,
                            "warnings": [],
                        }
                        for doc_id in doc_ids
                    ],
                    "warnings": [],
                }

            result = {}
            for dimension_id in prepared["dimension_ids"]:
                result = generate_compare_dimension(
                    db_path,
                    prepared["task_id"],
                    dimension_id,
                    json_generator=generate,
                )
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["llm_diagnostics"]["mode"], "staged_dimension_json")

            self.assertEqual(len(calls), len(prepared["dimension_ids"]))
            self.assertEqual(result["workflow"]["status"], "ready_to_finalize")
            finalized = finalize_compare_workflow(db_path, prepared["task_id"])
            self.assertNotEqual(finalized["status"], "incomplete")
            self.assertEqual(finalized["comparison_matrix"]["source"], "llm_staged")
            self.assertEqual(finalized["workflow"]["status"], "completed")


def _sync_compare_samples(root: Path) -> tuple[Path, list[str]]:
    papers = root / "papers"
    papers.mkdir()
    db_path = root / "kb.sqlite"
    (papers / "centralized.md").write_text(
        "# 摘要\n本文研究集中式机器人任务规划。\n\n"
        "# 方法\n中央规划器执行任务分解和工具选择。\n\n"
        "# 实验\n任务成功率高于规则基线。\n\n# 局限\n中央节点存在性能瓶颈。\n",
        encoding="utf-8",
    )
    (papers / "distributed.md").write_text(
        "# 摘要\n本文研究分布式多智能体任务规划。\n\n"
        "# 方法\n智能体通过局部决策和协同通信分配任务。\n\n"
        "# 实验\n负载均衡优于集中式基线。\n\n# 局限\n通信开销仍然较高。\n",
        encoding="utf-8",
    )
    report = sync_directory(papers, db_path)
    if report["failed"]:
        raise AssertionError(report["errors"])
    docs = search_documents(db_path, "任务规划", top_k=5)
    return db_path, [str(item["doc_id"]) for item in docs]


if __name__ == "__main__":
    unittest.main()
