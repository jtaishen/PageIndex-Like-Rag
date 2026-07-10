from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from kb_agent.ingest import sync_directory
from kb_agent.review_workflow import (
    draft_review_section,
    finalize_review_outline,
    generate_review_outline_section,
    prepare_review_workflow,
)
from kb_agent.search import search_documents
from kb_agent.task_artifacts import get_task_artifact


class ReviewWorkflowTest(unittest.TestCase):
    def test_staged_outline_calls_llm_once_per_section_and_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_review_samples(Path(tmp))
            prepared = prepare_review_workflow(
                db_path,
                "任务规划方法综述",
                doc_ids=doc_ids,
                top_k_docs=2,
                search_mode="fts",
            )
            task_id = prepared["task_id"]
            self.assertEqual(prepared["status"], "prepared")
            self.assertEqual(prepared["workflow"]["phase"], "outline")

            calls: list[str] = []

            def generate(_system: str, user: str) -> dict:
                calls.append(user)
                section_id = re.search(r"section_id: ([^\n]+)", user).group(1)  # type: ignore[union-attr]
                return {
                    "section_id": section_id,
                    "title": f"LLM {section_id}",
                    "purpose": "基于证据规划本节。",
                    "paper_ids": doc_ids,
                    "evidence": [],
                    "warnings": [],
                }

            for section_id in prepared["section_ids"]:
                result = generate_review_outline_section(
                    db_path,
                    task_id,
                    section_id,
                    json_generator=generate,
                )
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["llm_diagnostics"]["mode"], "staged_section_json")

            self.assertEqual(len(calls), len(prepared["section_ids"]))
            finalized = finalize_review_outline(db_path, task_id)
            self.assertNotEqual(finalized["status"], "incomplete")
            self.assertEqual(finalized["review_outline"]["source"], "llm_staged")
            self.assertEqual(finalized["workflow"]["phase"], "draft")
            persisted = get_task_artifact(db_path, task_id, "workflow_state.json")["content"]
            self.assertEqual(persisted["summary"]["completed_step_count"], len(prepared["section_ids"]))

            draft_calls = 0

            def draft_generate(_system: str, _user: str) -> dict:
                nonlocal draft_calls
                draft_calls += 1
                return {
                    "claim_plan": [{"claim": "基于证据形成章节结论。", "evidence": ["E1"]}],
                    "body_markdown": "本节依据候选论文证据归纳研究进展。[E1]",
                    "unsupported_claims": [],
                    "warnings": [],
                }

            drafted = draft_review_section(
                db_path,
                task_id,
                prepared["section_ids"][0],
                json_generator=draft_generate,
            )
            self.assertEqual(draft_calls, 1)
            self.assertEqual(drafted["drafted_section_count"], 1)

    def test_failed_outline_step_is_persisted_and_can_be_retried(self) -> None:
        from kb_agent.llm import LLMError

        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_review_samples(Path(tmp))
            prepared = prepare_review_workflow(
                db_path,
                "任务规划",
                doc_ids=doc_ids,
                top_k_docs=2,
                search_mode="fts",
            )
            section_id = prepared["section_ids"][0]
            failed = generate_review_outline_section(
                db_path,
                prepared["task_id"],
                section_id,
                json_generator=lambda _system, _user: (_ for _ in ()).throw(
                    LLMError("timeout", error_type="request_timeout")
                ),
            )

            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error_type"], "request_timeout")
            step = next(
                item
                for item in failed["workflow"]["steps"]
                if item["step_id"] == f"outline:{section_id}"
            )
            self.assertEqual(step["status"], "failed")
            self.assertEqual(step["attempt_count"], 1)


def _sync_review_samples(root: Path) -> tuple[Path, list[str]]:
    papers = root / "papers"
    papers.mkdir()
    db_path = root / "kb.sqlite"
    (papers / "paper_a.md").write_text(
        "# 摘要\n本文提出服务机器人任务分解与反馈修正方法。\n\n"
        "# 方法\n系统使用语言模型分解任务并根据执行反馈修正规划。\n\n"
        "# 实验\n实验显示任务完成率提升。\n\n# 局限\n真实环境验证仍不足。\n",
        encoding="utf-8",
    )
    (papers / "paper_b.md").write_text(
        "# 摘要\n本文研究多智能体动态任务分配。\n\n"
        "# 方法\n系统结合局部决策与全局协调完成任务分配。\n\n"
        "# 实验\n实验显示负载均衡有所改善。\n\n# 局限\n通信约束仍需研究。\n",
        encoding="utf-8",
    )
    report = sync_directory(papers, db_path)
    if report["failed"]:
        raise AssertionError(report["errors"])
    docs = search_documents(db_path, "任务规划 任务分配", top_k=5)
    return db_path, [str(item["doc_id"]) for item in docs]


if __name__ == "__main__":
    unittest.main()
