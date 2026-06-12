from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kb_agent.cli import main as cli_main
from kb_agent.memory import compile_memory_context, put_memory, put_memory_gated, resume_task
from kb_agent.tasks import _task_state_root


TASK_ID = "task_abcdef123456"


class MemoryCompilerTest(unittest.TestCase):
    def test_compile_memory_context_prefers_task_artifacts_and_filters_pollution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"
            _, task_dir = _create_review_task(db_path)
            _seed_memory(db_path, task_dir)

            result = compile_memory_context(
                db_path,
                "review",
                "继续写综述并完善任务规划方法比较",
                task_id=TASK_ID,
                skill_scope="review",
                max_items=6,
                max_chars=2400,
            )

            self.assertEqual(result["schema"], "memory_context.v1")
            self.assertEqual(result["task_id"], TASK_ID)
            self.assertTrue(result["read_policy"]["artifact_first"])
            self.assertGreaterEqual(result["artifact_ref_count"], 4)
            self.assertTrue(any(item["name"] == "review_outline.json" for item in result["artifact_refs"]))
            self.assertTrue(any(item["name"] == "section_evidence/*" for item in result["artifact_refs"]))
            self.assertGreaterEqual(result["selected_memory_count"], 2)
            self.assertIn("artifact_refs:", result["compiled_context"])
            self.assertIn("selected_memories:", result["compiled_context"])
            self.assertIn("下一步完善方法比较", result["compiled_context"])
            self.assertNotIn("node_id: n1", result["compiled_context"])
            self.assertTrue(any(item["reason"] == "paper_asset_boundary" for item in result["filtered_memories"]))

    def test_skill_scope_excludes_task_progress_for_paper_qa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"
            _, task_dir = _create_review_task(db_path)
            _seed_memory(db_path, task_dir)

            result = compile_memory_context(
                db_path,
                "paper_qa",
                "引用格式",
                task_id=TASK_ID,
                skill_scope="paper_qa",
                max_items=6,
                max_chars=1400,
            )

            selected_types = {item["type"] for item in result["selected_memories"]}
            self.assertNotIn("task_progress", selected_types)
            self.assertIn("preference", selected_types)
            self.assertTrue(
                any(item["type"] == "task_progress" and item["reason"] == "skill_scope_mismatch" for item in result["filtered_memories"])
            )

    def test_truncation_resume_preview_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"
            _create_review_task(db_path)
            _seed_memory(db_path, _task_state_root(db_path) / TASK_ID)

            short = compile_memory_context(
                db_path,
                "review",
                "继续写综述",
                task_id=TASK_ID,
                skill_scope="review",
                max_items=5,
                max_chars=80,
            )
            self.assertLessEqual(short["context_char_count"], 80)
            self.assertIn("context_truncated", short["warnings"])

            resumed = resume_task(db_path)
            preview = resumed["compiled_context_preview"]
            self.assertEqual(preview["schema"], "memory_context.v1")
            self.assertTrue(preview["available"])
            self.assertGreater(preview["artifact_ref_count"], 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(
                    [
                        "--db",
                        str(db_path),
                        "memory-compile",
                        "继续写综述",
                        "--intent",
                        "review",
                        "--task-id",
                        TASK_ID,
                        "--skill-scope",
                        "review",
                        "--max-chars",
                        "500",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema"], "memory_context.v1")
            self.assertEqual(payload["task_id"], TASK_ID)
            self.assertGreater(payload["artifact_ref_count"], 0)


def _create_review_task(db_path: Path) -> tuple[str, Path]:
    root = _task_state_root(db_path)
    task_dir = root / TASK_ID
    _write_json(
        root / "current_task.json",
        {
            "schema": "current_task.v1",
            "task_id": TASK_ID,
            "task_type": "review",
            "query": "任务规划方法研究综述",
        },
    )
    _write_json(
        task_dir / "manifest.json",
        {
            "schema": "task_manifest.v1",
            "task_id": TASK_ID,
            "task_type": "review",
            "query": "任务规划方法研究综述",
            "status": "planned",
        },
    )
    _write_json(
        task_dir / "selected_papers.json",
        {
            "schema": "selected_papers.v1",
            "paper_count": 2,
            "papers": [
                {"doc_id": "doc-a", "title": "服务机器人任务规划"},
                {"doc_id": "doc-b", "title": "多智能体协作任务规划"},
            ],
        },
    )
    _write_json(
        task_dir / "review_outline.json",
        {
            "schema": "review_outline.v1",
            "sections": [
                {"section_id": "background_problem", "title": "背景与问题"},
                {"section_id": "methods", "title": "任务规划方法"},
            ],
        },
    )
    _write_json(
        task_dir / "next_actions.json",
        {
            "schema": "next_actions.v1",
            "items": ["下一步完善方法比较", "检查引用覆盖"],
        },
    )
    _write_json(
        task_dir / "section_evidence" / "background_problem.json",
        {
            "schema": "section_evidence.v1",
            "section_id": "background_problem",
            "evidence_count": 2,
            "source_doc_count": 2,
            "evidence": [{"evidence_id": "E1"}, {"evidence_id": "E2"}],
        },
    )
    return TASK_ID, task_dir


def _seed_memory(db_path: Path, task_dir: Path) -> None:
    put_memory_gated(
        db_path,
        "project",
        "preference",
        "citation_style",
        "综述引用格式使用 GB/T 7714。",
        importance=0.9,
        confidence=0.95,
    )
    put_memory_gated(
        db_path,
        "project",
        "task_progress",
        f"task:{TASK_ID}",
        f"task_id: {TASK_ID}\n下一步完善方法比较，并检查引用覆盖。",
        refs=str(task_dir),
        importance=0.8,
        confidence=0.9,
        force=True,
    )
    put_memory(
        db_path,
        "project",
        "task_progress",
        "polluted_evidence",
        "node_id: n1 page_range: 1 excerpt: 这是一段不应该进入长期上下文的论文正文。",
        importance=1.0,
        confidence=0.9,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
