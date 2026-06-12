from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kb_agent.config import DEFAULT_DB_PATH, PROJECT_ROOT
from kb_agent.task_artifacts import (
    get_task_artifact,
    next_actions_artifact,
    open_questions_artifact,
    section_evidence_artifact,
    selected_papers_artifact,
    task_manifest,
    task_state_root,
    valid_task_artifact_name,
    write_task_artifacts,
)


class TaskArtifactsTest(unittest.TestCase):
    def test_task_state_root_uses_project_root_for_default_db_and_db_parent_for_custom_db(self) -> None:
        self.assertEqual(task_state_root(DEFAULT_DB_PATH), PROJECT_ROOT / ".kb_state")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "nested" / "kb.sqlite"

            self.assertEqual(task_state_root(db_path), db_path.expanduser().resolve().parent / ".kb_state")

    def test_valid_task_artifact_name_allows_known_artifacts_and_rejects_traversal(self) -> None:
        self.assertTrue(valid_task_artifact_name("manifest.json"))
        self.assertTrue(valid_task_artifact_name("current_task.json"))
        self.assertTrue(valid_task_artifact_name("section_evidence/background_problem.json"))
        self.assertTrue(valid_task_artifact_name("section_drafts/background_problem.md"))
        self.assertFalse(valid_task_artifact_name("../manifest.json"))
        self.assertFalse(valid_task_artifact_name("section_evidence/../bad.json"))
        self.assertFalse(valid_task_artifact_name("section_evidence/nested/bad.json"))
        self.assertFalse(valid_task_artifact_name("unknown.json"))

    def test_write_and_read_task_artifacts_preserves_existing_schema_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"
            task_id = "task_abcdef123456"
            contexts = [
                {
                    "doc_id": "doc-1",
                    "title": "Paper 1",
                    "path": "/tmp/paper1.txt",
                    "description": "short",
                    "abstract": "abstract",
                    "keywords": ["task"],
                    "quality": {"quality_warnings": ["warn"]},
                    "innovation": {"status": "ok", "items": [{}], "references": []},
                    "citation_map": {"references": [{"id": "r1"}]},
                    "facts": {
                        "available": True,
                        "claim_count": 1,
                        "entity_count": 2,
                        "relation_count": 3,
                        "table_backed_fact_count": 0,
                    },
                    "claim_frames": {"summary": {"available": True, "frame_count": 4, "verified_frame_rate": 0.5}},
                    "route_score": 0.7,
                    "node_matches": ["n1"],
                }
            ]
            manifest = task_manifest(task_id, "review", "任务规划", "partial", ["warn"])
            selected = selected_papers_artifact(task_id, "review", "任务规划", contexts)
            section = section_evidence_artifact(
                task_id,
                "background_problem",
                "任务规划",
                [
                    {
                        "doc_id": "doc-1",
                        "node_id": "n1",
                        "node_path": "1",
                        "summary": "任务规划证据",
                        "excerpt": "任务规划证据",
                    }
                ],
            )
            open_questions = open_questions_artifact(task_id, [], {"missing_sections": ["background_problem"]}, [])
            next_actions = next_actions_artifact(task_id, "review", {"source_doc_count": 1}, ["warn"])

            paths = write_task_artifacts(
                db_path,
                task_id,
                manifest=manifest,
                selected_papers=selected,
                open_questions=open_questions,
                next_actions=next_actions,
                review_outline={"schema": "review_outline.v1", "task_id": task_id},
                section_evidence={"background_problem": section},
            )

            self.assertEqual(get_task_artifact(db_path, task_id, "manifest.json")["content"]["schema"], "task_manifest.v1")
            self.assertEqual(get_task_artifact(db_path, task_id, "selected_papers.json")["content"]["paper_count"], 1)
            section_artifact = get_task_artifact(db_path, task_id, "section_evidence/background_problem.json")["content"]
            self.assertEqual(section_artifact["schema"], "section_evidence.v1")
            self.assertEqual(section_artifact["source_doc_ids"], ["doc-1"])
            current = get_task_artifact(db_path, "current", "current_task.json")["content"]
            self.assertEqual(current["task_id"], task_id)
            self.assertEqual(paths["current_task"], str(task_state_root(db_path) / "current_task.json"))

    def test_get_task_artifact_rejects_invalid_task_id_and_artifact_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"

            with self.assertRaises(ValueError):
                get_task_artifact(db_path, "bad-task", "manifest.json")
            with self.assertRaises(ValueError):
                get_task_artifact(db_path, "task_abcdef123456", "../manifest.json")


if __name__ == "__main__":
    unittest.main()
