from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / ".opencode" / "skills"
MCP_SERVER = ROOT / "kb_agent" / "mcp_server.py"
README = ROOT / "README.md"


class OpenCodeWorkflowTest(unittest.TestCase):
    def test_workflow_skills_have_frontmatter_and_required_sections(self) -> None:
        skill_files = sorted(SKILLS_DIR.glob("*/SKILL.md"))
        self.assertGreaterEqual(len(skill_files), 7)

        for path in skill_files:
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"), path)
            self.assertRegex(text, r"(?m)^name: [a-z0-9-]+$")
            self.assertRegex(text, r"(?m)^description: .+")
            self.assertIn("## 适用场景", text, path)
            self.assertIn("## 必调工具顺序", text, path)
            self.assertIn("## 输出要求", text, path)
            self.assertIn("## 禁止事项", text, path)

    def test_workflow_tool_references_exist_in_mcp_server(self) -> None:
        available = set(re.findall(r"def ((?:kb|memory)_[a-zA-Z0-9_]+)\(", MCP_SERVER.read_text(encoding="utf-8")))
        self.assertIn("kb_tree_search", available)
        self.assertIn("memory_resume_task", available)

        for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            referenced = set(re.findall(r"`((?:kb|memory)_[a-zA-Z0-9_]+)`", text))
            missing = sorted(referenced - available)
            self.assertEqual(missing, [], f"{path} references missing MCP tools")

        readme = README.read_text(encoding="utf-8")
        readme_refs = set(re.findall(r"`((?:kb|memory)_[a-zA-Z0-9_]+)`", readme))
        self.assertEqual(sorted(readme_refs - available), [])

    def test_paper_qa_is_focused_workflow_not_full_tool_catalog(self) -> None:
        text = (SKILLS_DIR / "paper-qa" / "SKILL.md").read_text(encoding="utf-8")
        referenced = re.findall(r"`((?:kb|memory)_[a-zA-Z0-9_]+)`", text)

        self.assertLessEqual(len(set(referenced)), 12)
        self.assertIn("kb_search_docs", referenced)
        self.assertIn("kb_tree_search", referenced)
        self.assertIn("kb_answer", referenced)
        self.assertNotIn("kb_sync -> kb_build_semantic_index", text)
        self.assertNotIn("kb_run_quality_baseline", referenced)

    def test_readme_documents_workflow_table(self) -> None:
        text = README.read_text(encoding="utf-8")

        self.assertIn("OpenCode 使用方式", text)
        for skill_name in [
            "ingest-papers",
            "paper-qa",
            "paper-insight",
            "compare-papers",
            "review-writing",
            "quality-review",
            "task-resume",
            "memory-hygiene",
        ]:
            self.assertIn(skill_name, text)

        for phrase in [
            "入库与质量检查",
            "证据优先问答",
            "单篇论文理解",
            "跨论文比较",
            "综述写作",
            "质量复盘",
            "任务恢复与记忆",
        ]:
            self.assertIn(phrase, text)

    def test_llm_heavy_workflows_use_resumable_staged_tools(self) -> None:
        expected_by_skill = {
            "paper-insight": {
                "kb_prepare_fact_extraction",
                "kb_extract_fact_batch",
                "kb_finalize_fact_extraction",
                "kb_get_workflow_status",
            },
            "compare-papers": {
                "kb_prepare_compare",
                "kb_generate_compare_dimension",
                "kb_finalize_compare",
                "kb_get_workflow_status",
            },
            "review-writing": {
                "kb_prepare_review",
                "kb_generate_review_outline_section",
                "kb_finalize_review_outline",
                "kb_draft_review_section",
                "kb_get_workflow_status",
            },
        }
        for skill_name, expected in expected_by_skill.items():
            text = (SKILLS_DIR / skill_name / "SKILL.md").read_text(encoding="utf-8")
            referenced = set(re.findall(r"`((?:kb|memory)_[a-zA-Z0-9_]+)`", text))
            self.assertTrue(expected.issubset(referenced), f"{skill_name} is missing staged tools")


if __name__ == "__main__":
    unittest.main()
