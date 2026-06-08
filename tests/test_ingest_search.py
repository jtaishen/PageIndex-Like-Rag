from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from kb_agent import db
from kb_agent.answer import answer_query
from kb_agent.artifacts import get_artifact, get_citation_map, get_doc_card, get_innovations, get_parse_quality, list_artifacts
from kb_agent.cli import main as cli_main
from kb_agent.ingest import sync_directory
from kb_agent.insights import extract_doc_insights
from kb_agent.llm import LLMError
from kb_agent.memory import compact_memory, put_memory_gated, remember_task, resume_task, search_memory
from kb_agent.review import assemble_review, check_review_citations, draft_review
from kb_agent.search import get_evidence, search_documents, search_nodes
from kb_agent.tasks import compare_papers, generate_review_plan, get_task_artifact


class IngestSearchTest(unittest.TestCase):
    def test_sync_markdown_and_search_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "memory.md").write_text(
                "# Agent Memory\n\n"
                "The paper studies memory compaction for tool-use agents.\n\n"
                "## Method\n\n"
                "It proposes a write gate that stores stable user preferences and task state.\n",
                encoding="utf-8",
            )

            report = sync_directory(papers, db_path)
            self.assertEqual(report["indexed"], 1)
            self.assertEqual(report["failed"], 0)

            docs = search_documents(db_path, "memory compaction", top_k=3)
            self.assertEqual(len(docs), 1)

            nodes = search_nodes(db_path, "write gate task state", top_k=3)
            self.assertGreaterEqual(len(nodes), 1)
            self.assertIn("write", nodes[0].snippet.lower())

            evidence = get_evidence(db_path, nodes[0].doc_id, [nodes[0].node_id])
            self.assertEqual(len(evidence), 1)
            self.assertIn("Agent Memory", evidence[0].node_path)

            answer = answer_query(db_path, "What does it store?", top_k=2, use_llm=False)
            self.assertIn("证据", answer["answer"])

    def test_sync_writes_v02_artifacts_and_doc_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "memory.md").write_text(
                "# Agent Memory\n\n"
                "摘要：This paper studies memory compaction for tool-use agents.\n\n"
                "关键词：memory, compaction\n\n"
                "## Method\n\n"
                "It proposes a write gate that stores stable user preferences and task state.\n",
                encoding="utf-8",
            )

            report = sync_directory(papers, db_path)
            self.assertEqual(report["indexed"], 1)
            doc_id = search_documents(db_path, "memory compaction", top_k=1)[0]["doc_id"]

            card = get_doc_card(db_path, str(doc_id))
            self.assertEqual(card["title"], "Agent Memory")
            self.assertIn("artifact_dir", card)
            self.assertGreater(card["node_count"], 0)

            listing = list_artifacts(db_path, str(doc_id))
            names = {item["name"] for item in listing["artifacts"]}
            self.assertIn("raw_text.txt", names)
            self.assertIn("body.md", names)
            self.assertIn("structured.json", names)
            self.assertIn("metadata.json", names)
            self.assertIn("references.json", names)
            self.assertIn("parse_report.json", names)
            self.assertIn("tree.json", names)
            self.assertIn("node_index.jsonl", names)
            self.assertIn("doc_card.json", names)
            self.assertIn("innovation.json", names)
            self.assertIn("citation_map.json", names)

            tree = get_artifact(db_path, str(doc_id), "tree.json")["content"]
            self.assertIn("node_path", tree)
            self.assertIn("page_range", tree)
            self.assertIn("keywords", tree)
            self.assertIn("source_offsets", tree)

            node_index = get_artifact(db_path, str(doc_id), "node_index.jsonl")["content"]
            self.assertGreater(len(node_index), 0)
            self.assertIn("doc_hash", node_index[0])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "card", str(doc_id)])
            self.assertIn("Agent Memory", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "artifacts", str(doc_id)])
            self.assertIn("node_index.jsonl", stdout.getvalue())

    def test_chinese_paper_structure_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "distributed_agents.txt").write_text(
                "摘要：本文围绕多智能体系统中的分布式任务规划展开研究，"
                "分析任务分解、协同调度和冲突消解中的关键问题，并提出面向动态环境的规划流程。\n"
                "关键词：多智能体；任务规划；协同调度\n\n"
                "第一章 绪论\n"
                "本章介绍分布式任务规划的研究背景和应用场景。\n\n"
                "1.1 研究背景\n"
                "多智能体协同需要在不确定环境下完成任务分配和路径协调。\n"
                "图 1 分布式任务规划流程\n"
                "表 1 算法能力对比\n\n"
                "1.1.1 关键挑战\n"
                "系统需要处理通信延迟、资源约束和任务冲突。\n\n"
                "结论\n"
                "实验分析表明，结构化任务规划可以提升系统鲁棒性。\n\n"
                "参考文献\n"
                "[1] 张三. 多智能体任务规划研究. 2025.\n",
                encoding="utf-8",
            )

            report = sync_directory(papers, db_path)
            self.assertEqual(report["indexed"], 1)
            doc_id = str(search_documents(db_path, "协同调度", top_k=1)[0]["doc_id"])

            tree = get_artifact(db_path, doc_id, "tree.json")["content"]
            types = _collect_tree_types(tree)
            self.assertIn("abstract", types)
            self.assertIn("keywords", types)
            self.assertIn("section", types)
            self.assertIn("figure", types)
            self.assertIn("table", types)
            self.assertIn("reference", types)

            node_index = get_artifact(db_path, doc_id, "node_index.jsonl")["content"]
            node_paths = [node["node_path"] for node in node_index]
            self.assertTrue(any("第一章 绪论" in path for path in node_paths))
            self.assertTrue(any("1.1 研究背景" in path for path in node_paths))

            card = get_doc_card(db_path, doc_id)
            self.assertIn("description", card)
            self.assertIn("多智能体", card["description"])
            self.assertGreaterEqual(card["section_count"], 3)
            self.assertGreaterEqual(len(card["sections"]), 3)
            self.assertNotIn("missing_abstract", card["quality_warnings"])
            self.assertNotIn("page_only_tree", card["quality_warnings"])

            quality = get_parse_quality(db_path, doc_id)
            self.assertFalse(quality["missing_abstract"])
            self.assertFalse(quality["page_only_tree"])
            self.assertGreaterEqual(quality["section_count"], 3)
            self.assertGreaterEqual(quality["reference_count"], 1)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "quality", doc_id])
            self.assertIn("section_count", stdout.getvalue())

    def test_rule_based_insight_extraction_writes_v1_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))

            result = extract_doc_insights(db_path, doc_id, use_llm=False)
            innovation = result["innovation"]
            citation_map = result["citation_map"]

            self.assertEqual(innovation["schema"], "innovation.v1")
            self.assertEqual(innovation["status"], "partial")
            self.assertGreaterEqual(len(innovation["items"]), 1)
            self.assertIn("llm_disabled", innovation["warnings"])
            evidence = innovation["items"][0]["evidence"][0]
            self.assertIn("node_id", evidence)
            self.assertIn("page_range", evidence)

            self.assertEqual(citation_map["schema"], "citation_map.v1")
            self.assertEqual(citation_map["status"], "extracted")
            self.assertGreaterEqual(len(citation_map["references"]), 3)
            ref_ids = {item["ref_id"] for item in citation_map["in_text_citations"]}
            self.assertIn("ref_1", ref_ids)
            self.assertIn("ref_2", ref_ids)
            self.assertIn("ref_3", ref_ids)
            self.assertGreaterEqual(len(citation_map["relations"]), 3)
            self.assertIn("node_id", citation_map["relations"][0])

            self.assertEqual(get_innovations(db_path, doc_id)["schema"], "innovation.v1")
            self.assertEqual(get_citation_map(db_path, doc_id)["schema"], "citation_map.v1")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "innovations", doc_id])
            self.assertIn("innovation.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "citations", doc_id])
            self.assertIn("citation_map.v1", stdout.getvalue())

    def test_llm_insight_extraction_normalizes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            first_node = get_artifact(db_path, doc_id, "node_index.jsonl")["content"][2]["node_id"]
            payload = {
                "items": [
                    {
                        "title": "动态角色发现机制",
                        "type": "method",
                        "claim": "提出动态角色发现机制以提升任务分解效率。",
                        "problem": "静态角色难以适应动态任务。",
                        "approach": "使用动作编码器和聚类构建角色模型。",
                        "evidence": [first_node],
                        "confidence": 0.82,
                    }
                ],
                "limitations": ["真实场景仍需要更多验证。"],
                "open_questions": ["复杂通信约束下如何扩展？"],
                "warnings": [],
            }

            with mock.patch("kb_agent.insights.generate_json_object", return_value=payload):
                result = extract_doc_insights(db_path, doc_id, force=True, use_llm=True)

            innovation = result["innovation"]
            self.assertEqual(innovation["status"], "extracted")
            self.assertEqual(innovation["source"], "llm")
            self.assertEqual(innovation["items"][0]["confidence"], 0.82)
            self.assertIn("node_id", innovation["items"][0]["evidence"][0])
            self.assertIn("真实场景", innovation["limitations"][0])

    def test_extract_requires_llm_does_not_overwrite_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))

            with mock.patch("kb_agent.insights.generate_json_object", side_effect=LLMError("boom")):
                with self.assertRaises(LLMError):
                    extract_doc_insights(db_path, doc_id, force=True, use_llm=True, require_llm=True)

            innovation = get_artifact(db_path, doc_id, "innovation.json")["content"]
            self.assertEqual(innovation["schema"], "innovation.v0")

            with mock.patch("kb_agent.insights.generate_json_object", side_effect=LLMError("boom")):
                result = extract_doc_insights(db_path, doc_id, force=True, use_llm=True)
            self.assertEqual(result["innovation"]["status"], "partial")
            self.assertTrue(any("llm_unavailable" in item for item in result["innovation"]["warnings"]))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "extract", doc_id, "--force", "--no-llm"])
            self.assertIn("innovation.v1", stdout.getvalue())

    def test_compare_writes_grounded_task_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))

            result = compare_papers(
                db_path,
                "服务机器人与多智能体任务规划方法对比",
                doc_ids=doc_ids,
                use_llm=False,
            )

            self.assertEqual(result["task_type"], "compare")
            self.assertEqual(result["selected_papers"]["paper_count"], 2)
            matrix = result["comparison_matrix"]
            self.assertEqual(matrix["schema"], "comparison_matrix.v1")
            self.assertEqual(len(matrix["dimensions"]), 6)
            dimension_ids = {item["id"] for item in matrix["dimensions"]}
            self.assertIn("problem_setting", dimension_ids)
            self.assertIn("evidence_strength", dimension_ids)
            for dimension in matrix["dimensions"]:
                self.assertEqual(len(dimension["cells"]), 2)
                for cell in dimension["cells"]:
                    self.assertTrue(cell["evidence"] or cell["warnings"])
            self.assertIn("comparison_matrix", result["artifact_paths"])

            artifact = get_task_artifact(db_path, result["task_id"], "comparison_matrix.json")
            self.assertEqual(artifact["content"]["schema"], "comparison_matrix.v1")
            current = get_task_artifact(db_path, "current", "current_task.json")
            self.assertEqual(current["content"]["task_id"], result["task_id"])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main([
                    "--db",
                    str(db_path),
                    "compare",
                    "服务机器人与多智能体任务规划方法对比",
                    "--doc-id",
                    doc_ids[0],
                    "--doc-id",
                    doc_ids[1],
                    "--no-llm",
                ])
            self.assertIn("task_id", stdout.getvalue())
            self.assertIn("comparison_matrix", stdout.getvalue())

    def test_generate_review_plan_writes_outline_and_section_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))

            result = generate_review_plan(
                db_path,
                "任务规划方法研究综述",
                doc_ids=doc_ids,
                use_llm=False,
            )

            self.assertEqual(result["task_type"], "review")
            outline = result["review_outline"]
            self.assertEqual(outline["schema"], "review_outline.v1")
            self.assertGreaterEqual(len(outline["sections"]), 5)
            self.assertIn("section_evidence/background_problem.json", result["artifact_paths"])
            background = get_task_artifact(
                db_path,
                result["task_id"],
                "section_evidence/background_problem.json",
            )
            self.assertEqual(background["content"]["schema"], "section_evidence.v1")
            self.assertGreaterEqual(background["content"]["source_doc_count"], 1)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main([
                    "--db",
                    str(db_path),
                    "generate-review",
                    "任务规划方法研究综述",
                    "--doc-id",
                    doc_ids[0],
                    "--doc-id",
                    doc_ids[1],
                    "--no-llm",
                ])
            self.assertIn("review_outline", stdout.getvalue())

    def test_llm_task_generation_and_require_llm_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            for doc_id in doc_ids:
                extract_doc_insights(db_path, doc_id, force=True, use_llm=False)

            payload = {
                "dimensions": [
                    {
                        "id": "problem_setting",
                        "synthesis": "两篇论文都关注动态任务规划问题。",
                        "overlaps": ["都涉及任务规划。"],
                        "differences": ["服务机器人强调工具调用，多智能体论文强调分布式协同。"],
                        "cells": [
                            {
                                "doc_id": doc_ids[0],
                                "claim": "服务机器人论文关注大模型任务规划。",
                                "evidence": [],
                                "confidence": 0.82,
                            },
                            {
                                "doc_id": doc_ids[1],
                                "claim": "多智能体论文关注分布式任务分配。",
                                "evidence": [],
                                "confidence": 0.8,
                            },
                        ],
                    }
                ],
                "open_questions": ["需要进一步比较真实实验设置。"],
                "warnings": [],
            }

            with mock.patch("kb_agent.tasks.generate_json_object", return_value=payload):
                result = compare_papers(
                    db_path,
                    "服务机器人与多智能体任务规划方法对比",
                    doc_ids=doc_ids,
                    use_llm=True,
                )
            self.assertEqual(result["comparison_matrix"]["source"], "llm")
            self.assertIn("需要进一步比较", result["comparison_matrix"]["open_questions"][0])
            self.assertIn("node_id", result["comparison_matrix"]["dimensions"][0]["cells"][0]["evidence"][0])

        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            state_root = db_path.parent / ".kb_state"
            with mock.patch("kb_agent.tasks.generate_json_object", side_effect=LLMError("boom")):
                with self.assertRaises(LLMError):
                    compare_papers(
                        db_path,
                        "服务机器人与多智能体任务规划方法对比",
                        doc_ids=doc_ids,
                        use_llm=True,
                        require_llm=True,
                    )
            self.assertFalse(state_root.exists())

    def test_draft_review_writes_section_drafts_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            task = generate_review_plan(
                db_path,
                "任务规划方法研究综述",
                doc_ids=doc_ids,
                use_llm=False,
            )

            result = draft_review(db_path, task["task_id"], use_llm=False)

            self.assertEqual(result["schema"], "review_draft_result.v1")
            self.assertEqual(result["drafted_section_count"], 5)
            self.assertIn("review_draft", result["artifact_paths"])
            draft = get_task_artifact(db_path, task["task_id"], "section_drafts/background_problem.json")
            self.assertEqual(draft["content"]["schema"], "section_draft.v1")
            self.assertEqual(draft["content"]["status"], "partial")
            self.assertGreaterEqual(len(draft["content"]["evidence"]), 1)

            review_draft = get_task_artifact(db_path, task["task_id"], "review_draft.md")
            self.assertIn("任务规划方法研究综述", review_draft["content"])
            citation_check = get_task_artifact(db_path, task["task_id"], "citation_check.json")
            self.assertEqual(citation_check["content"]["schema"], "citation_check.v1")
            report = get_task_artifact(db_path, task["task_id"], "review_report.json")
            self.assertEqual(report["content"]["schema"], "review_report.v1")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "assemble-review", task["task_id"]])
            self.assertIn("review_draft", stdout.getvalue())

    def test_llm_review_draft_normalizes_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            task = generate_review_plan(
                db_path,
                "任务规划方法研究综述",
                doc_ids=doc_ids,
                use_llm=False,
            )
            payload = {
                "claim_plan": [{"claim": "两类论文都关注任务规划问题。", "evidence": ["E1"]}],
                "body_markdown": "本节可先界定任务规划问题，并说明服务机器人和多智能体场景都需要处理动态约束。[E1]",
                "unsupported_claims": [],
                "warnings": [],
            }

            with mock.patch("kb_agent.review.generate_json_object", return_value=payload):
                result = draft_review(
                    db_path,
                    task["task_id"],
                    section_ids=["background_problem"],
                    use_llm=True,
                )

            draft = result["section_drafts"][0]
            self.assertEqual(draft["source"], "llm")
            self.assertEqual(draft["status"], "drafted")
            self.assertEqual(draft["used_evidence"][0]["ref_id"], "E1")
            self.assertFalse(result["citation_check"]["missing_refs"])
            self.assertEqual(result["citation_check"]["sections"][0]["coverage_score"], 1.0)

    def test_draft_review_requires_llm_does_not_write_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            task = generate_review_plan(
                db_path,
                "任务规划方法研究综述",
                doc_ids=doc_ids,
                use_llm=False,
            )
            draft_dir = db_path.parent / ".kb_state" / task["task_id"] / "section_drafts"

            with mock.patch("kb_agent.review.generate_json_object", side_effect=LLMError("boom")):
                with self.assertRaises(LLMError):
                    draft_review(
                        db_path,
                        task["task_id"],
                        section_ids=["background_problem"],
                        use_llm=True,
                        require_llm=True,
                    )

            self.assertFalse(draft_dir.exists())

    def test_check_review_detects_bad_refs_and_unsupported_paragraphs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            task = generate_review_plan(
                db_path,
                "任务规划方法研究综述",
                doc_ids=doc_ids,
                use_llm=False,
            )
            draft_review(db_path, task["task_id"], section_ids=["background_problem"], use_llm=False)
            artifact = get_task_artifact(db_path, task["task_id"], "section_drafts/background_problem.json")
            payload = artifact["content"]
            payload["body_markdown"] = (
                "这一段是没有任何证据标记的长段落，需要被引用一致性检查识别出来。\n\n"
                "这一段引用了不存在的证据编号，因此也应该被记录为缺失引用。[E99]"
            )
            Path(artifact["path"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            result = check_review_citations(db_path, task["task_id"])

            self.assertEqual(result["citation_check"]["status"], "partial")
            missing_refs = result["citation_check"]["missing_refs"]
            self.assertTrue(any(item["ref_id"] == "E99" for item in missing_refs))
            self.assertGreaterEqual(len(result["citation_check"]["unsupported_paragraphs"]), 1)

            assembled = assemble_review(db_path, task["task_id"])
            self.assertIn("review_draft", assembled["artifact_paths"])

    def test_memory_write_gate_allows_preferences_and_rejects_paper_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"

            accepted = put_memory_gated(
                db_path,
                "project",
                "preference",
                "citation_style",
                "默认引用格式使用 GB/T 7714。",
                confidence=0.9,
            )
            self.assertTrue(accepted["accepted"])
            self.assertEqual(accepted["action"], "accepted")

            merged = put_memory_gated(
                db_path,
                "project",
                "preference",
                "citation_style",
                "综述草稿也使用 GB/T 7714。",
                confidence=0.9,
            )
            self.assertEqual(merged["action"], "merged")
            memories = search_memory(db_path, "GB/T", scope="project")
            self.assertEqual(len(memories), 1)
            self.assertIn("综述草稿", memories[0]["content"])

            rejected = put_memory_gated(
                db_path,
                "project",
                "task_progress",
                "bad_evidence",
                "node_id=node_1 page_range=[1,2] excerpt=这是一段论文证据正文。",
                confidence=0.9,
            )
            self.assertFalse(rejected["accepted"])
            self.assertEqual(rejected["reason"], "paper_asset_boundary")

            low_confidence = put_memory_gated(
                db_path,
                "project",
                "preference",
                "uncertain",
                "也许用户喜欢某种格式。",
                confidence=0.2,
            )
            self.assertFalse(low_confidence["accepted"])
            self.assertEqual(low_confidence["reason"], "low_confidence")

    def test_memory_ttl_filters_expired_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"
            put_memory_gated(
                db_path,
                "project",
                "preference",
                "expired",
                "短期偏好。",
                ttl_days=-1,
                force=True,
            )
            put_memory_gated(
                db_path,
                "project",
                "preference",
                "active",
                "长期偏好。",
                ttl_days=1,
            )

            memories = search_memory(db_path, "偏好", scope="project")
            keys = {item["subject_key"] for item in memories}
            self.assertNotIn("expired", keys)
            self.assertIn("active", keys)

    def test_remember_resume_and_compact_task_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            task = generate_review_plan(
                db_path,
                "任务规划方法研究综述",
                doc_ids=doc_ids,
                use_llm=False,
            )
            draft_review(db_path, task["task_id"], section_ids=["background_problem"], use_llm=False)

            remembered = remember_task(db_path, task["task_id"])
            self.assertTrue(remembered["memory"]["accepted"])
            self.assertIn(task["task_id"], remembered["summary"])

            resumed = resume_task(db_path)
            self.assertEqual(resumed["current_task"]["task_id"], task["task_id"])
            self.assertIn("suggested_commands", resumed)
            self.assertTrue(any("remember-task" in item for item in resumed["suggested_commands"]))

            put_memory_gated(
                db_path,
                "project",
                "task_progress",
                "task:manual",
                "task_id: task_manual\nstatus: partial\nnext_actions: 继续检查",
                confidence=0.8,
            )
            compacted = compact_memory(db_path, scope="project")
            self.assertEqual(compacted["status"], "compacted")
            self.assertGreaterEqual(compacted["compacted_count"], 2)
            summary = search_memory(db_path, "recent task progress", scope="project")
            self.assertTrue(any(item["subject_key"] == "task_progress_summary" for item in summary))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "resume-task"])
            self.assertIn(task["task_id"], stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "memory-compact", "--scope", "project"])
            self.assertIn("memory_compact.v1", stdout.getvalue())

    def test_failed_parse_records_report_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "broken.docx").write_text("not a real docx", encoding="utf-8")

            report = sync_directory(papers, db_path)
            self.assertEqual(report["failed"], 1)
            self.assertEqual(len(report["errors"]), 1)

            conn = db.connect(db_path)
            db.init_db(conn)
            try:
                row = db.list_documents(conn)[0]
                self.assertEqual(row["status"], "failed")
                version = db.get_document_version(conn, row["doc_id"])
                self.assertIsNotNone(version)
            finally:
                conn.close()

            listing = list_artifacts(db_path, row["doc_id"])
            report_artifact = next(item for item in listing["artifacts"] if item["name"] == "parse_report.json")
            self.assertTrue(report_artifact["exists"])
            parse_report = get_artifact(db_path, row["doc_id"], "parse_report.json")["content"]
            self.assertEqual(parse_report["status"], "failed")
            self.assertIn("Cannot read DOCX", parse_report["error"])

    def test_v1_database_migrates_to_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "v1.sqlite"
            conn = sqlite3.connect(db_path)
            try:
                conn.executescript(
                    """
                    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    INSERT INTO meta(key, value) VALUES('schema_version', '1');
                    CREATE TABLE documents (
                        doc_id TEXT PRIMARY KEY,
                        path TEXT NOT NULL UNIQUE,
                        hash TEXT NOT NULL,
                        title TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        mtime REAL NOT NULL,
                        summary TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'ready',
                        error TEXT NOT NULL DEFAULT '',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE doc_nodes (
                        node_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        parent_id TEXT,
                        type TEXT NOT NULL,
                        heading TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        text TEXT NOT NULL,
                        level INTEGER NOT NULL,
                        node_path TEXT NOT NULL,
                        page_start INTEGER,
                        page_end INTEGER,
                        order_index INTEGER NOT NULL,
                        char_start INTEGER,
                        char_end INTEGER
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

            conn = db.connect(db_path)
            try:
                db.init_db(conn)
                doc_columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
                node_columns = {row["name"] for row in conn.execute("PRAGMA table_info(doc_nodes)")}
                self.assertIn("authors", doc_columns)
                self.assertIn("parser_version", doc_columns)
                self.assertIn("keywords", node_columns)
                self.assertIn("source_offsets", node_columns)
                self.assertIn("doc_hash", node_columns)
                self.assertEqual(conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()["value"], "2")
            finally:
                conn.close()


def _collect_tree_types(node: dict) -> set[str]:
    types = {str(node.get("type", ""))}
    for child in node.get("children", []):
        types.update(_collect_tree_types(child))
    return types


def _sync_insight_sample(root: Path) -> tuple[Path, str]:
    papers = root / "papers"
    papers.mkdir()
    db_path = root / "kb.sqlite"
    (papers / "insight.txt").write_text(
        "摘要：本文研究多智能体系统中的分布式任务规划，重点解决动态任务分配、"
        "角色适配和负载均衡问题，并提出可扩展的协同规划框架。\n"
        "关键词：多智能体；任务规划；动态角色\n\n"
        "第一章 绪论\n"
        "现有任务分配方法在动态任务环境中存在适应性不足的问题[1]。\n\n"
        "1.4 研究内容与主要贡献\n"
        "本文的研究内容包括三个方面：提出动态角色发现机制，设计动态任务重分配算法，"
        "并构建基于均衡性的分布式任务分配模型。该方法引用已有任务规划研究[1]，"
        "并对分布式系统综述和任务分配方法进行扩展[2-3]。\n\n"
        "2.1 方法设计\n"
        "本文设计动作编码器和聚类方法构建角色模型，提出任务驱动的重分配算法，"
        "以降低通信和计算开销。\n\n"
        "3.1 实验结果\n"
        "实验结果表明，该方法在任务完成率、响应时间和负载均衡方面优于基线方法。\n\n"
        "结论\n"
        "本文仍存在真实场景验证不足的局限，未来工作将扩展到复杂通信约束。\n\n"
        "参考文献\n"
        "[1] 张三. 多智能体任务规划研究. 2024.\n"
        "[2] 李四. 分布式系统综述. 2023.\n"
        "[3] Wang. Task Allocation for Multi-Agent Systems. 2022.\n",
        encoding="utf-8",
    )
    report = sync_directory(papers, db_path)
    if report["failed"]:
        raise AssertionError(report["errors"])
    doc_id = str(search_documents(db_path, "动态角色", top_k=1)[0]["doc_id"])
    return db_path, doc_id


def _sync_compare_samples(root: Path) -> tuple[Path, list[str]]:
    papers = root / "papers"
    papers.mkdir()
    db_path = root / "kb.sqlite"
    (papers / "service_robot.txt").write_text(
        "摘要：本文研究基于大语言模型的服务机器人任务规划方法，"
        "解决自然语言任务分解、工具调用和执行反馈中的不确定性问题。\n"
        "关键词：服务机器人；大语言模型；任务规划；工具调用\n\n"
        "第一章 绪论\n"
        "服务机器人需要在家庭和公共场景中完成复杂任务，现有规划方法存在泛化能力不足的问题。\n\n"
        "1.1 研究内容与主要贡献\n"
        "本文提出结合大语言模型和技能库的任务规划框架，设计任务分解、工具选择和反馈修正流程，"
        "用于提升机器人在开放环境中的任务完成率。\n\n"
        "2.1 方法设计\n"
        "系统通过大语言模型解析用户意图，再调用工具和技能模块生成可执行计划。\n\n"
        "3.1 实验结果\n"
        "实验结果表明，该方法在任务成功率和响应时间方面优于规则规划基线。\n\n"
        "结论\n"
        "本文仍存在真实家庭环境验证不足的局限，未来工作将扩展长期记忆和安全约束。\n\n"
        "参考文献\n"
        "[1] 张三. 服务机器人规划研究. 2025.\n",
        encoding="utf-8",
    )
    (papers / "multi_agent.txt").write_text(
        "摘要：本文研究多智能体系统中的分布式任务规划关键技术，"
        "重点解决动态任务分配、协同调度和冲突消解问题。\n"
        "关键词：多智能体；分布式任务规划；协同调度\n\n"
        "第一章 绪论\n"
        "多智能体系统在救援和物流场景中需要协同完成任务，通信延迟和资源约束带来规划挑战。\n\n"
        "1.1 研究内容与主要贡献\n"
        "本文提出动态任务重分配算法和负载均衡模型，构建面向动态环境的协同规划框架。\n\n"
        "2.1 方法设计\n"
        "系统通过局部决策和全局协调机制降低通信开销，并处理任务冲突。\n\n"
        "3.1 实验结果\n"
        "实验结果表明，该方法提升了任务完成率、负载均衡和系统鲁棒性。\n\n"
        "结论\n"
        "本文仍存在复杂通信约束下验证不足的局限，未来工作将扩展异构智能体场景。\n\n"
        "参考文献\n"
        "[1] 李四. 多智能体任务规划研究. 2024.\n",
        encoding="utf-8",
    )
    report = sync_directory(papers, db_path)
    if report["failed"]:
        raise AssertionError(report["errors"])
    docs = search_documents(db_path, "任务规划", top_k=5)
    doc_ids = [str(item["doc_id"]) for item in docs]
    if len(doc_ids) != 2:
        raise AssertionError(docs)
    return db_path, doc_ids


if __name__ == "__main__":
    unittest.main()
