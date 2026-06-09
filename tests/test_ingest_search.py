from __future__ import annotations

import contextlib
import io
import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from kb_agent import db
from kb_agent.answer import answer_query
from kb_agent.artifacts import get_artifact, get_citation_map, get_doc_card, get_innovations, get_parse_quality, get_parse_report, list_artifacts
from kb_agent.cli import main as cli_main
from kb_agent.embeddings import HashEmbeddingProvider, build_semantic_index
from kb_agent.eval import eval_memory, eval_review, eval_search
from kb_agent.ingest import sync_directory
from kb_agent.insights import extract_doc_insights
from kb_agent.llm import LLMError
from kb_agent.memory import compact_memory, put_memory_gated, remember_task, resume_task, search_memory
from kb_agent.models import ParsedBlock, ParsedDocument
from kb_agent.query import classify_query
from kb_agent.query_log import list_query_logs, query_stats
from kb_agent.review import assemble_review, check_review_citations, draft_review
from kb_agent.search import build_search_report, get_evidence, search_documents, search_nodes
from kb_agent.tasks import compare_papers, generate_review_plan, get_task_artifact
from kb_agent.tree_search import tree_search


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

    def test_hash_embedding_index_is_stable_and_skips_unchanged_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, _ = _sync_insight_sample(Path(tmp))
            provider = HashEmbeddingProvider()
            vector_a = provider.embed("多智能体任务规划")
            vector_b = provider.embed("多智能体任务规划")
            self.assertEqual(vector_a, vector_b)
            self.assertEqual(len(vector_a), 256)

            first = build_semantic_index(db_path, force=True, provider="hash")
            self.assertEqual(first["schema"], "semantic_index.v1")
            self.assertGreater(first["indexed_nodes"], 0)
            self.assertGreater(first["indexed_documents"], 0)

            second = build_semantic_index(db_path, provider="hash")
            self.assertEqual(second["indexed_nodes"], 0)
            self.assertGreater(second["skipped_nodes"], 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "embed", "--provider", "hash"])
            self.assertIn("semantic_index.v1", stdout.getvalue())

    def test_hybrid_search_falls_back_without_embedding_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            results = search_nodes(db_path, "动态角色", top_k=3, search_mode="hybrid")
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("fts_fallback", results[0].rank_reason)

            report = build_search_report(db_path, "动态角色", top_k=3, search_mode="hybrid")
            self.assertIn("missing_embedding_index", report["warnings"])
            self.assertEqual(report["effective_search_mode"], "fts")

            fts_results = search_nodes(db_path, "动态角色", top_k=3, search_mode="fts")
            self.assertGreaterEqual(len(fts_results), 1)
            self.assertEqual(fts_results[0].doc_id, doc_id)

    def test_hybrid_search_uses_vector_candidates(self) -> None:
        class FakeProvider:
            name = "hash"
            model = "fake-semantic-v1"
            dim = 2

            def embed(self, text: str) -> list[float]:
                if "目标节点" in text or "语义查询" in text:
                    return [1.0, 0.0]
                return [0.0, 1.0]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "target.txt").write_text(
                "摘要：本文提出目标节点方法，用于机器人协同规划。\n\n"
                "第一章 方法\n目标节点方法通过共享任务图完成协作。\n",
                encoding="utf-8",
            )
            (papers / "other.txt").write_text(
                "摘要：本文研究普通调度规则。\n\n第一章 方法\n普通调度规则依赖人工配置。\n",
                encoding="utf-8",
            )
            sync_directory(papers, db_path)
            provider = FakeProvider()
            with mock.patch("kb_agent.embeddings.get_embedding_provider", return_value=provider):
                build_semantic_index(db_path, force=True)
            with mock.patch("kb_agent.search.get_embedding_provider", return_value=provider):
                results = search_nodes(db_path, "语义查询", top_k=2, search_mode="hybrid")

            self.assertGreaterEqual(len(results), 1)
            self.assertIn("target", results[0].path)
            self.assertIn("vector", results[0].rank_reason)
            self.assertIsNotNone(results[0].vector_score)

    def test_search_eval_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, doc_id = _sync_insight_sample(root)
            build_semantic_index(db_path, force=True, provider="hash")
            queries_path = root / "queries.json"
            queries_path.write_text(
                json.dumps(
                    [
                        {
                            "query": "动态角色任务规划",
                            "expected_doc_ids": [doc_id],
                            "expected_node_keywords": ["动态角色"],
                            "intent": "method",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = eval_search(db_path, queries_path, search_mode="hybrid", top_k=3)
            self.assertEqual(report["schema"], "search_eval.v2")
            self.assertEqual(report["query_count"], 1)
            self.assertGreaterEqual(report["doc_recall_at_k"], 1.0)
            self.assertTrue(Path(report["path"]).exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "eval-search", str(queries_path), "--search-mode", "hybrid"])
            self.assertIn("search_eval.v2", stdout.getvalue())

    def test_query_classifier_recognizes_paper_intents(self) -> None:
        samples = {
            "method": "这篇论文的方法设计是什么？",
            "experiment": "实验结果和评价指标如何？",
            "limitation": "这项工作的局限和不足是什么？",
            "citation": "参考文献和引用关系有哪些？",
            "compare": "对比两篇论文的方法差异",
            "review": "生成任务规划方法研究综述",
        }
        for expected, query in samples.items():
            profile = classify_query(query, use_llm=False)
            self.assertEqual(profile["schema"], "query_profile.v1")
            self.assertEqual(profile["intent"], expected)
            self.assertTrue(profile["focus_terms"])
            self.assertTrue(profile["preferred_node_types"])

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli_main(["classify-query", "这篇论文的方法设计是什么？", "--no-llm"])
        self.assertIn('"intent": "method"', stdout.getvalue())

    def test_tree_search_value_function_selects_structured_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            trace = tree_search(db_path, doc_id, "这篇论文的方法设计是什么？", budget=3, use_llm=False)

            self.assertEqual(trace["schema"], "tree_search_trace.v1")
            self.assertEqual(trace["query_profile"]["intent"], "method")
            self.assertGreaterEqual(len(trace["evidence"]), 1)
            self.assertTrue(any("方法设计" in item["node_path"] for item in trace["evidence"]))
            self.assertIn("score_components", trace["evidence"][0])
            self.assertTrue(trace["expanded_nodes"])
            self.assertTrue(trace["selected_paths"])

            results = search_nodes(db_path, "这篇论文的方法设计是什么？", doc_id=doc_id, top_k=3, search_mode="tree")
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("tree:value", results[0].rank_reason)

            report = build_search_report(db_path, "这篇论文的方法设计是什么？", doc_id=doc_id, top_k=3, search_mode="tree")
            self.assertEqual(report["effective_search_mode"], "tree")
            self.assertIn("tree_search_trace", report)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "tree-search", doc_id, "这篇论文的方法设计是什么？", "--no-llm"])
            self.assertIn("tree_search_trace.v1", stdout.getvalue())

    def test_tree_search_llm_selection_and_require_llm_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            node_index = get_artifact(db_path, doc_id, "node_index.jsonl")["content"]
            target_id = next(
                node["node_id"]
                for node in node_index
                if "实验结果表明" in node.get("text", "")
            )
            payload = {
                "selected_node_ids": [target_id],
                "rationale": ["实验查询应优先选择实验结果节点。"],
                "warnings": [],
            }
            with mock.patch("kb_agent.tree_search.generate_json_object", return_value=payload):
                trace = tree_search(db_path, doc_id, "实验结果如何？", budget=3, use_llm=True)

            self.assertEqual(trace["evidence"][0]["node_id"], target_id)
            self.assertEqual(trace["llm_decisions"]["selected_node_ids"], [target_id])
            self.assertFalse(trace["llm_error"])

            with mock.patch("kb_agent.tree_search.generate_json_object", side_effect=LLMError("boom")):
                fallback = tree_search(db_path, doc_id, "实验结果如何？", budget=2, use_llm=True)
            self.assertIn("llm_unavailable:boom", fallback["warnings"])
            self.assertTrue(fallback["evidence"])

            with mock.patch("kb_agent.tree_search.generate_json_object", side_effect=LLMError("boom")):
                with self.assertRaises(LLMError):
                    tree_search(db_path, doc_id, "实验结果如何？", budget=2, use_llm=True, require_llm=True)

    def test_ask_compare_and_review_can_use_tree_search_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))

            answer = answer_query(
                db_path,
                "这两篇论文的任务规划方法有什么区别？",
                top_k=4,
                use_llm=False,
                search_mode="tree",
            )
            self.assertEqual(answer["search_mode"], "tree")
            self.assertTrue(answer["tree_search_trace"])
            self.assertGreaterEqual(len(answer["evidence"]), 1)
            self.assertIn("score_components", answer["evidence"][0])

            comparison = compare_papers(
                db_path,
                "服务机器人与多智能体任务规划方法对比",
                doc_ids=doc_ids,
                use_llm=False,
                search_mode="tree",
            )
            first_cell = comparison["comparison_matrix"]["dimensions"][0]["cells"][0]
            self.assertTrue(first_cell["evidence"])
            self.assertIn("tree:value", first_cell["evidence"][0]["rank_reason"])

            review = generate_review_plan(
                db_path,
                "任务规划方法研究综述",
                doc_ids=doc_ids,
                use_llm=False,
                search_mode="tree",
            )
            background = review["section_evidence"]["background_problem"]
            self.assertGreaterEqual(background["evidence_count"], 1)
            self.assertIn("score_components", background["evidence"][0])

    def test_query_log_schema_migrates_old_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "old.sqlite"
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                CREATE TABLE query_logs(
                    query_id TEXT PRIMARY KEY,
                    intent TEXT NOT NULL,
                    query TEXT NOT NULL,
                    docs_used TEXT NOT NULL DEFAULT '',
                    nodes_used TEXT NOT NULL DEFAULT '',
                    latency_ms REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO query_logs(query_id, intent, query, docs_used, nodes_used, latency_ms, created_at)
                VALUES('query_old', 'method', '旧查询', '[]', '[]', 12.0, 1.0)
                """
            )
            db.init_db(conn)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(query_logs)").fetchall()}
            conn.close()

            self.assertIn("operation", columns)
            self.assertIn("metrics_json", columns)
            logs = list_query_logs(db_path, limit=5)
            self.assertEqual(logs["count"], 1)
            self.assertEqual(logs["items"][0]["query"], "旧查询")
            self.assertEqual(logs["items"][0]["metrics"], {})

    def test_query_logs_capture_operations_without_evidence_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            doc_id = doc_ids[0]

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "search", "任务规划方法", "--top-k", "2"])
            answer_query(db_path, "服务机器人任务规划方法是什么？", top_k=2, use_llm=False)
            tree_search(db_path, doc_id, "方法设计", budget=2, use_llm=False)
            compare_papers(
                db_path,
                "服务机器人与多智能体任务规划方法对比",
                doc_ids=doc_ids,
                use_llm=False,
                search_mode="tree",
            )
            generate_review_plan(
                db_path,
                "任务规划方法研究综述",
                doc_ids=doc_ids,
                use_llm=False,
                search_mode="tree",
            )

            logs = list_query_logs(db_path, limit=200)
            operations = {item["operation"] for item in logs["items"]}
            self.assertIn("search", operations)
            self.assertIn("ask", operations)
            self.assertIn("tree-search", operations)
            self.assertIn("compare", operations)
            self.assertIn("generate-review", operations)
            dumped = json.dumps(logs, ensure_ascii=False)
            self.assertNotIn("excerpt", dumped)
            self.assertNotIn("系统通过大语言模型解析用户意图", dumped)

            stats = query_stats(db_path)
            self.assertGreaterEqual(stats["query_count"], 5)
            self.assertIn("tree-search", stats["operation_counts"])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "query-stats"])
            self.assertIn("query_stats.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "query-log", "--limit", "3"])
            self.assertIn("query_log_list.v1", stdout.getvalue())

    def test_eval_search_v2_compare_modes_and_expected_node_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, doc_id = _sync_insight_sample(root)
            build_semantic_index(db_path, force=True, provider="hash")
            node_index = get_artifact(db_path, doc_id, "node_index.jsonl")["content"]
            target_node = next(node for node in node_index if "动态角色发现机制" in node.get("text", ""))
            queries_path = root / "queries.json"
            queries_path.write_text(
                json.dumps(
                    [
                        {
                            "query": "动态角色任务规划",
                            "expected_doc_ids": [doc_id],
                            "expected_node_ids": [target_node["node_id"]],
                            "expected_node_keywords": ["动态角色"],
                            "intent": "method",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = eval_search(db_path, queries_path, search_mode="hybrid", top_k=3, compare_modes=["hybrid", "tree", "fts"])
            self.assertEqual(report["schema"], "search_eval.v2")
            self.assertEqual(set(report["mode_results"].keys()), {"hybrid", "tree", "fts"})
            self.assertIn("node_recall_at_k", report)
            self.assertTrue(Path(report["path"]).exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main([
                    "--db",
                    str(db_path),
                    "eval-search",
                    str(queries_path),
                    "--compare-modes",
                    "hybrid,tree,fts",
                ])
            self.assertIn("search_eval.v2", stdout.getvalue())

    def test_eval_review_and_memory_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            task = generate_review_plan(
                db_path,
                "任务规划方法研究综述",
                doc_ids=doc_ids,
                use_llm=False,
            )
            draft_review(db_path, task["task_id"], use_llm=False)
            check_review_citations(db_path, task["task_id"])

            review_eval = eval_review(db_path, task["task_id"])
            self.assertEqual(review_eval["schema"], "review_eval.v1")
            self.assertEqual(review_eval["task_id"], task["task_id"])
            self.assertIn("citation_coverage_score", review_eval)

            put_memory_gated(
                db_path,
                "project",
                "preference",
                "expired_pref",
                "偏好：优先检查引用覆盖。",
                ttl_days=-1,
            )
            conn = db.connect(db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO memory_items(
                        memory_id, scope, type, subject_key, content, refs, ttl,
                        importance, confidence, created_at, updated_at
                    )
                    VALUES('mem_polluted', 'project', 'task_progress', 'polluted',
                           'node_id=node_x excerpt=论文正文 page_range=[1,2]', '', NULL,
                           0.9, 0.9, 1.0, 1.0)
                    """
                )
                conn.commit()
            finally:
                conn.close()

            memory_eval = eval_memory(db_path)
            self.assertEqual(memory_eval["schema"], "memory_eval.v1")
            self.assertGreaterEqual(memory_eval["expired_count"], 1)
            self.assertGreaterEqual(memory_eval["suspected_pollution_count"], 1)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "eval-review", task["task_id"]])
            self.assertIn("review_eval.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "eval-memory"])
            self.assertIn("memory_eval.v1", stdout.getvalue())

    def test_opencode_observer_plugin_is_static_valid_and_sanitizes_sensitive_keys(self) -> None:
        plugin_path = Path(".opencode/plugins/kb-observer/index.mjs")
        self.assertTrue(plugin_path.exists())
        content = plugin_path.read_text(encoding="utf-8")
        self.assertIn("tool.execute.after", content)
        self.assertIn("experimental.session.compacting", content)
        self.assertIn("SENSITIVE_KEYS", content)
        node = shutil.which("node")
        if node:
            result = subprocess.run([node, "--check", str(plugin_path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_pdf_parser_choice_pypdf_records_quality_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "paper.pdf").write_bytes(b"%PDF fake")

            with mock.patch("kb_agent.parsers._parse_pypdf_pdf", return_value=_fake_pdf_document("pypdf")):
                report = sync_directory(papers, db_path, force=True, pdf_parser="pypdf")

            self.assertEqual(report["indexed"], 1)
            doc_id = str(search_documents(db_path, "多解析器", top_k=1)[0]["doc_id"])
            card = get_doc_card(db_path, doc_id)
            self.assertEqual(card["parser_name"], "pdf_pypdf")
            quality = get_parse_quality(db_path, doc_id)
            self.assertEqual(quality["parser_chain"], ["pypdf"])
            self.assertFalse(quality["fallback_used"])
            self.assertIn(quality["quality_level"], {"good", "usable"})

    def test_docling_adapter_enhances_structured_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "docling.pdf").write_bytes(b"%PDF fake")
            parsed = _fake_pdf_document(
                "docling",
                structured={
                    "schema": "structured.v0",
                    "blocks": [],
                    "tables": [{"caption": "表 1 解析质量对比"}],
                    "figures": [{"caption": "图 1 解析流程"}],
                },
            )

            with mock.patch("kb_agent.parsers._parse_docling_pdf", return_value=parsed):
                report = sync_directory(papers, db_path, force=True, pdf_parser="docling")

            self.assertEqual(report["indexed"], 1)
            doc_id = str(search_documents(db_path, "解析质量", top_k=1)[0]["doc_id"])
            structured = get_artifact(db_path, doc_id, "structured.json")["content"]
            self.assertGreaterEqual(len(structured["tables"]), 1)
            self.assertGreaterEqual(len(structured["figures"]), 1)
            tree = get_artifact(db_path, doc_id, "tree.json")["content"]
            types = _collect_tree_types(tree)
            self.assertIn("section", types)
            self.assertIn("table", types)
            self.assertIn("figure", types)
            parse_report = get_parse_report(db_path, doc_id)
            self.assertEqual(parse_report["parser_chain"], ["docling"])
            self.assertFalse(parse_report["fallback_used"])
            self.assertTrue(parse_report["adapter_statuses"]["marker"]["placeholder"])

    def test_grobid_enrichment_updates_metadata_and_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "grobid.pdf").write_bytes(b"%PDF fake")
            enrichment = {
                "metadata": {
                    "title": "GROBID 增强论文",
                    "authors": ["张三", "李四"],
                    "year": 2026,
                    "venue": "机器人学报",
                    "doi": "10.1234/example",
                    "abstract": "本文使用 GROBID 增强标题、作者、摘要和参考文献结构。",
                },
                "references": [
                    {"ref_id": "ref_1", "raw": "[1] 张三. 解析研究. 2025.", "authors": ["张三"], "title": "解析研究", "year": 2025}
                ],
            }

            with mock.patch("kb_agent.parsers._parse_pypdf_pdf", return_value=_fake_pdf_document("grobid")), \
                mock.patch("kb_agent.parsers._fetch_grobid_enrichment", return_value=enrichment), \
                mock.patch.dict("os.environ", {"GROBID_URL": "http://localhost:8070"}):
                report = sync_directory(papers, db_path, force=True, pdf_parser="grobid")

            self.assertEqual(report["indexed"], 1)
            doc_id = str(search_documents(db_path, "GROBID", top_k=1)[0]["doc_id"])
            card = get_doc_card(db_path, doc_id)
            self.assertEqual(card["title"], "GROBID 增强论文")
            self.assertEqual(card["authors"], ["张三", "李四"])
            self.assertEqual(card["doi"], "10.1234/example")
            references = get_artifact(db_path, doc_id, "references.json")["content"]
            self.assertEqual(references["source"], "grobid")
            self.assertEqual(len(references["references"]), 1)
            parse_report = get_parse_report(db_path, doc_id)
            self.assertEqual(parse_report["parser_chain"], ["pypdf", "grobid"])

    def test_pdf_external_failure_falls_back_to_pypdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "fallback.pdf").write_bytes(b"%PDF fake")

            with mock.patch("kb_agent.parsers._parse_docling_pdf", side_effect=RuntimeError("docling down")), \
                mock.patch("kb_agent.parsers._parse_pypdf_pdf", return_value=_fake_pdf_document("fallback")):
                report = sync_directory(papers, db_path, force=True, pdf_parser="auto")

            self.assertEqual(report["indexed"], 1)
            doc_id = str(search_documents(db_path, "回退", top_k=1)[0]["doc_id"])
            parse_report = get_parse_report(db_path, doc_id)
            self.assertEqual(parse_report["parser_name"], "pdf_auto")
            self.assertTrue(parse_report["fallback_used"])
            self.assertEqual(parse_report["parser_chain"], ["docling", "pypdf"])
            self.assertTrue(parse_report["external_parser_errors"])
            quality = get_parse_quality(db_path, doc_id)
            self.assertTrue(quality["fallback_used"])
            self.assertIn("external_parser_failed:docling:docling down", quality["quality_warnings"])

    def test_parse_report_cli_and_pdf_parser_argument_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "override.pdf").write_bytes(b"%PDF fake")

            with mock.patch("kb_agent.parsers._parse_docling_pdf", side_effect=AssertionError("should not call docling")), \
                mock.patch("kb_agent.parsers._parse_pypdf_pdf", return_value=_fake_pdf_document("override")), \
                mock.patch.dict("os.environ", {"KB_PDF_PARSER": "docling"}):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    cli_main(["--db", str(db_path), "sync", str(papers), "--force", "--pdf-parser", "pypdf"])

            self.assertIn('"indexed": 1', stdout.getvalue())
            doc_id = str(search_documents(db_path, "多解析器", top_k=1)[0]["doc_id"])
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "parse-report", doc_id])
            output = stdout.getvalue()
            self.assertIn("parser_chain", output)
            self.assertIn("pdf_pypdf", output)

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

    def test_v1_database_migrates_to_v4(self) -> None:
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
                query_log_columns = {row["name"] for row in conn.execute("PRAGMA table_info(query_logs)")}
                tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
                self.assertIn("authors", doc_columns)
                self.assertIn("parser_version", doc_columns)
                self.assertIn("keywords", node_columns)
                self.assertIn("source_offsets", node_columns)
                self.assertIn("doc_hash", node_columns)
                self.assertIn("node_embeddings", tables)
                self.assertIn("document_embeddings", tables)
                self.assertIn("operation", query_log_columns)
                self.assertIn("metrics_json", query_log_columns)
                self.assertEqual(conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()["value"], "4")
            finally:
                conn.close()


def _collect_tree_types(node: dict) -> set[str]:
    types = {str(node.get("type", ""))}
    for child in node.get("children", []):
        types.update(_collect_tree_types(child))
    return types


def _fake_pdf_document(label: str, structured=None) -> ParsedDocument:  # type: ignore[no-untyped-def]
    title = f"{label} PDF 论文"
    blocks = [
        ParsedBlock(kind="heading", text="", heading="摘要", level=1, page=1),
        ParsedBlock(kind="abstract", text=f"本文研究 PDF 多解析器融合与稳定回退，用于提升解析质量。{label}", page=1),
        ParsedBlock(kind="heading", text="", heading="关键词", level=1, page=1),
        ParsedBlock(kind="keywords", text="PDF；多解析器；解析质量；回退", page=1),
        ParsedBlock(kind="heading", text="", heading="1.1 方法设计", level=2, page=2),
        ParsedBlock(kind="paragraph", text=f"系统通过 {label} 解析器识别章节、图表和参考文献，并保留证据链。", page=2),
        ParsedBlock(kind="table", text="表 1 解析质量对比", page=2),
        ParsedBlock(kind="figure", text="图 1 解析流程", page=2),
        ParsedBlock(kind="heading", text="", heading="参考文献", level=1, page=3),
        ParsedBlock(kind="reference", text="[1] 张三. PDF 解析研究. 2025.", page=3),
    ]
    raw_text = "\n".join(block.heading or block.text for block in blocks)
    return ParsedDocument(
        title=title,
        file_type="pdf",
        raw_text=raw_text,
        blocks=blocks,
        metadata={
            "source_format": "pdf",
            "pages": 3,
            "title": title,
            "authors": ["张三"],
            "year": 2026,
            "doi": "10.1234/fake",
            "abstract": f"本文研究 PDF 多解析器融合与稳定回退，用于提升解析质量。{label}",
            "keywords": ["PDF", "多解析器", "解析质量", "回退"],
        },
        body_md="",
        structured=structured or {
            "schema": "structured.v0",
            "blocks": [],
            "tables": [{"caption": "表 1 解析质量对比"}],
            "figures": [{"caption": "图 1 解析流程"}],
            "formulas": [],
        },
        references={
            "schema": "references.v0",
            "status": "extracted",
            "references": [{"ref_id": "ref_1", "raw": "[1] 张三. PDF 解析研究. 2025."}],
            "citation_contexts": [],
        },
        parser_name=f"pdf_{label}",
        parser_version="test",
    )


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
