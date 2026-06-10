from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

from kb_agent import db
from kb_agent.answer import answer_query
from kb_agent.artifacts import (
    get_artifact,
    get_citation_map,
    get_doc_card,
    get_figures,
    get_innovations,
    get_layout_blocks,
    get_parse_quality,
    get_parse_report,
    get_table_content,
    get_table_summaries,
    get_tables,
    list_artifacts,
)
from kb_agent.benchmark import (
    analyze_failures,
    create_eval_suite,
    generate_case_study,
    get_eval_suite,
    list_eval_suites,
    run_benchmark,
)
from kb_agent.cli import main as cli_main
from kb_agent.embeddings import HashEmbeddingProvider, build_semantic_index, semantic_index_status
from kb_agent.eval import eval_facts, eval_memory, eval_review, eval_search
from kb_agent.fact_audit import audit_facts, fact_conflict_summary, get_fact_conflicts
from kb_agent.facts import extract_facts, fact_search, get_claims, get_entities, get_fact_graph, get_relations
from kb_agent.feedback import build_eval_set_from_feedback, eval_dashboard, list_feedback, put_feedback
from kb_agent.ingest import sync_directory
from kb_agent.insights import extract_doc_insights
from kb_agent.knowledge_graph import (
    build_knowledge_graph,
    export_knowledge_graph,
    get_graph_neighborhood,
    get_graph_report,
)
from kb_agent.llm import LLMError, generate_json_object, llm_payload_metadata, llm_status
from kb_agent.memory import compact_memory, put_memory_gated, remember_task, resume_task, search_memory
from kb_agent.models import ParsedBlock, ParsedDocument
from kb_agent.query import classify_query
from kb_agent.quality_baseline import latest_quality_baseline, run_quality_baseline
from kb_agent.query_log import list_query_logs, query_stats
from kb_agent.review import assemble_review, check_review_citations, draft_review
from kb_agent.search import build_search_report, get_evidence, search_documents, search_nodes
from kb_agent.search_profile import apply_search_profile, get_search_profile, list_search_profiles, tune_search
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
            self.assertIn("layout_blocks.json", names)
            self.assertIn("tables.json", names)
            self.assertIn("figures.json", names)
            self.assertIn("reference_sections.json", names)
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

    def test_llm_status_probe_is_sanitized(self) -> None:
        with mock.patch("kb_agent.config._ENV_LOADED", True), mock.patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "sk-test-secret",
                "DEEPSEEK_BASE_URL": "http://localhost:3000/v1",
                "DEEPSEEK_MODEL": "deepseek_v4",
                "DEEPSEEK_TEMPERATURE": "0",
                "DEEPSEEK_MAX_TOKENS": "300",
            },
            clear=True,
        ), mock.patch("kb_agent.llm._chat_completion_content", return_value="连接正常"):
            status = llm_status(probe=True)

        self.assertEqual(status["schema"], "llm_status.v1")
        self.assertTrue(status["configured"])
        self.assertTrue(status["reachable"])
        self.assertTrue(status["insecure_http"])
        self.assertEqual(status["model"], "deepseek_v4")
        self.assertNotIn("sk-test-secret", json.dumps(status, ensure_ascii=False))

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), mock.patch("kb_agent.config._ENV_LOADED", True), mock.patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "sk-test-secret",
                "DEEPSEEK_BASE_URL": "http://localhost:3000/v1",
                "DEEPSEEK_MODEL": "deepseek_v4",
            },
            clear=True,
        ), mock.patch("kb_agent.llm._chat_completion_content", return_value="连接正常"):
            cli_main(["llm-status", "--probe"])
        output = stdout.getvalue()
        self.assertIn("llm_status.v1", output)
        self.assertNotIn("sk-test-secret", output)

    def test_generate_json_object_repairs_and_retries_safely(self) -> None:
        env = {
            "DEEPSEEK_API_KEY": "sk-json-secret",
            "DEEPSEEK_BASE_URL": "http://localhost:3000/v1",
            "DEEPSEEK_MODEL": "deepseek_v4",
        }
        with mock.patch("kb_agent.config._ENV_LOADED", True), mock.patch.dict(os.environ, env, clear=True), mock.patch(
            "kb_agent.llm._chat_completion_content",
            side_effect=['{"ok": ', '{"ok": true}'],
        ):
            payload = generate_json_object("json only", "return ok")

        self.assertTrue(payload["ok"])
        metadata = llm_payload_metadata(payload)
        self.assertEqual(metadata["retry_count"], 1)
        self.assertEqual(metadata["first_error_type"], "truncated_json")
        self.assertNotIn("sk-json-secret", json.dumps(payload, ensure_ascii=False))

        with mock.patch("kb_agent.config._ENV_LOADED", True), mock.patch.dict(os.environ, env, clear=True), mock.patch(
            "kb_agent.llm._chat_completion_content",
            return_value='```json\n{"ok": true}\n```',
        ):
            fenced = generate_json_object("json only", "return ok")
        self.assertTrue(llm_payload_metadata(fenced)["repair_used"])

        with mock.patch("kb_agent.config._ENV_LOADED", True), mock.patch.dict(os.environ, env, clear=True), mock.patch(
            "kb_agent.llm._chat_completion_content",
            return_value='["not-object"]',
        ):
            with self.assertRaises(LLMError) as raised:
                generate_json_object("json only", "return object")
        self.assertEqual(raised.exception.error_type, "non_object_json")
        self.assertNotIn("not-object", str(raised.exception))

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

    def test_rule_based_fact_extraction_writes_artifacts_db_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            extract_doc_insights(db_path, doc_id, use_llm=False)

            result = extract_facts(db_path, doc_id, use_llm=False)

            self.assertEqual(result["schema"], "fact_extraction_result.v1")
            report = result["fact_report"]
            self.assertEqual(report["schema"], "fact_report.v1")
            self.assertEqual(report["status"], "partial")
            self.assertGreaterEqual(report["claim_count"], 3)
            self.assertGreaterEqual(report["entity_count"], 3)
            self.assertGreaterEqual(report["relation_count"], 3)

            claims = get_claims(db_path, doc_id)
            entities = get_entities(db_path, doc_id)
            relations = get_relations(db_path, doc_id)
            graph = get_fact_graph(db_path, doc_id)
            self.assertEqual(claims["schema"], "claims.v1")
            self.assertEqual(entities["schema"], "entities.v1")
            self.assertEqual(relations["schema"], "relations.v1")
            self.assertEqual(graph["schema"], "fact_graph.v1")

            claim_types = {item["type"] for item in claims["claims"]}
            self.assertIn("method", claim_types)
            self.assertIn("limitation", claim_types)
            entity_names = {item["name"] for item in entities["entities"]}
            self.assertIn("动态角色", entity_names)
            self.assertTrue(any("任务完成率" in name or "负载均衡" in name for name in entity_names))
            relation_types = {item["type"] for item in relations["relations"]}
            self.assertIn("cites", relation_types)
            self.assertIn("supports", relation_types)
            cite_relations = [item for item in relations["relations"] if item["type"] == "cites"]
            citation_map = get_citation_map(db_path, doc_id)
            self.assertGreaterEqual(len(cite_relations), len(citation_map["relations"]))
            self.assertEqual(report["entity_noise_filtered_count"], 0)

            for collection in (claims["claims"], entities["entities"], relations["relations"]):
                for item in collection:
                    self.assertEqual(item["doc_id"], doc_id)
                    self.assertTrue(item["version_id"])
                    self.assertTrue(item["node_id"])
                    self.assertIn("page_range", item)
                    self.assertGreaterEqual(item["confidence"], 0)
                    self.assertTrue(item["source"])

            search = fact_search(db_path, "动态角色任务规划", doc_ids=[doc_id], top_k=5)
            self.assertEqual(search["schema"], "fact_search.v1")
            self.assertGreaterEqual(search["count"], 1)
            self.assertTrue(any(item["fact_type"] in {"claim", "entity"} for item in search["items"]))

            search_report = build_search_report(db_path, "动态角色任务规划", doc_id=doc_id, top_k=3)
            self.assertIn("fact_matches", search_report)
            self.assertGreaterEqual(search_report["fact_matches"]["count"], 1)

            dashboard = eval_dashboard(db_path)
            self.assertIn("fact_coverage", dashboard)
            self.assertGreaterEqual(dashboard["fact_coverage"]["claim_count"], 3)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "extract-facts", doc_id, "--force", "--no-llm"])
            self.assertIn("fact_extraction_result.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "claims", doc_id])
            self.assertIn("claims.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "entities", doc_id])
            self.assertIn("entities.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "relations", doc_id])
            self.assertIn("relations.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "fact-graph", doc_id])
            self.assertIn("fact_graph.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "fact-search", "动态角色", "--doc-id", doc_id, "--type", "claim"])
            self.assertIn("fact_search.v1", stdout.getvalue())

    def test_table_content_artifacts_quality_and_fact_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_table_sample(Path(tmp))

            tables = get_tables(db_path, doc_id)
            self.assertEqual(tables["count"], 1)
            self.assertEqual(tables["tables"][0]["row_count"], 2)
            self.assertEqual(tables["tables"][0]["column_count"], 4)
            self.assertEqual(tables["tables"][0]["cell_count"], 12)
            self.assertEqual(tables["tables"][0]["source_parser"], "plain_text")

            table_content = get_table_content(db_path, doc_id)
            self.assertEqual(table_content["schema"], "table_content.v1")
            self.assertEqual(table_content["count"], 1)
            item = table_content["table_content"][0]
            self.assertEqual(item["headers"], ["方法", "任务完成率", "响应时间", "数据集"])
            self.assertEqual(item["rows"][0]["cells"][0]["text"], "基线方法")
            self.assertEqual(item["rows"][1]["cells"][1]["text"], "92%")
            self.assertEqual(item["source"], "table_rule")

            summaries = get_table_summaries(db_path, doc_id)
            summary = summaries["table_summaries"][0]
            self.assertIn("任务完成率", summary["metrics"])
            self.assertIn("本文方法", summary["methods"])
            self.assertIn("92%", summary["results"])

            quality = get_parse_quality(db_path, doc_id)
            self.assertEqual(quality["table_content_count"], 1)
            self.assertEqual(quality["table_parse_score"], 1.0)
            self.assertEqual(quality["table_warning_count"], 0)
            parse_report = get_parse_report(db_path, doc_id)
            self.assertEqual(parse_report["table_content_count"], 1)

            result = extract_facts(db_path, doc_id, force=True, use_llm=False)
            self.assertGreater(result["fact_report"]["table_backed_fact_count"], 0)
            entities = get_entities(db_path, doc_id)
            relations = get_relations(db_path, doc_id)
            table_entities = [item for item in entities["entities"] if item["source"] == "table_rule"]
            table_relations = [item for item in relations["relations"] if item["source"] == "table_rule"]
            self.assertTrue(any(item["type"] == "result" and "任务完成率" in item["name"] for item in table_entities))
            self.assertIn("reports_metric", {item["type"] for item in table_relations})
            self.assertIn("improves", {item["type"] for item in table_relations})
            for item in [*table_entities, *table_relations]:
                self.assertEqual(item["evidence"]["table_id"], "table_001")
                self.assertTrue(item["node_id"])

            table_search = fact_search(db_path, "任务完成率", doc_ids=[doc_id], source="table", min_confidence=0.5)
            self.assertGreaterEqual(table_search["count"], 1)
            self.assertTrue(all(item["source_kind"] == "table" for item in table_search["items"]))

            report = eval_facts(db_path, doc_ids=[doc_id])
            self.assertEqual(report["schema"], "fact_eval.v1")
            self.assertGreater(report["table_backed_fact_count"], 0)
            self.assertEqual(report["no_node_id_count"], 0)
            self.assertEqual(report["evidence_coverage_rate"], 1.0)

            search_report = build_search_report(db_path, "任务完成率", doc_id=doc_id, top_k=3)
            self.assertIn("table_backed_count", search_report["fact_matches"])

            dashboard = eval_dashboard(db_path)
            self.assertGreater(dashboard["fact_coverage"]["table_backed_fact_count"], 0)
            self.assertEqual(dashboard["latest_fact_eval"]["schema"], "fact_eval.v1")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "table-content", doc_id])
            self.assertIn("table_content.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "table-summaries", doc_id])
            self.assertIn("table_summaries.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main([
                    "--db",
                    str(db_path),
                    "fact-search",
                    "任务完成率",
                    "--doc-id",
                    doc_id,
                    "--source",
                    "table",
                    "--min-confidence",
                    "0.5",
                ])
            self.assertIn("source_kind", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "eval-facts", "--doc-id", doc_id])
            self.assertIn("fact_eval.v1", stdout.getvalue())

    def test_fact_audit_conflicts_dashboard_and_task_risk_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_fact_audit_samples(Path(tmp))

            audit = audit_facts(db_path, doc_ids=doc_ids, min_confidence=0.5)
            self.assertEqual(audit["schema"], "fact_audit.v1")
            self.assertEqual(audit["status"], "needs_review")
            self.assertGreaterEqual(audit["duplicate_group_count"], 1)
            self.assertGreaterEqual(audit["low_confidence_count"], 1)
            self.assertGreaterEqual(audit["no_evidence_count"], 1)
            self.assertGreaterEqual(audit["conflict_count"], 1)
            self.assertGreaterEqual(audit["table_text_mismatch_count"], 1)
            self.assertGreaterEqual(audit["citation_gap_count"], 1)
            self.assertTrue(Path(audit["path"]).exists())
            dumped = json.dumps(audit, ensure_ascii=False)
            self.assertNotIn("excerpt", dumped)
            self.assertNotIn("这是很长的论文正文", dumped)

            conflicts = get_fact_conflicts(db_path, doc_ids=doc_ids, severity="high", min_confidence=0.5)
            self.assertEqual(conflicts["schema"], "fact_conflicts.v1")
            self.assertGreaterEqual(conflicts["count"], 1)
            first = conflicts["conflicts"][0]
            for side in ("left", "right"):
                self.assertTrue(first[side]["doc_id"])
                self.assertTrue(first[side]["node_id"])
                self.assertIn("page_range", first[side])
                self.assertIn("fact_id", first[side])
                self.assertIn("confidence", first[side])
            self.assertIn("reason", first)

            summary = fact_conflict_summary(db_path, "任务完成率", doc_ids=doc_ids)
            self.assertEqual(summary["schema"], "fact_conflict_summary.v1")
            self.assertGreaterEqual(summary["conflict_count"], 1)

            comparison = compare_papers(db_path, "任务完成率方法对比", doc_ids=doc_ids, use_llm=False)
            matrix = comparison["comparison_matrix"]
            self.assertIn("fact_audit", matrix)
            self.assertTrue(any("fact_audit_conflicts" in warning for warning in matrix["warnings"]))
            evidence_strength = next(item for item in matrix["dimensions"] if item["id"] == "evidence_strength")
            self.assertTrue(any("fact_audit" in warning for warning in evidence_strength["warnings"]))

            review = generate_review_plan(db_path, "任务完成率研究综述", doc_ids=doc_ids, use_llm=False)
            outline = review["review_outline"]
            self.assertIn("fact_audit", outline)
            self.assertTrue(any("事实层存在" in item for item in outline["open_questions"]))

            case = generate_case_study(db_path, "任务完成率", doc_ids=doc_ids, compare_modes=["hybrid", "tree"], top_k=3)
            self.assertIn("fact_conflicts", case)
            self.assertGreaterEqual(case["fact_conflicts"]["conflict_count"], 1)

            dashboard = eval_dashboard(db_path)
            self.assertEqual(dashboard["latest_fact_audit"]["schema"], "fact_audit.v1")
            self.assertGreaterEqual(dashboard["latest_fact_audit"]["conflict_count"], 1)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "audit-facts", "--doc-id", doc_ids[0], "--doc-id", doc_ids[1]])
            self.assertIn("fact_audit.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "fact-conflicts", "--doc-id", doc_ids[0], "--doc-id", doc_ids[1], "--severity", "high"])
            self.assertIn("fact_conflicts.v1", stdout.getvalue())

    def test_claim_graph_navigation_exports_and_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_fact_audit_samples(Path(tmp))
            _insert_noisy_entity_for_graph(db_path, doc_ids[0])

            result = build_knowledge_graph(db_path, doc_ids=doc_ids, include_conflicts=True, min_confidence=0.5)

            self.assertEqual(result["knowledge_graph"]["schema"], "knowledge_graph.v1")
            graph = result["knowledge_graph"]
            node_types = {item["type"] for item in graph["nodes"]}
            edge_types = {item["type"] for item in graph["edges"]}
            self.assertIn("document", node_types)
            self.assertIn("claim", node_types)
            self.assertIn("evidence", node_types)
            self.assertIn("conflict", node_types)
            self.assertIn("has_claim", edge_types)
            self.assertIn("backed_by", edge_types)
            self.assertIn("conflicts_with", edge_types)
            self.assertGreaterEqual(result["graph_report"]["conflict_count"], 1)
            self.assertTrue(Path(result["knowledge_graph_path"]).exists())

            dumped = json.dumps(graph, ensure_ascii=False)
            self.assertNotIn("excerpt", dumped)
            self.assertNotIn("这是很长的论文正文", dumped)

            neighborhood = get_graph_neighborhood(db_path, "claim_a_positive", graph_id=result["graph_id"], depth=2)
            self.assertEqual(neighborhood["schema"], "knowledge_graph_neighborhood.v1")
            self.assertGreaterEqual(neighborhood["node_count"], 2)
            self.assertTrue(any(item["type"] == "evidence" for item in neighborhood["nodes"]))

            mermaid = export_knowledge_graph(db_path, result["graph_id"], format="mermaid")
            self.assertTrue(Path(mermaid["path"]).exists())
            self.assertIn("graph TD", Path(mermaid["path"]).read_text(encoding="utf-8"))

            html = export_knowledge_graph(db_path, result["graph_id"], format="html")
            self.assertTrue(Path(html["path"]).exists())
            html_text = Path(html["path"]).read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", html_text)
            self.assertNotIn("excerpt", html_text)

            report = get_graph_report(db_path, result["graph_id"])
            self.assertEqual(report["schema"], "knowledge_graph_report.v1")
            self.assertGreaterEqual(report["evidence_coverage_rate"], 0.5)
            self.assertGreaterEqual(report["noisy_entity_count"], 1)
            self.assertNotIn("No.", {item["label"] for item in report["top_entities"]})

            comparison = compare_papers(db_path, "任务完成率方法对比", doc_ids=doc_ids, use_llm=False)
            self.assertIn("claim_graph", comparison["comparison_matrix"])
            self.assertTrue(any("claim_graph" in warning for warning in comparison["comparison_matrix"]["warnings"]))

            review = generate_review_plan(db_path, "任务完成率研究综述", doc_ids=doc_ids, use_llm=False)
            self.assertIn("claim_graph", review["review_outline"])
            self.assertTrue(any("Claim Graph" in item for item in review["review_outline"]["open_questions"]))

            case = generate_case_study(db_path, "任务完成率", doc_ids=doc_ids, compare_modes=["hybrid"], top_k=3)
            self.assertIn("claim_graph", case)
            self.assertGreaterEqual(case["claim_graph"]["conflict_count"], 1)

            dashboard = eval_dashboard(db_path)
            self.assertIn("latest_claim_graph", dashboard)
            self.assertGreaterEqual(dashboard["latest_claim_graph"]["conflict_count"], 1)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "graph-build", "--doc-id", doc_ids[0], "--doc-id", doc_ids[1], "--include-conflicts"])
            self.assertIn("knowledge_graph_build_result.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "graph-neighborhood", "claim_a_positive", "--graph-id", result["graph_id"]])
            self.assertIn("knowledge_graph_neighborhood.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "graph-export", result["graph_id"], "--format", "mermaid"])
            self.assertIn("knowledge_graph_export.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "graph-report", result["graph_id"]])
            self.assertIn("knowledge_graph_report.v1", stdout.getvalue())

    def test_quality_baseline_runs_real_corpus_summary_without_sensitive_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "robot.txt").write_text(
                "摘要：本文研究服务机器人任务规划方法，解决任务分解和工具调用问题。\n"
                "关键词：服务机器人；任务规划；工具调用\n\n"
                "1 方法设计\n"
                "本文提出结合大语言模型和技能库的任务规划框架。\n\n"
                "2 实验结果\n"
                "实验结果表明，该方法提升任务成功率和响应时间表现。\n\n"
                "结论\n"
                "本文仍存在真实家庭环境验证不足的局限。\n",
                encoding="utf-8",
            )
            (papers / "agents.txt").write_text(
                "摘要：本文研究多智能体系统中的分布式任务规划。\n"
                "关键词：多智能体；分布式任务规划；协同调度\n\n"
                "1 方法设计\n"
                "本文提出动态任务重分配算法和负载均衡模型。\n\n"
                "2 实验结果\n"
                "实验结果表明，该方法提升任务完成率和系统鲁棒性。\n\n"
                "结论\n"
                "本文仍存在复杂通信约束验证不足的局限。\n",
                encoding="utf-8",
            )

            with mock.patch("kb_agent.quality_baseline.importlib.util.find_spec", return_value=None):
                result = run_quality_baseline(db_path, papers, use_llm=False, top_k=3)

            self.assertEqual(result["schema"], "quality_baseline.v1")
            self.assertEqual(result["doc_count"], 2)
            self.assertEqual(result["run_kind"], "test_fixture")
            self.assertFalse(result["is_real_corpus"])
            self.assertTrue(result["corpus_fingerprint"])
            self.assertEqual(result["fact_audit_delta"]["schema"], "fact_audit_delta.v1")
            self.assertTrue(Path(result["json_path"]).exists())
            self.assertTrue(Path(result["md_path"]).exists())
            self.assertTrue(Path(result["html_path"]).exists())
            self.assertEqual(result["embedding"]["hash"]["status"], "completed")
            self.assertEqual(result["embedding"]["sentence_transformers"]["status"], "skipped")
            self.assertEqual(result["benchmark"]["schema"], "benchmark_report.v1")
            self.assertIn(result["benchmark"]["best_mode_by_score"], {"fts", "hybrid", "tree"})
            self.assertEqual(result["tree_search"]["schema"], "tree_search_baseline.v1")
            self.assertEqual(result["tasks"]["schema"], "task_quality_baseline.v1")
            self.assertTrue(result["tasks"]["compare"].get("task_id"))
            self.assertTrue(result["tasks"]["review"].get("task_id"))
            self.assertEqual(result["memory"]["schema"], "memory_eval.v1")
            self.assertIn("claim_graph", result)
            providers = {item["provider"]: item for item in result["parser_comparison"]["providers"]}
            self.assertEqual(providers["docling"]["status"], "skipped")
            self.assertEqual(providers["grobid"]["status"], "skipped")

            html = Path(result["html_path"]).read_text(encoding="utf-8")
            self.assertIn("Quality Baseline", html)
            self.assertNotIn("excerpt", html)
            self.assertNotIn("evidence packet", html.lower())
            self.assertNotIn("本文提出结合大语言模型", html)

            latest = latest_quality_baseline(limit=1)
            self.assertEqual(latest["schema"], "quality_baseline_latest.v1")
            self.assertGreaterEqual(latest["count"], 1)
            filtered_latest = latest_quality_baseline(limit=1, corpus=papers)
            self.assertEqual(filtered_latest["count"], 1)
            self.assertEqual(filtered_latest["items"][0]["corpus_path"], str(papers.resolve()))
            self.assertEqual(filtered_latest["items"][0]["run_kind"], "test_fixture")
            real_filtered = latest_quality_baseline(limit=1, corpus=papers, real_only=True)
            self.assertEqual(real_filtered["count"], 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), mock.patch(
                "kb_agent.quality_baseline.importlib.util.find_spec",
                return_value=None,
            ):
                cli_main(["--db", str(db_path), "quality-baseline", str(papers), "--top-k", "2"])
            self.assertIn("quality_baseline.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["latest-quality-baseline", "--limit", "1", "--corpus", str(papers)])
            latest_stdout = stdout.getvalue()
            self.assertIn("quality_baseline_latest.v1", latest_stdout)
            self.assertIn("run_kind", latest_stdout)

    def test_quality_baseline_with_llm_records_sanitized_llm_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "robot.txt").write_text(
                "摘要：本文研究服务机器人任务规划方法。\n"
                "关键词：服务机器人；任务规划\n\n"
                "1 方法\n本文提出大语言模型任务规划框架。\n\n"
                "2 实验\n实验结果表明任务成功率提升。\n",
                encoding="utf-8",
            )
            (papers / "agents.txt").write_text(
                "摘要：本文研究多智能体分布式任务规划。\n"
                "关键词：多智能体；任务分配\n\n"
                "1 方法\n本文提出动态任务分配算法。\n\n"
                "2 实验\n实验结果表明鲁棒性提升。\n",
                encoding="utf-8",
            )
            status_payload = {
                "schema": "llm_status.v1",
                "provider": "deepseek",
                "configured": True,
                "reachable": True,
                "probe": True,
                "base_url": "http://localhost:3000/v1",
                "model": "deepseek_v4",
                "temperature": 0,
                "max_tokens": 300,
                "insecure_http": True,
                "error": "",
                "response_sample": "连接正常",
            }
            query_payload = {"intent": "experiment", "focus_terms": ["实验"], "preferred_node_types": ["paragraph"], "target_sections": ["实验"], "warnings": []}
            insight_payload = {
                "items": [{"title": "任务规划框架", "type": "method", "claim": "提出任务规划框架。", "evidence": ["N1"], "confidence": 0.8}],
                "limitations": [],
                "open_questions": [],
                "warnings": [],
            }
            fact_payload = {
                "claims": [{"type": "method", "text": "提出任务规划框架。", "evidence": ["N1"], "confidence": 0.8}],
                "entities": [{"type": "method", "name": "任务规划框架", "evidence": ["N1"], "confidence": 0.8}],
                "relations": [{"type": "uses", "subject": "任务规划框架", "object": "任务规划", "evidence": ["N1"], "confidence": 0.7}],
                "warnings": [],
            }
            tree_payload = {"selected_node_ids": [], "rationale": ["使用规则候选即可。"], "warnings": []}

            with mock.patch("kb_agent.quality_baseline.importlib.util.find_spec", return_value=None), mock.patch(
                "kb_agent.quality_baseline.llm_status",
                return_value=status_payload,
            ), mock.patch("kb_agent.query.generate_json_object", return_value=query_payload), mock.patch(
                "kb_agent.tree_search.generate_json_object",
                return_value=tree_payload,
            ), mock.patch("kb_agent.insights.generate_json_object", return_value=insight_payload), mock.patch(
                "kb_agent.facts.generate_json_object",
                return_value=fact_payload,
            ), mock.patch("kb_agent.tasks.generate_json_object", return_value={"warnings": []}):
                result = run_quality_baseline(db_path, papers, use_llm=True, top_k=2)

            self.assertEqual(result["llm_baseline"]["schema"], "llm_quality_baseline.v1")
            self.assertEqual(result["llm_baseline"]["status"], "completed")
            self.assertEqual(result["llm_status"]["model"], "deepseek_v4")
            self.assertEqual(result["tree_search"]["llm_enabled"], True)
            self.assertEqual(len(result["tree_search"]["llm_items"]), 2)
            self.assertGreaterEqual(result["llm_baseline"]["insights_and_facts"]["llm_used_count"], 1)
            self.assertIn("review_fallback_mode", result["llm_baseline"]["tasks"])
            self.assertIn("review_retry_count", result["llm_baseline"]["tasks"])
            self.assertIn("review_partial_reasons", result["llm_baseline"]["tasks"])
            self.assertIn("llm_diagnostics", result["tasks"]["review"])
            self.assertIn("comparison_summary", result["tree_search"])
            html = Path(result["html_path"]).read_text(encoding="utf-8")
            self.assertNotIn("sk-", html)
            self.assertNotIn("excerpt", html)

    def test_llm_fact_extraction_and_require_llm_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            node_id = get_artifact(db_path, doc_id, "node_index.jsonl")["content"][2]["node_id"]
            payload = {
                "claims": [
                    {
                        "type": "method",
                        "text": "提出动态角色发现机制以提升任务分解效率。",
                        "evidence": [node_id],
                        "confidence": 0.84,
                    },
                    {
                        "type": "result",
                        "text": "实验结果表明，" + "该方法能够提升任务完成率。" * 20,
                        "evidence": [node_id],
                        "confidence": 0.76,
                    }
                ],
                "entities": [
                    {
                        "type": "method",
                        "name": "动态角色发现机制",
                        "aliases": ["动态角色"],
                        "evidence": [node_id],
                        "confidence": 0.82,
                    },
                    {
                        "type": "noise",
                        "name": "A",
                        "evidence": [node_id],
                        "confidence": 0.4,
                    },
                    {
                        "type": "term",
                        "name": "No.",
                        "evidence": [node_id],
                        "confidence": 0.4,
                    },
                    {
                        "type": "term",
                        "name": "微调数据集则涵盖基于输入的总体任务规划文本进",
                        "evidence": [node_id],
                        "confidence": 0.4,
                    }
                ],
                "relations": [
                    {
                        "type": "uses",
                        "subject": "动态角色发现机制",
                        "object": "任务分解",
                        "evidence": [node_id],
                        "confidence": 0.78,
                    }
                ],
                "warnings": [],
            }

            with mock.patch("kb_agent.facts.generate_json_object", return_value=payload):
                result = extract_facts(db_path, doc_id, force=True, use_llm=True)

            self.assertEqual(result["fact_report"]["status"], "extracted")
            self.assertEqual(result["fact_report"]["source"], "llm")
            self.assertTrue(result["fact_report"]["llm_used"])
            self.assertGreaterEqual(result["fact_report"]["noise_filtered_count"], 2)
            self.assertIn("entity_noise_filtered_count", result["fact_report"])
            self.assertEqual(result["fact_report"]["long_claim_trimmed_count"], 1)
            self.assertEqual(result["claims"]["claims"][0]["confidence"], 0.84)
            self.assertLessEqual(len(result["claims"]["claims"][1]["text"]), 225)
            self.assertEqual(result["entities"]["entities"][0]["aliases"], ["动态角色"])
            entity_names = {item["name"] for item in result["entities"]["entities"]}
            self.assertNotIn("No.", entity_names)
            self.assertFalse(any("总体任务规划文本进" in name for name in entity_names))

        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            with mock.patch("kb_agent.facts.generate_json_object", side_effect=LLMError("boom")):
                with self.assertRaises(LLMError):
                    extract_facts(db_path, doc_id, force=True, use_llm=True, require_llm=True)
            with self.assertRaises(FileNotFoundError):
                get_claims(db_path, doc_id)

            with mock.patch("kb_agent.facts.generate_json_object", side_effect=LLMError("boom")):
                fallback = extract_facts(db_path, doc_id, force=True, use_llm=True)
            self.assertEqual(fallback["fact_report"]["status"], "partial")
            self.assertTrue(any("llm_unavailable:boom" in item for item in fallback["fact_report"]["warnings"]))

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
            self.assertIn("evidence_quality", matrix)
            self.assertIn("duplicate_evidence_removed", matrix)
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
            self.assertIn("evidence_quality", outline)
            self.assertIn("duplicate_evidence_removed", outline)
            self.assertIn("review_partial_reasons", outline)
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
            self.assertEqual(result["comparison_matrix"]["llm_diagnostics"]["mode"], "full_json")
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

    def test_review_llm_section_recovery_and_require_llm_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            full_error = LLMError("DeepSeek JSON parse failed: truncated_json", error_type="truncated_json", metadata={"retry_count": 1})
            section_payloads = [
                {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "purpose": section["purpose"],
                    "paper_ids": doc_ids,
                    "evidence": [],
                    "warnings": [],
                }
                for section in [
                    {"section_id": "background_problem", "title": "研究背景与问题定义", "purpose": "界定主题。"},
                    {"section_id": "method_paradigms", "title": "方法范式与系统框架", "purpose": "归纳方法。"},
                    {"section_id": "coordination_mechanisms", "title": "任务分解、分配与协同机制", "purpose": "整理机制。"},
                    {"section_id": "evaluation_evidence", "title": "实验评测与证据强度", "purpose": "比较证据。"},
                    {"section_id": "limitations_future", "title": "局限性与未来方向", "purpose": "汇总局限。"},
                ]
            ]
            with mock.patch("kb_agent.tasks.generate_json_object", side_effect=[full_error, *section_payloads]):
                result = generate_review_plan(
                    db_path,
                    "任务规划方法研究综述",
                    doc_ids=doc_ids,
                    use_llm=True,
                )

            outline = result["review_outline"]
            self.assertEqual(outline["source"], "llm_section")
            self.assertEqual(outline["llm_diagnostics"]["mode"], "section_json")
            self.assertEqual(outline["llm_diagnostics"]["retry_count"], 1)
            self.assertEqual(outline["llm_diagnostics"]["fallback_sections"], [])
            self.assertNotIn("rule_based_review_plan", outline["warnings"])
            self.assertEqual(len(outline["sections"]), 5)

        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_ids = _sync_compare_samples(Path(tmp))
            state_root = db_path.parent / ".kb_state"
            errors = [
                LLMError("DeepSeek JSON parse failed: truncated_json", error_type="truncated_json", metadata={"retry_count": 1}),
                *[LLMError("DeepSeek JSON parse failed: invalid_json", error_type="invalid_json") for _ in range(5)],
            ]
            with mock.patch("kb_agent.tasks.generate_json_object", side_effect=errors):
                with self.assertRaises(LLMError):
                    generate_review_plan(
                        db_path,
                        "任务规划方法研究综述",
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
            status = semantic_index_status(db_path, provider="hash")
            self.assertEqual(status["schema"], "semantic_index_status.v1")
            self.assertTrue(status["ready"])
            self.assertEqual(status["missing_node_embeddings"], 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "embed", "--provider", "hash", "--status"])
            self.assertIn("semantic_index_status.v1", stdout.getvalue())

    def test_sentence_transformers_model_option_and_incremental_status(self) -> None:
        class FakeSentenceTransformer:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def get_sentence_embedding_dimension(self) -> int:
                return 3

            def encode(self, texts, normalize_embeddings=True, batch_size=16):  # type: ignore[no-untyped-def]
                del normalize_embeddings, batch_size
                return [[1.0, 0.0, 0.0] if "动态角色" in text else [0.0, 1.0, 0.0] for text in texts]

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            db_path, _ = _sync_insight_sample(Path(tmp))
            first = build_semantic_index(
                db_path,
                provider="sentence-transformers",
                model="fake-model-v1",
                batch_size=2,
                force=True,
            )
            self.assertEqual(first["provider"], "sentence-transformers")
            self.assertEqual(first["model"], "fake-model-v1")
            self.assertEqual(first["dim"], 3)
            self.assertGreater(first["indexed_nodes"], 0)

            second = build_semantic_index(
                db_path,
                provider="sentence-transformers",
                model="fake-model-v1",
                batch_size=2,
            )
            self.assertEqual(second["indexed_nodes"], 0)
            self.assertGreater(second["skipped_nodes"], 0)

            status = semantic_index_status(db_path, provider="sentence-transformers", model="fake-model-v1")
            self.assertTrue(status["ready"])
            self.assertEqual(status["model"], "fake-model-v1")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main([
                    "--db",
                    str(db_path),
                    "embed",
                    "--provider",
                    "sentence-transformers",
                    "--model",
                    "fake-model-v1",
                    "--status",
                ])
            self.assertIn("fake-model-v1", stdout.getvalue())

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
            self.assertIn("selected_reason", trace["evidence"][0])
            self.assertIn("resolved_intent", trace)
            self.assertTrue(trace["expanded_nodes"])
            self.assertTrue(trace["selected_paths"])
            self.assertIn("score_components", trace["selected_paths"][0])

            results = search_nodes(db_path, "这篇论文的方法设计是什么？", doc_id=doc_id, top_k=3, search_mode="tree")
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("tree:value", results[0].rank_reason)

            report = build_search_report(db_path, "这篇论文的方法设计是什么？", doc_id=doc_id, top_k=3, search_mode="tree")
            self.assertEqual(report["effective_search_mode"], "tree")
            self.assertIn("tree_search_trace", report)
            self.assertTrue(report["tree_search_trace"]["query_profile"])
            self.assertTrue(report["tree_search_trace"]["selected_paths"])

            multi_report = build_search_report(db_path, "这篇论文的方法设计是什么？", top_k=3, search_mode="tree")
            multi_trace = multi_report["tree_search_trace"]
            self.assertTrue(multi_trace["query_profile"])
            self.assertTrue(multi_trace["expanded_nodes"])
            self.assertTrue(multi_trace["selected_paths"])
            self.assertTrue(multi_trace["evidence"])

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
            query_payload = {"intent": "experiment", "focus_terms": ["实验"], "preferred_node_types": ["paragraph"], "target_sections": ["实验"], "warnings": []}
            with mock.patch("kb_agent.query.generate_json_object", return_value=query_payload), mock.patch(
                "kb_agent.tree_search.generate_json_object",
                return_value=payload,
            ):
                trace = tree_search(db_path, doc_id, "实验结果如何？", budget=3, use_llm=True)

            self.assertEqual(trace["evidence"][0]["node_id"], target_id)
            self.assertEqual(trace["llm_decisions"]["selected_node_ids"], [target_id])
            self.assertTrue(trace["llm_used"])
            self.assertEqual(trace["llm_selected_count"], 1)
            self.assertEqual(trace["llm_warning_count"], 0)
            self.assertFalse(trace["llm_error"])

            with mock.patch("kb_agent.query.generate_json_object", return_value=query_payload), mock.patch(
                "kb_agent.tree_search.generate_json_object",
                side_effect=LLMError("boom"),
            ):
                fallback = tree_search(db_path, doc_id, "实验结果如何？", budget=2, use_llm=True)
            self.assertIn("llm_unavailable:boom", fallback["warnings"])
            self.assertFalse(fallback["llm_used"])
            self.assertEqual(fallback["fallback_reason"], "llm_unavailable:boom")
            self.assertTrue(fallback["evidence"])

            with mock.patch("kb_agent.query.generate_json_object", return_value=query_payload), mock.patch(
                "kb_agent.tree_search.generate_json_object",
                side_effect=LLMError("boom"),
            ):
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

    def test_eval_suite_benchmark_failure_analysis_and_case_study(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, doc_id = _sync_insight_sample(root)
            build_semantic_index(db_path, force=True, provider="hash")
            extract_doc_insights(db_path, doc_id, use_llm=False)
            extract_facts(db_path, doc_id, force=True, use_llm=False)
            node_index = get_artifact(db_path, doc_id, "node_index.jsonl")["content"]
            target_node = next(node for node in node_index if "动态角色发现机制" in node.get("text", ""))
            suite_name = f"paper_core_{root.name}"
            suite_json = root / "suite.json"
            suite_json.write_text(
                json.dumps(
                    {
                        "schema": "eval_suite_seed.v1",
                        "queries": [
                            {
                                "query": "动态角色任务规划方法",
                                "intent": "method",
                                "category": "core_retrieval",
                                "expected_doc_ids": [doc_id],
                                "expected_node_ids": [target_node["node_id"]],
                                "expected_node_keywords": ["动态角色"],
                            },
                            {
                                "query": "不存在的严格节点",
                                "intent": "qa",
                                "category": "expected_failure",
                                "expected_doc_ids": [doc_id],
                                "expected_node_ids": ["node_missing_for_test"],
                                "expected_node_keywords": ["不存在关键词"],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            suite = create_eval_suite(db_path, suite_name, input_json=suite_json, doc_ids=[doc_id])
            self.assertEqual(suite["schema"], "eval_suite.v1")
            self.assertEqual(suite["query_count"], 3)
            self.assertTrue(Path(suite["path"]).exists())
            self.assertTrue(any(item["name"] == suite["name"] for item in list_eval_suites()["items"]))
            self.assertEqual(get_eval_suite(suite_name)["suite_id"], suite["suite_id"])

            benchmark = run_benchmark(db_path, suite_name, compare_modes=["hybrid", "tree", "fts"], top_k=3)
            self.assertEqual(benchmark["schema"], "benchmark_report.v1")
            self.assertEqual(set(benchmark["mode_results"].keys()), {"hybrid", "tree", "fts"})
            self.assertTrue(Path(benchmark["path"]).exists())
            self.assertTrue(Path(benchmark["md_path"]).exists())
            self.assertIn(benchmark["best_mode_by_score"], {"hybrid", "tree", "fts"})
            dumped_benchmark = json.dumps(benchmark, ensure_ascii=False)
            self.assertNotIn("snippet", dumped_benchmark)
            self.assertNotIn("excerpt", dumped_benchmark)
            self.assertNotIn("实验结果表明，该方法在任务完成率", dumped_benchmark)

            failure = analyze_failures(db_path, benchmark["benchmark_id"])
            self.assertEqual(failure["schema"], "failure_analysis.v1")
            self.assertGreaterEqual(failure["failure_count"], 1)
            self.assertIn("node_recall_miss", failure["reason_counts"])
            self.assertTrue(failure["next_actions"])

            case = generate_case_study(
                db_path,
                "动态角色任务规划方法",
                doc_ids=[doc_id],
                compare_modes=["hybrid", "tree"],
                top_k=3,
            )
            self.assertEqual(case["schema"], "case_study.v1")
            self.assertEqual(set(case["mode_reports"].keys()), {"hybrid", "tree"})
            self.assertEqual(case["query_profile"]["intent"], "method")
            self.assertTrue(Path(case["path"]).exists())
            self.assertTrue(Path(case["md_path"]).exists())
            dumped_case = json.dumps(case, ensure_ascii=False)
            self.assertNotIn("snippet", dumped_case)
            self.assertNotIn("excerpt", dumped_case)
            self.assertNotIn("实验结果表明，该方法在任务完成率", dumped_case)

            dashboard = eval_dashboard(db_path)
            self.assertTrue(dashboard["latest_benchmarks"])
            self.assertTrue(dashboard["latest_failure_analyses"])
            self.assertTrue(dashboard["latest_case_studies"])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "eval-suite", "create", f"{suite_name}_cli", "--input-json", str(suite_json)])
            self.assertIn("eval_suite.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "eval-suite", "list"])
            self.assertIn("eval_suite_list.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "eval-suite", "show", suite_name])
            self.assertIn("eval_suite.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "benchmark", suite_name, "--compare-modes", "hybrid,tree", "--top-k", "3"])
            benchmark_stdout = stdout.getvalue()
            self.assertIn("benchmark_report.v1", benchmark_stdout)
            cli_benchmark_id = json.loads(benchmark_stdout)["benchmark_id"]

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "analyze-failures", cli_benchmark_id])
            self.assertIn("failure_analysis.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main([
                    "--db",
                    str(db_path),
                    "case-study",
                    "动态角色任务规划方法",
                    "--doc-id",
                    doc_id,
                    "--compare-modes",
                    "hybrid,tree",
                ])
            self.assertIn("case_study.v1", stdout.getvalue())

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

    def test_feedback_records_expected_targets_and_builds_eval_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path, doc_id = _sync_insight_sample(root)
            build_semantic_index(db_path, force=True, provider="hash")
            node_index = get_artifact(db_path, doc_id, "node_index.jsonl")["content"]
            target_node = next(node for node in node_index if "动态角色发现机制" in node.get("text", ""))

            result = put_feedback(
                db_path,
                query="动态角色任务规划",
                operation="ask",
                rating=5,
                label="good",
                comment="树搜索命中了方法章节。",
                expected_doc_ids=[doc_id],
                expected_node_ids=[target_node["node_id"]],
                expected_keywords=["动态角色"],
                preferred_search_mode="tree",
            )
            self.assertEqual(result["schema"], "feedback_write.v1")
            self.assertEqual(result["comment_status"], "accepted")

            feedback = list_feedback(db_path, limit=5, label="good")
            self.assertEqual(feedback["schema"], "feedback_list.v1")
            self.assertEqual(feedback["count"], 1)
            self.assertEqual(feedback["items"][0]["expected_doc_ids"], [doc_id])

            eval_path = root / "feedback_eval.json"
            eval_set = build_eval_set_from_feedback(db_path, output_path=eval_path, min_rating=4)
            self.assertEqual(eval_set["schema"], "search_eval_set.v1")
            self.assertEqual(eval_set["query_count"], 1)
            self.assertTrue(eval_path.exists())

            report = eval_search(db_path, eval_path, search_mode="hybrid", top_k=3, compare_modes=["hybrid", "tree", "fts"])
            self.assertEqual(report["eval_set_schema"], "search_eval_set.v1")
            self.assertEqual(set(report["mode_results"].keys()), {"hybrid", "tree", "fts"})

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main([
                    "--db",
                    str(db_path),
                    "feedback-put",
                    "动态角色任务规划",
                    "--operation",
                    "ask",
                    "--rating",
                    "4",
                    "--label",
                    "good",
                    "--expected-doc-id",
                    doc_id,
                    "--expected-node-id",
                    target_node["node_id"],
                    "--expected-keyword",
                    "动态角色",
                    "--preferred-search-mode",
                    "tree",
                ])
            self.assertIn("feedback_write.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "feedback-list", "--label", "good"])
            self.assertIn("feedback_list.v1", stdout.getvalue())

            cli_eval_path = root / "cli_feedback_eval.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "feedback-to-eval", str(cli_eval_path), "--min-rating", "4"])
            self.assertIn("search_eval_set.v1", stdout.getvalue())
            self.assertTrue(cli_eval_path.exists())

    def test_feedback_sanitizes_paper_assets_and_dashboard_reports_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, doc_id = _sync_insight_sample(Path(tmp))
            result = put_feedback(
                db_path,
                query="动态角色任务规划",
                operation="tree-search",
                rating=1,
                label="missing_evidence",
                comment="node_id=node_x page_range=[1,2] excerpt=这是一段论文正文，不能进入反馈评论。",
                expected_doc_ids=[doc_id],
                expected_keywords=["动态角色"],
                preferred_search_mode="hybrid",
            )
            self.assertEqual(result["comment_status"], "rejected")
            self.assertIn("comment_rejected:paper_asset_boundary", result["warnings"])

            feedback = list_feedback(db_path, limit=5)
            self.assertEqual(feedback["items"][0]["comment"], "")
            dumped = json.dumps(feedback, ensure_ascii=False)
            self.assertNotIn("这是一段论文正文", dumped)

            stats = query_stats(db_path)
            self.assertEqual(stats["feedback_count"], 1)
            self.assertEqual(stats["low_rating_count"], 1)
            self.assertIn("missing_evidence", stats["feedback_label_counts"])

            dashboard = eval_dashboard(db_path)
            self.assertEqual(dashboard["schema"], "eval_dashboard.v1")
            self.assertTrue(Path(dashboard["path"]).exists())
            self.assertTrue(Path(dashboard["json_path"]).exists())
            self.assertEqual(dashboard["feedback_summary"]["low_rating_count"], 1)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "eval-dashboard"])
            self.assertIn("eval_dashboard.v1", stdout.getvalue())

    def test_tune_search_profile_and_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_dir = root / "profiles"
            active_profile_path = profile_dir / "active.json"
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
                            "intent": "method",
                            "expected_doc_ids": [doc_id],
                            "expected_node_ids": [target_node["node_id"]],
                            "expected_node_keywords": ["动态角色"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with mock.patch("kb_agent.search_profile.PROFILE_DIR", profile_dir), \
                mock.patch("kb_agent.search_profile.ACTIVE_PROFILE_PATH", active_profile_path):
                tuning = tune_search(
                    db_path,
                    queries_path,
                    compare_modes=["hybrid", "tree", "fts"],
                    top_k=3,
                    save_profile="paper-v1",
                )
                self.assertEqual(tuning["schema"], "search_tuning.v1")
                self.assertTrue(tuning["mode_rankings"])
                self.assertIn("method", tuning["intent_modes"])
                self.assertIn("saved_profile", tuning)

                applied = apply_search_profile("paper-v1")
                self.assertEqual(applied["schema"], "search_profile_apply.v1")
                self.assertEqual(get_search_profile("active")["name"], "paper-v1")
                profiles = list_search_profiles()
                self.assertGreaterEqual(profiles["count"], 1)

                report = build_search_report(db_path, "动态角色任务规划", top_k=3, search_mode="auto")
                self.assertEqual(report["requested_search_mode"], "auto")
                self.assertIn(report["resolved_search_mode"], {"hybrid", "tree", "fts"})
                self.assertEqual(report["auto_resolution"]["profile_name"], "paper-v1")

                results = search_nodes(db_path, "动态角色任务规划", top_k=3, search_mode="auto")
                self.assertTrue(results)
                self.assertIn("auto:paper-v1", results[0].rank_reason)

                answer = answer_query(db_path, "动态角色任务规划", top_k=3, use_llm=False, search_mode="auto")
                self.assertEqual(answer["search_mode"], "auto")
                self.assertIn(answer["resolved_search_mode"], {"hybrid", "tree", "fts"})

                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    cli_main([
                        "--db",
                        str(db_path),
                        "tune-search",
                        str(queries_path),
                        "--compare-modes",
                        "hybrid,tree,fts",
                        "--save-profile",
                        "paper-cli",
                    ])
                self.assertIn("search_tuning.v1", stdout.getvalue())

                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    cli_main(["search-profile", "apply", "paper-cli"])
                self.assertIn("search_profile_apply.v1", stdout.getvalue())

    def test_auto_mode_falls_back_without_active_profile_and_dashboard_html_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_dir = root / "profiles"
            active_profile_path = profile_dir / "active.json"
            db_path, doc_id = _sync_insight_sample(root)
            with mock.patch("kb_agent.search_profile.PROFILE_DIR", profile_dir), \
                mock.patch("kb_agent.search_profile.ACTIVE_PROFILE_PATH", active_profile_path):
                report = build_search_report(db_path, "动态角色任务规划", top_k=2, search_mode="auto")
                self.assertEqual(report["requested_search_mode"], "auto")
                self.assertEqual(report["resolved_search_mode"], "hybrid")
                self.assertIn("auto_profile_missing", report["warnings"])

                put_feedback(
                    db_path,
                    query="动态角色任务规划",
                    operation="ask",
                    rating=2,
                    label="missing_evidence",
                    comment="node_id=node_x page_range=[1,2] excerpt=这是一段论文正文，不能进入 dashboard。",
                    expected_doc_ids=[doc_id],
                    expected_keywords=["动态角色"],
                )
                dashboard = eval_dashboard(db_path, output_format="html")
                self.assertTrue(Path(dashboard["html_path"]).exists())
                html = Path(dashboard["html_path"]).read_text(encoding="utf-8")
                self.assertIn("KB Eval Dashboard", html)
                self.assertIn("missing_evidence", html)
                self.assertNotIn("这是一段论文正文", html)
                self.assertNotIn("excerpt=", html)

    def test_opencode_observer_plugin_is_static_valid_and_sanitizes_sensitive_keys(self) -> None:
        plugin_path = Path(".opencode/plugins/kb-observer/index.mjs")
        self.assertTrue(plugin_path.exists())
        content = plugin_path.read_text(encoding="utf-8")
        self.assertIn("tool.execute.after", content)
        self.assertIn("experimental.session.compacting", content)
        self.assertIn("SENSITIVE_KEYS", content)
        self.assertIn("kb_put_feedback", content)
        self.assertIn("kb_tune_search", content)
        self.assertIn("kb_apply_search_profile", content)
        self.assertIn("kb_audit_facts", content)
        self.assertIn("kb_get_fact_conflicts", content)
        self.assertIn("kb_build_knowledge_graph", content)
        self.assertIn("kb_get_graph_neighborhood", content)
        self.assertIn("kb_run_quality_baseline", content)
        self.assertIn("kb_run_benchmark", content)
        self.assertIn("kb_analyze_failures", content)
        self.assertIn("kb_generate_case_study", content)
        self.assertIn("snippet", content)
        self.assertIn("feedback_hint", content)
        mcp_content = Path("kb_agent/mcp_server.py").read_text(encoding="utf-8")
        self.assertIn("kb_audit_facts", mcp_content)
        self.assertIn("kb_get_fact_conflicts", mcp_content)
        self.assertIn("kb_build_knowledge_graph", mcp_content)
        self.assertIn("kb_get_graph_neighborhood", mcp_content)
        self.assertIn("kb_export_knowledge_graph", mcp_content)
        self.assertIn("kb_run_quality_baseline", mcp_content)
        self.assertIn("kb_get_latest_quality_baseline", mcp_content)
        self.assertIn("kb_create_eval_suite", mcp_content)
        self.assertIn("kb_run_benchmark", mcp_content)
        self.assertIn("kb_analyze_failures", mcp_content)
        self.assertIn("kb_generate_case_study", mcp_content)
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

    def test_pypdf_layout_rules_extract_blocks_and_quality(self) -> None:
        class FakePage:
            def __init__(self, text: str) -> None:
                self._text = text

            def extract_text(self) -> str:
                return self._text

        class FakeReader:
            def __init__(self, path: str) -> None:
                del path
                self.pages = [
                    FakePage(
                        "期刊页眉\n"
                        "复杂 PDF 论文\n"
                        "摘要：本文研究复杂 PDF 版面解析，重点识别图题、表题和参考文献区域，"
                        "并清理页眉页脚以提升证据质量。\n"
                        "关键词：版面解析；图题；表题\n"
                        "1 引言\n"
                        "本文第一段介绍复杂论文版面的研究背景。\n"
                        "图 1 复杂 PDF 解析流程\n"
                        "表 1 版面块识别结果\n"
                        "DOI: 10.1234/noise\n"
                        "1\n"
                    ),
                    FakePage(
                        "期刊页眉\n"
                        "2 方法\n"
                        "本文提出基于 layout_block 的规则化解析方法。\n"
                        "参考文献\n"
                        "[1] 张三. PDF 解析研究. 2025.\n"
                        "2\n"
                    ),
                ]
                self.metadata = types.SimpleNamespace(title="复杂 PDF 论文", author="张三;李四")

        fake_module = types.SimpleNamespace(PdfReader=FakeReader)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"pypdf": fake_module}):
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            db_path = root / "kb.sqlite"
            (papers / "layout.pdf").write_bytes(b"%PDF fake")

            report = sync_directory(papers, db_path, force=True, pdf_parser="pypdf")

            self.assertEqual(report["indexed"], 1)
            doc_id = str(search_documents(db_path, "复杂 PDF 版面解析", top_k=1)[0]["doc_id"])
            layout = get_layout_blocks(db_path, doc_id)
            self.assertEqual(layout["schema"], "layout_blocks.v1")
            self.assertGreaterEqual(layout["count"], 8)
            self.assertGreaterEqual(layout["type_counts"]["heading"], 3)
            self.assertEqual(layout["type_counts"]["figure"], 1)
            self.assertEqual(layout["type_counts"]["table"], 1)
            self.assertGreaterEqual(layout["type_counts"]["reference"], 1)

            figures = get_figures(db_path, doc_id)
            tables = get_tables(db_path, doc_id)
            self.assertEqual(figures["count"], 1)
            self.assertIn("图 1", figures["figures"][0]["caption"])
            self.assertEqual(tables["count"], 1)
            self.assertIn("表 1", tables["tables"][0]["caption"])

            references = get_artifact(db_path, doc_id, "reference_sections.json")["content"]
            self.assertEqual(references["count"], 1)
            self.assertGreaterEqual(references["reference_sections"][0]["item_count"], 1)

            quality = get_parse_quality(db_path, doc_id)
            self.assertGreaterEqual(quality["layout_score"], 0.8)
            self.assertEqual(quality["caption_score"], 1.0)
            self.assertGreaterEqual(quality["noise_removed_count"], 3)
            self.assertEqual(quality["layout_block_count"], layout["count"])

            parse_report = get_parse_report(db_path, doc_id)
            self.assertEqual(parse_report["layout_block_count"], layout["count"])
            self.assertEqual(parse_report["figure_count"], 1)
            self.assertEqual(parse_report["table_count"], 1)
            self.assertGreaterEqual(parse_report["noise_removed_count"], 3)

            node_index = get_artifact(db_path, doc_id, "node_index.jsonl")["content"]
            visual_nodes = [node for node in node_index if node["kind"] in {"figure", "table"}]
            self.assertEqual(len(visual_nodes), 2)
            for node in visual_nodes:
                self.assertIn("layout_block_id", node["source_offsets"])
                self.assertIn("caption_id", node["source_offsets"])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "layout", doc_id])
            self.assertIn("layout_blocks.v1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "figures", doc_id])
            self.assertIn("图 1", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(["--db", str(db_path), "tables", doc_id])
            self.assertIn("表 1", stdout.getvalue())

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
                    "tables": [
                        {
                            "caption": "表 1 解析质量对比",
                            "rows": [["解析器", "章节数", "表格数"], ["Docling", "5", "2"]],
                            "bbox": [1, 2, 3, 4],
                        }
                    ],
                    "figures": [{"caption": "图 1 解析流程"}],
                },
            )

            with mock.patch("kb_agent.parsers._parse_docling_pdf", return_value=parsed):
                report = sync_directory(papers, db_path, force=True, pdf_parser="docling")

            self.assertEqual(report["indexed"], 1)
            doc_id = str(search_documents(db_path, "解析质量", top_k=1)[0]["doc_id"])
            structured = get_artifact(db_path, doc_id, "structured.json")["content"]
            self.assertEqual(structured["layout_schema"], "layout_blocks.v1")
            self.assertGreaterEqual(structured["layout_blocks_count"], 1)
            self.assertGreaterEqual(len(structured["tables"]), 1)
            self.assertGreaterEqual(len(structured["figures"]), 1)
            layout = get_layout_blocks(db_path, doc_id)
            self.assertGreaterEqual(layout["type_counts"]["table"], 1)
            self.assertGreaterEqual(layout["type_counts"]["figure"], 1)
            node_index = get_artifact(db_path, doc_id, "node_index.jsonl")["content"]
            table_node = next(node for node in node_index if node["kind"] == "table")
            self.assertIn("layout_block_id", table_node["source_offsets"])
            self.assertIn("caption_id", table_node["source_offsets"])
            tree = get_artifact(db_path, doc_id, "tree.json")["content"]
            types = _collect_tree_types(tree)
            self.assertIn("section", types)
            self.assertIn("table", types)
            self.assertIn("figure", types)
            parse_report = get_parse_report(db_path, doc_id)
            self.assertEqual(parse_report["parser_chain"], ["docling"])
            self.assertFalse(parse_report["fallback_used"])
            self.assertTrue(parse_report["adapter_statuses"]["marker"]["placeholder"])
            table_content = get_table_content(db_path, doc_id)
            self.assertEqual(table_content["count"], 1)
            self.assertEqual(table_content["table_content"][0]["source"], "docling_table")
            self.assertEqual(table_content["table_content"][0]["column_count"], 3)
            self.assertEqual(table_content["table_content"][0]["bbox"], [1.0, 2.0, 3.0, 4.0])

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
            reference_sections = get_artifact(db_path, doc_id, "reference_sections.json")["content"]
            self.assertEqual(reference_sections["schema"], "reference_sections.v1")
            self.assertEqual(reference_sections["count"], 1)
            self.assertGreaterEqual(reference_sections["reference_sections"][0]["references_count"], 1)
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

    def test_v1_database_migrates_to_v6(self) -> None:
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
                self.assertIn("feedback_items", tables)
                self.assertIn("paper_claims", tables)
                self.assertIn("paper_entities", tables)
                self.assertIn("paper_relations", tables)
                self.assertEqual(conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()["value"], "6")
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


def _sync_table_sample(root: Path) -> tuple[Path, str]:
    papers = root / "papers"
    papers.mkdir()
    db_path = root / "kb.sqlite"
    (papers / "table.txt").write_text(
        "摘要：本文研究任务规划方法的实验评测。\n"
        "关键词：任务规划；表格事实\n\n"
        "1 实验结果\n"
        "表 1 任务规划方法性能对比\n"
        "方法 任务完成率 响应时间 数据集\n"
        "基线方法 80% 12s 仿真任务集\n"
        "本文方法 92% 8s 仿真任务集\n\n"
        "结论\n"
        "本文方法优于基线方法。\n",
        encoding="utf-8",
    )
    report = sync_directory(papers, db_path, force=True)
    if report["failed"]:
        raise AssertionError(report["errors"])
    doc_id = str(search_documents(db_path, "任务规划 表格事实", top_k=1)[0]["doc_id"])
    return db_path, doc_id


def _sync_fact_audit_samples(root: Path) -> tuple[Path, list[str]]:
    papers = root / "papers"
    papers.mkdir()
    db_path = root / "kb.sqlite"
    (papers / "audit_a.txt").write_text(
        "摘要：本文研究任务完成率指标的一致性审计。\n"
        "关键词：任务完成率；事实审计\n\n"
        "1 实验结果\n"
        "本文方法在任务完成率上提升，并优于基线方法。\n"
        "参考文献\n"
        "[1] 张三. 任务规划评测. 2025.\n",
        encoding="utf-8",
    )
    (papers / "audit_b.txt").write_text(
        "摘要：本文研究另一个任务规划方法的实验结论。\n"
        "关键词：任务完成率；事实冲突\n\n"
        "1 实验结果\n"
        "本文方法在任务完成率上降低，并弱于基线方法。\n"
        "参考文献\n"
        "[1] 李四. 任务规划对比. 2024.\n",
        encoding="utf-8",
    )
    report = sync_directory(papers, db_path)
    if report["failed"]:
        raise AssertionError(report["errors"])

    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        rows = conn.execute("SELECT doc_id, path FROM documents ORDER BY path").fetchall()
        doc_ids = [str(row["doc_id"]) for row in rows]
    finally:
        conn.close()
    for doc_id in doc_ids:
        extract_doc_insights(db_path, doc_id, use_llm=False)

    node_ids: Dict[str, str] = {}
    version_ids: Dict[str, str] = {}
    for doc_id in doc_ids:
        node_index = get_artifact(db_path, doc_id, "node_index.jsonl")["content"]
        node_ids[doc_id] = next(node["node_id"] for node in node_index if node.get("text"))
        conn = db.connect(db_path)
        try:
            version = db.get_document_version(conn, doc_id)
            if version is None:
                raise AssertionError(doc_id)
            version_ids[doc_id] = str(version["version_id"])
        finally:
            conn.close()

    doc_a, doc_b = doc_ids
    now = 1.0
    claims = [
        {
            "claim_id": "claim_a_positive",
            "doc_id": doc_a,
            "version_id": version_ids[doc_a],
            "node_id": node_ids[doc_a],
            "type": "result",
            "text": "任务完成率提升，本文方法优于基线方法。",
            "normalized_text": "任务完成率提升本文方法优于基线方法",
            "page_range": [1, 1],
            "confidence": 0.86,
            "source": "rule",
            "evidence": {"node_id": node_ids[doc_a]},
            "created_at": now,
        },
        {
            "claim_id": "claim_a_duplicate",
            "doc_id": doc_a,
            "version_id": version_ids[doc_a],
            "node_id": node_ids[doc_a],
            "type": "result",
            "text": "任务完成率提升，本文方法优于基线方法。",
            "normalized_text": "任务完成率提升本文方法优于基线方法",
            "page_range": [1, 1],
            "confidence": 0.82,
            "source": "rule",
            "evidence": {"node_id": node_ids[doc_a]},
            "created_at": now,
        },
        {
            "claim_id": "claim_a_low",
            "doc_id": doc_a,
            "version_id": version_ids[doc_a],
            "node_id": node_ids[doc_a],
            "type": "result",
            "text": "响应时间改善但证据较弱。",
            "normalized_text": "响应时间改善但证据较弱",
            "page_range": [1, 1],
            "confidence": 0.3,
            "source": "rule",
            "evidence": {"node_id": node_ids[doc_a]},
            "created_at": now,
        },
        {
            "claim_id": "claim_a_no_evidence",
            "doc_id": doc_a,
            "version_id": version_ids[doc_a],
            "node_id": "",
            "type": "result",
            "text": "负载均衡提升但缺少证据绑定。",
            "normalized_text": "负载均衡提升但缺少证据绑定",
            "page_range": [],
            "confidence": 0.9,
            "source": "rule",
            "evidence": {},
            "created_at": now,
        },
        {
            "claim_id": "claim_a_table",
            "doc_id": doc_a,
            "version_id": version_ids[doc_a],
            "node_id": node_ids[doc_a],
            "type": "result",
            "text": "任务完成率提升，表格显示本文方法优于基线方法。",
            "normalized_text": "任务完成率提升表格显示本文方法优于基线方法",
            "page_range": [1, 1],
            "confidence": 0.9,
            "source": "table_rule",
            "evidence": {"node_id": node_ids[doc_a], "table_id": "table_001"},
            "created_at": now,
        },
        {
            "claim_id": "claim_a_text_mismatch",
            "doc_id": doc_a,
            "version_id": version_ids[doc_a],
            "node_id": node_ids[doc_a],
            "type": "result",
            "text": "任务完成率降低，正文认为本文方法弱于基线方法。",
            "normalized_text": "任务完成率降低正文认为本文方法弱于基线方法",
            "page_range": [1, 1],
            "confidence": 0.88,
            "source": "rule",
            "evidence": {"node_id": node_ids[doc_a]},
            "created_at": now,
        },
        {
            "claim_id": "claim_b_negative",
            "doc_id": doc_b,
            "version_id": version_ids[doc_b],
            "node_id": node_ids[doc_b],
            "type": "result",
            "text": "任务完成率降低，本文方法弱于基线方法。",
            "normalized_text": "任务完成率降低本文方法弱于基线方法",
            "page_range": [1, 1],
            "confidence": 0.87,
            "source": "rule",
            "evidence": {"node_id": node_ids[doc_b]},
            "created_at": now,
        },
    ]
    conn = db.connect(db_path)
    try:
        db.insert_paper_claims(conn, claims)
        conn.commit()
    finally:
        conn.close()
    return db_path, doc_ids


def _insert_noisy_entity_for_graph(db_path: Path, doc_id: str) -> None:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        version = db.get_document_version(conn, doc_id)
        node = conn.execute(
            "SELECT node_id FROM doc_nodes WHERE doc_id = ? AND COALESCE(text, '') != '' ORDER BY order_index LIMIT 1",
            (doc_id,),
        ).fetchone()
        if version is None or node is None:
            raise AssertionError(doc_id)
        db.insert_paper_entities(
            conn,
            [
                {
                    "entity_id": "entity_noisy_no",
                    "doc_id": doc_id,
                    "version_id": str(version["version_id"]),
                    "node_id": str(node["node_id"]),
                    "type": "term",
                    "name": "No.",
                    "normalized_name": "no",
                    "aliases": [],
                    "page_range": [1, 1],
                    "confidence": 0.8,
                    "source": "rule",
                    "evidence": {"node_id": str(node["node_id"])},
                    "created_at": 1.0,
                }
            ],
        )
        conn.commit()
    finally:
        conn.close()


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
