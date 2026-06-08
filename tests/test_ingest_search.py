from __future__ import annotations

import contextlib
import io
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
from kb_agent.search import get_evidence, search_documents, search_nodes


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


if __name__ == "__main__":
    unittest.main()
