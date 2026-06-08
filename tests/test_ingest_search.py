from __future__ import annotations

import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from kb_agent import db
from kb_agent.answer import answer_query
from kb_agent.artifacts import get_artifact, get_doc_card, list_artifacts
from kb_agent.cli import main as cli_main
from kb_agent.ingest import sync_directory
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


if __name__ == "__main__":
    unittest.main()
