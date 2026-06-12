from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kb_agent import db
from kb_agent.fact_artifacts import build_fact_artifacts, read_existing_facts, replace_fact_rows, write_fact_artifacts
from kb_agent.fact_records import claim_record, entity_record, relation_record
from kb_agent.models import DocumentRecord


class FactArtifactsTest(unittest.TestCase):
    def test_build_fact_artifacts_preserves_report_and_collection_schemas(self) -> None:
        facts = _facts()

        artifacts = build_fact_artifacts(
            "doc-1",
            "v1",
            {"title": "Doc A"},
            {"quality_level": "usable", "quality_warnings": []},
            facts,
            "",
        )

        self.assertEqual(artifacts["claims"]["schema"], "claims.v1")
        self.assertEqual(artifacts["entities"]["schema"], "entities.v1")
        self.assertEqual(artifacts["relations"]["schema"], "relations.v1")
        self.assertEqual(artifacts["fact_graph"]["schema"], "fact_graph.v1")
        report = artifacts["fact_report"]
        self.assertEqual(report["schema"], "fact_report.v1")
        self.assertEqual(report["claim_count"], 1)
        self.assertEqual(report["entity_count"], 1)
        self.assertEqual(report["relation_count"], 1)
        self.assertEqual(report["table_backed_fact_count"], 0)
        self.assertEqual(report["fact_dedupe"]["schema"], "fact_dedupe.v1")

    def test_write_read_and_replace_fact_rows_keep_current_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"
            artifact_dir = Path(tmp) / "artifacts"
            artifact_dir.mkdir()
            _seed_doc(db_path, artifact_dir)
            facts = _facts()
            artifacts = build_fact_artifacts(
                "doc-1",
                "v1",
                {"title": "Doc A"},
                {"quality_level": "usable", "quality_warnings": []},
                facts,
                "",
            )

            write_fact_artifacts(artifact_dir, artifacts)
            replace_fact_rows(db_path, "doc-1", "v1", facts)
            existing = read_existing_facts(db_path, "doc-1")

        assert existing is not None
        self.assertEqual(existing["claims"]["schema"], "claims.v1")
        self.assertEqual(existing["entities"]["count"], 1)
        self.assertEqual(existing["relations"]["count"], 1)
        self.assertEqual(existing["fact_graph"]["schema"], "fact_graph.v1")
        self.assertEqual(existing["fact_report"]["schema"], "fact_report.v1")


def _facts() -> dict:
    node = {
        "doc_id": "doc-1",
        "node_id": "node-1",
        "node_path": "1 Method",
        "page_start": 1,
        "page_end": 1,
        "source_offsets": {},
    }
    claim = claim_record("doc-1", "v1", "method", "提出动态角色发现机制。", node, "rule", 0.7, 0)
    entity = entity_record("doc-1", "v1", "method", "动态角色发现机制", node, "rule", 0.7)
    relation = relation_record("doc-1", "v1", "uses", "动态角色发现机制", "任务分解", node, "rule", 0.7)
    assert claim is not None and entity is not None and relation is not None
    return {
        "claims": [claim],
        "entities": [entity],
        "relations": [relation],
        "status": "partial",
        "source": "rule",
        "warnings": ["rule_based_fact_extraction"],
        "quality_stats": {},
        "dedupe_stats": {
            "schema": "fact_dedupe.v1",
            "dedupe_input_count": 3,
            "dedupe_output_count": 3,
            "dedupe_merged_count": 0,
            "post_dedupe_duplicate_count": 0,
            "by_type": {},
        },
    }


def _seed_doc(db_path: Path, artifact_dir: Path) -> None:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        db.upsert_document(
            conn,
            DocumentRecord(
                doc_id="doc-1",
                path="/tmp/doc-1.md",
                hash="hash-1",
                title="Doc A",
                file_type="markdown",
                size=10,
                mtime=1.0,
                summary="summary",
            ),
        )
        db.insert_document_version(
            conn,
            version_id="v1",
            doc_id="doc-1",
            file_hash="hash-1",
            parser_name="test",
            parser_version="1",
            artifact_dir=str(artifact_dir),
            parse_status="ok",
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    unittest.main()
