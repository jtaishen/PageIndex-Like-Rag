from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kb_agent import db
from kb_agent.fact_queries import fact_coverage_summary, fact_search
from kb_agent.models import DocumentRecord


def _insert_document(conn) -> None:  # type: ignore[no-untyped-def]
    db.upsert_document(
        conn,
        DocumentRecord(
            doc_id="doc_a",
            path="/tmp/doc_a.md",
            hash="hash_a",
            title="Doc A",
            file_type="markdown",
            size=100,
            mtime=1.0,
            summary="summary",
        ),
    )


class FactQueriesTest(unittest.TestCase):
    def test_fact_search_filters_by_type_source_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"
            conn = db.connect(db_path)
            db.init_db(conn)
            _insert_document(conn)
            db.insert_paper_claims(
                conn,
                [
                    {
                        "claim_id": "claim_table",
                        "doc_id": "doc_a",
                        "version_id": "v1",
                        "node_id": "node_1",
                        "claim_type": "result",
                        "text": "任务完成率提升 12%",
                        "page_range": [3],
                        "confidence": 0.91,
                        "source": "rule_table",
                        "evidence": {"table_id": "table_001"},
                    },
                    {
                        "claim_id": "claim_text",
                        "doc_id": "doc_a",
                        "version_id": "v1",
                        "node_id": "node_2",
                        "claim_type": "method",
                        "text": "动态角色任务规划用于开放环境",
                        "page_range": [1],
                        "confidence": 0.4,
                        "source": "rule_text",
                        "evidence": {"node_id": "node_2"},
                    },
                ],
            )
            conn.commit()
            conn.close()

            result = fact_search(
                db_path,
                "任务完成率",
                doc_ids=["doc_a"],
                fact_type="claim",
                source="table",
                min_confidence=0.8,
                top_k=5,
            )

        self.assertEqual(result["schema"], "fact_search.v1")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["fact_id"], "claim_table")
        self.assertEqual(result["items"][0]["source_kind"], "table")

    def test_fact_coverage_summary_counts_table_and_text_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "kb.sqlite"
            conn = db.connect(db_path)
            db.init_db(conn)
            _insert_document(conn)
            db.insert_paper_claims(
                conn,
                [
                    {
                        "claim_id": "claim_table",
                        "doc_id": "doc_a",
                        "version_id": "v1",
                        "node_id": "node_1",
                        "claim_type": "result",
                        "text": "任务完成率提升 12%",
                        "confidence": 0.91,
                        "source": "rule_table",
                    },
                    {
                        "claim_id": "claim_text",
                        "doc_id": "doc_a",
                        "version_id": "v1",
                        "node_id": "node_2",
                        "claim_type": "method",
                        "text": "动态角色任务规划用于开放环境",
                        "confidence": 0.4,
                        "source": "rule_text",
                    },
                ],
            )
            conn.commit()
            conn.close()

            summary = fact_coverage_summary(db_path, doc_id="doc_a")

        self.assertEqual(summary["schema"], "fact_coverage.v1")
        self.assertEqual(summary["claim_count"], 2)
        self.assertEqual(summary["table_backed_fact_count"], 1)
        self.assertEqual(summary["text_backed_fact_count"], 1)
        self.assertEqual(summary["table_backed_fact_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
