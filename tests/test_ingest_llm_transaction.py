from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kb_agent import db
from kb_agent.ingest import sync_directory


class IngestLLMTransactionTest(unittest.TestCase):
    def test_doc_card_llm_runs_before_sqlite_write_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "papers"
            papers.mkdir()
            (papers / "paper.md").write_text(
                "# 摘要\n本文研究任务规划。\n\n# 方法\n本文提出结构化任务规划方法。\n",
                encoding="utf-8",
            )
            db_path = root / "kb.sqlite"
            conn = db.connect(db_path)
            observed_transactions: list[bool] = []

            def generate(_system: str, _user: str, **_options: object) -> dict:
                observed_transactions.append(conn.in_transaction)
                return {
                    "description": "任务规划论文。",
                    "method_summary": "提出结构化任务规划方法。",
                    "innovation_summary": "",
                    "limitation_summary": "",
                }

            with mock.patch("kb_agent.ingest.db.connect", return_value=conn), mock.patch(
                "kb_agent.ingest.get_llm_settings",
                return_value=object(),
            ), mock.patch("kb_agent.ingest.generate_json_object", side_effect=generate):
                report = sync_directory(papers, db_path, doc_card_use_llm=True)

            self.assertEqual(report["indexed"], 1)
            self.assertEqual(observed_transactions, [False])


if __name__ == "__main__":
    unittest.main()
