from __future__ import annotations

import unittest
from pathlib import Path

from kb_agent.ingest_artifacts import build_failure_parse_report, build_ingest_artifacts
from kb_agent.models import NodeRecord, ParsedBlock, ParsedDocument


class IngestArtifactsTest(unittest.TestCase):
    def test_build_ingest_artifacts_preserves_payload_shapes(self) -> None:
        parsed = _parsed_document()
        nodes = [
            _node("section-1", "section", "1 方法", "方法"),
            _node("paragraph-1", "paragraph", "1 方法", "", text="本文提出结构化解析方法。"),
            _node("table-1", "table", "1 方法", "", text="表 1 解析质量对比"),
            _node("figure-1", "figure", "1 方法", "", text="图 1 解析流程"),
            _node("ref-1", "reference", "参考文献", "", text="[1] 张三. 解析研究. 2026."),
        ]

        artifacts = build_ingest_artifacts(
            "doc-1",
            "ver-1",
            Path("/tmp/artifacts"),
            Path("/tmp/paper.md"),
            "hash-1",
            parsed,
            nodes,
        )

        self.assertEqual(artifacts["structured"]["layout_schema"], "layout_blocks.v1")
        self.assertEqual(artifacts["structured"]["table_content_count"], 1)
        self.assertEqual(artifacts["layout_blocks"]["schema"], "layout_blocks.v1")
        self.assertEqual(artifacts["tables"]["schema"], "tables.v1")
        self.assertEqual(artifacts["table_content"]["schema"], "table_content.v1")
        self.assertEqual(artifacts["table_content"]["table_content"][0]["headers"], ["方法", "准确率"])
        self.assertEqual(artifacts["table_summaries"]["schema"], "table_summaries.v1")
        self.assertEqual(artifacts["figures"]["schema"], "figures.v1")
        self.assertEqual(artifacts["reference_sections"]["schema"], "reference_sections.v1")
        self.assertEqual(artifacts["parse_report"]["schema"], "parse_report.v0")
        self.assertEqual(artifacts["parse_report"]["parser_chain"], ["pypdf"])
        self.assertEqual(artifacts["parse_report"]["table_parse_score"], 1.0)
        self.assertEqual(artifacts["parse_report"]["node_count"], len(nodes))
        self.assertIn("components", artifacts)

    def test_failure_parse_report_keeps_existing_fields(self) -> None:
        report = build_failure_parse_report(
            "doc-1",
            "ver-1",
            Path("/tmp/artifacts"),
            Path("/tmp/broken.docx"),
            "hash-1",
            "docx",
            "0.16.0",
            "Cannot read DOCX",
        )

        self.assertEqual(report["schema"], "parse_report.v0")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "Cannot read DOCX")
        self.assertEqual(report["parser_chain"], ["docx"])
        self.assertEqual(report["external_parser_errors"], ["Cannot read DOCX"])
        self.assertEqual(report["table_parse_score"], 0.0)
        self.assertEqual(report["table_warning_count"], 0)
        self.assertEqual(report["layout_block_count"], 0)


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        title="结构化解析论文",
        file_type="pdf",
        raw_text="摘要：本文研究结构化解析。\n关键词：解析\n1 方法\n表 1 解析质量对比",
        blocks=[
            ParsedBlock(kind="heading", text="", heading="1 方法", level=1, page=1, source_parser="pypdf"),
            ParsedBlock(kind="paragraph", text="本文提出结构化解析方法。", page=1, source_parser="pypdf"),
            ParsedBlock(kind="table", text="表 1 解析质量对比", page=2, source_parser="pypdf"),
            ParsedBlock(kind="paragraph", text="方法  准确率\nBaseline  80%\nOurs  91%", page=2, source_parser="pypdf"),
            ParsedBlock(kind="figure", text="图 1 解析流程", page=3, source_parser="pypdf"),
            ParsedBlock(kind="heading", text="", heading="参考文献", level=1, page=4, source_parser="pypdf"),
            ParsedBlock(kind="reference", text="[1] 张三. 解析研究. 2026.", page=4, source_parser="pypdf"),
        ],
        metadata={
            "authors": ["张三"],
            "year": 2026,
            "abstract": "本文研究结构化解析。",
            "keywords": ["解析"],
            "_parse_diagnostics": {"parser_chain": ["pypdf"], "fallback_used": False},
        },
        structured={},
        references={"schema": "references.v0", "status": "extracted", "references": [{"raw": "ref"}]},
        parser_name="pdf_pypdf",
        parser_version="0.16.0",
    )


def _node(node_id: str, kind: str, node_path: str, heading: str, *, text: str = "") -> NodeRecord:
    return NodeRecord(
        doc_id="doc-1",
        node_id=node_id,
        parent_id=None,
        kind=kind,
        heading=heading,
        summary=text[:40],
        text=text,
        level=1,
        node_path=node_path,
        page_start=1,
        page_end=1,
        order_index=1,
    )


if __name__ == "__main__":
    unittest.main()
