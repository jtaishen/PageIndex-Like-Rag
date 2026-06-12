from __future__ import annotations

import unittest

from kb_agent.models import NodeRecord, ParsedBlock, ParsedDocument
from kb_agent.parse_quality import build_parse_quality


class ParseQualityTest(unittest.TestCase):
    def test_good_quality_keeps_existing_schema_and_counts(self) -> None:
        parsed = _parsed_document(
            metadata={
                "authors": ["张三"],
                "year": 2026,
                "doi": "10.1234/example",
                "abstract": "本文研究结构化解析。",
                "_parse_diagnostics": {"parser_chain": ["pypdf"], "fallback_used": False},
            },
            references={"status": "extracted"},
        )
        nodes = [
            *[_node(f"section-{index}", "section", text="section") for index in range(4)],
            *[_node(f"paragraph-{index}", "paragraph", text="paragraph") for index in range(3)],
            *[_node(f"ref-{index}", "reference", text="reference") for index in range(10)],
            _node("figure-1", "figure", text="图 1"),
            _node("table-1", "table", text="表 1"),
        ]
        layout_blocks = [
            {"type": "heading", "bbox": [1, 2, 3, 4]},
            {"type": "paragraph"},
            {"type": "reference"},
            {"type": "table"},
        ]
        tables = [{"caption": "表 1", "layout_block_id": "layout-table"}]
        table_content = [{"row_count": 2, "column_count": 2, "quality_warnings": []}]
        figures = [{"caption": "图 1", "layout_block_id": "layout-figure"}]

        quality = build_parse_quality("doc-1", "ver-1", parsed, nodes, layout_blocks, tables, table_content, figures)

        self.assertEqual(quality["schema"], "parse_quality.v0")
        self.assertEqual(quality["quality_level"], "good")
        self.assertEqual(quality["section_count"], 4)
        self.assertEqual(quality["reference_count"], 10)
        self.assertEqual(quality["table_parse_score"], 1.0)
        self.assertEqual(quality["table_warning_count"], 0)
        self.assertEqual(quality["parser_chain"], ["pypdf"])
        self.assertFalse(quality["fallback_used"])
        self.assertEqual(quality["quality_warnings"], [])

    def test_weak_pdf_quality_exposes_warnings(self) -> None:
        parsed = _parsed_document(metadata={}, references={"status": "not_extracted"}, file_type="pdf")
        nodes = [
            _node("page-1", "page", text="page text"),
            _node("paragraph-1", "paragraph", text="paragraph text"),
        ]
        tables = [{"caption": "表 1", "layout_block_id": ""}]

        quality = build_parse_quality("doc-1", "ver-1", parsed, nodes, [], tables, [], [])

        self.assertEqual(quality["quality_level"], "weak")
        self.assertTrue(quality["missing_abstract"])
        self.assertTrue(quality["page_only_tree"])
        self.assertEqual(quality["layout_score"], 0.0)
        self.assertIn("missing_abstract", quality["quality_warnings"])
        self.assertIn("page_only_tree", quality["quality_warnings"])
        self.assertIn("low_section_count", quality["quality_warnings"])
        self.assertIn("missing_references", quality["quality_warnings"])
        self.assertIn("weak_layout_blocks", quality["quality_warnings"])
        self.assertIn("missing_table_content", quality["quality_warnings"])
        self.assertIn("weak_table_parse", quality["quality_warnings"])


def _parsed_document(
    *,
    metadata: dict,
    references: dict,
    file_type: str = "pdf",
) -> ParsedDocument:
    return ParsedDocument(
        title="结构化解析论文",
        file_type=file_type,
        raw_text="结构化解析正文。",
        blocks=[ParsedBlock(kind="paragraph", text="结构化解析正文。")],
        metadata=metadata,
        structured={},
        references=references,
        parser_name="pdf_pypdf" if file_type == "pdf" else "markdown",
        parser_version="0.16.0",
    )


def _node(node_id: str, kind: str, *, text: str) -> NodeRecord:
    return NodeRecord(
        doc_id="doc-1",
        node_id=node_id,
        parent_id=None,
        kind=kind,
        heading=text if kind == "section" else "",
        summary=text,
        text=text,
        level=1,
        node_path=text,
        page_start=1,
        page_end=1,
        order_index=1,
    )


if __name__ == "__main__":
    unittest.main()
