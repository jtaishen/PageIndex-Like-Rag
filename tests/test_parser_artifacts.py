from __future__ import annotations

import unittest

from kb_agent.models import ParsedBlock
from kb_agent.parser_artifacts import (
    build_layout_blocks,
    build_reference_sections,
    build_table_content,
    build_table_summaries,
    build_visual_items,
    enhance_table_items,
    table_parse_score,
    table_warning_count,
)


class ParserArtifactsTest(unittest.TestCase):
    def test_layout_visual_table_and_reference_artifacts_keep_schema(self) -> None:
        blocks = [
            ParsedBlock(kind="heading", text="", heading="1 方法", level=1, page=1),
            ParsedBlock(kind="paragraph", text="本文提出结构化解析流程。", page=1),
            ParsedBlock(kind="table", text="表 1 解析质量对比", page=2),
            ParsedBlock(kind="paragraph", text="方法  准确率\nBaseline  80%\nOurs  91%", page=2),
            ParsedBlock(kind="figure", text="图 1 解析流程", page=3),
            ParsedBlock(kind="heading", text="", heading="参考文献", level=1, page=4),
            ParsedBlock(kind="reference", text="[1] 张三. 解析研究. 2025.", page=4),
        ]

        layout = build_layout_blocks(blocks, "pypdf")
        table_items = build_visual_items(layout, "table")
        figure_items = build_visual_items(layout, "figure")
        table_content = build_table_content(blocks, layout)
        enhanced_tables = enhance_table_items(table_items, table_content)
        summaries = build_table_summaries(table_content)
        references = build_reference_sections(layout, {"references": [{"raw": "ref"}]})

        self.assertEqual(layout[0]["schema"], "layout_block.v1")
        self.assertEqual(layout[0]["section_path"], ["1 方法"])
        self.assertEqual(table_items[0]["schema"], "table.v1")
        self.assertEqual(figure_items[0]["schema"], "figure.v1")
        self.assertEqual(table_content[0]["schema"], "table_content.v1")
        self.assertEqual(table_content[0]["caption"], "表 1 解析质量对比")
        self.assertEqual(table_content[0]["headers"], ["方法", "准确率"])
        self.assertEqual(table_content[0]["row_count"], 2)
        self.assertEqual(table_content[0]["column_count"], 2)
        self.assertEqual(enhanced_tables[0]["quality_warnings"], [])
        self.assertEqual(summaries[0]["schema"], "table_summary.v1")
        self.assertIn("准确率", summaries[0]["metrics"])
        self.assertEqual(table_parse_score(table_content, enhanced_tables), 1.0)
        self.assertEqual(table_warning_count(table_content), 0)
        self.assertEqual(references[0]["schema"], "reference_section.v1")
        self.assertEqual(references[0]["references_count"], 1)

    def test_raw_table_payload_preserves_docling_fields(self) -> None:
        caption_block = ParsedBlock(
            kind="table",
            text="表 2 消融实验",
            page=5,
            bbox=[1.0, 2.0, 3.0, 4.0],
            source_parser="docling",
        )
        layout = build_layout_blocks([caption_block], "docling")

        table_content = build_table_content(
            [caption_block],
            layout,
            raw_tables=[
                {
                    "caption": "表 2 消融实验",
                    "rows": [["设置", "F1"], ["Full", "92%"]],
                    "bbox": {"x0": 1, "y0": 2, "x1": 3, "y1": 4},
                    "confidence": 0.87,
                }
            ],
        )

        self.assertEqual(len(table_content), 1)
        self.assertEqual(table_content[0]["source"], "docling_table")
        self.assertEqual(table_content[0]["source_parser"], "docling")
        self.assertEqual(table_content[0]["page"], 5)
        self.assertEqual(table_content[0]["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(table_content[0]["confidence"], 0.87)
        self.assertEqual(table_parse_score(table_content, build_visual_items(layout, "table")), 1.0)


if __name__ == "__main__":
    unittest.main()
