from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from kb_agent import parsers
from kb_agent.models import ParsedBlock, ParsedDocument
from kb_agent.pdf_parsers import parse_grobid_tei


class PdfParsersTest(unittest.TestCase):
    def test_pypdf_wrapper_extracts_layout_and_noise_quality(self) -> None:
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
                        "摘要：本文研究 PDF 解析。\n"
                        "1 方法\n"
                        "图 1 解析流程\n"
                        "表 1 指标结果\n"
                        "DOI: 10.1234/noise\n"
                        "1\n"
                    ),
                    FakePage(
                        "期刊页眉\n"
                        "参考文献\n"
                        "[1] 张三. PDF 解析. 2026.\n"
                        "2\n"
                    ),
                ]
                self.metadata = types.SimpleNamespace(title="复杂 PDF 论文", author="张三;李四")

        fake_module = types.SimpleNamespace(PdfReader=FakeReader)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"pypdf": fake_module}):
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF fake")

            doc = parsers._parse_pypdf_pdf(path)

        self.assertEqual(doc.parser_name, "pdf_pypdf")
        self.assertEqual(doc.metadata["authors"], ["张三", "李四"])
        self.assertGreaterEqual(doc.metadata["noise_removed_count"], 3)
        self.assertGreaterEqual(doc.structured["layout_blocks_count"], 5)
        self.assertGreaterEqual(doc.structured["figure_count"], 1)
        self.assertGreaterEqual(doc.structured["table_count"], 1)
        self.assertGreaterEqual(doc.structured["reference_section_count"], 1)

    def test_docling_wrapper_converts_structured_payload(self) -> None:
        class FakeDocument:
            def export_to_dict(self) -> dict:
                return {
                    "texts": [
                        {"label": "title", "text": "Docling 标题", "prov": [{"page": 1}]},
                        {"label": "section_header", "text": "方法", "prov": [{"page": 1}]},
                    ],
                    "tables": [
                        {
                            "caption": "表 1 解析质量",
                            "rows": [["方法", "准确率"], ["Docling", "90%"]],
                            "prov": [{"page": 2, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}],
                        }
                    ],
                    "figures": [{"caption": "图 1 解析流程", "prov": [{"page": 3}]}],
                    "pages": [{"page": 1}, {"page": 2}, {"page": 3}],
                }

            def export_to_markdown(self) -> str:
                return "# Docling 标题\n\n方法正文"

        class FakeConverter:
            def convert(self, path: str) -> object:
                del path
                return types.SimpleNamespace(document=FakeDocument())

        docling_module = types.ModuleType("docling")
        converter_module = types.ModuleType("docling.document_converter")
        converter_module.DocumentConverter = FakeConverter

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules,
            {"docling": docling_module, "docling.document_converter": converter_module},
        ):
            path = Path(tmp) / "docling.pdf"
            path.write_bytes(b"%PDF fake")

            doc = parsers._parse_docling_pdf(path)

        self.assertEqual(doc.parser_name, "pdf_docling")
        self.assertEqual(doc.metadata["pages"], 3)
        self.assertEqual(doc.structured["layout_schema"], "layout_blocks.v1")
        self.assertGreaterEqual(doc.structured["table_content_count"], 1)
        self.assertEqual(doc.structured["table_content"][0]["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(doc.structured["table_content"][0]["source"], "docling_table")

    def test_grobid_tei_parse_preserves_metadata_and_references(self) -> None:
        tei = """
        <TEI>
          <teiHeader>
            <fileDesc>
              <titleStmt><title>GROBID 增强论文</title></titleStmt>
              <sourceDesc>
                <biblStruct>
                  <analytic>
                    <author><persName><forename>San</forename><surname>Zhang</surname></persName></author>
                    <title>备用标题</title>
                  </analytic>
                  <monogr><title>机器人学报</title></monogr>
                  <idno type="doi">10.1234/example</idno>
                  <date when="2026-01-01" />
                </biblStruct>
              </sourceDesc>
            </fileDesc>
            <profileDesc><abstract>本文使用 GROBID 增强元数据。</abstract></profileDesc>
          </teiHeader>
          <text>
            <back>
              <listBibl>
                <biblStruct>
                  <analytic>
                    <author><persName><surname>Li</surname></persName></author>
                    <title>参考论文</title>
                  </analytic>
                  <monogr><date when="2025" /></monogr>
                </biblStruct>
              </listBibl>
            </back>
          </text>
        </TEI>
        """

        result = parse_grobid_tei(tei, parse_error_cls=parsers.ParseError)

        self.assertEqual(result["metadata"]["title"], "GROBID 增强论文")
        self.assertEqual(result["metadata"]["authors"], ["San Zhang"])
        self.assertEqual(result["metadata"]["year"], 2026)
        self.assertEqual(result["metadata"]["venue"], "机器人学报")
        self.assertEqual(result["metadata"]["doi"], "10.1234/example")
        self.assertEqual(result["references"][0]["title"], "参考论文")

    def test_pdf_parser_still_uses_patchable_facade_wrapper(self) -> None:
        fake_doc = ParsedDocument(
            title="patched",
            file_type="pdf",
            raw_text="patched text",
            blocks=[ParsedBlock(kind="paragraph", text="patched text")],
            metadata={},
            structured={},
            references={},
            parser_name="pdf_pypdf",
            parser_version=parsers.PARSER_VERSION,
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "kb_agent.parsers._parse_pypdf_pdf",
            return_value=fake_doc,
        ) as patched:
            path = Path(tmp) / "patched.pdf"
            path.write_bytes(b"%PDF fake")

            doc = parsers.PdfParser().parse(path, pdf_parser="pypdf")

        patched.assert_called_once_with(path)
        self.assertEqual(doc.title, "patched")
        self.assertEqual(doc.parser_name, "pdf_pypdf")
        self.assertEqual(doc.structured["parser_chain"], ["pypdf"])
        self.assertFalse(doc.structured["fallback_used"])


if __name__ == "__main__":
    unittest.main()
