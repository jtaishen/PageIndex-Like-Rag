from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from kb_agent.doc_cards import build_doc_card, doc_card_summaries, rule_doc_card_summaries
from kb_agent.llm import LLMError
from kb_agent.models import NodeRecord, ParsedBlock, ParsedDocument


class DocCardsTest(unittest.TestCase):
    def test_build_doc_card_keeps_schema_and_disabled_llm_warning(self) -> None:
        parsed = _parsed_document()
        nodes = [_node("section-1", "section", "方法"), _node("paragraph-1", "paragraph", "本文提出结构化解析方法。")]
        generator = mock.Mock(side_effect=AssertionError("LLM should not be called"))

        card = build_doc_card(
            "doc-1",
            "ver-1",
            Path("/tmp/artifacts"),
            Path("/tmp/paper.md"),
            "hash-1",
            parsed,
            nodes,
            doc_card_use_llm=False,
            json_generator=generator,
            llm_settings_getter=lambda: object(),
            llm_error_cls=LLMError,
        )

        self.assertEqual(card["schema"], "doc_card.v0")
        self.assertEqual(card["summary_source"], "rule")
        self.assertEqual(card["summary_warnings"], ["deepseek_summary_skipped:disabled"])
        self.assertEqual(card["parse_quality"]["schema"], "parse_quality.v0")
        self.assertEqual(card["artifacts"][0], "raw_text.txt")
        generator.assert_not_called()

    def test_llm_summary_success_and_failure_paths_are_visible(self) -> None:
        parsed = _parsed_document()
        payload = {
            "description": "LLM 生成的短描述。",
            "method_summary": "LLM 方法摘要。",
            "innovation_summary": "LLM 创新摘要。",
            "limitation_summary": "LLM 局限摘要。",
        }
        success_generator = mock.Mock(return_value=payload)

        success = doc_card_summaries(
            parsed,
            use_llm=True,
            json_generator=success_generator,
            llm_settings_getter=lambda: object(),
            llm_error_cls=LLMError,
        )

        self.assertEqual(success["summary_source"], "deepseek")
        self.assertEqual(success["description"], "LLM 生成的短描述。")
        success_generator.assert_called_once()

        failure = doc_card_summaries(
            parsed,
            use_llm=True,
            json_generator=mock.Mock(side_effect=LLMError("boom", error_type="request_failed")),
            llm_settings_getter=lambda: object(),
            llm_error_cls=LLMError,
        )

        self.assertEqual(failure["summary_source"], "rule")
        self.assertIn("deepseek_summary_failed:request_failed", failure["summary_warnings"])

    def test_not_configured_llm_does_not_call_generator(self) -> None:
        generator = mock.Mock(side_effect=AssertionError("LLM should not be called"))

        summary = doc_card_summaries(
            _parsed_document(),
            use_llm=True,
            json_generator=generator,
            llm_settings_getter=lambda: None,
            llm_error_cls=LLMError,
        )

        self.assertEqual(summary["summary_source"], "rule")
        self.assertEqual(summary["summary_warnings"], ["deepseek_summary_skipped:not_configured"])
        generator.assert_not_called()

    def test_rule_summary_filters_front_matter_noise(self) -> None:
        parsed = _parsed_document(
            blocks=[
                ParsedBlock(kind="heading", text="", heading="封面", level=1),
                ParsedBlock(kind="paragraph", text="指导教师 张三 学号 123456"),
                ParsedBlock(kind="heading", text="", heading="方法", level=1),
                ParsedBlock(kind="paragraph", text="本文提出结构化解析方法，用于稳定生成文档卡片。"),
            ],
            raw_text="指导教师 张三 学号 123456\n本文提出结构化解析方法。",
        )

        summary = rule_doc_card_summaries(parsed)

        self.assertIn("结构化解析方法", summary["method_summary"])
        self.assertNotIn("指导教师", summary["method_summary"])


def _parsed_document(
    *,
    blocks: list[ParsedBlock] | None = None,
    raw_text: str = "摘要：本文研究结构化解析。\n关键词：解析\n方法：本文提出结构化解析方法。",
) -> ParsedDocument:
    return ParsedDocument(
        title="结构化解析论文",
        file_type="markdown",
        raw_text=raw_text,
        blocks=blocks
        or [
            ParsedBlock(kind="abstract", text="本文研究结构化解析。"),
            ParsedBlock(kind="keywords", text="解析"),
            ParsedBlock(kind="heading", text="", heading="方法", level=1),
            ParsedBlock(kind="paragraph", text="本文提出结构化解析方法。"),
        ],
        metadata={
            "authors": ["张三"],
            "year": 2026,
            "abstract": "本文研究结构化解析。",
            "keywords": ["解析"],
        },
        structured={},
        references={"status": "extracted", "references": [{"raw": "ref"}]},
        parser_name="markdown",
        parser_version="0.16.0",
    )


def _node(node_id: str, kind: str, text: str) -> NodeRecord:
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
