from __future__ import annotations

import html
import importlib.util
import os
import re
import zipfile
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

from . import pdf_parsers
from .models import ParsedBlock, ParsedDocument
from .parser_artifacts import (
    _ensure_layout_metadata,
    _is_table_line,
    _looks_like_table_data_line,
    build_layout_blocks,
    build_reference_sections,
    build_table_content,
    build_table_summaries,
    build_visual_items,
    enhance_table_items,
    table_parse_score,
    table_warning_count,
)
from .utils import compact_whitespace, read_text_lossy, split_paragraphs


PARSER_VERSION = "0.16.0"
PDF_PARSER_CHOICES = {"auto", "pypdf", "docling", "grobid"}
DEFAULT_PDF_PARSER = "auto"


class ParseError(RuntimeError):
    pass


def resolve_pdf_parser(pdf_parser: Optional[str] = None) -> str:
    requested = (pdf_parser or os.environ.get("KB_PDF_PARSER") or DEFAULT_PDF_PARSER).strip().lower()
    if requested not in PDF_PARSER_CHOICES:
        choices = ", ".join(sorted(PDF_PARSER_CHOICES))
        raise ParseError(f"Unsupported PDF parser '{requested}'. Expected one of: {choices}")
    return requested


def pdf_adapter_statuses() -> Dict[str, Dict[str, Any]]:
    return {
        "pypdf": {
            "available": _module_available("pypdf") or _module_available("PyPDF2"),
            "default_fallback": True,
        },
        "docling": {
            "available": _module_available("docling"),
            "default_in_auto": True,
        },
        "grobid": {
            "available": bool(os.environ.get("GROBID_URL", "").strip()),
            "configured_url": bool(os.environ.get("GROBID_URL", "").strip()),
        },
        "marker": {
            "available": _module_available("marker"),
            "placeholder": True,
            "enabled": False,
        },
    }


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class ParserAdapter(ABC):
    name = "base"
    version = PARSER_VERSION
    file_type = "unknown"
    suffixes: set[str] = set()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.suffixes

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        raise NotImplementedError

    def finish(
        self,
        path: Path,
        *,
        title: str,
        raw_text: str,
        blocks: List[ParsedBlock],
        metadata: Optional[Dict[str, Any]] = None,
        body_md: str = "",
        references: Optional[Dict[str, Any]] = None,
        structured: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        parser_name: Optional[str] = None,
        parser_version: Optional[str] = None,
    ) -> ParsedDocument:
        normalized_blocks, normalize_warnings = normalize_blocks(blocks)
        source_parser = parser_name or self.name
        _ensure_layout_metadata(normalized_blocks, source_parser)
        enriched = _enrich_metadata(title, raw_text, metadata or {})
        if not enriched.get("abstract"):
            enriched["abstract"] = _extract_abstract_from_blocks(normalized_blocks)
        if not enriched.get("keywords"):
            enriched["keywords"] = _extract_keywords_from_blocks(normalized_blocks)
        references_payload = references or extract_references(raw_text)
        layout_blocks = build_layout_blocks(normalized_blocks, source_parser)
        raw_structured = dict(structured or {})
        raw_tables = raw_structured.get("tables") if isinstance(raw_structured.get("tables"), list) else []
        table_items = build_visual_items(layout_blocks, "table")
        table_content = build_table_content(normalized_blocks, layout_blocks, raw_tables=raw_tables)
        table_items = enhance_table_items(table_items, table_content)
        table_summaries = build_table_summaries(table_content)
        figure_items = build_visual_items(layout_blocks, "figure")
        reference_sections = build_reference_sections(layout_blocks, references_payload)
        structured_payload = dict(raw_structured or {
            "schema": "structured.v0",
            "blocks": [_block_to_dict(block, index) for index, block in enumerate(normalized_blocks)],
            "tables": [],
            "figures": [],
            "formulas": [],
        })
        if "blocks" not in structured_payload:
            structured_payload["blocks"] = [_block_to_dict(block, index) for index, block in enumerate(normalized_blocks)]
        structured_payload["layout_schema"] = "layout_blocks.v1"
        structured_payload["layout_blocks"] = layout_blocks
        structured_payload["layout_blocks_count"] = len(layout_blocks)
        structured_payload["table_count"] = len(table_items)
        structured_payload["figure_count"] = len(figure_items)
        structured_payload["table_content"] = table_content
        structured_payload["table_summaries"] = table_summaries
        structured_payload["table_content_count"] = len(table_content)
        structured_payload["table_parse_score"] = table_parse_score(table_content, table_items)
        structured_payload["table_warning_count"] = table_warning_count(table_content)
        structured_payload["reference_section_count"] = len(reference_sections)
        if table_items:
            structured_payload["tables"] = table_items
        else:
            structured_payload.setdefault("tables", [])
        if figure_items:
            structured_payload["figures"] = figure_items
        else:
            structured_payload.setdefault("figures", [])
        structured_payload["reference_sections"] = reference_sections
        structured_payload.setdefault("formulas", [])
        return ParsedDocument(
            title=enriched.get("title") or title or path.stem,
            file_type=self.file_type,
            raw_text=raw_text,
            blocks=normalized_blocks,
            metadata=enriched,
            body_md=body_md or blocks_to_markdown(normalized_blocks, title),
            structured=structured_payload,
            references=references_payload,
            parser_name=source_parser,
            parser_version=parser_version or self.version,
            parse_warnings=[*(warnings or []), *normalize_warnings],
        )


class MarkdownParser(ParserAdapter):
    name = "markdown"
    file_type = "markdown"
    suffixes = {".md", ".markdown"}

    def parse(self, path: Path) -> ParsedDocument:
        text = read_text_lossy(path)
        blocks: List[ParsedBlock] = []
        current: List[str] = []
        char_cursor = 0
        title = path.stem

        def flush_paragraph() -> None:
            nonlocal current, char_cursor
            if not current:
                return
            paragraph = "\n".join(current).strip()
            if paragraph:
                blocks.append(
                    ParsedBlock(
                        kind="paragraph",
                        text=paragraph,
                        char_start=char_cursor,
                        char_end=char_cursor + len(paragraph),
                    )
                )
                char_cursor += len(paragraph)
            current = []

        for line in text.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if match:
                flush_paragraph()
                level = len(match.group(1))
                heading = match.group(2).strip()
                if title == path.stem:
                    title = heading
                blocks.append(ParsedBlock(kind="heading", text="", heading=heading, level=level))
                continue
            if line.strip():
                current.append(line)
            else:
                flush_paragraph()
        flush_paragraph()

        return self.finish(
            path,
            title=title,
            raw_text=text,
            blocks=blocks,
            metadata={"source_format": "markdown"},
            body_md=text,
        )


class PlainTextParser(ParserAdapter):
    name = "plain_text"
    file_type = "text"
    suffixes = {".txt"}

    def parse(self, path: Path) -> ParsedDocument:
        text = read_text_lossy(path)
        blocks = [
            ParsedBlock(kind="paragraph", text=paragraph)
            for paragraph in split_paragraphs(text)
        ]
        return self.finish(
            path,
            title=path.stem,
            raw_text=text,
            blocks=blocks,
            metadata={"source_format": "text"},
        )


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        if tag.lower() in {"p", "br", "div", "section", "article", "h1", "h2", "h3"}:
            self.parts.append("\n")


class HtmlParser(ParserAdapter):
    name = "html_parser"
    file_type = "html"
    suffixes = {".html", ".htm"}

    def parse(self, path: Path) -> ParsedDocument:
        raw = read_text_lossy(path)
        extractor = _HTMLTextExtractor()
        extractor.feed(raw)
        text = html.unescape("\n".join(extractor.parts))
        blocks = [
            ParsedBlock(kind="paragraph", text=paragraph)
            for paragraph in split_paragraphs(text)
        ]
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
        title = compact_whitespace(html.unescape(title_match.group(1))) if title_match else path.stem
        return self.finish(
            path,
            title=title or path.stem,
            raw_text=text,
            blocks=blocks,
            metadata={"source_format": "html"},
        )


class DocxParser(ParserAdapter):
    name = "docx_zip_xml"
    file_type = "docx"
    suffixes = {".docx"}

    def parse(self, path: Path) -> ParsedDocument:
        try:
            with zipfile.ZipFile(path) as docx:
                xml = docx.read("word/document.xml")
        except (KeyError, zipfile.BadZipFile) as exc:
            raise ParseError(f"Cannot read DOCX document XML: {exc}") from exc

        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        root = ElementTree.fromstring(xml)
        paragraphs: List[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
            merged = "".join(texts).strip()
            if merged:
                paragraphs.append(merged)

        raw_text = "\n\n".join(paragraphs)
        blocks = [ParsedBlock(kind="paragraph", text=part) for part in paragraphs]
        title = paragraphs[0][:120] if paragraphs else path.stem
        return self.finish(
            path,
            title=compact_whitespace(title) or path.stem,
            raw_text=raw_text,
            blocks=blocks,
            metadata={"source_format": "docx"},
        )


class PdfParser(ParserAdapter):
    name = "pdf_auto"
    file_type = "pdf"
    suffixes = {".pdf"}

    def parse(self, path: Path, pdf_parser: Optional[str] = None) -> ParsedDocument:
        provider = resolve_pdf_parser(pdf_parser)
        parser_chain: List[str] = []
        external_errors: List[str] = []
        fallback_used = False

        doc: Optional[ParsedDocument] = None
        if provider in {"auto", "docling"}:
            try:
                parser_chain.append("docling")
                doc = _parse_docling_pdf(path)
            except Exception as exc:
                external_errors.append(f"docling:{exc}")
                if provider == "docling":
                    fallback_used = True
                elif provider == "auto":
                    fallback_used = True

        if doc is None:
            parser_chain.append("pypdf")
            doc = _parse_pypdf_pdf(path)

        grobid_url = os.environ.get("GROBID_URL", "").strip()
        if provider == "grobid" or (provider == "auto" and grobid_url):
            if grobid_url:
                parser_chain.append("grobid")
                try:
                    _merge_grobid_enrichment(doc, _fetch_grobid_enrichment(path, grobid_url))
                except Exception as exc:
                    external_errors.append(f"grobid:{exc}")
                    fallback_used = True
            else:
                external_errors.append("grobid:GROBID_URL is not configured")
                fallback_used = True

        diagnostics = {
            "requested_pdf_parser": provider,
            "parser_chain": parser_chain,
            "fallback_used": fallback_used,
            "external_parser_errors": external_errors,
            "adapter_statuses": pdf_adapter_statuses(),
        }
        doc.metadata["_parse_diagnostics"] = diagnostics
        doc.structured["parser_chain"] = parser_chain
        doc.structured["fallback_used"] = fallback_used
        if external_errors:
            doc.parse_warnings.extend(f"external_parser_failed:{item}" for item in external_errors)
        if fallback_used:
            doc.parse_warnings.append("fallback_used")
        doc.parser_name = f"pdf_{provider}"
        doc.parser_version = PARSER_VERSION
        return doc


def _pdf_parse_context() -> pdf_parsers.PdfParseContext:
    return pdf_parsers.PdfParseContext(
        parser_version=PARSER_VERSION,
        parse_error_cls=ParseError,
        finish_document=PdfParser().finish,
        split_authors=_split_authors,
        split_special_line=_split_special_line,
        classify_heading_line=_classify_heading_line,
        classify_semantic_line=_classify_semantic_line,
        is_reference_heading=_is_reference_heading,
    )


def _parse_pypdf_pdf(path: Path) -> ParsedDocument:
    return pdf_parsers.parse_pypdf_pdf(path, _pdf_parse_context())


def _parse_docling_pdf(path: Path) -> ParsedDocument:
    return pdf_parsers.parse_docling_pdf(path, _pdf_parse_context())


def _fetch_grobid_enrichment(path: Path, url: str) -> Dict[str, Any]:
    return pdf_parsers.fetch_grobid_enrichment(path, url, parse_error_cls=ParseError)


def _merge_grobid_enrichment(doc: ParsedDocument, enrichment: Dict[str, Any]) -> None:
    pdf_parsers.merge_grobid_enrichment(doc, enrichment)


def _parse_grobid_tei(tei: str) -> Dict[str, Any]:
    return pdf_parsers.parse_grobid_tei(tei, parse_error_cls=ParseError)


SEMANTIC_BLOCK_KINDS = {"abstract", "keywords", "figure", "table", "reference", "formula"}
_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千万0-9]+章\s*[\w\u4e00-\u9fff（）()《》:：、，,\- ]{0,80}$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})[\s、.．]+(.{1,80})$")
_CHINESE_ORDER_HEADING_RE = re.compile(r"^[一二三四五六七八九十]+[、.．]\s*.{1,80}$")
_REFERENCE_LINE_RE = re.compile(r"^(\[\d+\]|\d+[.)、]\s+).+")


def normalize_blocks(blocks: List[ParsedBlock]) -> tuple[List[ParsedBlock], List[str]]:
    """Split parser output into lightweight paper-structure blocks."""
    normalized: List[ParsedBlock] = []
    warnings: List[str] = []
    in_references = False
    context_kind = ""
    split_count = 0

    for block in blocks:
        if block.kind == "heading":
            heading = compact_whitespace(block.heading or block.text)
            if not heading:
                continue
            level = block.level or _infer_heading_level(heading)
            normalized.append(_copy_block(block, kind="heading", text="", heading=heading, level=level))
            in_references = _is_reference_heading(heading)
            context_kind = _context_for_heading(heading)
            continue

        if block.kind in SEMANTIC_BLOCK_KINDS:
            normalized.append(block)
            in_references = block.kind == "reference" or in_references
            context_kind = block.kind if block.kind in {"abstract", "keywords", "reference"} else context_kind
            continue

        emitted, in_references, context_kind = _normalize_paragraph_block(block, in_references, context_kind)
        if len(emitted) > 1:
            split_count += len(emitted) - 1
        normalized.extend(emitted)

    if split_count:
        warnings.append(f"normalize_split_blocks:{split_count}")
    return normalized, warnings


def _normalize_paragraph_block(
    block: ParsedBlock,
    in_references: bool,
    context_kind: str,
) -> tuple[List[ParsedBlock], bool, str]:
    emitted: List[ParsedBlock] = []
    buffer: List[str] = []
    table_rows: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        text = compact_whitespace(" ".join(buffer))
        if text:
            kind = _contextual_paragraph_kind(in_references, context_kind)
            emitted.append(_copy_block(block, kind=kind, text=text, heading="", level=0))
        buffer = []

    def flush_table_rows() -> None:
        nonlocal table_rows
        if table_rows:
            emitted.append(_copy_block(block, kind="table", text="\n".join(table_rows), heading="", level=0))
        table_rows = []

    lines = [compact_whitespace(line) for line in block.text.replace("\r\n", "\n").replace("\r", "\n").splitlines()]
    lines = [line for line in lines if line]
    if not lines and block.text.strip():
        lines = [compact_whitespace(block.text)]

    for line in lines:
        special = _split_special_line(line)
        if special:
            flush_buffer()
            flush_table_rows()
            heading, level, body_kind, body = special
            emitted.append(_copy_block(block, kind="heading", text="", heading=heading, level=level))
            in_references = _is_reference_heading(heading)
            context_kind = _context_for_heading(heading)
            if body:
                emitted.append(_copy_block(block, kind=body_kind, text=body, heading="", level=0))
            continue

        heading = _classify_heading_line(line)
        if heading:
            flush_buffer()
            flush_table_rows()
            emitted.append(_copy_block(block, kind="heading", text="", heading=heading[0], level=heading[1]))
            in_references = _is_reference_heading(heading[0])
            context_kind = _context_for_heading(heading[0])
            continue

        semantic_kind = _classify_semantic_line(line, in_references)
        if semantic_kind:
            flush_buffer()
            flush_table_rows()
            emitted.append(_copy_block(block, kind=semantic_kind, text=line, heading="", level=0))
            if semantic_kind == "table":
                context_kind = "table"
            continue

        if context_kind == "table" and _looks_like_table_data_line(line):
            flush_buffer()
            table_rows.append(line)
            continue

        flush_table_rows()
        buffer.append(line)

    flush_table_rows()
    flush_buffer()
    return emitted, in_references, context_kind


def _split_special_line(line: str) -> Optional[tuple[str, int, str, str]]:
    abstract = re.match(r"^(摘\s*要|Abstract)\s*[:：]?\s*(.*)$", line, re.IGNORECASE)
    if abstract and (abstract.group(2) or compact_whitespace(abstract.group(1)) in {"摘要", "摘 要", "Abstract"}):
        return "摘要", 1, "abstract", compact_whitespace(abstract.group(2))

    keywords = re.match(r"^(关键词|关键字|Keywords?)\s*[:：]?\s*(.*)$", line, re.IGNORECASE)
    if keywords:
        return "关键词", 1, "keywords", compact_whitespace(keywords.group(2))

    references = re.match(r"^(参考文献|References|Bibliography)\s*[:：]?\s*$", line, re.IGNORECASE)
    if references:
        return "参考文献", 1, "reference", ""
    return None


def _classify_heading_line(line: str) -> Optional[tuple[str, int]]:
    text = compact_whitespace(line)
    if not text or len(text) > 96:
        return None
    if _CHAPTER_RE.match(text):
        return text, 1
    if _is_reference_heading(text):
        return "参考文献", 1
    if _is_conclusion_heading(text):
        return text, 1
    numbered = _NUMBERED_HEADING_RE.match(text)
    if numbered:
        number = numbered.group(1)
        if _looks_like_heading_text(numbered.group(2)):
            return text, min(number.count(".") + 1, 4)
    if _CHINESE_ORDER_HEADING_RE.match(text) and _looks_like_heading_text(text):
        return text, 2
    return None


def _classify_semantic_line(line: str, in_references: bool) -> str:
    if _is_figure_line(line):
        return "figure"
    if _is_table_line(line):
        return "table"
    if in_references or _REFERENCE_LINE_RE.match(line):
        return "reference"
    return ""


def _context_for_heading(heading: str) -> str:
    normalized = re.sub(r"\s+", "", heading).lower()
    if normalized in {"摘要", "abstract"}:
        return "abstract"
    if normalized in {"关键词", "关键字", "keywords", "keyword"}:
        return "keywords"
    if normalized in {"参考文献", "references", "bibliography"}:
        return "reference"
    return ""


def _contextual_paragraph_kind(in_references: bool, context_kind: str) -> str:
    if in_references:
        return "reference"
    if context_kind in {"abstract", "keywords"}:
        return context_kind
    return "paragraph"


def _copy_block(
    block: ParsedBlock,
    *,
    kind: str,
    text: str,
    heading: str,
    level: int,
) -> ParsedBlock:
    return ParsedBlock(
        kind=kind,
        text=text,
        heading=heading,
        level=level,
        page=block.page,
        char_start=block.char_start,
        char_end=block.char_end,
        bbox=block.bbox,
        layout_block_id=block.layout_block_id,
        caption_id=block.caption_id,
        confidence=block.confidence,
        source_parser=block.source_parser,
    )


def _infer_heading_level(heading: str) -> int:
    classified = _classify_heading_line(heading)
    return classified[1] if classified else 1


def _is_reference_heading(text: str) -> bool:
    return bool(re.match(r"^(参考文献|References|Bibliography)\s*[:：]?$", compact_whitespace(text), re.IGNORECASE))


def _is_conclusion_heading(text: str) -> bool:
    normalized = compact_whitespace(text)
    return bool(re.match(r"^(结论|总结|讨论|致谢|Conclusion|Discussion)\s*[:：]?$", normalized, re.IGNORECASE))


def _looks_like_heading_text(text: str) -> bool:
    stripped = compact_whitespace(text)
    if not stripped or len(stripped) > 72:
        return False
    return not re.search(r"[。！？!?；;]$", stripped)


def _is_figure_line(text: str) -> bool:
    return bool(re.match(r"^(图\s*\d+|Figure\s+\d+)", compact_whitespace(text), re.IGNORECASE))





PARSERS: List[ParserAdapter] = [
    MarkdownParser(),
    PlainTextParser(),
    PdfParser(),
    DocxParser(),
    HtmlParser(),
]


def get_parser(path: Path) -> ParserAdapter:
    for parser in PARSERS:
        if parser.supports(path):
            return parser
    raise ParseError(f"Unsupported file type: {path.suffix.lower()}")


def parser_identity_for_path(path: Path, pdf_parser: Optional[str] = None) -> tuple[str, str]:
    parser = get_parser(path)
    if isinstance(parser, PdfParser):
        return f"pdf_{resolve_pdf_parser(pdf_parser)}", PARSER_VERSION
    return parser.name, parser.version


def parse_document(path: Path, pdf_parser: Optional[str] = None) -> ParsedDocument:
    parser = get_parser(path)
    if isinstance(parser, PdfParser):
        return parser.parse(path, pdf_parser=pdf_parser)
    return parser.parse(path)


def blocks_to_markdown(blocks: List[ParsedBlock], title: str) -> str:
    parts: List[str] = []
    has_title_heading = False
    for block in blocks:
        if block.kind == "heading":
            level = max(1, min(block.level or 1, 6))
            parts.append(f"{'#' * level} {block.heading}".strip())
            if level == 1 and block.heading == title:
                has_title_heading = True
            continue
        if block.text.strip():
            parts.append(block.text.strip())
    if title and not has_title_heading:
        parts.insert(0, f"# {title}")
    return "\n\n".join(parts).strip() + ("\n" if parts else "")


def extract_references(raw_text: str) -> Dict[str, Any]:
    match = re.search(r"(参考文献|References)\s*[:：]?\s*(.+)$", raw_text, re.IGNORECASE | re.DOTALL)
    references: List[Dict[str, Any]] = []
    if match:
        tail = match.group(2)
        lines = [compact_whitespace(line) for line in tail.splitlines() if compact_whitespace(line)]
        for index, line in enumerate(lines[:200], start=1):
            references.append({"ref_id": f"ref_{index}", "raw": line})
    return {
        "schema": "references.v0",
        "status": "extracted" if references else "not_extracted",
        "references": references,
        "citation_contexts": [],
    }


def _block_to_dict(block: ParsedBlock, index: int) -> Dict[str, Any]:
    return {
        "block_id": f"block_{index}",
        "kind": block.kind,
        "text": block.text,
        "heading": block.heading,
        "level": block.level,
        "page": block.page,
        "char_start": block.char_start,
        "char_end": block.char_end,
        "bbox": block.bbox,
        "layout_block_id": block.layout_block_id,
        "caption_id": block.caption_id,
        "confidence": block.confidence,
        "source_parser": block.source_parser,
    }


def _enrich_metadata(title: str, raw_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(metadata)
    enriched["title"] = enriched.get("title") or title
    enriched["authors"] = enriched.get("authors") or []
    enriched["year"] = enriched.get("year") or _extract_year(raw_text)
    enriched["venue"] = enriched.get("venue") or ""
    enriched["doi"] = enriched.get("doi") or _extract_doi(raw_text)
    enriched["abstract"] = enriched.get("abstract") or _extract_abstract(raw_text)
    enriched["keywords"] = enriched.get("keywords") or _extract_keywords(raw_text)
    return enriched


def _extract_year(text: str) -> Optional[int]:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group(0)) if match else None


def _extract_doi(text: str) -> str:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
    return match.group(0) if match else ""


def _extract_abstract(text: str) -> str:
    patterns = [
        r"(?:^|\n)\s*摘要\s*[:：]?\s*(.{40,900}?)(?:\n\s*(?:关键词|关键字|Abstract|1[.、\s]|一、)|$)",
        r"(?:^|\n)\s*Abstract\s*[:：]?\s*(.{40,900}?)(?:\n\s*(?:Keywords|1[.、\s])|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return _clean_abstract_text(match.group(1))[:1200]
    return ""


def _extract_keywords(text: str) -> List[str]:
    match = re.search(r"(?:关键词|关键字|Keywords)\s*[:：]\s*(.+)", text, re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1).strip()
    parts = re.split(r"[;,，；、\s]+", raw)
    return [part for part in (compact_whitespace(part) for part in parts) if part][:20]


def _extract_abstract_from_blocks(blocks: List[ParsedBlock]) -> str:
    parts = [block.text for block in blocks if block.kind == "abstract" and block.text.strip()]
    return _clean_abstract_text(" ".join(parts))[:1200]


def _extract_keywords_from_blocks(blocks: List[ParsedBlock]) -> List[str]:
    raw = " ".join(block.text for block in blocks if block.kind == "keywords")
    if not raw:
        return []
    parts = re.split(r"[;,，；、\s]+", raw)
    return [part for part in (compact_whitespace(part) for part in parts) if part][:20]


def _split_authors(raw: str) -> List[str]:
    parts = re.split(r"[,;，；、/]+", raw)
    return [part for part in (compact_whitespace(part) for part in parts) if part]


def _clean_abstract_text(text: str) -> str:
    cleaned = compact_whitespace(text)
    cleaned = re.sub(r"^(?:[IVXLC]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ])\s+", "", cleaned)
    cleaned = re.sub(r"扬州大学博士学位论文\s*(?:[IVXLC]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ])?", "", cleaned)
    return compact_whitespace(cleaned)
