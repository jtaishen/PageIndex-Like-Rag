from __future__ import annotations

import html
import re
import zipfile
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

from .models import ParsedBlock, ParsedDocument
from .utils import compact_whitespace, read_text_lossy, split_paragraphs


PARSER_VERSION = "0.2.0"


class ParseError(RuntimeError):
    pass


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
        warnings: Optional[List[str]] = None,
    ) -> ParsedDocument:
        enriched = _enrich_metadata(title, raw_text, metadata or {})
        structured = {
            "schema": "structured.v0",
            "blocks": [_block_to_dict(block, index) for index, block in enumerate(blocks)],
            "tables": [],
            "figures": [],
            "formulas": [],
        }
        return ParsedDocument(
            title=enriched.get("title") or title or path.stem,
            file_type=self.file_type,
            raw_text=raw_text,
            blocks=blocks,
            metadata=enriched,
            body_md=body_md or blocks_to_markdown(blocks, title),
            structured=structured,
            references=references or extract_references(raw_text),
            parser_name=self.name,
            parser_version=self.version,
            parse_warnings=warnings or [],
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
    name = "pypdf"
    file_type = "pdf"
    suffixes = {".pdf"}

    def parse(self, path: Path) -> ParsedDocument:
        reader = None
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
        except Exception:
            try:
                from PyPDF2 import PdfReader  # type: ignore

                reader = PdfReader(str(path))
            except Exception as exc:
                raise ParseError(
                    "PDF parsing requires optional dependency pypdf. Install with: uv sync --extra pdf"
                ) from exc

        blocks: List[ParsedBlock] = []
        page_texts: List[str] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            page_texts.append(text)
            for paragraph in split_paragraphs(text):
                blocks.append(ParsedBlock(kind="paragraph", text=paragraph, page=index))

        raw_text = "\n\n".join(page_texts)
        title = path.stem
        metadata: Dict[str, Any] = {"source_format": "pdf", "pages": len(page_texts)}
        reader_metadata = getattr(reader, "metadata", None)
        if reader_metadata:
            candidate = getattr(reader_metadata, "title", None) or reader_metadata.get("/Title", "")
            if candidate:
                title = compact_whitespace(str(candidate))
            author = getattr(reader_metadata, "author", None) or reader_metadata.get("/Author", "")
            if author:
                metadata["authors"] = _split_authors(str(author))
        return self.finish(path, title=title or path.stem, raw_text=raw_text, blocks=blocks, metadata=metadata)


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


def parser_identity_for_path(path: Path) -> tuple[str, str]:
    parser = get_parser(path)
    return parser.name, parser.version


def parse_document(path: Path) -> ParsedDocument:
    return get_parser(path).parse(path)


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
            return compact_whitespace(match.group(1))[:1200]
    return ""


def _extract_keywords(text: str) -> List[str]:
    match = re.search(r"(?:关键词|关键字|Keywords)\s*[:：]\s*(.+)", text, re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1).strip()
    parts = re.split(r"[;,，；、\s]+", raw)
    return [part for part in (compact_whitespace(part) for part in parts) if part][:20]


def _split_authors(raw: str) -> List[str]:
    parts = re.split(r"[,;，；、/]+", raw)
    return [part for part in (compact_whitespace(part) for part in parts) if part]
