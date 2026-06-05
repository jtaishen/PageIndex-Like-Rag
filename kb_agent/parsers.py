from __future__ import annotations

import html
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import List
from xml.etree import ElementTree

from .models import ParsedBlock, ParsedDocument
from .utils import compact_whitespace, read_text_lossy, split_paragraphs


class ParseError(RuntimeError):
    pass


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


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _parse_markdown(path)
    if suffix == ".txt":
        return _parse_plain_text(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix in {".html", ".htm"}:
        return _parse_html(path)
    raise ParseError(f"Unsupported file type: {suffix}")


def _parse_markdown(path: Path) -> ParsedDocument:
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

    return ParsedDocument(
        title=title,
        file_type="markdown",
        raw_text=text,
        blocks=blocks,
        metadata={"parser": "markdown"},
    )


def _parse_plain_text(path: Path) -> ParsedDocument:
    text = read_text_lossy(path)
    blocks = [
        ParsedBlock(kind="paragraph", text=paragraph)
        for paragraph in split_paragraphs(text)
    ]
    return ParsedDocument(
        title=path.stem,
        file_type="text",
        raw_text=text,
        blocks=blocks,
        metadata={"parser": "plain_text"},
    )


def _parse_html(path: Path) -> ParsedDocument:
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
    return ParsedDocument(
        title=title or path.stem,
        file_type="html",
        raw_text=text,
        blocks=blocks,
        metadata={"parser": "html_parser"},
    )


def _parse_docx(path: Path) -> ParsedDocument:
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
    return ParsedDocument(
        title=compact_whitespace(title) or path.stem,
        file_type="docx",
        raw_text=raw_text,
        blocks=blocks,
        metadata={"parser": "docx_zip_xml"},
    )


def _parse_pdf(path: Path) -> ParsedDocument:
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
    metadata = getattr(reader, "metadata", None)
    if metadata:
        candidate = getattr(metadata, "title", None) or metadata.get("/Title", "")
        if candidate:
            title = compact_whitespace(str(candidate))
    return ParsedDocument(
        title=title or path.stem,
        file_type="pdf",
        raw_text=raw_text,
        blocks=blocks,
        metadata={"parser": "pypdf", "pages": len(page_texts)},
    )

