from __future__ import annotations

import html
import importlib.util
import os
import re
import urllib.error
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

from .models import ParsedBlock, ParsedDocument
from .utils import compact_whitespace, read_text_lossy, split_paragraphs


PARSER_VERSION = "0.14.0"
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
        table_items = build_visual_items(layout_blocks, "table")
        figure_items = build_visual_items(layout_blocks, "figure")
        reference_sections = build_reference_sections(layout_blocks, references_payload)
        structured_payload = dict(structured or {
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


def _ensure_layout_metadata(blocks: List[ParsedBlock], source_parser: str) -> None:
    counters = Counter()
    for index, block in enumerate(blocks, start=1):
        if not block.source_parser:
            block.source_parser = source_parser
        if not block.layout_block_id:
            block.layout_block_id = f"layout_{index:04d}"
        if block.kind in {"figure", "table"} and not block.caption_id:
            counters[block.kind] += 1
            prefix = "fig" if block.kind == "figure" else "table"
            block.caption_id = f"{prefix}_{counters[block.kind]:03d}"


def build_layout_blocks(blocks: List[ParsedBlock], source_parser: str = "") -> List[Dict[str, Any]]:
    current_section: List[str] = []
    layout_blocks: List[Dict[str, Any]] = []
    _ensure_layout_metadata(blocks, source_parser or "unknown")
    for block in blocks:
        text = compact_whitespace(block.heading or block.text)
        if block.kind == "heading" and text:
            level = max(1, block.level or 1)
            current_section = current_section[: level - 1]
            current_section.append(text)
        if not text:
            continue
        layout_blocks.append(
            {
                "schema": "layout_block.v1",
                "block_id": block.layout_block_id,
                "type": block.kind,
                "text": text,
                "page": block.page,
                "bbox": block.bbox,
                "section_path": list(current_section),
                "caption_id": block.caption_id,
                "confidence": round(float(block.confidence), 3),
                "source_parser": block.source_parser or source_parser or "unknown",
            }
        )
    return layout_blocks


def build_visual_items(layout_blocks: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for index, block in enumerate((item for item in layout_blocks if item.get("type") == kind), start=1):
        caption = compact_whitespace(str(block.get("text") or ""))
        item_id = str(block.get("caption_id") or f"{kind}_{index:03d}")
        items.append(
            {
                "schema": f"{kind}.v1",
                "id": item_id,
                "caption_id": item_id,
                "layout_block_id": block.get("block_id"),
                "caption": caption,
                "text": caption,
                "page": block.get("page"),
                "bbox": block.get("bbox"),
                "section_path": block.get("section_path") or [],
                "confidence": block.get("confidence", 1.0),
                "source_parser": block.get("source_parser") or "",
            }
        )
    return items


def build_reference_sections(
    layout_blocks: List[Dict[str, Any]],
    references: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    reference_blocks = [block for block in layout_blocks if block.get("type") == "reference"]
    pages = [int(block["page"]) for block in reference_blocks if isinstance(block.get("page"), int)]
    if reference_blocks:
        return [
            {
                "schema": "reference_section.v1",
                "section_id": "references",
                "source": _dominant_source(reference_blocks),
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "block_ids": [str(block.get("block_id") or "") for block in reference_blocks if block.get("block_id")],
                "item_count": len(reference_blocks),
                "references_count": len((references or {}).get("references") or []),
            }
        ]
    ref_items = (references or {}).get("references") or []
    if ref_items:
        return [
            {
                "schema": "reference_section.v1",
                "section_id": "references",
                "source": str((references or {}).get("source") or "references"),
                "page_start": None,
                "page_end": None,
                "block_ids": [],
                "item_count": len(ref_items),
                "references_count": len(ref_items),
            }
        ]
    return []


def _dominant_source(blocks: List[Dict[str, Any]]) -> str:
    sources = [str(block.get("source_parser") or "") for block in blocks if block.get("source_parser")]
    if not sources:
        return ""
    return Counter(sources).most_common(1)[0][0]


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


def _parse_pypdf_pdf(path: Path) -> ParsedDocument:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional package
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found,no-redef]
        except Exception as fallback_exc:  # pragma: no cover - depends on optional package
            raise ParseError(
                "PDF parsing requires optional dependency 'pypdf'. Install with: uv sync --extra pdf"
            ) from fallback_exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ParseError(f"Cannot read PDF: {exc}") from exc

    page_lines: List[Tuple[int, List[str]]] = []
    extract_errors: List[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = ""
            extract_errors.append(f"page_{page_number}:{exc}")
        page_lines.append((page_number, text.splitlines()))

    cleaned_pages, noise_removed_count = _clean_pdf_page_lines(page_lines)
    blocks = []
    page_texts: List[str] = []
    for page_number, lines in cleaned_pages:
        if lines:
            page_texts.append("\n".join(lines))
            blocks.extend(_pypdf_blocks_from_lines(lines, page_number))
    raw_text = "\n\n".join(page_texts)
    metadata = {
        "source_format": "pdf",
        "pages": len(reader.pages),
        "pdf_parser": "pypdf",
        "noise_removed_count": noise_removed_count,
    }
    if extract_errors:
        metadata["page_extract_errors"] = extract_errors
    pdf_metadata = getattr(reader, "metadata", None)
    title = path.stem
    if pdf_metadata:
        candidate_title = getattr(pdf_metadata, "title", None) or _metadata_get(pdf_metadata, "/Title")
        if candidate_title:
            title = compact_whitespace(str(candidate_title))
        author = getattr(pdf_metadata, "author", None) or _metadata_get(pdf_metadata, "/Author")
        if author:
            metadata["authors"] = _split_authors(str(author))

    warnings = []
    if extract_errors:
        warnings.append(f"page_extract_errors:{len(extract_errors)}")
    if not raw_text.strip():
        warnings.append("scanned_pdf_or_empty_text")

    return PdfParser().finish(
        path,
        title=title or path.stem,
        raw_text=raw_text,
        blocks=blocks,
        metadata=metadata,
        warnings=warnings,
        parser_name="pdf_pypdf",
        parser_version=PARSER_VERSION,
    )


def _clean_pdf_page_lines(page_lines: List[Tuple[int, List[str]]]) -> Tuple[List[Tuple[int, List[str]]], int]:
    normalized_pages: List[Tuple[int, List[str]]] = []
    line_counts: Counter[str] = Counter()
    for page_number, lines in page_lines:
        cleaned = [compact_whitespace(line) for line in lines]
        cleaned = [line for line in cleaned if line]
        normalized_pages.append((page_number, cleaned))
        for line in cleaned:
            key = _noise_key(line)
            if key:
                line_counts[key] += 1

    repeated_noise = {
        line
        for line, count in line_counts.items()
        if count >= 2 and len(line) <= 120 and not _classify_heading_line(line) and not _classify_semantic_line(line, False)
    }
    result: List[Tuple[int, List[str]]] = []
    removed = 0
    for page_number, lines in normalized_pages:
        kept: List[str] = []
        for line in lines:
            key = _noise_key(line)
            if _is_pdf_noise_line(line) or key in repeated_noise:
                removed += 1
                continue
            kept.append(line)
        result.append((page_number, kept))
    return result, removed


def _noise_key(line: str) -> str:
    text = compact_whitespace(line)
    text = re.sub(r"\d+", "#", text)
    return text.lower()


def _is_pdf_noise_line(line: str) -> bool:
    text = compact_whitespace(line)
    if not text:
        return True
    if re.match(r"^[-—–]?\s*\d+\s*[-—–]?$", text):
        return True
    if re.match(r"^第\s*\d+\s*页(?:\s*/\s*共\s*\d+\s*页)?$", text):
        return True
    if re.search(r"\.{5,}\s*\d+$", text):
        return True
    if re.search(r"\b(?:doi|issn|cn)\b\s*[:：]", text, re.IGNORECASE):
        return True
    if re.search(r"(收稿日期|基金项目|作者简介|版权|©|Copyright|All rights reserved)", text, re.IGNORECASE):
        return True
    return False


def _pypdf_blocks_from_lines(lines: List[str], page_number: int) -> List[ParsedBlock]:
    blocks: List[ParsedBlock] = []
    paragraph: List[str] = []
    in_references = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        text = compact_whitespace(" ".join(paragraph))
        if text:
            blocks.append(
                ParsedBlock(
                    kind="paragraph",
                    text=text,
                    page=page_number,
                    source_parser="pypdf",
                    confidence=0.72,
                )
            )
        paragraph = []

    for line in lines:
        text = compact_whitespace(line)
        if not text:
            flush_paragraph()
            continue

        special = _split_special_line(text)
        if special:
            flush_paragraph()
            heading, level, body_kind, body = special
            blocks.append(
                ParsedBlock(
                    kind="heading",
                    text="",
                    heading=heading,
                    level=level,
                    page=page_number,
                    source_parser="pypdf",
                    confidence=0.88,
                )
            )
            in_references = _is_reference_heading(heading)
            if body:
                blocks.append(
                    ParsedBlock(
                        kind=body_kind,
                        text=body,
                        page=page_number,
                        source_parser="pypdf",
                        confidence=0.82,
                    )
                )
            continue

        heading = _classify_heading_line(text)
        if heading:
            flush_paragraph()
            blocks.append(
                ParsedBlock(
                    kind="heading",
                    text="",
                    heading=heading[0],
                    level=heading[1],
                    page=page_number,
                    source_parser="pypdf",
                    confidence=0.86,
                )
            )
            in_references = _is_reference_heading(heading[0])
            continue

        semantic_kind = _classify_semantic_line(text, in_references)
        if semantic_kind:
            flush_paragraph()
            confidence = 0.86 if semantic_kind in {"figure", "table", "reference"} else 0.78
            blocks.append(
                ParsedBlock(
                    kind=semantic_kind,
                    text=text,
                    page=page_number,
                    source_parser="pypdf",
                    confidence=confidence,
                )
            )
            continue

        if _is_formula_line(text):
            flush_paragraph()
            blocks.append(
                ParsedBlock(
                    kind="formula",
                    text=text,
                    page=page_number,
                    source_parser="pypdf",
                    confidence=0.68,
                )
            )
            continue

        paragraph.append(text)

    flush_paragraph()
    return blocks


def _is_formula_line(line: str) -> bool:
    text = compact_whitespace(line)
    if len(text) > 120:
        return False
    if re.search(r"[=≈≤≥∑∏√∫]", text) and re.search(r"[A-Za-zα-ωΑ-Ω]", text):
        return True
    return bool(re.match(r"^\(?\d+\)?\s*[A-Za-z]\s*=", text))


def _metadata_get(metadata: Any, key: str) -> Any:
    try:
        return metadata.get(key)
    except Exception:
        return None


def _parse_docling_pdf(path: Path) -> ParsedDocument:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ParseError("Docling is not installed. Install with: uv sync --extra docling") from exc

    try:
        result = DocumentConverter().convert(str(path))
    except Exception as exc:  # pragma: no cover - external parser behavior
        raise ParseError(f"Docling conversion failed: {exc}") from exc

    document = getattr(result, "document", result)
    structured = _docling_structured_payload(document)
    body_md = _docling_markdown(document)
    return _parsed_document_from_pdf_payload(
        path,
        {
            "title": _title_from_structured(structured) or path.stem,
            "raw_text": body_md,
            "body_md": body_md,
            "structured": structured,
            "blocks": _blocks_from_docling_structured(structured, body_md),
            "metadata": {
                "source_format": "pdf",
                "pdf_parser": "docling",
                "pages": _page_count_from_structured(structured),
            },
        },
        parser_name="pdf_docling",
    )


def _docling_structured_payload(document: Any) -> Dict[str, Any]:
    for method in ("export_to_dict", "to_dict", "as_dict"):
        candidate = getattr(document, method, None)
        if callable(candidate):
            try:
                payload = candidate()
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    if isinstance(document, dict):
        return document
    return {"schema": "structured.v0", "docling_repr": str(document)}


def _docling_markdown(document: Any) -> str:
    for method in ("export_to_markdown", "to_markdown"):
        candidate = getattr(document, method, None)
        if callable(candidate):
            try:
                text = candidate()
            except Exception:
                continue
            if text:
                return str(text)
    return ""


def _parsed_document_from_pdf_payload(path: Path, payload: Dict[str, Any], parser_name: str) -> ParsedDocument:
    blocks = payload.get("blocks") or []
    parsed_blocks = [
        block if isinstance(block, ParsedBlock) else _block_from_payload(block)
        for block in blocks
        if isinstance(block, (ParsedBlock, dict))
    ]
    raw_text = str(payload.get("raw_text") or "")
    if not raw_text:
        raw_text = "\n\n".join(
            block.heading if block.kind == "heading" else block.text
            for block in parsed_blocks
            if block.heading or block.text
        )
    if not parsed_blocks and raw_text:
        parsed_blocks = [ParsedBlock(kind="paragraph", text=part) for part in split_paragraphs(raw_text)]
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("source_format", "pdf")
    metadata.setdefault("pdf_parser", parser_name.removeprefix("pdf_"))
    structured = dict(payload.get("structured") or {})
    references = payload.get("references")
    return PdfParser().finish(
        path,
        title=str(payload.get("title") or metadata.get("title") or path.stem),
        raw_text=raw_text,
        blocks=parsed_blocks,
        metadata=metadata,
        body_md=str(payload.get("body_md") or ""),
        references=references if isinstance(references, dict) else None,
        structured=structured if structured else None,
        warnings=list(payload.get("warnings") or []),
        parser_name=parser_name,
        parser_version=PARSER_VERSION,
    )


def _block_from_payload(payload: Dict[str, Any]) -> ParsedBlock:
    kind = str(payload.get("kind") or payload.get("type") or payload.get("label") or "paragraph").lower()
    if kind in {"section_header", "title", "heading"}:
        kind = "heading"
    if kind in {"picture", "image"}:
        kind = "figure"
    heading = str(payload.get("heading") or "")
    text = str(payload.get("text") or payload.get("content") or payload.get("caption") or "")
    if kind == "heading" and not heading:
        heading = text
        text = ""
    raw_level = payload.get("level")
    try:
        level = int(raw_level) if raw_level is not None else (1 if kind == "heading" else 0)
    except (TypeError, ValueError):
        level = 1 if kind == "heading" else 0
    return ParsedBlock(
        kind=kind,
        text=text,
        heading=heading,
        level=level,
        page=_page_from_payload(payload),
        char_start=payload.get("char_start"),
        char_end=payload.get("char_end"),
        bbox=_bbox_from_payload(payload),
        layout_block_id=str(payload.get("block_id") or payload.get("layout_block_id") or payload.get("self_ref") or ""),
        caption_id=str(payload.get("caption_id") or payload.get("caption_ref") or ""),
        confidence=_confidence_from_payload(payload),
        source_parser=str(payload.get("source_parser") or payload.get("source") or "docling"),
    )


def _blocks_from_docling_structured(structured: Dict[str, Any], markdown: str) -> List[ParsedBlock]:
    blocks: List[ParsedBlock] = []
    raw_blocks = structured.get("blocks")
    if isinstance(raw_blocks, list):
        for item in raw_blocks:
            if isinstance(item, dict):
                blocks.append(_block_from_payload(item))
    raw_texts = structured.get("texts")
    if not blocks and isinstance(raw_texts, list):
        for item in raw_texts:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("type") or "").lower()
            text = str(item.get("text") or item.get("orig") or item.get("content") or "").strip()
            if not text:
                continue
            kind = "heading" if label in {"section_header", "title"} else "paragraph"
            blocks.append(
                ParsedBlock(
                    kind=kind,
                    text="" if kind == "heading" else text,
                    heading=text if kind == "heading" else "",
                    level=1 if label == "title" else 2 if kind == "heading" else 0,
                    page=_page_from_payload(item),
                    bbox=_bbox_from_payload(item),
                    layout_block_id=str(item.get("block_id") or item.get("self_ref") or ""),
                    confidence=_confidence_from_payload(item),
                    source_parser="docling",
                )
            )
    for item in structured.get("tables") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("caption") or item.get("name") or "table").strip()
            blocks.append(
                ParsedBlock(
                    kind="table",
                    text=text,
                    page=_page_from_payload(item),
                    bbox=_bbox_from_payload(item),
                    layout_block_id=str(item.get("block_id") or item.get("self_ref") or ""),
                    caption_id=str(item.get("caption_id") or ""),
                    confidence=_confidence_from_payload(item),
                    source_parser="docling",
                )
            )
    for item in structured.get("figures") or structured.get("pictures") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("caption") or item.get("name") or "figure").strip()
            blocks.append(
                ParsedBlock(
                    kind="figure",
                    text=text,
                    page=_page_from_payload(item),
                    bbox=_bbox_from_payload(item),
                    layout_block_id=str(item.get("block_id") or item.get("self_ref") or ""),
                    caption_id=str(item.get("caption_id") or ""),
                    confidence=_confidence_from_payload(item),
                    source_parser="docling",
                )
            )
    if not blocks and markdown:
        blocks = [ParsedBlock(kind="paragraph", text=part) for part in split_paragraphs(markdown)]
    return blocks


def _page_from_payload(payload: Dict[str, Any]) -> Optional[int]:
    for key in ("page", "page_no", "page_number"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    prov = payload.get("prov")
    if isinstance(prov, list) and prov and isinstance(prov[0], dict):
        return _page_from_payload(prov[0])
    return None


def _bbox_from_payload(payload: Dict[str, Any]) -> Optional[List[float]]:
    for key in ("bbox", "box", "bounding_box"):
        bbox = payload.get(key)
        parsed = _parse_bbox(bbox)
        if parsed:
            return parsed
    prov = payload.get("prov")
    if isinstance(prov, list) and prov and isinstance(prov[0], dict):
        return _bbox_from_payload(prov[0])
    return None


def _parse_bbox(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(item) for item in value[:4]]
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        keys = ("x0", "y0", "x1", "y1")
        if all(key in value for key in keys):
            try:
                return [float(value[key]) for key in keys]
            except (TypeError, ValueError):
                return None
        keys = ("l", "t", "r", "b")
        if all(key in value for key in keys):
            try:
                return [float(value[key]) for key in keys]
            except (TypeError, ValueError):
                return None
        if all(key in value for key in ("left", "top", "right", "bottom")):
            try:
                return [float(value[key]) for key in ("left", "top", "right", "bottom")]
            except (TypeError, ValueError):
                return None
    return None


def _confidence_from_payload(payload: Dict[str, Any]) -> float:
    for key in ("confidence", "score", "probability"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return 0.95


def _page_count_from_structured(structured: Dict[str, Any]) -> Optional[int]:
    pages = structured.get("pages")
    if isinstance(pages, list):
        return len(pages)
    if isinstance(pages, int):
        return pages
    max_page = 0
    for collection in (structured.get("blocks"), structured.get("texts"), structured.get("tables"), structured.get("figures")):
        if not isinstance(collection, list):
            continue
        for item in collection:
            if isinstance(item, dict):
                page = _page_from_payload(item)
                if page:
                    max_page = max(max_page, page)
    return max_page or None


def _title_from_structured(structured: Dict[str, Any]) -> str:
    for key in ("title", "name"):
        value = structured.get(key)
        if isinstance(value, str) and value.strip():
            return compact_whitespace(value)
    for item in structured.get("texts") or structured.get("blocks") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("kind") or item.get("type") or "").lower()
        if label in {"title", "document_title"}:
            return compact_whitespace(str(item.get("text") or item.get("heading") or ""))
    return ""


def _fetch_grobid_enrichment(path: Path, url: str) -> Dict[str, Any]:
    endpoint = url.rstrip("/") + "/api/processFulltextDocument"
    boundary = "----kb-agent-grobid-boundary"
    data = path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="input"; filename="{path.name}"\r\n'.encode("utf-8"),
            b"Content-Type: application/pdf\r\n\r\n",
            data,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            tei = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise ParseError(f"GROBID request failed: {exc}") from exc
    return _parse_grobid_tei(tei)


def _merge_grobid_enrichment(doc: ParsedDocument, enrichment: Dict[str, Any]) -> None:
    metadata = enrichment.get("metadata") or {}
    for key in ("title", "authors", "year", "venue", "doi", "abstract"):
        value = metadata.get(key)
        if value:
            doc.metadata[key] = value
            if key == "title":
                doc.title = str(value)
    references = enrichment.get("references") or []
    if references:
        doc.references = {
            "schema": "references.v0",
            "status": "extracted",
            "source": "grobid",
            "references": references,
            "citation_contexts": doc.references.get("citation_contexts", []),
        }
        doc.structured["reference_sections"] = build_reference_sections(
            doc.structured.get("layout_blocks") or [],
            doc.references,
        )
        doc.structured["reference_section_count"] = len(doc.structured.get("reference_sections") or [])
    doc.metadata["grobid_enriched"] = bool(metadata or references)


def _parse_grobid_tei(tei: str) -> Dict[str, Any]:
    try:
        root = ElementTree.fromstring(tei)
    except ElementTree.ParseError as exc:
        raise ParseError(f"Invalid GROBID TEI XML: {exc}") from exc

    title_stmt = _first_descendant(root, "titleStmt")
    source_desc = _first_descendant(root, "sourceDesc")
    top_bibl = _first_child(source_desc, "biblStruct") if source_desc is not None else None
    analytic = _first_child(top_bibl, "analytic") if top_bibl is not None else None
    monogr = _first_child(top_bibl, "monogr") if top_bibl is not None else None
    abstract_el = _first_descendant(root, "abstract")

    title = _text_from_first(title_stmt, "title") or _text_from_first(analytic, "title")
    metadata = {
        "title": title,
        "authors": _authors_from_element(analytic),
        "year": _year_from_element(top_bibl),
        "venue": _text_from_first(monogr, "title"),
        "doi": _idno_from_element(top_bibl, "doi"),
        "abstract": _element_text(abstract_el),
    }
    metadata = {key: value for key, value in metadata.items() if value}

    references: List[Dict[str, Any]] = []
    list_bibl = _first_descendant(root, "listBibl")
    if list_bibl is not None:
        for index, bibl in enumerate(_children(list_bibl, "biblStruct"), start=1):
            ref_analytic = _first_child(bibl, "analytic")
            ref_monogr = _first_child(bibl, "monogr")
            raw = _element_text(bibl)
            references.append(
                {
                    "ref_id": f"ref_{index}",
                    "raw": raw,
                    "authors": _authors_from_element(ref_analytic),
                    "title": _text_from_first(ref_analytic, "title") or _text_from_first(ref_monogr, "title"),
                    "year": _year_from_element(bibl),
                }
            )
    return {"metadata": metadata, "references": references}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: Optional[ElementTree.Element], name: str) -> List[ElementTree.Element]:
    if element is None:
        return []
    return [child for child in list(element) if _local_name(child.tag) == name]


def _first_child(element: Optional[ElementTree.Element], name: str) -> Optional[ElementTree.Element]:
    children = _children(element, name)
    return children[0] if children else None


def _first_descendant(element: Optional[ElementTree.Element], name: str) -> Optional[ElementTree.Element]:
    if element is None:
        return None
    for child in element.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _element_text(element: Optional[ElementTree.Element]) -> str:
    if element is None:
        return ""
    return compact_whitespace(" ".join(text for text in element.itertext() if text))


def _text_from_first(element: Optional[ElementTree.Element], name: str) -> str:
    return _element_text(_first_descendant(element, name))


def _idno_from_element(element: Optional[ElementTree.Element], wanted_type: str) -> str:
    if element is None:
        return ""
    for child in element.iter():
        if _local_name(child.tag) != "idno":
            continue
        id_type = str(child.attrib.get("type") or "").lower()
        if id_type == wanted_type.lower():
            return compact_whitespace(child.text or "")
    return ""


def _year_from_element(element: Optional[ElementTree.Element]) -> Optional[int]:
    if element is None:
        return None
    for child in element.iter():
        if _local_name(child.tag) != "date":
            continue
        for key in ("when", "from", "notBefore"):
            value = child.attrib.get(key)
            if value:
                match = re.search(r"\b(19|20)\d{2}\b", value)
                if match:
                    return int(match.group(0))
        match = re.search(r"\b(19|20)\d{2}\b", _element_text(child))
        if match:
            return int(match.group(0))
    return None


def _authors_from_element(element: Optional[ElementTree.Element]) -> List[str]:
    authors: List[str] = []
    for author in _children(element, "author"):
        pers = _first_descendant(author, "persName")
        if pers is not None:
            forenames = [_element_text(item) for item in _children(pers, "forename")]
            surname = _text_from_first(pers, "surname")
            name = compact_whitespace(" ".join([*forenames, surname]))
        else:
            name = _element_text(author)
        if name:
            authors.append(name)
    return authors


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

    def flush_buffer() -> None:
        nonlocal buffer
        text = compact_whitespace(" ".join(buffer))
        if text:
            kind = _contextual_paragraph_kind(in_references, context_kind)
            emitted.append(_copy_block(block, kind=kind, text=text, heading="", level=0))
        buffer = []

    lines = [compact_whitespace(line) for line in block.text.replace("\r\n", "\n").replace("\r", "\n").splitlines()]
    lines = [line for line in lines if line]
    if not lines and block.text.strip():
        lines = [compact_whitespace(block.text)]

    for line in lines:
        special = _split_special_line(line)
        if special:
            flush_buffer()
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
            emitted.append(_copy_block(block, kind="heading", text="", heading=heading[0], level=heading[1]))
            in_references = _is_reference_heading(heading[0])
            context_kind = _context_for_heading(heading[0])
            continue

        semantic_kind = _classify_semantic_line(line, in_references)
        if semantic_kind:
            flush_buffer()
            emitted.append(_copy_block(block, kind=semantic_kind, text=line, heading="", level=0))
            continue

        buffer.append(line)

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


def _is_table_line(text: str) -> bool:
    return bool(re.match(r"^(表\s*\d+|Table\s+\d+)", compact_whitespace(text), re.IGNORECASE))


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
