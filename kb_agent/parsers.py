from __future__ import annotations

import html
import importlib.util
import os
import re
import urllib.error
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

from .models import ParsedBlock, ParsedDocument
from .utils import compact_whitespace, read_text_lossy, split_paragraphs


PARSER_VERSION = "0.8.0"
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
        enriched = _enrich_metadata(title, raw_text, metadata or {})
        if not enriched.get("abstract"):
            enriched["abstract"] = _extract_abstract_from_blocks(normalized_blocks)
        if not enriched.get("keywords"):
            enriched["keywords"] = _extract_keywords_from_blocks(normalized_blocks)
        structured_payload = structured or {
            "schema": "structured.v0",
            "blocks": [_block_to_dict(block, index) for index, block in enumerate(normalized_blocks)],
            "tables": [],
            "figures": [],
            "formulas": [],
        }
        structured_payload.setdefault("blocks", [_block_to_dict(block, index) for index, block in enumerate(normalized_blocks)])
        structured_payload.setdefault("tables", [])
        structured_payload.setdefault("figures", [])
        structured_payload.setdefault("formulas", [])
        return ParsedDocument(
            title=enriched.get("title") or title or path.stem,
            file_type=self.file_type,
            raw_text=raw_text,
            blocks=normalized_blocks,
            metadata=enriched,
            body_md=body_md or blocks_to_markdown(normalized_blocks, title),
            structured=structured_payload,
            references=references or extract_references(raw_text),
            parser_name=parser_name or self.name,
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

    blocks: List[ParsedBlock] = []
    page_texts: List[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = ""
            blocks.append(ParsedBlock(kind="paragraph", text=f"[page_extract_error:{exc}]", page=page_number))
        text = text.strip()
        if not text:
            continue
        page_texts.append(text)
        blocks.append(ParsedBlock(kind="paragraph", text=text, page=page_number))

    raw_text = "\n\n".join(page_texts)
    metadata = {
        "source_format": "pdf",
        "pages": len(reader.pages),
        "pdf_parser": "pypdf",
    }
    pdf_metadata = getattr(reader, "metadata", None)
    title = path.stem
    if pdf_metadata:
        candidate_title = getattr(pdf_metadata, "title", None) or _metadata_get(pdf_metadata, "/Title")
        if candidate_title:
            title = compact_whitespace(str(candidate_title))
        author = getattr(pdf_metadata, "author", None) or _metadata_get(pdf_metadata, "/Author")
        if author:
            metadata["authors"] = _split_authors(str(author))

    return PdfParser().finish(
        path,
        title=title or path.stem,
        raw_text=raw_text,
        blocks=blocks,
        metadata=metadata,
        parser_name="pdf_pypdf",
        parser_version=PARSER_VERSION,
    )


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
    return ParsedBlock(
        kind=kind,
        text=text,
        heading=heading,
        level=int(payload.get("level") or 1 if kind == "heading" else 0),
        page=_page_from_payload(payload),
        char_start=payload.get("char_start"),
        char_end=payload.get("char_end"),
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
                )
            )
    for item in structured.get("tables") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("caption") or item.get("name") or "table").strip()
            blocks.append(ParsedBlock(kind="table", text=text, page=_page_from_payload(item)))
    for item in structured.get("figures") or structured.get("pictures") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("caption") or item.get("name") or "figure").strip()
            blocks.append(ParsedBlock(kind="figure", text=text, page=_page_from_payload(item)))
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


SEMANTIC_BLOCK_KINDS = {"abstract", "keywords", "figure", "table", "reference"}
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
