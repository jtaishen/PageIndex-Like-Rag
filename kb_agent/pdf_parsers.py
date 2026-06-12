from __future__ import annotations

import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree

from .models import ParsedBlock, ParsedDocument
from .parser_artifacts import (
    bbox_from_payload,
    confidence_from_payload,
    page_from_payload,
    build_reference_sections,
)
from .utils import compact_whitespace, split_paragraphs


@dataclass(frozen=True)
class PdfParseContext:
    parser_version: str
    parse_error_cls: type[Exception]
    finish_document: Callable[..., ParsedDocument]
    split_authors: Callable[[str], List[str]]
    split_special_line: Callable[[str], Optional[tuple[str, int, str, str]]]
    classify_heading_line: Callable[[str], Optional[tuple[str, int]]]
    classify_semantic_line: Callable[[str, bool], str]
    is_reference_heading: Callable[[str], bool]


def parse_pypdf_pdf(path: Path, context: PdfParseContext) -> ParsedDocument:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - depends on optional package
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found,no-redef]
        except Exception as fallback_exc:  # pragma: no cover - depends on optional package
            raise context.parse_error_cls(
                "PDF parsing requires optional dependency 'pypdf'. Install with: uv sync --extra pdf"
            ) from fallback_exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise context.parse_error_cls(f"Cannot read PDF: {exc}") from exc

    page_lines: List[Tuple[int, List[str]]] = []
    extract_errors: List[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = ""
            extract_errors.append(f"page_{page_number}:{exc}")
        page_lines.append((page_number, text.splitlines()))

    cleaned_pages, noise_removed_count = _clean_pdf_page_lines(page_lines, context)
    blocks = []
    page_texts: List[str] = []
    for page_number, lines in cleaned_pages:
        if lines:
            page_texts.append("\n".join(lines))
            blocks.extend(_pypdf_blocks_from_lines(lines, page_number, context))
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
            metadata["authors"] = context.split_authors(str(author))

    warnings = []
    if extract_errors:
        warnings.append(f"page_extract_errors:{len(extract_errors)}")
    if not raw_text.strip():
        warnings.append("scanned_pdf_or_empty_text")

    return context.finish_document(
        path,
        title=title or path.stem,
        raw_text=raw_text,
        blocks=blocks,
        metadata=metadata,
        warnings=warnings,
        parser_name="pdf_pypdf",
        parser_version=context.parser_version,
    )


def parse_docling_pdf(path: Path, context: PdfParseContext) -> ParsedDocument:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise context.parse_error_cls("Docling is not installed. Install with: uv sync --extra docling") from exc

    try:
        result = DocumentConverter().convert(str(path))
    except Exception as exc:  # pragma: no cover - external parser behavior
        raise context.parse_error_cls(f"Docling conversion failed: {exc}") from exc

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
        context=context,
    )


def fetch_grobid_enrichment(path: Path, url: str, parse_error_cls: type[Exception]) -> Dict[str, Any]:
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
        raise parse_error_cls(f"GROBID request failed: {exc}") from exc
    return parse_grobid_tei(tei, parse_error_cls=parse_error_cls)


def merge_grobid_enrichment(doc: ParsedDocument, enrichment: Dict[str, Any]) -> None:
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


def _clean_pdf_page_lines(
    page_lines: List[Tuple[int, List[str]]],
    context: PdfParseContext,
) -> Tuple[List[Tuple[int, List[str]]], int]:
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
        if count >= 2
        and len(line) <= 120
        and not context.classify_heading_line(line)
        and not context.classify_semantic_line(line, False)
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


def _pypdf_blocks_from_lines(lines: List[str], page_number: int, context: PdfParseContext) -> List[ParsedBlock]:
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

        special = context.split_special_line(text)
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
            in_references = context.is_reference_heading(heading)
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

        heading = context.classify_heading_line(text)
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
            in_references = context.is_reference_heading(heading[0])
            continue

        semantic_kind = context.classify_semantic_line(text, in_references)
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


def _parsed_document_from_pdf_payload(
    path: Path,
    payload: Dict[str, Any],
    parser_name: str,
    context: PdfParseContext,
) -> ParsedDocument:
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
    return context.finish_document(
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
        parser_version=context.parser_version,
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
        page=page_from_payload(payload),
        char_start=payload.get("char_start"),
        char_end=payload.get("char_end"),
        bbox=bbox_from_payload(payload),
        layout_block_id=str(payload.get("block_id") or payload.get("layout_block_id") or payload.get("self_ref") or ""),
        caption_id=str(payload.get("caption_id") or payload.get("caption_ref") or ""),
        confidence=confidence_from_payload(payload),
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
                    page=page_from_payload(item),
                    bbox=bbox_from_payload(item),
                    layout_block_id=str(item.get("block_id") or item.get("self_ref") or ""),
                    confidence=confidence_from_payload(item),
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
                    page=page_from_payload(item),
                    bbox=bbox_from_payload(item),
                    layout_block_id=str(item.get("block_id") or item.get("self_ref") or ""),
                    caption_id=str(item.get("caption_id") or ""),
                    confidence=confidence_from_payload(item),
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
                    page=page_from_payload(item),
                    bbox=bbox_from_payload(item),
                    layout_block_id=str(item.get("block_id") or item.get("self_ref") or ""),
                    caption_id=str(item.get("caption_id") or ""),
                    confidence=confidence_from_payload(item),
                    source_parser="docling",
                )
            )
    if not blocks and markdown:
        blocks = [ParsedBlock(kind="paragraph", text=part) for part in split_paragraphs(markdown)]
    return blocks


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
                page = page_from_payload(item)
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


def parse_grobid_tei(tei: str, *, parse_error_cls: type[Exception]) -> Dict[str, Any]:
    try:
        root = ElementTree.fromstring(tei)
    except ElementTree.ParseError as exc:
        raise parse_error_cls(f"Invalid GROBID TEI XML: {exc}") from exc

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
