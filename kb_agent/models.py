from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParsedBlock:
    kind: str
    text: str
    heading: str = ""
    level: int = 0
    page: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    bbox: Optional[List[float]] = None
    layout_block_id: str = ""
    caption_id: str = ""
    confidence: float = 1.0
    source_parser: str = ""


@dataclass
class ParsedDocument:
    title: str
    file_type: str
    raw_text: str
    blocks: List[ParsedBlock]
    metadata: Dict[str, Any]
    body_md: str = ""
    structured: Dict[str, Any] = field(default_factory=dict)
    references: Dict[str, Any] = field(default_factory=dict)
    parser_name: str = ""
    parser_version: str = ""
    parse_warnings: List[str] = field(default_factory=list)


@dataclass
class DocumentRecord:
    doc_id: str
    path: str
    hash: str
    title: str
    file_type: str
    size: int
    mtime: float
    summary: str
    status: str = "ready"
    error: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    parser_name: str = ""
    parser_version: str = ""


@dataclass
class NodeRecord:
    node_id: str
    doc_id: str
    parent_id: Optional[str]
    kind: str
    heading: str
    summary: str
    text: str
    level: int
    node_path: str
    page_start: Optional[int]
    page_end: Optional[int]
    order_index: int
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    source_offsets: Dict[str, Any] = field(default_factory=dict)
    doc_hash: str = ""


@dataclass
class EvidencePacket:
    doc_id: str
    node_id: str
    node_path: str
    page_range: Optional[Tuple[Optional[int], Optional[int]]]
    excerpt: str
    evidence_type: str
    confidence: float
    title: str = ""
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    doc_id: str
    node_id: str
    title: str
    path: str
    node_path: str
    heading: str
    snippet: str
    score: float
    page_start: Optional[int]
    page_end: Optional[int]
    fts_score: Optional[float] = None
    vector_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rank_reason: str = ""
