from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass
class ParsedDocument:
    title: str
    file_type: str
    raw_text: str
    blocks: List[ParsedBlock]
    metadata: Dict[str, Any]


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

