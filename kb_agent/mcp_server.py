from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .answer import answer_query, route_documents
from .artifacts import get_artifact, get_doc_card
from .config import resolve_db_path
from .ingest import sync_directory
from .memory import put_memory, search_memory
from .search import get_evidence, search_nodes

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - depends on optional package
    FastMCP = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


if FastMCP is not None:
    mcp = FastMCP("paper-kb")

    @mcp.tool()
    def kb_sync(path: str, force: bool = False, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Scan a directory or file and update the local knowledge base index."""
        return sync_directory(Path(path), resolve_db_path(db_path), force=force)

    @mcp.tool()
    def kb_search_docs(query: str, top_k: int = 8, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Route a query to candidate documents using the local full-text index."""
        return route_documents(resolve_db_path(db_path), query, top_k=top_k)

    @mcp.tool()
    def kb_search_tree(
        doc_id: str,
        query: str,
        top_k: int = 8,
        db_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search evidence nodes inside one indexed document."""
        return [
            result.__dict__
            for result in search_nodes(resolve_db_path(db_path), query, doc_id=doc_id, top_k=top_k)
        ]

    @mcp.tool()
    def kb_get_evidence(
        doc_id: str,
        node_ids: List[str],
        db_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return evidence packets for selected document nodes."""
        return [
            packet.to_dict()
            for packet in get_evidence(resolve_db_path(db_path), doc_id, node_ids)
        ]

    @mcp.tool()
    def kb_get_doc_card(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the structured document card for one indexed document."""
        return get_doc_card(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_artifact(
        doc_id: str,
        name: str,
        version_id: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a whitelisted generated artifact for one indexed document."""
        return get_artifact(resolve_db_path(db_path), doc_id, name, version_id=version_id)

    @mcp.tool()
    def kb_answer(
        query: str,
        top_k: int = 6,
        use_llm: bool = True,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a grounded answer draft plus evidence packets."""
        return answer_query(resolve_db_path(db_path), query, top_k=top_k, use_llm=use_llm)

    @mcp.tool()
    def memory_put(
        scope: str,
        type: str,
        subject_key: str,
        content: str,
        importance: float = 0.5,
        confidence: float = 1.0,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store explicit, gated long-term memory."""
        return put_memory(
            resolve_db_path(db_path),
            scope,
            type,
            subject_key,
            content,
            importance=importance,
            confidence=confidence,
        )

    @mcp.tool()
    def memory_get(
        query: str,
        scope: Optional[str] = None,
        top_k: int = 8,
        db_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search relevant long-term memory items."""
        return search_memory(resolve_db_path(db_path), query, scope=scope, top_k=top_k)


def main() -> None:
    if FastMCP is None:
        raise SystemExit(
            "The MCP server requires optional dependency 'mcp'. "
            "Install it with: uv sync --extra mcp\n"
            f"Import error: {_IMPORT_ERROR}"
        )
    mcp.run()  # type: ignore[union-attr]


if __name__ == "__main__":
    main()
