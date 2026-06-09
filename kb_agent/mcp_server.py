from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .answer import answer_query, route_documents
from .artifacts import get_artifact, get_citation_map, get_doc_card, get_innovations, get_parse_quality, get_parse_report
from .config import resolve_db_path
from .embeddings import build_semantic_index
from .ingest import sync_directory
from .insights import extract_doc_insights
from .memory import compact_memory, put_memory_gated as write_memory_gated, remember_task, resume_task, search_memory
from .review import assemble_review, check_review_citations, draft_review
from .search import get_evidence, search_nodes
from .tasks import compare_papers, generate_review_plan, get_task_artifact

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
    def kb_sync(
        path: str,
        force: bool = False,
        pdf_parser: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scan a directory or file and update the local knowledge base index."""
        return sync_directory(Path(path), resolve_db_path(db_path), force=force, pdf_parser=pdf_parser)

    @mcp.tool()
    def kb_search_docs(
        query: str,
        top_k: int = 8,
        search_mode: str = "hybrid",
        db_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Route a query to candidate documents using hybrid or FTS search."""
        return route_documents(resolve_db_path(db_path), query, top_k=top_k, search_mode=search_mode)

    @mcp.tool()
    def kb_build_semantic_index(
        doc_ids: Optional[List[str]] = None,
        force: bool = False,
        provider: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build or refresh semantic embeddings for ready documents."""
        return build_semantic_index(resolve_db_path(db_path), doc_ids=doc_ids, force=force, provider=provider)

    @mcp.tool()
    def kb_search_tree(
        doc_id: str,
        query: str,
        top_k: int = 8,
        search_mode: str = "hybrid",
        db_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search evidence nodes inside one indexed document."""
        return [
            result.__dict__
            for result in search_nodes(resolve_db_path(db_path), query, doc_id=doc_id, top_k=top_k, search_mode=search_mode)
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
    def kb_get_parse_quality(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return parse quality metrics and warnings for one indexed document."""
        return get_parse_quality(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_parse_report(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return parser diagnostics and fallback details for one indexed document."""
        return get_parse_report(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_extract_doc_insights(
        doc_id: str,
        force: bool = False,
        use_llm: bool = True,
        require_llm: bool = False,
        search_mode: str = "hybrid",
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract or refresh innovation and citation-map artifacts for one document."""
        return extract_doc_insights(
            resolve_db_path(db_path),
            doc_id,
            force=force,
            use_llm=use_llm,
            require_llm=require_llm,
        )

    @mcp.tool()
    def kb_get_innovations(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the extracted innovation artifact for one document."""
        return get_innovations(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_citation_map(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the extracted citation map artifact for one document."""
        return get_citation_map(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_compare(
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k_docs: int = 5,
        use_llm: bool = True,
        require_llm: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare candidate papers and write grounded comparison task artifacts."""
        return compare_papers(
            resolve_db_path(db_path),
            query,
            doc_ids=doc_ids,
            top_k_docs=top_k_docs,
            use_llm=use_llm,
            require_llm=require_llm,
            search_mode=search_mode,
        )

    @mcp.tool()
    def kb_generate_review(
        topic: str,
        doc_ids: Optional[List[str]] = None,
        top_k_docs: int = 8,
        use_llm: bool = True,
        require_llm: bool = False,
        search_mode: str = "hybrid",
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate review planning artifacts with section-level evidence."""
        return generate_review_plan(
            resolve_db_path(db_path),
            topic,
            doc_ids=doc_ids,
            top_k_docs=top_k_docs,
            use_llm=use_llm,
            require_llm=require_llm,
            search_mode=search_mode,
        )

    @mcp.tool()
    def kb_get_task_artifact(
        task_id: str,
        name: str,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read a whitelisted compare/review task artifact."""
        return get_task_artifact(resolve_db_path(db_path), task_id, name)

    @mcp.tool()
    def kb_draft_review(
        task_id: str,
        section_ids: Optional[List[str]] = None,
        use_llm: bool = True,
        require_llm: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Draft review sections from a generated review task and section evidence."""
        return draft_review(
            resolve_db_path(db_path),
            task_id,
            section_ids=section_ids,
            use_llm=use_llm,
            require_llm=require_llm,
        )

    @mcp.tool()
    def kb_assemble_review(task_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Assemble review section drafts into a Markdown review draft."""
        return assemble_review(resolve_db_path(db_path), task_id)

    @mcp.tool()
    def kb_check_review_citations(task_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Check that review draft citations map to section evidence."""
        return check_review_citations(resolve_db_path(db_path), task_id)

    @mcp.tool()
    def kb_answer(
        query: str,
        top_k: int = 6,
        use_llm: bool = True,
        search_mode: str = "hybrid",
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a grounded answer draft plus evidence packets."""
        return answer_query(resolve_db_path(db_path), query, top_k=top_k, use_llm=use_llm, search_mode=search_mode)

    @mcp.tool()
    def memory_put(
        scope: str,
        type: str,
        subject_key: str,
        content: str,
        importance: float = 0.5,
        confidence: float = 1.0,
        ttl_days: Optional[float] = None,
        refs: str = "",
        force: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store explicit, gated long-term memory."""
        return write_memory_gated(
            resolve_db_path(db_path),
            scope,
            type,
            subject_key,
            content,
            importance=importance,
            confidence=confidence,
            ttl_days=ttl_days,
            refs=refs,
            force=force,
        )

    @mcp.tool()
    def memory_put_gated(
        scope: str,
        type: str,
        subject_key: str,
        content: str,
        refs: str = "",
        ttl_days: Optional[float] = None,
        force: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store long-term memory after applying the write gate."""
        return write_memory_gated(
            resolve_db_path(db_path),
            scope,
            type,
            subject_key,
            content,
            refs=refs,
            ttl_days=ttl_days,
            force=force,
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

    @mcp.tool()
    def memory_remember_task(task_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Store a compressed memory entry for a task artifact directory."""
        return remember_task(resolve_db_path(db_path), task_id)

    @mcp.tool()
    def memory_resume_task(db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return current task status, remembered tasks, and suggested next commands."""
        return resume_task(resolve_db_path(db_path))

    @mcp.tool()
    def memory_compact(scope: Optional[str] = None, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Compact task progress memory entries into a short summary."""
        return compact_memory(resolve_db_path(db_path), scope=scope)


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
