from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .answer import answer_query, route_documents
from .artifacts import (
    get_artifact,
    get_citation_map,
    get_doc_card,
    get_figures,
    get_innovations,
    get_layout_blocks,
    get_parse_quality,
    get_parse_report,
    get_table_content,
    get_table_summaries,
    get_tables,
)
from .benchmark import (
    analyze_failures,
    create_eval_suite,
    generate_case_study,
    get_eval_suite,
    list_eval_suites,
    run_benchmark,
)
from .config import resolve_db_path
from .embeddings import build_semantic_index, semantic_index_status
from .eval import eval_facts, eval_memory, eval_review, eval_search
from .fact_audit import audit_facts, get_fact_conflicts
from .facts import extract_facts, fact_search, get_claims, get_entities, get_fact_graph, get_relations
from .feedback import build_eval_set_from_feedback, eval_dashboard, list_feedback, put_feedback
from .ingest import sync_directory
from .insights import extract_doc_insights
from .llm import llm_status
from .knowledge_graph import (
    build_knowledge_graph,
    export_knowledge_graph,
    get_graph_neighborhood,
    get_graph_report,
    get_knowledge_graph,
)
from .memory import compact_memory, put_memory_gated as write_memory_gated, remember_task, resume_task, search_memory
from .quality_baseline import latest_quality_baseline, run_quality_baseline
from .query import classify_query
from .query_log import list_query_logs, query_stats
from .review import assemble_review, check_review_citations, draft_review
from .search import get_evidence, search_nodes
from .search_profile import apply_search_profile, get_search_profile, list_search_profiles, tune_search
from .tasks import compare_papers, generate_review_plan, get_task_artifact
from .tree_search import tree_search

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
        model: Optional[str] = None,
        batch_size: int = 16,
        status: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build or refresh semantic embeddings for ready documents."""
        if status:
            return semantic_index_status(resolve_db_path(db_path), provider=provider, model=model)
        return build_semantic_index(
            resolve_db_path(db_path),
            doc_ids=doc_ids,
            force=force,
            provider=provider,
            model=model,
            batch_size=batch_size,
        )

    @mcp.tool()
    def kb_get_llm_status(probe: bool = False) -> Dict[str, Any]:
        """Return sanitized DeepSeek configuration and optional connectivity state."""
        return llm_status(probe=probe)

    @mcp.tool()
    def kb_eval_search(
        queries_json: str,
        search_mode: str = "hybrid",
        top_k: int = 5,
        compare_modes: Optional[List[str]] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run retrieval evaluation and optionally compare search modes."""
        return eval_search(
            resolve_db_path(db_path),
            Path(queries_json),
            search_mode=search_mode,
            top_k=top_k,
            compare_modes=compare_modes,
        )

    @mcp.tool()
    def kb_eval_review(task_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate review task citation coverage and unsupported paragraphs."""
        return eval_review(resolve_db_path(db_path), task_id)

    @mcp.tool()
    def kb_eval_memory(db_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate long-term memory hygiene and task resume readiness."""
        return eval_memory(resolve_db_path(db_path))

    @mcp.tool()
    def kb_eval_facts(doc_ids: Optional[List[str]] = None, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate grounded fact coverage, confidence, duplicates, and table-backed facts."""
        return eval_facts(resolve_db_path(db_path), doc_ids=doc_ids)

    @mcp.tool()
    def kb_audit_facts(
        doc_ids: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Audit grounded facts for duplicates, conflicts, citation gaps, and evidence gaps."""
        return audit_facts(resolve_db_path(db_path), doc_ids=doc_ids, min_confidence=min_confidence)

    @mcp.tool()
    def kb_get_fact_conflicts(
        doc_ids: Optional[List[str]] = None,
        severity: Optional[str] = None,
        min_confidence: float = 0.5,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return fact conflicts from a fresh audit without long evidence text."""
        return get_fact_conflicts(
            resolve_db_path(db_path),
            doc_ids=doc_ids,
            severity=severity,
            min_confidence=min_confidence,
        )

    @mcp.tool()
    def kb_build_knowledge_graph(
        doc_ids: Optional[List[str]] = None,
        include_conflicts: bool = False,
        min_confidence: float = 0.0,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a lightweight claim graph from facts, evidence ids, and optional audit conflicts."""
        return build_knowledge_graph(
            resolve_db_path(db_path),
            doc_ids=doc_ids,
            include_conflicts=include_conflicts,
            min_confidence=min_confidence,
        )

    @mcp.tool()
    def kb_get_knowledge_graph(graph_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return a previously built lightweight claim graph."""
        return get_knowledge_graph(resolve_db_path(db_path), graph_id)

    @mcp.tool()
    def kb_get_graph_neighborhood(
        node_or_fact_id: str,
        depth: int = 1,
        graph_id: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a bounded graph neighborhood for a claim/entity/relation/conflict/evidence id."""
        return get_graph_neighborhood(resolve_db_path(db_path), node_or_fact_id, depth=depth, graph_id=graph_id)

    @mcp.tool()
    def kb_export_knowledge_graph(
        graph_id: str,
        format: str = "json",
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export a claim graph as json, mermaid, or static html without evidence text."""
        return export_knowledge_graph(resolve_db_path(db_path), graph_id, format=format)

    @mcp.tool()
    def kb_get_graph_report(graph_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return a graph quality report with evidence coverage, conflicts, and isolated facts."""
        return get_graph_report(resolve_db_path(db_path), graph_id)

    @mcp.tool()
    def kb_run_quality_baseline(
        corpus_path: str = "articles",
        force: bool = True,
        top_k: int = 5,
        use_llm: bool = False,
        embedding_model: Optional[str] = None,
        llm_timeout_seconds: Optional[int] = None,
        llm_stage_timeout_seconds: Optional[int] = None,
        llm_max_docs: Optional[int] = None,
        skip_llm_tasks: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a real-corpus quality baseline across parsing, embeddings, retrieval, tasks, memory, and graph risks."""
        return run_quality_baseline(
            resolve_db_path(db_path),
            Path(corpus_path),
            force=force,
            top_k=top_k,
            use_llm=use_llm,
            embedding_model=embedding_model,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_stage_timeout_seconds=llm_stage_timeout_seconds,
            llm_max_docs=llm_max_docs,
            skip_llm_tasks=skip_llm_tasks,
        )

    @mcp.tool()
    def kb_get_latest_quality_baseline(
        limit: int = 1,
        corpus: Optional[str] = None,
        real_only: bool = False,
        exclude_temp: bool = False,
    ) -> Dict[str, Any]:
        """Return latest quality baseline summaries without evidence text."""
        return latest_quality_baseline(limit=limit, corpus=corpus, real_only=real_only, exclude_temp=exclude_temp)

    @mcp.tool()
    def kb_create_eval_suite(
        name: str,
        input_json: Optional[str] = None,
        from_feedback: bool = False,
        from_query_log: bool = False,
        doc_ids: Optional[List[str]] = None,
        limit: int = 100,
        min_rating: int = 4,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a reusable evaluation suite without storing evidence text."""
        return create_eval_suite(
            resolve_db_path(db_path),
            name,
            input_json=Path(input_json) if input_json else None,
            from_feedback=from_feedback,
            from_query_log=from_query_log,
            doc_ids=doc_ids,
            limit=limit,
            min_rating=min_rating,
        )

    @mcp.tool()
    def kb_list_eval_suites() -> Dict[str, Any]:
        """List saved local evaluation suites."""
        return list_eval_suites()

    @mcp.tool()
    def kb_get_eval_suite(name: str) -> Dict[str, Any]:
        """Return one saved local evaluation suite."""
        return get_eval_suite(name)

    @mcp.tool()
    def kb_run_benchmark(
        suite_name: str,
        compare_modes: Optional[List[str]] = None,
        top_k: int = 5,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a retrieval benchmark across fts/hybrid/tree/auto modes."""
        return run_benchmark(
            resolve_db_path(db_path),
            suite_name,
            compare_modes=compare_modes,
            top_k=top_k,
        )

    @mcp.tool()
    def kb_analyze_failures(benchmark_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze benchmark misses, fallbacks, weak parse warnings, and next actions."""
        return analyze_failures(resolve_db_path(db_path), benchmark_id)

    @mcp.tool()
    def kb_generate_case_study(
        query: str,
        doc_ids: Optional[List[str]] = None,
        compare_modes: Optional[List[str]] = None,
        top_k: int = 5,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a retrieval case study with mode summaries and no evidence body text."""
        return generate_case_study(
            resolve_db_path(db_path),
            query,
            doc_ids=doc_ids,
            compare_modes=compare_modes,
            top_k=top_k,
        )

    @mcp.tool()
    def kb_get_query_log(
        limit: int = 20,
        operation: Optional[str] = None,
        intent: Optional[str] = None,
        status: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return recent sanitized query logs without evidence excerpts."""
        return list_query_logs(
            resolve_db_path(db_path),
            limit=limit,
            operation=operation,
            intent=intent,
            status=status,
        )

    @mcp.tool()
    def kb_get_query_stats(since_days: Optional[float] = None, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return aggregate query log metrics and warning rates."""
        return query_stats(resolve_db_path(db_path), since_days=since_days)

    @mcp.tool()
    def kb_put_feedback(
        query: str,
        rating: int,
        query_id: str = "",
        operation: str = "",
        label: str = "",
        comment: str = "",
        expected_doc_ids: Optional[List[str]] = None,
        expected_node_ids: Optional[List[str]] = None,
        expected_keywords: Optional[List[str]] = None,
        preferred_search_mode: str = "",
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record sanitized human feedback for a query result without evidence text."""
        return put_feedback(
            resolve_db_path(db_path),
            query=query,
            query_id=query_id,
            operation=operation,
            rating=rating,
            label=label,
            comment=comment,
            expected_doc_ids=expected_doc_ids,
            expected_node_ids=expected_node_ids,
            expected_keywords=expected_keywords,
            preferred_search_mode=preferred_search_mode,
        )

    @mcp.tool()
    def kb_get_feedback(
        limit: int = 20,
        operation: Optional[str] = None,
        label: Optional[str] = None,
        rating: Optional[int] = None,
        min_rating: Optional[int] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return sanitized human feedback items."""
        return list_feedback(
            resolve_db_path(db_path),
            limit=limit,
            operation=operation,
            label=label,
            rating=rating,
            min_rating=min_rating,
        )

    @mcp.tool()
    def kb_build_eval_set_from_feedback(
        output_json: Optional[str] = None,
        min_rating: int = 4,
        label: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 200,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a search evaluation set from high-quality feedback."""
        return build_eval_set_from_feedback(
            resolve_db_path(db_path),
            output_path=Path(output_json) if output_json else None,
            min_rating=min_rating,
            label=label,
            operation=operation,
            limit=limit,
        )

    @mcp.tool()
    def kb_eval_dashboard(
        since_days: Optional[float] = None,
        format: str = "html",
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write a static dashboard that summarizes query logs, eval reports, and feedback."""
        return eval_dashboard(resolve_db_path(db_path), since_days=since_days, output_format=format)

    @mcp.tool()
    def kb_tune_search(
        queries_json: str,
        compare_modes: Optional[List[str]] = None,
        top_k: int = 5,
        save_profile: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tune search mode preferences from a search evaluation set."""
        return tune_search(
            resolve_db_path(db_path),
            Path(queries_json),
            compare_modes=compare_modes,
            top_k=top_k,
            save_profile=save_profile,
        )

    @mcp.tool()
    def kb_get_search_profile(name: str = "active") -> Dict[str, Any]:
        """Return a saved or active local search profile."""
        if name == "list":
            return list_search_profiles()
        return get_search_profile(name)

    @mcp.tool()
    def kb_apply_search_profile(name: str) -> Dict[str, Any]:
        """Apply a saved search profile for explicit auto search mode."""
        return apply_search_profile(name)

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
    def kb_classify_query(
        query: str,
        use_llm: bool = True,
        require_llm: bool = False,
    ) -> Dict[str, Any]:
        """Classify query intent and preferred tree-search targets."""
        return classify_query(query, use_llm=use_llm, require_llm=require_llm)

    @mcp.tool()
    def kb_tree_search(
        doc_id: str,
        query: str,
        budget: int = 8,
        use_llm: bool = True,
        require_llm: bool = False,
        search_mode: str = "hybrid",
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run explainable value-function or LLM-guided tree search inside one document."""
        return tree_search(
            resolve_db_path(db_path),
            doc_id,
            query,
            budget=budget,
            use_llm=use_llm,
            require_llm=require_llm,
            search_mode=search_mode,
        )

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
    def kb_get_layout_blocks(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return normalized layout blocks for one indexed document."""
        return get_layout_blocks(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_figures(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return parsed figure captions and layout links for one indexed document."""
        return get_figures(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_tables(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return parsed table captions and layout links for one indexed document."""
        return get_tables(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_table_content(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return parsed table rows, cells, and conservative quality warnings."""
        return get_table_content(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_table_summaries(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return compact table summaries for metrics, methods, and results."""
        return get_table_summaries(resolve_db_path(db_path), doc_id)

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
    def kb_extract_facts(
        doc_id: str,
        force: bool = False,
        use_llm: bool = True,
        require_llm: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract grounded claims, entities, relations, and fact graph artifacts for one document."""
        return extract_facts(
            resolve_db_path(db_path),
            doc_id,
            force=force,
            use_llm=use_llm,
            require_llm=require_llm,
        )

    @mcp.tool()
    def kb_get_claims(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the extracted claims artifact for one document."""
        return get_claims(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_entities(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the extracted entities artifact for one document."""
        return get_entities(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_relations(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the extracted relations artifact for one document."""
        return get_relations(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_get_fact_graph(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the extracted fact graph artifact for one document."""
        return get_fact_graph(resolve_db_path(db_path), doc_id)

    @mcp.tool()
    def kb_fact_search(
        query: str,
        doc_ids: Optional[List[str]] = None,
        type: Optional[str] = None,
        source: str = "all",
        min_confidence: float = 0.0,
        top_k: int = 20,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search grounded claims, entities, and relations without returning long excerpts."""
        return fact_search(
            resolve_db_path(db_path),
            query,
            doc_ids=doc_ids,
            fact_type=type,
            source=source,
            min_confidence=min_confidence,
            top_k=top_k,
        )

    @mcp.tool()
    def kb_compare(
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k_docs: int = 5,
        use_llm: bool = True,
        require_llm: bool = False,
        search_mode: str = "hybrid",
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
