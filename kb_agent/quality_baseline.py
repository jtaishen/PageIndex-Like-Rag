from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .artifacts import get_doc_card, get_parse_quality, get_parse_report
from .benchmark import create_eval_suite, generate_case_study, run_benchmark
from .config import DATA_DIR, PROJECT_ROOT
from .embeddings import EmbeddingError, build_semantic_index, semantic_index_status
from .eval import eval_memory
from .facts import extract_facts
from .ingest import discover_files, sync_directory
from .insights import extract_doc_insights
from .knowledge_graph import graph_summary
from .parsers import pdf_adapter_statuses
from .tasks import compare_papers, generate_review_plan
from .tree_search import tree_search
from .utils import compact_whitespace, stable_id, write_json


BASELINE_SCHEMA = "quality_baseline.v1"
BASELINE_DIR = DATA_DIR / "eval"
EVAL_SET_DIR = DATA_DIR / "eval_sets"


def run_quality_baseline(
    db_path: Path,
    corpus_path: Path = Path("articles"),
    *,
    force: bool = True,
    top_k: int = 5,
    use_llm: bool = False,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.time()
    root = corpus_path.expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    files = discover_files(root)
    pdf_files = [path for path in files if path.suffix.lower() == ".pdf"]
    warnings: List[str] = []
    if not files:
        warnings.append("empty_corpus")

    primary_sync = sync_directory(root, db_path, force=force, pdf_parser="pypdf")
    doc_ids = _doc_ids_for_corpus(db_path, root)
    doc_reports = [_doc_quality_summary(db_path, doc_id) for doc_id in doc_ids]
    parser_comparison = _parser_comparison(root, pdf_files, primary_sync)
    insights = _prepare_insights_and_facts(db_path, doc_ids, use_llm=use_llm)
    embedding = _embedding_baseline(db_path, doc_ids, embedding_model=embedding_model)
    eval_set = _write_baseline_eval_set(doc_reports)
    suite = create_eval_suite(db_path, f"quality_baseline_{int(started)}", input_json=Path(eval_set["path"]))
    benchmark = run_benchmark(db_path, str(suite["name"]), compare_modes=["fts", "hybrid", "tree"], top_k=top_k)
    tree = _tree_search_baseline(db_path, doc_reports, top_k=top_k)
    tasks = _task_baseline(db_path, doc_ids, use_llm=use_llm)
    memory = eval_memory(db_path)
    graph = graph_summary(db_path, doc_ids=doc_ids, include_conflicts=True) if doc_ids else {
        "schema": "knowledge_graph_summary.v1",
        "available": False,
        "warnings": ["no_ready_documents_for_graph"],
    }
    recommendations = _recommendations(doc_reports, parser_comparison, embedding, benchmark, tree, tasks, memory, graph)
    warnings.extend(_baseline_warnings(doc_reports, parser_comparison, embedding, benchmark, tasks, memory, graph))
    baseline_id = stable_id("quality_baseline", str(root), ",".join(doc_ids), started, length=12)
    report = {
        "schema": BASELINE_SCHEMA,
        "baseline_id": baseline_id,
        "corpus_path": str(root),
        "file_count": len(files),
        "pdf_count": len(pdf_files),
        "doc_ids": doc_ids,
        "doc_count": len(doc_ids),
        "primary_parser": "pypdf",
        "primary_sync": _sync_summary(primary_sync),
        "parser_comparison": parser_comparison,
        "documents": doc_reports,
        "insights_and_facts": insights,
        "embedding": embedding,
        "eval_set": eval_set,
        "eval_suite": _suite_summary(suite),
        "benchmark": _benchmark_summary(benchmark),
        "tree_search": tree,
        "tasks": tasks,
        "memory": _memory_summary(memory),
        "claim_graph": graph,
        "recommendations": recommendations,
        "warnings": _unique_strings(warnings),
        "created_at": started,
        "duration_ms": round((time.time() - started) * 1000, 3),
    }
    json_path = BASELINE_DIR / f"quality_baseline_{baseline_id}.json"
    md_path = BASELINE_DIR / f"quality_baseline_{baseline_id}.md"
    html_path = BASELINE_DIR / f"quality_baseline_{baseline_id}.html"
    write_json(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_baseline_markdown(report), encoding="utf-8")
    html_path.write_text(_baseline_html(report), encoding="utf-8")
    return {
        **report,
        "path": str(json_path),
        "json_path": str(json_path),
        "md_path": str(md_path),
        "html_path": str(html_path),
    }


def latest_quality_baseline(limit: int = 1) -> Dict[str, Any]:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(BASELINE_DIR.glob("quality_baseline_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = _read_json(path, {})
        if payload.get("schema") != BASELINE_SCHEMA:
            continue
        items.append(
            {
                "path": str(path),
                "baseline_id": payload.get("baseline_id") or "",
                "corpus_path": payload.get("corpus_path") or "",
                "doc_count": payload.get("doc_count", 0),
                "pdf_count": payload.get("pdf_count", 0),
                "best_search_mode": (payload.get("benchmark") or {}).get("best_mode_by_score") or "",
                "weak_doc_count": sum(1 for item in payload.get("documents") or [] if item.get("quality_level") == "weak"),
                "real_embedding_status": (payload.get("embedding") or {}).get("sentence_transformers", {}).get("status", ""),
                "warning_count": len(payload.get("warnings") or []),
                "warnings": payload.get("warnings") or [],
                "created_at": payload.get("created_at"),
            }
        )
        if len(items) >= limit:
            break
    return {"schema": "quality_baseline_latest.v1", "count": len(items), "items": items}


def _doc_ids_for_corpus(db_path: Path, root: Path) -> List[str]:
    conn = db.connect(db_path)
    db.init_db(conn)
    root_text = str(root)
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT doc_id, path
                FROM documents
                WHERE status = 'ready'
                ORDER BY path ASC
                """
            ).fetchall()
        ]
    finally:
        conn.close()
    return [
        str(row["doc_id"])
        for row in rows
        if str(row.get("path") or "").startswith(root_text)
    ]


def _doc_quality_summary(db_path: Path, doc_id: str) -> Dict[str, Any]:
    card = get_doc_card(db_path, doc_id)
    quality = get_parse_quality(db_path, doc_id)
    try:
        parse_report = get_parse_report(db_path, doc_id)
    except (FileNotFoundError, KeyError, ValueError):
        parse_report = {}
    return {
        "doc_id": doc_id,
        "title": card.get("title") or doc_id,
        "path": card.get("path") or "",
        "parser_name": card.get("parser_name") or "",
        "parser_version": card.get("parser_version") or "",
        "quality_level": quality.get("quality_level") or "",
        "section_count": quality.get("section_count", 0),
        "paragraph_count": quality.get("paragraph_count", 0),
        "reference_count": quality.get("reference_count", 0),
        "figure_count": quality.get("figure_count", 0),
        "table_count": quality.get("table_count", 0),
        "table_content_count": quality.get("table_content_count", 0),
        "layout_block_count": quality.get("layout_block_count", 0),
        "metadata_score": quality.get("metadata_score"),
        "structure_score": quality.get("structure_score"),
        "reference_score": quality.get("reference_score"),
        "layout_score": quality.get("layout_score"),
        "table_parse_score": quality.get("table_parse_score"),
        "missing_abstract": bool(quality.get("missing_abstract")),
        "page_only_tree": bool(quality.get("page_only_tree")),
        "fallback_used": bool(quality.get("fallback_used") or parse_report.get("fallback_used")),
        "parser_chain": quality.get("parser_chain") or parse_report.get("parser_chain") or [],
        "warning_count": len(quality.get("quality_warnings") or []),
        "warnings": quality.get("quality_warnings") or [],
        "parse_report_path": parse_report.get("path", ""),
        "artifact_dir": parse_report.get("artifact_dir", ""),
    }


def _parser_comparison(root: Path, pdf_files: List[Path], primary_sync: Dict[str, Any]) -> Dict[str, Any]:
    statuses = pdf_adapter_statuses()
    providers = []
    for provider in ("pypdf", "docling", "grobid"):
        providers.append(_parser_provider_result(root, pdf_files, provider, statuses, primary_sync))
    return {
        "schema": "parser_comparison.v1",
        "pdf_count": len(pdf_files),
        "adapter_statuses": statuses,
        "providers": providers,
    }


def _parser_provider_result(
    root: Path,
    pdf_files: List[Path],
    provider: str,
    statuses: Dict[str, Any],
    primary_sync: Dict[str, Any],
) -> Dict[str, Any]:
    if not pdf_files:
        return {"provider": provider, "status": "skipped", "reason": "no_pdf_files", "documents": []}
    if provider == "docling" and not (statuses.get("docling") or {}).get("available"):
        return {"provider": provider, "status": "skipped", "reason": "docling_not_installed", "documents": []}
    if provider == "grobid" and not os.environ.get("GROBID_URL", "").strip():
        return {"provider": provider, "status": "skipped", "reason": "GROBID_URL_not_configured", "documents": []}
    if provider == "pypdf":
        return {
            "provider": provider,
            "status": "completed" if int(primary_sync.get("failed") or 0) == 0 else "partial",
            "sync": _sync_summary(primary_sync),
            "documents": [],
        }
    with tempfile.TemporaryDirectory(prefix=f"kb_baseline_{provider}_") as tmp:
        temp_db = Path(tmp) / "kb.sqlite"
        report = sync_directory(root, temp_db, force=True, pdf_parser=provider)
        doc_ids = _doc_ids_for_corpus(temp_db, root)
        documents = [_doc_quality_summary(temp_db, doc_id) for doc_id in doc_ids]
    status = "completed" if int(report.get("failed") or 0) == 0 else "partial"
    return {
        "provider": provider,
        "status": status,
        "sync": _sync_summary(report),
        "documents": documents,
    }


def _prepare_insights_and_facts(db_path: Path, doc_ids: List[str], *, use_llm: bool) -> Dict[str, Any]:
    items = []
    warnings: List[str] = []
    for doc_id in doc_ids:
        insight_status = "skipped"
        fact_status = "skipped"
        try:
            insight = extract_doc_insights(db_path, doc_id, force=True, use_llm=use_llm, require_llm=False)
            innovation = insight.get("innovation") or {}
            citation_map = insight.get("citation_map") or {}
            insight_status = str(innovation.get("status") or "partial")
            fact = extract_facts(db_path, doc_id, force=True, use_llm=use_llm, require_llm=False)
            fact_report = fact.get("fact_report") or {}
            fact_status = str(fact_report.get("status") or "partial")
            items.append(
                {
                    "doc_id": doc_id,
                    "innovation_status": insight_status,
                    "innovation_count": len(innovation.get("items") or []),
                    "citation_reference_count": len(citation_map.get("references") or []),
                    "citation_relation_count": len(citation_map.get("relations") or []),
                    "fact_status": fact_status,
                    "claim_count": fact_report.get("claim_count", 0),
                    "entity_count": fact_report.get("entity_count", 0),
                    "relation_count": fact_report.get("relation_count", 0),
                    "warnings": _unique_strings([*innovation.get("warnings", []), *fact_report.get("warnings", [])]),
                }
            )
        except Exception as exc:
            warnings.append(f"insight_fact_failed:{doc_id}:{exc}")
            items.append({"doc_id": doc_id, "innovation_status": insight_status, "fact_status": fact_status, "error": str(exc)})
    return {
        "schema": "baseline_insights_facts.v1",
        "use_llm": use_llm,
        "doc_count": len(items),
        "items": items,
        "warnings": _unique_strings(warnings),
    }


def _embedding_baseline(db_path: Path, doc_ids: List[str], *, embedding_model: Optional[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "embedding_baseline.v1",
        "doc_ids": doc_ids,
        "hash": {},
        "sentence_transformers": {},
        "warnings": [],
    }
    if not doc_ids:
        result["hash"] = {"status": "skipped", "reason": "no_ready_documents"}
        result["sentence_transformers"] = {"status": "skipped", "reason": "no_ready_documents"}
        result["warnings"].append("embedding_skipped:no_ready_documents")
        return result
    try:
        build = build_semantic_index(db_path, doc_ids=doc_ids or None, force=True, provider="hash")
        result["hash"] = {"status": "completed", "build": build, "status_report": semantic_index_status(db_path, provider="hash")}
    except Exception as exc:
        result["hash"] = {"status": "failed", "error": str(exc)}
        result["warnings"].append("hash_embedding_failed")
    if importlib.util.find_spec("sentence_transformers") is None:
        result["sentence_transformers"] = {
            "status": "skipped",
            "reason": "sentence_transformers_not_installed",
        }
        result["warnings"].append("real_embedding_not_enabled")
        return result
    try:
        build = build_semantic_index(
            db_path,
            doc_ids=doc_ids or None,
            force=True,
            provider="sentence-transformers",
            model=embedding_model,
        )
        status = semantic_index_status(db_path, provider="sentence-transformers", model=build.get("model"))
        result["sentence_transformers"] = {"status": "completed", "build": build, "status_report": status}
    except EmbeddingError as exc:
        result["sentence_transformers"] = {"status": "skipped", "reason": str(exc)}
        result["warnings"].append("real_embedding_skipped")
    except Exception as exc:
        result["sentence_transformers"] = {"status": "failed", "error": str(exc)}
        result["warnings"].append("real_embedding_failed")
    return result


def _write_baseline_eval_set(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    EVAL_SET_DIR.mkdir(parents=True, exist_ok=True)
    queries = []
    for doc in documents:
        doc_id = str(doc.get("doc_id") or "")
        title = str(doc.get("title") or doc_id)
        queries.extend(
            [
                _eval_query(f"{title} 的摘要和主要研究内容是什么？", "summary", doc_id, ["摘要"]),
                _eval_query(f"{title} 的方法设计是什么？", "method", doc_id, ["方法", "框架", "算法"]),
                _eval_query(f"{title} 的实验结果和评价指标是什么？", "experiment", doc_id, ["实验", "结果", "指标"]),
                _eval_query(f"{title} 的局限和未来工作是什么？", "limitation", doc_id, ["局限", "未来"]),
                _eval_query(f"{title} 的引用关系和参考文献情况是什么？", "citation", doc_id, ["参考文献", "引用"]),
            ]
        )
    if len(documents) >= 2:
        queries.append(
            {
                "query": "这些论文的方法和实验评测有什么区别？",
                "intent": "compare",
                "category": "baseline_compare",
                "expected_doc_ids": [str(item.get("doc_id") or "") for item in documents],
                "expected_node_ids": [],
                "expected_keywords": ["方法", "实验"],
                "expected_fact_sources": [],
            }
        )
    created_at = time.time()
    payload = {
        "schema": "search_eval_set.v1",
        "source": "quality_baseline",
        "query_count": len(queries),
        "queries": queries,
        "created_at": created_at,
    }
    path = EVAL_SET_DIR / f"quality_baseline_eval_{int(created_at)}.json"
    write_json(path, payload)
    return {**payload, "path": str(path)}


def _eval_query(query: str, intent: str, doc_id: str, keywords: List[str]) -> Dict[str, Any]:
    return {
        "query": query,
        "intent": intent,
        "category": "baseline_doc",
        "expected_doc_ids": [doc_id],
        "expected_node_ids": [],
        "expected_keywords": keywords,
        "expected_fact_sources": [],
    }


def _tree_search_baseline(db_path: Path, documents: List[Dict[str, Any]], *, top_k: int) -> Dict[str, Any]:
    items = []
    warnings = []
    for doc in documents:
        doc_id = str(doc.get("doc_id") or "")
        title = str(doc.get("title") or doc_id)
        query = f"{title} 的方法设计和实验结果是什么？"
        try:
            trace = tree_search(db_path, doc_id, query, budget=top_k, use_llm=False, search_mode="hybrid")
        except Exception as exc:
            warnings.append(f"tree_search_failed:{doc_id}:{exc}")
            items.append({"doc_id": doc_id, "status": "failed", "error": str(exc)})
            continue
        items.append(
            {
                "doc_id": doc_id,
                "query": query,
                "status": "completed",
                "intent": (trace.get("query_profile") or {}).get("intent", ""),
                "expanded_node_count": len(trace.get("expanded_nodes") or []),
                "selected_path_count": len(trace.get("selected_paths") or []),
                "evidence_count": len(trace.get("evidence") or []),
                "fallback_reason": trace.get("fallback_reason") or "",
                "warnings": trace.get("warnings") or [],
            }
        )
    return {
        "schema": "tree_search_baseline.v1",
        "doc_count": len(items),
        "items": items,
        "warnings": _unique_strings(warnings),
    }


def _task_baseline(db_path: Path, doc_ids: List[str], *, use_llm: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "task_quality_baseline.v1",
        "use_llm": use_llm,
        "compare": {},
        "review": {},
        "case_study": {},
        "warnings": [],
    }
    if len(doc_ids) < 2:
        result["warnings"].append("insufficient_docs_for_compare_review")
        return result
    try:
        compare = compare_papers(
            db_path,
            "真实论文集方法与实验评测对比",
            doc_ids=doc_ids,
            use_llm=use_llm,
            require_llm=False,
            search_mode="tree",
        )
        matrix = compare.get("comparison_matrix") or {}
        result["compare"] = {
            "task_id": compare.get("task_id"),
            "status": compare.get("status"),
            "artifact_paths": compare.get("artifact_paths") or {},
            "evidence_coverage": matrix.get("evidence_coverage") or {},
            "fact_audit": matrix.get("fact_audit") or {},
            "claim_graph": matrix.get("claim_graph") or {},
            "warning_count": len(matrix.get("warnings") or []),
            "warnings": matrix.get("warnings") or [],
            "llm_error": compare.get("llm_error", ""),
        }
    except Exception as exc:
        result["compare"] = {"status": "failed", "error": str(exc)}
        result["warnings"].append(f"compare_failed:{exc}")
    try:
        review = generate_review_plan(
            db_path,
            "真实论文集任务规划方法研究综述",
            doc_ids=doc_ids,
            use_llm=use_llm,
            require_llm=False,
            search_mode="tree",
        )
        outline = review.get("review_outline") or {}
        result["review"] = {
            "task_id": review.get("task_id"),
            "status": review.get("status"),
            "artifact_paths": review.get("artifact_paths") or {},
            "evidence_coverage": outline.get("evidence_coverage") or {},
            "fact_audit": outline.get("fact_audit") or {},
            "claim_graph": outline.get("claim_graph") or {},
            "open_questions": outline.get("open_questions") or [],
            "warning_count": len(outline.get("warnings") or []),
            "warnings": outline.get("warnings") or [],
            "llm_error": review.get("llm_error", ""),
        }
    except Exception as exc:
        result["review"] = {"status": "failed", "error": str(exc)}
        result["warnings"].append(f"review_failed:{exc}")
    try:
        case = generate_case_study(db_path, "真实论文集方法与实验评测对比", doc_ids=doc_ids, compare_modes=["hybrid", "tree"], top_k=5)
        result["case_study"] = {
            "case_id": case.get("case_id"),
            "path": case.get("path"),
            "md_path": case.get("md_path"),
            "evidence_count": (case.get("evidence_summary") or {}).get("count", 0),
            "fact_match_count": (case.get("fact_matches") or {}).get("count", 0),
            "fact_conflict_count": (case.get("fact_conflicts") or {}).get("conflict_count", 0),
            "claim_graph": case.get("claim_graph") or {},
            "warnings": case.get("warnings") or [],
        }
    except Exception as exc:
        result["case_study"] = {"status": "failed", "error": str(exc)}
        result["warnings"].append(f"case_study_failed:{exc}")
    return result


def _recommendations(
    documents: List[Dict[str, Any]],
    parser_comparison: Dict[str, Any],
    embedding: Dict[str, Any],
    benchmark: Dict[str, Any],
    tree: Dict[str, Any],
    tasks: Dict[str, Any],
    memory: Dict[str, Any],
    graph: Dict[str, Any],
) -> List[str]:
    items = []
    if any(doc.get("quality_level") == "weak" for doc in documents):
        items.append("存在弱解析论文；优先使用 Docling 重建 PDF 工件，并复查章节树和表格内容。")
    if any(provider.get("status") == "skipped" for provider in parser_comparison.get("providers") or [] if provider.get("provider") in {"docling", "grobid"}):
        items.append("Docling 或 GROBID 未完成真实对比；补齐依赖/服务后重跑 quality-baseline。")
    if (embedding.get("sentence_transformers") or {}).get("status") != "completed":
        items.append("真实 embedding 尚未完成；安装 sentence-transformers 并指定适合中文论文的模型后重跑。")
    if (benchmark.get("summary") or {}).get("tree", {}).get("benchmark_score", 0) < (benchmark.get("summary") or {}).get("fts", {}).get("benchmark_score", 0):
        items.append("tree 检索未优于 fts；需要复核 query intent、章节树质量和 value function 权重。")
    if any(not item.get("evidence_count") for item in tree.get("items") or []):
        items.append("部分 tree-search 无证据；优先检查对应文档的 node_index 和解析质量。")
    if (tasks.get("compare") or {}).get("warning_count", 0) or (tasks.get("review") or {}).get("warning_count", 0):
        items.append("比较/综述任务存在 warning；查看任务工件中的 open_questions 和 evidence_coverage。")
    if memory.get("status") == "needs_review":
        items.append("memory 评测需要复核；确认没有论文资产进入长期记忆。")
    if graph.get("conflict_count", 0) or graph.get("isolated_fact_count", 0):
        items.append("Claim Graph 存在冲突或孤立事实；用 graph-neighborhood 回到 evidence packet 核验。")
    if not items:
        items.append("当前基线未发现阻塞项；下一步可扩展到 10-30 篇真实论文并重跑 benchmark。")
    return _unique_strings(items)


def _baseline_warnings(
    documents: List[Dict[str, Any]],
    parser_comparison: Dict[str, Any],
    embedding: Dict[str, Any],
    benchmark: Dict[str, Any],
    tasks: Dict[str, Any],
    memory: Dict[str, Any],
    graph: Dict[str, Any],
) -> List[str]:
    warnings = []
    if not documents:
        warnings.append("no_ready_documents")
    if any(doc.get("quality_level") == "weak" for doc in documents):
        warnings.append("weak_parse_quality")
    if any(provider.get("status") == "skipped" for provider in parser_comparison.get("providers") or []):
        warnings.append("optional_parser_skipped")
    warnings.extend(embedding.get("warnings") or [])
    warnings.extend(benchmark.get("warnings") or [])
    warnings.extend((tasks.get("compare") or {}).get("warnings") or [])
    warnings.extend((tasks.get("review") or {}).get("warnings") or [])
    warnings.extend(memory.get("warnings") or [])
    warnings.extend(graph.get("warnings") or [])
    return _unique_strings(str(item) for item in warnings)


def _sync_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "root": report.get("root", ""),
        "discovered": report.get("discovered", 0),
        "indexed": report.get("indexed", 0),
        "skipped": report.get("skipped", 0),
        "failed": report.get("failed", 0),
        "error_count": len(report.get("errors") or []),
        "errors": report.get("errors") or [],
    }


def _suite_summary(suite: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": suite.get("schema"),
        "suite_id": suite.get("suite_id"),
        "name": suite.get("name"),
        "path": suite.get("path"),
        "query_count": suite.get("query_count", 0),
        "warnings": suite.get("warnings") or [],
    }


def _benchmark_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": report.get("schema"),
        "benchmark_id": report.get("benchmark_id"),
        "path": report.get("path"),
        "md_path": report.get("md_path"),
        "suite_name": report.get("suite_name"),
        "compare_modes": report.get("compare_modes") or [],
        "query_count": report.get("query_count", 0),
        "best_mode_by_score": report.get("best_mode_by_score"),
        "best_mode_by_node_recall": report.get("best_mode_by_node_recall"),
        "summary": report.get("summary") or {},
        "warnings": report.get("warnings") or [],
    }


def _memory_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": report.get("schema"),
        "path": report.get("path"),
        "status": report.get("status"),
        "memory_count": report.get("memory_count", 0),
        "expired_count": report.get("expired_count", 0),
        "duplicate_subject_count": report.get("duplicate_subject_count", 0),
        "suspected_pollution_count": report.get("suspected_pollution_count", 0),
        "resume_available": report.get("resume_available", False),
        "warnings": report.get("warnings") or [],
    }


def _baseline_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Quality Baseline",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- baseline_id: `{report.get('baseline_id')}`",
        f"- corpus_path: `{report.get('corpus_path')}`",
        f"- doc_count: `{report.get('doc_count')}`",
        f"- pdf_count: `{report.get('pdf_count')}`",
        f"- best_search_mode: `{(report.get('benchmark') or {}).get('best_mode_by_score', '')}`",
        f"- real_embedding_status: `{(report.get('embedding') or {}).get('sentence_transformers', {}).get('status', '')}`",
        f"- warning_count: `{len(report.get('warnings') or [])}`",
        "",
        "## Documents",
    ]
    for doc in report.get("documents") or []:
        lines.append(
            f"- `{doc.get('doc_id')}` quality=`{doc.get('quality_level')}` "
            f"sections=`{doc.get('section_count')}` tables=`{doc.get('table_count')}` warnings=`{doc.get('warning_count')}`"
        )
    lines.extend(["", "## Parser Comparison"])
    for provider in (report.get("parser_comparison") or {}).get("providers") or []:
        lines.append(f"- `{provider.get('provider')}` status=`{provider.get('status')}` reason=`{provider.get('reason', '')}`")
    lines.extend(["", "## Recommendations"])
    for item in report.get("recommendations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _baseline_html(report: Dict[str, Any]) -> str:
    cards = [
        ("Docs", report.get("doc_count", 0)),
        ("PDFs", report.get("pdf_count", 0)),
        ("Warnings", len(report.get("warnings") or [])),
        ("Best Mode", (report.get("benchmark") or {}).get("best_mode_by_score", "")),
        ("Real Embedding", (report.get("embedding") or {}).get("sentence_transformers", {}).get("status", "")),
        ("Memory", (report.get("memory") or {}).get("status", "")),
        ("Graph Conflicts", (report.get("claim_graph") or {}).get("conflict_count", 0)),
        ("Graph Isolated", (report.get("claim_graph") or {}).get("isolated_fact_count", 0)),
    ]
    card_html = "\n".join(
        f"<section class='card'><div class='label'>{escape(str(label))}</div><div class='value'>{escape(str(value))}</div></section>"
        for label, value in cards
    )
    docs_html = _table(
        "Documents",
        ["Doc", "Quality", "Sections", "Tables", "Warnings"],
        [
            [
                doc.get("doc_id", ""),
                doc.get("quality_level", ""),
                doc.get("section_count", 0),
                doc.get("table_count", 0),
                doc.get("warning_count", 0),
            ]
            for doc in report.get("documents") or []
        ],
    )
    parser_html = _table(
        "Parser Comparison",
        ["Provider", "Status", "Reason"],
        [
            [item.get("provider", ""), item.get("status", ""), item.get("reason", "")]
            for item in (report.get("parser_comparison") or {}).get("providers") or []
        ],
    )
    links = _list_section(
        "Artifacts",
        [
            f"eval_set: {(report.get('eval_set') or {}).get('path', '')}",
            f"benchmark: {(report.get('benchmark') or {}).get('path', '')}",
            f"compare_task: {((report.get('tasks') or {}).get('compare') or {}).get('task_id', '')}",
            f"review_task: {((report.get('tasks') or {}).get('review') or {}).get('task_id', '')}",
            f"case_study: {((report.get('tasks') or {}).get('case_study') or {}).get('path', '')}",
            f"claim_graph: {(report.get('claim_graph') or {}).get('graph_dir', '')}",
        ],
    )
    recs = _list_section("Recommendations", report.get("recommendations") or [])
    warnings = _list_section("Warnings", report.get("warnings") or [])
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Quality Baseline</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; background: #f7f8fa; }}
    header {{ padding: 26px 32px; background: #fff; border-bottom: 1px solid #d9dee7; }}
    main {{ padding: 24px 32px 40px; }}
    h1 {{ margin: 0; font-size: 26px; letter-spacing: 0; }}
    h2 {{ font-size: 16px; margin: 0 0 10px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card, .panel {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
    .label {{ font-size: 12px; color: #64748b; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ border-top: 1px solid #edf1f5; padding: 8px; font-size: 13px; overflow-wrap: anywhere; text-align: left; }}
    ul {{ padding-left: 18px; margin: 0; }}
    li {{ margin: 6px 0; font-size: 13px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>Quality Baseline</h1>
    <p>{escape(str(report.get('baseline_id') or ''))}</p>
  </header>
  <main>
    <div class="grid">{card_html}</div>
    {docs_html}
    {parser_html}
    {links}
    {recs}
    {warnings}
  </main>
</body>
</html>
"""


def _table(title: str, headers: List[str], rows: List[List[Any]]) -> str:
    head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<section class='panel'><h2>{escape(title)}</h2><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>"


def _list_section(title: str, items: Iterable[Any]) -> str:
    rows = "\n".join(f"<li>{escape(str(item))}</li>" for item in items)
    return f"<section class='panel'><h2>{escape(title)}</h2><ul>{rows}</ul></section>"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _unique_strings(values: Iterable[Any]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = compact_whitespace(str(value))
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
