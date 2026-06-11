from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .artifacts import get_doc_card, get_parse_quality, get_parse_report
from .baseline_renderers import baseline_html, baseline_markdown
from .baseline_runtime import LLMBaselineRuntime, null_stage
from .benchmark import create_eval_suite, generate_case_study, run_benchmark
from .claim_frames import verify_claim_frames
from .config import DATA_DIR, PROJECT_ROOT, baseline_llm_stage_timeout_seconds, baseline_llm_timeout_seconds, deepseek_timeout_seconds
from .embeddings import (
    EmbeddingError,
    build_semantic_index,
    resolve_embedding_model,
    resolve_embedding_provider,
    semantic_index_status,
    sentence_transformers_available,
)
from .eval import eval_memory
from .fact_audit import fact_audit_summary
from .facts import extract_facts
from .ingest import discover_files, sync_directory
from .insights import extract_doc_insights
from .knowledge_graph import graph_summary
from .llm import llm_status
from .parsers import pdf_adapter_statuses
from .review import draft_review
from .tasks import COMPARE_DIMENSIONS, compare_papers, generate_review_plan
from .tree_search import tree_search
from .utils import compact_whitespace, read_json as _read_json, stable_id, unique_strings as _unique_strings, write_json


BASELINE_SCHEMA = "quality_baseline.v1"
CODE_VERSION = "v0.38"
BASELINE_FEATURE_FLAGS = {
    "review_draft_baseline": True,
    "baseline_staleness": True,
    "section_budgeted_review_draft": True,
    "evidence_first_review_draft": True,
    "baseline_llm_fast_path": True,
    "blocker_limitation_split": True,
    "parser_quality_comparison": True,
    "doc_card_summary_fields": True,
    "document_routing_explanations": True,
    "openai_compatible_embedding": True,
    "bge_hybrid_calibration": True,
    "doc_card_noise_filtering": True,
    "evidence_unit_claim_frame_chain": True,
    "claim_frame_verifier": True,
}
BASELINE_DIR = DATA_DIR / "eval"
EVAL_SET_DIR = DATA_DIR / "eval_sets"


def run_quality_baseline(
    db_path: Path,
    corpus_path: Path = Path("articles"),
    *,
    force: bool = True,
    top_k: int = 5,
    use_llm: bool = False,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_timeout_seconds: Optional[int] = None,
    llm_stage_timeout_seconds: Optional[int] = None,
    llm_max_docs: Optional[int] = None,
    skip_llm_tasks: bool = False,
) -> Dict[str, Any]:
    started = time.time()
    llm_runtime = LLMBaselineRuntime(
        enabled=use_llm,
        timeout_seconds=llm_timeout_seconds or deepseek_timeout_seconds(),
        total_timeout_seconds=baseline_llm_timeout_seconds(),
        stage_timeout_seconds=llm_stage_timeout_seconds or baseline_llm_stage_timeout_seconds(),
        max_docs=llm_max_docs if llm_max_docs is not None else (2 if use_llm else 0),
        skip_tasks=skip_llm_tasks,
    )
    root = corpus_path.expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    files = discover_files(root)
    pdf_files = [path for path in files if path.suffix.lower() == ".pdf"]
    warnings: List[str] = []
    if not files:
        warnings.append("empty_corpus")

    primary_sync = sync_directory(root, db_path, force=force, pdf_parser="pypdf", doc_card_use_llm=False)
    doc_ids = _doc_ids_for_corpus(db_path, root)
    llm_doc_ids = llm_runtime.limit_doc_ids(doc_ids)
    corpus_meta = _corpus_metadata(root, files)
    doc_reports = [_doc_quality_summary(db_path, doc_id) for doc_id in doc_ids]
    parser_comparison = _parser_comparison(root, pdf_files, primary_sync, doc_reports)
    llm = _baseline_llm_probe(llm_runtime)
    fact_audit_before = fact_audit_summary(db_path, doc_ids=doc_ids) if doc_ids else {}
    insights = _prepare_insights_and_facts(
        db_path,
        doc_ids,
        use_llm=use_llm,
        runtime=llm_runtime,
        llm_doc_ids=llm_doc_ids,
    )
    fact_audit_after = fact_audit_summary(db_path, doc_ids=doc_ids) if doc_ids else {}
    claim_frame_verification = verify_claim_frames(db_path, doc_ids=doc_ids) if doc_ids else {
        "schema": "claim_frame_verifier_result.v1",
        "status": "skipped",
        "doc_count": 0,
        "frame_count": 0,
        "verified_frame_count": 0,
        "verified_frame_rate": 0.0,
        "unsupported_frame_count": 0,
        "low_confidence_frame_count": 0,
        "missing_evidence_unit_count": 0,
        "warnings": ["no_ready_documents"],
    }
    embedding = _embedding_baseline(db_path, doc_ids, embedding_provider=embedding_provider, embedding_model=embedding_model)
    with _embedding_search_env(embedding):
        eval_set = _write_baseline_eval_set(doc_reports)
        suite = create_eval_suite(db_path, f"quality_baseline_{int(started)}", input_json=Path(eval_set["path"]))
        benchmark_modes = _baseline_benchmark_modes(embedding)
        benchmark = run_benchmark(db_path, str(suite["name"]), compare_modes=benchmark_modes, top_k=top_k)
        tree = _tree_search_baseline(
            db_path,
            doc_reports,
            top_k=top_k,
            use_llm=use_llm,
            runtime=llm_runtime,
            llm_doc_ids=llm_doc_ids,
        )
        tasks = _task_baseline(
            db_path,
            doc_ids,
            use_llm=use_llm,
            runtime=llm_runtime,
            llm_doc_ids=llm_doc_ids,
            skip_llm_tasks=skip_llm_tasks,
        )
    memory = eval_memory(db_path)
    graph = graph_summary(db_path, doc_ids=doc_ids, include_conflicts=True) if doc_ids else {
        "schema": "knowledge_graph_summary.v1",
        "available": False,
        "warnings": ["no_ready_documents_for_graph"],
    }
    llm_baseline = _llm_baseline_summary(llm, insights, tree, tasks, graph, enabled=use_llm, runtime=llm_runtime.summary())
    recommendations = _recommendations(doc_reports, parser_comparison, embedding, benchmark, tree, tasks, memory, graph, claim_frame_verification)
    warnings.extend(
        _baseline_warnings(
            doc_reports,
            parser_comparison,
            embedding,
            benchmark,
            tasks,
            memory,
            graph,
            llm_baseline,
            claim_frame_verification,
        )
    )
    baseline_id = stable_id("quality_baseline", str(root), ",".join(doc_ids), started, length=12)
    git_commit = _current_git_commit()
    report = {
        "schema": BASELINE_SCHEMA,
        "code_version": CODE_VERSION,
        "git_commit": git_commit,
        "feature_flags": dict(BASELINE_FEATURE_FLAGS),
        "is_current_code_baseline": True,
        "baseline_stale_reason": "",
        "baseline_id": baseline_id,
        **corpus_meta,
        "corpus_path": str(root),
        "file_count": len(files),
        "pdf_count": len(pdf_files),
        "doc_ids": doc_ids,
        "doc_count": len(doc_ids),
        "primary_parser": "pypdf",
        "primary_sync": _sync_summary(primary_sync),
        "llm_status": llm,
        "llm_runtime": llm_runtime.summary(),
        "parser_comparison": parser_comparison,
        "documents": doc_reports,
        "insights_and_facts": insights,
        "fact_audit_delta": _fact_audit_delta(fact_audit_before, fact_audit_after),
        "claim_frame_verification": claim_frame_verification,
        "embedding": embedding,
        "eval_set": eval_set,
        "eval_suite": _suite_summary(suite),
        "benchmark": _benchmark_summary(benchmark),
        "tree_search": tree,
        "llm_baseline": llm_baseline,
        "tasks": tasks,
        "memory": _memory_summary(memory),
        "claim_graph": graph,
        "recommendations": recommendations,
        "warnings": _unique_strings(warnings),
        "created_at": started,
        "duration_ms": round((time.time() - started) * 1000, 3),
    }
    report["top_review_blockers"] = _top_review_blockers(report)
    report["baseline_limitations"] = _baseline_limitations(report)
    report["llm_runtime_limitations"] = _llm_runtime_limitations(report)
    json_path = BASELINE_DIR / f"quality_baseline_{baseline_id}.json"
    md_path = BASELINE_DIR / f"quality_baseline_{baseline_id}.md"
    html_path = BASELINE_DIR / f"quality_baseline_{baseline_id}.html"
    write_json(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(baseline_markdown(report), encoding="utf-8")
    html_path.write_text(baseline_html(report), encoding="utf-8")
    return {
        **report,
        "path": str(json_path),
        "json_path": str(json_path),
        "md_path": str(md_path),
        "html_path": str(html_path),
    }


def latest_quality_baseline(
    limit: int = 1,
    *,
    corpus: Optional[Any] = None,
    real_only: bool = False,
    exclude_temp: bool = False,
) -> Dict[str, Any]:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = _resolve_corpus_filter(corpus) if corpus is not None else None
    current_commit = _current_git_commit()
    items = []
    paths = sorted(
        BASELINE_DIR.glob("quality_baseline_*.json"),
        key=lambda item: _baseline_sort_key(item),
    )
    for path in paths:
        payload = _read_json(path, {})
        if payload.get("schema") != BASELINE_SCHEMA:
            continue
        if corpus_path is not None and str(payload.get("corpus_path") or "") != str(corpus_path):
            continue
        is_real = _payload_is_real_corpus(payload)
        if real_only and not is_real:
            continue
        if exclude_temp and payload.get("run_kind") == "test_fixture":
            continue
        review = ((payload.get("tasks") or {}).get("review") or {})
        review_diagnostics = review.get("llm_diagnostics") or {}
        tree_delta = (payload.get("tree_search") or {}).get("comparison_summary") or {}
        fact_delta = payload.get("fact_audit_delta") or {}
        claim_frame_verification = payload.get("claim_frame_verification") or {}
        llm_baseline = payload.get("llm_baseline") or {}
        stage_summary = llm_baseline.get("stage_summary") or {}
        llm_facts = llm_baseline.get("insights_and_facts") or {}
        llm_tasks = llm_baseline.get("tasks") or {}
        review_draft = ((payload.get("tasks") or {}).get("review_draft") or {})
        stale_reason = _baseline_stale_reason(payload, current_commit)
        items.append(
            {
                "path": str(path),
                "baseline_id": payload.get("baseline_id") or "",
                "code_version": payload.get("code_version") or "",
                "git_commit": payload.get("git_commit") or "",
                "baseline_git_commit_short": _short_commit(payload.get("git_commit") or ""),
                "current_git_commit": current_commit,
                "current_git_commit_short": _short_commit(current_commit),
                "feature_flags": payload.get("feature_flags") or {},
                "is_current_code_baseline": not bool(stale_reason),
                "baseline_stale_reason": stale_reason,
                "run_kind": payload.get("run_kind") or "",
                "corpus_name": payload.get("corpus_name") or "",
                "corpus_fingerprint": payload.get("corpus_fingerprint") or "",
                "is_real_corpus": is_real,
                "corpus_path": payload.get("corpus_path") or "",
                "doc_count": payload.get("doc_count", 0),
                "pdf_count": payload.get("pdf_count", 0),
                "best_search_mode": (payload.get("benchmark") or {}).get("best_mode_by_score") or "",
                "llm_baseline_status": llm_baseline.get("status", ""),
                "llm_reachable": ((payload.get("llm_status") or {}).get("reachable")),
                "llm_stage_status": {name: (stage.get("status") if isinstance(stage, dict) else "") for name, stage in stage_summary.items()},
                "llm_timeout_count": llm_baseline.get("timeout_count", 0),
                "llm_hard_timeout_count": llm_baseline.get("hard_timeout_count", 0),
                "llm_slow_call_count": llm_baseline.get("slow_call_count", 0),
                "llm_total_duration_ms": llm_baseline.get("total_llm_duration_ms", 0.0),
                "llm_budget_exhausted": bool(llm_baseline.get("budget_exhausted")),
                "llm_facts_success_rate": llm_facts.get("llm_facts_success_rate", 0.0),
                "llm_facts_batch_timeout_count": llm_facts.get("llm_facts_batch_timeout_count", 0),
                "llm_compare_dimension_success_rate": llm_tasks.get("llm_compare_dimension_success_rate", 0.0),
                "llm_compare_dimension_timeout_count": llm_tasks.get("compare_dimension_timeout_count", 0),
                "review_llm_error": (((payload.get("tasks") or {}).get("review") or {}).get("llm_error") or ""),
                "review_fallback_mode": review_diagnostics.get("mode") or "",
                "review_partial_reasons": review.get("review_partial_reasons") or [],
                "review_draft_status": review_draft.get("status", ""),
                "review_draft_skip_reason": review_draft.get("review_draft_skip_reason", "") or review_draft.get("reason", ""),
                "review_draft_quality_level": review_draft.get("draft_quality_level", ""),
                "citation_coverage_score": review_draft.get("citation_coverage_score", 0.0),
                "missing_ref_count": review_draft.get("missing_ref_count", 0),
                "unsupported_paragraph_count": review_draft.get("unsupported_paragraph_count", 0),
                "optional_unused_evidence_count": review_draft.get("optional_unused_evidence_count", 0),
                "removed_paragraph_count": (review_draft.get("paragraph_support_report") or {}).get("removed_paragraph_count", 0),
                "drafted_section_count": review_draft.get("drafted_section_count", 0),
                "skipped_section_count": review_draft.get("skipped_section_count", 0),
                "review_draft_path": review_draft.get("review_draft_path", ""),
                "section_revision_actions": review_draft.get("section_revision_actions") or [],
                "top_review_blockers": _top_review_blockers(payload),
                "baseline_limitations": _baseline_limitations(payload),
                "llm_runtime_limitations": _llm_runtime_limitations(payload),
                "duplicate_evidence_removed": review.get("duplicate_evidence_removed", 0),
                "citation_gap_count_before": fact_delta.get("citation_gap_count_before", 0),
                "citation_gap_count_after": fact_delta.get("citation_gap_count_after", 0),
                "evidence_unit_count": sum(
                    int(item.get("evidence_unit_count") or 0)
                    for item in (payload.get("insights_and_facts") or {}).get("items") or []
                    if isinstance(item, dict)
                ),
                "claim_frame_count": claim_frame_verification.get("frame_count", 0),
                "verified_frame_rate": claim_frame_verification.get("verified_frame_rate", 0.0),
                "unsupported_frame_count": claim_frame_verification.get("unsupported_frame_count", 0),
                "tree_trace_completeness_before": tree_delta.get("rule_trace_completeness_avg", 0.0),
                "tree_trace_completeness_after": tree_delta.get("llm_trace_completeness_avg", 0.0),
                "weak_doc_count": sum(1 for item in payload.get("documents") or [] if item.get("quality_level") == "weak"),
                "real_embedding_provider": (payload.get("embedding") or {}).get("real_embedding_provider", ""),
                "real_embedding_status": (payload.get("embedding") or {}).get("real_embedding_status")
                or (payload.get("embedding") or {}).get("sentence_transformers", {}).get("status", ""),
                "real_embedding_model": (payload.get("embedding") or {}).get("real_embedding_model", ""),
                "real_embedding_dim": (payload.get("embedding") or {}).get("real_embedding_dim", 0),
                "real_embedding_node_coverage": (payload.get("embedding") or {}).get("real_embedding_node_coverage", 0.0),
                "real_embedding_doc_coverage": (payload.get("embedding") or {}).get("real_embedding_doc_coverage", 0.0),
                "hybrid_embedding_provider": (payload.get("embedding") or {}).get("hybrid_embedding_provider", ""),
                "hybrid_embedding_model": (payload.get("embedding") or {}).get("hybrid_embedding_model", ""),
                "embedding_rebuild_needed": bool((payload.get("embedding") or {}).get("embedding_rebuild_needed")),
                "warning_count": len(payload.get("warnings") or []),
                "warnings": payload.get("warnings") or [],
                "created_at": payload.get("created_at"),
            }
        )
        if len(items) >= limit:
            break
    return {"schema": "quality_baseline_latest.v1", "count": len(items), "items": items}


def _corpus_metadata(root: Path, files: List[Path]) -> Dict[str, Any]:
    articles_root = (PROJECT_ROOT / "articles").resolve()
    is_real = root == articles_root
    run_kind = "real_articles" if is_real else ("test_fixture" if _looks_like_temp_path(root) else "local_corpus")
    return {
        "run_kind": run_kind,
        "corpus_name": root.name or str(root),
        "corpus_fingerprint": _corpus_fingerprint(root, files),
        "is_real_corpus": is_real,
    }


def _current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip()


def _short_commit(value: str) -> str:
    text = str(value or "").strip()
    return text[:7] if text else ""


def _baseline_stale_reason(payload: Dict[str, Any], current_commit: str = "") -> str:
    flags = payload.get("feature_flags") or {}
    if not isinstance(flags, dict) or not flags.get("review_draft_baseline"):
        return "missing_review_draft_baseline"
    if payload.get("code_version") != CODE_VERSION:
        return "older_code_version"
    baseline_commit = str(payload.get("git_commit") or "")
    if not baseline_commit:
        return "missing_git_commit"
    if current_commit and baseline_commit != current_commit:
        return "older_git_commit"
    return ""


def _top_review_blockers(payload: Dict[str, Any]) -> List[str]:
    tasks = payload.get("tasks") or {}
    review = tasks.get("review") or {}
    review_draft = tasks.get("review_draft") or {}
    llm = payload.get("llm_baseline") or {}
    blockers: List[str] = []
    hard_review_reasons = {
        "missing_refs",
        "unsupported_paragraphs",
        "low_evidence_coverage",
        "section_draft_missing",
        "section_draft_skipped",
        "duplicate_evidence",
    }
    blockers.extend(reason for reason in review.get("review_partial_reasons") or [] if reason in hard_review_reasons)
    blockers.extend(reason for reason in review_draft.get("quality_reasons") or [] if reason in hard_review_reasons)
    if not review_draft or review_draft.get("status") in {"", "skipped"}:
        blockers.append("review_draft_skipped")
    if int(review_draft.get("skipped_section_count") or 0) > 0:
        blockers.append("section_draft_skipped")
    if review_draft.get("status") == "partial" and any(reason in blockers for reason in hard_review_reasons):
        blockers.append("review_draft_partial")
    if int(review_draft.get("skipped_section_count") or 0) > 0 and (review_draft.get("review_draft_skip_reason") or review_draft.get("reason")):
        blockers.append(str(review_draft.get("review_draft_skip_reason") or review_draft.get("reason")))
    if int(review_draft.get("missing_ref_count") or 0) > 0:
        blockers.append("missing_refs")
    if int(review_draft.get("unsupported_paragraph_count") or 0) > 0:
        blockers.append("unsupported_paragraphs")
    if review_draft.get("status") and float(review_draft.get("citation_coverage_score") or 0.0) < 0.8:
        blockers.append("low_evidence_coverage")
    if "duplicate_evidence" in review.get("review_partial_reasons", []) and int((review.get("evidence_coverage") or {}).get("post_dedupe_duplicate_count") or 0) > 0:
        blockers.append("duplicate_evidence")
    priority = [
        "missing_refs",
        "unsupported_paragraphs",
        "low_evidence_coverage",
        "review_draft_skipped",
        "section_draft_skipped",
        "review_draft_partial",
        "duplicate_evidence",
    ]
    limitation_tags = set(_baseline_limitations(payload))
    post_duplicate_count = int((review.get("evidence_coverage") or {}).get("post_dedupe_duplicate_count") or 0)
    operational_tags = {
        "llm_unavailable",
        "rule_based_section_draft",
        "llm_budget_exhausted",
        "review_draft_partial",
    }
    unique = [
        item
        for item in _unique_strings(blockers)
        if item not in limitation_tags
        and item not in operational_tags
        and not str(item).startswith("llm_review_draft_stage_budget_exhausted")
        and not (item == "duplicate_evidence" and post_duplicate_count <= 0)
    ]
    return sorted(unique, key=lambda item: priority.index(item) if item in priority else len(priority))[:8]


def _baseline_limitations(payload: Dict[str, Any]) -> List[str]:
    llm = payload.get("llm_baseline") or {}
    runtime = payload.get("llm_runtime") or {}
    embedding = payload.get("embedding") or {}
    limitations: List[str] = []
    if int(payload.get("doc_count") or 0) < 3:
        limitations.append("small_corpus")
    if embedding.get("real_embedding_status") != "completed":
        limitations.append("real_embedding_not_enabled")
    if int(llm.get("timeout_count") or runtime.get("timeout_count") or 0) > 0:
        limitations.append("llm_timeout")
    if int(llm.get("hard_timeout_count") or runtime.get("hard_timeout_count") or 0) > 0:
        limitations.append("hard_llm_timeout")
    if int(llm.get("slow_call_count") or runtime.get("slow_call_count") or 0) > 0:
        limitations.append("slow_llm_calls")
    if llm.get("budget_exhausted") or runtime.get("budget_exhausted"):
        limitations.append("baseline_llm_budget_exhausted")
    tasks = payload.get("tasks") or {}
    review_draft = tasks.get("review_draft") or {}
    quality_reasons = set(review_draft.get("quality_reasons") or [])
    warnings = set((review_draft.get("warnings") or []) + (payload.get("warnings") or []))
    if {"llm_unavailable", "rule_based_section_draft", "llm_budget_exhausted"} & quality_reasons or any(
        str(warning).startswith("llm_unavailable") or str(warning).startswith("llm_review_draft_stage_budget_exhausted")
        for warning in warnings
    ):
        limitations.append("llm_rule_fallback")
    if "optional_parser_skipped" in (payload.get("warnings") or []):
        limitations.append("optional_parser_skipped")
    return _unique_strings(limitations)


def _llm_runtime_limitations(payload: Dict[str, Any]) -> List[str]:
    llm = payload.get("llm_baseline") or {}
    runtime = payload.get("llm_runtime") or {}
    limitations: List[str] = []
    if int(llm.get("timeout_count") or runtime.get("timeout_count") or 0) > 0:
        limitations.append("llm_timeout")
    if int(llm.get("hard_timeout_count") or runtime.get("hard_timeout_count") or 0) > 0:
        limitations.append("hard_llm_timeout")
    if int(llm.get("slow_call_count") or runtime.get("slow_call_count") or 0) > 0:
        limitations.append("slow_llm_calls")
    if llm.get("budget_exhausted") or runtime.get("budget_exhausted"):
        limitations.append("baseline_llm_budget_exhausted")
    return _unique_strings(limitations)


def _corpus_fingerprint(root: Path, files: List[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(str(root).encode("utf-8"))
    for path in sorted(files, key=lambda item: str(item)):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
        digest.update(rel.encode("utf-8"))
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(str(int(stat.st_mtime)).encode("ascii"))
    return digest.hexdigest()[:16]


def _looks_like_temp_path(path: Path) -> bool:
    text = str(path)
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        if path.is_relative_to(temp_root):
            return True
    except ValueError:
        pass
    return "/private/var/folders/" in text or "/tmp/" in text


def _resolve_corpus_filter(corpus: Any) -> Path:
    path = Path(corpus).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _baseline_sort_key(path: Path) -> tuple[int, float]:
    payload = _read_json(path, {})
    if payload.get("schema") != BASELINE_SCHEMA:
        return (3, -path.stat().st_mtime)
    if _payload_is_real_corpus(payload):
        priority = 0
    elif payload.get("run_kind") == "test_fixture":
        priority = 2
    else:
        priority = 1
    return (priority, -float(path.stat().st_mtime))


def _payload_is_real_corpus(payload: Dict[str, Any]) -> bool:
    if bool(payload.get("is_real_corpus")):
        return True
    try:
        return Path(str(payload.get("corpus_path") or "")).resolve() == (PROJECT_ROOT / "articles").resolve()
    except OSError:
        return False


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


PARSER_QUALITY_METRICS = ["metadata", "abstract", "section", "reference", "table", "figure", "warning"]


def _parser_comparison(
    root: Path,
    pdf_files: List[Path],
    primary_sync: Dict[str, Any],
    primary_documents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    statuses = pdf_adapter_statuses()
    providers = []
    for provider in ("pypdf", "docling", "grobid"):
        providers.append(_parser_provider_result(root, pdf_files, provider, statuses, primary_sync, primary_documents))
    return {
        "schema": "parser_comparison.v1",
        "pdf_count": len(pdf_files),
        "quality_metrics": PARSER_QUALITY_METRICS,
        "adapter_statuses": statuses,
        "providers": providers,
        "summary_by_provider": {
            str(provider.get("provider")): provider.get("quality_summary", {})
            for provider in providers
        },
    }


def _parser_provider_result(
    root: Path,
    pdf_files: List[Path],
    provider: str,
    statuses: Dict[str, Any],
    primary_sync: Dict[str, Any],
    primary_documents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not pdf_files:
        return _skipped_parser_provider(provider, "no_pdf_files")
    if provider == "docling" and not (statuses.get("docling") or {}).get("available"):
        return _skipped_parser_provider(provider, "docling_not_installed")
    if provider == "grobid" and not os.environ.get("GROBID_URL", "").strip():
        return _skipped_parser_provider(provider, "GROBID_URL_not_configured")
    if provider == "pypdf":
        status = "completed" if int(primary_sync.get("failed") or 0) == 0 else "partial"
        return {
            "provider": provider,
            "status": status,
            "sync": _sync_summary(primary_sync),
            "documents": primary_documents,
            "quality_summary": _parser_quality_summary(primary_documents, status=status),
        }
    with tempfile.TemporaryDirectory(prefix=f"kb_baseline_{provider}_") as tmp:
        temp_db = Path(tmp) / "kb.sqlite"
        report = sync_directory(root, temp_db, force=True, pdf_parser=provider, doc_card_use_llm=False)
        doc_ids = _doc_ids_for_corpus(temp_db, root)
        documents = [_doc_quality_summary(temp_db, doc_id) for doc_id in doc_ids]
    status = "completed" if int(report.get("failed") or 0) == 0 else "partial"
    return {
        "provider": provider,
        "status": status,
        "sync": _sync_summary(report),
        "documents": documents,
        "quality_summary": _parser_quality_summary(documents, status=status),
    }


def _skipped_parser_provider(provider: str, reason: str) -> Dict[str, Any]:
    return {
        "provider": provider,
        "status": "skipped",
        "reason": reason,
        "documents": [],
        "quality_summary": _parser_quality_summary([], status="skipped", reason=reason),
    }


def _parser_quality_summary(
    documents: List[Dict[str, Any]],
    *,
    status: str,
    reason: str = "",
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "schema": "parser_quality_summary.v1",
        "status": status,
        "reason": reason,
        "document_count": len(documents),
        "metadata": {
            "avg_score": _avg_metric(documents, "metadata_score"),
            "min_score": _min_metric(documents, "metadata_score"),
        },
        "abstract": {
            "present_count": sum(1 for doc in documents if not doc.get("missing_abstract")),
            "missing_count": sum(1 for doc in documents if doc.get("missing_abstract")),
        },
        "section": {
            "total_count": sum(int(doc.get("section_count") or 0) for doc in documents),
            "avg_count": _avg_metric(documents, "section_count"),
            "page_only_tree_count": sum(1 for doc in documents if doc.get("page_only_tree")),
        },
        "reference": {
            "total_count": sum(int(doc.get("reference_count") or 0) for doc in documents),
            "avg_count": _avg_metric(documents, "reference_count"),
            "avg_score": _avg_metric(documents, "reference_score"),
        },
        "table": {
            "total_count": sum(int(doc.get("table_count") or 0) for doc in documents),
            "content_count": sum(int(doc.get("table_content_count") or 0) for doc in documents),
            "avg_parse_score": _avg_metric(documents, "table_parse_score"),
        },
        "figure": {
            "total_count": sum(int(doc.get("figure_count") or 0) for doc in documents),
            "avg_count": _avg_metric(documents, "figure_count"),
        },
        "warning": {
            "total_count": sum(int(doc.get("warning_count") or 0) for doc in documents),
            "documents_with_warnings": sum(1 for doc in documents if int(doc.get("warning_count") or 0) > 0),
            "items": _parser_warning_items(documents),
        },
    }
    return summary


def _avg_metric(documents: List[Dict[str, Any]], field: str) -> float:
    values = [_numeric_metric(doc.get(field)) for doc in documents]
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _min_metric(documents: List[Dict[str, Any]], field: str) -> float:
    values = [_numeric_metric(doc.get(field)) for doc in documents]
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return round(min(values), 3)


def _numeric_metric(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parser_warning_items(documents: List[Dict[str, Any]]) -> List[str]:
    items: List[str] = []
    seen = set()
    for doc in documents:
        for warning in doc.get("warnings") or []:
            text = str(warning)
            if text and text not in seen:
                seen.add(text)
                items.append(text)
    return items[:12]


def _baseline_llm_probe(runtime: LLMBaselineRuntime) -> Dict[str, Any]:
    if not runtime.enabled:
        return llm_status(probe=False)
    with runtime.stage("llm_probe") as stage:
        if not stage.allowed:
            status = llm_status(probe=False)
            return {
                **status,
                "probe": True,
                "reachable": False,
                "error": stage.reason or "baseline_llm_budget_exhausted",
            }
        status = llm_status(probe=True, timeout_seconds=min(runtime.timeout_seconds, 15))
        if not status.get("reachable"):
            stage.mark_fallback(str(status.get("error") or "llm_unreachable"))
        return status


def _prepare_insights_and_facts(
    db_path: Path,
    doc_ids: List[str],
    *,
    use_llm: bool,
    runtime: Optional[LLMBaselineRuntime] = None,
    llm_doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    items = []
    warnings: List[str] = []
    llm_doc_set = set(llm_doc_ids or doc_ids)
    rule_by_doc: Dict[str, Dict[str, Any]] = {}
    for doc_id in doc_ids:
        try:
            rule_summary = _extract_insight_fact_summary(db_path, doc_id, use_llm=False)
            rule_by_doc[doc_id] = rule_summary
            if not use_llm:
                items.append(rule_summary)
        except Exception as exc:
            warnings.append(f"insight_fact_failed:{doc_id}:{exc}")
            rule_by_doc[doc_id] = {"doc_id": doc_id, "innovation_status": "skipped", "fact_status": "skipped", "error": str(exc)}
            if not use_llm:
                items.append(rule_by_doc[doc_id])
    if use_llm:
        llm_by_doc: Dict[str, Dict[str, Any]] = {}
        with (runtime.stage("llm_insights") if runtime else null_stage("llm_insights")) as stage:
            for doc_id in doc_ids:
                if doc_id not in llm_doc_set:
                    continue
                if not stage.can_continue():
                    warnings.append("llm_insights_skipped:baseline_llm_budget_exhausted")
                    break
                try:
                    llm_by_doc.setdefault(doc_id, {})["insight"] = _extract_insight_summary(db_path, doc_id, use_llm=True)
                except Exception as exc:
                    stage.mark_fallback(getattr(exc, "error_type", "") or "llm_insight_failed")
                    llm_by_doc.setdefault(doc_id, {})["insight"] = {"doc_id": doc_id, "mode": "llm", "innovation_status": "partial", "llm_error": str(exc)}
        with (runtime.stage("llm_facts") if runtime else null_stage("llm_facts")) as stage:
            for doc_id in doc_ids:
                if doc_id not in llm_doc_set:
                    continue
                if not stage.can_continue():
                    warnings.append("llm_facts_skipped:baseline_llm_budget_exhausted")
                    break
                try:
                    llm_by_doc.setdefault(doc_id, {})["fact"] = _extract_fact_summary(db_path, doc_id, use_llm=True)
                except Exception as exc:
                    stage.mark_fallback(getattr(exc, "error_type", "") or "llm_fact_failed")
                    llm_by_doc.setdefault(doc_id, {})["fact"] = {"doc_id": doc_id, "mode": "llm", "fact_status": "partial", "llm_error": str(exc)}
        for doc_id in doc_ids:
            rule_summary = rule_by_doc.get(doc_id) or {"doc_id": doc_id}
            if doc_id not in llm_doc_set:
                items.append({**rule_summary, "llm": {"status": "skipped", "reason": "llm_max_docs"}, "warnings": _unique_strings([*rule_summary.get("warnings", []), "llm_doc_skipped:max_docs"])})
                continue
            llm_summary = _merge_insight_fact_summaries(rule_summary, llm_by_doc.get(doc_id) or {})
            items.append(
                {
                    "doc_id": doc_id,
                    "innovation_status": llm_summary.get("innovation_status", "partial"),
                    "innovation_count": llm_summary.get("innovation_count", 0),
                    "citation_reference_count": llm_summary.get("citation_reference_count", 0),
                    "citation_relation_count": llm_summary.get("citation_relation_count", 0),
                    "fact_status": llm_summary.get("fact_status", "partial"),
                    "claim_count": llm_summary.get("claim_count", 0),
                    "entity_count": llm_summary.get("entity_count", 0),
                    "relation_count": llm_summary.get("relation_count", 0),
                    "evidence_unit_count": llm_summary.get("evidence_unit_count", 0),
                    "claim_frame_count": llm_summary.get("claim_frame_count", 0),
                    "verified_frame_rate": llm_summary.get("verified_frame_rate", 0.0),
                    "source": llm_summary.get("source", ""),
                    "llm_used": llm_summary.get("llm_used", False),
                    "llm_error": llm_summary.get("llm_error", ""),
                        "noise_filtered_count": llm_summary.get("noise_filtered_count", 0),
                        "entity_noise_filtered_count": llm_summary.get("entity_noise_filtered_count", 0),
                        "long_claim_trimmed_count": llm_summary.get("long_claim_trimmed_count", 0),
                        "llm_mode": llm_summary.get("llm_mode", ""),
                        "batch_count": llm_summary.get("batch_count", 0),
                        "batch_success_count": llm_summary.get("batch_success_count", 0),
                        "batch_timeout_count": llm_summary.get("batch_timeout_count", 0),
                        "batch_fallback_count": llm_summary.get("batch_fallback_count", 0),
                        "llm_batch_success_rate": llm_summary.get("llm_batch_success_rate", 0.0),
                        "rule": rule_summary,
                        "llm": llm_summary,
                    "warnings": _unique_strings(llm_summary.get("warnings") or []),
                }
            )
    return {
        "schema": "baseline_insights_facts.v1",
        "use_llm": use_llm,
        "llm_doc_ids": list(llm_doc_ids or []),
        "doc_count": len(items),
        "items": items,
        "warnings": _unique_strings(warnings),
    }


def _extract_insight_fact_summary(db_path: Path, doc_id: str, *, use_llm: bool) -> Dict[str, Any]:
    insight_summary = _extract_insight_summary(db_path, doc_id, use_llm=use_llm)
    fact_summary = _extract_fact_summary(db_path, doc_id, use_llm=use_llm)
    return _merge_insight_fact_summaries({}, {"insight": insight_summary, "fact": fact_summary})


def _extract_insight_summary(db_path: Path, doc_id: str, *, use_llm: bool) -> Dict[str, Any]:
    insight = extract_doc_insights(db_path, doc_id, force=True, use_llm=use_llm, require_llm=False)
    innovation = insight.get("innovation") or {}
    citation_map = insight.get("citation_map") or {}
    return {
        "doc_id": doc_id,
        "mode": "llm" if use_llm else "rule",
        "innovation_status": str(innovation.get("status") or "partial"),
        "innovation_source": innovation.get("source") or "",
        "innovation_count": len(innovation.get("items") or []),
        "citation_reference_count": len(citation_map.get("references") or []),
        "citation_relation_count": len(citation_map.get("relations") or []),
        "llm_error": insight.get("llm_error") or "",
        "warnings": _unique_strings(innovation.get("warnings") or []),
    }


def _extract_fact_summary(db_path: Path, doc_id: str, *, use_llm: bool) -> Dict[str, Any]:
    fact = extract_facts(db_path, doc_id, force=True, use_llm=use_llm, require_llm=False)
    fact_report = fact.get("fact_report") or {}
    return {
        "doc_id": doc_id,
        "mode": "llm" if use_llm else "rule",
        "fact_status": str(fact_report.get("status") or "partial"),
        "source": fact_report.get("source") or "",
        "llm_used": bool(fact_report.get("llm_used")),
        "llm_error": fact_report.get("llm_error") or "",
        "claim_count": fact_report.get("claim_count", 0),
        "entity_count": fact_report.get("entity_count", 0),
        "relation_count": fact_report.get("relation_count", 0),
        "evidence_unit_count": fact_report.get("evidence_unit_count", 0),
        "claim_frame_count": fact_report.get("claim_frame_count", 0),
        "verified_frame_rate": fact_report.get("verified_frame_rate", 0.0),
        "noise_filtered_count": fact_report.get("noise_filtered_count", 0),
        "entity_noise_filtered_count": fact_report.get("entity_noise_filtered_count", 0),
        "long_claim_trimmed_count": fact_report.get("long_claim_trimmed_count", 0),
        "llm_mode": fact_report.get("llm_mode") or "",
        "batch_count": fact_report.get("batch_count", 0),
        "batch_success_count": fact_report.get("batch_success_count", 0),
        "batch_timeout_count": fact_report.get("batch_timeout_count", 0),
        "batch_fallback_count": fact_report.get("batch_fallback_count", 0),
        "llm_batch_success_rate": fact_report.get("llm_batch_success_rate", 0.0),
        "warnings": _unique_strings(fact_report.get("warnings", [])),
    }


def _merge_insight_fact_summaries(rule_summary: Dict[str, Any], llm_parts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    insight = llm_parts.get("insight") or {}
    fact = llm_parts.get("fact") or {}
    merged = {
        "doc_id": insight.get("doc_id") or fact.get("doc_id") or rule_summary.get("doc_id", ""),
        "mode": insight.get("mode") or fact.get("mode") or rule_summary.get("mode", ""),
        "innovation_status": insight.get("innovation_status") or rule_summary.get("innovation_status", "partial"),
        "innovation_source": insight.get("innovation_source") or rule_summary.get("innovation_source", ""),
        "innovation_count": insight.get("innovation_count", rule_summary.get("innovation_count", 0)),
        "citation_reference_count": insight.get("citation_reference_count", rule_summary.get("citation_reference_count", 0)),
        "citation_relation_count": insight.get("citation_relation_count", rule_summary.get("citation_relation_count", 0)),
        "fact_status": fact.get("fact_status") or rule_summary.get("fact_status", "partial"),
        "source": fact.get("source") or insight.get("innovation_source") or rule_summary.get("source", ""),
        "llm_used": bool(fact.get("llm_used")) or bool(rule_summary.get("llm_used")),
        "llm_error": fact.get("llm_error") or insight.get("llm_error") or "",
        "claim_count": fact.get("claim_count", rule_summary.get("claim_count", 0)),
        "entity_count": fact.get("entity_count", rule_summary.get("entity_count", 0)),
        "relation_count": fact.get("relation_count", rule_summary.get("relation_count", 0)),
        "evidence_unit_count": fact.get("evidence_unit_count", rule_summary.get("evidence_unit_count", 0)),
        "claim_frame_count": fact.get("claim_frame_count", rule_summary.get("claim_frame_count", 0)),
        "verified_frame_rate": fact.get("verified_frame_rate", rule_summary.get("verified_frame_rate", 0.0)),
        "noise_filtered_count": fact.get("noise_filtered_count", rule_summary.get("noise_filtered_count", 0)),
        "entity_noise_filtered_count": fact.get("entity_noise_filtered_count", rule_summary.get("entity_noise_filtered_count", 0)),
        "long_claim_trimmed_count": fact.get("long_claim_trimmed_count", rule_summary.get("long_claim_trimmed_count", 0)),
        "llm_mode": fact.get("llm_mode", rule_summary.get("llm_mode", "")),
        "batch_count": fact.get("batch_count", rule_summary.get("batch_count", 0)),
        "batch_success_count": fact.get("batch_success_count", rule_summary.get("batch_success_count", 0)),
        "batch_timeout_count": fact.get("batch_timeout_count", rule_summary.get("batch_timeout_count", 0)),
        "batch_fallback_count": fact.get("batch_fallback_count", rule_summary.get("batch_fallback_count", 0)),
        "llm_batch_success_rate": fact.get("llm_batch_success_rate", rule_summary.get("llm_batch_success_rate", 0.0)),
        "warnings": _unique_strings([*rule_summary.get("warnings", []), *insight.get("warnings", []), *fact.get("warnings", [])]),
    }
    return merged


def _embedding_baseline(
    db_path: Path,
    doc_ids: List[str],
    *,
    embedding_provider: Optional[str],
    embedding_model: Optional[str],
) -> Dict[str, Any]:
    real_provider = _baseline_real_embedding_provider(embedding_provider)
    real_model = resolve_embedding_model(real_provider, embedding_model)
    result: Dict[str, Any] = {
        "schema": "embedding_baseline.v1",
        "doc_ids": doc_ids,
        "hash": {},
        "sentence_transformers": {},
        "openai_compatible": {},
        "real_embedding_provider": real_provider,
        "hybrid_embedding_provider": "hash",
        "hybrid_embedding_model": "hash-ngram-v1",
        "hybrid_embedding_ready": False,
        "real_embedding_status": "skipped",
        "real_embedding_model": real_model,
        "real_embedding_dim": 0,
        "real_embedding_node_coverage": 0.0,
        "real_embedding_doc_coverage": 0.0,
        "embedding_rebuild_needed": True,
        "warnings": [],
    }
    if not doc_ids:
        result["hash"] = {"status": "skipped", "reason": "no_ready_documents"}
        result["sentence_transformers"] = {"status": "skipped", "reason": "no_ready_documents"}
        result["openai_compatible"] = {"status": "skipped", "reason": "no_ready_documents"}
        result["warnings"].append("embedding_skipped:no_ready_documents")
        return _finalize_embedding_baseline(result)
    try:
        build = build_semantic_index(db_path, doc_ids=doc_ids or None, force=True, provider="hash")
        result["hash"] = {"status": "completed", "build": build, "status_report": semantic_index_status(db_path, provider="hash")}
    except Exception as exc:
        result["hash"] = {"status": "failed", "error": str(exc)}
        result["warnings"].append("hash_embedding_failed")
    if real_provider == "hash":
        result["sentence_transformers"] = {"status": "skipped", "reason": "real_embedding_provider_is_hash"}
        result["openai_compatible"] = {"status": "skipped", "reason": "real_embedding_provider_is_hash"}
        result["warnings"].append("real_embedding_not_enabled")
        return _finalize_embedding_baseline(result)
    if real_provider == "openai-compatible":
        result["sentence_transformers"] = {"status": "skipped", "reason": "real_embedding_provider_openai_compatible"}
        try:
            build = build_semantic_index(
                db_path,
                doc_ids=doc_ids or None,
                force=True,
                provider="openai-compatible",
                model=embedding_model,
            )
            status = semantic_index_status(db_path, provider="openai-compatible", model=build.get("model"))
            result["openai_compatible"] = {"status": "completed", "build": build, "status_report": status}
        except EmbeddingError as exc:
            result["openai_compatible"] = {"status": "skipped", "reason": str(exc)}
            result["warnings"].append("real_embedding_skipped")
        except Exception as exc:
            result["openai_compatible"] = {"status": "failed", "error": str(exc)}
            result["warnings"].append("real_embedding_failed")
        return _finalize_embedding_baseline(result)

    result["openai_compatible"] = {"status": "skipped", "reason": "real_embedding_provider_sentence_transformers"}
    if not _baseline_sentence_transformers_available():
        result["sentence_transformers"] = {
            "status": "skipped",
            "reason": "sentence_transformers_not_installed",
            "install_command": "uv sync --extra embeddings",
        }
        result["warnings"].append("real_embedding_not_enabled")
        return _finalize_embedding_baseline(result)
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
    return _finalize_embedding_baseline(result)


def _finalize_embedding_baseline(result: Dict[str, Any]) -> Dict[str, Any]:
    real_provider = str(result.get("real_embedding_provider") or "sentence-transformers")
    real_key = "openai_compatible" if real_provider == "openai-compatible" else "sentence_transformers"
    real = result.get(real_key) or {}
    hash_part = result.get("hash") or {}
    real_status = str(real.get("status") or "skipped")
    result["real_embedding_provider"] = real_provider
    result["real_embedding_status"] = real_status
    real_status_report = real.get("status_report") if isinstance(real.get("status_report"), dict) else {}
    real_build = real.get("build") if isinstance(real.get("build"), dict) else {}
    result["real_embedding_model"] = str(
        real_status_report.get("model")
        or real_build.get("model")
        or result.get("real_embedding_model")
        or ""
    )
    result["real_embedding_dim"] = int(real_status_report.get("dim") or real_build.get("dim") or 0)
    result["real_embedding_node_coverage"] = float(real_status_report.get("node_coverage") or real_build.get("node_coverage") or 0.0)
    result["real_embedding_doc_coverage"] = float(
        real_status_report.get("document_coverage") or real_build.get("document_coverage") or 0.0
    )
    if real_status == "completed":
        result["hybrid_embedding_provider"] = real_provider
        result["hybrid_embedding_model"] = result["real_embedding_model"]
        result["hybrid_embedding_ready"] = bool(real_status_report.get("ready", True))
        result["embedding_rebuild_needed"] = bool(real_status_report.get("needs_rebuild", False))
    else:
        hash_status_report = hash_part.get("status_report") if isinstance(hash_part.get("status_report"), dict) else {}
        hash_build = hash_part.get("build") if isinstance(hash_part.get("build"), dict) else {}
        result["hybrid_embedding_provider"] = "hash"
        result["hybrid_embedding_model"] = str(hash_status_report.get("model") or hash_build.get("model") or "hash-ngram-v1")
        result["hybrid_embedding_ready"] = bool(hash_status_report.get("ready") or hash_build.get("total_node_embeddings"))
        result["embedding_rebuild_needed"] = bool(hash_status_report.get("needs_rebuild", True))
        if hash_part.get("status") == "completed":
            result["warnings"].append("hybrid_uses_hash_embedding")
    result["warnings"] = _unique_strings(result.get("warnings") or [])
    return result


def _baseline_real_embedding_provider(embedding_provider: Optional[str]) -> str:
    if not embedding_provider:
        return "sentence-transformers"
    return resolve_embedding_provider(embedding_provider)


def _baseline_benchmark_modes(embedding: Dict[str, Any]) -> List[str]:
    if embedding.get("hybrid_embedding_provider") == "openai-compatible":
        return ["fts", "hash-hybrid", "bge-m3-hybrid", "tree", "auto"]
    return ["fts", "hybrid", "tree", "auto"]


def _baseline_sentence_transformers_available() -> bool:
    if "sentence_transformers" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return sentence_transformers_available()


@contextmanager
def _embedding_search_env(embedding: Dict[str, Any]):
    provider = str(embedding.get("hybrid_embedding_provider") or "hash")
    model = str(embedding.get("hybrid_embedding_model") or "")
    previous_provider = os.environ.get("KB_EMBEDDING_PROVIDER")
    previous_model = os.environ.get("KB_EMBEDDING_MODEL")
    if provider:
        os.environ["KB_EMBEDDING_PROVIDER"] = provider
    if model:
        os.environ["KB_EMBEDDING_MODEL"] = model
    try:
        yield
    finally:
        if previous_provider is None:
            os.environ.pop("KB_EMBEDDING_PROVIDER", None)
        else:
            os.environ["KB_EMBEDDING_PROVIDER"] = previous_provider
        if previous_model is None:
            os.environ.pop("KB_EMBEDDING_MODEL", None)
        else:
            os.environ["KB_EMBEDDING_MODEL"] = previous_model


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


def _tree_search_baseline(
    db_path: Path,
    documents: List[Dict[str, Any]],
    *,
    top_k: int,
    use_llm: bool,
    runtime: Optional[LLMBaselineRuntime] = None,
    llm_doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    items = []
    llm_items = []
    comparisons = []
    warnings = []
    llm_doc_set = set(llm_doc_ids or [str(item.get("doc_id") or "") for item in documents])
    for doc in documents:
        doc_id = str(doc.get("doc_id") or "")
        title = str(doc.get("title") or doc_id)
        query = f"{title} 的方法设计和实验结果是什么？"
        rule_item = _tree_search_item(db_path, doc_id, query, top_k=top_k, use_llm=False)
        items.append(rule_item)
        if rule_item.get("status") == "failed":
            warnings.append(f"tree_search_failed:{doc_id}:{rule_item.get('error', '')}")
    if use_llm:
        with (runtime.stage("llm_tree_search") if runtime else null_stage("llm_tree_search")) as stage:
            for doc in documents:
                doc_id = str(doc.get("doc_id") or "")
                if doc_id not in llm_doc_set:
                    llm_items.append({"doc_id": doc_id, "status": "skipped", "mode": "llm", "reason": "llm_max_docs"})
                    continue
                if not stage.can_continue():
                    warnings.append("llm_tree_search_skipped:baseline_llm_budget_exhausted")
                    break
                title = str(doc.get("title") or doc_id)
                query = f"{title} 的方法设计和实验结果是什么？"
                llm_item = _tree_search_item(db_path, doc_id, query, top_k=top_k, use_llm=True)
                if llm_item.get("fallback_reason"):
                    stage.mark_fallback(str(llm_item.get("fallback_reason")))
                elif not llm_item.get("llm_used"):
                    stage.mark_fallback("llm_not_used")
                llm_items.append(llm_item)
                if llm_item.get("status") == "failed":
                    warnings.append(f"llm_tree_search_failed:{doc_id}:{llm_item.get('error', '')}")
    rule_by_doc = {str(item.get("doc_id") or ""): item for item in items}
    for llm_item in llm_items:
        doc_id = str(llm_item.get("doc_id") or "")
        if llm_item.get("status") == "skipped":
            continue
        rule_item = rule_by_doc.get(doc_id) or {}
        comparisons.append(
            {
                "doc_id": doc_id,
                "rule_evidence_count": rule_item.get("evidence_count", 0),
                "llm_evidence_count": llm_item.get("evidence_count", 0),
                "evidence_delta": int(llm_item.get("evidence_count", 0) or 0) - int(rule_item.get("evidence_count", 0) or 0),
                "rule_warning_count": len(rule_item.get("warnings") or []),
                "llm_warning_count": len(llm_item.get("warnings") or []),
                "llm_used": bool(llm_item.get("llm_used")),
                "fallback_reason": llm_item.get("fallback_reason") or "",
            }
        )
    return {
        "schema": "tree_search_baseline.v1",
        "llm_enabled": use_llm,
        "doc_count": len(items),
        "items": items,
        "llm_items": llm_items,
        "comparison": comparisons,
        "comparison_summary": _tree_comparison_summary(items, llm_items),
        "warnings": _unique_strings(warnings),
    }


def _tree_search_item(db_path: Path, doc_id: str, query: str, *, top_k: int, use_llm: bool) -> Dict[str, Any]:
    try:
        trace = tree_search(db_path, doc_id, query, budget=top_k, use_llm=use_llm, require_llm=False, search_mode="hybrid")
    except Exception as exc:
        return {"doc_id": doc_id, "query": query, "status": "failed", "mode": "llm" if use_llm else "rule", "error": str(exc)}
    return {
        "doc_id": doc_id,
        "query": query,
        "mode": "llm" if use_llm else "rule",
        "status": "completed",
        "intent": (trace.get("query_profile") or {}).get("intent", ""),
        "resolved_intent": trace.get("resolved_intent") or (trace.get("query_profile") or {}).get("intent", ""),
        "expanded_node_count": len(trace.get("expanded_nodes") or []),
        "selected_path_count": len(trace.get("selected_paths") or []),
        "evidence_count": len(trace.get("evidence") or []),
        "selected_node_ids": [str(item.get("node_id") or "") for item in trace.get("selected_paths") or [] if item.get("node_id")],
        "trace_completeness": _trace_completeness(trace),
        "llm_used": bool(trace.get("llm_used")),
        "llm_selected_count": trace.get("llm_selected_count", 0),
        "llm_warning_count": trace.get("llm_warning_count", 0),
        "fallback_reason": trace.get("fallback_reason") or "",
        "warnings": trace.get("warnings") or [],
    }


def _trace_completeness(trace: Dict[str, Any]) -> float:
    checks = [
        bool(trace.get("query_profile")),
        bool(trace.get("expanded_nodes")),
        bool(trace.get("selected_paths")),
        bool(trace.get("evidence")),
    ]
    return round(sum(1 for item in checks if item) / max(1, len(checks)), 4)


def _tree_comparison_summary(items: List[Dict[str, Any]], llm_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    rule_avg = _avg(float(item.get("trace_completeness") or 0.0) for item in items)
    llm_avg = _avg(float(item.get("trace_completeness") or 0.0) for item in llm_items)
    overlaps = []
    llm_by_doc = {str(item.get("doc_id") or ""): item for item in llm_items}
    for rule_item in items:
        doc_id = str(rule_item.get("doc_id") or "")
        llm_item = llm_by_doc.get(doc_id) or {}
        rule_ids = set(str(item) for item in rule_item.get("selected_node_ids") or [] if item)
        llm_ids = set(str(item) for item in llm_item.get("selected_node_ids") or [] if item)
        denominator = len(rule_ids | llm_ids)
        overlaps.append(round(len(rule_ids & llm_ids) / denominator, 4) if denominator else 0.0)
    return {
        "schema": "tree_search_comparison_summary.v1",
        "rule_trace_completeness_avg": rule_avg,
        "llm_trace_completeness_avg": llm_avg,
        "trace_completeness_delta": round(llm_avg - rule_avg, 4) if llm_items else 0.0,
        "selected_node_overlap_avg": _avg(overlaps),
    }


def _task_baseline(
    db_path: Path,
    doc_ids: List[str],
    *,
    use_llm: bool,
    runtime: Optional[LLMBaselineRuntime] = None,
    llm_doc_ids: Optional[List[str]] = None,
    skip_llm_tasks: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "task_quality_baseline.v1",
        "use_llm": use_llm,
        "skip_llm_tasks": skip_llm_tasks,
        "compare": {},
        "review": {},
        "review_draft": {},
        "case_study": {},
        "warnings": [],
    }
    if len(doc_ids) < 2:
        result["warnings"].append("insufficient_docs_for_compare_review")
        return result
    if use_llm and skip_llm_tasks and runtime:
        for name in ("llm_compare", "llm_review", "llm_review_draft"):
            stage = runtime.stage(name)
            stage.status = "skipped"
            stage.reason = "skip_llm_tasks"
            stage.warnings.append("llm_tasks_skipped")
        result["review_draft"] = _review_draft_skip_summary("skip_llm_tasks")
    task_doc_ids = list(llm_doc_ids or doc_ids)
    if len(task_doc_ids) < 2:
        task_doc_ids = list(doc_ids[:2])
    compare_use_llm = use_llm and not skip_llm_tasks
    try:
        stage_ctx = runtime.stage("llm_compare") if runtime and use_llm and not skip_llm_tasks else null_stage("llm_compare")
        with stage_ctx as stage:
            if use_llm and skip_llm_tasks:
                result["warnings"].append("llm_tasks_skipped")
            if use_llm and not skip_llm_tasks and not stage.allowed:
                compare_use_llm = False
                result["warnings"].append(stage.reason or "llm_compare_skipped")
            elif use_llm and not skip_llm_tasks and not stage.can_continue():
                compare_use_llm = False
                result["warnings"].append("llm_compare_skipped:baseline_llm_budget_exhausted")
            compare = compare_papers(
                db_path,
                "真实论文集方法与实验评测对比",
                doc_ids=task_doc_ids,
                use_llm=compare_use_llm,
                require_llm=False,
                search_mode="tree",
            )
            if compare_use_llm and compare.get("llm_error"):
                stage.mark_fallback(str(compare.get("llm_error")))
            elif compare_use_llm and (((compare.get("comparison_matrix") or {}).get("llm_diagnostics") or {}).get("mode") == "fallback_rule"):
                stage.mark_fallback("compare_fallback_rule")
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
            "llm_diagnostics": matrix.get("llm_diagnostics") or {},
            "duplicate_evidence_removed": matrix.get("duplicate_evidence_removed", 0),
        }
    except Exception as exc:
        result["compare"] = {"status": "failed", "error": str(exc)}
        result["warnings"].append(f"compare_failed:{exc}")
    review_use_llm = use_llm and not skip_llm_tasks
    try:
        stage_ctx = runtime.stage("llm_review") if runtime and use_llm and not skip_llm_tasks else null_stage("llm_review")
        with stage_ctx as stage:
            if use_llm and not skip_llm_tasks and not stage.allowed:
                review_use_llm = False
                result["warnings"].append(stage.reason or "llm_review_skipped")
            elif use_llm and not skip_llm_tasks and not stage.can_continue():
                review_use_llm = False
                result["warnings"].append("llm_review_skipped:baseline_llm_budget_exhausted")
            review = generate_review_plan(
                db_path,
                "真实论文集任务规划方法研究综述",
                doc_ids=task_doc_ids,
                use_llm=review_use_llm,
                require_llm=False,
                search_mode="tree",
                prefer_section_llm=review_use_llm,
            )
            if review_use_llm and review.get("llm_error"):
                stage.mark_fallback(str(review.get("llm_error")))
            elif review_use_llm and (((review.get("review_outline") or {}).get("llm_diagnostics") or {}).get("mode") == "fallback_rule"):
                stage.mark_fallback("review_fallback_rule")
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
            "llm_diagnostics": outline.get("llm_diagnostics") or {},
            "duplicate_evidence_removed": outline.get("duplicate_evidence_removed", 0),
            "review_partial_reasons": outline.get("review_partial_reasons") or [],
        }
    except Exception as exc:
        result["review"] = {"status": "failed", "error": str(exc)}
        result["warnings"].append(f"review_failed:{exc}")
    if use_llm and not skip_llm_tasks:
        review_task_id = str((result.get("review") or {}).get("task_id") or "")
        if review_task_id:
            try:
                stage_ctx = runtime.stage("llm_review_draft") if runtime else null_stage("llm_review_draft")
                with stage_ctx as stage:
                    if not stage.allowed:
                        result["review_draft"] = _review_draft_skip_summary(stage.reason or "llm_review_draft_skipped")
                    elif not stage.can_continue():
                        result["review_draft"] = _review_draft_skip_summary("llm_review_draft_skipped:baseline_llm_budget_exhausted")
                    else:
                        draft = draft_review(
                            db_path,
                            review_task_id,
                            use_llm=True,
                            require_llm=False,
                            should_continue=stage.can_continue,
                            skip_reason="llm_review_draft_stage_budget_exhausted",
                            budget_fallback_to_rule=True,
                        )
                        result["review_draft"] = _review_draft_summary(draft)
                        if draft.get("llm_error"):
                            stage.mark_fallback(str(draft.get("llm_error")))
                        if result["review_draft"].get("skipped_section_count"):
                            stage.mark_warning("review_draft_sections_skipped")
                        if result["review_draft"].get("status") == "partial":
                            stage.mark_warning("review_draft_partial")
            except Exception as exc:
                result["review_draft"] = _review_draft_failed_summary(str(exc))
                result["warnings"].append(f"review_draft_failed:{exc}")
        else:
            result["review_draft"] = _review_draft_skip_summary("missing_review_task_id")
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
    claim_frame_verification: Dict[str, Any],
) -> List[str]:
    items = []
    if any(doc.get("quality_level") == "weak" for doc in documents):
        items.append("存在弱解析论文；优先使用 Docling 重建 PDF 工件，并复查章节树和表格内容。")
    if any(provider.get("status") == "skipped" for provider in parser_comparison.get("providers") or [] if provider.get("provider") in {"docling", "grobid"}):
        items.append("Docling 或 GROBID 未完成真实对比；补齐依赖/服务后重跑 quality-baseline。")
    if embedding.get("real_embedding_status") != "completed":
        items.append("真实 embedding 尚未完成；配置 sentence-transformers 或 openai-compatible/bge-m3 后重跑。")
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
    if int(claim_frame_verification.get("unsupported_frame_count") or 0) > 0:
        items.append("ClaimFrame 存在无证据支撑项；运行 verify-claim-frames 并回到 EvidenceUnit 检查来源节点。")
    if not items:
        items.append("当前基线未发现阻塞项；下一步可扩展到 10-30 篇真实论文并重跑 benchmark。")
    return _unique_strings(items)


def _llm_baseline_summary(
    status: Dict[str, Any],
    insights: Dict[str, Any],
    tree: Dict[str, Any],
    tasks: Dict[str, Any],
    graph: Dict[str, Any],
    *,
    enabled: bool,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runtime = runtime or {}
    if not enabled:
        return {
            "schema": "llm_quality_baseline.v1",
            "status": "skipped",
            "llm_status": status,
            "stage_summary": runtime.get("stage_summary") or {},
            "total_llm_duration_ms": runtime.get("total_llm_duration_ms", 0.0),
            "total_llm_call_count": runtime.get("total_llm_call_count", 0),
            "timeout_count": runtime.get("timeout_count", 0),
            "hard_timeout_count": runtime.get("hard_timeout_count", 0),
            "slow_call_count": runtime.get("slow_call_count", 0),
            "budget_exhausted": bool(runtime.get("budget_exhausted")),
            "tree_search": {"rule_doc_count": len(tree.get("items") or []), "llm_doc_count": 0, "llm_used_count": 0, "fallback_count": 0, "comparison": []},
            "insights_and_facts": {
                "doc_count": len(insights.get("items") or []),
                "llm_doc_count": 0,
                "llm_used_count": 0,
                "llm_error_count": 0,
                "noise_filtered_count": 0,
                "long_claim_trimmed_count": 0,
                "llm_facts_success_rate": 0.0,
                "llm_facts_batch_count": 0,
                "llm_facts_batch_success_count": 0,
                "llm_facts_batch_timeout_count": 0,
                "llm_facts_batch_fallback_count": 0,
            },
            "tasks": {},
            "fact_conflict_count": graph.get("conflict_count", 0),
            "warnings": [],
        }
    llm_items = tree.get("llm_items") or []
    insight_items = insights.get("items") or []
    llm_fact_items = [item for item in insight_items if isinstance(item, dict) and item.get("llm")]
    compare = tasks.get("compare") or {}
    review = tasks.get("review") or {}
    review_draft = tasks.get("review_draft") or {}
    compare_diag = compare.get("llm_diagnostics") or {}
    fact_batch_count = sum(int((item.get("llm") or {}).get("batch_count") or 0) for item in llm_fact_items)
    fact_batch_success = sum(int((item.get("llm") or {}).get("batch_success_count") or 0) for item in llm_fact_items)
    warning_tags = []
    if status.get("configured") and status.get("probe") and not status.get("reachable"):
        warning_tags.append("llm_failed")
    elif not status.get("configured"):
        warning_tags.append("llm_skipped")
    if any(item.get("fallback_reason") for item in llm_items if isinstance(item, dict)):
        warning_tags.append("llm_tree_search_fallback")
    if any((item.get("llm") or {}).get("llm_error") for item in llm_fact_items):
        warning_tags.append("llm_fact_extraction_fallback")
    if review_draft.get("status") == "partial":
        warning_tags.append("review_draft_partial")
    if review_draft.get("draft_quality_level") in {"weak", "failed"}:
        warning_tags.append(f"review_draft_{review_draft.get('draft_quality_level')}")
    if runtime.get("timeout_count"):
        warning_tags.append("llm_timeout")
    if runtime.get("budget_exhausted"):
        warning_tags.append("baseline_llm_budget_exhausted")
    return {
        "schema": "llm_quality_baseline.v1",
        "status": "completed" if status.get("reachable") or (status.get("configured") and status.get("reachable") is None) else "partial",
        "llm_status": status,
        "stage_summary": runtime.get("stage_summary") or {},
        "total_llm_duration_ms": runtime.get("total_llm_duration_ms", 0.0),
        "total_llm_call_count": runtime.get("total_llm_call_count", 0),
        "timeout_count": runtime.get("timeout_count", 0),
        "hard_timeout_count": runtime.get("hard_timeout_count", 0),
        "slow_call_count": runtime.get("slow_call_count", 0),
        "budget_exhausted": bool(runtime.get("budget_exhausted")),
        "tree_search": {
            "rule_doc_count": len(tree.get("items") or []),
            "llm_doc_count": len(llm_items),
            "llm_used_count": sum(1 for item in llm_items if isinstance(item, dict) and item.get("llm_used")),
            "fallback_count": sum(1 for item in llm_items if isinstance(item, dict) and item.get("fallback_reason")),
            "comparison": tree.get("comparison") or [],
        },
        "insights_and_facts": {
            "doc_count": len(insight_items),
            "llm_doc_count": len(llm_fact_items),
            "llm_used_count": sum(1 for item in llm_fact_items if (item.get("llm") or {}).get("llm_used")),
            "llm_error_count": sum(1 for item in llm_fact_items if (item.get("llm") or {}).get("llm_error")),
            "noise_filtered_count": sum(int((item.get("llm") or {}).get("noise_filtered_count") or 0) for item in llm_fact_items),
            "entity_noise_filtered_count": sum(int((item.get("llm") or {}).get("entity_noise_filtered_count") or 0) for item in llm_fact_items),
            "long_claim_trimmed_count": sum(int((item.get("llm") or {}).get("long_claim_trimmed_count") or 0) for item in llm_fact_items),
            "llm_facts_success_rate": round(fact_batch_success / max(1, fact_batch_count), 4),
            "llm_facts_batch_count": fact_batch_count,
            "llm_facts_batch_success_count": fact_batch_success,
            "llm_facts_batch_timeout_count": sum(int((item.get("llm") or {}).get("batch_timeout_count") or 0) for item in llm_fact_items),
            "llm_facts_batch_fallback_count": sum(int((item.get("llm") or {}).get("batch_fallback_count") or 0) for item in llm_fact_items),
        },
        "tasks": {
            "compare_status": compare.get("status") or "",
            "compare_warning_count": compare.get("warning_count", 0),
            "compare_llm_error": bool(compare.get("llm_error")),
            "compare_fallback_mode": compare_diag.get("mode", ""),
            "compare_dimension_success_count": compare_diag.get("dimension_success_count", 0),
            "compare_dimension_timeout_count": compare_diag.get("dimension_timeout_count", 0),
            "compare_fallback_dimensions": compare_diag.get("fallback_dimensions", []),
            "llm_compare_dimension_success_rate": round(
                int(compare_diag.get("dimension_success_count") or 0) / max(1, int(compare_diag.get("dimension_count") or len(COMPARE_DIMENSIONS))),
                4,
            )
            if compare_diag
            else 0.0,
            "review_status": review.get("status") or "",
            "review_warning_count": review.get("warning_count", 0),
            "review_llm_error": bool(review.get("llm_error")),
            "review_fallback_mode": (review.get("llm_diagnostics") or {}).get("mode", ""),
            "review_retry_count": (review.get("llm_diagnostics") or {}).get("retry_count", 0),
            "review_repair_used": bool((review.get("llm_diagnostics") or {}).get("repair_used")),
            "review_fallback_sections": (review.get("llm_diagnostics") or {}).get("fallback_sections", []),
            "review_partial_reasons": review.get("review_partial_reasons") or [],
            "duplicate_evidence_removed": review.get("duplicate_evidence_removed", 0),
            "review_draft_status": review_draft.get("status", ""),
            "review_draft_skip_reason": review_draft.get("review_draft_skip_reason", "") or review_draft.get("reason", ""),
            "review_draft_quality_level": review_draft.get("draft_quality_level", ""),
            "citation_coverage_score": review_draft.get("citation_coverage_score", 0.0),
            "missing_ref_count": review_draft.get("missing_ref_count", 0),
            "unsupported_paragraph_count": review_draft.get("unsupported_paragraph_count", 0),
            "optional_unused_evidence_count": review_draft.get("optional_unused_evidence_count", 0),
            "removed_paragraph_count": (review_draft.get("paragraph_support_report") or {}).get("removed_paragraph_count", 0),
            "drafted_section_count": review_draft.get("drafted_section_count", 0),
            "skipped_section_count": review_draft.get("skipped_section_count", 0),
            "review_draft_path": review_draft.get("review_draft_path", ""),
            "section_revision_actions": review_draft.get("section_revision_actions") or [],
        },
        "fact_conflict_count": graph.get("conflict_count", 0),
        "warnings": _unique_strings(warning_tags),
    }


def _baseline_warnings(
    documents: List[Dict[str, Any]],
    parser_comparison: Dict[str, Any],
    embedding: Dict[str, Any],
    benchmark: Dict[str, Any],
    tasks: Dict[str, Any],
    memory: Dict[str, Any],
    graph: Dict[str, Any],
    llm_baseline: Dict[str, Any],
    claim_frame_verification: Dict[str, Any],
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
    warnings.extend((tasks.get("review_draft") or {}).get("warnings") or [])
    warnings.extend(memory.get("warnings") or [])
    warnings.extend(graph.get("warnings") or [])
    warnings.extend(llm_baseline.get("warnings") or [])
    warnings.extend(claim_frame_verification.get("warnings") or [])
    return _unique_strings(str(item) for item in warnings)


def _fact_audit_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before_citations = int(before.get("citation_gap_count") or 0)
    after_citations = int(after.get("citation_gap_count") or 0)
    before_duplicates = int(before.get("duplicate_group_count") or 0)
    after_duplicates = int(after.get("duplicate_group_count") or 0)
    return {
        "schema": "fact_audit_delta.v1",
        "citation_gap_count_before": before_citations,
        "citation_gap_count_after": after_citations,
        "citation_gap_count_delta": after_citations - before_citations,
        "duplicate_group_count_before": before_duplicates,
        "duplicate_group_count_after": after_duplicates,
        "duplicate_group_count_delta": after_duplicates - before_duplicates,
    }


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


def _review_draft_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    report = result.get("review_report") or {}
    citation_check = result.get("citation_check") or {}
    artifacts = result.get("artifact_paths") or {}
    return {
        "schema": "review_draft_baseline_summary.v1",
        "task_id": result.get("task_id") or "",
        "status": result.get("status") or report.get("status") or "",
        "review_draft_skip_reason": "",
        "draft_quality_level": report.get("draft_quality_level") or "",
        "quality_reasons": report.get("quality_reasons") or [],
        "drafted_section_count": result.get("drafted_section_count") or report.get("drafted_section_count", 0),
        "skipped_section_count": result.get("skipped_section_count") or report.get("skipped_section_count", 0),
        "section_count": report.get("section_count", 0),
        "citation_coverage_score": report.get("citation_coverage_score") or citation_check.get("coverage_score", 0.0),
        "missing_ref_count": len(citation_check.get("missing_refs") or []),
        "unused_evidence_count": len(citation_check.get("unused_evidence") or []),
        "unsupported_paragraph_count": len(citation_check.get("unsupported_paragraphs") or []),
        "optional_unused_evidence_count": len(citation_check.get("optional_unused_evidence") or []),
        "paragraph_support_report": report.get("paragraph_support_report") or {},
        "review_draft_path": artifacts.get("review_draft", ""),
        "citation_check_path": artifacts.get("citation_check", ""),
        "review_report_path": artifacts.get("review_report", ""),
        "section_statuses": report.get("section_statuses") or [],
        "section_revision_actions": report.get("section_revision_actions") or [],
        "warning_count": len(report.get("warnings") or []),
        "warnings": report.get("warnings") or [],
        "llm_error": result.get("llm_error", ""),
    }


def _review_draft_skip_summary(reason: str) -> Dict[str, Any]:
    return {
        "schema": "review_draft_baseline_summary.v1",
        "task_id": "",
        "status": "skipped",
        "review_draft_skip_reason": reason,
        "reason": reason,
        "draft_quality_level": "",
        "quality_reasons": ["review_draft_skipped"],
            "drafted_section_count": 0,
        "skipped_section_count": 0,
        "section_count": 0,
        "citation_coverage_score": 0.0,
        "missing_ref_count": 0,
        "unused_evidence_count": 0,
        "unsupported_paragraph_count": 0,
        "optional_unused_evidence_count": 0,
        "paragraph_support_report": {},
        "review_draft_path": "",
        "citation_check_path": "",
        "review_report_path": "",
        "section_statuses": [],
        "section_revision_actions": [],
        "warning_count": 1,
        "warnings": [reason],
        "llm_error": "",
    }


def _review_draft_failed_summary(error: str) -> Dict[str, Any]:
    return {
        **_review_draft_skip_summary("review_draft_failed"),
        "status": "failed",
        "review_draft_skip_reason": "",
        "reason": "",
        "draft_quality_level": "failed",
        "quality_reasons": ["review_draft_failed"],
        "warnings": ["review_draft_failed"],
        "llm_error": error,
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



def _avg(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)
