from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DEFAULT_DB_PATH, PROJECT_ROOT
from .task_evidence import compact_section_evidence
from .utils import compact_whitespace, stable_id, unique_strings, write_json


TASK_ARTIFACT_WHITELIST = {
    "manifest.json",
    "selected_papers.json",
    "comparison_matrix.json",
    "review_outline.json",
    "review_draft.md",
    "citation_check.json",
    "review_report.json",
    "open_questions.json",
    "next_actions.json",
}
TASK_ID_RE = re.compile(r"^task_[0-9a-f]{12}$")


def task_state_root(db_path: Path) -> Path:
    resolved = db_path.expanduser().resolve()
    if resolved == DEFAULT_DB_PATH.expanduser().resolve():
        return PROJECT_ROOT / ".kb_state"
    return resolved.parent / ".kb_state"


def new_task_id(task_type: str, query: str, doc_ids: List[str]) -> str:
    return stable_id("task", task_type, query, ",".join(doc_ids), time.time(), length=12)


def valid_task_artifact_name(name: str) -> bool:
    if name == "current_task.json":
        return True
    if name in TASK_ARTIFACT_WHITELIST:
        return True
    if name.startswith("section_evidence/") and name.endswith(".json"):
        parts = Path(name).parts
        return len(parts) == 2 and parts[0] == "section_evidence" and ".." not in parts
    if name.startswith("section_drafts/") and (name.endswith(".json") or name.endswith(".md")):
        parts = Path(name).parts
        return len(parts) == 2 and parts[0] == "section_drafts" and ".." not in parts
    return False


def get_task_artifact(db_path: Path, task_id: str, name: str) -> Dict[str, Any]:
    if not valid_task_artifact_name(name):
        raise ValueError(f"Unsupported task artifact name: {name}")
    if task_id != "current" and not TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"Unsupported task id: {task_id}")
    root = task_state_root(db_path)
    if task_id == "current":
        path = root / "current_task.json"
    else:
        path = root / task_id / name
    if not path.exists():
        raise FileNotFoundError(f"Task artifact not found: {path}")
    text = path.read_text(encoding="utf-8")
    content: Any = text
    if path.suffix == ".json":
        content = json.loads(text)
    return {
        "task_id": task_id,
        "name": name,
        "path": str(path),
        "content": content,
    }


def selected_papers_artifact(
    task_id: str,
    task_type: str,
    query: str,
    contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    papers = []
    for context in contexts:
        citation_map = context.get("citation_map") or {}
        innovation = context.get("innovation") or {}
        facts = context.get("facts") or {}
        claim_frame_summary = (context.get("claim_frames") or {}).get("summary") or {}
        papers.append(
            {
                "doc_id": context["doc_id"],
                "title": context["title"],
                "path": context["path"],
                "description": context.get("description") or "",
                "abstract": context.get("abstract") or "",
                "keywords": context.get("keywords") or [],
                "quality_warnings": (context.get("quality") or {}).get("quality_warnings") or [],
                "innovation_status": innovation.get("status") or "",
                "innovation_count": len(innovation.get("items") or []),
                "citation_count": len(citation_map.get("references") or []),
                "fact_available": bool(facts.get("available")),
                "claim_count": facts.get("claim_count", 0),
                "entity_count": facts.get("entity_count", 0),
                "relation_count": facts.get("relation_count", 0),
                "table_backed_fact_count": facts.get("table_backed_fact_count", 0),
                "claim_frame_available": bool(claim_frame_summary.get("available")),
                "claim_frame_count": claim_frame_summary.get("frame_count", (context.get("claim_frames") or {}).get("count", 0)),
                "verified_frame_rate": claim_frame_summary.get("verified_frame_rate", 0.0),
                "unsupported_frame_count": claim_frame_summary.get("unsupported_frame_count", 0),
                "route_score": context.get("route_score"),
                "node_matches": context.get("node_matches"),
            }
        )
    return {
        "schema": "selected_papers.v1",
        "task_id": task_id,
        "task_type": task_type,
        "query": query,
        "papers": papers,
        "paper_count": len(papers),
        "created_at": time.time(),
    }


def section_evidence_artifact(
    task_id: str,
    section_id: str,
    topic: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    compacted, compaction_report = compact_section_evidence(evidence)
    doc_ids = unique_strings(str(item.get("doc_id") or "") for item in compacted if item.get("doc_id"))
    return {
        "schema": "section_evidence.v1",
        "task_id": task_id,
        "section_id": section_id,
        "topic": topic,
        "evidence": compacted,
        "evidence_count": len(compacted),
        "source_doc_count": len(doc_ids),
        "source_doc_ids": doc_ids,
        "compaction_report": compaction_report,
        "created_at": time.time(),
    }


def open_questions_artifact(
    task_id: str,
    questions: Any,
    coverage: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    items = _string_list(questions)
    if coverage.get("missing_cells"):
        items.append("部分比较单元缺少证据，需要补充检索或人工阅读。")
    if coverage.get("missing_sections"):
        items.append("部分综述章节缺少证据，需要补充论文或重新解析。")
    if not items and warnings:
        items.append("当前任务存在质量告警，需要确认是否影响结论可信度。")
    return {
        "schema": "open_questions.v1",
        "task_id": task_id,
        "items": unique_strings(items),
        "created_at": time.time(),
    }


def next_actions_artifact(
    task_id: str,
    task_type: str,
    coverage: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    actions = []
    if warnings:
        actions.append("查看 warnings，确认是否需要重新同步或重新抽取论文工件。")
    if coverage.get("missing_cells") or coverage.get("missing_sections"):
        actions.append("针对缺证据维度补充关键词检索，必要时人工指定 doc_id。")
    if coverage.get("source_doc_count", 0) < 2 and task_type in {"compare", "review"}:
        actions.append("补充更多同主题论文后重新运行任务。")
    if task_type == "review":
        actions.append("基于 section_evidence 逐节撰写综述正文，并做引用一致性检查。")
    else:
        actions.append("基于 comparison_matrix 复核差异点，再决定是否进入综述规划。")
    return {
        "schema": "next_actions.v1",
        "task_id": task_id,
        "items": unique_strings(actions),
        "created_at": time.time(),
    }


def task_manifest(task_id: str, task_type: str, query: str, status: str, warnings: List[str]) -> Dict[str, Any]:
    return {
        "schema": "task_manifest.v1",
        "task_id": task_id,
        "task_type": task_type,
        "query": query,
        "status": status,
        "warnings": unique_strings(warnings),
        "created_at": time.time(),
    }


def write_task_artifacts(
    db_path: Path,
    task_id: str,
    *,
    manifest: Dict[str, Any],
    selected_papers: Dict[str, Any],
    open_questions: Dict[str, Any],
    next_actions: Dict[str, Any],
    comparison_matrix: Optional[Dict[str, Any]] = None,
    review_outline: Optional[Dict[str, Any]] = None,
    section_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, str]:
    root = task_state_root(db_path)
    task_dir = root / task_id
    paths = {
        "state_root": str(root),
        "task_dir": str(task_dir),
    }
    write_json(task_dir / "manifest.json", manifest)
    write_json(task_dir / "selected_papers.json", selected_papers)
    write_json(task_dir / "open_questions.json", open_questions)
    write_json(task_dir / "next_actions.json", next_actions)
    paths["manifest"] = str(task_dir / "manifest.json")
    paths["selected_papers"] = str(task_dir / "selected_papers.json")
    paths["open_questions"] = str(task_dir / "open_questions.json")
    paths["next_actions"] = str(task_dir / "next_actions.json")
    if comparison_matrix is not None:
        write_json(task_dir / "comparison_matrix.json", comparison_matrix)
        paths["comparison_matrix"] = str(task_dir / "comparison_matrix.json")
    if review_outline is not None:
        write_json(task_dir / "review_outline.json", review_outline)
        paths["review_outline"] = str(task_dir / "review_outline.json")
    if section_evidence:
        section_dir = task_dir / "section_evidence"
        for section_id, payload in section_evidence.items():
            path = section_dir / f"{section_id}.json"
            write_json(path, payload)
            paths[f"section_evidence/{section_id}.json"] = str(path)
    write_json(
        root / "current_task.json",
        {
            "schema": "current_task.v1",
            "task_id": task_id,
            "task_type": manifest["task_type"],
            "query": manifest["query"],
            "status": manifest["status"],
            "task_dir": str(task_dir),
            "updated_at": time.time(),
        },
    )
    paths["current_task"] = str(root / "current_task.json")
    return paths


def _string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = compact_whitespace(str(item)) if item is not None else ""
        if text:
            result.append(text)
    return result
