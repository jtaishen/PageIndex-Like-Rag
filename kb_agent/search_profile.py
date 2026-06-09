from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import DATA_DIR
from .eval import eval_search
from .query import classify_query
from .utils import compact_whitespace, write_json


PROFILE_MODES = {"hybrid", "tree", "fts"}
PROFILE_DIR = DATA_DIR / "state" / "search_profiles"
ACTIVE_PROFILE_PATH = PROFILE_DIR / "active.json"


def tune_search(
    db_path: Path,
    queries_path: Path,
    *,
    compare_modes: Optional[List[str]] = None,
    top_k: int = 5,
    save_profile: Optional[str] = None,
) -> Dict[str, Any]:
    modes = _clean_modes(compare_modes or ["hybrid", "tree", "fts"])
    eval_report = eval_search(db_path, queries_path, search_mode=modes[0], top_k=top_k, compare_modes=modes)
    mode_rankings = _rank_modes(eval_report.get("mode_results") or {})
    intent_modes = _intent_modes(eval_report.get("mode_results") or {})
    default_mode = mode_rankings[0]["search_mode"] if mode_rankings else "hybrid"
    warnings = []
    if not mode_rankings:
        warnings.append("no_eval_modes")
    if default_mode == "hybrid" and any(item.get("fallback_rate", 0) > 0 for item in mode_rankings):
        warnings.append("hybrid_fallback_present")

    created_at = time.time()
    tuning = {
        "schema": "search_tuning.v1",
        "queries_path": str(queries_path),
        "eval_report_path": eval_report.get("path"),
        "top_k": top_k,
        "compare_modes": modes,
        "default_mode": default_mode,
        "intent_modes": intent_modes,
        "mode_rankings": mode_rankings,
        "warnings": warnings,
        "created_at": created_at,
    }
    out_dir = DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = out_dir / f"search_tuning_{int(created_at)}.json"
    write_json(tuning_path, tuning)
    tuning["path"] = str(tuning_path)

    if save_profile:
        tuning["saved_profile"] = save_search_profile(
            name=save_profile,
            default_mode=default_mode,
            intent_modes=intent_modes,
            mode_rankings=mode_rankings,
            source_eval_report=str(eval_report.get("path") or ""),
            warnings=warnings,
        )
    return tuning


def save_search_profile(
    *,
    name: str,
    default_mode: str,
    intent_modes: Dict[str, str],
    mode_rankings: List[Dict[str, Any]],
    source_eval_report: str,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_name = _profile_name(name)
    now = time.time()
    profile = {
        "schema": "search_profile.v1",
        "name": profile_name,
        "default_mode": _valid_mode(default_mode),
        "intent_modes": {
            str(intent): _valid_mode(mode)
            for intent, mode in (intent_modes or {}).items()
            if _valid_mode(mode)
        },
        "mode_scores": mode_rankings,
        "source_eval_report": source_eval_report,
        "warnings": warnings or [],
        "created_at": now,
    }
    path = PROFILE_DIR / f"{profile_name}.json"
    write_json(path, profile)
    return {**profile, "path": str(path)}


def list_search_profiles() -> Dict[str, Any]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    active = _active_pointer()
    profiles = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        if path.name == ACTIVE_PROFILE_PATH.name:
            continue
        payload = _read_json(path)
        if not payload:
            continue
        profiles.append(
            {
                "name": payload.get("name") or path.stem,
                "path": str(path),
                "default_mode": payload.get("default_mode") or "",
                "intent_modes": payload.get("intent_modes") or {},
                "created_at": payload.get("created_at"),
                "active": str(path) == str(active.get("path") or ""),
            }
        )
    return {
        "schema": "search_profile_list.v1",
        "active": active,
        "count": len(profiles),
        "profiles": profiles,
    }


def get_search_profile(name: Optional[str] = None) -> Dict[str, Any]:
    path = _profile_path(name or "active")
    payload = _read_json(path)
    if not payload:
        raise FileNotFoundError(f"Search profile not found: {name or 'active'}")
    return {**payload, "path": str(path)}


def apply_search_profile(name: str) -> Dict[str, Any]:
    profile = get_search_profile(name)
    pointer = {
        "schema": "active_search_profile.v1",
        "name": profile.get("name") or _profile_name(name),
        "path": profile.get("path") or str(_profile_path(name)),
        "applied_at": time.time(),
    }
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ACTIVE_PROFILE_PATH, pointer)
    return {"schema": "search_profile_apply.v1", "active": pointer, "profile": profile}


def resolve_auto_search_mode(db_path: Path, query: str) -> Dict[str, Any]:
    del db_path
    warnings: List[str] = []
    try:
        profile = get_search_profile("active")
    except FileNotFoundError:
        return {
            "schema": "auto_search_resolution.v1",
            "requested_search_mode": "auto",
            "resolved_search_mode": "hybrid",
            "profile_name": "",
            "intent": str(classify_query(query, use_llm=False).get("intent") or ""),
            "warnings": ["auto_profile_missing"],
        }
    profile_intent = classify_query(query, use_llm=False)
    intent = str(profile_intent.get("intent") or "")
    intent_modes = profile.get("intent_modes") or {}
    resolved = _valid_mode(str(intent_modes.get(intent) or profile.get("default_mode") or "hybrid"))
    if not resolved:
        resolved = "hybrid"
        warnings.append("invalid_profile_mode")
    return {
        "schema": "auto_search_resolution.v1",
        "requested_search_mode": "auto",
        "resolved_search_mode": resolved,
        "profile_name": profile.get("name") or "",
        "intent": intent,
        "warnings": warnings,
    }


def latest_search_tuning_reports(limit: int = 5) -> List[Dict[str, Any]]:
    out_dir = DATA_DIR / "eval"
    if not out_dir.exists():
        return []
    reports = []
    for path in sorted(out_dir.glob("search_tuning_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = _read_json(path)
        if not payload:
            continue
        reports.append(
            {
                "path": str(path),
                "schema": payload.get("schema") or "",
                "default_mode": payload.get("default_mode") or "",
                "intent_modes": payload.get("intent_modes") or {},
                "mode_rankings": payload.get("mode_rankings") or [],
                "created_at": payload.get("created_at"),
                "warnings": payload.get("warnings") or [],
            }
        )
        if len(reports) >= limit:
            break
    return reports


def _rank_modes(mode_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rankings = []
    for mode, result in mode_results.items():
        query_count = max(1, int(result.get("query_count") or 0))
        fallback_rate = float(result.get("fallback_count") or 0) / query_count
        weak_rate = float(result.get("weak_parse_quality_count") or 0) / query_count
        score = _score_metrics(
            doc_recall=float(result.get("doc_recall_at_k") or 0.0),
            node_recall=float(result.get("node_recall_at_k") or 0.0),
            precision=float(result.get("evidence_precision") or 0.0),
            mrr=float(result.get("mrr") or 0.0),
            keyword_hit=float(result.get("node_keyword_hit_rate") or 0.0),
            fallback_rate=fallback_rate,
            weak_rate=weak_rate,
        )
        rankings.append(
            {
                "search_mode": mode,
                "score": score,
                "doc_recall_at_k": result.get("doc_recall_at_k") or 0.0,
                "node_recall_at_k": result.get("node_recall_at_k") or 0.0,
                "evidence_precision": result.get("evidence_precision") or 0.0,
                "mrr": result.get("mrr") or 0.0,
                "fallback_rate": round(fallback_rate, 4),
                "weak_parse_quality_rate": round(weak_rate, 4),
            }
        )
    return sorted(rankings, key=lambda item: (-float(item["score"]), str(item["search_mode"])))


def _intent_modes(mode_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    by_intent: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for mode, result in mode_results.items():
        for item in result.get("items") or []:
            if not isinstance(item, dict):
                continue
            intent = _item_intent(item)
            if not intent:
                continue
            by_intent.setdefault(intent, {}).setdefault(mode, []).append(item)
    selected: Dict[str, str] = {}
    for intent, mode_items in by_intent.items():
        ranked = []
        for mode, items in mode_items.items():
            ranked.append({"search_mode": mode, "score": _score_eval_items(items)})
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["search_mode"])))
        if ranked:
            selected[intent] = str(ranked[0]["search_mode"])
    return selected


def _score_eval_items(items: List[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    count = len(items)
    doc_recall = sum(float(item.get("doc_recall_at_k") or 0.0) for item in items) / count
    node_recall = sum(float(item.get("node_recall_at_k") or 0.0) for item in items) / count
    precision = sum(float(item.get("evidence_precision") or 0.0) for item in items) / count
    mrr = sum(float(item.get("mrr") or 0.0) for item in items) / count
    keyword_hit = sum(1.0 if item.get("node_keyword_hit") else 0.0 for item in items) / count
    fallback_rate = sum(1.0 if item.get("fallback_used") else 0.0 for item in items) / count
    return _score_metrics(doc_recall, node_recall, precision, mrr, keyword_hit, fallback_rate, 0.0)


def _score_metrics(
    doc_recall: float,
    node_recall: float,
    precision: float,
    mrr: float,
    keyword_hit: float,
    fallback_rate: float,
    weak_rate: float,
) -> float:
    return round(
        doc_recall * 0.3
        + node_recall * 0.35
        + precision * 0.15
        + mrr * 0.15
        + keyword_hit * 0.05
        - fallback_rate * 0.05
        - weak_rate * 0.03,
        6,
    )


def _item_intent(item: Dict[str, Any]) -> str:
    intent = compact_whitespace(str(item.get("intent") or ""))
    if intent:
        return intent
    query = str(item.get("query") or "")
    if not query:
        return ""
    return str(classify_query(query, use_llm=False).get("intent") or "")


def _profile_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", compact_whitespace(name)).strip("-._")
    return cleaned[:64] or "default"


def _profile_path(name: str) -> Path:
    if name == "active":
        active = _active_pointer()
        if active.get("path"):
            return Path(str(active["path"]))
        return ACTIVE_PROFILE_PATH
    candidate = Path(name).expanduser()
    if candidate.suffix == ".json" or candidate.is_absolute():
        return candidate.resolve()
    return PROFILE_DIR / f"{_profile_name(name)}.json"


def _active_pointer() -> Dict[str, Any]:
    payload = _read_json(ACTIVE_PROFILE_PATH)
    return payload or {}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_modes(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        mode = _valid_mode(str(value))
        if mode and mode not in seen:
            seen.add(mode)
            result.append(mode)
    return result or ["hybrid"]


def _valid_mode(mode: str) -> str:
    text = compact_whitespace(mode).lower()
    return text if text in PROFILE_MODES else ""
