from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from . import db


ARTIFACT_WHITELIST = {
    "raw_text.txt",
    "body.md",
    "structured.json",
    "metadata.json",
    "references.json",
    "parse_report.json",
    "tree.json",
    "node_index.jsonl",
    "doc_card.json",
    "innovation.json",
    "citation_map.json",
}


def get_doc_card(db_path: Path, doc_id: str) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        card = db.get_doc_card(conn, doc_id)
        if card is None:
            raise KeyError(f"No doc card found for doc_id: {doc_id}")
        return card
    finally:
        conn.close()


def list_artifacts(db_path: Path, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        version = db.get_document_version(conn, doc_id, version_id)
        if version is None:
            raise KeyError(f"No document version found for doc_id: {doc_id}")
        artifact_dir = Path(version["artifact_dir"])
        artifacts = []
        for name in sorted(ARTIFACT_WHITELIST):
            path = artifact_dir / name
            artifacts.append(
                {
                    "name": name,
                    "path": str(path),
                    "exists": path.exists(),
                    "bytes": path.stat().st_size if path.exists() else 0,
                }
            )
        return {
            "doc_id": doc_id,
            "version_id": version["version_id"],
            "artifact_dir": str(artifact_dir),
            "parse_status": version["parse_status"],
            "error": version["error"],
            "artifacts": artifacts,
        }
    finally:
        conn.close()


def get_artifact(db_path: Path, doc_id: str, name: str, version_id: Optional[str] = None) -> Dict[str, Any]:
    if name not in ARTIFACT_WHITELIST:
        raise ValueError(f"Unsupported artifact name: {name}")
    listing = list_artifacts(db_path, doc_id, version_id)
    artifact_dir = Path(str(listing["artifact_dir"]))
    path = artifact_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    text = path.read_text(encoding="utf-8")
    content: Any = text
    if name.endswith(".json"):
        content = json.loads(text)
    elif name.endswith(".jsonl"):
        content = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {
        "doc_id": doc_id,
        "version_id": listing["version_id"],
        "name": name,
        "path": str(path),
        "content": content,
    }
