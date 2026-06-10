from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    joined = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_words(text: str, limit: int = 48) -> str:
    words = compact_whitespace(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + " ..."


def split_paragraphs(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", normalized)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_chars: int = 1800) -> List[str]:
    paragraphs = split_paragraphs(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if current and current_len + paragraph_len > max_chars:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0
        if paragraph_len > max_chars:
            for start in range(0, paragraph_len, max_chars):
                piece = paragraph[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue
        current.append(paragraph)
        current_len += paragraph_len
    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def read_text_lossy(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def unique_strings(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = compact_whitespace(str(value))
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def string_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def excerpt(text: str, max_chars: int) -> str:
    compacted = compact_whitespace(str(text or ""))
    if len(compacted) <= max_chars:
        return compacted
    return compacted[:max_chars].rstrip() + "..."


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
