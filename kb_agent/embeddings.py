from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import db
from .config import load_env_file
from .utils import compact_whitespace


EMBEDDING_PROVIDERS = {"hash", "sentence-transformers"}
DEFAULT_EMBEDDING_PROVIDER = "hash"
DEFAULT_HASH_MODEL = "hash-ngram-v1"
DEFAULT_HASH_DIM = 256
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingError(RuntimeError):
    pass


@dataclass
class EmbeddingProvider:
    name: str
    model: str
    dim: int

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = DEFAULT_HASH_DIM) -> None:
        super().__init__(name="hash", model=DEFAULT_HASH_MODEL, dim=dim)

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        tokens = _embedding_tokens(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize_vector(vector)


class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self, model: Optional[str] = None) -> None:
        model_name = model or os.environ.get("KB_EMBEDDING_MODEL") or DEFAULT_SENTENCE_TRANSFORMERS_MODEL
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise EmbeddingError(
                "sentence-transformers is not installed. Install with: uv sync --extra embeddings"
            ) from exc
        self._model = SentenceTransformer(model_name)
        dim = int(getattr(self._model, "get_sentence_embedding_dimension", lambda: 0)() or 0)
        super().__init__(name="sentence-transformers", model=model_name, dim=dim)

    def embed(self, text: str) -> List[float]:  # pragma: no cover - optional dependency
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        values = [float(item) for item in vector]
        if self.dim <= 0:
            self.dim = len(values)
        return values


def resolve_embedding_provider(provider: Optional[str] = None) -> str:
    load_env_file()
    requested = (provider or os.environ.get("KB_EMBEDDING_PROVIDER") or DEFAULT_EMBEDDING_PROVIDER).strip().lower()
    if requested not in EMBEDDING_PROVIDERS:
        choices = ", ".join(sorted(EMBEDDING_PROVIDERS))
        raise EmbeddingError(f"Unsupported embedding provider '{requested}'. Expected one of: {choices}")
    return requested


def get_embedding_provider(provider: Optional[str] = None) -> EmbeddingProvider:
    resolved = resolve_embedding_provider(provider)
    if resolved == "hash":
        return HashEmbeddingProvider()
    return SentenceTransformersProvider()


def content_hash(text: str) -> str:
    return hashlib.sha256(compact_whitespace(text).encode("utf-8")).hexdigest()


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def vector_from_json(raw: str) -> List[float]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [float(item) for item in payload]


def build_semantic_index(
    db_path: Path,
    doc_ids: Optional[List[str]] = None,
    force: bool = False,
    provider: Optional[str] = None,
) -> Dict[str, object]:
    resolved = get_embedding_provider(provider)
    conn = db.connect(db_path)
    db.init_db(conn)
    indexed_nodes = 0
    skipped_nodes = 0
    indexed_documents = 0
    skipped_documents = 0
    try:
        documents = db.get_ready_document_rows(conn, doc_ids=doc_ids)
        for row in documents:
            text = document_embedding_text(row)
            digest = content_hash(text)
            existing = db.get_existing_document_embedding(conn, row["doc_id"], resolved.name, resolved.model)
            if existing and existing["content_hash"] == digest and not force:
                skipped_documents += 1
                continue
            vector = resolved.embed(text)
            db.upsert_document_embedding(
                conn,
                doc_id=row["doc_id"],
                content_hash=digest,
                provider=resolved.name,
                model=resolved.model,
                dim=len(vector),
                vector=vector,
            )
            indexed_documents += 1

        nodes = db.get_indexable_node_rows(conn, doc_ids=doc_ids)
        for row in nodes:
            text = node_embedding_text(row)
            digest = content_hash(text)
            existing = db.get_existing_node_embedding(conn, row["node_id"], resolved.name, resolved.model)
            if existing and existing["content_hash"] == digest and not force:
                skipped_nodes += 1
                continue
            vector = resolved.embed(text)
            db.upsert_node_embedding(
                conn,
                node_id=row["node_id"],
                doc_id=row["doc_id"],
                content_hash=digest,
                provider=resolved.name,
                model=resolved.model,
                dim=len(vector),
                vector=vector,
            )
            indexed_nodes += 1
        conn.commit()
        counts = db.embedding_counts(conn, resolved.name, resolved.model)
    finally:
        conn.close()

    return {
        "schema": "semantic_index.v1",
        "provider": resolved.name,
        "model": resolved.model,
        "dim": resolved.dim,
        "requested_doc_ids": doc_ids or [],
        "indexed_documents": indexed_documents,
        "skipped_documents": skipped_documents,
        "indexed_nodes": indexed_nodes,
        "skipped_nodes": skipped_nodes,
        "total_document_embeddings": counts["document_count"],
        "total_node_embeddings": counts["node_count"],
    }


def semantic_index_status(db_path: Path, provider: Optional[str] = None) -> Dict[str, object]:
    resolved = get_embedding_provider(provider)
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        counts = db.embedding_counts(conn, resolved.name, resolved.model)
    finally:
        conn.close()
    return {
        "schema": "semantic_index_status.v1",
        "provider": resolved.name,
        "model": resolved.model,
        "dim": resolved.dim,
        "ready": counts["node_count"] > 0,
        **counts,
    }


def node_embedding_text(row) -> str:  # type: ignore[no-untyped-def]
    node_path = str(row["node_path"] or "")
    path_parts = [part.strip() for part in node_path.split(">") if part.strip()]
    local_path = " > ".join(path_parts[1:]) if len(path_parts) > 1 else node_path
    return compact_whitespace(
        "\n".join(
            [
                local_path,
                str(row["heading"] or ""),
                str(row["summary"] or ""),
                str(row["text"] or ""),
            ]
        )
    )


def document_embedding_text(row) -> str:  # type: ignore[no-untyped-def]
    keywords = row["keywords"] or ""
    try:
        parsed_keywords = json.loads(keywords)
        if isinstance(parsed_keywords, list):
            keywords = " ".join(str(item) for item in parsed_keywords)
    except json.JSONDecodeError:
        pass
    return compact_whitespace(
        "\n".join(
            [
                str(row["title"] or ""),
                str(row["abstract"] or ""),
                str(keywords or ""),
                str(row["summary"] or ""),
                str(row["path"] or ""),
            ]
        )
    )


def _embedding_tokens(text: str) -> List[str]:
    normalized = compact_whitespace(text).lower()
    tokens: List[str] = []
    for token in re.findall(r"[a-z][a-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", normalized):
        tokens.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            for size in (2, 3):
                tokens.extend(token[index : index + size] for index in range(0, len(token) - size + 1))
    return tokens[:1000]


def _normalize_vector(vector: Iterable[float]) -> List[float]:
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0:
        return values
    return [round(item / norm, 8) for item in values]
