from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import db
from .config import load_env_file
from .utils import compact_whitespace


EMBEDDING_PROVIDERS = {"hash", "sentence-transformers", "openai-compatible"}
DEFAULT_EMBEDDING_PROVIDER = "hash"
DEFAULT_HASH_MODEL = "hash-ngram-v1"
DEFAULT_HASH_DIM = 256
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OPENAI_COMPATIBLE_EMBEDDING_MODEL = "bge-m3"
DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 45


class EmbeddingError(RuntimeError):
    pass


@dataclass
class EmbeddingProvider:
    name: str
    model: str
    dim: int

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        return [self.embed(text) for text in texts]


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

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:  # pragma: no cover - optional dependency
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=max(1, batch_size))
        values = [[float(item) for item in vector] for vector in vectors]
        if self.dim <= 0 and values:
            self.dim = len(values[0])
        return values


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: Optional[str] = None) -> None:
        config = _openai_compatible_embedding_config(model=model)
        model_name = str(config["model"])
        base_url = str(config["base_url"])
        api_key = str(config["api_key"])
        if not base_url:
            raise EmbeddingError("KB_EMBEDDING_BASE_URL is not configured.")
        if not api_key:
            raise EmbeddingError("KB_EMBEDDING_API_KEY is not configured.")
        self._endpoint = f"{base_url.rstrip('/')}/embeddings"
        self._api_key = api_key
        self._timeout_seconds = int(config["timeout_seconds"])
        super().__init__(name="openai-compatible", model=model_name, dim=0)

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text], batch_size=1)[0]

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        del batch_size
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": texts}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:  # nosec B310
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = _read_error_detail(exc)
            message = f"openai-compatible embeddings request failed: HTTP {exc.code}"
            if detail:
                message = f"{message}: {detail}"
            raise EmbeddingError(_redact_embedding_secret(message)) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EmbeddingError(
                _redact_embedding_secret(f"openai-compatible embeddings request failed: {exc}")
            ) from exc

        try:
            response_payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingError("openai-compatible embeddings response is not valid JSON.") from exc
        vectors = _parse_openai_embedding_vectors(response_payload, expected_count=len(texts))
        if self.dim <= 0 and vectors:
            self.dim = len(vectors[0])
        return vectors


def resolve_embedding_provider(provider: Optional[str] = None) -> str:
    load_env_file()
    requested = (provider or os.environ.get("KB_EMBEDDING_PROVIDER") or DEFAULT_EMBEDDING_PROVIDER).strip().lower()
    if requested not in EMBEDDING_PROVIDERS:
        choices = ", ".join(sorted(EMBEDDING_PROVIDERS))
        raise EmbeddingError(f"Unsupported embedding provider '{requested}'. Expected one of: {choices}")
    return requested


def resolve_embedding_model(provider: str, model: Optional[str] = None) -> str:
    load_env_file()
    if provider == "hash":
        return DEFAULT_HASH_MODEL
    if provider == "openai-compatible":
        return model or os.environ.get("KB_EMBEDDING_MODEL") or DEFAULT_OPENAI_COMPATIBLE_EMBEDDING_MODEL
    return model or os.environ.get("KB_EMBEDDING_MODEL") or DEFAULT_SENTENCE_TRANSFORMERS_MODEL


def get_embedding_provider(provider: Optional[str] = None, model: Optional[str] = None) -> EmbeddingProvider:
    resolved = resolve_embedding_provider(provider)
    if resolved == "hash":
        return HashEmbeddingProvider()
    if resolved == "openai-compatible":
        return OpenAICompatibleEmbeddingProvider(resolve_embedding_model(resolved, model))
    return SentenceTransformersProvider(resolve_embedding_model(resolved, model))


def _openai_compatible_embedding_config(model: Optional[str] = None) -> Dict[str, object]:
    load_env_file()
    base_url = (os.environ.get("KB_EMBEDDING_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "").rstrip("/")
    api_key = os.environ.get("KB_EMBEDDING_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    model_name = model or os.environ.get("KB_EMBEDDING_MODEL") or DEFAULT_OPENAI_COMPATIBLE_EMBEDDING_MODEL
    return {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "timeout_seconds": _env_int("KB_EMBEDDING_TIMEOUT_SECONDS", DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS),
        "configured": bool(base_url and api_key),
        "base_url_configured": bool(base_url),
        "api_key_configured": bool(api_key),
        "base_url_source": _first_env_source(["KB_EMBEDDING_BASE_URL", "DEEPSEEK_BASE_URL"]),
        "api_key_source": _first_env_source(["KB_EMBEDDING_API_KEY", "DEEPSEEK_API_KEY"]),
    }


def openai_compatible_embedding_status(model: Optional[str] = None) -> Dict[str, object]:
    config = _openai_compatible_embedding_config(model=model)
    missing = []
    if not config["base_url_configured"]:
        missing.append("KB_EMBEDDING_BASE_URL")
    if not config["api_key_configured"]:
        missing.append("KB_EMBEDDING_API_KEY")
    return {
        "configured": bool(config["configured"]),
        "base_url_configured": bool(config["base_url_configured"]),
        "api_key_configured": bool(config["api_key_configured"]),
        "base_url_source": config["base_url_source"],
        "api_key_source": config["api_key_source"],
        "timeout_seconds": config["timeout_seconds"],
        "missing_config": missing,
    }


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
    model: Optional[str] = None,
    batch_size: int = 16,
) -> Dict[str, object]:
    resolved = get_embedding_provider(provider, model)
    conn = db.connect(db_path)
    db.init_db(conn)
    indexed_nodes = 0
    skipped_nodes = 0
    indexed_documents = 0
    skipped_documents = 0
    try:
        documents = db.get_ready_document_rows(conn, doc_ids=doc_ids)
        document_jobs = []
        for row in documents:
            text = document_embedding_text(row)
            digest = content_hash(text)
            existing = db.get_existing_document_embedding(conn, row["doc_id"], resolved.name, resolved.model)
            if existing and existing["content_hash"] == digest and not force:
                skipped_documents += 1
                continue
            document_jobs.append((row, text, digest))
        for batch in _batched(document_jobs, batch_size):
            vectors = _embed_batch(resolved, [text for _row, text, _digest in batch], batch_size=batch_size)
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"Embedding provider returned {len(vectors)} vectors for {len(batch)} document inputs."
                )
            for (row, _text, digest), vector in zip(batch, vectors):
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
        node_jobs = []
        for row in nodes:
            text = node_embedding_text(row)
            digest = content_hash(text)
            existing = db.get_existing_node_embedding(conn, row["node_id"], resolved.name, resolved.model)
            if existing and existing["content_hash"] == digest and not force:
                skipped_nodes += 1
                continue
            node_jobs.append((row, text, digest))
        for batch in _batched(node_jobs, batch_size):
            vectors = _embed_batch(resolved, [text for _row, text, _digest in batch], batch_size=batch_size)
            if len(vectors) != len(batch):
                raise EmbeddingError(f"Embedding provider returned {len(vectors)} vectors for {len(batch)} node inputs.")
            for (row, _text, digest), vector in zip(batch, vectors):
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
        document_total = len(documents)
        node_total = len(nodes)
    finally:
        conn.close()

    return {
        "schema": "semantic_index.v1",
        "provider": resolved.name,
        "model": resolved.model,
        "dim": resolved.dim,
        "batch_size": max(1, batch_size),
        "requested_doc_ids": doc_ids or [],
        "indexed_documents": indexed_documents,
        "skipped_documents": skipped_documents,
        "indexed_nodes": indexed_nodes,
        "skipped_nodes": skipped_nodes,
        "total_document_embeddings": counts["document_count"],
        "total_node_embeddings": counts["node_count"],
        "document_total": document_total,
        "node_total": node_total,
        "document_coverage": _coverage(counts["document_count"], document_total),
        "node_coverage": _coverage(counts["node_count"], node_total),
        "needs_rebuild": counts["document_count"] < document_total or counts["node_count"] < node_total,
    }


def semantic_index_status(db_path: Path, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, object]:
    provider_name = resolve_embedding_provider(provider)
    model_name = resolve_embedding_model(provider_name, model)
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        counts = db.embedding_counts(conn, provider_name, model_name)
        document_total = len(db.get_ready_document_rows(conn))
        node_total = len(db.get_indexable_node_rows(conn))
        dim_row = conn.execute(
            """
            SELECT MAX(dim) AS dim
            FROM (
              SELECT dim FROM node_embeddings WHERE provider = ? AND model = ?
              UNION ALL
              SELECT dim FROM document_embeddings WHERE provider = ? AND model = ?
            )
            """,
            (provider_name, model_name, provider_name, model_name),
        ).fetchone()
    finally:
        conn.close()
    stored_dim = int((dim_row["dim"] if dim_row else 0) or 0)
    dim = stored_dim or (DEFAULT_HASH_DIM if provider_name == "hash" else 0)
    document_coverage = _coverage(counts["document_count"], document_total)
    node_coverage = _coverage(counts["node_count"], node_total)
    needs_rebuild = counts["document_count"] < document_total or counts["node_count"] < node_total
    if provider_name == "sentence-transformers":
        package_available = sentence_transformers_available()
        provider_status: Dict[str, object] = {"configured": package_available}
        install_command = "" if package_available else "uv sync --extra embeddings"
    elif provider_name == "openai-compatible":
        package_available = True
        provider_status = openai_compatible_embedding_status(model=model_name)
        install_command = ""
    else:
        package_available = True
        provider_status = {"configured": True}
        install_command = ""
    return {
        "schema": "semantic_index_status.v1",
        "provider": provider_name,
        "model": model_name,
        "dim": dim,
        "ready": counts["node_count"] > 0,
        "package_available": package_available,
        "install_command": install_command,
        "document_total": document_total,
        "node_total": node_total,
        "missing_document_embeddings": max(0, document_total - counts["document_count"]),
        "missing_node_embeddings": max(0, node_total - counts["node_count"]),
        "document_coverage": document_coverage,
        "node_coverage": node_coverage,
        "needs_rebuild": needs_rebuild,
        **provider_status,
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


def sentence_transformers_available() -> bool:
    if "sentence_transformers" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


def _parse_openai_embedding_vectors(payload: Any, *, expected_count: int) -> List[List[float]]:
    if not isinstance(payload, dict):
        raise EmbeddingError("openai-compatible embeddings response must be a JSON object.")
    data = payload.get("data")
    if not isinstance(data, list):
        raise EmbeddingError("openai-compatible embeddings response missing data list.")
    if len(data) != expected_count:
        raise EmbeddingError(
            f"openai-compatible embeddings response returned {len(data)} vectors for {expected_count} inputs."
        )
    use_indexes = all(isinstance(item, dict) and "index" in item for item in data)
    indexed: Dict[int, List[float]] = {}
    ordered: List[List[float]] = []
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise EmbeddingError("openai-compatible embeddings data item must be an object.")
        vector = _coerce_embedding_vector(item.get("embedding"), position)
        if use_indexes:
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("openai-compatible embeddings data item has invalid index.") from exc
            indexed[index] = vector
        else:
            ordered.append(vector)
    if use_indexes:
        expected_indexes = set(range(expected_count))
        if set(indexed) != expected_indexes:
            raise EmbeddingError("openai-compatible embeddings response indexes do not match inputs.")
        ordered = [indexed[index] for index in range(expected_count)]
    return ordered


def _coerce_embedding_vector(raw: Any, position: int) -> List[float]:
    if not isinstance(raw, list) or not raw:
        raise EmbeddingError(f"openai-compatible embeddings data item {position} has an empty embedding.")
    try:
        return _normalize_vector(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(f"openai-compatible embeddings data item {position} has non-numeric values.") from exc


def _read_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:
        return ""
    try:
        return raw.decode("utf-8", errors="replace")[:500]
    except Exception:
        return ""


def _redact_embedding_secret(text: str) -> str:
    load_env_file()
    redacted = text
    for key in ("KB_EMBEDDING_API_KEY", "DEEPSEEK_API_KEY"):
        secret = os.environ.get(key)
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _first_env_source(names: List[str]) -> str:
    for name in names:
        if os.environ.get(name):
            return name
    return ""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _coverage(count: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return round(float(count) / float(total), 4)


def _normalize_vector(vector: Iterable[float]) -> List[float]:
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0:
        return values
    return [round(item / norm, 8) for item in values]


def _batched(values: List[object], batch_size: int) -> Iterable[List[object]]:
    size = max(1, int(batch_size or 1))
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _embed_batch(provider: EmbeddingProvider, texts: List[str], batch_size: int) -> List[List[float]]:
    embed_batch = getattr(provider, "embed_batch", None)
    if callable(embed_batch):
        return embed_batch(texts, batch_size=batch_size)
    return [provider.embed(text) for text in texts]
