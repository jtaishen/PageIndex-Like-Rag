from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json

from .models import DocumentRecord, EvidencePacket, NodeRecord


SCHEMA_VERSION = 2


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            hash TEXT NOT NULL,
            title TEXT NOT NULL,
            file_type TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime REAL NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'ready',
            error TEXT NOT NULL DEFAULT '',
            authors TEXT NOT NULL DEFAULT '[]',
            year INTEGER,
            venue TEXT NOT NULL DEFAULT '',
            doi TEXT NOT NULL DEFAULT '',
            abstract TEXT NOT NULL DEFAULT '',
            keywords TEXT NOT NULL DEFAULT '[]',
            parser_name TEXT NOT NULL DEFAULT '',
            parser_version TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS doc_nodes (
            node_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            parent_id TEXT,
            type TEXT NOT NULL,
            heading TEXT NOT NULL,
            summary TEXT NOT NULL,
            text TEXT NOT NULL,
            level INTEGER NOT NULL,
            node_path TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            order_index INTEGER NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            keywords TEXT NOT NULL DEFAULT '[]',
            source_offsets TEXT NOT NULL DEFAULT '{}',
            doc_hash TEXT NOT NULL DEFAULT ''
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS doc_nodes_fts USING fts5(
            node_id UNINDEXED,
            doc_id UNINDEXED,
            heading,
            summary,
            text,
            node_path
        );

        CREATE TABLE IF NOT EXISTS memory_items (
            memory_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            type TEXT NOT NULL,
            subject_key TEXT NOT NULL,
            content TEXT NOT NULL,
            refs TEXT NOT NULL DEFAULT '',
            ttl REAL,
            importance REAL NOT NULL DEFAULT 0.5,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS query_logs (
            query_id TEXT PRIMARY KEY,
            intent TEXT NOT NULL,
            query TEXT NOT NULL,
            docs_used TEXT NOT NULL DEFAULT '',
            nodes_used TEXT NOT NULL DEFAULT '',
            latency_ms REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS document_versions (
            version_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            file_hash TEXT NOT NULL,
            parser_name TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            artifact_dir TEXT NOT NULL,
            parse_status TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS doc_cards (
            doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
            version_id TEXT NOT NULL,
            card_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_doc_nodes_doc_id ON doc_nodes(doc_id);
        CREATE INDEX IF NOT EXISTS idx_doc_nodes_parent_id ON doc_nodes(parent_id);
        CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
        CREATE INDEX IF NOT EXISTS idx_document_versions_doc_id ON document_versions(doc_id);
        """
    )
    _ensure_columns(
        conn,
        "documents",
        {
            "authors": "TEXT NOT NULL DEFAULT '[]'",
            "year": "INTEGER",
            "venue": "TEXT NOT NULL DEFAULT ''",
            "doi": "TEXT NOT NULL DEFAULT ''",
            "abstract": "TEXT NOT NULL DEFAULT ''",
            "keywords": "TEXT NOT NULL DEFAULT '[]'",
            "parser_name": "TEXT NOT NULL DEFAULT ''",
            "parser_version": "TEXT NOT NULL DEFAULT ''",
        },
    )
    _ensure_columns(
        conn,
        "doc_nodes",
        {
            "keywords": "TEXT NOT NULL DEFAULT '[]'",
            "source_offsets": "TEXT NOT NULL DEFAULT '{}'",
            "doc_hash": "TEXT NOT NULL DEFAULT ''",
        },
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def get_document_by_path(conn: sqlite3.Connection, path: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()


def delete_document_by_path(conn: sqlite3.Connection, path: str) -> None:
    row = get_document_by_path(conn, path)
    if not row:
        return
    delete_document(conn, row["doc_id"])


def delete_document(conn: sqlite3.Connection, doc_id: str) -> None:
    conn.execute("DELETE FROM doc_nodes_fts WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def upsert_document(conn: sqlite3.Connection, record: DocumentRecord) -> None:
    now = time.time()
    conn.execute(
        """
        INSERT INTO documents(
            doc_id, path, hash, title, file_type, size, mtime, summary, status, error,
            authors, year, venue, doi, abstract, keywords, parser_name, parser_version,
            created_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            path = excluded.path,
            hash = excluded.hash,
            title = excluded.title,
            file_type = excluded.file_type,
            size = excluded.size,
            mtime = excluded.mtime,
            summary = excluded.summary,
            status = excluded.status,
            error = excluded.error,
            authors = excluded.authors,
            year = excluded.year,
            venue = excluded.venue,
            doi = excluded.doi,
            abstract = excluded.abstract,
            keywords = excluded.keywords,
            parser_name = excluded.parser_name,
            parser_version = excluded.parser_version,
            updated_at = excluded.updated_at
        """,
        (
            record.doc_id,
            record.path,
            record.hash,
            record.title,
            record.file_type,
            record.size,
            record.mtime,
            record.summary,
            record.status,
            record.error,
            json.dumps(record.authors, ensure_ascii=False),
            record.year,
            record.venue,
            record.doi,
            record.abstract,
            json.dumps(record.keywords, ensure_ascii=False),
            record.parser_name,
            record.parser_version,
            now,
            now,
        ),
    )


def insert_nodes(conn: sqlite3.Connection, nodes: Iterable[NodeRecord]) -> None:
    node_rows = []
    fts_rows = []
    for node in nodes:
        node_rows.append(
            (
                node.node_id,
                node.doc_id,
                node.parent_id,
                node.kind,
                node.heading,
                node.summary,
                node.text,
                node.level,
                node.node_path,
                node.page_start,
                node.page_end,
                node.order_index,
                node.char_start,
                node.char_end,
                json.dumps(node.keywords, ensure_ascii=False),
                json.dumps(node.source_offsets, ensure_ascii=False),
                node.doc_hash,
            )
        )
        fts_rows.append(
            (
                node.node_id,
                node.doc_id,
                node.heading,
                node.summary,
                node.text,
                node.node_path,
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO doc_nodes(
            node_id, doc_id, parent_id, type, heading, summary, text, level,
            node_path, page_start, page_end, order_index, char_start, char_end,
            keywords, source_offsets, doc_hash
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        node_rows,
    )
    conn.executemany(
        """
        INSERT INTO doc_nodes_fts(node_id, doc_id, heading, summary, text, node_path)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        fts_rows,
    )


def list_documents(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents ORDER BY updated_at DESC, title ASC"
    ).fetchall()


def insert_document_version(
    conn: sqlite3.Connection,
    *,
    version_id: str,
    doc_id: str,
    file_hash: str,
    parser_name: str,
    parser_version: str,
    artifact_dir: str,
    parse_status: str,
    error: str = "",
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO document_versions(
            version_id, doc_id, file_hash, parser_name, parser_version,
            artifact_dir, parse_status, error, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version_id,
            doc_id,
            file_hash,
            parser_name,
            parser_version,
            artifact_dir,
            parse_status,
            error,
            time.time(),
        ),
    )


def upsert_doc_card(conn: sqlite3.Connection, doc_id: str, version_id: str, card: Dict[str, object]) -> None:
    now = time.time()
    conn.execute(
        """
        INSERT INTO doc_cards(doc_id, version_id, card_json, created_at, updated_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            version_id = excluded.version_id,
            card_json = excluded.card_json,
            updated_at = excluded.updated_at
        """,
        (doc_id, version_id, json.dumps(card, ensure_ascii=False), now, now),
    )


def get_doc_card(conn: sqlite3.Connection, doc_id: str) -> Optional[Dict[str, object]]:
    row = conn.execute("SELECT card_json FROM doc_cards WHERE doc_id = ?", (doc_id,)).fetchone()
    if not row:
        return None
    return json.loads(row["card_json"])


def get_document_version(conn: sqlite3.Connection, doc_id: str, version_id: Optional[str] = None) -> Optional[sqlite3.Row]:
    if version_id:
        return conn.execute(
            "SELECT * FROM document_versions WHERE doc_id = ? AND version_id = ?",
            (doc_id, version_id),
        ).fetchone()
    return conn.execute(
        """
        SELECT *
        FROM document_versions
        WHERE doc_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (doc_id,),
    ).fetchone()


def get_doc_tree_rows(conn: sqlite3.Connection, doc_id: str) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM doc_nodes
        WHERE doc_id = ?
        ORDER BY order_index ASC
        """,
        (doc_id,),
    ).fetchall()


def get_evidence_packets(
    conn: sqlite3.Connection,
    doc_id: str,
    node_ids: Iterable[str],
) -> List[EvidencePacket]:
    ids = list(node_ids)
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT n.*, d.title, d.path
        FROM doc_nodes n
        JOIN documents d ON d.doc_id = n.doc_id
        WHERE n.doc_id = ? AND n.node_id IN ({placeholders})
        ORDER BY n.order_index ASC
        """,
        [doc_id] + ids,
    ).fetchall()
    packets = []
    for row in rows:
        text = row["text"] or row["summary"] or row["heading"]
        packets.append(
            EvidencePacket(
                doc_id=row["doc_id"],
                node_id=row["node_id"],
                node_path=row["node_path"],
                page_range=(row["page_start"], row["page_end"]),
                excerpt=text,
                evidence_type=_classify_evidence_type(row["type"], row["node_path"], row["heading"]),
                confidence=0.75,
                title=row["title"],
                path=row["path"],
            )
        )
    return packets


def _classify_evidence_type(kind: str, node_path: str, heading: str) -> str:
    text = f"{node_path} {heading}".lower()
    if any(token in text for token in ("abstract", "摘要")):
        return "abstract"
    if any(token in text for token in ("method", "方法", "算法", "模型", "框架")):
        return "method"
    if any(token in text for token in ("experiment", "evaluation", "result", "实验", "评估", "结果", "消融")):
        return "result"
    if any(token in text for token in ("limitation", "discussion", "局限", "不足", "讨论")):
        return "limitation"
    if any(token in text for token in ("reference", "citation", "参考文献", "引用")):
        return "citation"
    if kind in {"document", "section", "page", "paragraph"}:
        return kind
    return "paragraph"
