from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable, List, Optional

from .models import DocumentRecord, EvidencePacket, NodeRecord


SCHEMA_VERSION = 1


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
            char_end INTEGER
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

        CREATE INDEX IF NOT EXISTS idx_doc_nodes_doc_id ON doc_nodes(doc_id);
        CREATE INDEX IF NOT EXISTS idx_doc_nodes_parent_id ON doc_nodes(parent_id);
        CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


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
            created_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            node_path, page_start, page_end, order_index, char_start, char_end
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                evidence_type=row["type"],
                confidence=0.75,
                title=row["title"],
                path=row["path"],
            )
        )
    return packets

