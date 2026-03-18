"""MemoryStore: save memories with embeddings, retrieve by similarity. SQLite + numpy vectors."""

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    created_at: float = 0.0
    accessed_at: float = 0.0
    access_count: int = 0
    importance: float = 0.5
    entities: List[str] = field(default_factory=list)


class MemoryStore:
    """Persistent memory store using SQLite for metadata and numpy for vectors."""

    def __init__(self, db_path: str = ":memory:", embedding_dim: int = 128) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._embeddings: Dict[str, np.ndarray] = {}
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                session_id TEXT,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.5,
                entities TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

            CREATE TABLE IF NOT EXISTS memory_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT DEFAULT 'related',
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES memories(id),
                FOREIGN KEY (target_id) REFERENCES memories(id)
            );
        """)
        self.conn.commit()
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load all memory IDs (embeddings stored in-memory dict)."""
        rows = self.conn.execute("SELECT id FROM memories").fetchall()
        for row in rows:
            mid = row["id"]
            if mid not in self._embeddings:
                self._embeddings[mid] = np.zeros(self.embedding_dim, dtype=np.float32)

    def save(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        importance: float = 0.5,
        entities: Optional[List[str]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Save a memory with its embedding vector."""
        mid = memory_id or str(uuid.uuid4())
        now = time.time()
        ents = entities or []

        self.conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, metadata, session_id, created_at, accessed_at, access_count, importance, entities)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)""",
            (mid, content, json.dumps(metadata or {}), session_id, now, now, importance, json.dumps(ents)),
        )
        self.conn.commit()

        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self._embeddings[mid] = emb
        return mid

    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID, updating access stats."""
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None

        self.conn.execute(
            "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
            (time.time(), memory_id),
        )
        self.conn.commit()

        return Memory(
            id=row["id"],
            content=row["content"],
            embedding=self._embeddings.get(memory_id, np.zeros(self.embedding_dim)),
            metadata=json.loads(row["metadata"]),
            session_id=row["session_id"],
            created_at=row["created_at"],
            accessed_at=row["accessed_at"],
            access_count=row["access_count"] + 1,
            importance=row["importance"],
            entities=json.loads(row["entities"]),
        )

    def search_by_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
        session_id: Optional[str] = None,
    ) -> List[Tuple[Memory, float]]:
        """Find memories most similar to query embedding using cosine similarity."""
        if not self._embeddings:
            return []

        query = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Filter by session if specified
        candidate_ids = list(self._embeddings.keys())
        if session_id:
            session_rows = self.conn.execute(
                "SELECT id FROM memories WHERE session_id = ?", (session_id,)
            ).fetchall()
            session_set = {r["id"] for r in session_rows}
            candidate_ids = [mid for mid in candidate_ids if mid in session_set]

        if not candidate_ids:
            return []

        # Vectorized cosine similarity
        ids = candidate_ids
        vectors = np.array([self._embeddings[mid] for mid in ids])
        similarities = vectors @ query

        # Filter and sort
        ranked = sorted(zip(ids, similarities), key=lambda x: x[1], reverse=True)
        results: List[Tuple[Memory, float]] = []
        for mid, sim in ranked[:top_k]:
            if sim < threshold:
                break
            memory = self.get(mid)
            if memory:
                results.append((memory, float(sim)))
        return results

    def search_by_content(self, query: str, top_k: int = 10) -> List[Memory]:
        """Full-text search on memory content."""
        rows = self.conn.execute(
            "SELECT id FROM memories WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
            (f"%{query}%", top_k),
        ).fetchall()
        return [self.get(r["id"]) for r in rows if self.get(r["id"]) is not None]

    def get_by_session(self, session_id: str) -> List[Memory]:
        """Get all memories from a specific session."""
        rows = self.conn.execute(
            "SELECT id FROM memories WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [m for r in rows if (m := self.get(r["id"])) is not None]

    def update_importance(self, memory_id: str, importance: float) -> None:
        """Update the importance score of a memory."""
        self.conn.execute("UPDATE memories SET importance = ? WHERE id = ?", (importance, memory_id))
        self.conn.commit()

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        self._embeddings.pop(memory_id, None)
        self.conn.execute("DELETE FROM memory_links WHERE source_id = ? OR target_id = ?", (memory_id, memory_id))
        cursor = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def add_link(self, source_id: str, target_id: str, relation: str = "related", weight: float = 1.0) -> None:
        """Create a link between two memories."""
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_links (source_id, target_id, relation_type, weight) VALUES (?, ?, ?, ?)",
            (source_id, target_id, relation, weight),
        )
        self.conn.commit()

    def get_links(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get all links from a memory."""
        rows = self.conn.execute(
            "SELECT * FROM memory_links WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Total number of memories."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return row["cnt"]

    def get_all_ids(self) -> List[str]:
        """Get all memory IDs."""
        rows = self.conn.execute("SELECT id FROM memories").fetchall()
        return [r["id"] for r in rows]

    def close(self) -> None:
        self.conn.close()
