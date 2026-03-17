"""Storage adapters — SQLite, ChromaDB, FAISS, filesystem backends."""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class StorageAdapter(ABC):
    """Abstract storage adapter interface."""

    @abstractmethod
    def save(self, memory_id, data):
        pass

    @abstractmethod
    def load(self, memory_id):
        pass

    @abstractmethod
    def delete(self, memory_id):
        pass

    @abstractmethod
    def list_ids(self):
        pass

    @abstractmethod
    def clear(self):
        pass


class FilesystemAdapter(StorageAdapter):
    """Store memories as JSON files on disk."""

    def __init__(self, base_dir="./memory_store"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, memory_id):
        return os.path.join(self.base_dir, f"{memory_id}.json")

    def save(self, memory_id, data):
        serializable = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                serializable[k] = {"__ndarray__": True, "data": v.tolist(), "dtype": str(v.dtype)}
            else:
                serializable[k] = v
        with open(self._path(memory_id), "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    def load(self, memory_id):
        path = self._path(memory_id)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        # Restore numpy arrays
        for k, v in data.items():
            if isinstance(v, dict) and v.get("__ndarray__"):
                data[k] = np.array(v["data"], dtype=v.get("dtype", "float32"))
        return data

    def delete(self, memory_id):
        path = self._path(memory_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_ids(self):
        return [f.replace(".json", "") for f in os.listdir(self.base_dir) if f.endswith(".json")]

    def clear(self):
        for f in os.listdir(self.base_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(self.base_dir, f))


class SQLiteAdapter(StorageAdapter):
    """Store memories in SQLite database."""

    def __init__(self, db_path="./memories.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT DEFAULT 'episodic',
                importance REAL DEFAULT 0.5,
                created_at REAL,
                accessed_at REAL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()

    def _conn(self):
        import sqlite3
        return sqlite3.connect(self.db_path)

    def save(self, memory_id, data):
        conn = self._conn()
        embedding = data.get("embedding")
        emb_bytes = embedding.tobytes() if isinstance(embedding, np.ndarray) else None
        conn.execute("""
            INSERT OR REPLACE INTO memories (id, content, memory_type, importance,
                created_at, accessed_at, access_count, metadata, tags, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, data.get("content", ""), data.get("memory_type", "episodic"),
              data.get("importance", 0.5), data.get("created_at", time.time()),
              data.get("accessed_at", time.time()), data.get("access_count", 0),
              json.dumps(data.get("metadata", {})), json.dumps(data.get("tags", [])),
              emb_bytes))
        conn.commit()
        conn.close()

    def load(self, memory_id):
        conn = self._conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row[0], "content": row[1], "memory_type": row[2], "importance": row[3],
            "created_at": row[4], "accessed_at": row[5], "access_count": row[6],
            "metadata": json.loads(row[7]), "tags": json.loads(row[8]),
            "embedding": np.frombuffer(row[9], dtype=np.float32) if row[9] else None,
        }

    def delete(self, memory_id):
        conn = self._conn()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        conn.close()
        return True

    def list_ids(self):
        conn = self._conn()
        rows = conn.execute("SELECT id FROM memories").fetchall()
        conn.close()
        return [r[0] for r in rows]

    def clear(self):
        conn = self._conn()
        conn.execute("DELETE FROM memories")
        conn.commit()
        conn.close()


class ChromaDBAdapter(StorageAdapter):
    """Store memories using ChromaDB for native vector search."""

    def __init__(self, collection_name="memoryweave", persist_dir="./chroma_store"):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.persist_dir)
                self._collection = client.get_or_create_collection(self.collection_name)
            except ImportError:
                raise ImportError("chromadb is required. Install with: pip install chromadb")
        return self._collection

    def save(self, memory_id, data):
        collection = self._get_collection()
        embedding = data.get("embedding")
        emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else None
        metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in data.get("metadata", {}).items()}
        metadata["memory_type"] = data.get("memory_type", "episodic")
        metadata["importance"] = data.get("importance", 0.5)

        kwargs = {"ids": [memory_id], "documents": [data.get("content", "")], "metadatas": [metadata]}
        if emb_list:
            kwargs["embeddings"] = [emb_list]
        collection.upsert(**kwargs)

    def load(self, memory_id):
        collection = self._get_collection()
        result = collection.get(ids=[memory_id], include=["documents", "metadatas", "embeddings"])
        if not result["ids"]:
            return None
        return {
            "id": memory_id, "content": result["documents"][0] if result["documents"] else "",
            "metadata": result["metadatas"][0] if result["metadatas"] else {},
            "embedding": np.array(result["embeddings"][0]) if result.get("embeddings") and result["embeddings"][0] else None,
        }

    def delete(self, memory_id):
        self._get_collection().delete(ids=[memory_id])
        return True

    def list_ids(self):
        result = self._get_collection().get(include=[])
        return result["ids"]

    def clear(self):
        collection = self._get_collection()
        ids = self.list_ids()
        if ids:
            collection.delete(ids=ids)


class FAISSAdapter(StorageAdapter):
    """Store embeddings using FAISS for fast similarity search."""

    def __init__(self, dimension=384, index_path="./faiss_index"):
        self.dimension = dimension
        self.index_path = index_path
        self._metadata = {}  # id -> data dict
        self._id_map = {}    # int_id -> str_id
        self._reverse_map = {}  # str_id -> int_id
        self._index = None
        self._next_id = 0

    def _get_index(self):
        if self._index is None:
            try:
                import faiss
                self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine on normalized)
            except ImportError:
                raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        return self._index

    def save(self, memory_id, data):
        index = self._get_index()
        embedding = data.get("embedding")
        if embedding is not None and isinstance(embedding, np.ndarray):
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embedding = embedding.reshape(1, -1).astype(np.float32)

            if memory_id in self._reverse_map:
                int_id = self._reverse_map[memory_id]
            else:
                int_id = self._next_id
                self._next_id += 1

            self._id_map[int_id] = memory_id
            self._reverse_map[memory_id] = int_id
            index.add(embedding)

        self._metadata[memory_id] = {k: v for k, v in data.items() if k != "embedding"}

    def load(self, memory_id):
        return self._metadata.get(memory_id)

    def delete(self, memory_id):
        self._metadata.pop(memory_id, None)
        return True

    def list_ids(self):
        return list(self._metadata.keys())

    def clear(self):
        self._metadata.clear()
        self._id_map.clear()
        self._reverse_map.clear()
        self._next_id = 0
        self._index = None
