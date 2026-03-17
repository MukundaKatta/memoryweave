"""MemoryStore — hybrid storage combining vector DB, key-value, and graph."""

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np


@dataclass
class Memory:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: str = ""
    embedding: Optional[np.ndarray] = None
    memory_type: str = "episodic"     # episodic, semantic, procedural
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    links: list = field(default_factory=list)   # IDs of linked memories


class MemoryStore:
    """Hybrid memory storage: vector similarity + key-value + graph links.

    Stores memories with embeddings for semantic search, metadata for
    key-value lookup, and bidirectional links for graph traversal.
    """

    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self._memories = {}             # id -> Memory
        self._embeddings = {}           # id -> np.ndarray
        self._tags_index = {}           # tag -> set of ids
        self._type_index = {}           # type -> set of ids
        self._graph = {}               # id -> set of linked ids

    def _compute_embedding(self, text):
        """Compute a simple embedding. Replace with sentence-transformers for production."""
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, "_model"):
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._model.encode(text)
        except ImportError:
            # Fallback: hash-based pseudo-embedding
            np.random.seed(int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**31))
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def add(self, content, memory_type="episodic", importance=0.5, metadata=None, tags=None):
        """Add a memory to the store."""
        embedding = self._compute_embedding(content)
        memory = Memory(content=content, embedding=embedding, memory_type=memory_type,
                        importance=importance, metadata=metadata or {}, tags=tags or [])

        self._memories[memory.id] = memory
        self._embeddings[memory.id] = embedding

        # Update indices
        for tag in memory.tags:
            self._tags_index.setdefault(tag, set()).add(memory.id)
        self._type_index.setdefault(memory_type, set()).add(memory.id)
        self._graph.setdefault(memory.id, set())

        return memory

    def get(self, memory_id):
        """Retrieve a memory by ID."""
        memory = self._memories.get(memory_id)
        if memory:
            memory.accessed_at = time.time()
            memory.access_count += 1
        return memory

    def update(self, memory_id, content=None, importance=None, metadata=None, tags=None):
        """Update an existing memory."""
        memory = self._memories.get(memory_id)
        if not memory:
            return None
        if content is not None:
            memory.content = content
            memory.embedding = self._compute_embedding(content)
            self._embeddings[memory_id] = memory.embedding
        if importance is not None:
            memory.importance = importance
        if metadata is not None:
            memory.metadata.update(metadata)
        if tags is not None:
            for old_tag in memory.tags:
                self._tags_index.get(old_tag, set()).discard(memory_id)
            memory.tags = tags
            for tag in tags:
                self._tags_index.setdefault(tag, set()).add(memory_id)
        return memory

    def delete(self, memory_id):
        """Delete a memory and clean up indices."""
        memory = self._memories.pop(memory_id, None)
        if not memory:
            return False
        self._embeddings.pop(memory_id, None)
        for tag in memory.tags:
            self._tags_index.get(tag, set()).discard(memory_id)
        self._type_index.get(memory.memory_type, set()).discard(memory_id)
        # Remove graph links
        for linked_id in self._graph.pop(memory_id, set()):
            self._graph.get(linked_id, set()).discard(memory_id)
        return True

    def link(self, id_a, id_b, relation="related"):
        """Create a bidirectional link between two memories."""
        if id_a not in self._memories or id_b not in self._memories:
            return False
        self._graph.setdefault(id_a, set()).add(id_b)
        self._graph.setdefault(id_b, set()).add(id_a)
        self._memories[id_a].links.append(id_b)
        self._memories[id_b].links.append(id_a)
        return True

    def unlink(self, id_a, id_b):
        """Remove link between two memories."""
        self._graph.get(id_a, set()).discard(id_b)
        self._graph.get(id_b, set()).discard(id_a)

    def get_linked(self, memory_id, depth=1):
        """Get memories linked to a given memory, up to depth hops."""
        visited = set()
        current_level = {memory_id}
        results = []
        for _ in range(depth):
            next_level = set()
            for mid in current_level:
                if mid in visited:
                    continue
                visited.add(mid)
                for linked in self._graph.get(mid, set()):
                    if linked not in visited:
                        next_level.add(linked)
                        mem = self._memories.get(linked)
                        if mem:
                            results.append(mem)
            current_level = next_level
        return results

    def search_by_vector(self, query_text, top_k=5, memory_type=None):
        """Semantic search using cosine similarity."""
        query_emb = self._compute_embedding(query_text)
        candidates = self._embeddings
        if memory_type and memory_type in self._type_index:
            candidates = {mid: self._embeddings[mid] for mid in self._type_index[memory_type]
                          if mid in self._embeddings}

        scores = []
        for mid, emb in candidates.items():
            norm_q = np.linalg.norm(query_emb)
            norm_e = np.linalg.norm(emb)
            if norm_q == 0 or norm_e == 0:
                sim = 0.0
            else:
                sim = float(np.dot(query_emb, emb) / (norm_q * norm_e))
            scores.append((sim, mid))

        scores.sort(key=lambda x: -x[0])
        results = []
        for sim, mid in scores[:top_k]:
            mem = self.get(mid)
            if mem:
                results.append((sim, mem))
        return results

    def search_by_tag(self, tag):
        """Find memories by tag."""
        ids = self._tags_index.get(tag, set())
        return [self._memories[mid] for mid in ids if mid in self._memories]

    def search_by_type(self, memory_type):
        """Find memories by type."""
        ids = self._type_index.get(memory_type, set())
        return [self._memories[mid] for mid in ids if mid in self._memories]

    def get_all(self):
        return list(self._memories.values())

    def count(self):
        return len(self._memories)

    def stats(self):
        types = {}
        for m in self._memories.values():
            types[m.memory_type] = types.get(m.memory_type, 0) + 1
        return {"total": len(self._memories), "types": types,
                "tags": len(self._tags_index), "links": sum(len(v) for v in self._graph.values()) // 2}
