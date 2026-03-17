"""SmartRetrieval — multi-strategy retrieval: semantic, temporal, causal, episodic."""

import time
from typing import Any, Optional

import numpy as np

from memoryweave.store import MemoryStore, Memory


class SmartRetrieval:
    """Multi-strategy memory retrieval combining semantic, temporal, and graph-based approaches."""

    def __init__(self, store, recency_weight=0.2, importance_weight=0.2, semantic_weight=0.6):
        self.store = store
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.semantic_weight = semantic_weight

    def retrieve(self, query, top_k=5, strategy="auto", memory_type=None, tags=None, time_range=None):
        """Retrieve relevant memories using the specified strategy."""
        if strategy == "auto":
            return self._auto_retrieve(query, top_k, memory_type, tags, time_range)
        elif strategy == "semantic":
            return self._semantic_retrieve(query, top_k, memory_type)
        elif strategy == "temporal":
            return self._temporal_retrieve(query, top_k, time_range)
        elif strategy == "episodic":
            return self._episodic_retrieve(query, top_k)
        elif strategy == "graph":
            return self._graph_retrieve(query, top_k)
        else:
            return self._auto_retrieve(query, top_k, memory_type, tags, time_range)

    def _auto_retrieve(self, query, top_k, memory_type=None, tags=None, time_range=None):
        """Automatically combine multiple retrieval strategies."""
        candidates = {}

        # Semantic search
        semantic_results = self.store.search_by_vector(query, top_k=top_k * 2, memory_type=memory_type)
        for sim, mem in semantic_results:
            candidates[mem.id] = {"memory": mem, "semantic_score": sim,
                                   "temporal_score": 0.0, "importance_score": mem.importance}

        # Tag filtering
        if tags:
            for tag in tags:
                tag_results = self.store.search_by_tag(tag)
                for mem in tag_results:
                    if mem.id not in candidates:
                        candidates[mem.id] = {"memory": mem, "semantic_score": 0.0,
                                              "temporal_score": 0.0, "importance_score": mem.importance}
                    candidates[mem.id]["importance_score"] += 0.1  # Tag match bonus

        # Temporal scoring
        now = time.time()
        for mid, data in candidates.items():
            mem = data["memory"]
            age = now - mem.created_at
            # Exponential decay: half-life of 1 day
            data["temporal_score"] = np.exp(-age / 86400)

        # Combined scoring
        scored = []
        for mid, data in candidates.items():
            combined = (self.semantic_weight * data["semantic_score"]
                        + self.recency_weight * data["temporal_score"]
                        + self.importance_weight * data["importance_score"])
            scored.append((combined, data["memory"]))

        # Time range filter
        if time_range:
            start, end = time_range
            scored = [(s, m) for s, m in scored if start <= m.created_at <= end]

        scored.sort(key=lambda x: -x[0])
        return [(score, mem) for score, mem in scored[:top_k]]

    def _semantic_retrieve(self, query, top_k, memory_type=None):
        """Pure semantic similarity search."""
        results = self.store.search_by_vector(query, top_k=top_k, memory_type=memory_type)
        return results

    def _temporal_retrieve(self, query, top_k, time_range=None):
        """Retrieve most recent memories, optionally within a time range."""
        memories = self.store.get_all()
        if time_range:
            start, end = time_range
            memories = [m for m in memories if start <= m.created_at <= end]

        memories.sort(key=lambda m: m.created_at, reverse=True)
        return [(1.0 - i * 0.1, m) for i, m in enumerate(memories[:top_k])]

    def _episodic_retrieve(self, query, top_k):
        """Retrieve episodic memories (events, experiences)."""
        return self.store.search_by_vector(query, top_k=top_k, memory_type="episodic")

    def _graph_retrieve(self, query, top_k):
        """Retrieve by finding semantically similar memories then expanding via graph."""
        seed_results = self.store.search_by_vector(query, top_k=2)
        if not seed_results:
            return []

        expanded = set()
        results = []
        for sim, mem in seed_results:
            if mem.id not in expanded:
                expanded.add(mem.id)
                results.append((sim, mem))
            # Expand via links
            linked = self.store.get_linked(mem.id, depth=2)
            for linked_mem in linked:
                if linked_mem.id not in expanded:
                    expanded.add(linked_mem.id)
                    # Score linked memories lower
                    link_score = sim * 0.7
                    results.append((link_score, linked_mem))

        results.sort(key=lambda x: -x[0])
        return results[:top_k]

    def retrieve_context(self, query, max_tokens=2000):
        """Retrieve memories and format as a context string."""
        results = self.retrieve(query, top_k=10, strategy="auto")
        lines = []
        estimated_tokens = 0

        for score, mem in results:
            line = f"[{mem.memory_type}] {mem.content}"
            tokens = len(line.split()) * 1.3
            if estimated_tokens + tokens > max_tokens:
                break
            lines.append(line)
            estimated_tokens += tokens

        return "\n".join(lines)
