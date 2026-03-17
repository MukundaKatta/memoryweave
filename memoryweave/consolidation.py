"""MemoryConsolidator — compress, merge, and prune old memories (like sleep)."""

import time
from typing import Any, Optional

import numpy as np

from memoryweave.store import MemoryStore, Memory
from memoryweave.importance import ImportanceScorer


class MemoryConsolidator:
    """Consolidate memories: merge similar ones, compress old ones, prune unimportant ones.

    Inspired by how biological memory consolidation works during sleep.
    """

    def __init__(self, store, importance_scorer=None, merge_threshold=0.85,
                 prune_threshold=0.1, max_age_days=30):
        self.store = store
        self.scorer = importance_scorer or ImportanceScorer()
        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold
        self.max_age_seconds = max_age_days * 86400

    def consolidate(self):
        """Run full consolidation: score -> merge -> prune -> compress."""
        # 1. Re-score all memories
        self._rescore_all()

        # 2. Merge highly similar memories
        merged = self._merge_similar()

        # 3. Prune low-importance old memories
        pruned = self._prune_old()

        # 4. Compress remaining memories
        compressed = self._compress_memories()

        return {"merged": merged, "pruned": pruned, "compressed": compressed,
                "remaining": self.store.count()}

    def _rescore_all(self):
        """Recalculate importance scores for all memories."""
        for mem in self.store.get_all():
            new_score = self.scorer.score(mem)
            self.store.update(mem.id, importance=new_score)

    def _merge_similar(self):
        """Merge memories that are highly similar into single consolidated memories."""
        memories = self.store.get_all()
        merged_count = 0
        to_delete = set()

        for i, mem_a in enumerate(memories):
            if mem_a.id in to_delete:
                continue
            for j in range(i + 1, len(memories)):
                mem_b = memories[j]
                if mem_b.id in to_delete:
                    continue

                if mem_a.embedding is not None and mem_b.embedding is not None:
                    norm_a = np.linalg.norm(mem_a.embedding)
                    norm_b = np.linalg.norm(mem_b.embedding)
                    if norm_a > 0 and norm_b > 0:
                        similarity = float(np.dot(mem_a.embedding, mem_b.embedding) / (norm_a * norm_b))
                        if similarity >= self.merge_threshold:
                            # Merge: keep the more important one, combine content
                            if mem_a.importance >= mem_b.importance:
                                keeper, loser = mem_a, mem_b
                            else:
                                keeper, loser = mem_b, mem_a

                            merged_content = f"{keeper.content}\n[Merged]: {loser.content[:200]}"
                            merged_importance = max(keeper.importance, loser.importance)
                            merged_tags = list(set(keeper.tags + loser.tags))

                            self.store.update(keeper.id, content=merged_content,
                                              importance=merged_importance, tags=merged_tags)
                            to_delete.add(loser.id)
                            merged_count += 1

        for mid in to_delete:
            self.store.delete(mid)

        return merged_count

    def _prune_old(self):
        """Remove old, low-importance memories."""
        now = time.time()
        pruned = 0
        to_delete = []

        for mem in self.store.get_all():
            age = now - mem.created_at
            if age > self.max_age_seconds and mem.importance < self.prune_threshold:
                # Check if memory has been accessed recently
                recency = now - mem.accessed_at
                if recency > self.max_age_seconds * 0.5:
                    to_delete.append(mem.id)

        for mid in to_delete:
            self.store.delete(mid)
            pruned += 1

        return pruned

    def _compress_memories(self):
        """Compress verbose memories by truncating content."""
        compressed = 0
        for mem in self.store.get_all():
            if len(mem.content) > 1000 and mem.importance < 0.5:
                truncated = mem.content[:500] + "... [compressed]"
                self.store.update(mem.id, content=truncated)
                compressed += 1
        return compressed

    def decay_importance(self, decay_rate=0.01):
        """Apply time-based decay to memory importance scores."""
        now = time.time()
        for mem in self.store.get_all():
            age_days = (now - mem.accessed_at) / 86400
            decay = np.exp(-decay_rate * age_days)
            new_importance = mem.importance * decay
            # Floor: never go below 0.01
            self.store.update(mem.id, importance=max(0.01, new_importance))
