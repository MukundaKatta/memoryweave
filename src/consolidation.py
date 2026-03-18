"""Sleep-like consolidation: merge similar memories, compress old ones, prune below importance."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .importance import ImportanceScorer
from .store import Memory, MemoryStore


@dataclass
class ConsolidationResult:
    """Statistics from a consolidation pass."""
    memories_before: int
    memories_after: int
    merged: int
    pruned: int
    compressed: int
    duration_seconds: float


class MemoryConsolidator:
    """Consolidate memories: merge duplicates, compress old, prune unimportant."""

    def __init__(
        self,
        store: MemoryStore,
        importance_scorer: Optional[ImportanceScorer] = None,
        merge_threshold: float = 0.92,
        prune_threshold: float = 0.1,
        compression_age_hours: float = 168.0,  # 1 week
    ) -> None:
        self.store = store
        self.scorer = importance_scorer or ImportanceScorer()
        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold
        self.compression_age_seconds = compression_age_hours * 3600.0

    def consolidate(self) -> ConsolidationResult:
        """Run a full consolidation pass (like memory consolidation during sleep)."""
        start = time.time()
        memories_before = self.store.count()

        merged = self._merge_similar()
        pruned = self._prune_unimportant()
        compressed = self._compress_old()

        return ConsolidationResult(
            memories_before=memories_before,
            memories_after=self.store.count(),
            merged=merged,
            pruned=pruned,
            compressed=compressed,
            duration_seconds=time.time() - start,
        )

    def _merge_similar(self) -> int:
        """Find and merge near-duplicate memories."""
        all_ids = self.store.get_all_ids()
        merged_count = 0
        to_delete: set = set()

        for i, id_a in enumerate(all_ids):
            if id_a in to_delete:
                continue
            mem_a = self.store.get(id_a)
            if not mem_a:
                continue

            for id_b in all_ids[i + 1 :]:
                if id_b in to_delete:
                    continue
                mem_b = self.store.get(id_b)
                if not mem_b:
                    continue

                similarity = self._cosine_similarity(mem_a.embedding, mem_b.embedding)
                if similarity >= self.merge_threshold:
                    self._merge_memories(mem_a, mem_b)
                    to_delete.add(id_b)
                    merged_count += 1

        for mid in to_delete:
            self.store.delete(mid)

        return merged_count

    def _merge_memories(self, primary: Memory, secondary: Memory) -> None:
        """Merge secondary into primary, combining content and metadata."""
        # Average embeddings
        merged_embedding = (primary.embedding + secondary.embedding) / 2.0
        norm = np.linalg.norm(merged_embedding)
        if norm > 0:
            merged_embedding /= norm

        # Combine content (keep primary, add note about merge)
        merged_content = primary.content
        if secondary.content not in primary.content:
            merged_content = f"{primary.content}\n[Also: {secondary.content[:200]}]"

        # Merge entities
        merged_entities = list(set(primary.entities + secondary.entities))

        # Take higher importance
        merged_importance = max(primary.importance, secondary.importance)

        # Merge metadata
        merged_meta = {**secondary.metadata, **primary.metadata}
        merged_meta["merged_from"] = merged_meta.get("merged_from", [])
        if isinstance(merged_meta["merged_from"], list):
            merged_meta["merged_from"].append(secondary.id)

        # Update primary
        self.store.save(
            content=merged_content,
            embedding=merged_embedding,
            metadata=merged_meta,
            session_id=primary.session_id,
            importance=merged_importance,
            entities=merged_entities,
            memory_id=primary.id,
        )

    def _prune_unimportant(self) -> int:
        """Remove memories below importance threshold."""
        all_ids = self.store.get_all_ids()
        pruned = 0
        now = time.time()

        for mid in all_ids:
            memory = self.store.get(mid)
            if not memory:
                continue

            # Score importance dynamically
            score = self.scorer.score(memory, current_time=now)
            self.store.update_importance(mid, score)

            if score < self.prune_threshold:
                self.store.delete(mid)
                pruned += 1

        return pruned

    def _compress_old(self) -> int:
        """Compress old memories by reducing embedding precision and content length."""
        all_ids = self.store.get_all_ids()
        compressed = 0
        now = time.time()

        for mid in all_ids:
            memory = self.store.get(mid)
            if not memory:
                continue

            age = now - memory.created_at
            if age < self.compression_age_seconds:
                continue

            # Reduce content to summary-like form
            if len(memory.content) > 500:
                sentences = memory.content.split(". ")
                compressed_content = ". ".join(sentences[:3]) + "..."

                # Quantize embedding (reduce precision)
                quantized = np.round(memory.embedding * 100) / 100

                self.store.save(
                    content=compressed_content,
                    embedding=quantized,
                    metadata={**memory.metadata, "compressed": True, "original_length": len(memory.content)},
                    session_id=memory.session_id,
                    importance=memory.importance,
                    entities=memory.entities,
                    memory_id=memory.id,
                )
                compressed += 1

        return compressed

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_consolidation_candidates(self, top_k: int = 20) -> Dict[str, List]:
        """Preview what would happen in a consolidation pass without executing."""
        all_ids = self.store.get_all_ids()
        now = time.time()
        merge_pairs: List[Tuple[str, str, float]] = []
        prune_candidates: List[Tuple[str, float]] = []
        compress_candidates: List[str] = []

        for i, id_a in enumerate(all_ids[:100]):  # Limit scan
            mem_a = self.store.get(id_a)
            if not mem_a:
                continue

            score = self.scorer.score(mem_a, current_time=now)
            if score < self.prune_threshold:
                prune_candidates.append((id_a, score))

            age = now - mem_a.created_at
            if age >= self.compression_age_seconds and len(mem_a.content) > 500:
                compress_candidates.append(id_a)

            for id_b in all_ids[i + 1 : i + 50]:
                mem_b = self.store.get(id_b)
                if not mem_b:
                    continue
                sim = self._cosine_similarity(mem_a.embedding, mem_b.embedding)
                if sim >= self.merge_threshold:
                    merge_pairs.append((id_a, id_b, sim))

        return {
            "merge_pairs": merge_pairs[:top_k],
            "prune_candidates": prune_candidates[:top_k],
            "compress_candidates": compress_candidates[:top_k],
        }
