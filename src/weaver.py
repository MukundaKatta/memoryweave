"""MemoryWeaver: automatically find connections between memories. Build knowledge graph
edges based on entity co-occurrence and embedding proximity."""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .store import Memory, MemoryStore


class MemoryWeaver:
    """Discovers and creates relationships between memories automatically."""

    def __init__(
        self,
        store: MemoryStore,
        similarity_threshold: float = 0.7,
        entity_weight: float = 0.4,
        embedding_weight: float = 0.6,
    ) -> None:
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.entity_weight = entity_weight
        self.embedding_weight = embedding_weight
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)

    def build_entity_index(self) -> None:
        """Build an inverted index: entity -> set of memory IDs."""
        self._entity_index.clear()
        all_ids = self.store.get_all_ids()
        for mid in all_ids:
            memory = self.store.get(mid)
            if memory and memory.entities:
                for entity in memory.entities:
                    self._entity_index[entity.lower()].add(mid)

    def extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction: capitalized multi-word phrases and proper nouns."""
        entities: List[str] = []
        # Match capitalized sequences (potential proper nouns)
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        matches = re.findall(pattern, text)
        entities.extend(matches)

        # Also extract quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)

        # Deduplicate, preserving order
        seen: Set[str] = set()
        unique: List[str] = []
        for e in entities:
            lower = e.lower()
            if lower not in seen:
                seen.add(lower)
                unique.append(e)
        return unique

    def compute_entity_similarity(self, mem_a: Memory, mem_b: Memory) -> float:
        """Jaccard similarity based on shared entities."""
        ents_a = set(e.lower() for e in mem_a.entities)
        ents_b = set(e.lower() for e in mem_b.entities)
        if not ents_a or not ents_b:
            return 0.0
        intersection = ents_a & ents_b
        union = ents_a | ents_b
        return len(intersection) / len(union) if union else 0.0

    def compute_embedding_similarity(self, mem_a: Memory, mem_b: Memory) -> float:
        """Cosine similarity between memory embeddings."""
        dot = float(np.dot(mem_a.embedding, mem_b.embedding))
        norm_a = float(np.linalg.norm(mem_a.embedding))
        norm_b = float(np.linalg.norm(mem_b.embedding))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def compute_combined_similarity(self, mem_a: Memory, mem_b: Memory) -> float:
        """Weighted combination of entity and embedding similarity."""
        entity_sim = self.compute_entity_similarity(mem_a, mem_b)
        embedding_sim = self.compute_embedding_similarity(mem_a, mem_b)
        return (self.entity_weight * entity_sim + self.embedding_weight * embedding_sim)

    def find_connections(
        self,
        memory_id: str,
        top_k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[str, float, str]]:
        """Find memories connected to the given memory.

        Returns list of (connected_memory_id, similarity_score, relation_type).
        """
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold
        memory = self.store.get(memory_id)
        if not memory:
            return []

        connections: List[Tuple[str, float, str]] = []

        # Entity-based candidates
        entity_candidates: Set[str] = set()
        for entity in memory.entities:
            for mid in self._entity_index.get(entity.lower(), set()):
                if mid != memory_id:
                    entity_candidates.add(mid)

        # Check embedding similarity for entity candidates first
        all_ids = self.store.get_all_ids()
        candidates = entity_candidates | set(all_ids[:100])  # Also check top-100 by recency

        for cid in candidates:
            if cid == memory_id:
                continue
            other = self.store.get(cid)
            if not other:
                continue

            sim = self.compute_combined_similarity(memory, other)
            if sim >= threshold:
                # Determine relation type
                relation = self._classify_relation(memory, other, sim)
                connections.append((cid, sim, relation))

        connections.sort(key=lambda x: x[1], reverse=True)
        return connections[:top_k]

    def _classify_relation(self, mem_a: Memory, mem_b: Memory, similarity: float) -> str:
        """Classify the type of relationship between two memories."""
        entity_sim = self.compute_entity_similarity(mem_a, mem_b)
        time_diff = abs(mem_a.created_at - mem_b.created_at)

        if entity_sim > 0.5:
            return "entity_overlap"
        if mem_a.session_id and mem_a.session_id == mem_b.session_id:
            return "same_session"
        if time_diff < 60:  # Within 1 minute
            return "temporal_proximity"
        if similarity > 0.9:
            return "near_duplicate"
        return "semantic_similarity"

    def weave_all(self, batch_size: int = 50) -> Dict[str, Any]:
        """Discover and create links between all memories.

        Returns statistics about discovered connections.
        """
        self.build_entity_index()
        all_ids = self.store.get_all_ids()
        total_links = 0
        relation_counts: Dict[str, int] = defaultdict(int)

        for i in range(0, len(all_ids), batch_size):
            batch = all_ids[i : i + batch_size]
            for mid in batch:
                connections = self.find_connections(mid, top_k=5)
                for target_id, sim, relation in connections:
                    self.store.add_link(mid, target_id, relation=relation, weight=sim)
                    total_links += 1
                    relation_counts[relation] += 1

        return {
            "total_memories": len(all_ids),
            "total_links_created": total_links,
            "relation_types": dict(relation_counts),
            "unique_entities": len(self._entity_index),
        }

    def get_memory_neighborhood(
        self, memory_id: str, depth: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get the neighborhood of a memory up to a given depth (BFS)."""
        visited: Set[str] = set()
        layers: Dict[str, List[Dict[str, Any]]] = {}
        current_layer = {memory_id}

        for d in range(depth):
            next_layer: Set[str] = set()
            layer_data: List[Dict[str, Any]] = []

            for mid in current_layer:
                if mid in visited:
                    continue
                visited.add(mid)
                links = self.store.get_links(mid)
                for link in links:
                    other = link["target_id"] if link["source_id"] == mid else link["source_id"]
                    if other not in visited:
                        next_layer.add(other)
                        layer_data.append({
                            "from": mid,
                            "to": other,
                            "relation": link["relation_type"],
                            "weight": link["weight"],
                        })

            layers[f"depth_{d + 1}"] = layer_data
            current_layer = next_layer

        return layers
