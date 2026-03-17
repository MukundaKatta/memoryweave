"""MemoryWeaver — automatically link related memories and build knowledge graph."""

import numpy as np
from typing import Optional

from memoryweave.store import MemoryStore, Memory


class MemoryWeaver:
    """Automatically discover and create links between related memories.

    Analyzes embeddings, temporal proximity, shared tags, and content
    overlap to build a knowledge graph.
    """

    def __init__(self, store, similarity_threshold=0.7, temporal_window=3600):
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.temporal_window = temporal_window  # seconds

    def weave(self, memory_id=None):
        """Find and create links for a memory (or all memories if id is None)."""
        if memory_id:
            memories = [self.store.get(memory_id)]
            if not memories[0]:
                return []
        else:
            memories = self.store.get_all()

        new_links = []
        all_memories = self.store.get_all()

        for mem in memories:
            for other in all_memories:
                if mem.id == other.id:
                    continue
                if other.id in self.store._graph.get(mem.id, set()):
                    continue  # Already linked

                similarity = self._compute_similarity(mem, other)
                if similarity >= self.similarity_threshold:
                    self.store.link(mem.id, other.id, relation="semantic")
                    new_links.append((mem.id, other.id, similarity))

        return new_links

    def _compute_similarity(self, mem_a, mem_b):
        """Compute multi-factor similarity between two memories."""
        score = 0.0

        # Semantic similarity (embedding cosine)
        if mem_a.embedding is not None and mem_b.embedding is not None:
            norm_a = np.linalg.norm(mem_a.embedding)
            norm_b = np.linalg.norm(mem_b.embedding)
            if norm_a > 0 and norm_b > 0:
                cosine = float(np.dot(mem_a.embedding, mem_b.embedding) / (norm_a * norm_b))
                score += cosine * 0.5

        # Temporal proximity
        time_diff = abs(mem_a.created_at - mem_b.created_at)
        if time_diff < self.temporal_window:
            temporal_score = 1.0 - (time_diff / self.temporal_window)
            score += temporal_score * 0.2

        # Tag overlap
        tags_a, tags_b = set(mem_a.tags), set(mem_b.tags)
        if tags_a and tags_b:
            jaccard = len(tags_a & tags_b) / len(tags_a | tags_b)
            score += jaccard * 0.2

        # Type match
        if mem_a.memory_type == mem_b.memory_type:
            score += 0.1

        return score

    def find_clusters(self, min_cluster_size=3):
        """Find clusters of related memories using connected components."""
        visited = set()
        clusters = []

        for mem_id in self.store._graph:
            if mem_id in visited:
                continue
            cluster = []
            queue = [mem_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                for linked in self.store._graph.get(current, set()):
                    if linked not in visited:
                        queue.append(linked)
            if len(cluster) >= min_cluster_size:
                cluster_memories = [self.store.get(mid) for mid in cluster if self.store.get(mid)]
                clusters.append(cluster_memories)

        return clusters

    def get_knowledge_graph(self):
        """Export the memory graph as nodes and edges."""
        nodes = []
        edges = []
        seen_edges = set()

        for mem in self.store.get_all():
            nodes.append({"id": mem.id, "content": mem.content[:100], "type": mem.memory_type,
                          "importance": mem.importance, "tags": mem.tags})

        for mem_id, linked_ids in self.store._graph.items():
            for linked_id in linked_ids:
                edge_key = tuple(sorted([mem_id, linked_id]))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({"source": mem_id, "target": linked_id})

        return {"nodes": nodes, "edges": edges}

    def suggest_tags(self, memory_id, max_tags=5):
        """Suggest tags for a memory based on linked memories."""
        linked = self.store.get_linked(memory_id, depth=2)
        tag_counts = {}
        for mem in linked:
            for tag in mem.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        current_tags = set(self.store.get(memory_id).tags) if self.store.get(memory_id) else set()
        suggestions = [(tag, count) for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])
                       if tag not in current_tags]
        return [tag for tag, _ in suggestions[:max_tags]]
