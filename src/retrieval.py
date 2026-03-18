"""Multi-strategy retrieval: semantic, temporal, episodic, causal."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .store import Memory, MemoryStore


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    memory: Memory
    score: float
    strategy: str
    details: Dict[str, Any]


class SemanticRetriever:
    """Retrieve memories by embedding similarity (cosine)."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        results = self.store.search_by_similarity(query_embedding, top_k=top_k, threshold=threshold)
        return [
            RetrievalResult(
                memory=mem,
                score=sim,
                strategy="semantic",
                details={"cosine_similarity": sim},
            )
            for mem, sim in results
        ]


class TemporalRetriever:
    """Retrieve memories with recency decay weighting."""

    def __init__(self, store: MemoryStore, half_life_hours: float = 24.0) -> None:
        self.store = store
        self.half_life_seconds = half_life_hours * 3600.0

    def _recency_weight(self, created_at: float, now: float) -> float:
        """Exponential decay: weight = 2^(-age / half_life)."""
        age = now - created_at
        return float(np.power(2.0, -age / self.half_life_seconds))

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        recency_bias: float = 0.3,
    ) -> List[RetrievalResult]:
        """Combine semantic similarity with recency weighting."""
        now = time.time()
        raw_results = self.store.search_by_similarity(query_embedding, top_k=top_k * 3)

        scored: List[Tuple[Memory, float, float, float]] = []
        for mem, sim in raw_results:
            recency = self._recency_weight(mem.created_at, now)
            combined = (1 - recency_bias) * sim + recency_bias * recency
            scored.append((mem, combined, sim, recency))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                memory=mem,
                score=score,
                strategy="temporal",
                details={"semantic_similarity": sim, "recency_weight": rec, "combined": score},
            )
            for mem, score, sim, rec in scored[:top_k]
        ]


class EpisodicRetriever:
    """Retrieve memories grouped by session/episode."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        expand_session: bool = True,
    ) -> List[RetrievalResult]:
        """Find relevant memories and optionally include their full session context."""
        base_results = self.store.search_by_similarity(query_embedding, top_k=top_k)
        results: List[RetrievalResult] = []
        seen_ids: set = set()

        for mem, sim in base_results:
            if mem.id in seen_ids:
                continue
            seen_ids.add(mem.id)
            results.append(
                RetrievalResult(
                    memory=mem, score=sim, strategy="episodic",
                    details={"source": "direct_match", "session_id": mem.session_id},
                )
            )

            # Expand to include session context
            if expand_session and mem.session_id:
                session_memories = self.store.get_by_session(mem.session_id)
                for sm in session_memories:
                    if sm.id not in seen_ids:
                        seen_ids.add(sm.id)
                        # Session memories get a decayed score
                        time_dist = abs(sm.created_at - mem.created_at)
                        session_score = sim * np.exp(-time_dist / 3600.0)
                        results.append(
                            RetrievalResult(
                                memory=sm, score=float(session_score), strategy="episodic",
                                details={"source": "session_expansion", "session_id": sm.session_id,
                                         "time_distance_seconds": time_dist},
                            )
                        )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class CausalRetriever:
    """Retrieve memories following causal/link chains in the memory graph."""

    def __init__(self, store: MemoryStore, max_hops: int = 3) -> None:
        self.store = store
        self.max_hops = max_hops

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        decay_per_hop: float = 0.7,
    ) -> List[RetrievalResult]:
        """Find memories through link traversal with score decay per hop."""
        seed_results = self.store.search_by_similarity(query_embedding, top_k=3)
        results: Dict[str, RetrievalResult] = {}

        for mem, sim in seed_results:
            results[mem.id] = RetrievalResult(
                memory=mem, score=sim, strategy="causal",
                details={"hop": 0, "path": [mem.id]},
            )

            # BFS through links
            frontier = [(mem.id, sim, [mem.id])]
            visited = {mem.id}

            for hop in range(self.max_hops):
                next_frontier: List[Tuple[str, float, List[str]]] = []
                for current_id, current_score, path in frontier:
                    links = self.store.get_links(current_id)
                    for link in links:
                        neighbor = link["target_id"] if link["source_id"] == current_id else link["source_id"]
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        new_score = current_score * decay_per_hop * link["weight"]
                        new_path = path + [neighbor]
                        neighbor_mem = self.store.get(neighbor)
                        if neighbor_mem and new_score > 0.01:
                            results[neighbor] = RetrievalResult(
                                memory=neighbor_mem, score=new_score, strategy="causal",
                                details={"hop": hop + 1, "path": new_path, "link_type": link["relation_type"]},
                            )
                            next_frontier.append((neighbor, new_score, new_path))
                frontier = next_frontier

        sorted_results = sorted(results.values(), key=lambda r: r.score, reverse=True)
        return sorted_results[:top_k]


class MultiStrategyRetriever:
    """Combines multiple retrieval strategies with configurable weights."""

    def __init__(
        self,
        store: MemoryStore,
        semantic_weight: float = 0.4,
        temporal_weight: float = 0.2,
        episodic_weight: float = 0.2,
        causal_weight: float = 0.2,
    ) -> None:
        self.store = store
        self.weights = {
            "semantic": semantic_weight,
            "temporal": temporal_weight,
            "episodic": episodic_weight,
            "causal": causal_weight,
        }
        self.semantic = SemanticRetriever(store)
        self.temporal = TemporalRetriever(store)
        self.episodic = EpisodicRetriever(store)
        self.causal = CausalRetriever(store)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        strategies: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Execute retrieval across all (or selected) strategies and merge with RRF."""
        active = strategies or list(self.weights.keys())
        all_results: Dict[str, Dict[str, float]] = {}  # memory_id -> {strategy: score}
        memory_map: Dict[str, Memory] = {}

        retriever_map = {
            "semantic": self.semantic,
            "temporal": self.temporal,
            "episodic": self.episodic,
            "causal": self.causal,
        }

        for strategy_name in active:
            retriever = retriever_map.get(strategy_name)
            if not retriever:
                continue
            results = retriever.retrieve(query_embedding, top_k=top_k * 2)
            for r in results:
                mid = r.memory.id
                if mid not in all_results:
                    all_results[mid] = {}
                    memory_map[mid] = r.memory
                all_results[mid][strategy_name] = r.score

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        fused_scores: Dict[str, float] = {}
        for strategy_name in active:
            ranked = sorted(
                [(mid, scores.get(strategy_name, 0)) for mid, scores in all_results.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            weight = self.weights.get(strategy_name, 0.25)
            for rank, (mid, _) in enumerate(ranked):
                rrf = weight / (k + rank + 1)
                fused_scores[mid] = fused_scores.get(mid, 0) + rrf

        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)  # type: ignore
        return [
            RetrievalResult(
                memory=memory_map[mid],
                score=fused_scores[mid],
                strategy="multi",
                details={"strategy_scores": all_results.get(mid, {}), "fused_score": fused_scores[mid]},
            )
            for mid in sorted_ids[:top_k]
        ]
