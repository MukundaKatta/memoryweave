"""ImportanceScorer: score by recency, frequency, emotional weight, relevance."""

import re
import time
from typing import Any, Dict, Optional

import numpy as np

from .store import Memory


# Sentiment word lists for emotional weight estimation
POSITIVE_WORDS = {
    "love", "great", "excellent", "amazing", "wonderful", "fantastic", "happy",
    "joy", "excited", "beautiful", "perfect", "brilliant", "outstanding", "success",
    "celebrate", "triumph", "delighted", "grateful", "thankful", "inspiring",
}

NEGATIVE_WORDS = {
    "hate", "terrible", "awful", "horrible", "sad", "angry", "fear", "pain",
    "disaster", "failure", "crisis", "danger", "threat", "tragedy", "death",
    "suffering", "miserable", "devastating", "catastrophe", "emergency",
}

SURPRISE_WORDS = {
    "unexpected", "surprising", "shocking", "remarkable", "unprecedented",
    "incredible", "unbelievable", "astonishing", "bizarre", "extraordinary",
}


class ImportanceScorer:
    """Score memory importance using multiple signals."""

    def __init__(
        self,
        recency_half_life_hours: float = 24.0,
        frequency_weight: float = 0.2,
        recency_weight: float = 0.3,
        emotional_weight: float = 0.2,
        relevance_weight: float = 0.3,
        base_importance_weight: float = 0.1,
    ) -> None:
        self.half_life_seconds = recency_half_life_hours * 3600.0
        self.weights = {
            "frequency": frequency_weight,
            "recency": recency_weight,
            "emotional": emotional_weight,
            "relevance": relevance_weight,
        }
        self.base_weight = base_importance_weight
        self._query_embedding: Optional[np.ndarray] = None

    def set_query_context(self, query_embedding: np.ndarray) -> None:
        """Set the current query embedding for relevance scoring."""
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            self._query_embedding = query_embedding / norm
        else:
            self._query_embedding = query_embedding

    def score(
        self,
        memory: Memory,
        current_time: Optional[float] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """Compute composite importance score in [0, 1]."""
        now = current_time or time.time()
        qe = query_embedding if query_embedding is not None else self._query_embedding

        components: Dict[str, float] = {}

        # 1. Recency score: exponential decay
        components["recency"] = self._recency_score(memory, now)

        # 2. Frequency score: logarithmic scaling of access count
        components["frequency"] = self._frequency_score(memory)

        # 3. Emotional weight: sentiment intensity
        components["emotional"] = self._emotional_score(memory)

        # 4. Relevance: cosine similarity with query
        components["relevance"] = self._relevance_score(memory, qe) if qe is not None else 0.5

        # Weighted combination
        total = sum(
            self.weights[k] * components[k]
            for k in self.weights
            if k in components
        )

        # Add base importance
        total = (1 - self.base_weight) * total + self.base_weight * memory.importance

        return float(np.clip(total, 0.0, 1.0))

    def score_detailed(
        self,
        memory: Memory,
        current_time: Optional[float] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Return detailed breakdown of importance components."""
        now = current_time or time.time()
        qe = query_embedding if query_embedding is not None else self._query_embedding

        return {
            "recency": self._recency_score(memory, now),
            "frequency": self._frequency_score(memory),
            "emotional": self._emotional_score(memory),
            "relevance": self._relevance_score(memory, qe) if qe is not None else 0.0,
            "base_importance": memory.importance,
            "final_score": self.score(memory, now, qe),
        }

    def _recency_score(self, memory: Memory, now: float) -> float:
        """Exponential decay based on time since creation.

        score = 2^(-age / half_life)
        """
        age = max(0, now - memory.created_at)
        return float(np.power(2.0, -age / self.half_life_seconds))

    def _frequency_score(self, memory: Memory) -> float:
        """Logarithmic scaling of access count.

        score = log(1 + access_count) / log(1 + max_expected_count)
        """
        max_expected = 100  # Normalization constant
        return float(np.log1p(memory.access_count) / np.log1p(max_expected))

    def _emotional_score(self, memory: Memory) -> float:
        """Estimate emotional weight from content using sentiment word lists.

        Higher emotional intensity (positive or negative) = higher importance.
        """
        words = set(re.findall(r"\b\w+\b", memory.content.lower()))

        positive_count = len(words & POSITIVE_WORDS)
        negative_count = len(words & NEGATIVE_WORDS)
        surprise_count = len(words & SURPRISE_WORDS)

        # Total emotional intensity
        total_emotional = positive_count + negative_count + surprise_count * 1.5
        total_words = max(len(words), 1)

        # Normalize: emotional word density
        density = total_emotional / total_words

        # Scale to [0, 1] with saturation at ~10% emotional density
        return float(np.clip(density / 0.10, 0.0, 1.0))

    def _relevance_score(self, memory: Memory, query_embedding: Optional[np.ndarray]) -> float:
        """Cosine similarity between memory and current query."""
        if query_embedding is None:
            return 0.0

        mem_emb = memory.embedding
        norm = np.linalg.norm(mem_emb)
        if norm > 0:
            mem_emb = mem_emb / norm

        similarity = float(np.dot(mem_emb, query_embedding))
        return float(np.clip((similarity + 1) / 2, 0.0, 1.0))  # Map [-1,1] to [0,1]

    def batch_score(
        self,
        memories: list,
        current_time: Optional[float] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> list:
        """Score multiple memories and return sorted by importance."""
        now = current_time or time.time()
        scored = [
            (mem, self.score(mem, now, query_embedding))
            for mem in memories
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
