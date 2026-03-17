"""ImportanceScorer — score memories by relevance, recency, frequency, emotional weight."""

import time
from typing import Any, Optional

import numpy as np


class ImportanceScorer:
    """Score memory importance using multiple factors.

    Combines recency, access frequency, emotional content, uniqueness,
    and explicit user ratings.
    """

    EMOTIONAL_WORDS = {
        "love", "hate", "fear", "joy", "anger", "surprise", "disgust", "trust",
        "amazing", "terrible", "wonderful", "horrible", "exciting", "scary",
        "beautiful", "ugly", "brilliant", "stupid", "incredible", "devastating",
        "thrilled", "furious", "delighted", "terrified", "proud", "ashamed",
    }

    def __init__(self, recency_weight=0.25, frequency_weight=0.2, emotional_weight=0.15,
                 length_weight=0.1, uniqueness_weight=0.15, explicit_weight=0.15):
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.emotional_weight = emotional_weight
        self.length_weight = length_weight
        self.uniqueness_weight = uniqueness_weight
        self.explicit_weight = explicit_weight

    def score(self, memory, all_memories=None):
        """Compute overall importance score for a memory (0.0 to 1.0)."""
        scores = {
            "recency": self._recency_score(memory),
            "frequency": self._frequency_score(memory),
            "emotional": self._emotional_score(memory),
            "length": self._length_score(memory),
            "uniqueness": self._uniqueness_score(memory, all_memories),
            "explicit": self._explicit_score(memory),
        }

        weighted = (
            self.recency_weight * scores["recency"]
            + self.frequency_weight * scores["frequency"]
            + self.emotional_weight * scores["emotional"]
            + self.length_weight * scores["length"]
            + self.uniqueness_weight * scores["uniqueness"]
            + self.explicit_weight * scores["explicit"]
        )

        return float(np.clip(weighted, 0.0, 1.0))

    def _recency_score(self, memory):
        """Score based on how recently the memory was created/accessed."""
        now = time.time()
        age_hours = (now - memory.accessed_at) / 3600
        # Half-life of 24 hours
        return float(np.exp(-age_hours / 24))

    def _frequency_score(self, memory):
        """Score based on how often the memory has been accessed."""
        # Logarithmic scaling
        return float(min(1.0, np.log1p(memory.access_count) / 5))

    def _emotional_score(self, memory):
        """Score based on emotional content of the memory."""
        words = set(memory.content.lower().split())
        emotional_count = len(words & self.EMOTIONAL_WORDS)
        total = max(len(words), 1)
        return float(min(1.0, emotional_count / total * 10))

    def _length_score(self, memory):
        """Score based on content length (moderate length is optimal)."""
        length = len(memory.content)
        # Bell curve: optimal at ~200 chars
        return float(np.exp(-((length - 200) / 300) ** 2))

    def _uniqueness_score(self, memory, all_memories=None):
        """Score based on how unique this memory is relative to others."""
        if not all_memories or memory.embedding is None:
            return 0.5

        similarities = []
        for other in all_memories:
            if other.id == memory.id or other.embedding is None:
                continue
            norm_m = np.linalg.norm(memory.embedding)
            norm_o = np.linalg.norm(other.embedding)
            if norm_m > 0 and norm_o > 0:
                sim = float(np.dot(memory.embedding, other.embedding) / (norm_m * norm_o))
                similarities.append(sim)

        if not similarities:
            return 0.5

        avg_sim = np.mean(similarities)
        # More unique = higher score
        return float(1.0 - avg_sim)

    def _explicit_score(self, memory):
        """Score from explicit user rating in metadata."""
        rating = memory.metadata.get("user_rating", None)
        if rating is not None:
            return float(np.clip(rating, 0, 1))
        # Check for pinned/starred
        if memory.metadata.get("pinned", False):
            return 1.0
        if memory.metadata.get("starred", False):
            return 0.8
        return 0.5

    def batch_score(self, memories):
        """Score a batch of memories, using cross-memory uniqueness."""
        scored = []
        for mem in memories:
            score = self.score(mem, all_memories=memories)
            scored.append((score, mem))
        scored.sort(key=lambda x: -x[0])
        return scored
