"""Tests for ImportanceScorer."""

import time
import numpy as np
import pytest
from src.store import Memory
from src.importance import ImportanceScorer


def make_memory(
    content: str = "test",
    access_count: int = 0,
    importance: float = 0.5,
    created_at: float = 0.0,
    embedding_dim: int = 16,
) -> Memory:
    return Memory(
        id="test-id",
        content=content,
        embedding=np.random.randn(embedding_dim).astype(np.float32),
        created_at=created_at or time.time(),
        accessed_at=time.time(),
        access_count=access_count,
        importance=importance,
    )


class TestImportanceScorer:
    def test_score_range(self) -> None:
        scorer = ImportanceScorer()
        mem = make_memory()
        score = scorer.score(mem)
        assert 0.0 <= score <= 1.0

    def test_recency_decay(self) -> None:
        scorer = ImportanceScorer(recency_half_life_hours=1.0)
        now = time.time()
        recent = make_memory(created_at=now)
        old = make_memory(created_at=now - 7200)  # 2 hours ago
        assert scorer.score(recent, current_time=now) > scorer.score(old, current_time=now)

    def test_frequency_increases_score(self) -> None:
        scorer = ImportanceScorer(frequency_weight=0.5, recency_weight=0.1, emotional_weight=0.1, relevance_weight=0.3)
        low_access = make_memory(access_count=0)
        high_access = make_memory(access_count=50)
        assert scorer.score(high_access) > scorer.score(low_access)

    def test_emotional_content(self) -> None:
        scorer = ImportanceScorer(emotional_weight=0.5, recency_weight=0.1, frequency_weight=0.1, relevance_weight=0.3)
        neutral = make_memory(content="The weather is mild today")
        emotional = make_memory(content="I love this amazing wonderful fantastic day!")
        assert scorer.score(emotional) > scorer.score(neutral)

    def test_relevance_with_query(self) -> None:
        scorer = ImportanceScorer()
        query = np.ones(16, dtype=np.float32)
        mem_similar = make_memory()
        mem_similar.embedding = np.ones(16, dtype=np.float32)
        mem_dissimilar = make_memory()
        mem_dissimilar.embedding = -np.ones(16, dtype=np.float32)
        s1 = scorer.score(mem_similar, query_embedding=query)
        s2 = scorer.score(mem_dissimilar, query_embedding=query)
        assert s1 > s2

    def test_detailed_breakdown(self) -> None:
        scorer = ImportanceScorer()
        mem = make_memory(content="An amazing surprise event happened")
        details = scorer.score_detailed(mem)
        assert "recency" in details
        assert "frequency" in details
        assert "emotional" in details
        assert "final_score" in details

    def test_batch_score(self) -> None:
        scorer = ImportanceScorer()
        memories = [make_memory(access_count=i) for i in range(5)]
        scored = scorer.batch_score(memories)
        assert len(scored) == 5
        # Should be sorted by importance descending
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)
