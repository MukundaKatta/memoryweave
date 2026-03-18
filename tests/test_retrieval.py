"""Tests for multi-strategy retrieval."""

import time
import numpy as np
import pytest
from src.store import MemoryStore
from src.retrieval import SemanticRetriever, TemporalRetriever, MultiStrategyRetriever


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(db_path=":memory:", embedding_dim=16)
    # Add test memories
    for i in range(5):
        v = np.zeros(16, dtype=np.float32)
        v[i % 16] = 1.0
        s.save(f"memory {i}", v, session_id=f"session_{i // 3}")
    return s


class TestSemanticRetriever:
    def test_basic_retrieval(self, store: MemoryStore) -> None:
        retriever = SemanticRetriever(store)
        query = np.zeros(16, dtype=np.float32)
        query[0] = 1.0
        results = retriever.retrieve(query, top_k=3)
        assert len(results) > 0
        assert results[0].strategy == "semantic"
        assert results[0].score > 0

    def test_top_k_limit(self, store: MemoryStore) -> None:
        retriever = SemanticRetriever(store)
        query = np.ones(16, dtype=np.float32)
        results = retriever.retrieve(query, top_k=2)
        assert len(results) <= 2


class TestTemporalRetriever:
    def test_recency_boosting(self, store: MemoryStore) -> None:
        retriever = TemporalRetriever(store)
        query = np.ones(16, dtype=np.float32) / 4
        results = retriever.retrieve(query, top_k=5, recency_bias=0.5)
        assert len(results) > 0
        assert results[0].strategy == "temporal"


class TestMultiStrategyRetriever:
    def test_multi_strategy(self, store: MemoryStore) -> None:
        retriever = MultiStrategyRetriever(store)
        query = np.zeros(16, dtype=np.float32)
        query[0] = 1.0
        results = retriever.retrieve(query, top_k=3)
        assert len(results) > 0
        assert results[0].strategy == "multi"

    def test_strategy_selection(self, store: MemoryStore) -> None:
        retriever = MultiStrategyRetriever(store)
        query = np.ones(16, dtype=np.float32)
        results = retriever.retrieve(query, top_k=3, strategies=["semantic"])
        assert len(results) > 0
