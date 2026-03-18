"""Tests for MemoryStore."""

import numpy as np
import pytest
from src.store import MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:", embedding_dim=32)


def random_embedding(dim: int = 32) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestMemoryStore:
    def test_save_and_get(self, store: MemoryStore) -> None:
        emb = random_embedding()
        mid = store.save("Hello world", emb, metadata={"key": "value"}, entities=["World"])
        memory = store.get(mid)
        assert memory is not None
        assert memory.content == "Hello world"
        assert memory.metadata["key"] == "value"
        assert "World" in memory.entities
        assert memory.access_count == 1

    def test_search_by_similarity(self, store: MemoryStore) -> None:
        emb1 = np.ones(32, dtype=np.float32)
        emb2 = -np.ones(32, dtype=np.float32)
        emb_query = np.ones(32, dtype=np.float32) * 0.9
        store.save("similar", emb1)
        store.save("dissimilar", emb2)
        results = store.search_by_similarity(emb_query, top_k=2)
        assert len(results) == 2
        assert results[0][0].content == "similar"
        assert results[0][1] > results[1][1]

    def test_search_by_content(self, store: MemoryStore) -> None:
        store.save("The quick brown fox", random_embedding())
        store.save("Lazy dog", random_embedding())
        results = store.search_by_content("fox")
        assert len(results) == 1
        assert "fox" in results[0].content

    def test_delete(self, store: MemoryStore) -> None:
        mid = store.save("to delete", random_embedding())
        assert store.count() == 1
        assert store.delete(mid)
        assert store.count() == 0
        assert store.get(mid) is None

    def test_session_filtering(self, store: MemoryStore) -> None:
        emb = random_embedding()
        store.save("session A", emb, session_id="s1")
        store.save("session B", random_embedding(), session_id="s2")
        results = store.get_by_session("s1")
        assert len(results) == 1
        assert results[0].content == "session A"

    def test_links(self, store: MemoryStore) -> None:
        id1 = store.save("mem1", random_embedding())
        id2 = store.save("mem2", random_embedding())
        store.add_link(id1, id2, relation="related", weight=0.9)
        links = store.get_links(id1)
        assert len(links) >= 1
        assert links[0]["weight"] == 0.9

    def test_update_importance(self, store: MemoryStore) -> None:
        mid = store.save("test", random_embedding(), importance=0.5)
        store.update_importance(mid, 0.9)
        mem = store.get(mid)
        assert mem is not None
        assert mem.importance == 0.9

    def test_count(self, store: MemoryStore) -> None:
        assert store.count() == 0
        store.save("a", random_embedding())
        store.save("b", random_embedding())
        assert store.count() == 2
