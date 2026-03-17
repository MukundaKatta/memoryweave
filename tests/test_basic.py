"""Tests for MemoryWeave modules."""

import time
import pytest
import numpy as np

from memoryweave.store import MemoryStore, Memory
from memoryweave.weaver import MemoryWeaver
from memoryweave.retrieval import SmartRetrieval
from memoryweave.consolidation import MemoryConsolidator
from memoryweave.importance import ImportanceScorer
from memoryweave.adapters import FilesystemAdapter


class TestMemoryStore:
    def setup_method(self):
        self.store = MemoryStore(embedding_dim=64)

    def test_add_and_get(self):
        mem = self.store.add("Hello world", tags=["test"])
        retrieved = self.store.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "Hello world"
        assert "test" in retrieved.tags

    def test_update(self):
        mem = self.store.add("Original content")
        self.store.update(mem.id, content="Updated content", importance=0.9)
        updated = self.store.get(mem.id)
        assert updated.content == "Updated content"
        assert updated.importance == 0.9

    def test_delete(self):
        mem = self.store.add("To be deleted")
        assert self.store.delete(mem.id)
        assert self.store.get(mem.id) is None

    def test_link(self):
        m1 = self.store.add("Memory A")
        m2 = self.store.add("Memory B")
        assert self.store.link(m1.id, m2.id)
        linked = self.store.get_linked(m1.id)
        assert len(linked) == 1
        assert linked[0].id == m2.id

    def test_search_by_vector(self):
        self.store.add("Python programming language")
        self.store.add("JavaScript web development")
        self.store.add("Cooking recipes and food")
        results = self.store.search_by_vector("Python code", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_by_tag(self):
        self.store.add("Tagged memory", tags=["special"])
        self.store.add("Untagged memory")
        results = self.store.search_by_tag("special")
        assert len(results) == 1

    def test_count_and_stats(self):
        self.store.add("A", memory_type="episodic")
        self.store.add("B", memory_type="semantic")
        assert self.store.count() == 2
        stats = self.store.stats()
        assert stats["total"] == 2
        assert "episodic" in stats["types"]


class TestMemoryWeaver:
    def test_weave(self):
        store = MemoryStore(embedding_dim=64)
        store.add("Python programming")
        store.add("Python coding language")
        store.add("Cooking Italian pasta")
        weaver = MemoryWeaver(store, similarity_threshold=0.3)
        links = weaver.weave()
        assert isinstance(links, list)

    def test_knowledge_graph(self):
        store = MemoryStore(embedding_dim=64)
        store.add("A")
        store.add("B")
        weaver = MemoryWeaver(store)
        graph = weaver.get_knowledge_graph()
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 2


class TestSmartRetrieval:
    def test_auto_retrieve(self):
        store = MemoryStore(embedding_dim=64)
        store.add("Machine learning algorithms")
        store.add("Web development frameworks")
        retrieval = SmartRetrieval(store)
        results = retrieval.retrieve("machine learning", top_k=2)
        assert len(results) <= 2

    def test_context_retrieval(self):
        store = MemoryStore(embedding_dim=64)
        store.add("Important fact about AI")
        retrieval = SmartRetrieval(store)
        context = retrieval.retrieve_context("AI")
        assert isinstance(context, str)


class TestImportanceScorer:
    def test_score(self):
        scorer = ImportanceScorer()
        mem = Memory(content="An amazing and exciting discovery!", importance=0.5)
        score = scorer.score(mem)
        assert 0 <= score <= 1

    def test_emotional_content(self):
        scorer = ImportanceScorer()
        emotional = Memory(content="I love this amazing and wonderful experience!")
        neutral = Memory(content="The meeting is scheduled for Tuesday at noon")
        assert scorer._emotional_score(emotional) > scorer._emotional_score(neutral)


class TestConsolidator:
    def test_consolidate(self):
        store = MemoryStore(embedding_dim=64)
        store.add("Test memory 1")
        store.add("Test memory 2")
        consolidator = MemoryConsolidator(store)
        result = consolidator.consolidate()
        assert "merged" in result
        assert "pruned" in result
        assert "remaining" in result


class TestFilesystemAdapter:
    def test_save_load(self, tmp_path):
        adapter = FilesystemAdapter(base_dir=str(tmp_path))
        data = {"content": "test", "importance": 0.7,
                "embedding": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        adapter.save("test_id", data)
        loaded = adapter.load("test_id")
        assert loaded is not None
        assert loaded["content"] == "test"
        assert isinstance(loaded["embedding"], np.ndarray)

    def test_delete(self, tmp_path):
        adapter = FilesystemAdapter(base_dir=str(tmp_path))
        adapter.save("del_id", {"content": "delete me"})
        assert adapter.delete("del_id")
        assert adapter.load("del_id") is None

    def test_list_ids(self, tmp_path):
        adapter = FilesystemAdapter(base_dir=str(tmp_path))
        adapter.save("id1", {"content": "a"})
        adapter.save("id2", {"content": "b"})
        ids = adapter.list_ids()
        assert set(ids) == {"id1", "id2"}
