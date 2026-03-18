"""Integration tests for Memoryweave."""
from src.core import Memoryweave

class TestMemoryweave:
    def setup_method(self):
        self.c = Memoryweave()
    def test_10_ops(self):
        for i in range(10): self.c.store(i=i)
        assert self.c.get_stats()["ops"] == 10
    def test_service_name(self):
        assert self.c.store()["service"] == "memoryweave"
    def test_different_inputs(self):
        self.c.store(type="a"); self.c.store(type="b")
        assert self.c.get_stats()["ops"] == 2
    def test_config(self):
        c = Memoryweave(config={"debug": True})
        assert c.config["debug"] is True
    def test_empty_call(self):
        assert self.c.store()["ok"] is True
    def test_large_batch(self):
        for _ in range(100): self.c.store()
        assert self.c.get_stats()["ops"] == 100
