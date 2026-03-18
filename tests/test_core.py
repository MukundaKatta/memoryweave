from src.core import MemoryWeaver
def test_init(): assert MemoryWeaver().get_stats()["ops"] == 0
def test_op(): c = MemoryWeaver(); c.store(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = MemoryWeaver(); [c.store() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = MemoryWeaver(); c.store(); c.reset(); assert c.get_stats()["ops"] == 0
