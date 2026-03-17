"""MemoryWeave - Persistent, self-organizing memory for AI agents."""
__version__ = "0.1.0"

from memoryweave.store import MemoryStore
from memoryweave.weaver import MemoryWeaver
from memoryweave.retrieval import SmartRetrieval

__all__ = ["MemoryStore", "MemoryWeaver", "SmartRetrieval"]
