# MemoryWeave

Persistent, self-organizing memory system for AI agents.

## Features

- **MemoryStore**: Hybrid storage combining vector similarity, key-value, and graph
- **MemoryWeaver**: Automatically link related memories and build knowledge graphs
- **SmartRetrieval**: Multi-strategy retrieval (semantic, temporal, episodic, graph)
- **MemoryConsolidator**: Compress, merge, and prune memories (like sleep consolidation)
- **ImportanceScorer**: Score by relevance, recency, frequency, emotional weight
- **Storage Adapters**: SQLite, ChromaDB, FAISS, filesystem backends

## Quick Start

```python
from memoryweave import MemoryStore, MemoryWeaver, SmartRetrieval

store = MemoryStore()
store.add("Python is a programming language", tags=["python"])
store.add("FastAPI is a web framework", tags=["python", "web"])

weaver = MemoryWeaver(store)
weaver.weave()

retrieval = SmartRetrieval(store)
results = retrieval.retrieve("web development with Python")
```

## Installation

```bash
pip install -e ".[full]"
```

## Testing

```bash
pytest tests/
```

## License

© 2026 Officethree Technologies. All Rights Reserved.
