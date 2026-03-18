"""FastAPI endpoints for MemoryWeave."""

import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .consolidation import MemoryConsolidator
from .graph import MemoryGraph
from .importance import ImportanceScorer
from .retrieval import MultiStrategyRetriever
from .store import MemoryStore
from .weaver import MemoryWeaver

app = FastAPI(title="MemoryWeave", version="0.1.0")

_store = MemoryStore(embedding_dim=128)
_scorer = ImportanceScorer()
_weaver = MemoryWeaver(_store)
_retriever = MultiStrategyRetriever(_store)
_graph = MemoryGraph(_store)
_consolidator = MemoryConsolidator(_store, _scorer)


class SaveMemoryRequest(BaseModel):
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = {}
    session_id: Optional[str] = None
    importance: float = 0.5
    entities: List[str] = []


class SearchRequest(BaseModel):
    query_embedding: List[float]
    top_k: int = 10
    strategies: Optional[List[str]] = None


class WeaveRequest(BaseModel):
    batch_size: int = 50


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "memoryweave"}


@app.post("/memories")
async def save_memory(request: SaveMemoryRequest) -> Dict[str, Any]:
    """Save a new memory."""
    embedding = np.array(request.embedding, dtype=np.float32)
    mid = _store.save(
        content=request.content, embedding=embedding, metadata=request.metadata,
        session_id=request.session_id, importance=request.importance, entities=request.entities,
    )
    return {"memory_id": mid, "status": "saved"}


@app.get("/memories/{memory_id}")
async def get_memory(memory_id: str) -> Dict[str, Any]:
    """Retrieve a memory by ID."""
    memory = _store.get(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "id": memory.id, "content": memory.content, "metadata": memory.metadata,
        "session_id": memory.session_id, "importance": memory.importance,
        "access_count": memory.access_count, "entities": memory.entities,
        "created_at": memory.created_at,
    }


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str) -> Dict[str, bool]:
    return {"deleted": _store.delete(memory_id)}


@app.post("/search")
async def search_memories(request: SearchRequest) -> Dict[str, Any]:
    """Multi-strategy memory retrieval."""
    query = np.array(request.query_embedding, dtype=np.float32)
    results = _retriever.retrieve(query, top_k=request.top_k, strategies=request.strategies)
    return {
        "count": len(results),
        "results": [
            {"memory_id": r.memory.id, "content": r.memory.content, "score": r.score,
             "strategy": r.strategy, "details": r.details}
            for r in results
        ],
    }


@app.post("/weave")
async def weave_connections(request: WeaveRequest) -> Dict[str, Any]:
    """Discover and create connections between memories."""
    return _weaver.weave_all(batch_size=request.batch_size)


@app.post("/consolidate")
async def consolidate_memories() -> Dict[str, Any]:
    """Run memory consolidation pass."""
    result = _consolidator.consolidate()
    return {
        "memories_before": result.memories_before, "memories_after": result.memories_after,
        "merged": result.merged, "pruned": result.pruned, "compressed": result.compressed,
        "duration_seconds": result.duration_seconds,
    }


@app.get("/graph/stats")
async def graph_stats() -> Dict[str, Any]:
    """Get memory graph statistics."""
    _graph.build_from_store()
    return _graph.get_stats()


@app.get("/graph/path/{source}/{target}")
async def find_path(source: str, target: str) -> Dict[str, Any]:
    """Find shortest path between two memories."""
    _graph.build_from_store()
    path = _graph.shortest_path(source, target)
    if path is None:
        raise HTTPException(status_code=404, detail="No path found")
    return {"path": path, "length": len(path) - 1}


@app.get("/memories/{memory_id}/importance")
async def get_importance(memory_id: str) -> Dict[str, Any]:
    """Get detailed importance breakdown for a memory."""
    memory = _store.get(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _scorer.score_detailed(memory)


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    return {"total_memories": _store.count(), "embedding_dim": _store.embedding_dim}
