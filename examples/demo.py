"""Example: Use MemoryWeave for persistent agent memory."""

from memoryweave.store import MemoryStore
from memoryweave.weaver import MemoryWeaver
from memoryweave.retrieval import SmartRetrieval
from memoryweave.consolidation import MemoryConsolidator
from memoryweave.importance import ImportanceScorer


def main():
    print("=== MemoryWeave Demo ===\n")

    store = MemoryStore()

    # Add some memories
    m1 = store.add("Python is a versatile programming language", memory_type="semantic", tags=["python", "programming"])
    m2 = store.add("I had a great meeting with the team about the new API design", memory_type="episodic", tags=["work", "api"])
    m3 = store.add("Machine learning models need training data", memory_type="semantic", tags=["ml", "data"])
    m4 = store.add("The team decided to use FastAPI for the new project", memory_type="episodic", tags=["work", "api", "python"])
    m5 = store.add("Neural networks are inspired by biological neurons", memory_type="semantic", tags=["ml", "neuroscience"])
    m6 = store.add("Today I debugged a tricky async issue in the API", memory_type="episodic", tags=["work", "api", "python"])

    print(f"Added {store.count()} memories")
    print(f"Stats: {store.stats()}\n")

    # Semantic search
    print("--- Semantic Search: 'API development' ---")
    results = store.search_by_vector("API development", top_k=3)
    for sim, mem in results:
        print(f"  [{sim:.3f}] {mem.content[:80]}")

    # Weave connections
    print("\n--- Weaving Memory Links ---")
    weaver = MemoryWeaver(store, similarity_threshold=0.3)
    new_links = weaver.weave()
    print(f"Created {len(new_links)} new links")

    clusters = weaver.find_clusters(min_cluster_size=2)
    print(f"Found {len(clusters)} memory clusters")

    graph = weaver.get_knowledge_graph()
    print(f"Knowledge graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    # Smart retrieval
    print("\n--- Smart Retrieval: 'python web framework' ---")
    retrieval = SmartRetrieval(store)
    results = retrieval.retrieve("python web framework", top_k=3)
    for score, mem in results:
        print(f"  [{score:.3f}] {mem.content[:80]}")

    # Importance scoring
    print("\n--- Importance Scoring ---")
    scorer = ImportanceScorer()
    scored = scorer.batch_score(store.get_all())
    for score, mem in scored[:5]:
        print(f"  [{score:.3f}] {mem.content[:60]}")

    # Consolidation
    print("\n--- Memory Consolidation ---")
    consolidator = MemoryConsolidator(store)
    result = consolidator.consolidate()
    print(f"Merged: {result['merged']}, Pruned: {result['pruned']}, "
          f"Compressed: {result['compressed']}, Remaining: {result['remaining']}")


if __name__ == "__main__":
    main()
