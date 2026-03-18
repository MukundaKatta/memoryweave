"""Basic usage example for memoryweave."""
from src.core import Memoryweave

def main():
    instance = Memoryweave(config={"verbose": True})

    print("=== memoryweave Example ===\n")

    # Run primary operation
    result = instance.store(input="example data", mode="demo")
    print(f"Result: {result}")

    # Run multiple operations
    ops = ["store", "retrieve", "consolidate]
    for op in ops:
        r = getattr(instance, op)(source="example")
        print(f"  {op}: {"✓" if r.get("ok") else "✗"}")

    # Check stats
    print(f"\nStats: {instance.get_stats()}")

if __name__ == "__main__":
    main()
