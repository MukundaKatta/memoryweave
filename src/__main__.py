"""CLI for memoryweave."""
import sys, json, argparse
from .core import Memoryweave

def main():
    parser = argparse.ArgumentParser(description="Agent memory that learns — persistent, searchable, and self-organizing memory for AI agents")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Memoryweave()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.store(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"memoryweave v0.1.0 — Agent memory that learns — persistent, searchable, and self-organizing memory for AI agents")

if __name__ == "__main__":
    main()
