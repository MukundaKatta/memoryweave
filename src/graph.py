"""Memory graph: nodes=memories, edges=relationships. BFS/DFS traversal,
shortest path, connected components."""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .store import MemoryStore


class MemoryGraph:
    """Graph structure over memories with traversal and analysis algorithms."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._adjacency: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
        self._nodes: Set[str] = set()

    def build_from_store(self) -> None:
        """Build graph from all memories and links in the store."""
        self._adjacency.clear()
        self._nodes.clear()

        all_ids = self.store.get_all_ids()
        self._nodes = set(all_ids)

        for mid in all_ids:
            links = self.store.get_links(mid)
            for link in links:
                src = link["source_id"]
                tgt = link["target_id"]
                weight = link["weight"]
                relation = link["relation_type"]
                self._adjacency[src].append((tgt, weight, relation))
                self._adjacency[tgt].append((src, weight, relation))
                self._nodes.add(src)
                self._nodes.add(tgt)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(neighbors) for neighbors in self._adjacency.values()) // 2

    def neighbors(self, node_id: str) -> List[Tuple[str, float, str]]:
        """Get neighbors of a node: (neighbor_id, weight, relation_type)."""
        return self._adjacency.get(node_id, [])

    def degree(self, node_id: str) -> int:
        """Number of edges connected to a node."""
        return len(self._adjacency.get(node_id, []))

    def bfs(self, start_id: str, max_depth: Optional[int] = None) -> List[Tuple[str, int]]:
        """Breadth-first search from start node. Returns (node_id, depth) pairs."""
        if start_id not in self._nodes:
            return []

        visited: Set[str] = {start_id}
        queue: deque = deque([(start_id, 0)])
        result: List[Tuple[str, int]] = [(start_id, 0)]

        while queue:
            current, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            for neighbor, _, _ in self._adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    result.append((neighbor, depth + 1))

        return result

    def dfs(self, start_id: str, max_depth: Optional[int] = None) -> List[Tuple[str, int]]:
        """Depth-first search from start node. Returns (node_id, depth) pairs."""
        if start_id not in self._nodes:
            return []

        visited: Set[str] = set()
        result: List[Tuple[str, int]] = []

        def _dfs_recurse(node: str, depth: int) -> None:
            visited.add(node)
            result.append((node, depth))
            if max_depth is not None and depth >= max_depth:
                return
            for neighbor, _, _ in self._adjacency.get(node, []):
                if neighbor not in visited:
                    _dfs_recurse(neighbor, depth + 1)

        _dfs_recurse(start_id, 0)
        return result

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest unweighted path using BFS. Returns path or None."""
        if source not in self._nodes or target not in self._nodes:
            return None
        if source == target:
            return [source]

        visited: Set[str] = {source}
        queue: deque = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            for neighbor, _, _ in self._adjacency.get(current, []):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found

    def weighted_shortest_path(self, source: str, target: str) -> Optional[Tuple[List[str], float]]:
        """Dijkstra's algorithm for weighted shortest path.

        Uses inverse weight (1/weight) as distance since higher weight = stronger connection.
        Returns (path, total_distance) or None.
        """
        if source not in self._nodes or target not in self._nodes:
            return None

        import heapq

        dist: Dict[str, float] = {source: 0.0}
        prev: Dict[str, Optional[str]] = {source: None}
        heap: List[Tuple[float, str]] = [(0.0, source)]
        visited: Set[str] = set()

        while heap:
            d, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current == target:
                # Reconstruct path
                path: List[str] = []
                node: Optional[str] = target
                while node is not None:
                    path.append(node)
                    node = prev.get(node)
                return list(reversed(path)), d

            for neighbor, weight, _ in self._adjacency.get(current, []):
                if neighbor in visited:
                    continue
                edge_dist = 1.0 / max(weight, 1e-6)
                new_dist = d + edge_dist
                if new_dist < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_dist
                    prev[neighbor] = current
                    heapq.heappush(heap, (new_dist, neighbor))

        return None

    def connected_components(self) -> List[Set[str]]:
        """Find all connected components using BFS."""
        visited: Set[str] = set()
        components: List[Set[str]] = []

        for node in self._nodes:
            if node in visited:
                continue
            component: Set[str] = set()
            queue: deque = deque([node])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor, _, _ in self._adjacency.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)

        return sorted(components, key=len, reverse=True)

    def pagerank(self, damping: float = 0.85, iterations: int = 50, tolerance: float = 1e-6) -> Dict[str, float]:
        """Compute PageRank scores for all nodes."""
        nodes = list(self._nodes)
        n = len(nodes)
        if n == 0:
            return {}

        node_idx = {node: i for i, node in enumerate(nodes)}
        scores = np.ones(n, dtype=np.float64) / n

        for _ in range(iterations):
            new_scores = np.ones(n, dtype=np.float64) * (1 - damping) / n

            for node in nodes:
                i = node_idx[node]
                neighbors = self._adjacency.get(node, [])
                if not neighbors:
                    # Dangling node: distribute evenly
                    new_scores += damping * scores[i] / n
                else:
                    share = damping * scores[i] / len(neighbors)
                    for neighbor, _, _ in neighbors:
                        if neighbor in node_idx:
                            new_scores[node_idx[neighbor]] += share

            # Check convergence
            diff = np.abs(new_scores - scores).sum()
            scores = new_scores
            if diff < tolerance:
                break

        return {nodes[i]: float(scores[i]) for i in range(n)}

    def clustering_coefficient(self, node_id: str) -> float:
        """Compute local clustering coefficient for a node."""
        neighbors_list = [n for n, _, _ in self._adjacency.get(node_id, [])]
        k = len(neighbors_list)
        if k < 2:
            return 0.0

        neighbor_set = set(neighbors_list)
        triangles = 0
        for i, n1 in enumerate(neighbors_list):
            for n2 in neighbors_list[i + 1 :]:
                # Check if n1 and n2 are connected
                n1_neighbors = {nb for nb, _, _ in self._adjacency.get(n1, [])}
                if n2 in n1_neighbors:
                    triangles += 1

        max_triangles = k * (k - 1) / 2
        return triangles / max_triangles if max_triangles > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        components = self.connected_components()
        degrees = [self.degree(n) for n in self._nodes]
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_components": len(components),
            "largest_component_size": len(components[0]) if components else 0,
            "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
            "max_degree": max(degrees) if degrees else 0,
        }
