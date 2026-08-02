"""
Synthetic directed-graph reachability generator for the latent-CoT experiments.

This is Phase 1 of ``local/latent-cot-superposition-prd.md``. It produces
controllable reachability problems on *layered* directed graphs, where every
edge advances exactly one level. That structure gives three properties the
experiment relies on:

- **Controllable difficulty.** For a reachable instance the shortest
  source-to-target distance is exactly the requested ``depth`` (no shortcuts are
  possible, because every edge advances one level), so difficulty scales cleanly
  with ``depth`` — the axis the superposition theory predicts the continuous-CoT
  advantage grows along.
- **A ready reasoning trace / probing target.** The breadth-first-search
  frontiers from the source (``frontiers[k]`` = the set of nodes reachable in
  exactly ``k`` hops) are exactly the "set of candidates held at reasoning step
  ``k``" that a superposition state should encode. They serve both as the
  teacher chain-of-thought (Phase 2/4) and as the per-step probing labels
  (Phase 6/L8).
- **Non-trivial negatives.** Unreachable instances still expand a wide frontier
  through the graph; the target simply has no in-edges, so concluding "not
  reachable" requires exploring the reachable set rather than a one-step check.

Node ``0`` is always the source. Only the standard library is used so the
generator has no heavy dependencies.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "Example",
    "generate",
    "build_adjacency",
    "bfs_distances",
    "bfs_layers",
    "shortest_path",
]


@dataclass
class Example:
    """
    A single reachability problem instance.

    :param num_nodes: Total number of nodes; ids are ``0 .. num_nodes - 1``.
    :param edges: Directed edges as ``(u, v)`` pairs, sorted and de-duplicated.
    :param source: The source node (always ``0``).
    :param target: The target node whose reachability is queried.
    :param reachable: Whether ``target`` is reachable from ``source``.
    :param depth: The requested difficulty ``D``. For reachable instances this
        equals the true shortest source-to-target distance.
    :param branching: The maximum out-degree used when wiring each node to the
        next level.
    :param distance: The true shortest source-to-target distance, or ``None``
        when unreachable.
    :param frontiers: BFS layers from ``source``; ``frontiers[k]`` lists the
        nodes at distance exactly ``k``.
    :param path: A shortest source-to-target path, or ``None`` when unreachable.
    :param seed: The RNG seed used to build this instance.
    """

    num_nodes: int
    edges: List[Tuple[int, int]]
    source: int
    target: int
    reachable: bool
    depth: int
    branching: int
    distance: Optional[int]
    frontiers: List[List[int]]
    path: Optional[List[int]]
    seed: int

    @property
    def graph_hash(self) -> str:
        """A stable hash of the graph + query, used to check train/test disjointness."""
        canon = json.dumps(
            {
                "n": self.num_nodes,
                "edges": [list(e) for e in sorted(self.edges)],
                "s": self.source,
                "t": self.target,
            },
            sort_keys=True,
        )
        return hashlib.sha1(canon.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (tuples become lists)."""
        return {
            "num_nodes": self.num_nodes,
            "edges": [list(e) for e in self.edges],
            "source": self.source,
            "target": self.target,
            "reachable": self.reachable,
            "depth": self.depth,
            "branching": self.branching,
            "distance": self.distance,
            "frontiers": self.frontiers,
            "path": self.path,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        """Inverse of :meth:`to_dict`."""
        return cls(
            num_nodes=d["num_nodes"],
            edges=[tuple(e) for e in d["edges"]],
            source=d["source"],
            target=d["target"],
            reachable=d["reachable"],
            depth=d["depth"],
            branching=d["branching"],
            distance=d["distance"],
            frontiers=d["frontiers"],
            path=d["path"],
            seed=d["seed"],
        )


def build_adjacency(edges: List[Tuple[int, int]], num_nodes: int) -> Dict[int, List[int]]:
    """Build an adjacency list (every node present, possibly with an empty list)."""
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        adj[u].append(v)
    return adj


def bfs_distances(adj: Dict[int, List[int]], source: int) -> Dict[int, int]:
    """Return a map from node to its BFS distance from ``source`` (reachable nodes only)."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def bfs_layers(adj: Dict[int, List[int]], source: int) -> List[List[int]]:
    """Return BFS layers from ``source``; index ``k`` is the sorted nodes at distance ``k``."""
    dist = bfs_distances(adj, source)
    if not dist:
        return []
    max_d = max(dist.values())
    layers: List[List[int]] = [[] for _ in range(max_d + 1)]
    for node in sorted(dist):
        layers[dist[node]].append(node)
    return layers


def shortest_path(adj: Dict[int, List[int]], source: int, target: int) -> Optional[List[int]]:
    """Return one shortest ``source``-to-``target`` path, or ``None`` if unreachable."""
    if source == target:
        return [source]
    prev: Dict[int, int] = {source: source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, ()):
            if v not in prev:
                prev[v] = u
                if v == target:
                    path = [v]
                    while path[-1] != source:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                queue.append(v)
    return None


def generate(
    *,
    num_nodes: int,
    branching: int,
    depth: int,
    seed: int,
    reachable: bool,
) -> Example:
    """
    Generate one layered directed-graph reachability instance.

    The graph is organized into ``depth + 1`` levels with the source alone on
    level 0 and the target on level ``depth``. Every edge goes from level ``i``
    to level ``i + 1``, so any source-to-target path has length exactly
    ``depth``. For reachable instances a backbone path guarantees the target is
    reached at distance ``depth``; for unreachable instances the target is given
    no in-edges while the rest of the graph still expands a wide frontier.

    :param num_nodes: Total nodes; must be at least ``depth + 1``.
    :param branching: Maximum out-degree from a node to the next level.
    :param depth: Requested difficulty ``D`` (and the exact distance when reachable).
    :param seed: RNG seed for deterministic generation.
    :param reachable: Whether the instance should be reachable.

    :returns: The generated :class:`Example`.

    :raises ValueError: If ``num_nodes < depth + 1`` or ``depth < 1`` or ``branching < 1``.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if branching < 1:
        raise ValueError(f"branching must be >= 1, got {branching}")
    if num_nodes < depth + 1:
        raise ValueError(f"num_nodes ({num_nodes}) must be >= depth + 1 ({depth + 1})")

    rng = random.Random(seed)

    # Assign nodes to levels: source alone at level 0, then guarantee one node
    # per level 1..depth, then a guaranteed second node on the last level (a
    # non-target "sink" so unreachable instances can expand a frontier of the same
    # depth as reachable ones), then scatter the remainder.
    levels: List[List[int]] = [[] for _ in range(depth + 1)]
    levels[0].append(0)
    rest = list(range(1, num_nodes))
    rng.shuffle(rest)
    for i in range(1, depth + 1):
        levels[i].append(rest[i - 1])
    idx = depth
    if idx < len(rest):
        levels[depth].append(rest[idx])
        idx += 1
    for node in rest[idx:]:
        levels[rng.randint(1, depth)].append(node)

    source = 0
    target = levels[depth][0]

    edges = set()

    # Random forward edges (level i -> level i+1), up to `branching` per node.
    # For unreachable instances the target is excluded from every candidate set,
    # so it receives no in-edges.
    for i in range(depth):
        candidates = [v for v in levels[i + 1] if reachable or v != target]
        if not candidates:
            continue
        for u in levels[i]:
            k = rng.randint(1, min(branching, len(candidates)))
            for v in rng.sample(candidates, k):
                edges.add((u, v))

    # Backbone: a guaranteed source path down the levels so the frontier expands
    # to full depth. For reachable instances it ends at the target (fixing the
    # shortest distance = depth). For unreachable instances it runs through levels
    # 0..depth-1 and on to a non-target node on the last level when one exists, so
    # negatives expand a frontier of matched depth while the target stays isolated
    # (no in-edges) -- this prevents "frontier depth" from leaking the answer.
    backbone = [levels[i][0] for i in range(depth)]  # one node per level 0..depth-1
    if reachable:
        backbone.append(target)
    else:
        others = [v for v in levels[depth] if v != target]
        if others:
            backbone.append(others[0])
    for u, v in zip(backbone, backbone[1:]):
        edges.add((u, v))

    edge_list = sorted(edges)
    adj = build_adjacency(edge_list, num_nodes)
    dist = bfs_distances(adj, source)
    frontiers = bfs_layers(adj, source)

    # Construction invariants (cheap, and they catch wiring bugs early).
    is_reachable = target in dist
    assert is_reachable == reachable, "constructed reachability disagrees with request"
    distance = dist.get(target)
    if reachable:
        assert distance == depth, f"expected distance {depth}, got {distance}"

    path = shortest_path(adj, source, target) if reachable else None

    return Example(
        num_nodes=num_nodes,
        edges=edge_list,
        source=source,
        target=target,
        reachable=reachable,
        depth=depth,
        branching=branching,
        distance=distance,
        frontiers=frontiers,
        path=path,
        seed=seed,
    )
