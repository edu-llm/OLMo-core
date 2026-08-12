"""
Synthetic directed-graph reachability generator for the latent-CoT experiments.

This is Phase 1 of ``docs/latent-cot/latent-cot-superposition-prd.md``. It produces
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
- **Non-trivial negatives.** Every instance contains two parallel layered chains -- a *main*
  chain rooted at the source and a *decoy* chain whose root has no in-edges -- and the classes
  differ only in which chain the target sits on. Deciding reachability means tracing the target's
  ancestors back to see which root they reach, so the negatives cannot be recognised locally.

  This is the second attempt at that property and the first one was wrong, which is worth
  recording because it cost a training run and an eval. Unreachable targets used to be given no
  in-edges at all. That makes ``> T does not appear in the edge list`` exactly equivalent to the
  label, so the whole dataset is solvable by one substring test with no search: measured on the
  held-out split, that single rule scores **1.000 at every depth**, out-of-distribution depths
  included. The eval of ``run_019ff280`` showed it as a no-CoT lower anchor scoring a perfect
  1.000 at all six depths while the explicit-CoT upper anchor sat at chance -- an inverted pair
  of anchors is what a shortcut looks like from the outside. A rule that never consults depth
  transfers across depth perfectly, so held-out depths do not catch it either.

  Any change to :func:`generate` should be checked against
  ``src/test/latentcot/test_graph_gen.py::test_no_surface_heuristic_separates_the_classes``,
  which runs a battery of local rules and requires all of them to stay near chance. A dataset
  that a one-line rule can solve measures nothing about reasoning, and the gates built on it are
  vacuous rather than merely noisy.

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
            # JSON has no tuples, so edges round-trip as 2-element lists.
            edges=[(int(u), int(v)) for u, v in d["edges"]],
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

    The graph holds **two parallel layered chains**, and every instance contains both:

    - the **main** chain, rooted at the source on level 0;
    - the **decoy** chain, rooted on level 0 at its own node, which has no in-edges. No edge ever
      crosses between chains, so nothing on the decoy chain is reachable from the source.

    Every edge runs from level ``i`` to level ``i + 1`` within one chain, so any
    source-to-target path has length exactly ``depth``. The **only** difference between a
    reachable and an unreachable instance is which chain the target sits on. Both chains are
    wired by the same random process, so in-degree, out-degree, level occupancy, frontier depth
    and the count of in-degree-zero nodes are matched across the two classes by construction.

    WHY IT IS BUILT THIS WAY, BECAUSE THE OBVIOUS CONSTRUCTION IS SILENTLY BROKEN. Giving an
    unreachable target no in-edges is the natural way to make it unreachable, and it was what
    this function did. It makes the dataset solvable without any search at all: ``target has no
    in-edge`` is exactly ``not reachable``, so the single surface test "does ``> T`` appear in
    the edge list" scores **1.000 at every depth**, including depths held out as
    out-of-distribution -- measured, on the eval of ``run_019ff280``, where the no-CoT lower
    anchor scored a perfect 1.000 at all six depths while the explicit-CoT upper anchor sat at
    chance. A shortcut that ignores depth generalizes across depth perfectly, which is what that
    result was. The previous construction also defended against the subtler *frontier-depth*
    leak, which is why the trivial one is easy to miss: the negatives expand a full-depth
    frontier and look non-trivial while the answer is one substring away.

    Deciding reachability here requires tracing the target's ancestors back ``depth`` levels to
    see whether they terminate at the source or at the decoy root. One backward step does not
    separate the classes: every node on the decoy chain except its root has an in-edge too.

    :param num_nodes: Total nodes; must be at least ``1 + 2 * depth`` -- the source, plus one
        node per level on each of the two chains.
    :param branching: Maximum out-degree from a node to the next level within its chain.
    :param depth: Requested difficulty ``D`` (and the exact distance when reachable).
    :param seed: RNG seed for deterministic generation.
    :param reachable: Whether the instance should be reachable.

    :returns: The generated :class:`Example`.

    :raises ValueError: If ``num_nodes < 1 + 2 * depth`` or ``depth < 1`` or ``branching < 1``.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if branching < 1:
        raise ValueError(f"branching must be >= 1, got {branching}")
    if num_nodes < 2 + 2 * depth:
        raise ValueError(
            f"num_nodes ({num_nodes}) must be >= 2 + 2 * depth ({2 + 2 * depth}): the source and "
            "the decoy root, plus one node per level on each of the two chains. Both chains exist "
            "in every instance so that the two classes differ only in where the target sits."
        )

    rng = random.Random(seed)

    # Level assignment. Two chains of EQUAL length, each with its own level-0 root: the source
    # (always node 0) roots the main chain, and a separate in-degree-zero node roots the decoy
    # chain. Nothing on the decoy chain is reachable from the source because no edge ever crosses
    # between chains.
    #
    # THE DECOY ROOT SITS ON LEVEL 0 AND THAT IS LOAD-BEARING, NOT TIDINESS. With the decoy chain
    # starting on level 1 instead, its root is the target's direct predecessor whenever
    # depth == 2, so "does a predecessor of the target itself have a predecessor" separates the
    # classes perfectly at that depth -- caught by
    # test_no_surface_heuristic_separates_the_classes, which is the reason that test is a battery.
    # Giving the decoy chain a level-0 root makes the two chains structurally identical, so no
    # bounded number of backward steps distinguishes them: the ancestor walk has to run the full
    # `depth` levels and see which root it lands on. A level-0 node that is not the source does
    # not disturb the layering invariant, which only constrains edges whose tail is reachable.
    main: List[List[int]] = [[] for _ in range(depth + 1)]
    decoy: List[List[int]] = [[] for _ in range(depth + 1)]
    main[0].append(0)

    rest = list(range(1, num_nodes))
    rng.shuffle(rest)
    decoy[0].append(rest[0])
    cursor = 1
    for i in range(1, depth + 1):
        main[i].append(rest[cursor])
        decoy[i].append(rest[cursor + 1])
        cursor += 2
    # Scatter the remainder, alternating sides so the two chains stay comparably wide.
    for offset, node in enumerate(rest[cursor:]):
        (main if offset % 2 == 0 else decoy)[rng.randint(1, depth)].append(node)

    source = 0
    target = (main if reachable else decoy)[depth][0]

    edges = set()

    # Random forward edges within each chain, up to `branching` per node. Never between chains:
    # one cross edge into the decoy chain would make it reachable and the label a lie. The decoy
    # chain has an empty level 0, so its level-1 root receives nothing here.
    for chain in (main, decoy):
        for i in range(depth):
            if not chain[i] or not chain[i + 1]:
                continue
            for u in chain[i]:
                k = rng.randint(1, min(branching, len(chain[i + 1])))
                for v in rng.sample(chain[i + 1], k):
                    edges.add((u, v))

    # One backbone per chain, so both expand to full depth and the target always has ancestors
    # running back to its chain's root. Levels 1..depth are non-empty on both chains and main
    # level 0 holds the source, so each backbone is contiguous and stays layered.
    for chain in (main, decoy):
        spine = [chain[i][0] for i in range(depth + 1) if chain[i]]
        for u, v in zip(spine, spine[1:]):
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
