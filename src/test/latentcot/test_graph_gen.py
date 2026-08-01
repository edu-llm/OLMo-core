"""Tests for the synthetic graph-reachability generator (PRD Phase 1)."""

from collections import deque
from typing import Dict, List, Tuple

import pytest

from olmo_core.latentcot.data.graph_gen import Example, generate

# A grid that spans the difficulty axis, including OOD-ish depths.
DEPTHS = [2, 3, 4, 5, 6, 8]
BRANCHINGS = [2, 3, 4]


def _independent_layers(
    edges: List[Tuple[int, int]], num_nodes: int, source: int
) -> List[List[int]]:
    """A from-scratch BFS, independent of the generator's own BFS."""
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        adj[u].append(v)
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    if not dist:
        return []
    layers: List[List[int]] = [[] for _ in range(max(dist.values()) + 1)]
    for node in sorted(dist):
        layers[dist[node]].append(node)
    return layers


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("branching", BRANCHINGS)
def test_reachable_distance_equals_depth(depth: int, branching: int):
    ex = generate(num_nodes=6 * depth, branching=branching, depth=depth, seed=depth, reachable=True)
    assert ex.reachable
    assert ex.distance == depth
    assert ex.target in ex.frontiers[depth]
    assert ex.path is not None and len(ex.path) == depth + 1
    assert ex.path[0] == ex.source and ex.path[-1] == ex.target


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("branching", BRANCHINGS)
def test_unreachable_target_isolated(depth: int, branching: int):
    ex = generate(
        num_nodes=6 * depth, branching=branching, depth=depth, seed=1000 + depth, reachable=False
    )
    assert not ex.reachable
    assert ex.distance is None and ex.path is None
    assert all(v != ex.target for (_, v) in ex.edges), "target must have no in-edges"
    assert all(ex.target not in layer for layer in ex.frontiers)


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("reachable", [True, False])
def test_frontiers_match_independent_bfs(depth: int, reachable: bool):
    ex = generate(num_nodes=6 * depth, branching=3, depth=depth, seed=42, reachable=reachable)
    assert _independent_layers(ex.edges, ex.num_nodes, ex.source) == ex.frontiers


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("reachable", [True, False])
def test_layered_no_shortcuts(depth: int, reachable: bool):
    # Every edge from a reachable node must advance the BFS distance by exactly 1.
    ex = generate(num_nodes=6 * depth, branching=4, depth=depth, seed=7, reachable=reachable)
    dist = {node: k for k, layer in enumerate(ex.frontiers) for node in layer}
    assert all(dist[v] == dist[u] + 1 for (u, v) in ex.edges if u in dist)


def test_determinism_and_hash():
    a = generate(num_nodes=30, branching=3, depth=4, seed=7, reachable=True)
    b = generate(num_nodes=30, branching=3, depth=4, seed=7, reachable=True)
    c = generate(num_nodes=30, branching=3, depth=4, seed=8, reachable=True)
    assert a.edges == b.edges and a.graph_hash == b.graph_hash
    assert a.graph_hash != c.graph_hash  # different seed -> different graph


def test_roundtrip_to_from_dict():
    ex = generate(num_nodes=24, branching=3, depth=4, seed=3, reachable=True)
    restored = Example.from_dict(ex.to_dict())
    assert restored == ex


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_nodes=3, branching=2, depth=5, seed=0, reachable=True),  # n < depth+1
        dict(num_nodes=10, branching=0, depth=3, seed=0, reachable=True),  # branching < 1
        dict(num_nodes=10, branching=2, depth=0, seed=0, reachable=True),  # depth < 1
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        generate(**kwargs)
