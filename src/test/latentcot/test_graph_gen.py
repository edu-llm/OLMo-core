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
def test_unreachable_target_still_has_in_edges(depth: int, branching: int):
    """
    An unreachable target must be unreachable *and* have in-edges.

    This assertion is the reverse of the one it replaces, which required the target to have no
    in-edges. That was the bug: no-in-edges is equivalent to the label, so the dataset was
    solvable by one substring test and the no-CoT anchor scored a perfect 1.000 on every depth.
    The target now hangs off the decoy chain, so it has ancestors -- they just do not lead back
    to the source.
    """
    ex = generate(
        num_nodes=6 * depth, branching=branching, depth=depth, seed=1000 + depth, reachable=False
    )
    assert not ex.reachable
    assert ex.distance is None and ex.path is None
    assert any(v == ex.target for (_, v) in ex.edges), "unreachable target must have in-edges"
    assert all(ex.target not in layer for layer in ex.frontiers)


def _surface_rules():
    """Local rules that must NOT decide reachability. Name -> predicate over an Example."""
    return {
        "target is an edge destination": lambda ex: any(v == ex.target for _, v in ex.edges),
        "target appears anywhere in edges": lambda ex: any(
            ex.target in (u, v) for u, v in ex.edges
        ),
        "target has an outgoing edge": lambda ex: any(u == ex.target for u, _ in ex.edges),
        "target in-degree > 1": lambda ex: sum(1 for _, v in ex.edges if v == ex.target) > 1,
        # One backward step: does any direct predecessor of the target itself have a predecessor?
        # True on the decoy chain too for depth >= 2, so it must not separate the classes.
        "a predecessor of target has a predecessor": lambda ex: any(
            any(w == u for _, w in ex.edges) for u, v in ex.edges if v == ex.target
        ),
        "target id is below the midpoint": lambda ex: ex.target < ex.num_nodes / 2,
    }


@pytest.mark.parametrize("depth", [2, 4, 8])
def test_no_surface_heuristic_separates_the_classes(depth: int):
    """
    No local rule may predict reachability better than chance, at any depth.

    This is the regression test for the leak that made the first sweep meaningless. It is written
    as a battery rather than one assertion because the failure mode is a rule nobody thought of:
    the original bug was found only after a trained model scored 1.000 on held-out
    out-of-distribution depths, which is what a depth-independent shortcut looks like.

    The bound is deliberately loose (0.60). These are small samples, and the point is to catch a
    rule that *decides* the label, not to police sampling noise.
    """
    n = 6 * depth
    # Paired by seed, which is the strongest form this test can take: nothing in the construction
    # except the choice of target consults `reachable`, so one seed yields the SAME graph for both
    # classes with a different node named as the sink. Any rule that reads only graph structure is
    # then exactly at chance by construction, pairing removes the variance that would otherwise
    # need a large sample, and an even/odd-seed confound (which an earlier version of this test
    # had, and which showed up as a spurious 0.606) cannot arise.
    examples = []
    for i in range(80):
        for reachable in (True, False):
            examples.append(
                generate(
                    num_nodes=n, branching=3, depth=depth, seed=90_000 + i, reachable=reachable
                )
            )
    assert sum(ex.reachable for ex in examples) == len(examples) // 2  # balanced, chance = 0.5
    # The matched-pair property itself, asserted so a future change cannot quietly lose it.
    assert examples[0].edges == examples[1].edges
    assert examples[0].target != examples[1].target

    for name, rule in _surface_rules().items():
        agree = sum(1 for ex in examples if rule(ex) == ex.reachable) / len(examples)
        # A rule that is reliably *anti*-correlated is just as strong a shortcut as one that is
        # correlated, so score the distance from chance in either direction.
        assert abs(agree - 0.5) < 0.10, (
            f"surface rule {name!r} predicts reachability at {agree:.3f} on depth {depth} "
            "(chance is 0.500). The dataset is solvable without search and the gates built on "
            "it would be vacuous."
        )


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


@pytest.mark.parametrize("depth", DEPTHS)
def test_matched_frontier_depth_prevents_answer_leak(depth: int):
    # Confound guard: with enough nodes, unreachable instances expand a frontier of
    # the SAME depth as reachable ones, so the label can't be read off frontier depth.
    reach = generate(
        num_nodes=6 * depth, branching=3, depth=depth, seed=900 + depth, reachable=True
    )
    unreach = generate(
        num_nodes=6 * depth, branching=3, depth=depth, seed=900 + depth, reachable=False
    )
    assert len(reach.frontiers) - 1 == depth
    assert len(unreach.frontiers) - 1 == depth


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
