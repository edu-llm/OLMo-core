"""
Generate and verify the synthetic graph-reachability dataset (PRD Phase 1.2/1.3).

Writes ``train.jsonl``, ``test.jsonl`` and ``meta.json`` under an output
directory (default ``local/latentcot/data``), then verifies every instance with
an *independent* BFS, prints the depth (difficulty) histogram and label balance,
and asserts the train/test graphs are disjoint by hash.

The test split is deliberately harder than a seed reshuffle: it uses a disjoint
seed range *and* includes out-of-distribution ``(num_nodes, branching, depth)``
combinations (unseen depths) so we can later separate memorization from
generalization.

Usage::

    .venv/bin/python src/scripts/latentcot/gen_graph_data.py
    .venv/bin/python src/scripts/latentcot/gen_graph_data.py --out local/latentcot/data --per-combo 150
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Tuple

from olmo_core.latentcot.data.graph_gen import Example, generate

# Difficulty grid. Nodes-per-level is held ~constant so `num_nodes` scales with depth.
WIDTH = 6
BRANCHINGS = [3, 4]
TRAIN_DEPTHS = [2, 3, 4, 6]
TEST_INDIST_DEPTHS = [2, 3, 4, 6]  # same combos as train, disjoint seeds
TEST_OOD_DEPTHS = [5, 8]  # unseen depths -> unseen (num_nodes, branching, depth)

# Seed ranges are far apart so train and test never share a seed (or a graph).
TRAIN_SEED_BASE = 0
TEST_SEED_BASE = 10_000_000


def num_nodes_for(depth: int) -> int:
    """Nodes for a given depth, keeping ~WIDTH nodes per level (>= depth + 1)."""
    return max(depth + 1, WIDTH * depth)


def build_split(
    depths: List[int], branchings: List[int], per_combo: int, seed_base: int
) -> List[Example]:
    """Generate a balanced (reachable/unreachable) split over the given grid."""
    examples: List[Example] = []
    seed = seed_base
    for depth in depths:
        n = num_nodes_for(depth)
        for branching in branchings:
            for _ in range(per_combo):
                for reachable in (True, False):
                    examples.append(
                        generate(
                            num_nodes=n,
                            branching=branching,
                            depth=depth,
                            seed=seed,
                            reachable=reachable,
                        )
                    )
                    seed += 1
    return examples


# --- Independent verification (a second BFS implementation, not graph_gen's) ---


def _independent_bfs_layers(
    edges: List[Tuple[int, int]], num_nodes: int, source: int
) -> List[List[int]]:
    """A from-scratch BFS used only to cross-check the generator's stored fields."""
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


def verify(ex: Example) -> None:
    """Cross-check one instance against an independent BFS; raise on any mismatch."""
    layers = _independent_bfs_layers(ex.edges, ex.num_nodes, ex.source)
    assert layers == ex.frontiers, f"frontiers mismatch (seed {ex.seed})"
    reachable = any(ex.target in layer for layer in layers)
    assert reachable == ex.reachable, f"reachability mismatch (seed {ex.seed})"
    if ex.reachable:
        distance = next(k for k, layer in enumerate(layers) if ex.target in layer)
        assert distance == ex.depth == ex.distance, f"distance mismatch (seed {ex.seed})"
        assert ex.path is not None and len(ex.path) == ex.depth + 1
    else:
        assert ex.distance is None and ex.path is None
    # No shortcuts / backward edges: every edge from a reachable node advances distance by 1.
    dist = {node: k for k, layer in enumerate(layers) for node in layer}
    assert all(
        dist[v] == dist[u] + 1 for (u, v) in ex.edges if u in dist
    ), f"non-layered edge found (seed {ex.seed})"


def histogram(examples: List[Example]) -> Dict[int, Dict[str, int]]:
    """Count reachable/unreachable per depth."""
    hist: Dict[int, Counter] = {}
    for ex in examples:
        hist.setdefault(ex.depth, Counter())["reach" if ex.reachable else "unreach"] += 1
    return {d: dict(c) for d, c in sorted(hist.items())}


def write_jsonl(path: Path, examples: List[Example]) -> None:
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("local/latentcot/data"))
    parser.add_argument(
        "--per-combo",
        type=int,
        default=150,
        help="reachable+unreachable pairs per (branching, depth)",
    )
    parser.add_argument("--test-per-combo", type=int, default=40)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    train = build_split(TRAIN_DEPTHS, BRANCHINGS, args.per_combo, TRAIN_SEED_BASE)
    test_indist = build_split(TEST_INDIST_DEPTHS, BRANCHINGS, args.test_per_combo, TEST_SEED_BASE)
    test_ood = build_split(
        TEST_OOD_DEPTHS, BRANCHINGS, args.test_per_combo, TEST_SEED_BASE + 5_000_000
    )
    test = test_indist + test_ood

    print(f"Generated {len(train)} train / {len(test)} test instances. Verifying...")
    for ex in train + test:
        verify(ex)
    print("Independent-BFS verification: PASSED for all instances.")

    train_hashes = {ex.graph_hash for ex in train}
    test_hashes = {ex.graph_hash for ex in test}
    overlap = train_hashes & test_hashes
    assert not overlap, f"train/test graph overlap: {len(overlap)} shared graphs"
    print(
        f"Train/test disjointness: OK ({len(train_hashes)} unique train, "
        f"{len(test_hashes)} unique test, 0 shared)."
    )

    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "test.jsonl", test)

    meta = {
        "width_per_level": WIDTH,
        "branchings": BRANCHINGS,
        "train_depths": TRAIN_DEPTHS,
        "test_indist_depths": TEST_INDIST_DEPTHS,
        "test_ood_depths": TEST_OOD_DEPTHS,
        "counts": {
            "train": len(train),
            "test": len(test),
            "test_indist": len(test_indist),
            "test_ood": len(test_ood),
        },
        "train_hist": histogram(train),
        "test_hist": histogram(test),
        "num_nodes_by_depth": {
            d: num_nodes_for(d)
            for d in sorted(set(TRAIN_DEPTHS + TEST_INDIST_DEPTHS + TEST_OOD_DEPTHS))
        },
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\nDepth histogram (train):")
    for d, c in meta["train_hist"].items():
        print(f"  D={d}: reachable={c.get('reach', 0):4d}  unreachable={c.get('unreach', 0):4d}")
    print("Depth histogram (test):")
    for d, c in meta["test_hist"].items():
        tag = " (OOD)" if d in TEST_OOD_DEPTHS else ""
        print(
            f"  D={d}{tag}: reachable={c.get('reach', 0):4d}  unreachable={c.get('unreach', 0):4d}"
        )
    print(f"\nWrote {args.out}/train.jsonl, test.jsonl, meta.json")


if __name__ == "__main__":
    main()
