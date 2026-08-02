"""
Generate + verify the graph-reachability dataset in the platform-compliant shape (PRD Phase 1.2/1.3).

Emits the ``sft-conversations/v1`` layout expected by the eduLLM dataset validator
(`edullm-datasets` skill): a single group directory ``conversations/`` holding
``train-00000.jsonl`` and ``heldout-00000.jsonl``, where each row is the full ``Example``
(so our latent-CoT pipeline can reconstruct it) plus a ``messages`` array
(``user`` = reachability query + edges, ``assistant`` = BFS reasoning + yes/no).

It then verifies every instance with an *independent* BFS, prints the depth histogram and
label balance, asserts train/test graphs are disjoint, and asserts **0 train/heldout leakage**
by the validator's own dedup key (sha256 of the message contents) so publishing won't be rejected.

Publish it with ``src/scripts/latentcot/publish_dataset.py`` (dataset id
``sft/graph-reachability-depth``). The out dir is the publish ``source`` — it contains ONLY the
group directory (meta.json is written outside it, since a file not under a group prefix is a
hard publish error).

Usage::

    .venv/bin/python src/scripts/latentcot/gen_graph_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Tuple

from olmo_core.latentcot.data.encode import to_sft_record
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

GROUP = "conversations"


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
    dist = {node: k for k, layer in enumerate(layers) for node in layer}
    assert all(
        dist[v] == dist[u] + 1 for (u, v) in ex.edges if u in dist
    ), f"non-layered edge found (seed {ex.seed})"


def histogram(examples: List[Example]) -> Dict[int, Dict[str, int]]:
    hist: Dict[int, Counter] = {}
    for ex in examples:
        hist.setdefault(ex.depth, Counter())["reach" if ex.reachable else "unreach"] += 1
    return {d: dict(c) for d, c in sorted(hist.items())}


def dedup_key(record: dict) -> str:
    """The validator's default sft leakage key: sha256 over each message's role + content."""
    parts = [f"{m['role']}\x1f{m['content']}" for m in record["messages"]]
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def write_shard(path: Path, examples: List[Example]) -> List[str]:
    """Write one JSONL shard of sft records; return the per-row dedup keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    with path.open("w") as f:
        for ex in examples:
            record = to_sft_record(ex)
            keys.append(dedup_key(record))
            f.write(json.dumps(record) + "\n")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("local/latentcot/graph-reachability-depth"),
        help="publish source dir (contains ONLY the group directory)",
    )
    parser.add_argument("--per-combo", type=int, default=150)
    parser.add_argument("--test-per-combo", type=int, default=40)
    args = parser.parse_args()

    train = build_split(TRAIN_DEPTHS, BRANCHINGS, args.per_combo, TRAIN_SEED_BASE)
    test_indist = build_split(TEST_INDIST_DEPTHS, BRANCHINGS, args.test_per_combo, TEST_SEED_BASE)
    test_ood = build_split(
        TEST_OOD_DEPTHS, BRANCHINGS, args.test_per_combo, TEST_SEED_BASE + 5_000_000
    )
    test = test_indist + test_ood

    print(f"Generated {len(train)} train / {len(test)} heldout instances. Verifying...")
    for ex in train + test:
        verify(ex)
    print("Independent-BFS verification: PASSED for all instances.")

    train_hashes = {ex.graph_hash for ex in train}
    test_hashes = {ex.graph_hash for ex in test}
    assert not (train_hashes & test_hashes), "train/test graph overlap"
    print(
        f"Train/test graph disjointness: OK ({len(train_hashes)}/{len(test_hashes)} unique, 0 shared)."
    )

    group_dir = args.out / GROUP
    train_keys = write_shard(group_dir / "train-00000.jsonl", train)
    heldout_keys = write_shard(group_dir / "heldout-00000.jsonl", test)

    # The validator recomputes train/heldout leakage from these keys and rejects any overlap.
    leak = set(train_keys) & set(heldout_keys)
    assert not leak, f"train/heldout leakage: {len(leak)} shared conversations (would be rejected)"
    print("Train/heldout leakage (validator dedup key): 0 shared. OK.")

    meta = {
        "dataset_id": "sft/graph-reachability-depth",
        "profile": "sft-conversations/v1",
        "group": GROUP,
        "counts": {"train": len(train), "heldout": len(test)},
        "train_hist": histogram(train),
        "test_hist": histogram(test),
        "num_nodes_by_depth": {
            d: num_nodes_for(d)
            for d in sorted(set(TRAIN_DEPTHS + TEST_INDIST_DEPTHS + TEST_OOD_DEPTHS))
        },
    }
    # meta.json lives OUTSIDE the publish source dir (a non-group file there is a publish error).
    meta_path = args.out.parent / f"{args.out.name}-meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\nDepth histogram (train):")
    for d, c in meta["train_hist"].items():
        print(f"  D={d}: reachable={c.get('reach', 0):4d}  unreachable={c.get('unreach', 0):4d}")
    print("Depth histogram (heldout):")
    for d, c in meta["test_hist"].items():
        tag = " (OOD)" if d in TEST_OOD_DEPTHS else ""
        print(
            f"  D={d}{tag}: reachable={c.get('reach', 0):4d}  unreachable={c.get('unreach', 0):4d}"
        )
    print(
        f"\nWrote {group_dir}/train-00000.jsonl + heldout-00000.jsonl  (publish source: {args.out})"
    )
    print(f"Wrote {meta_path} (diagnostics, outside the publish dir)")


if __name__ == "__main__":
    main()
