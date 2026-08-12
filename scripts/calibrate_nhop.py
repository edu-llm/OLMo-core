#!/usr/bin/env python3
"""Calibrate the n-hop endpoint at every depth. Run this BEFORE training anything.

This is the cheap check that has to come first. It trains nothing: it scores a
perfect oracle, a noisy oracle at a given per-hop reliability, and a set of
degenerate policies through the *production* parser, then reports whether the
endpoint has enough dynamic range to register an effect.

    python scripts/calibrate_nhop.py --depths 1 2 3 4 5 --n-items 300

Exit status is 1 if the endpoint is unusable, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memsplit import bios, calibration, nhop  # noqa: E402


def build_items(
    n_entities: int, n_layers: int, depths: list[int], n_items: int, seed: int
) -> tuple[dict[int, list], list[str]]:
    recs = bios.generate_records(n_entities, seed=seed)
    graph = nhop.build_graph(recs, n_layers=n_layers, seed=seed)
    by_id = {r.entity_id: r for r in recs}
    # One start pool for every depth, so depth is not entangled with entity.
    starts = nhop.eligible_starts(graph, max(depths))
    if not starts:
        raise SystemExit(
            f"no entity can start a depth-{max(depths)} chain; raise --n-layers"
        )

    # Every (start, attribute) pair is an item, not just the first attribute that
    # works. That is 5x the items per entity and it spreads the target attribute
    # across the eval, so a per-attribute quirk cannot carry a depth stratum.
    out: dict[int, list] = {}
    for d in depths:
        items = []
        for eid in starts:
            for attr in bios.ATTRIBUTES:
                got = nhop.sample_item(graph, by_id, eid, d, attr, seed=seed)
                if got:
                    chain, end, value = got
                    items.append(
                        nhop.make_item(graph, by_id, eid, chain, attr, value, "comp")
                    )
                if len(items) >= n_items:
                    break
            if len(items) >= n_items:
                break
        out[d] = items
    vocab = [v for a in bios.ATTRIBUTES for v in bios.VALUE_POOLS[a]]
    return out, vocab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-entities", type=int, default=600)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--per-hop-noisy", type=float, default=0.93)
    ap.add_argument("--min-range-pp", type=float, default=10.0)
    ap.add_argument("--mde-pp", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    items_by_depth, vocab = build_items(
        args.n_entities, args.n_layers, args.depths, args.n_items, args.seed
    )
    chance = bios.chance_accuracy("employer")
    verdict = calibration.calibrate_endpoint(
        items_by_depth,
        vocab,
        chance=chance,
        min_range_pp=args.min_range_pp,
        per_hop_noisy=args.per_hop_noisy,
        seed=args.seed,
    )

    print(f"pool sizes        : {bios.pool_sizes()}")
    print(f"bits per entity   : {bios.bits_per_entity():.2f}")
    print(f"exact-match chance: {chance:.5f}")
    print(f"templates per slot: {nhop.n_templates()}")
    print()
    hdr = (
        f"{'depth':>5} {'n':>5} {'lookups':>7} {'floor%':>7} {'degen%':>7} "
        f"{'ceil%':>6} {'obs%':>6} {'p^n%':>6} {'range pp':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in verdict.per_depth:
        print(
            f"{c.depth:>5} {c.n_items:>5} {c.depth + 1:>7} "
            f"{100 * c.best_constant:>7.2f} {100 * c.untrained_accuracy:>7.2f} "
            f"{100 * c.oracle_accuracy:>6.1f} {100 * c.oracle_noisy_accuracy:>6.1f} "
            f"{100 * c.oracle_noisy_expected:>6.1f} {c.dynamic_range_pp:>9.1f}"
        )
    print()
    print("p**n null, the curve every depth plot must be read against:")
    for row in nhop.pn_table({"dense": 0.93, "split": 0.999}, args.depths)["rows"]:
        print(
            f"  depth {row['depth']}  lookups {row['n_lookups']}  "
            f"dense {row['pred_dense']:.3f}  split {row['pred_split']:.3f}  "
            f"gap {row['pred_gap_pp']:.1f}pp"
        )
    print()
    n_needed = calibration.required_n_for_mde(args.mde_pp, sd_pp=33.3)
    print(f"items needed for an {args.mde_pp}pp MDE at 80% power: {n_needed}")
    for c in verdict.per_depth:
        if c.n_items < n_needed:
            print(
                f"  ! depth {c.depth} has {c.n_items} items, under-powered for "
                f"{args.mde_pp}pp"
            )

    print()
    if verdict.usable:
        print("VERDICT: usable")
    else:
        print("VERDICT: UNUSABLE")
        for r in verdict.reasons:
            print(f"  - {r}")

    if args.json:
        args.json.write_text(json.dumps(verdict.to_dict(), indent=2))
        print(f"\nwrote {args.json}")
    return 0 if verdict.usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
