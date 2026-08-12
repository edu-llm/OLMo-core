#!/usr/bin/env python3
"""Materialise the n-hop corpus: one token stream, four loss-weight sidecars.

    python scripts/build_corpus.py --out data/nhop_v1 --n-entities 10000 \
        --total-tokens 500_000_000

Writes:

    tokens.bin              uint16, shared by every arm
    weights.{cond}.bin      uint8, one per condition, same length
    organizer.jsonl         the external store
    eval/{name}.jsonl       held-out evaluation strata
    report.json             lane shares, exposures, integrity gates

Aborts on any failed gate. Two of them exist because of specific prior failures:

* **Lane shares.** A finite lane can exhaust and have its budget silently
  reallocated while the total token count stays exact and every other check
  passes -- measured once at a fact lane requesting 50% and realising **3.75%**,
  with web text absorbing 46.3 points. Deep composition lanes are the most
  exhaustible thing here, so realised shares are asserted to 1% and the build
  dies rather than quietly shipping a corpus with no depth-5 items.
* **Population leakage.** Held-out entities must never appear in a composition
  document, and held-out depths must never appear in training at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memsplit import bios, masking, nhop  # noqa: E402
from memsplit.records import spans_from_roles  # noqa: E402
from memsplit.store import Organizer  # noqa: E402
from memsplit.tokenizer import get_tok, require_production_tokenizer  # noqa: E402


def _cycle(seq):
    while True:
        for x in seq:
            yield x


def build(args) -> dict:
    out = Path(args.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    tok = get_tok()
    if not args.allow_fallback:
        require_production_tokenizer(tok)

    train_depths = tuple(args.train_depths)
    eval_depths = tuple(args.eval_depths)
    max_depth = max(train_depths + eval_depths)

    # ---- entities, pools, graph -------------------------------------------
    train_pools, novel_pools = bios.split_pools(args.seed, args.novel_frac)
    recs = bios.generate_records(args.n_entities, seed=args.seed, pools=train_pools)
    graph = nhop.build_graph(recs, n_layers=max_depth + 2, seed=args.seed)
    by_id = {r.entity_id: r for r in recs}

    n_comp = int(round(args.n_entities * (1.0 - args.held_frac)))
    P_comp = [r.entity_id for r in recs[:n_comp]]
    P_held = [r.entity_id for r in recs[n_comp:]]
    comp_set, held_set = set(P_comp), set(P_held)

    # Composition starts must be depth-eligible AND in P_comp. Every depth draws
    # from this one pool so depth stays orthogonal to entity identity.
    eligible = [e for e in nhop.eligible_starts(graph, max_depth) if e in comp_set]
    held_eligible = [e for e in nhop.eligible_starts(graph, max_depth) if e in held_set]
    if len(eligible) < 50:
        raise SystemExit(
            f"only {len(eligible)} depth-{max_depth} eligible P_comp entities; "
            "raise --n-entities"
        )

    # ---- the store -------------------------------------------------------
    org = Organizer()
    for r in recs:
        for attr in bios.ATTRIBUTES:
            org.add(r.name, attr, r.attrs[attr])
        for rel, tgt in graph.edges[r.entity_id].items():
            org.add(r.name, rel, by_id[tgt].name)
    org.save(out / "organizer.jsonl")

    # ---- document streams (infinite, exposure-cycling) --------------------
    def atomic_stream():
        for exposure in _cycle(range(1 << 30)):
            for r in recs:
                for attr in bios.ATTRIBUTES:
                    yield nhop.render_atomic_doc(r, attr, exposure)

    def bridge_stream():
        for exposure in _cycle(range(1 << 30)):
            for r in recs:
                for rel, tgt in sorted(graph.edges[r.entity_id].items()):
                    yield nhop.render_bridge_doc(r, rel, by_id[tgt], exposure)

    def compose_stream():
        # Depth cycles FASTEST. Deeper documents are longer, so a token-budgeted
        # lane that looped depth-outermost would truncate mid-cycle and leave ~2x
        # as many depth-1 documents as depth-3 -- and a depth curve would then
        # show decline partly from exposure imbalance rather than from difficulty,
        # which is exactly the confound the depth axis exists to avoid.
        for exposure in _cycle(range(1 << 30)):
            for eid in eligible:
                for attr in bios.ATTRIBUTES:
                    for depth in train_depths:
                        got = nhop.sample_item(
                            graph, by_id, eid, depth, attr, seed=args.seed
                        )
                        if not got:
                            continue
                        chain, end, value = got
                        yield nhop.render_doc(
                            graph, by_id, eid, chain, attr, value, exposure
                        )

    lanes = {
        "atomic": (atomic_stream(), args.atomic_share),
        "bridge": (bridge_stream(), args.bridge_share),
        "compose": (compose_stream(), args.compose_share),
    }
    budgets = {k: int(round(args.total_tokens * s)) for k, (_, s) in lanes.items()}

    # ---- encode into one stream ------------------------------------------
    stream: list[int] = []
    side: dict[str, list[int]] = {c: [] for c in masking.CONDITIONS}
    emitted = Counter()
    doc_counts = Counter()
    exposures: Counter = Counter()
    depth_docs = Counter()
    diags: list[dict] = []
    skipped = 0

    while any(emitted[k] < budgets[k] for k in lanes):
        # Largest relative deficit first, deterministic tie-break.
        lane = max(
            sorted(lanes),
            key=lambda k: (budgets[k] - emitted[k]) / max(budgets[k], 1),
        )
        if emitted[lane] >= budgets[lane]:
            break
        doc = next(lanes[lane][0])
        ids, spans = spans_from_roles(tok, doc.segments, doc.roles)
        try:
            plan = masking.derive_weights(
                spans, len(ids), seed=len(stream), strict=True,
                mask_restatements=args.mask_restatements,
            )
        except masking.ControlUndersupply:
            skipped += 1
            continue
        stream.extend(ids)
        for cond in masking.CONDITIONS:
            side[cond].extend(plan.weights[cond].tolist())
        diags.append(plan.diagnostics)
        emitted[lane] += len(ids)
        doc_counts[lane] += 1
        if lane == "compose":
            depth_docs[doc.meta["depth"]] += 1
        if lane in ("atomic", "bridge"):
            exposures[(lane, doc.meta["entity_id"])] += 1

    n = len(stream)
    np.array(stream, dtype=np.uint16).tofile(out / "tokens.bin")
    for cond in masking.CONDITIONS:
        assert len(side[cond]) == n
        np.array(side[cond], dtype=np.uint8).tofile(out / f"weights.{cond}.bin")

    # ---- evaluation strata ------------------------------------------------
    novel = bios.generate_records(
        args.n_eval, seed=args.seed + 7777, pools=novel_pools, name_offset=311
    )
    novel_graph = nhop.build_graph(novel, n_layers=max_depth + 2, seed=args.seed + 7777)
    novel_by_id = {r.entity_id: r for r in novel}
    novel_eligible = nhop.eligible_starts(novel_graph, max_depth)

    def _items(g, bid, starts, depth, population, cap):
        got_items = []
        for eid in starts:
            for attr in bios.ATTRIBUTES:
                s = nhop.sample_item(g, bid, eid, depth, attr, seed=args.seed)
                if s:
                    chain, end, value = s
                    got_items.append(
                        nhop.make_item(g, bid, eid, chain, attr, value, population)
                    )
                if len(got_items) >= cap:
                    return got_items
        return got_items

    strata: dict[str, list] = {}
    for depth in sorted(set(train_depths + eval_depths)):
        tag = "trained" if depth in train_depths else "heldout"
        strata[f"comp_d{depth}_{tag}"] = _items(
            graph, by_id, eligible, depth, "comp", args.n_eval
        )
        strata[f"held_d{depth}_{tag}"] = _items(
            graph, by_id, held_eligible, depth, "held", args.n_eval
        )
        strata[f"novel_d{depth}_{tag}"] = _items(
            novel_graph, novel_by_id, novel_eligible, depth, "novel", args.n_eval
        )
    for name, items in strata.items():
        with open(out / "eval" / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps({
                    "task": it.task, "prompt": it.prompt,
                    "answer": it.answer, "meta": it.meta,
                }) + "\n")

    novel_org = Organizer()
    for r in novel:
        for attr in bios.ATTRIBUTES:
            novel_org.add(r.name, attr, r.attrs[attr])
        for rel, tgt in novel_graph.edges[r.entity_id].items():
            novel_org.add(r.name, rel, novel_by_id[tgt].name)
    novel_org.save(out / "organizer_novel.jsonl")

    # ---- gates -----------------------------------------------------------
    realised = {k: emitted[k] / n for k in lanes}
    share_ok = {
        k: abs(realised[k] - lanes[k][1]) <= args.share_tol for k in lanes
    }
    atomic_exp = [v for (lane, _), v in exposures.items() if lane == "atomic"]
    z = {c: int((np.array(side[c], dtype=np.uint8) == 0).sum()) for c in masking.CONDITIONS}

    train_names = {r.name for r in recs}
    novel_names = {r.name for r in novel}
    gates = {
        "populations_disjoint": not (comp_set & held_set),
        "lane_shares_within_tol": all(share_ok.values()),
        "all_train_depths_present": all(depth_docs[d] > 0 for d in train_depths),
        # Equal DOCUMENTS per trained depth, not equal tokens. Otherwise the
        # depth axis is partly an exposure axis.
        "compose_depths_balanced": (
            (max(depth_docs[d] for d in train_depths)
             - min(depth_docs[d] for d in train_depths))
            / max(max(depth_docs[d] for d in train_depths), 1) <= args.depth_tol
        ),
        "no_heldout_depth_in_training": all(depth_docs[d] == 0 for d in eval_depths),
        "equal_mass_contig": z["random_contig"] == z["split"],
        "equal_mass_scatter": z["random_scatter"] == z["split"],
        "dense_supervises_all": z["dense"] == 0,
        "novel_names_disjoint": not (train_names & novel_names),
        "novel_values_disjoint": not (
            {v for r in recs for v in r.attrs.values()}
            & {v for r in novel for v in r.attrs.values()}
        ),
        "organizer_covers_all_eval_hops": all(
            k in org or k in novel_org
            for items in strata.values() for it in items for k in it.meta["hop_keys"]
        ),
        "skip_rate_ok": skipped / max(skipped + sum(doc_counts.values()), 1) < 0.25,
    }

    report = {
        "config": {k: v for k, v in vars(args).items() if k != "func"},
        "tokenizer": tok.name,
        "n_tokens": n,
        "n_docs": dict(doc_counts),
        "skipped_docs": skipped,
        "lane_requested": {k: lanes[k][1] for k in lanes},
        "lane_realised": realised,
        "compose_docs_by_depth": {str(k): v for k, v in sorted(depth_docs.items())},
        "atomic_exposures_per_entity_attr": {
            "mean": float(np.mean(atomic_exp)) if atomic_exp else 0.0,
            "min": int(np.min(atomic_exp)) if atomic_exp else 0,
            "max": int(np.max(atomic_exp)) if atomic_exp else 0,
        },
        "populations": {"P_comp": len(P_comp), "P_held": len(P_held),
                        "eligible_comp": len(eligible),
                        "eligible_held": len(held_eligible)},
        "bits_per_entity": bios.bits_per_entity(train_pools),
        "pool_sizes": bios.pool_sizes(train_pools),
        "masked_tokens": z,
        "mask_report": masking.aggregate_report(diags),
        "eval_strata": {k: len(v) for k, v in strata.items()},
        "organizer_size": len(org),
        "organizer_novel_size": len(novel_org),
        "gates": gates,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    for name, ok in gates.items():
        print(f"  gate {name}: {'OK' if ok else 'FAIL'}")
    if not all(gates.values()):
        raise SystemExit("integrity gates FAILED; see report.json")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-entities", type=int, default=10_000)
    ap.add_argument("--total-tokens", type=int, default=200_000_000)
    ap.add_argument("--train-depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eval-depths", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--held-frac", type=float, default=0.2)
    ap.add_argument("--novel-frac", type=float, default=0.5)
    ap.add_argument("--atomic-share", type=float, default=0.35)
    ap.add_argument("--bridge-share", type=float, default=0.15)
    ap.add_argument("--compose-share", type=float, default=0.50)
    ap.add_argument("--share-tol", type=float, default=0.01)
    ap.add_argument("--depth-tol", type=float, default=0.05,
                    help="max relative spread in documents per trained depth")
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-restatements", action="store_true")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="permit the byte tokenizer (shape validation only)")
    args = ap.parse_args()

    total = args.atomic_share + args.bridge_share + args.compose_share
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"lane shares must sum to 1.0, got {total}")

    rep = build(args)
    print()
    print(f"tokens        : {rep['n_tokens']:,}")
    print(f"docs          : {rep['n_docs']}")
    print(f"lane realised : "
          + ", ".join(f"{k} {v:.3f}" for k, v in rep["lane_realised"].items()))
    print(f"compose depths: {rep['compose_docs_by_depth']}")
    print(f"atomic exposures/fact: {rep['atomic_exposures_per_entity_attr']}")
    print(f"masked split  : {rep['masked_tokens']['split']:,} "
          f"({rep['mask_report']['masked_token_frac_split']:.4f})")
    print(f"eval strata   : {len(rep['eval_strata'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
