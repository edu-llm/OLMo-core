"""Evaluate several trained mixers against one another: length sweep + associative recall.

    python .edullm/eval_arms.py "$EDULLM_RUN_ID" \
        --arms linear=s3://.../stage-linear/checkpoints/step12716,gdn=s3://.../stage-gdn/checkpoints/step12716 \
        --eval-data s3://.../stage-evaldata/arxiv/part-00-00000.npy,...

ONE PROCESS FOR EVERY ARM, WHICH IS A MEASUREMENT DECISION AND NOT A CONVENIENCE. Container
start, the S3 dataset open and the CUDA context are paid once and shared, so they cancel out of
the arm-vs-arm contrast instead of adding tens of seconds of unpaired jitter per arm. The A100
playbook's B8 records that this jitter was *larger than the effect* in a comparable setup. It is
also one queue wait and one machine instead of N.

WHAT IS HELD IDENTICAL ACROSS ARMS: the eval shards, the window construction, the length sweep,
the NIAH conditions, the marker ids and the seed. Everything that could move a number other than
the checkpoint itself is fixed before the loop starts, and the markers in particular are chosen
ONCE from the eval data rather than per arm, so no arm gets an easier needle.

TWO KINDS OF NUMBER COME OUT, AND THEY ARE NOT EQUALLY TRUSTWORTHY.

  * WITHIN-MODEL, safe to compare across arms: perplexity at length L *relative to that same
    model's own 4096 baseline* (its degradation curve), and the NIAH retrieval gain
    ``ce_ctrl - ce``. Both are ratios or differences taken inside one model, so a training-recipe
    difference between arms largely cancels.
  * ABSOLUTE, to be read with the recipe in hand: raw perplexity at 4096. The arms here were not
    all trained with the same optimizer, learning rate, batch size or z-loss -- GDN-2 in
    particular used adamw at 4e-4 with no z-loss and a 524,288 batch, where the linear and GDN
    arms used skip_step_adamw at ~7.7e-4 with z-loss 1e-5 and a 786,432 batch. Architecture,
    mixer geometry, sequence length and token budget DO match. So an absolute gap is a gap
    between two trained models, not attributable to the mixer alone, and the summary prints that
    warning next to the column rather than burying it in a document.

Nothing here trains anything, and no arm's weights are modified.
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "linear-attn-vs-gdn"
    ),
)

import olmo_linear_attn  # noqa: E402,F401  (registers the linear_attention mixer)
from _survive import survive  # noqa: E402
from eval_long_context import (  # noqa: E402
    BASE_LEN,
    _eval_length,
    _load_model,
    _open_tokens,
    _patch_head_for_chunked_ce,
    _windows,
)
from eval_niah import (  # noqa: E402
    build_items,
    build_items_english,
    load_assets,
    pick_marker_ids,
    score,
)

from olmo_core.io import upload  # noqa: E402
from olmo_core.train import (  # noqa: E402
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.utils import get_default_device  # noqa: E402

log = logging.getLogger("eval_arms")


def phase(msg: str) -> None:
    """Print a locatable marker and flush it.

    run_019ff4dc exited 1 with no traceback inside the fifty lines `edullm logs` returns -- the
    last thing it printed was the model build, so the failure could have been anywhere after it.
    A killed process (host OOM, for instance) leaves no traceback at all, so the only way to know
    where it stopped is to have said where it was. Flushed because a buffered marker is no marker.
    """
    print(f"[PHASE] {msg}", flush=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Compare trained sequence mixers.")
    p.add_argument("run_name", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    p.add_argument("--arms", required=True, help="name=checkpoint_uri, comma separated.")
    p.add_argument("--eval-data", required=True)
    p.add_argument("--seq-lens", default="4096,8192,16384,32768,65536")
    # TOKENS PER LENGTH, NOT WINDOWS PER LENGTH, AND THE DIFFERENCE BIASED THE LAST RUN.
    # Holding windows constant makes the token count GROW with L, so 4096 is the noisiest point
    # -- and it is the denominator of every x-own-4096 ratio, which pushed all of them low: my
    # GDN read 0.91x where the published run read 0.99x. The published harness held tokens at
    # ~2M and let the window count fall (256 windows at 4096 down to 30 at 65536); this now does
    # the same, so the ratios are comparable both to it and between lengths.
    p.add_argument("--tokens-per-len", type=int, default=2_000_000)
    p.add_argument("--max-windows", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--ce-chunk", type=int, default=4096)
    # NIAH scores argmax over the vocabulary, so it needs FULL (B, T, V) logits where the length
    # sweep needs only per-token CE. At L=16384 against a 100,352 vocab that is 3.3 GiB in bf16
    # for a single sequence before anything is cast to fp32, which is why its ceiling is lower
    # than the sweep's and why it is capped here rather than sharing --seq-lens.
    p.add_argument("--niah-lens", default="1024,2048,4096,8192")
    p.add_argument("--niah-depths", default="0.1,0.5,0.9")
    p.add_argument("--niah-keys", default="1,4")
    # 96 rather than 32: the last run's retrieval cells were single estimates over 32 items with
    # no repeats, and differences between GDN and GDN-2 sat inside that noise. Tripling the sample
    # is cheap here -- the whole NIAH pass was minutes -- and is the difference between "level"
    # and "indistinguishable at this sample size".
    p.add_argument("--niah-items", type=int, default=96)
    p.add_argument("--value-len", type=int, default=4)
    p.add_argument(
        "--needle-style",
        choices=["tokens", "english"],
        default="english",
        help="english uses pre-tokenized natural-language needles (see make_needle_assets.py) and "
        "is the default because the token-space task is too hard at this scale to discriminate: "
        "every arm sat near the floor, inside a measured noise band of 0.43 nats. tokens keeps the "
        "harder probe, which isolates state capacity from linguistic priors.",
    )
    p.add_argument("--value-kind", default="digits", choices=["digits", "uuid", "words"])
    p.add_argument(
        "--needle-assets",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "experiments",
            "linear-attn-vs-gdn",
            "needle_assets.json",
        ),
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", default="/tmp/evalarms")
    p.add_argument("--skip-niah", action="store_true")
    p.add_argument("--upload-to", default=None)
    opts, _ = p.parse_known_args()
    survive(f"eval_arms_{opts.run_name}")

    arms = []
    for spec in opts.arms.split(","):
        if not spec.strip():
            continue
        name, _, uri = spec.partition("=")
        if not uri:
            raise SystemExit(f"--arms entry {spec!r} must be name=checkpoint_uri")
        arms.append((name.strip(), uri.strip()))
    log.info(f"{len(arms)} arm(s): {[a for a, _ in arms]}")

    prepare_training_environment()
    try:
        device = get_default_device()
        shards = [s.strip() for s in opts.eval_data.split(",") if s.strip()]
        streams = [_open_tokens(s, os.path.join(opts.work_dir, "data")) for s in shards]
        log.info(f"{len(streams)} eval shard(s), sizes={[len(s) for s in streams]}")

        seq_lens = [int(x) for x in opts.seq_lens.split(",") if x]
        niah_lens = [int(x) for x in opts.niah_lens.split(",") if x]
        niah_depths = [float(x) for x in opts.niah_depths.split(",") if x]
        niah_keys = [int(x) for x in opts.niah_keys.split(",") if x]

        # WINDOWS AND MARKERS ARE BUILT ONCE, BEFORE ANY MODEL LOADS. Every arm then sees
        # byte-identical evaluation inputs; nothing about arm order can change what is scored.
        windows = {}
        for L in seq_lens:
            n = max(2, min(opts.max_windows, opts.tokens_per_len // L))
            windows[L] = _windows(streams, L, n)
            log.info(f"  L={L}: {len(windows[L])} window(s) -> {len(windows[L]) * L:,} tokens")

        phase(f"windows built for {len(windows)} length(s)")
        assets = None
        if opts.needle_style == "english":
            assets = load_assets(opts.needle_assets)
            n_items = {k: len(v) for k, v in assets["items"].items()}
            phase(f"english needles from {assets['tokenizer']}, items per kind {n_items}")

        results: Dict[str, Any] = {}
        markers = pool = freq = None

        for name, ckpt in arms:
            phase(f"{name}: loading {ckpt}")
            model, model_cfg = _load_model(ckpt, device, os.path.join(opts.work_dir, name))
            phase(f"{name}: loaded, {model_cfg.num_params:,} params")
            if markers is None and opts.needle_style == "tokens":
                phase("choosing needle markers by measured rarity")
                markers, pool, freq = pick_marker_ids(
                    streams, n_markers=max(niah_keys) + 4, vocab=model_cfg.vocab_size
                )
                log.info(f"markers (shared by all arms) {markers}, haystack counts {freq}")

            arm: Dict[str, Any] = {
                "checkpoint": ckpt,
                "params": model_cfg.num_params,
                "non_embedding_params": model_cfg.num_non_embedding_params,
                "lengths": {},
                "niah": [],
            }

            # ---- NIAH first, while the LM head is still intact (it needs full logits) --------
            if not opts.skip_niah:
                phase(f"{name}: NIAH begins")
                for K in niah_keys:
                    for L in niah_lens:
                        for d in niah_depths:
                            rng = np.random.default_rng(opts.seed)
                            if opts.needle_style == "english":
                                toks, spans, answers, ctrl_qids = build_items_english(
                                    streams,
                                    L,
                                    d,
                                    K,
                                    opts.niah_items,
                                    assets,
                                    opts.value_kind,
                                    rng,
                                )
                            else:
                                toks, spans, answers = build_items(
                                    streams,
                                    L,
                                    d,
                                    K,
                                    opts.niah_items,
                                    markers,
                                    pool,
                                    opts.value_len,
                                    rng,
                                )
                                ctrl_qids = None
                            if len(toks) == 0:
                                continue
                            got = score(model, toks, spans, answers, device, opts.micro_batch)

                            # THE CONTROL: the same row with the query's KEY replaced by one never
                            # planted, and nothing else touched -- same haystack, same needles,
                            # same teacher-forced answer, same targets, same length.
                            ctl = toks.copy()
                            if ctrl_qids is not None:
                                for i, (sp, _e) in enumerate(spans):
                                    q = ctrl_qids[i]
                                    ctl[i, sp - len(q) : sp] = q
                            else:
                                # markers[-1] is reserved; build_items samples keys from
                                # markers[:-1] so it is never inserted as a needle.
                                ctl[:, L - (1 + opts.value_len)] = markers[-1]
                            ctlr = score(model, ctl, spans, answers, device, opts.micro_batch)

                            # PAIRED BOOTSTRAP over items. gain is a difference on the same items,
                            # so its uncertainty has to be resampled over items; an aggregate
                            # cannot say whether a 0.2-nat gap is real. The last run had no
                            # interval at all and I reported differences that turned out to sit
                            # inside a 0.43-nat noise band.
                            lo = hi = boot = None
                            a_ce = np.asarray(got.get("per_item_ce") or [], dtype=np.float64)
                            b_ce = np.asarray(ctlr.get("per_item_ce") or [], dtype=np.float64)
                            if opts.bootstrap and len(a_ce) == len(b_ce) and len(a_ce) > 1:
                                diff = b_ce - a_ce
                                brng = np.random.default_rng(opts.seed + 1)
                                idx = brng.integers(0, len(diff), size=(opts.bootstrap, len(diff)))
                                means = diff[idx].mean(axis=1)
                                lo, hi = (
                                    float(np.percentile(means, 2.5)),
                                    float(np.percentile(means, 97.5)),
                                )
                                # THE INTERVAL'S OWN CENTRE, RECORDED SEPARATELY. `gain` is a
                                # difference of TOKEN-weighted means and this is a mean of
                                # ITEM-weighted ones, so they coincide only when every answer has
                                # the same token count. They nearly do here (4-6 tokens), but a
                                # silent divergence between an estimate and the interval drawn
                                # around it is the kind of thing that invalidates a table without
                                # looking wrong, so both are written down and can be compared.
                                boot = float(diff.mean())
                            elif opts.bootstrap:
                                # LOUD, BECAUSE THE LAST RUN FAILED HERE IN SILENCE. per_item_ce
                                # came back empty (nothing appended to it), this guard fell
                                # through, and the summary printed a "gain [lo,hi]" header over
                                # bare gains for all 90 cells. A skipped interval is now a line in
                                # the log rather than an absence nobody can see.
                                log.warning(
                                    f"  no CI at K={K} L={L} d={d}: per-item CE lengths "
                                    f"{len(a_ce)} and {len(b_ce)}"
                                )
                            # per_item_ce is dropped from the record: it is what the bootstrap
                            # above consumed, and keeping thousands of floats per cell would
                            # bloat the results JSON without adding anything a reader needs.
                            rec = {k: v for k, v in got.items() if k != "per_item_ce"}
                            arm["niah"].append(
                                {
                                    "n_keys": K,
                                    "len": L,
                                    "depth": d,
                                    **rec,
                                    "ce_ctrl": ctlr["ce"],
                                    "gain": ctlr["ce"] - got["ce"],
                                    "gain_ci95": [lo, hi],
                                    "gain_boot": boot,
                                }
                            )
                            ci = f" [{lo:+.2f},{hi:+.2f}]" if lo is not None else ""
                            log.info(
                                f"  niah K={K} L={L:<6} d={d:<4} acc={got['acc']:.3f} "
                                f"gain={ctlr['ce'] - got['ce']:+.3f}{ci}"
                            )

            # ---- length sweep. The head is patched for chunked CE and cannot give logits
            # afterwards, which is why NIAH runs above rather than below. --------------------
            phase(f"{name}: NIAH done; length sweep begins")
            _patch_head_for_chunked_ce(model, chunk=opts.ce_chunk)
            for L in seq_lens:
                w = windows[L]
                if len(w) == 0:
                    continue
                try:
                    ce, ntok, buckets = _eval_length(model, w, device, opts.micro_batch)
                except torch.cuda.OutOfMemoryError:
                    arm["lengths"][str(L)] = {"oom": True}
                    log.info(f"  len L={L:<6} OOM")
                    continue
                arm["lengths"][str(L)] = {
                    "ce": ce,
                    "ppl": math.exp(min(ce, 20.0)),
                    "tokens": ntok,
                    "position_buckets": {str(k): v for k, v in buckets.items()},
                    "oom": False,
                }
                log.info(f"  len L={L:<6} ce={ce:.4f} ppl={math.exp(min(ce, 20.0)):.3f}")

            phase(f"{name}: complete")
            results[name] = arm
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # ---- summary LAST: `edullm logs` shows only the final fifty lines -------------------
        print()
        print("=" * 78)
        print("LENGTH SWEEP -- ppl, and ppl relative to each model's OWN 4096 baseline")
        print("=" * 78)
        hdr = f"{'arm':<16}" + "".join(f"{L:>12}" for L in seq_lens)
        print(hdr)
        for name in results:
            row = f"{name:<16}"
            for L in seq_lens:
                cell = results[name]["lengths"].get(str(L))
                row += f"{'OOM':>12}" if (not cell or cell.get("oom")) else f"{cell['ppl']:>12.2f}"
            print(row)
        print()
        print(f"{'arm':<16}" + "".join(f"{L:>12}" for L in seq_lens) + "   (x own 4096)")
        for name in results:
            base = results[name]["lengths"].get(str(BASE_LEN))
            row = f"{name:<16}"
            for L in seq_lens:
                cell = results[name]["lengths"].get(str(L))
                if not cell or cell.get("oom") or not base or base.get("oom"):
                    row += f"{'-':>12}"
                else:
                    row += f"{cell['ppl'] / base['ppl']:>11.2f}x"
            print(row)

        if not opts.skip_niah:
            print()
            print("=" * 78)
            print("ASSOCIATIVE RECALL -- gain (ce_ctrl - ce) with 95% bootstrap CI, PER DEPTH")
            print("=" * 78)
            print("A cell is gain [lo,hi]. The CI is a paired bootstrap over items, so an")
            print("interval spanning zero means NO retrieval was demonstrated at that cell.")
            print("NOT averaged over depths: depth is the best-resolved axis in this eval and")
            print("averaging it away turns a monotone effect into one uninterpretable number.")
            for K in niah_keys:
                for d in niah_depths:
                    print(f"\n  n_keys={K}  depth={d}")
                    print(f"    {'arm':<10}" + "".join(f"{L:>22}" for L in niah_lens))
                    for name in results:
                        row = f"    {name:<10}"
                        for L in niah_lens:
                            c = next(
                                (
                                    x
                                    for x in results[name]["niah"]
                                    if x["n_keys"] == K and x["len"] == L and x["depth"] == d
                                ),
                                None,
                            )
                            if not c:
                                row += f"{'-':>22}"
                            else:
                                lo_, hi_ = c.get("gain_ci95") or [None, None]
                                cell = (
                                    f"{c['gain']:+.2f}"
                                    if lo_ is None
                                    else f"{c['gain']:+.2f}[{lo_:+.2f},{hi_:+.2f}]"
                                )
                                row += f"{cell:>22}"
                        print(row)
            print("\n  accuracy, same layout (fraction of answer tokens whose argmax is right)")
            for K in niah_keys:
                print(f"    n_keys={K}")
                for name in results:
                    row = f"      {name:<10}"
                    for L in niah_lens:
                        cs = [
                            x for x in results[name]["niah"] if x["n_keys"] == K and x["len"] == L
                        ]
                        row += (
                            f"{'-':>10}"
                            if not cs
                            else f"{sum(c['acc'] for c in cs) / len(cs):>10.3f}"
                        )
                    print(row)

        print()
        print("READ THE RELATIVE TABLE FOR THE ARCHITECTURE COMPARISON, NOT THE ABSOLUTE ONE.")
        print("These arms do not share a training recipe: architecture, mixer geometry, sequence")
        print("length and token budget match, but optimizer, LR, batch size and z-loss do not.")
        print("A gap in absolute ppl is a gap between two trained models. The x-own-4096 column")
        print("and the retrieval gain are taken inside one model, so the recipe largely cancels.")

        # ---- DIGEST, PRINTED LAST, BECAUSE `edullm logs` SHOWS FIFTY LINES. -----------------
        # The tables above are about sixty and the model repr each arm prints is longer than the
        # whole window, so on the previous run every number had scrolled out of reach and the
        # results were recoverable only by reading the raw stream. Whatever a reader most needs
        # has to be the last thing printed, and short enough to survive on its own.
        print()
        print("=" * 78)
        print("DIGEST (last, so it survives the 50-line `edullm logs` window)")
        print("=" * 78)
        for name in results:
            # arm["lengths"] is a dict keyed by str(L), and a cell that ran out of memory carries
            # only {"oom": True} -- so filter on a present ppl rather than assuming every
            # requested length produced one.
            sweep = {
                int(k): v
                for k, v in results[name]["lengths"].items()
                if not v.get("oom") and v.get("ppl") is not None
            }
            if sweep:
                lo_l, hi_l = min(sweep), max(sweep)
                ratio = sweep[hi_l]["ppl"] / sweep[lo_l]["ppl"]
                print(
                    f"  {name:<8} ppl {sweep[lo_l]['ppl']:.2f} @{lo_l} -> "
                    f"{sweep[hi_l]['ppl']:.2f} @{hi_l}  ({ratio:.2f}x own baseline)"
                )
            # The deepest, longest cell that still demonstrates retrieval: the CI must clear zero,
            # so a cell survives here only if the interval says it did rather than the point
            # estimate looking positive.
            alive = [
                c
                for c in results[name]["niah"]
                if (c.get("gain_ci95") or [None])[0] is not None and c["gain_ci95"][0] > 0
            ]
            if alive:
                best = max(alive, key=lambda c: (c["len"], c["depth"]))
                print(
                    f"  {name:<8} retrieval holds to L={best['len']} "
                    f"(K={best['n_keys']}, depth={best['depth']}, gain {best['gain']:+.2f} "
                    f"CI [{best['gain_ci95'][0]:+.2f},{best['gain_ci95'][1]:+.2f}])"
                )
            else:
                print(f"  {name:<8} retrieval: NO cell had a CI clear of zero")

        out = os.path.join(opts.work_dir, f"eval_arms_{opts.run_name}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"opts": vars(opts), "markers": markers, "results": results}, f, indent=2)
        log.info(f"wrote {out}")
        if opts.upload_to:
            upload(out, opts.upload_to, save_overwrite=True)
        return 0
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    sys.exit(main())
