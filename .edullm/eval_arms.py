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
from eval_long_context import (  # noqa: E402
    BASE_LEN,
    _eval_length,
    _load_model,
    _open_tokens,
    _patch_head_for_chunked_ce,
    _windows,
)
from eval_niah import build_items, pick_marker_ids, score  # noqa: E402

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
    p.add_argument("--max-windows", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--ce-chunk", type=int, default=4096)
    # NIAH scores argmax over the vocabulary, so it needs FULL (B, T, V) logits where the length
    # sweep needs only per-token CE. At L=16384 against a 100,352 vocab that is 3.3 GiB in bf16
    # for a single sequence before anything is cast to fp32, which is why its ceiling is lower
    # than the sweep's and why it is capped here rather than sharing --seq-lens.
    p.add_argument("--niah-lens", default="1024,2048,4096,8192")
    p.add_argument("--niah-depths", default="0.1,0.5,0.9")
    p.add_argument("--niah-keys", default="1,4")
    p.add_argument("--niah-items", type=int, default=32)
    p.add_argument("--value-len", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", default="/tmp/evalarms")
    p.add_argument("--skip-niah", action="store_true")
    p.add_argument("--upload-to", default=None)
    opts, _ = p.parse_known_args()

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
        windows = {L: _windows(streams, L, opts.max_windows) for L in seq_lens}
        for L, w in windows.items():
            log.info(f"  L={L}: {len(w)} window(s)")

        phase(f"windows built for {len(windows)} length(s)")
        results: Dict[str, Any] = {}
        markers = pool = freq = None

        for name, ckpt in arms:
            phase(f"{name}: loading {ckpt}")
            model, model_cfg = _load_model(ckpt, device, os.path.join(opts.work_dir, name))
            phase(f"{name}: loaded, {model_cfg.num_params:,} params")
            if markers is None:
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
                            if len(toks) == 0:
                                continue
                            got = score(model, toks, spans, answers, device, opts.micro_batch)
                            ctl = toks.copy()
                            ctl[:, L - (1 + opts.value_len)] = markers[-1]
                            ctlr = score(model, ctl, spans, answers, device, opts.micro_batch)
                            arm["niah"].append(
                                {
                                    "n_keys": K,
                                    "len": L,
                                    "depth": d,
                                    **got,
                                    "ce_ctrl": ctlr["ce"],
                                    "gain": ctlr["ce"] - got["ce"],
                                }
                            )
                            log.info(
                                f"  niah K={K} L={L:<6} d={d:<4} acc={got['acc']:.3f} "
                                f"gain={ctlr['ce'] - got['ce']:+.3f}"
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
            print("ASSOCIATIVE RECALL -- acc, and retrieval gain (ce_ctrl - ce)")
            print("=" * 78)
            for K in niah_keys:
                print(f"  n_keys={K}")
                print(f"    {'arm':<16}" + "".join(f"{L:>14}" for L in niah_lens))
                for name in results:
                    row = f"    {name:<16}"
                    for L in niah_lens:
                        cells = [
                            c for c in results[name]["niah"] if c["n_keys"] == K and c["len"] == L
                        ]
                        if not cells:
                            row += f"{'-':>14}"
                        else:
                            a = sum(c["acc"] for c in cells) / len(cells)
                            g = sum(c["gain"] for c in cells) / len(cells)
                            row += f"  {a:.2f}/{g:+.2f}".rjust(14)
                    print(row)
            print("    cells are acc/gain, averaged over depths. gain near zero means NO")
            print("    retrieval however high acc looks -- acc alone rides the value prior.")

        print()
        print("READ THE RELATIVE TABLE FOR THE ARCHITECTURE COMPARISON, NOT THE ABSOLUTE ONE.")
        print("These arms do not share a training recipe: architecture, mixer geometry, sequence")
        print("length and token budget match, but optimizer, LR, batch size and z-loss do not.")
        print("A gap in absolute ppl is a gap between two trained models. The x-own-4096 column")
        print("and the retrieval gain are taken inside one model, so the recipe largely cancels.")

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
