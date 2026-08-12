"""Associative recall in a long context: needle-in-a-haystack for GDN vs linear attention.

    torchrun --standalone --nproc-per-node=1 experiments/linear-attn-vs-gdn/eval_niah.py NAME \
        --checkpoint s3://.../stepNNNNN --eval-data shard1.npy,shard2.npy

WHY THIS EXISTS ALONGSIDE THE PERPLEXITY SWEEP. Cross-entropy over natural text is dominated by
local prediction: a model can hold a very good average while retaining nothing from 30k tokens
ago, because almost every token is predictable from its neighbours. Perplexity therefore cannot
separate a state-carrying recurrence that *remembers* from one that merely smooths. Retrieval can,
which is why the GDN-2 paper leans on needle-in-a-haystack rather than on its own loss numbers.

WHY THE NEEDLES ARE TOKEN IDS AND NOT ENGLISH SENTENCES. RULER's NIAH inserts text, which needs
an encoder. `TokenizerConfig` in this repository carries only metadata -- vocab size and special
ids, no encoder -- so building one means `AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")`,
a public-internet fetch from inside a run whose whole claim is that it read a sealed corpus, over
a network path nothing here establishes exists. So the needle is built in token space instead.

That makes this an ASSOCIATIVE-RECALL probe rather than a reproduction of RULER, and the numbers
here are NOT comparable to published RULER scores. They are comparable across our arms, which is
what the comparison needs. The capability under test is the same one: bind a value to a key, keep
it across a long span of distractor text, and produce it on demand.

WHAT IS MEASURED, AND WHY A CONTROL IS NOT OPTIONAL. For each item the model sees
``haystack ... MARKER_i VALUE_i ... haystack ... MARKER_q`` and must continue with ``VALUE_q``.
Reported per condition:

  * ``acc``     -- teacher-forced fraction of value tokens whose argmax is correct
  * ``exact``   -- fraction of items where EVERY value token is correct
  * ``ce``      -- mean cross-entropy on the value tokens
  * ``ce_ctrl`` -- the same, for a marker that was never inserted

``ce_ctrl`` is the load-bearing one. Value tokens are drawn from a small fixed pool, so a model
that has learnt nothing but the pool's marginal distribution already scores well above chance on
``acc``. The retrieval signal is ``ce_ctrl - ce``: how much better the model does when the answer
IS in the context than when it is not. Without it this measures a prior and calls it memory.

MARKERS ARE CHOSEN BY MEASURED RARITY. A marker that occurs naturally in the haystack creates a
second, spurious binding and depresses the score for a reason that has nothing to do with the
mixer. So the eval shards are counted first and the rarest ids become markers; their haystack
frequency is reported so a reader can see the collision risk rather than trust it.

CONDITIONS. Lengths span the training length and beyond (these models trained at 4096 with no
positional encoding, so past 4096 is extrapolation). Depth is where the needle sits as a fraction
of the context, which separates "forgot it" from "never encoded it". ``--n-keys`` > 1 is the
multi-key case, where several bindings compete for one fixed-size state -- the setting the GDN-2
paper reports its largest margins on, and the one a decoupled erase/write gate should help most.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import olmo_linear_attn  # noqa: E402,F401  (registers the linear_attention mixer)
import s3_io_robustness  # noqa: E402,F401
from eval_long_context import _load_model, _open_tokens  # noqa: E402  (shared loader)

from olmo_core.io import upload  # noqa: E402
from olmo_core.train import (  # noqa: E402
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.utils import get_default_device  # noqa: E402

log = logging.getLogger("eval_niah")

DEFAULT_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]


def pick_marker_ids(streams: List[np.ndarray], n_markers: int, vocab: int, sample: int = 4_000_000):
    """The rarest ids in the eval data, so a marker cannot collide with the haystack by accident.

    Returns ``(marker_ids, value_pool, freq)`` where ``freq`` maps each chosen id to how many
    times it occurred in the sample -- reported rather than assumed, because "rare" is a
    measurement and a marker that is merely *probably* rare is a confound nobody can see later.
    """
    counts = np.zeros(vocab, dtype=np.int64)
    for s in streams:
        take = min(len(s), sample // max(1, len(streams)))
        ids = np.asarray(s[:take], dtype=np.int64)
        counts += np.bincount(ids, minlength=vocab)[:vocab]
    # Skip id 0 and the very top of the range: specials and unused slots are rare for reasons
    # that have nothing to do with the corpus, and an unused id is out of distribution in a way
    # that would make the probe easier than the task it stands for.
    order = np.argsort(counts[1 : vocab - 8]) + 1
    markers = [int(i) for i in order[:n_markers]]
    # Values come from the NEXT band of rarity: distinct from markers, still uncommon, but real
    # tokens the model has seen. A value pool of unused ids would be trivially separable.
    pool = [int(i) for i in order[n_markers : n_markers + 32]]
    freq = {int(i): int(counts[i]) for i in markers + pool[:4]}
    return markers, pool, freq


def build_items(
    streams: List[np.ndarray],
    length: int,
    depth: float,
    n_keys: int,
    n_items: int,
    markers: List[int],
    pool: List[int],
    value_len: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Build ``n_items`` haystacks. Returns ``(tokens, value_spans, values)``.

    ``value_spans[i]`` is the ``(start, end)`` of the QUERIED value inside row ``i``, which is
    where the metrics are read; ``values[i]`` is the answer.
    """
    seqs, spans, answers = [], [], []
    for _ in range(n_items):
        s = streams[int(rng.integers(len(streams)))]
        if len(s) <= length + 8:
            continue
        off = int(rng.integers(0, len(s) - length - 8))
        row = np.asarray(s[off : off + length], dtype=np.int64).copy()

        # KEYS COME FROM markers[:-1]; THE LAST MARKER IS RESERVED FOR THE CONTROL ALONE.
        # Sampling keys from ALL markers meant the marker the control queries as "never
        # inserted" was in fact inserted in about half of K=4 items (4 chosen out of 8) and one
        # in eight K=1 items. In those, the control query hit a REAL binding to the WRONG value,
        # so ce_ctrl came out high and the retrieval gain was inflated -- upward, and worse for
        # larger K, which is exactly where the arms were reported to separate.
        keys = list(rng.choice(len(markers) - 1, size=n_keys, replace=False))
        vals = [
            np.asarray(rng.choice(pool, size=value_len, replace=True), dtype=np.int64) for _ in keys
        ]
        # Needles are spread around the requested depth so several keys do not sit on top of
        # each other; the queried one is placed AT the depth so `depth` means what it says.
        q = int(rng.integers(n_keys))
        unit = 1 + value_len
        base = int(depth * (length - unit * (n_keys + 2)))
        base = max(1, base)
        for j, (k, v) in enumerate(zip(keys, vals)):
            at = base + (j - q) * unit * 3
            at = max(1, min(at, length - unit * 3))
            row[at] = markers[k]
            row[at + 1 : at + 1 + value_len] = v
        # The query goes at the very end, so the answer is the model's continuation.
        qpos = length - unit
        row[qpos] = markers[keys[q]]
        row[qpos + 1 : qpos + 1 + value_len] = vals[q]
        seqs.append(row)
        spans.append((qpos + 1, qpos + 1 + value_len))
        answers.append(vals[q])
    if not seqs:
        return np.empty((0, length), dtype=np.int64), [], np.empty((0, value_len), dtype=np.int64)
    return np.stack(seqs), spans, np.stack(answers)


@torch.no_grad()
def score(model, tokens: np.ndarray, spans, answers: np.ndarray, device, micro: int = 1):
    """Teacher-forced accuracy, exact-match and CE over the queried value tokens only."""
    n_correct = n_total = n_exact = 0
    ce_sum = 0.0
    for i in range(0, len(tokens), micro):
        batch = torch.from_numpy(tokens[i : i + micro]).to(device)
        logits = model(input_ids=batch)  # (B, T, V) -- value spans are short, so this is safe
        if isinstance(logits, tuple):
            logits = logits[0]
        for b in range(batch.shape[0]):
            s, e = spans[i + b]
            # Position p predicts token p+1, so the logits for the value at [s, e) are at [s-1, e-1).
            lg = logits[b, s - 1 : e - 1].float()
            tgt = torch.from_numpy(answers[i + b]).to(device)
            pred = lg.argmax(-1)
            ok = pred == tgt
            n_correct += int(ok.sum().item())
            n_total += int(tgt.numel())
            n_exact += int(bool(ok.all().item()))
            ce_sum += float(torch.nn.functional.cross_entropy(lg, tgt, reduction="sum").item())
    return {
        "acc": n_correct / max(1, n_total),
        "exact": n_exact / max(1, len(tokens)),
        "ce": ce_sum / max(1, n_total),
        "n_items": len(tokens),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Associative-recall NIAH for recurrent mixers.")
    p.add_argument("run_name")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--lens", default=",".join(map(str, DEFAULT_LENS)))
    p.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    p.add_argument("--n-keys", default="1,4", help="Comma-separated key counts; >1 is multi-key.")
    p.add_argument("--items", type=int, default=64, help="Haystacks per condition.")
    p.add_argument("--value-len", type=int, default=4, help="Tokens in the value to be recalled.")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", default="/tmp/niah")
    p.add_argument("--output", default=None)
    p.add_argument("--upload-to", default=None)
    opts, _ = p.parse_known_args()

    prepare_training_environment()
    try:
        device = get_default_device()
        model, model_cfg = _load_model(opts.checkpoint, device, os.path.join(opts.work_dir, "ckpt"))
        vocab = model_cfg.vocab_size
        shards = [s.strip() for s in opts.eval_data.split(",") if s.strip()]
        streams = [_open_tokens(s, os.path.join(opts.work_dir, "data")) for s in shards]

        lens = [int(x) for x in opts.lens.split(",") if x]
        depths = [float(x) for x in opts.depths.split(",") if x]
        keycounts = [int(x) for x in opts.n_keys.split(",") if x]

        markers, pool, freq = pick_marker_ids(streams, n_markers=max(keycounts) + 4, vocab=vocab)
        log.info(f"markers {markers} (haystack counts {freq})")
        log.info(f"value pool of {len(pool)} ids, value_len={opts.value_len}")

        rows: List[Dict] = []
        for K in keycounts:
            for L in lens:
                for d in depths:
                    rng = np.random.default_rng(opts.seed)
                    toks, spans, answers = build_items(
                        streams, L, d, K, opts.items, markers, pool, opts.value_len, rng
                    )
                    if len(toks) == 0:
                        continue
                    got = score(model, toks, spans, answers, device, opts.micro_batch)

                    # CONTROL: identical haystack, but query a marker that was never inserted.
                    ctl = toks.copy()
                    unseen = markers[-1]
                    ctl[:, L - (1 + opts.value_len)] = unseen
                    ctl_got = score(model, ctl, spans, answers, device, opts.micro_batch)

                    row = {
                        "n_keys": K,
                        "len": L,
                        "depth": d,
                        **got,
                        "ce_ctrl": ctl_got["ce"],
                        "acc_ctrl": ctl_got["acc"],
                        "retrieval_gain": ctl_got["ce"] - got["ce"],
                    }
                    rows.append(row)
                    log.info(
                        f"  K={K} L={L:<6} d={d:<5} acc={got['acc']:.3f} exact={got['exact']:.3f} "
                        f"ce={got['ce']:.3f} ce_ctrl={ctl_got['ce']:.3f} "
                        f"gain={row['retrieval_gain']:+.3f}"
                    )

        # ---- summary LAST: only the final fifty log lines are readable -----------------------
        print()
        print("=" * 78)
        print(f"ASSOCIATIVE-RECALL NIAH -- {opts.run_name}")
        print(f"checkpoint {opts.checkpoint}")
        print("=" * 78)
        print(f"{'K':>3} {'len':>7} {'acc':>6} {'exact':>6} {'ce':>7} {'ce_ctrl':>8} {'gain':>7}")
        for K in keycounts:
            for L in lens:
                sub = [r for r in rows if r["n_keys"] == K and r["len"] == L]
                if not sub:
                    continue
                m = lambda k: sum(r[k] for r in sub) / len(sub)  # noqa: E731 -- depth-mean
                print(
                    f"{K:>3} {L:>7} {m('acc'):>6.3f} {m('exact'):>6.3f} {m('ce'):>7.3f} "
                    f"{m('ce_ctrl'):>8.3f} {m('retrieval_gain'):>+7.3f}"
                )
        print()
        print("gain = ce_ctrl - ce: how much the answer being IN context helps. Near zero means")
        print("no retrieval, whatever acc says -- acc alone is inflated by the value pool's prior.")
        print("Rows are averaged over depths; the JSON keeps every (K, len, depth) cell.")

        out = opts.output or os.path.join(opts.work_dir, f"niah_{opts.run_name}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "checkpoint": opts.checkpoint,
                    "markers": markers,
                    "marker_freq": freq,
                    "opts": vars(opts),
                    "rows": rows,
                },
                f,
                indent=2,
            )
        log.info(f"wrote {out}")
        if opts.upload_to:
            upload(out, opts.upload_to, save_overwrite=True)
        return 0
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    sys.exit(main())
