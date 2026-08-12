"""
Long-context evaluation: sweep INCREASING sequence lengths on a trained checkpoint
to locate the point at which the model *breaks* for sequence length.

This is the read-out for the plain-linear-attention vs Gated-DeltaNet experiment.
Every model here was trained at seq-len 4096 with a recurrent (state-carrying)
sequence mixer and **no positional encoding** (neither ``LinearAttention`` nor
``GatedDeltaNet`` applies RoPE -- position is carried purely by the recurrence and
the short causal conv). So the degradation this probe measures is genuine
length-generalization / state-capacity behaviour, not a positional-encoding artifact.

What it does
------------
For each evaluation length ``L`` in an increasing sweep (default 4096 -> 262144),
it packs a held-out token stream into non-overlapping windows of length ``L``,
runs the model, and reports the mean next-token cross-entropy and perplexity over
that length. The *breaking point* is the ``L`` at which perplexity turns sharply
upward away from the L=4096 (training-length) baseline.

It additionally records a **loss-vs-absolute-position** curve (bucketed every 4096
tokens) so you can see not just *which* length breaks but *where inside* the
context the model starts to fall apart.

Memory
------
Materializing full logits over a long sequence would be enormous
(``L x vocab`` -- 26 GB at L=131072). The LM head is therefore monkeypatched
on the instance to compute cross-entropy in position-chunks against the tied
output embedding, bounding peak memory to ``chunk x vocab`` regardless of ``L``.
Nothing in OLMo-core ``src/`` is modified -- this is an additive experiment script.

Run (one GPU, under torchrun so the distributed checkpoint loader is happy)::

    CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc-per-node=1 \
        experiments/linear-attn-vs-gdn/eval_long_context.py linear-attn-370m-10b \
        --checkpoint s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/linear/stepNNNNN \
        --eval-data s3://.../arxiv/part-00-00000.npy,s3://.../wikipedia/part-00-00000.npy \
        --work-dir /mnt/nvme/olmo-work-linear
"""

import argparse
import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Make sibling `olmo_linear_attn` importable so the serialized LinearAttentionConfig
# ("_CLASS_": "olmo_linear_attn.LinearAttentionConfig") resolves during from_dict.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import olmo_linear_attn  # noqa: E402,F401  (registers the linear_attention mixer)
import s3_io_robustness  # noqa: E402,F401  (resilient S3 sizing; additive)

from olmo_core.data.utils import get_labels  # noqa: E402
from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402
from olmo_core.io import copy_file, get_bytes_range, get_file_size, upload  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.train import (  # noqa: E402
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.utils import get_default_device  # noqa: E402

log = logging.getLogger("eval_long_context")

BASE_LEN = 4096  # training sequence length; also the position-bucket granularity
IGNORE = -100
DEFAULT_SEQ_LENS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def _open_tokens(path: str, cache_dir: str) -> np.ndarray:
    """
    Memory-map a tokenized shard. Handles both real ``.npy`` (with header) and raw
    ``uint32`` dumps. Remote (s3://) paths are downloaded to ``cache_dir`` first.
    """
    local = path
    if "://" in path:
        os.makedirs(cache_dir, exist_ok=True)
        local = os.path.join(cache_dir, path.replace("://", "_").replace("/", "_"))
        if not os.path.exists(local):
            # copy_file rather than `aws s3 cp`, for the reason given on _load_model below:
            # the platform image carries no AWS CLI, and a subprocess to a binary that is not
            # there fails after the machine is billed.
            log.info(f"downloading {path} -> {local}")
            copy_file(path, local)
    with open(local, "rb") as f:
        magic = f.read(6)
    if magic == b"\x93NUMPY":
        return np.load(local, mmap_mode="r")
    # Raw dump: dolma2 padded vocab (100352) > 2**16, so tokens are uint32.
    return np.memmap(local, dtype=np.uint32, mode="r")


def _windows(streams: List[np.ndarray], seq_len: int, max_windows: int) -> np.ndarray:
    """
    Non-overlapping length-``seq_len`` windows drawn round-robin across shards (so a
    single dominant shard does not monopolize a length). Returns ``(n, seq_len)`` int64.
    """
    out: List[np.ndarray] = []
    offsets = [0] * len(streams)
    exhausted = [False] * len(streams)
    while len(out) < max_windows and not all(exhausted):
        for i, s in enumerate(streams):
            if len(out) >= max_windows:
                break
            o = offsets[i]
            if o + seq_len <= len(s):
                out.append(np.asarray(s[o : o + seq_len], dtype=np.int64))
                offsets[i] = o + seq_len
            else:
                exhausted[i] = True
    if not out:
        return np.empty((0, seq_len), dtype=np.int64)
    return np.stack(out, axis=0)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def _load_model(checkpoint: str, device: torch.device, cache_dir: str):
    """Rebuild the model from the checkpoint's serialized config and load weights in-place."""
    os.makedirs(cache_dir, exist_ok=True)

    # THROUGH olmo_core.io, NOT `aws s3 cp`, AND THIS IS A PORTING FIX RATHER THAN A STYLE ONE.
    # This harness was written for a box that had the AWS CLI and its own credentials. The
    # platform image has neither: `.edullm/Dockerfile` installs no awscli, so a subprocess call
    # to `aws` dies with FileNotFoundError after the instance is billed -- and AGENTS.md forbids
    # reaching AWS that way regardless. `get_bytes_range` is the same layer the checkpoint loader
    # below already uses, so if it can read the weights it can read the config beside them.
    cfg_uri = f"{checkpoint.rstrip('/')}/config.json"
    full_cfg = json.loads(get_bytes_range(cfg_uri, 0, get_file_size(cfg_uri)).decode("utf-8"))

    model_cfg = TransformerConfig.from_dict(full_cfg["model"])
    model = model_cfg.build(init_device=str(device))
    model.eval()

    # Load the fp32 checkpoint into the fp32 model, THEN cast the whole model to
    # bf16. Training ran under FSDP with param_dtype=bfloat16 (uniform bf16 compute),
    # NOT autocast -- and the fla chunked-scan kernel asserts both dot operands share
    # a dtype. Casting the model to bf16 reproduces the training compute dtype exactly
    # and keeps q/k/v uniformly bf16 through the kernel. (Weights are tied, so a single
    # .to() casts embeddings and the LM head together and they stay tied.)
    load_model_and_optim_state(f"{checkpoint.rstrip('/')}/model_and_optim", model)
    model = model.to(torch.bfloat16)
    log.info(
        f"loaded {model_cfg.num_params:,} params "
        f"({model_cfg.num_non_embedding_params:,} non-embedding) from {checkpoint}"
    )
    return model, model_cfg


def _patch_head_for_chunked_ce(model, chunk: int = 4096):
    """
    Replace the LM head's forward with a memory-bounded, per-token CE computation.
    Returns per-token cross-entropy of shape (B, T) in float32 (0 at ignored positions).
    Reuses the head's own norm + tied output weight, so numerics match the real head.
    """
    head = model.lm_head
    W = head.w_out.weight  # (vocab, d_model), tied to input embeddings
    vocab = head.vocab_size

    def chunked_forward(x, *, labels=None, ignore_index=IGNORE, **_kw):
        assert labels is not None
        h = head.norm(x) if head.norm is not None else x
        B, T, D = h.shape
        h = h.reshape(-1, D)
        lbl = labels.reshape(-1)
        ce = torch.empty(h.shape[0], dtype=torch.float32, device=h.device)
        for i in range(0, h.shape[0], chunk):
            hs = h[i : i + chunk]
            logits = F.linear(hs, W).float()
            ce[i : i + hs.shape[0]] = F.cross_entropy(
                logits, lbl[i : i + hs.shape[0]], ignore_index=ignore_index, reduction="none"
            )
        return ce.view(B, T)

    head.forward = chunked_forward  # instance-level override (this process only)
    log.info(f"LM head patched for chunked CE (chunk={chunk}, vocab={vocab})")


# --------------------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _eval_length(
    model,
    windows: np.ndarray,
    device: torch.device,
    micro_batch: int,
) -> Tuple[float, int, Dict[int, Tuple[float, int]]]:
    """
    Returns (mean_ce, n_tokens, position_buckets) where position_buckets maps
    bucket_index -> (ce_sum, token_count) with bucket = position // BASE_LEN.
    """
    total_ce = 0.0
    total_tok = 0
    buckets: Dict[int, list] = {}

    for start in range(0, len(windows), micro_batch):
        batch = torch.from_numpy(windows[start : start + micro_batch]).to(device)
        labels = get_labels({"input_ids": batch}, label_ignore_index=IGNORE)
        # Model is bf16 (uniform, matching training); no autocast needed.
        ce = model(input_ids=batch, labels=labels, loss_reduction="none")  # (B, T)
        ce = ce.float()
        mask = labels != IGNORE
        total_ce += ce[mask].sum().item()
        total_tok += int(mask.sum().item())

        # position-bucketed accumulation
        T = ce.shape[1]
        pos = torch.arange(T, device=device)
        bidx = (pos // BASE_LEN).unsqueeze(0).expand_as(ce)
        for b in range(int(bidx.max().item()) + 1):
            bmask = (bidx == b) & mask
            n = int(bmask.sum().item())
            if n == 0:
                continue
            s = ce[bmask].sum().item()
            if b not in buckets:
                buckets[b] = [0.0, 0]
            buckets[b][0] += s
            buckets[b][1] += n

    mean_ce = total_ce / max(1, total_tok)
    return mean_ce, total_tok, {b: (v[0], v[1]) for b, v in buckets.items()}


def main():
    p = argparse.ArgumentParser(
        description="Long-context (increasing seq-len) breaking-point eval."
    )
    p.add_argument("run_name", type=str)
    p.add_argument("--checkpoint", required=True, help="s3://.../stepNNNNN (final checkpoint dir).")
    p.add_argument("--eval-data", required=True, help="Comma-separated held-out .npy shard paths.")
    p.add_argument("--seq-lens", type=str, default=",".join(map(str, DEFAULT_SEQ_LENS)))
    p.add_argument(
        "--tokens-per-len",
        type=int,
        default=2_000_000,
        help="Approx tokens evaluated per length (windows = tokens // L, min 2).",
    )
    p.add_argument("--max-windows", type=int, default=512, help="Hard cap on windows per length.")
    p.add_argument(
        "--micro-batch", type=int, default=1, help="Windows per forward (keep 1 for long L)."
    )
    p.add_argument("--ce-chunk", type=int, default=4096, help="Position chunk for the LM-head CE.")
    p.add_argument("--work-dir", type=str, default="/mnt/nvme/olmo-longctx-cache")
    p.add_argument("--output", type=str, default=None, help="Local JSON results path.")
    p.add_argument(
        "--upload-to", type=str, default=None, help="Optional s3:// path to upload results JSON."
    )
    p.add_argument("--wandb", action="store_true", default=False)
    opts, _ = p.parse_known_args()

    prepare_training_environment()
    try:
        device = get_default_device()
        seq_lens = [int(x) for x in opts.seq_lens.split(",") if x.strip()]

        model, model_cfg = _load_model(opts.checkpoint, device, os.path.join(opts.work_dir, "ckpt"))
        _patch_head_for_chunked_ce(model, chunk=opts.ce_chunk)

        shard_paths = [s.strip() for s in opts.eval_data.split(",") if s.strip()]
        streams = [_open_tokens(sp, os.path.join(opts.work_dir, "data")) for sp in shard_paths]
        log.info(f"loaded {len(streams)} shards, sizes={[len(s) for s in streams]}")

        run = None
        if opts.wandb:
            import wandb

            run = wandb.init(
                entity="eduLLM",
                project="pretraining",
                name=f"{opts.run_name}-longctx",
                config=dict(
                    checkpoint=opts.checkpoint,
                    seq_lens=seq_lens,
                    tokens_per_len=opts.tokens_per_len,
                ),
            )

        results: List[dict] = []
        baseline_ppl: Optional[float] = None
        print(
            f"\n{'seq_len':>10} {'windows':>8} {'tokens':>12} {'mean_ce':>9} {'ppl':>12} {'x_base':>8}"
        )
        print("-" * 66)
        for L in seq_lens:
            n_windows = max(2, min(opts.max_windows, opts.tokens_per_len // L))
            windows = _windows(streams, L, n_windows)
            if len(windows) == 0:
                log.warning(f"L={L}: no windows (shards too short); stopping sweep.")
                break
            mean_ce, n_tok, buckets = _eval_length(model, windows, device, opts.micro_batch)
            ppl = math.exp(mean_ce)
            if baseline_ppl is None:
                baseline_ppl = ppl
            x_base = ppl / baseline_ppl
            print(
                f"{L:>10} {len(windows):>8} {n_tok:>12,} {mean_ce:>9.4f} {ppl:>12.3f} {x_base:>8.2f}"
            )
            pos_curve = {
                b * BASE_LEN: (s / c if c else None) for b, (s, c) in sorted(buckets.items())
            }
            results.append(
                dict(
                    seq_len=L,
                    n_windows=int(len(windows)),
                    n_tokens=n_tok,
                    mean_ce=mean_ce,
                    ppl=ppl,
                    x_baseline=x_base,
                    pos_ce=pos_curve,
                )
            )
            if run is not None:
                run.log(dict(seq_len=L, ppl=ppl, mean_ce=mean_ce, x_baseline=x_base), step=L)

        summary = dict(
            run_name=opts.run_name,
            checkpoint=opts.checkpoint,
            num_params=model_cfg.num_params,
            num_non_embedding_params=model_cfg.num_non_embedding_params,
            base_len=BASE_LEN,
            results=results,
        )

        out_path = opts.output or os.path.join(opts.work_dir, f"longctx_{opts.run_name}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"wrote results -> {out_path}")
        if opts.upload_to:
            # The third and last `aws s3 cp` in this file, replaced for the same reason as the
            # other two. This one is the most costly to leave: it fires AFTER the whole sweep,
            # so the CLI's absence would throw away every number the run just spent its time
            # computing. save_overwrite because a retry re-derives the same path.
            upload(out_path, opts.upload_to, save_overwrite=True)
            log.info(f"uploaded results -> {opts.upload_to}")
        if run is not None:
            run.finish()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
