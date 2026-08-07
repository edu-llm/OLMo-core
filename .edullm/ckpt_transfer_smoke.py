"""
MoE checkpoint-transfer smoke (Ben / Lin et al. 2025) — more accurate v2.

Creates Ms → Mt on one MoE trajectory (4-of-40 experts, small width so it fits
gpu-1xl4), continues training on *real* eduLLM corpus tokens, SFT-style continues
Ms only, then applies Δ = FT − Ms onto Mt and reports CE retention on a held-out
real split.

Data: prefers EDULLM_DATASET_* from the platform; otherwise resolves
pretrain/math-frontload-100m@v1 (dolma2).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

log = logging.getLogger("ckpt_transfer_smoke")

DEFAULT_DATASET_ID = "pretrain/math-frontload-100m"
DEFAULT_DATASET_VERSION = "v1"
DEFAULT_TOKENIZER_ID = "tokenizer/dolma2-bpe"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_moe_model(
    vocab_size: int,
    device: torch.device,
    *,
    d_model: int = 512,
    n_layers: int = 8,
    n_heads: int = 8,
    num_experts: int = 40,
    top_k: int = 4,
):
    """
    Team-shaped MoE (4 active / 40 experts) at a width that fits one L4.

    Joe-scale 4/40 at 7B is the same *routing pattern*; this is the affordable twin.
    """
    from olmo_core.nn.transformer import TransformerConfig

    config = TransformerConfig.llama_like_moe(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden_size=max(128, d_model // 2),
        reordered_norm=True,
        dropless=True,
    )
    model = config.build(init_device="cpu")
    n = sum(p.numel() for p in model.parameters())
    log.info(
        "built MoE: experts=%d top_k=%d d_model=%d layers=%d params=%s",
        num_experts,
        top_k,
        d_model,
        n_layers,
        f"{n:,}",
    )
    return model.to(device), config


def resolve_corpus_tokens() -> Tuple[np.ndarray, int, Dict[str, str]]:
    """Load a 1D token array from the platform corpus (or default math-frontload)."""
    from olmo_core.data import TokenizerConfig

    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    dataset_id = os.environ.get("EDULLM_DATASET_ID", DEFAULT_DATASET_ID)
    version = os.environ.get("EDULLM_DATASET_VERSION", DEFAULT_DATASET_VERSION)
    tokenizer_id = os.environ.get("EDULLM_DATASET_TOKENIZER", DEFAULT_TOKENIZER_ID)
    # Accept registry reference ids like math-frontload-100m-v1.
    if dataset_id == "math-frontload-100m-v1" or (
        dataset_id.endswith("-v1") and dataset_id.startswith("math-frontload")
    ):
        dataset_id = DEFAULT_DATASET_ID
        version = DEFAULT_DATASET_VERSION
    if "/" not in dataset_id and dataset_id.startswith("math-frontload"):
        dataset_id = DEFAULT_DATASET_ID

    s3 = Boto3S3.default()
    if version in ("", "latest"):
        resolved = resolve_latest(dataset_id, s3=s3)
        if resolved is None:
            raise RuntimeError(f"no published version of {dataset_id}")
        version = resolved

    read = dataset_paths(dataset_id, version, s3=s3)
    if not read.paths:
        raise RuntimeError(f"{dataset_id}/{version} has no trainable shards")
    if read.dtype is None:
        raise RuntimeError(f"{dataset_id}/{version} declares no dtype")

    # Vocab from dolma2 (matches math-frontload).
    if tokenizer_id != "tokenizer/dolma2-bpe":
        log.warning("unexpected tokenizer %s; using dolma2 vocab size", tokenizer_id)
    tok = TokenizerConfig.dolma2()
    vocab_size = tok.padded_vocab_size()

    from olmo_core.io import cached_path

    path = str(cached_path(read.paths[0]))
    dtype = np.dtype(read.dtype)
    # Respect little-endian declaration when present.
    if getattr(read, "byte_order", "little") == "big":
        dtype = dtype.newbyteorder(">")
    tokens = np.memmap(path, mode="r", dtype=dtype)
    meta = {
        "dataset_id": dataset_id,
        "dataset_version": version,
        "tokenizer_id": tokenizer_id,
        "shard": read.paths[0],
        "dtype": str(dtype),
        "token_count": str(int(tokens.shape[0])),
    }
    log.info(
        "corpus %s/%s shard0 tokens=%s dtype=%s vocab=%d",
        dataset_id,
        version,
        f"{tokens.shape[0]:,}",
        dtype,
        vocab_size,
    )
    return tokens, vocab_size, meta


def make_batches_from_tokens(
    tokens: np.ndarray,
    *,
    n_batches: int,
    batch_size: int,
    seq_len: int,
    start: int,
    stride: int,
    vocab_size: int,
) -> List[torch.Tensor]:
    """Contiguous windows from a real shard (not random ids)."""
    need = start + n_batches * batch_size * seq_len * max(stride, 1)
    if tokens.shape[0] < need:
        raise RuntimeError(
            f"shard too short for requested windows: have {tokens.shape[0]}, need ~{need}"
        )
    batches: List[torch.Tensor] = []
    cursor = start
    for _ in range(n_batches):
        rows = []
        for _ in range(batch_size):
            window = np.asarray(tokens[cursor : cursor + seq_len], dtype=np.int64)
            # Clamp accidental OOV to pad/eos range inside vocab.
            window = np.clip(window, 0, vocab_size - 1)
            rows.append(torch.from_numpy(window.copy()))
            cursor += seq_len * stride
        batches.append(torch.stack(rows, dim=0))
    return batches


def clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model: torch.nn.Module, sd: Mapping[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=True)


def apply_delta(
    base: Mapping[str, torch.Tensor],
    ms: Mapping[str, torch.Tensor],
    ft: Mapping[str, torch.Tensor],
    lam: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, b in base.items():
        if key not in ms or key not in ft or not torch.is_floating_point(b):
            out[key] = b.clone()
            continue
        out[key] = b + lam * (ft[key].float() - ms[key].float()).to(dtype=b.dtype)
    return out


def _ce_loss(model: torch.nn.Module, batch: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    input_ids = batch[:, :-1].contiguous()
    labels = batch[:, 1:].contiguous()
    out = model(input_ids=input_ids, labels=labels, loss_reduction=reduction)
    if hasattr(out, "loss"):
        return out.loss
    raise TypeError(f"expected LMOutputWithLoss, got {type(out)}")


@torch.no_grad()
def mean_ce(model: torch.nn.Module, batches: Sequence[torch.Tensor], device: torch.device) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    for batch in batches:
        batch = batch.to(device)
        loss = _ce_loss(model, batch, reduction="sum")
        total += float(loss.item())
        tokens += batch[:, 1:].numel()
    return total / max(tokens, 1)


def train_on_batches(
    model: torch.nn.Module,
    batches: Sequence[torch.Tensor],
    *,
    steps: int,
    device: torch.device,
    lr: float,
) -> float:
    if steps <= 0 or not batches:
        return float("nan")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    last = float("nan")
    n = len(batches)
    for step in range(steps):
        batch = batches[step % n].to(device)
        opt.zero_grad(set_to_none=True)
        loss = _ce_loss(model, batch, reduction="mean")
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step % max(steps // 5, 1) == 0 or step + 1 == steps:
            log.info("train step %d/%d ce=%.4f", step + 1, steps, last)
    return last


def retention(score_mt: float, score_merge: float, score_ms: float, score_ft: float) -> float:
    denom = score_ms - score_ft
    if abs(denom) < 1e-8:
        return float("nan")
    return (score_mt - score_merge) / denom


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-folder", type=str, required=True)
    p.add_argument("--steps-pre", type=int, default=200)
    p.add_argument("--steps-gap", type=int, default=200)
    p.add_argument("--steps-sft", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--train-batches", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--sft-batches", type=int, default=16)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=40)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sft-lr", type=float, default=1e-4)
    p.add_argument("--lambda-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--param-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="named for the platform command scanner",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--min-sft-gain",
        type=float,
        default=0.05,
        help="require Ms CE − FT CE ≥ this before calling transfer decisive",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    log.info("param-dtype flag=%s (train loop float32)", args.param_dtype)

    save = Path(args.save_folder)
    if str(save).startswith("s3://"):
        local_out = Path(os.environ.get("TMPDIR", "/tmp")) / "ckpt_transfer_smoke"
        local_out.mkdir(parents=True, exist_ok=True)
    else:
        save.mkdir(parents=True, exist_ok=True)
        local_out = save

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("device=%s MoE %d/%d", device, args.top_k, args.num_experts)

    tokens, vocab_size, corpus_meta = resolve_corpus_tokens()

    # Disjoint regions of the shard: pretrain traj | SFT | eval
    pre_batches = make_batches_from_tokens(
        tokens,
        n_batches=args.train_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        start=0,
        stride=1,
        vocab_size=vocab_size,
    )
    sft_batches = make_batches_from_tokens(
        tokens,
        n_batches=args.sft_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        start=args.train_batches * args.batch_size * args.seq_len * 2,
        stride=1,
        vocab_size=vocab_size,
    )
    eval_batches = make_batches_from_tokens(
        tokens,
        n_batches=args.eval_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        start=args.train_batches * args.batch_size * args.seq_len * 4,
        stride=1,
        vocab_size=vocab_size,
    )

    t0 = time.time()
    model, _ = build_moe_model(
        vocab_size,
        device,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )

    log.info("phase Ms: %d steps on real pretrain windows", args.steps_pre)
    train_on_batches(model, pre_batches, steps=args.steps_pre, device=device, lr=args.lr)
    ms = clone_state(model)

    log.info("phase Mt: +%d steps (gap) on real pretrain windows", args.steps_gap)
    train_on_batches(model, pre_batches, steps=args.steps_gap, device=device, lr=args.lr)
    mt = clone_state(model)

    load_state(model, ms, device)
    log.info("phase SFT on Ms: %d steps on held-in SFT windows", args.steps_sft)
    train_on_batches(model, sft_batches, steps=args.steps_sft, device=device, lr=args.sft_lr)
    ft = clone_state(model)

    merged = apply_delta(mt, ms, ft, args.lambda_scale)

    load_state(model, ms, device)
    ce_ms = mean_ce(model, eval_batches, device)
    load_state(model, ft, device)
    ce_ft = mean_ce(model, eval_batches, device)
    load_state(model, mt, device)
    ce_mt = mean_ce(model, eval_batches, device)
    load_state(model, merged, device)
    ce_merge = mean_ce(model, eval_batches, device)

    sft_gain = ce_ms - ce_ft
    ret = retention(ce_mt, ce_merge, ce_ms, ce_ft)
    sft_ok = sft_gain >= args.min_sft_gain
    go = bool(sft_ok and ret == ret and ret > 0.25 and ce_merge < ce_mt)

    result: Dict[str, Any] = {
        "version": 2,
        "architecture": {
            "type": "moe",
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "note": "4/40 routing pattern at smoke width; not full 7B Joe scale",
        },
        "corpus": corpus_meta,
        "steps_pre": args.steps_pre,
        "steps_gap": args.steps_gap,
        "steps_sft": args.steps_sft,
        "lambda": args.lambda_scale,
        "ce_heldout_real_tokens": {
            "ms": ce_ms,
            "ft_ms": ce_ft,
            "mt": ce_mt,
            "mt_plus_delta": ce_merge,
        },
        "sft_gain_ce": sft_gain,
        "sft_gain_sufficient": sft_ok,
        "min_sft_gain": args.min_sft_gain,
        "retention_ce": ret,
        "go_for_close_gap_transfer": go,
        "interpretation": (
            "go: SFT improved Ms on held-out real tokens, and Mt+Δ beat Mt keeping >25% of that gain"
            if go
            else (
                "inconclusive: SFT did not create a clear gain on held-out tokens — rerun with more --steps-sft"
                if not sft_ok
                else "no-go / weak transfer: prefer transfer-then-finetune, regenerate, or redo on final"
            )
        ),
        "paper": "https://arxiv.org/abs/2503.20110",
        "elapsed_sec": round(time.time() - t0, 1),
        "device": str(device),
    }

    out_path = local_out / "transfer_smoke_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("wrote %s", out_path)
    print(json.dumps(result, indent=2))

    if str(save).startswith("s3://"):
        from olmo_core.io import upload

        dest = f"{str(save).rstrip('/')}/transfer_smoke_results.json"
        upload(out_path, dest, save_overwrite=True)
        log.info("uploaded %s", dest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
