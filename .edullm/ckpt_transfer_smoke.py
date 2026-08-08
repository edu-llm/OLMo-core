"""
Checkpoint-transfer smoke v4 — Ben's parallel early post-train idea.

MoE 4-of-40 twin (same routing pattern as team final): short pretrain to Ms then
Mt on real dolma2 tokens, full-weight assistant-masked SFT on Ms only
(math-sft-60m), Δ = FT − Ms onto Mt, zero-shot held-out SFT CE retention.

Actual post-train stage = SFT (not continued pretrain). Not DPO/RLVR.
Never push to main.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("ckpt_transfer_smoke")

DEFAULT_DATASET_ID = "pretrain/math-frontload-100m"
DEFAULT_DATASET_VERSION = "v1"
DEFAULT_TOKENIZER_ID = "tokenizer/dolma2-bpe"
DEFAULT_SFT_DATASET_ID = "sft/math-sft-60m"
DEFAULT_SFT_VERSION = "v1"
LABEL_IGNORE_INDEX = -100


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
    """Team-shaped MoE (4 active / 40 experts) at a width that fits one L4."""
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
    from olmo_core.data.tokenizer import TokenizerConfig
    from olmo_core.io import cached_path

    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    dataset_id = os.environ.get("EDULLM_DATASET_ID", DEFAULT_DATASET_ID)
    version = os.environ.get("EDULLM_DATASET_VERSION", DEFAULT_DATASET_VERSION)
    tokenizer_id = os.environ.get("EDULLM_DATASET_TOKENIZER", DEFAULT_TOKENIZER_ID)

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

    tok = TokenizerConfig.dolma2()
    vocab_size = tok.padded_vocab_size()

    path = str(cached_path(read.paths[0]))
    dtype = np.dtype(read.dtype)
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
        "corpus %s/%s shard0 tokens=%s vocab=%d",
        dataset_id,
        version,
        f"{tokens.shape[0]:,}",
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
            window = np.clip(window, 0, vocab_size - 1)
            rows.append(torch.from_numpy(window.copy()))
            cursor += seq_len * stride
        batches.append(torch.stack(rows, dim=0))
    return batches


def clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(
    model: torch.nn.Module, sd: Mapping[str, torch.Tensor], device: torch.device
) -> None:
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


def _ce_loss_lm(
    model: torch.nn.Module, batch: torch.Tensor, *, reduction: str = "mean"
) -> torch.Tensor:
    input_ids = batch[:, :-1].contiguous()
    labels = batch[:, 1:].contiguous()
    out = model(input_ids=input_ids, labels=labels, loss_reduction=reduction)
    if hasattr(out, "loss"):
        return out.loss
    raise TypeError(f"expected LMOutputWithLoss, got {type(out)}")


def train_pretrain_batches(
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
        loss = _ce_loss_lm(model, batch, reduction="mean")
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step % max(steps // 5, 1) == 0 or step + 1 == steps:
            log.info("pretrain step %d/%d ce=%.4f", step + 1, steps, last)
    return last


def _s3_uri_list_for_sft(split: str) -> List[str]:
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    dataset_id = os.environ.get("EDULLM_SFT_DATASET_ID", DEFAULT_SFT_DATASET_ID)
    version = os.environ.get("EDULLM_SFT_DATASET_VERSION", DEFAULT_SFT_VERSION)
    s3 = Boto3S3.default()
    if version in ("", "latest"):
        resolved = resolve_latest(dataset_id, s3=s3)
        if resolved is None:
            raise RuntimeError(f"no published version of {dataset_id}")
        version = resolved
    read = dataset_paths(dataset_id, version, split=split, s3=s3)
    if not read.paths:
        raise RuntimeError(f"{dataset_id}/{version} split={split} has no paths")
    return list(read.paths)


def _is_gzip_file(path: str, *, source_uri: str = "") -> bool:
    """
    ``cached_path`` often stores objects under a hash name with no ``.gz`` suffix.
    Prefer the source URI suffix, then sniff the gzip magic bytes.
    """
    if source_uri.endswith(".gz") or path.endswith(".gz"):
        return True
    with open(path, "rb") as bf:
        return bf.read(2) == b"\x1f\x8b"


def _open_maybe_gz(path: str, *, source_uri: str = ""):
    if _is_gzip_file(path, source_uri=source_uri):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def load_conversation_rows(uris: Sequence[str], *, max_rows: int) -> List[Dict[str, Any]]:
    from olmo_core.io import cached_path

    rows: List[Dict[str, Any]] = []
    for uri in uris:
        local = str(cached_path(uri))
        with _open_maybe_gz(local, source_uri=uri) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "messages" not in row:
                    continue
                rows.append(row)
                if len(rows) >= max_rows:
                    return rows
    return rows


def _get_hf_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers is required to tokenize SFT conversations") from e

    local = os.environ.get("EDULLM_TOKENIZER_PATH") or os.environ.get(
        "EDULLM_DATASET_TOKENIZER_PATH"
    )
    candidates = []
    if local:
        candidates.append(local)
    candidates.append("allenai/dolma2-tokenizer")
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            log.info("loaded tokenizer %s", name)
            return tok
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("tokenizer load failed for %s: %s", name, e)
    raise RuntimeError(f"could not load dolma2 tokenizer: {last_err}")


def tokenize_conversations(
    rows: Sequence[Mapping[str, Any]],
    tokenizer,
    *,
    seq_len: int,
    eos_token_id: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dolma2-style chat markup; loss only on assistant content + trailing eos."""
    input_rows: List[torch.Tensor] = []
    mask_rows: List[torch.Tensor] = []

    for row in rows:
        ids: List[int] = []
        mask: List[bool] = []
        for msg in row["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                piece = f"<|user|>\n{content}\n"
                toks = tokenizer.encode(piece, add_special_tokens=False)
                ids.extend(toks)
                mask.extend([False] * len(toks))
            elif role == "assistant":
                prefix = tokenizer.encode("<|assistant|>\n", add_special_tokens=False)
                body = tokenizer.encode(content, add_special_tokens=False)
                ids.extend(prefix + body + [eos_token_id])
                mask.extend([False] * len(prefix) + [True] * len(body) + [True])
            else:
                piece = f"<|{role}|>\n{content}\n"
                toks = tokenizer.encode(piece, add_special_tokens=False)
                ids.extend(toks)
                mask.extend([False] * len(toks))

        if not any(mask):
            continue
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            mask = mask[:seq_len]
        if len(ids) < seq_len:
            pad_n = seq_len - len(ids)
            ids = ids + [pad_token_id] * pad_n
            mask = mask + [False] * pad_n
        input_rows.append(torch.tensor(ids, dtype=torch.long))
        mask_rows.append(torch.tensor(mask, dtype=torch.bool))

    if not input_rows:
        raise RuntimeError("no tokenized SFT rows with assistant labels")
    return torch.stack(input_rows, dim=0), torch.stack(mask_rows, dim=0)


def make_sft_batches(
    input_ids: torch.Tensor,
    label_mask: torch.Tensor,
    *,
    batch_size: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    n = input_ids.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append((input_ids[start:end], label_mask[start:end]))
    return batches


def get_shifted_labels(input_ids: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    labels.masked_fill_(~label_mask, LABEL_IGNORE_INDEX)
    return F.pad(labels[..., 1:], (0, 1, 0, 0), value=LABEL_IGNORE_INDEX)


def _ce_sum_sft(
    model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    out = model(input_ids=input_ids, labels=labels, loss_reduction="sum")
    if hasattr(out, "ce_loss") and out.ce_loss is not None:
        return out.ce_loss
    if hasattr(out, "loss"):
        return out.loss
    raise TypeError(f"expected LMOutputWithLoss, got {type(out)}")


@torch.no_grad()
def mean_assistant_ce(
    model: torch.nn.Module,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    for input_ids, label_mask in batches:
        input_ids = input_ids.to(device)
        label_mask = label_mask.to(device)
        labels = get_shifted_labels(input_ids, label_mask)
        loss = _ce_sum_sft(model, input_ids, labels)
        total += float(loss.item())
        tokens += int((labels != LABEL_IGNORE_INDEX).sum().item())
    return total / max(tokens, 1)


def train_sft(
    model: torch.nn.Module,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
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
        input_ids, label_mask = batches[step % n]
        input_ids = input_ids.to(device)
        label_mask = label_mask.to(device)
        labels = get_shifted_labels(input_ids, label_mask)
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels, loss_reduction="mean")
        loss = out.loss if hasattr(out, "loss") else out
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step % max(steps // 5, 1) == 0 or step + 1 == steps:
            log.info("sft step %d/%d loss=%.4f", step + 1, steps, last)
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
    p.add_argument("--sft-train-rows", type=int, default=1024)
    p.add_argument("--sft-eval-rows", type=int, default=128)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=40)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sft-lr", type=float, default=5e-5)
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
    log.info("param-dtype flag=%s", args.param_dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save = Path(args.save_folder)
    if str(save).startswith("s3://"):
        local_out = Path(os.environ.get("TMPDIR", "/tmp")) / "ckpt_transfer_smoke"
        local_out.mkdir(parents=True, exist_ok=True)
    else:
        save.mkdir(parents=True, exist_ok=True)
        local_out = save

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("device=%s MoE %d/%d", device, args.top_k, args.num_experts)

    from olmo_core.data.tokenizer import TokenizerConfig

    tok_cfg = TokenizerConfig.dolma2()
    eos_id = tok_cfg.eos_token_id
    pad_id = tok_cfg.pad_token_id if tok_cfg.pad_token_id is not None else eos_id

    tokens, vocab_size, corpus_meta = resolve_corpus_tokens()
    pre_batches = make_batches_from_tokens(
        tokens,
        n_batches=args.train_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        start=0,
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

    log.info("phase Ms: %d pretrain steps", args.steps_pre)
    train_pretrain_batches(
        model, pre_batches, steps=args.steps_pre, device=device, lr=args.lr
    )
    ms = clone_state(model)

    log.info("phase Mt: +%d pretrain gap steps", args.steps_gap)
    train_pretrain_batches(
        model, pre_batches, steps=args.steps_gap, device=device, lr=args.lr
    )
    mt = clone_state(model)

    log.info("loading math-sft conversations")
    train_rows = load_conversation_rows(
        _s3_uri_list_for_sft("train"), max_rows=args.sft_train_rows
    )
    eval_rows = load_conversation_rows(
        _s3_uri_list_for_sft("val"), max_rows=args.sft_eval_rows
    )
    hf_tok = _get_hf_tokenizer()
    train_ids, train_mask = tokenize_conversations(
        train_rows, hf_tok, seq_len=args.seq_len, eos_token_id=eos_id, pad_token_id=pad_id
    )
    eval_ids, eval_mask = tokenize_conversations(
        eval_rows, hf_tok, seq_len=args.seq_len, eos_token_id=eos_id, pad_token_id=pad_id
    )
    sft_batches = make_sft_batches(train_ids, train_mask, batch_size=args.batch_size)
    eval_batches = make_sft_batches(eval_ids, eval_mask, batch_size=args.batch_size)
    log.info(
        "sft rows train=%d eval=%d batches=%d/%d",
        len(train_rows),
        len(eval_rows),
        len(sft_batches),
        len(eval_batches),
    )

    load_state(model, ms, device)
    ce_ms = mean_assistant_ce(model, eval_batches, device)
    log.info("held-out assistant CE Ms=%.4f", ce_ms)

    log.info("SFT on Ms: %d steps lr=%g", args.steps_sft, args.sft_lr)
    train_sft(model, sft_batches, steps=args.steps_sft, device=device, lr=args.sft_lr)
    ft = clone_state(model)
    ce_ft = mean_assistant_ce(model, eval_batches, device)

    merged = apply_delta(mt, ms, ft, args.lambda_scale)

    load_state(model, mt, device)
    ce_mt = mean_assistant_ce(model, eval_batches, device)
    load_state(model, merged, device)
    ce_merge = mean_assistant_ce(model, eval_batches, device)

    sft_gain = ce_ms - ce_ft
    ret = retention(ce_mt, ce_merge, ce_ms, ce_ft)
    sft_ok = sft_gain >= args.min_sft_gain
    go = bool(sft_ok and ret == ret and ret > 0.25 and ce_merge < ce_mt)

    result: Dict[str, Any] = {
        "version": 4,
        "question": (
            "Can early SFT on a MoE Ms be fit onto later MoE Mt via Δ without "
            "re-running post-train? (zero-shot Mt+Δ)"
        ),
        "architecture": {
            "type": "moe",
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "note": "4/40 routing at smoke width; not full Joe-scale MoE",
        },
        "bases": {
            "kind": "synthetic_twin_on_shared_pretrain_shard",
            "steps_pre": args.steps_pre,
            "steps_gap": args.steps_gap,
            "corpus": corpus_meta,
        },
        "sft": {
            "dataset_id": DEFAULT_SFT_DATASET_ID,
            "dataset_version": DEFAULT_SFT_VERSION,
            "steps": args.steps_sft,
            "lr": args.sft_lr,
            "seq_len": args.seq_len,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "loss": "assistant_only_label_mask",
        },
        "lambda": args.lambda_scale,
        "ce_heldout_assistant": {
            "ms": ce_ms,
            "ft_ms": ce_ft,
            "mt": ce_mt,
            "mt_plus_delta": ce_merge,
        },
        "sft_gain_ce": sft_gain,
        "sft_gain_sufficient": sft_ok,
        "min_sft_gain": args.min_sft_gain,
        "retention_ce": ret,
        "go_for_parallel_early_posttrain_fit": go,
        "interpretation": (
            "go: SFT improved Ms on held-out assistant CE, and Mt+Δ beat Mt keeping >25% of that gain"
            if go
            else (
                "inconclusive: SFT did not create a clear held-out gain — increase --steps-sft"
                if not sft_ok
                else "no-go: Δ did not transfer enough zero-shot on this MoE twin"
            )
        ),
        "caveats": [
            "SFT only — not DPO or RLVR",
            "MoE twin at smoke width — not full Joe-scale MoE",
            "synthetic Ms/Mt gap on real tokens — not team MoE intermediates",
            "held-out assistant CE — not generation evals",
            "zero-shot fit only",
        ],
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
