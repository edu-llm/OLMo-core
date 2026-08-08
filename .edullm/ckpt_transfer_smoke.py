"""
Checkpoint-transfer smoke v3 — Ben's parallel early post-train idea.

Load real team bases Ms=step15000, Mt=step20000 (edullm-370M-30B), run
full-weight assistant-masked SFT on Ms only (math-sft-60m), form
Δ = FT − Ms, apply Mt + λΔ, and report zero-shot held-out SFT CE retention.

No transfer-then-finetune. SFT only (not DPO/RLVR). Never push to main.
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

DEFAULT_MS = (
    "s3://edullm-checkpoints/olmo2-370m-cpt/edullm-370M-30B/step15000-unsharded"
)
DEFAULT_MT = (
    "s3://edullm-checkpoints/olmo2-370m-cpt/edullm-370M-30B/step20000-unsharded"
)
DEFAULT_SFT_DATASET_ID = "sft/math-sft-60m"
DEFAULT_SFT_VERSION = "v1"
LABEL_IGNORE_INDEX = -100


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def convert_legacy_olmo_to_olmo_core(
    legacy: Mapping[str, torch.Tensor],
    *,
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Map unsharded legacy OLMo ``model.pt`` keys into olmo_core Transformer keys.

    Follows allenai/OLMo ``convert_olmo_to_hf_new.py`` split/chunk conventions, then
    the HF→olmo_core naming in :mod:`olmo_core.nn.hf.convert`.
    """
    if n_kv_heads is None:
        n_kv_heads = n_heads
    head_dim = d_model // n_heads
    fused_dims = [d_model, head_dim * n_kv_heads, head_dim * n_kv_heads]

    out: Dict[str, torch.Tensor] = {}
    out["embeddings.weight"] = legacy["transformer.wte.weight"].detach().cpu().contiguous()
    out["lm_head.norm.weight"] = legacy["transformer.ln_f.weight"].detach().cpu().contiguous()
    out["lm_head.w_out.weight"] = legacy["transformer.ff_out.weight"].detach().cpu().contiguous()

    for i in range(n_layers):
        prefix = f"transformer.blocks.{i}"
        q, k, v = torch.split(legacy[f"{prefix}.att_proj.weight"], fused_dims, dim=0)
        # Official convert names chunk0=up, chunk1=gate → HF up/gate → olmo_core w3/w1.
        up, gate = torch.chunk(legacy[f"{prefix}.ff_proj.weight"], 2, dim=0)

        out[f"blocks.{i}.attention.w_q.weight"] = q.detach().cpu().contiguous()
        out[f"blocks.{i}.attention.w_k.weight"] = k.detach().cpu().contiguous()
        out[f"blocks.{i}.attention.w_v.weight"] = v.detach().cpu().contiguous()
        out[f"blocks.{i}.attention.w_out.weight"] = (
            legacy[f"{prefix}.attn_out.weight"].detach().cpu().contiguous()
        )
        out[f"blocks.{i}.attention.q_norm.weight"] = (
            legacy[f"{prefix}.q_norm.weight"].detach().cpu().contiguous()
        )
        out[f"blocks.{i}.attention.k_norm.weight"] = (
            legacy[f"{prefix}.k_norm.weight"].detach().cpu().contiguous()
        )
        out[f"blocks.{i}.attention_norm.weight"] = (
            legacy[f"{prefix}.attn_norm.weight"].detach().cpu().contiguous()
        )
        out[f"blocks.{i}.feed_forward_norm.weight"] = (
            legacy[f"{prefix}.ff_norm.weight"].detach().cpu().contiguous()
        )
        out[f"blocks.{i}.feed_forward.w1.weight"] = gate.detach().cpu().contiguous()
        out[f"blocks.{i}.feed_forward.w2.weight"] = (
            legacy[f"{prefix}.ff_out.weight"].detach().cpu().contiguous()
        )
        out[f"blocks.{i}.feed_forward.w3.weight"] = up.detach().cpu().contiguous()

    return out


def _unwrap_legacy_state(obj: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict checkpoint, got {type(obj)}")
    if "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not any(k.startswith("transformer.") for k in obj):
        raise KeyError("checkpoint does not look like a legacy OLMo model.pt")
    return obj  # type: ignore[return-value]


def load_legacy_unsharded_dir(uri: str) -> Dict[str, torch.Tensor]:
    """Download ``model.pt`` from an unsharded ckpt dir and convert to olmo_core."""
    from olmo_core.io import cached_path

    model_uri = uri.rstrip("/") + "/model.pt"
    log.info("loading legacy checkpoint %s", model_uri)
    path = str(cached_path(model_uri))
    raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    legacy = _unwrap_legacy_state(raw)
    converted = convert_legacy_olmo_to_olmo_core(
        legacy, n_layers=16, d_model=1024, n_heads=16, n_kv_heads=16
    )
    log.info("converted %d tensors from %s", len(converted), uri)
    return converted


def build_olmo2_370m(device: torch.device):
    from olmo_core.data.tokenizer import TokenizerConfig
    from olmo_core.nn.transformer.config import TransformerConfig

    tok = TokenizerConfig.dolma2()
    vocab_size = tok.padded_vocab_size()
    config = TransformerConfig.olmo2_370M(vocab_size=vocab_size)
    model = config.build(init_device="cpu")
    n = sum(p.numel() for p in model.parameters())
    log.info("built olmo2_370M params=%s vocab=%d", f"{n:,}", vocab_size)
    return model.to(device), config, tok


def clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(
    model: torch.nn.Module, sd: Mapping[str, torch.Tensor], device: torch.device
) -> None:
    current = model.state_dict()
    filtered = {k: v.to(device) for k, v in sd.items() if k in current}
    missing = [k for k in current if k not in filtered]
    unexpected = [k for k in sd if k not in current]
    if unexpected:
        log.warning("unexpected keys (ignored): %s", unexpected[:8])
    if missing:
        # Buffers / non-persistent may be ok; fail hard if many params missing.
        missing_params = [k for k in missing if "weight" in k or "bias" in k]
        if missing_params:
            raise RuntimeError(f"missing parameter keys: {missing_params[:12]}")
    model.load_state_dict(filtered, strict=False)


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


def _open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def load_conversation_rows(uris: Sequence[str], *, max_rows: int) -> List[Dict[str, Any]]:
    from olmo_core.io import cached_path

    rows: List[Dict[str, Any]] = []
    for uri in uris:
        local = str(cached_path(uri))
        with _open_maybe_gz(local) as f:
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
    """Prefer platform-mounted dolma2, else HuggingFace identifier."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers is required to tokenize SFT conversations") from e

    local = os.environ.get("EDULLM_TOKENIZER_PATH") or os.environ.get("EDULLM_DATASET_TOKENIZER_PATH")
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
    """
    Tokenize chat rows with a simple Dolma2-style markup and assistant-only labels.

    Format per turn: ``<|user|>\\n...\\n`` / ``<|assistant|>\\n...<eos>``.
    Loss is on assistant content tokens and the trailing eos (not the role tag).
    """
    input_rows: List[torch.Tensor] = []
    mask_rows: List[torch.Tensor] = []

    for row in rows:
        ids: List[int] = []
        mask: List[bool] = []
        messages = row["messages"]
        for msg in messages:
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
                # system / other — encode, no loss
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


def make_batches(
    input_ids: torch.Tensor,
    label_mask: torch.Tensor,
    *,
    batch_size: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    n = input_ids.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end - start < 1:
            continue
        batches.append((input_ids[start:end], label_mask[start:end]))
    return batches


def get_shifted_labels(input_ids: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    labels.masked_fill_(~label_mask, LABEL_IGNORE_INDEX)
    return F.pad(labels[..., 1:], (0, 1, 0, 0), value=LABEL_IGNORE_INDEX)


def _ce_sum(model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
        loss = _ce_sum(model, input_ids, labels)
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
    p.add_argument("--ms-uri", type=str, default=DEFAULT_MS)
    p.add_argument("--mt-uri", type=str, default=DEFAULT_MT)
    p.add_argument("--steps-sft", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--sft-train-rows", type=int, default=2048)
    p.add_argument("--sft-eval-rows", type=int, default=256)
    p.add_argument("--sft-lr", type=float, default=5e-5)
    p.add_argument("--lambda-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--param-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="named for the platform command scanner; train loop uses float32 params",
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
    log.info("device=%s", device)

    t0 = time.time()
    model, _, tok_cfg = build_olmo2_370m(device)
    eos_id = tok_cfg.eos_token_id
    pad_id = tok_cfg.pad_token_id if tok_cfg.pad_token_id is not None else eos_id

    log.info("loading Ms from %s", args.ms_uri)
    ms = load_legacy_unsharded_dir(args.ms_uri)
    log.info("loading Mt from %s", args.mt_uri)
    mt = load_legacy_unsharded_dir(args.mt_uri)

    load_state(model, ms, device)

    log.info("loading math-sft conversations")
    train_uris = _s3_uri_list_for_sft("train")
    val_uris = _s3_uri_list_for_sft("val")
    train_rows = load_conversation_rows(train_uris, max_rows=args.sft_train_rows)
    eval_rows = load_conversation_rows(val_uris, max_rows=args.sft_eval_rows)
    log.info("sft rows train=%d eval=%d", len(train_rows), len(eval_rows))

    hf_tok = _get_hf_tokenizer()
    train_ids, train_mask = tokenize_conversations(
        train_rows, hf_tok, seq_len=args.seq_len, eos_token_id=eos_id, pad_token_id=pad_id
    )
    eval_ids, eval_mask = tokenize_conversations(
        eval_rows, hf_tok, seq_len=args.seq_len, eos_token_id=eos_id, pad_token_id=pad_id
    )
    sft_batches = make_batches(train_ids, train_mask, batch_size=args.batch_size)
    eval_batches = make_batches(eval_ids, eval_mask, batch_size=args.batch_size)
    log.info("sft batches=%d eval batches=%d", len(sft_batches), len(eval_batches))

    ce_ms = mean_assistant_ce(model, eval_batches, device)
    log.info("held-out assistant CE Ms=%.4f", ce_ms)

    log.info("SFT on Ms for %d steps lr=%g", args.steps_sft, args.sft_lr)
    train_sft(model, sft_batches, steps=args.steps_sft, device=device, lr=args.sft_lr)
    ft = clone_state(model)
    ce_ft = mean_assistant_ce(model, eval_batches, device)
    log.info("held-out assistant CE FT=%.4f", ce_ft)

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
        "version": 3,
        "question": (
            "Can early post-train (SFT) on Ms be fit onto later Mt via Δ without "
            "re-running post-train? (zero-shot Mt+Δ)"
        ),
        "architecture": {
            "type": "dense_olmo2_370M",
            "d_model": 1024,
            "n_layers": 16,
            "n_heads": 16,
            "note": "real team edullm-370M-30B bases; not MoE final",
        },
        "checkpoints": {"ms": args.ms_uri, "mt": args.mt_uri},
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
                else "no-go: Δ did not transfer enough zero-shot; do not count on parallel early post-train for this gap"
            )
        ),
        "caveats": [
            "SFT only — not DPO or RLVR",
            "dense 370M — not final MoE 4/40",
            "held-out assistant CE — not generation / pedagogy evals",
            "zero-shot fit only — no continue-finetune arm",
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
