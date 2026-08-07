"""
Smoke test for Ben's early-checkpoint post-train → transfer idea (Lin et al. 2025).

One process:
  1. Obtain an earlier base Ms and a later base Mt from the *same* trajectory.
  2. Short SFT / continued train on Ms only → FT.
  3. Build Δ = FT − Ms, form Mt + λΔ.
  4. Compare CE on a held-out batch; write retention JSON.

Modes:
  synthetic_twin (default)
      Create Ms/Mt by training one olmo2_370M-scale model for N then N+gap steps
      on synthetic tokens. Same scientific question as Lin's OLMo-2 intermediate study,
      without depending on old-OLMo ``model.pt`` key layouts.
  team_s3
      Load ``model.pt`` from the eduLLM 370M ladder run (step15000 / step20000).
      Requires state-dict keys that load into olmo_core; otherwise exits with a clear error.

Never pushes. Results go under ``--save-folder`` (platform: ``$EDULLM_CHECKPOINT_DIR``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

log = logging.getLogger("ckpt_transfer_smoke")

DEFAULT_MS = (
    "s3://edullm-checkpoints/olmo2-370m-cpt/edullm-370M-30B/step15000-unsharded/model.pt"
)
DEFAULT_MT = (
    "s3://edullm-checkpoints/olmo2-370m-cpt/edullm-370M-30B/step20000-unsharded/model.pt"
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_model(vocab_size: int = 100_278, device: torch.device | str = "cpu"):
    """370M-class OLMo2 config (matches team ladder width/depth)."""
    from olmo_core.nn.feed_forward import FeedForwardConfig
    from olmo_core.nn.transformer import TransformerConfig

    # Ladder YAML used mlp_ratio 8 → hidden 8192; pin that so shapes can match team dumps.
    config = TransformerConfig.olmo2_370M(
        vocab_size=vocab_size,
        feed_forward=FeedForwardConfig(hidden_size=8192, bias=False),
    )
    model = config.build(init_device="cpu")
    return model.to(device), config


def unwrap_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("model", "state_dict", "module"):
            inner = raw.get(key)
            if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
                raw = inner
                break
    if not isinstance(raw, dict):
        raise TypeError(f"expected a state dict, got {type(raw)}")
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            out[str(k)] = v
    if not out:
        raise ValueError("state dict contained no tensors")
    return out


def load_state_dict(uri: str) -> Dict[str, torch.Tensor]:
    from olmo_core.io import cached_path

    log.info("fetching %s", uri)
    path = cached_path(uri)
    log.info("loading %s", path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return unwrap_state_dict(raw)


def remap_old_olmo_keys(sd: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Best-effort rename from classic OLMo trainer keys → olmo_core."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        k = key
        if k.startswith("model."):
            k = k[len("model.") :]
        if k.startswith("transformer."):
            k = k[len("transformer.") :]
        k = k.replace("wte.weight", "embeddings.weight")
        k = k.replace("wte.embedding.weight", "embeddings.weight")
        k = k.replace("ln_f.weight", "lm_head.norm.weight")
        k = k.replace("ff_out.weight", "lm_head.w_out.weight")
        remapped[k] = value
    return remapped


def load_into_model(model: torch.nn.Module, sd: Mapping[str, torch.Tensor]) -> List[str]:
    """Load with strict=False; return missing material keys (skip buffers)."""
    candidates = [dict(sd), remap_old_olmo_keys(sd)]
    # Also try stripping a leading 'module.'
    stripped = {k[len("module.") :]: v for k, v in sd.items() if k.startswith("module.")}
    if stripped:
        candidates.append(stripped)
        candidates.append(remap_old_olmo_keys(stripped))

    best_missing: Optional[List[str]] = None
    best_unexpected = 10**9
    for cand in candidates:
        missing, unexpected = model.load_state_dict(cand, strict=False)
        # Ignore missing that are clearly non-critical if almost all matched.
        if best_missing is None or len(unexpected) + len(missing) < best_unexpected + len(
            best_missing
        ):
            best_missing = list(missing)
            best_unexpected = len(unexpected)
        n_model = sum(1 for _ in model.state_dict())
        n_loaded = n_model - len(missing)
        if n_loaded >= 0.85 * n_model and len(unexpected) < 0.25 * n_model:
            log.info(
                "loaded state dict (~%.0f%% params matched, %d missing, %d unexpected)",
                100.0 * n_loaded / max(n_model, 1),
                len(missing),
                len(unexpected),
            )
            return list(missing)
    assert best_missing is not None
    raise RuntimeError(
        "could not load checkpoint into olmo_core Transformer "
        f"(missing={len(best_missing)}, unexpected≈{best_unexpected}). "
        "Team ladder dumps use the old OLMo trainer layout; use --mode synthetic_twin "
        "or add a converter. Sample missing: "
        + ", ".join(best_missing[:8])
    )


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
        if key not in ms or key not in ft:
            out[key] = b.clone()
            continue
        if not torch.is_floating_point(b):
            out[key] = b.clone()
            continue
        out[key] = b + lam * (ft[key].float() - ms[key].float()).to(dtype=b.dtype)
    return out


def _shifted_ce(
    model: torch.nn.Module, batch: torch.Tensor, *, reduction: str = "mean"
) -> torch.Tensor:
    """CE with labels shifted left by one (olmo_core does not shift for you)."""
    input_ids = batch[:, :-1].contiguous()
    labels = batch[:, 1:].contiguous()
    out = model(input_ids=input_ids, labels=labels, loss_reduction=reduction)
    if hasattr(out, "loss"):
        return out.loss
    raise TypeError(f"expected LMOutputWithLoss, got {type(out)}")


@torch.no_grad()
def mean_ce(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    for batch in batches:
        batch = batch.to(device)
        loss = _shifted_ce(model, batch, reduction="sum")
        total += float(loss.item())
        tokens += batch[:, 1:].numel()
    return total / max(tokens, 1)


def train_steps(
    model: torch.nn.Module,
    steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    lr: float,
    seed: int,
) -> float:
    """Short CE train on synthetic token batches. Returns final train CE."""
    if steps <= 0:
        return float("nan")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    last = float("nan")
    for step in range(steps):
        batch = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        loss = _shifted_ce(model, batch, reduction="mean")
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step % max(steps // 5, 1) == 0 or step + 1 == steps:
            log.info("train step %d/%d ce=%.4f", step + 1, steps, last)
    return last


def make_eval_batches(
    n: int, batch_size: int, seq_len: int, vocab_size: int, seed: int
) -> List[torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return [
        torch.randint(0, vocab_size, (batch_size, seq_len), generator=g) for _ in range(n)
    ]


def retention(score_mt: float, score_merge: float, score_ms: float, score_ft: float) -> float:
    """
    For CE (lower is better): gain on source is score_ms - score_ft.
    Gain kept on target is score_mt - score_merge.
    """
    denom = score_ms - score_ft
    if abs(denom) < 1e-8:
        return float("nan")
    return (score_mt - score_merge) / denom


def run_synthetic_twin(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    vocab = args.vocab_size
    model, _ = build_model(vocab_size=vocab, device=device)
    log.info("synthetic_twin: warming Ms for %d steps", args.steps_pre)
    train_steps(
        model,
        args.steps_pre,
        args.batch_size,
        args.seq_len,
        vocab,
        device,
        args.lr,
        seed=args.seed,
    )
    ms = clone_state(model)
    log.info("synthetic_twin: continuing %d steps to form Mt", args.steps_gap)
    train_steps(
        model,
        args.steps_gap,
        args.batch_size,
        args.seq_len,
        vocab,
        device,
        args.lr,
        seed=args.seed + 1,
    )
    mt = clone_state(model)

    load_state(model, ms, device)
    log.info("synthetic_twin: SFT on Ms for %d steps", args.steps_sft)
    train_steps(
        model,
        args.steps_sft,
        args.batch_size,
        args.seq_len,
        vocab,
        device,
        args.lr,
        seed=args.seed + 2,
    )
    ft = clone_state(model)

    return evaluate_transfer(model, ms, mt, ft, args, device, mode="synthetic_twin")


def run_team_s3(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    vocab = args.vocab_size
    model, _ = build_model(vocab_size=vocab, device=device)
    ms_sd = load_state_dict(args.source_uri)
    mt_sd = load_state_dict(args.target_uri)
    load_into_model(model, ms_sd)
    ms = clone_state(model)
    # Load Mt into a fresh copy for the base of the merge.
    model_t, _ = build_model(vocab_size=vocab, device=device)
    load_into_model(model_t, mt_sd)
    mt = clone_state(model_t)
    del model_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    load_state(model, ms, device)
    log.info("team_s3: SFT on Ms for %d steps", args.steps_sft)
    train_steps(
        model,
        args.steps_sft,
        args.batch_size,
        args.seq_len,
        vocab,
        device,
        args.lr,
        seed=args.seed + 2,
    )
    ft = clone_state(model)
    return evaluate_transfer(
        model,
        ms,
        mt,
        ft,
        args,
        device,
        mode="team_s3",
        extra={"source_uri": args.source_uri, "target_uri": args.target_uri},
    )


def evaluate_transfer(
    model: torch.nn.Module,
    ms: Mapping[str, torch.Tensor],
    mt: Mapping[str, torch.Tensor],
    ft: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    eval_batches = make_eval_batches(
        args.eval_batches, args.batch_size, args.seq_len, args.vocab_size, seed=args.seed + 99
    )
    merged = apply_delta(mt, ms, ft, args.lambda_scale)

    load_state(model, ms, device)
    ce_ms = mean_ce(model, eval_batches, device)
    load_state(model, ft, device)
    ce_ft = mean_ce(model, eval_batches, device)
    load_state(model, mt, device)
    ce_mt = mean_ce(model, eval_batches, device)
    load_state(model, merged, device)
    ce_merge = mean_ce(model, eval_batches, device)

    ret = retention(ce_mt, ce_merge, ce_ms, ce_ft)
    # CE: lower is better. Positive retention means merge improved Mt toward FT's gain.
    go = bool(ret == ret and ret > 0.25 and ce_merge < ce_mt)

    result: Dict[str, Any] = {
        "mode": mode,
        "lambda": args.lambda_scale,
        "steps_pre": args.steps_pre,
        "steps_gap": args.steps_gap,
        "steps_sft": args.steps_sft,
        "ce": {
            "ms": ce_ms,
            "ft_ms": ce_ft,
            "mt": ce_mt,
            "mt_plus_delta": ce_merge,
        },
        "retention_ce": ret,
        "go_for_close_gap_transfer": go,
        "interpretation": (
            "go: Mt+Δ beat Mt and kept >25% of the Ms→FT CE gain (close-gap transfer looks usable)"
            if go
            else "no-go / weak: prefer transfer-then-finetune, regenerate, or redo post-train on final"
        ),
        "paper": "https://arxiv.org/abs/2503.20110",
    }
    if extra:
        result.update(extra)
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("synthetic_twin", "team_s3"), default="synthetic_twin")
    p.add_argument("--source-uri", default=DEFAULT_MS)
    p.add_argument("--target-uri", default=DEFAULT_MT)
    p.add_argument("--save-folder", type=str, required=True)
    p.add_argument("--steps-pre", type=int, default=60)
    p.add_argument("--steps-gap", type=int, default=60)
    p.add_argument("--steps-sft", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=100_278)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--param-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="named for the platform command scanner; smoke uses float32 compute by default",
    )
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    # Acknowledge platform-visible flag (dtype is forced float32 for the smoke loop).
    log.info("param-dtype flag=%s (smoke train loop uses float32)", args.param_dtype)

    save = Path(args.save_folder)
    if str(save).startswith("s3://"):
        local_out = Path(os.environ.get("TMPDIR", "/tmp")) / "ckpt_transfer_smoke"
        local_out.mkdir(parents=True, exist_ok=True)
    else:
        save.mkdir(parents=True, exist_ok=True)
        local_out = save

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s mode=%s", device, args.mode)

    t0 = time.time()
    if args.mode == "team_s3":
        result = run_team_s3(args, device)
    else:
        result = run_synthetic_twin(args, device)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["device"] = str(device)

    out_path = local_out / "transfer_smoke_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("wrote %s", out_path)
    print(json.dumps(result, indent=2))

    if str(save).startswith("s3://"):
        from olmo_core.io import upload

        dest = f"{save.rstrip('/')}/transfer_smoke_results.json"
        upload(out_path, dest, save_overwrite=True)
        log.info("uploaded %s", dest)

    # Exit 0 even on no-go — the experiment succeeded; the scientific answer may be negative.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
