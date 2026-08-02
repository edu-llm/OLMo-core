"""Export a trained OLMo-core checkpoint to a HuggingFace directory for evaluation.

    python src/scripts/train/p3_math_split/export_checkpoint.py --run runs/split

Reads the latest checkpoint under ``<run>/``, unshards it, maps parameter names back to
Qwen2 via ``olmo_core.nn.transformer.qwen.export_to_hf_state_dict``, and writes a directory
``transformers.AutoModelForCausalLM.from_pretrained`` can load. ``run_eval.py`` takes that
directory. Generation through HuggingFace is much faster than through the training model
(KV cache, batched ``generate``) and keeps the eval harness usable with vLLM later.
"""

from __future__ import annotations

import argparse
import json
import os
import re

from olmo_core.nn.transformer.qwen import QWEN2_0_5B_HF_ID, export_to_hf_state_dict


def latest_step_dir(run_dir: str) -> str:
    """Pick the highest-numbered ``stepN`` directory the checkpointer wrote."""
    candidates = []
    for name in os.listdir(run_dir):
        m = re.fullmatch(r"step(\d+)", name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            candidates.append((int(m.group(1)), name))
    if not candidates:
        raise SystemExit(
            f"no stepN checkpoint directories under {run_dir}. "
            f"Contents: {sorted(os.listdir(run_dir))[:10]}"
        )
    return os.path.join(run_dir, max(candidates)[1])


def write_hf_dir(olmo_state_dict, out_dir: str, *, tied: bool, hf_id: str) -> None:
    """Materialize a HuggingFace model directory from an OLMo-core state dict."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(hf_id)
    config.tie_word_embeddings = tied
    model = AutoModelForCausalLM.from_config(config)

    hf_sd = export_to_hf_state_dict(olmo_state_dict, tied=tied)
    hf_sd = {k: v.to(dtype=torch.float32) for k, v in hf_sd.items()}

    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    # With tying, HuggingFace materializes lm_head.weight from the embedding, so it is
    # expected to be missing here. Anything else is a real gap and must not pass silently.
    real_missing = [k for k in missing if k != "lm_head.weight"]
    if real_missing or unexpected:
        raise RuntimeError(
            f"export mismatch: missing={real_missing[:5]} unexpected={list(unexpected)[:5]}"
        )

    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(hf_id).save_pretrained(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/split")
    ap.add_argument("--out", default=None, help="defaults to <run>/hf")
    ap.add_argument("--step", default=None, help="specific checkpoint dir; default is latest")
    ap.add_argument("--hf-id", default=QWEN2_0_5B_HF_ID)
    args = ap.parse_args()

    import torch

    from olmo_core.distributed.checkpoint import unshard_checkpoint

    out_dir = args.out or os.path.join(args.run, "hf")
    ckpt = args.step or latest_step_dir(args.run)
    model_dir = os.path.join(ckpt, "model_and_optim")
    if not os.path.exists(model_dir):
        model_dir = ckpt
    print(f"unsharding {model_dir}")

    tmp = os.path.join(args.run, "_unsharded")
    os.makedirs(tmp, exist_ok=True)
    model_path, _ = unshard_checkpoint(
        dir=model_dir, target_dir=tmp, optim=False, save_overwrite=True
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = state.get("model", state)

    tied = True
    fingerprint_path = os.path.join(args.run, "arm_fingerprint.json")
    if os.path.exists(fingerprint_path):
        with open(fingerprint_path, encoding="utf-8") as f:
            tied = json.load(f).get("tie_embeddings", True)

    print(f"exporting (tied={tied}) -> {out_dir}")
    write_hf_dir(state, out_dir, tied=tied, hf_id=args.hf_id)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
