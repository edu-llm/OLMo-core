"""Shared CLI plumbing so each implementation entrypoint stays short.

``build_sft_parser`` exposes every SFTConfig knob as a flag; ``--config file.yaml``
supplies defaults that flags then override. ``sft_config_from_args`` maps the parsed
namespace onto a ``common.sft_train.SFTConfig``.
"""
from __future__ import annotations

import argparse

from .sft_train import SFTConfig


def _load_yaml(path):
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_sft_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=None, help="YAML file of SFTConfig defaults (flags override).")
    p.add_argument("--base_model", default=None, help="HF id / path. Default: OLMo-2-0425-1B-Instruct.")
    p.add_argument("--start_from", choices=["base", "instruct"], default=None,
                   help="Shorthand for --base_model (instruct = the PRD base / KL reference).")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--hf_dataset", default=None,
                   help="HuggingFace Hub dataset id to train on (e.g. meric533/socrateach-sft); overrides --data_dir.")
    p.add_argument("--output_dir", default=None)

    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_epochs", type=float, default=None)
    p.add_argument("--train_total", type=int, default=None, help="Cap train examples (0 = whole file).")

    p.add_argument("--full_finetune", action="store_true", help="Full fine-tune instead of LoRA.")
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)

    p.add_argument("--per_device_batch", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--no_grad_checkpointing", action="store_true", help="Disable gradient checkpointing.")

    p.add_argument("--min_checkpoints", type=int, default=None, help="Target >= this many checkpoints.")
    p.add_argument("--save_steps", type=int, default=None, help="0 = auto from min_checkpoints.")
    p.add_argument("--save_total_limit", type=int, default=None, help="Default None = keep ALL.")
    p.add_argument("--checkpoint_schedule", choices=["uniform", "log"], default=None,
                   help="'log' = powers-of-two steps to densely sample the fast early trajectory.")
    p.add_argument("--eval_cap", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)

    p.add_argument("--resume", default=None, help="Checkpoint path or 'auto'.")
    p.add_argument("--report_to", default=None, help="'wandb' (default) or 'none'.")
    p.add_argument("--wandb_project", default=None, help="W&B project (default edullm-p7). Entity via WANDB_ENTITY env.")
    p.add_argument("--run_name", default=None)
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B (sets report_to=none).")
    return p


_START_FROM = {"base": "allenai/OLMo-2-0425-1B", "instruct": "allenai/OLMo-2-0425-1B-Instruct"}


def sft_config_from_args(args, **hard_overrides) -> SFTConfig:
    cfg_kwargs = {}
    if getattr(args, "config", None):
        cfg_kwargs.update({k: v for k, v in _load_yaml(args.config).items() if v is not None})

    # Map flags -> SFTConfig fields (only when explicitly provided).
    direct = ["base_model", "data_dir", "hf_dataset", "output_dir", "max_len", "seed", "num_epochs",
              "train_total", "lora_r", "lora_alpha", "lora_dropout", "per_device_batch",
              "grad_accum", "learning_rate", "warmup_ratio", "weight_decay", "min_checkpoints",
              "save_steps", "save_total_limit", "checkpoint_schedule", "eval_cap", "eval_steps",
              "logging_steps", "resume", "report_to", "wandb_project", "run_name"]
    for name in direct:
        v = getattr(args, name, None)
        if v is not None:
            cfg_kwargs[name] = v

    if getattr(args, "start_from", None):
        cfg_kwargs["base_model"] = _START_FROM[args.start_from]
    if getattr(args, "full_finetune", False):
        cfg_kwargs["use_lora"] = False
    if getattr(args, "no_grad_checkpointing", False):
        cfg_kwargs["grad_checkpointing"] = False
    if getattr(args, "no_wandb", False):
        cfg_kwargs["report_to"] = "none"

    cfg_kwargs.update(hard_overrides)
    return SFTConfig(**cfg_kwargs)
