"""SFT entrypoint for the dense/split comparison. One arm per invocation.

    torchrun --nproc-per-node=1 src/scripts/train/p3_math_split/train.py --arm dense --config configs/dense.yaml
    torchrun --nproc-per-node=1 src/scripts/train/p3_math_split/train.py --arm split --config configs/split.yaml

Everything that must be identical between the arms is derived from the shared config
block and asserted at startup — see `assert_controls()`. The only thing `--arm` changes
is which `label_mask_*.npy` the dataset reads. If you find yourself adding a second
`if arm == ...` anywhere in this file, that is the experiment breaking.

The run writes `<save_folder>/arm_fingerprint.json`, which records the seed, the sha256
of the token array, the step/token/batch plan, and the resolved optimizer settings.
`compare_arms.py` refuses to compare two runs whose fingerprints disagree on
anything except the arm name and the mask file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_module import FixedDivisorTransformerTrainModule  # noqa: E402

from olmo_core.config import DType  # noqa: E402
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig  # noqa: E402
from olmo_core.distributed.utils import get_world_size  # noqa: E402
from olmo_core.nn.transformer import (  # noqa: E402
    build_qwen2_0_5b,
    load_hf_weights,
    parameter_report,
    qwen2_tokenizer_config,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup  # noqa: E402
from olmo_core.train import (  # noqa: E402
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (  # noqa: E402
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.utils import seed_all  # noqa: E402

log = logging.getLogger(__name__)

ARMS = ("dense", "split")


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def assert_controls(cfg: dict, arm: str, data_meta: dict) -> dict:
    """Resolve the run plan and refuse to start if a control is unspecified.

    The point of this function is that the plan is computed from the *shared* section
    of the config, so the two arms cannot be given different values by editing one file
    and forgetting the other.
    """
    if arm not in ARMS:
        raise SystemExit(f"--arm must be one of {ARMS}")

    shared = cfg["shared"]
    S = data_meta["sequence_length"]
    if shared["sequence_length"] != S:
        raise SystemExit(
            f"config sequence_length {shared['sequence_length']} != tokenized data "
            f"{S}. Re-run tokenize_corpus.py or fix the config; they must agree or the "
            f"dataset will slice instances across example boundaries."
        )

    world = get_world_size()
    gbs_seqs = shared["global_batch_size_sequences"]
    gbs_tokens = gbs_seqs * S
    if gbs_tokens % world != 0:
        raise SystemExit(f"global batch {gbs_tokens} tokens not divisible by world size {world}")

    micro_seqs = shared["rank_microbatch_size_sequences"]
    rank_micro_tokens = micro_seqs * S
    rank_tokens = gbs_tokens // world
    if rank_tokens % rank_micro_tokens != 0:
        raise SystemExit(
            f"rank batch {rank_tokens} tokens is not a multiple of rank microbatch "
            f"{rank_micro_tokens}; grad accumulation would be uneven between arms"
        )

    n_instances = data_meta["n_instances"]
    steps = shared.get("max_steps")
    if steps is None:
        epochs = shared.get("epochs", 1)
        steps = (n_instances * epochs) // gbs_seqs
    if steps < 1:
        raise SystemExit("computed 0 training steps — corpus too small for this batch size")

    warmup = shared["warmup_steps"]
    if warmup >= steps:
        raise SystemExit(
            f"warmup_steps ({warmup}) >= max_steps ({steps}): the learning rate would "
            f"never reach peak and never decay, so the schedule control is meaningless. "
            f"Lower warmup_steps in BOTH configs, or train longer. "
            f"({n_instances:,} instances / {gbs_seqs} per step x "
            f"{shared.get('epochs', 1)} epochs = {steps} steps)"
        )
    if warmup > steps // 10:
        print(
            f"  note: warmup is {warmup}/{steps} steps ({warmup / steps:.0%} of the run); "
            f"under ~10% is typical"
        )

    return {
        "arm": arm,
        "seed": shared["seed"],
        "sequence_length": S,
        "global_batch_size_sequences": gbs_seqs,
        "global_batch_size_tokens": gbs_tokens,
        "rank_microbatch_size_tokens": rank_micro_tokens,
        "grad_accum_steps": rank_tokens // rank_micro_tokens,
        "world_size": world,
        "max_steps": steps,
        "total_input_tokens": steps * gbs_tokens,
        "n_instances": n_instances,
        "epochs_equivalent": round(steps * gbs_seqs / n_instances, 3),
        "learning_rate": shared["learning_rate"],
        "warmup_steps": shared["warmup_steps"],
        "weight_decay": shared["weight_decay"],
        "betas": tuple(shared["betas"]),
        "eps": shared["eps"],
        "max_grad_norm": shared["max_grad_norm"],
        "lr_alpha_f": shared["lr_alpha_f"],
        "tie_embeddings": shared["tie_embeddings"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="tokenized")
    ap.add_argument("--split", default="train")
    ap.add_argument("--save-folder", default=None)
    ap.add_argument("--work-dir", default="/tmp/olmo_work")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("arm") != args.arm:
        raise SystemExit(
            f"--arm {args.arm} but {args.config} declares arm={cfg.get('arm')!r}. "
            f"Refusing to run: this is the one setting that must not be ambiguous."
        )

    prefix = os.path.join(args.data_dir, args.split)
    with open(f"{prefix}_meta.json", encoding="utf-8") as f:
        data_meta = json.load(f)

    plan = assert_controls(cfg, args.arm, data_meta)
    save_folder = args.save_folder or f"runs/{args.arm}"

    tokens_path = f"{prefix}_tokens.npy"
    mask_path = f"{prefix}_label_mask_{args.arm}.npy"
    for p in (tokens_path, mask_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"{p} not found — run src/scripts/train/p3_math_split/tokenize_corpus.py"
            )

    if args.dry_run:
        print(json.dumps({**plan, "tokens": tokens_path, "label_mask": mask_path}, indent=2))
        return

    prepare_training_environment(seed=plan["seed"])
    try:
        seed_all(plan["seed"])  # explicit: init, shuffling, and dropout all key off this

        tokenizer = qwen2_tokenizer_config()

        # Both arms read the SAME tokens file. Only label_mask_paths differs.
        dataset_config = NumpyFSLDatasetConfig(
            tokenizer=tokenizer,
            paths=[tokens_path],
            label_mask_paths=[mask_path],
            sequence_length=plan["sequence_length"],
            work_dir=args.work_dir,
        )
        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=plan["global_batch_size_tokens"],
            seed=plan["seed"],
            num_workers=cfg["shared"].get("num_workers", 2),
            work_dir=args.work_dir,
        )

        model = build_qwen2_0_5b(
            dtype=DType.bfloat16,
            tie=plan["tie_embeddings"],
            init_seed=plan["seed"],
        )
        load_hf_weights(model)
        log.info("model: %s", parameter_report(model))

        train_module = FixedDivisorTransformerTrainModule(
            model=model,
            # The whole reason this subclass exists. Constant, and identical in both
            # arms, so the split arm's proof tokens are not silently up-weighted.
            fixed_loss_div_factor=plan["global_batch_size_tokens"],
            optim=AdamWConfig(
                lr=plan["learning_rate"],
                betas=plan["betas"],
                eps=plan["eps"],
                weight_decay=plan["weight_decay"],
            ),
            rank_microbatch_size=plan["rank_microbatch_size_tokens"],
            max_sequence_length=plan["sequence_length"],
            scheduler=CosWithWarmup(warmup=plan["warmup_steps"], alpha_f=plan["lr_alpha_f"]),
            max_grad_norm=plan["max_grad_norm"],
            compile_model=cfg["shared"].get("compile_model", False),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        dataset = dataset_config.build()
        data_loader = data_loader_config.build(dataset)

        trainer_config = (
            TrainerConfig(
                save_folder=save_folder,
                work_dir=args.work_dir,
                load_strategy=LoadStrategy.if_available,  # resume-safe; init is from HF
                max_duration=Duration.steps(plan["max_steps"]),
                metrics_collect_interval=cfg["shared"].get("log_every", 10),
                save_overwrite=cfg["shared"].get("save_overwrite", False),
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback("gc", GarbageCollectorCallback())
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=cfg["shared"].get("save_every", 500),
                    ephemeral_save_interval=None,
                    save_async=False,
                ),
            )
        )

        if cfg["shared"].get("wandb_project"):
            from olmo_core.train.callbacks.wandb import WandBCallback

            trainer_config = trainer_config.with_callback(
                "wandb",
                WandBCallback(
                    project=cfg["shared"]["wandb_project"],
                    name=f"{cfg['shared'].get('run_name', 'qwen-mm')}-{args.arm}",
                    cancel_check_interval=10,
                ),
            )

        trainer = trainer_config.build(train_module, data_loader)

        fingerprint = {
            **plan,
            "tokens_sha256": sha256_file(tokens_path),
            "label_mask_path": os.path.basename(mask_path),
            "label_mask_sha256": sha256_file(mask_path),
            "data_meta": data_meta,
            "supervised_tokens_this_arm": data_meta[f"supervised_tokens_{args.arm}"],
        }
        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, "arm_fingerprint.json"), "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)
        log.info("fingerprint: %s", json.dumps(fingerprint, indent=2))

        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
