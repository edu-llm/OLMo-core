"""P1 quality pilot: does the activation-energy proxy predict from-scratch training quality?

THE QUESTION
------------
``F-r128`` (low-rank gates) and ``G-grouped`` (block-diagonal gates) are **exactly**
parameter-matched -- 323,157,760 each, a bit-identical 15,728,640 below ``L0``. So any difference
between them is pure quality, with cost held fixed.

On Liquid's *already-trained* weights, low-rank r=128 retains **0.929** of activation-weighted
energy while grouped g=4 retains **0.130** -- and 0.130 is identical to a random mask of the same
density, so block structure buys nothing there. If that proxy predicts from-scratch training, this
pilot sees a large gap. If it does not, the 12-day study is built on a metric that does not transfer.
GaLore is a documented case of exactly this failure: plain ``W = BA`` collapses to 142.53 ppl vs
15.56 at 1B.

**A null result here is informative, not a wasted run** -- but only if it can be distinguished from
undertraining. See the next section, which is the load-bearing part of this design.

CAN A NULL JUST MEAN "TOO SHORT"?
---------------------------------
Yes, and that is the sharpest objection to any pilot at 1.18 tokens/param. Two arms that would
diverge at 20 tok/param can look identical at 1, because early loss is dominated by learning
token frequencies -- something every arm does equally well regardless of gate structure. A single
end-of-run number cannot tell the two situations apart.

So the pilot does not report a single number. It evaluates held-out loss at a **geometric ladder**
of steps (5%, 10%, 20%, 35%, 50%, 75%, 100% of the run) and the endpoint is the **trajectory of the
between-arm gap**, not the final value:

* gap **grows** across rungs -> the effect is real and this budget already sees it;
* gap **flat and near zero while loss is still falling steeply** -> UNDERTRAINED. Not a null.
  The honest conclusion is "this budget cannot answer the question", and the next move is more
  tokens on the two cost-matched arms only, not a verdict;
* gap **flat and near zero while the curves have flattened** -> a real null over this regime.

That third case is still not a claim about 20 tok/param. It is a claim that the 0.929-vs-0.130
energy proxy does not predict *early* training, which is exactly what would make the proxy unusable
as a cheap pre-screen -- the use the 12-day design had in mind for it.

**Two things this pilot cannot do, regardless of outcome:**

1. **Rank architectures.** An ordering at 1.18 tok/param can reverse by 20. This screens a *metric*.
2. **Rule out a recall difference.** The endpoint is loss. Hymba measured a 20.75-point recall gap
   at near-identical perplexity, so a loss null says nothing about retrieval. MQAR is calibrated
   (``mqar/``) and is the follow-up if loss comes back flat.

WHY THESE FOUR ARMS
-------------------
``L0``         stock LIV, the control every P1 claim is measured against.
``F-r128``     low-rank gates. The treatment.
``G-grouped``  block-diagonal gates. Exactly cost-matched to ``F-r128`` -- this is the clean pair.
``N-narrow``   just build a narrower dense model. Answers the reviewer question "why not do the
               obvious thing instead?", and is matched to ``F-r128`` within 0.0095%.

TOKEN BUDGET, AND WHY IT IS NOT "2 GPU-HOURS"
---------------------------------------------
An earlier plan said ~2 GPU-hours total. Measured throughput on an L40S is ~36,800 tok/s, so 2
GPU-hours over 16 runs is ~17M tokens each = **0.05 tokens/param**, against a Chinchilla-optimal 20.
Every arm would still be in the initial loss drop; the comparison would be seed noise.

The floor is set by measured noise instead. Repeated seeds of identical configs in this repo's KDA
study (13 runs, same cluster) give a within-config SD of **0.0105 nats**, so with paired seeds:

    n=2 -> resolves 0.015 nats      n=4 -> resolves 0.010
    n=3 -> resolves 0.012 nats      n=8 -> resolves 0.007

For scale: Mamba-2's whole 4-23%-attention sweep spans 0.06 nats, and DeltaProduct's
parameter-matched contrast is 0.0053 (needing ~43 seeds). So a pilot can only resolve a *large*
effect -- which is precisely what the 0.929-vs-0.130 proxy predicts. ``--tokens 400M --seeds 3`` is
the default: 4 arms x 3 seeds x 400M = 4.8B tokens, ~36 GPU-hours, ~9 h wall-clock on 4 GPUs.

PAIRING
-------
Seed ``s`` fixes model init *and* data order identically across arms, so arm differences are
measured on the same data in the same order. That is what makes the paired MDD above valid rather
than the ~1.4x-larger independent-samples figure.

USAGE
-----
    python run_pilot.py --arm F-r128 --seed 0 --tokens 400M
    python run_pilot.py --dry-run          # print the plan for every arm, build nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.nn.transformer.liv_arms import ARMS, VOCAB_SIZE, build_arm
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    MetricSaverCallback,
)
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

PILOT_ARMS = ("L0", "F-r128", "G-grouped", "N-narrow")

DATA_ROOT = "/scratch/users/ericrcwu/liv/data"
OUT_ROOT = "/scratch/users/ericrcwu/liv/pilot"

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 128 * SEQUENCE_LENGTH  # 524,288 tokens/step
RANK_MICROBATCH_SIZE = 2 * SEQUENCE_LENGTH  # measured: mbs=4 fits, mbs=8 OOMs at 44 GiB
LEARNING_RATE = 3e-4
WARMUP_FRACTION = 0.02

# The corpus is GPT-2 tokenized; its largest id is 50,256 (EOS, at every document boundary).
# OLMo-core derives the same padded vocab independently -- asserted in main().
TOKENIZER = TokenizerConfig.gpt2()


def parse_tokens(s: str) -> int:
    """Accept 400M / 1.5B / 400_000_000."""
    s = s.strip().replace("_", "")
    mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    if s and s[-1].lower() in mult:
        return int(float(s[:-1]) * mult[s[-1].lower()])
    return int(s)


def build_everything(arm: str, seed: int, tokens: int) -> Dict[str, Any]:
    """Assemble model/data/train-module/trainer configs for one run."""
    if tokens % GLOBAL_BATCH_SIZE:
        log.warning(
            "token budget %s is not a multiple of the global batch size %s; rounding down",
            f"{tokens:,}",
            f"{GLOBAL_BATCH_SIZE:,}",
        )
    steps = tokens // GLOBAL_BATCH_SIZE
    if steps < 100:
        raise ValueError(f"{steps} steps is too few for a cosine schedule to mean anything")

    run_name = f"{arm}-s{seed}-{tokens // 1_000_000}M"

    model_cfg = build_arm(arm)
    # Same seed -> same init across arms, so the comparison is paired.
    model_cfg.init_seed = seed

    dataset_cfg = NumpyFSLDatasetConfig(
        paths=[f"{DATA_ROOT}/train_raw.bin"],
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=TOKENIZER,
        work_dir=f"{OUT_ROOT}/work",
    )

    # Same seed -> same data order across arms. This is the other half of the pairing.
    data_loader_cfg = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=seed, num_workers=4
    )

    train_module_cfg = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=LEARNING_RATE,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=max(10, int(WARMUP_FRACTION * steps))),
        compile_model=True,
    )

    # Validation loss on a held-out split at a LADDER of checkpoints, not just at the end.
    #
    # This is what makes an absence of a gap interpretable. A single end-of-run number cannot
    # distinguish "these arms are equivalent" from "400M tokens is too early for any arm to have
    # differentiated yet" -- and at 1.18 tokens/param the second is a live possibility. A ladder
    # answers it directly: if the between-arm gap is flat and near zero at every rung while the
    # loss itself is still dropping steeply, the run is too short to conclude anything. If the gap
    # is near zero and *stable* while the curves have flattened, the arms really are equivalent
    # over this regime.
    #
    # Geometric spacing because loss falls roughly log-linearly in tokens, so evenly-spaced steps
    # would put most rungs in the flat tail where they carry the least information.
    eval_steps = sorted({max(1, int(steps * f)) for f in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)})

    eval_cfg = LMEvaluatorCallbackConfig(
        eval_dataset=NumpyFSLDatasetConfig(
            paths=[f"{DATA_ROOT}/val_raw.bin"],
            sequence_length=SEQUENCE_LENGTH,
            tokenizer=TOKENIZER,
            work_dir=f"{OUT_ROOT}/work",
        ),
        eval_interval=None,
        fixed_steps=eval_steps,
        eval_on_finish=True,
        # 8M val tokens is far more than needed per rung and would cost more than training.
        # 64 batches x 128 x 4096 = ~33M tokens... cap by duration instead.
        eval_duration=Duration.steps(16),
    )

    trainer_cfg = (
        TrainerConfig(
            save_folder=f"{OUT_ROOT}/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=50,
            max_duration=Duration.steps(steps),
            no_checkpoints=True,  # a pilot needs the loss curve, not resumable weights
            # NOTE: no_evals must stay False. Trainer._iter_callbacks filters out every
            # EvaluatorCallback when it is True (trainer.py:1225), which would silently drop the
            # ladder above and leave only the final training loss -- reintroducing exactly the
            # ambiguity the ladder exists to remove.
            no_evals=False,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        # gc_interval must be small enough to actually fire within this run.
        #
        # GarbageCollectorCallback calls gc.disable() in pre_train and then only ever runs
        # gc.collect(1) every gc_interval steps (garbage_collector.py:30-42). At the default
        # 1000 a run shorter than 1000 steps disables automatic GC and never collects -- so
        # cyclic garbage, including the CPU staging buffers a checkpoint leaves behind,
        # accumulates untouched for the whole run. The pilot is 762 steps, i.e. squarely in
        # that window. 100 keeps collection happening without paying for it every step.
        .with_callback("garbage_collector", GarbageCollectorCallback(gc_interval=100))
        .with_callback("metric_saver", MetricSaverCallback())
        .with_callback("lm_eval", eval_cfg)
    )

    return {
        "run_name": run_name,
        "arm": arm,
        "seed": seed,
        "tokens": steps * GLOBAL_BATCH_SIZE,
        "steps": steps,
        "model": model_cfg,
        "dataset": dataset_cfg,
        "data_loader": data_loader_cfg,
        "train_module": train_module_cfg,
        "trainer": trainer_cfg,
    }


def assert_gate_variance_parity(model, arm: str) -> Dict[str, float]:
    """Gate output variance at step 0 must match ``L0``'s, or the sweep measures init scale.

    Low-rank gate init has ``Var(y) = d * r * sigma_A^2 * sigma_B^2``. Using a fixed 0.02 for both
    factors is 24-48x too small, and **the error is monotone in r** -- so a rank sweep with fixed
    std produces a smooth, plausible "higher rank is better" curve that is really an init-scale
    curve. Measuring this is the only way to know which one you have.

    Reported, not enforced, because the right reference is ``L0`` measured in the same process;
    the driver compares across arms.
    """
    del arm
    stats: Dict[str, float] = {}
    with torch.no_grad():
        for i, block in model.blocks.items():
            mixer = block.attention
            if type(mixer).__name__ != "ShortConv":
                continue
            device = next(mixer.parameters()).device
            x = torch.randn(4, 64, mixer.d_model, device=device, dtype=torch.float32)
            # in_proj returns (pre_gate, post_gate, value) -- the gates are the first two.
            pre_gate, post_gate, _ = mixer.in_proj(x)
            stats[f"block{i}.pre_gate_var"] = float(pre_gate.var())
            stats[f"block{i}.post_gate_var"] = float(post_gate.var())
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=PILOT_ARMS, help="which arm to train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens", default="400M", help="token budget per run, e.g. 400M")
    p.add_argument("--seeds", type=int, default=3, help="for --dry-run accounting only")
    p.add_argument("--dry-run", action="store_true", help="print the plan; build nothing")
    args = p.parse_args()

    tokens = parse_tokens(args.tokens)

    if args.dry_run:
        print(f"vocab {VOCAB_SIZE:,} (OLMo-core gpt2 padded: {TOKENIZER.padded_vocab_size():,})")
        assert TOKENIZER.padded_vocab_size() == VOCAB_SIZE
        print(f"seq {SEQUENCE_LENGTH:,}  global batch {GLOBAL_BATCH_SIZE:,} tok")
        print(f"budget {tokens:,} tok/run  ->  {tokens // GLOBAL_BATCH_SIZE:,} steps")
        print(f"tokens/param at 338.9M: {tokens / 338_886_400:.3f}  (Chinchilla-optimal is 20)")
        print()
        print(f"{'arm':<12}{'params':>13}{'vs L0':>9}{'flops@4K':>12}{'ShortConv':>11}")
        base = None
        for arm in PILOT_ARMS:
            cfg = build_arm(arm)
            n = cfg.num_params
            base = base or n
            # num_flops_per_token lives on the built model, not the config. Build on meta so
            # this stays cheap and needs no GPU.
            built = cfg.build(init_device="meta")
            kinds = [type(b.attention).__name__ for b in built.blocks.values()]
            assert kinds.count("ShortConv") == ARMS[arm].n_liv_layers, arm
            print(
                f"{arm:<12}{n:>13,}{n / base:>8.4f}x"
                f"{built.num_flops_per_token(SEQUENCE_LENGTH):>12.3e}"
                f"{kinds.count('ShortConv'):>11}"
            )
        total = len(PILOT_ARMS) * args.seeds * tokens
        print()
        print(f"{len(PILOT_ARMS)} arms x {args.seeds} seeds = "
              f"{len(PILOT_ARMS) * args.seeds} runs, {total:,} tokens")
        for tps in (30_000, 36_800):
            gpu_h = total / tps / 3600
            print(f"  at {tps:,} tok/s: {gpu_h:6.1f} GPU-h -> {gpu_h / 4:5.1f} h wall on 4 GPUs")
        print()
        print("noise floor: within-config SD 0.0105 nats (13 KDA runs, same cluster)")
        for n in (2, 3, 4, 8):
            print(f"  n={n} paired seeds resolves {2.0 * 0.0105 / math.sqrt(n):.4f} nats")
        print()
        steps = tokens // GLOBAL_BATCH_SIZE
        ladder = sorted({max(1, int(steps * f)) for f in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)})
        print("held-out eval ladder (steps):", ladder)
        print("  = tokens:", [f"{s * GLOBAL_BATCH_SIZE / 1e6:.0f}M" for s in ladder])
        print("  The endpoint is the TRAJECTORY of the between-arm gap across these rungs.")
        print("  A flat ~zero gap while loss is still falling steeply means UNDERTRAINED,")
        print("  not equivalent -- that distinction is why the ladder exists.")
        return

    if args.arm is None:
        p.error("--arm is required unless --dry-run")

    prepare_training_environment()
    try:
        plan = build_everything(args.arm, args.seed, tokens)
        run_name = plan["run_name"]
        log.info("=== %s: %s steps, %s tokens ===", run_name, plan["steps"], f"{plan['tokens']:,}")

        seed_all(args.seed)

        model = plan["model"].build(init_device="meta" if is_distributed() else "cuda")
        # build() constructs modules but does NOT initialize them -- on a real device the
        # parameter memory is left uninitialized. Skipping this gives a step-0 loss around 900
        # instead of ln(vocab)=10.83, and everything downstream still "runs".
        model.init_weights(max_seq_len=SEQUENCE_LENGTH, device=torch.device("cuda"))
        gate_stats = assert_gate_variance_parity(model, args.arm)

        train_module = plan["train_module"].build(model)

        # Assert the topology is what was declared. Per-layer overrides go through
        # `block.sequence_mixer`; setting `block.attention` on a *config* silently no-ops and
        # yields an all-attention model that trains fine and answers a different question.
        kinds = [type(b.attention).__name__ for b in model.blocks.values()]
        n_liv = kinds.count("ShortConv")
        expected_liv = ARMS[args.arm].n_liv_layers
        assert n_liv == expected_liv, f"{args.arm}: {n_liv} ShortConv layers, want {expected_liv}"
        log.info("topology OK: %d ShortConv, %d Attention", n_liv, kinds.count("Attention"))

        dataset = plan["dataset"].build()
        dataset.prepare()
        data_loader = plan["data_loader"].build(dataset)
        trainer = plan["trainer"].build(train_module, data_loader)

        t0 = time.perf_counter()
        trainer.fit()
        wall = time.perf_counter() - t0

        if get_rank() == 0:
            out = {
                "run_name": run_name,
                "arm": args.arm,
                "seed": args.seed,
                "tokens": plan["tokens"],
                "steps": plan["steps"],
                "params": plan["model"].num_params,
                "flops_per_token_4k": plan["model"].num_flops_per_token(SEQUENCE_LENGTH),
                "wall_seconds": wall,
                "tokens_per_sec": plan["tokens"] / wall,
                "world_size": get_world_size(),
                "gate_variance_step0": gate_stats,
                "vocab_size": VOCAB_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            }
            Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
            dest = Path(OUT_ROOT) / f"{run_name}.json"
            dest.write_text(json.dumps(out, indent=2))
            log.info("wrote %s (%.0f tok/s)", dest, out["tokens_per_sec"])
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
    main()
