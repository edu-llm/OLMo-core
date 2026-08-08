"""
Does raising --batch-size buy any throughput in the current loop?

codi_loss iterates examples one at a time (batch dim 1), so a bigger --batch-size adds
sequential forwards rather than widening any tensor. If that is true, step time should scale
~linearly with batch size and time-PER-EXAMPLE should stay flat. Real batching would show
per-example time falling.

CPU + tiny model: the point is the shape of the curve, not absolute speed. The sequential
Python loop is device-independent, so the conclusion carries to the GPU (where it is worse,
since batch-1 kernels are launch-bound).

Run: .venv/bin/python local/latentcot_batch_scaling.py
"""

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.arms import ARMS
from olmo_core.latentcot.data.dataset import LatentCotDataset
from olmo_core.latentcot.data.encode import to_sft_record
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.train_driver import train_arm
from olmo_core.nn.transformer import TransformerConfig

K = 10
STEPS = 4
BATCHES = [1, 2, 4, 8]


def main():
    with TemporaryDirectory() as td:
        path = Path(td) / "conversations" / "train-00000.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w") as f:
            for s in range(16):
                ex = generate(num_nodes=14, branching=2, depth=3, seed=s, reachable=bool(s % 2))
                f.write(json.dumps(to_sft_record(ex)) + "\n")
        dataset = LatentCotDataset(path, num_continuous_thoughts=K)

        cfg = TransformerConfig.llama_like(
            d_model=128, n_layers=2, n_heads=4, vocab_size=T.PADDED_VOCAB_SIZE
        )

        print(f"K={K}, {STEPS} steps per setting, arm A2, CPU tiny model")
        print(f"{'batch':>6} {'step time':>11} {'per example':>13} {'fwd/step':>9} {'speedup/ex':>11}")
        base = None
        for bs in BATCHES:
            torch.manual_seed(0)
            model = cfg.build(init_device="cpu")
            t0 = time.perf_counter()
            train_arm(
                model, ARMS["A2"], dataset, steps=STEPS, batch_size=bs,
                warmup_steps=1, seed=0, log_every=1000, precision="fp32",
            )
            per_step = (time.perf_counter() - t0) / STEPS
            per_ex = per_step / bs
            base = base or per_ex
            print(f"{bs:>6} {per_step*1e3:>9.1f}ms {per_ex*1e3:>11.1f}ms "
                  f"{bs*(K+2):>9} {base/per_ex:>10.2f}x")

        print("\nper-example time flat => --batch-size adds sequential work, no parallelism.")
        print("It is an effective-batch / gradient-noise knob at linear wall-clock cost,")
        print("NOT a throughput knob -- until the per-example loop is packed.")


if __name__ == "__main__":
    main()
