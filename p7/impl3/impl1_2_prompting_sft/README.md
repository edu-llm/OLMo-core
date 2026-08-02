# Implementation 2 — SI-conditioned SFT

Vanilla SFT on the Socratic mix, conditioned on a per-dialogue system instruction. In the
Impl-3 experiment this is the **baseline** — the thing every reweighted configuration has to
beat — and it appears in the results as `SFT` (the run tag is `impl2-rerun`).

Impl 1 (prompting-only, no training) is part of the same PRD section but contributed nothing
to the 192-checkpoint sweep, so `prompt_tutor.py` and the 2x2 generation script are not on
this branch. See the `p7/POC` branch for that work.

## Files
| File | What it does |
|---|---|
| `train_sft.py` | Vanilla SI-conditioned SFT (LoRA). Thin wrapper over `common.sft_train`. |
| `config.yaml`  | Recipe defaults (PRD §2.6). |

## Run
```bash
# Data streams from the Hub (hf_dataset: meric533/socrateach-sft in config.yaml).
python train_sft.py --config config.yaml --output_dir out/impl2-rerun --run_name impl2-rerun
```

Two things make the baseline comparable to the Impl-3 runs, and both are already in
`config.yaml`: `checkpoint_schedule: log` (steps 1, 2, 3, 4, 8, ... 923, so the fast early
trajectory is sampled densely) and the same LR, LoRA rank, and epoch count as the sweep.
Changing any of them makes the baseline a different experiment rather than a control.

## Note on run-to-run variance
Two vanilla SFT runs of this exact recipe scored 0.64 and 0.53 on the pedagogy judge. The gap
between them is larger than several of the differences the sweep is trying to resolve, so treat
any single baseline number as noisy and prefer the in-batch anchor from the same judging round.
