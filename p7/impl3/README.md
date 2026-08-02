# P7 — The Tutor Layer ("Rosenshine at the Interface")

Post-training code that turns an instruction-following model into a **step-level
Socratic tutor**, and the low-KL/forgetting-aware SFT family that keeps the tutoring
gains without forgetting math. This is a fresh, cluster-first implementation of the
P7 PRD (`../PRDs/P7 PRD.docx`), redoing Impls 1 & 2 more rigorously and then building
the new Impls 3 & 4.

> **Data & evals:** reused from the POC. Training data is the published dataset
> `meric533/socrateach-sft` (the POC 30k mix: 75% Socratic pedagogy + 25% general
> replay; `train`/`validation`/`test`). The POC math-retention, general-IF, and
> Socratic pedagogy evals are ported under `eval/` and apply directly. See `eval/README.md`.

## What's current (this phase)
**Impls 1 & 2 are done** — the recipe is unchanged from the POC, and their data +
checkpoints were saved earlier, so we **do not rerun them**. This phase runs the new
**Impls 3 & 4** on the *same* `socrateach-sft` data, using the saved Impl-2 model as
the vanilla baseline (and for Impl-3 variant b). The only intended difference from the
POC is **log-spaced checkpointing** (`checkpoint_schedule: log`, default in the Impl-3/4
configs) — steps 1,2,3,4,8,16,32,64,128,… — which densely samples the fast early
trajectory so the low-KL knee of the RL's-Razor curve is resolved (the POC's uniform
steps 20/40 were already far from base). LR stays at `2e-4`.

```bash
pip install -r requirements.txt
export WANDB_API_KEY=...
# Data streams from the Hub (hf_dataset in each config). Offline GPU node? Run
#   python snapshot_hf_dataset.py         # -> data/socrateach_sft_{train,val,test}.jsonl
# on the login node, then pass --data_dir data (drops the Hub dependency at train time).

# Impl 3 sweep (variant a shown); variant b reuses the saved Impl-2 model:
for T in 2 4 8 16 32; do
  python impl3_kl_reweighted_sft/train_kl_sft.py --variant a --temperature $T \
      --config impl3_kl_reweighted_sft/config.yaml; done
```
Then build the KL–forgetting curve (KL axis + retention/pedagogy + plots) per `eval/README.md`.

## Layout
```
post-training/
├── common/                     shared library (imported by every entrypoint)
│   ├── system_instructions.py  Impl-1 prompt, canonical eval SI, per-dialogue SI generator (§2.2)
│   ├── chat.py                 chat template + assistant-only loss masking (§2.4)
│   ├── data.py                 SI-prefix + co-training mix + group splits (§2.1–2.5)  [data hooks]
│   ├── modeling.py             model/tokenizer/LoRA/dtype loading
│   ├── sft_train.py            the shared SFT core + WeightedTrainer (Impl 2/3/4)
│   ├── weighting.py            Impl-3 per-token weight signals + global norm (§3.2–3.3)
│   ├── kl.py                   forward KL(pi_0 || pi) per PRD convention
│   └── cli.py                  shared CLI flags -> SFTConfig
├── impl1_2_prompting_sft/      Impl 2 (SI-conditioned SFT) — the vanilla baseline
├── impl3_kl_reweighted_sft/    Impl 3 (KL-reweighted loss)
├── clusters/orcd/              SLURM runners: the sweep, the T=451 control, per-checkpoint eval
├── eval/                       math forgetting probe, pedagogy judge, per-checkpoint sweep + figures
├── snapshot_hf_dataset.py      snapshot meric533/socrateach-sft to local JSONL (offline nodes)
├── run_kl_curve.py             KL axis of the KL–forgetting curve, over any run's checkpoints
└── requirements.txt
```

This branch is scoped to the Impl-3 experiment. Impl 1 (prompting-only) and Impl 4 (SDFT) are
part of the P7 PRD but contributed nothing to these results, so their scripts are not carried
here; the AWS runner is likewise absent because every run was done on ORCD.

## The four implementations (increasing difficulty)
| # | What | Where | Changes vs Impl 2 |
|---|------|-------|-------------------|
| **1** | Prompting-only Socratic tutor (system prompt) | not on this branch | — (no training) |
| **2** | SI-conditioned SFT (LoRA, co-trained 75/25) | `impl1_2_prompting_sft/train_sft.py` | baseline |
| **3** | Low-KL SFT via **KL-reweighted loss** | `impl3_kl_reweighted_sft/` | loss reweighting (objective change) |
| **4** | Low-KL SFT via **self-distillation (SDFT)** | not on this branch | self-distilled targets (data change) |

Impls 1 & 2 are one 2×2 experiment (`{Raw,SFT}×{no-SI,+SI}`); Impls 3 & 4 each target
the RL's-Razor goal of matching Impl-2 pedagogy at lower new-task KL / less forgetting,
and must **Pareto-beat** Impl 2. Impl 5 (a second SDFT) and Impl 6 (RLHF) are **not to
be implemented yet** (PRD) and are absent here by design.

### Cross-cutting principles (baked in)
- **SI vs no-SI** is a first-class axis everywhere (two of the four eval cells; both
  KL numbers `kl_new_SI` / `kl_ped_noSI`).
- **Checkpoint sweep:** every training run keeps **≥10 checkpoints** (auto-sized
  `save_steps`, `save_total_limit=None`) so the full trajectory feeds the KL–forgetting
  curve — not just the end state. Set `checkpoint_schedule: log` (default in the Impl-3/4
  configs) to sample steps 1,2,3,4,8,16,32,64,… — dense where the trajectory moves fastest,
  so the low-KL knee is resolved instead of jumping straight to a far-from-base point.
- **Base model / KL reference:** `allenai/OLMo-2-0425-1B-Instruct` (the Instruct
  checkpoint, not the pretrained base). Swap with `--base_model` to post-train our own
  model instead.

## Logging
W&B logging is **on by default** (`report_to: wandb`, project `edullm-p7`) — set
`WANDB_API_KEY` (and optionally `WANDB_PROJECT` / `WANDB_ENTITY`) before running, or pass
`--no_wandb` to disable. HF Trainer also records per-checkpoint train/eval loss to each
`output_dir/trainer_state.json` regardless.

## An `s3://` output dir (the eduLLM platform)
`--output_dir` accepts an `s3://` URI, which is what `$EDULLM_CHECKPOINT_DIR` holds on the
platform. The run trains to a local staging directory and mirrors it to that prefix as each
checkpoint lands. HF's `Trainer` has no notion of object storage, so without this it creates
a local directory literally named `s3:`, exits 0, and leaves the prefix empty. See
`common/s3_io.py` for the full argument.

```bash
python impl1_2_prompting_sft/train_sft.py --config impl1_2_prompting_sft/config.yaml \
    --output_dir "$EDULLM_CHECKPOINT_DIR" --run_name "$EDULLM_RUN_ID"
```

| Behaviour | What happens |
| --- | --- |
| Staging | `/tmp/edullm-sft/<prefix slug>`, or `$EDULLM_LOCAL_OUTPUT_DIR` if set. A full run stages roughly 1.7 GB |
| Upload cadence | Every checkpoint, plus the final model and tokenizer. Never on a timer, so no half-written file is ever sent |
| Resume | `--resume auto` downloads the newest remote `checkpoint-N` first, so a Batch retry continues instead of restarting |
| Failure | Any upload error ends the run non-zero. Write access is probed before the model loads, so a bad prefix costs seconds rather than a GPU hour |
| Deletes | Never. `save_total_limit` rotation applies to local disk only, and S3 keeps every checkpoint |

Nothing changes for a local `--output_dir`, which is what every ORCD run uses.

## Quick start (once you have data)
```bash
pip install -r requirements.txt          # install torch first for your CUDA (see clusters/orcd/setup_orcd_env.sh)
export WANDB_API_KEY=...                  # W&B is on by default
# Data streams from the Hub; python snapshot_hf_dataset.py materialises it for an offline node.

# 1. Impl 2 baseline (the vanilla SFT every Impl-3 config is judged against)
python impl1_2_prompting_sft/train_sft.py --config impl1_2_prompting_sft/config.yaml --output_dir out/impl2-sft

# 2. Impl 3 sweep (variant a shown; variant b also needs --sft_model_id <the Impl-2 adapter>)
for T in 2 4 8 16 32; do python impl3_kl_reweighted_sft/train_kl_sft.py --variant a --temperature $T --config impl3_kl_reweighted_sft/config.yaml; done

# 3. Score every checkpoint, then plot
python eval/sweep_ckpt_eval.py --out out/ckpt_sweep_bare_hint250.jsonl
bash eval/make_figures.sh
```
On a cluster use `clusters/orcd/` (SLURM), which wraps
the same entrypoints; sweeps fan out as independent single-GPU jobs.

## Stack note — why HF Transformers + PEFT, not OLMo-core's native SFT
OLMo-core's `src/scripts/train/sft/*` is its own high-throughput distributed trainer
(sharded DCP checkpoints, its own data/optim stack) built for **pretraining-scale**
runs on the dolma2 pipeline. This P7 work is small (1B, LoRA) and needs custom,
per-token control of the loss — Impl 3 reweights individual tokens' cross-entropy, and
Impls 3/4 lean on the HF/PEFT ecosystem (adapters, `generate`, chat templates) for the
self-distillation and KL passes. Doing that inside OLMo-core's trainer would mean
fighting its abstractions; the HF `Trainer` + a small `WeightedTrainer` subclass makes
the reweighting a few lines and keeps the four implementations sharing one core. The
tradeoff is we don't get OLMo-core's distributed throughput — fine at this scale (a LoRA
epoch is ~1h on one L40S). If we later post-train a much larger model, revisit.
