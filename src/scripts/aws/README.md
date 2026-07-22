# AWS training scheduler (Training team)

This folder holds the tooling for running OLMo-core training on **AWS** via a shared
**SLURM** scheduler that every hypothesis group submits to. It is the AWS counterpart to
`../lambda/` (which targets Ai2's own clusters) and deliberately avoids the Beaker/Gantry
`launch` path — we use the plain `train` command under `torchrun`.

> **Status: scaffold.** The scripts here are ready in structure but have a few `FINALIZE ME`
> spots that depend on which compute channel DevOps gives us (a self-managed SLURM GPU box,
> AWS ParallelCluster, or SageMaker HyperPod). See [What still needs deciding](#what-still-needs-deciding).

## Why this exists

Different groups do different things — OLMo pre-training architecture experiments, SFT on
OLMo, open-source (e.g. Qwen) runs. The scheduler's only job is: **allocate N GPUs and run
whatever branch + environment a group hands it.** Each job is self-describing (branch +
training script + GPU count + config overrides), so environments never collide and runs
queue automatically when GPUs are busy.

**Policy:** heavy pre-training experiments run at the **~370M** rung
(`src/scripts/train/OLMo2/OLMo2-370M.py`). An even smaller/faster `OLMo2-190M.py` is also
available for quick iteration. Both are small-model variants of the official `OLMo2-1B.py`.

## Workflow for hypothesis groups

1. **Branch.** Do your work on a branch of `edu-llm/OLMo-core` and push it:
   ```bash
   git checkout -b my-group/my-experiment
   # ...make changes...
   git push -u origin my-group/my-experiment
   ```
2. **Submit.** From a login node (reachable via AWS SSM), source your env and submit:
   ```bash
   source env.sh
   ./submit.sh my-group/my-experiment OLMo2/OLMo2-370M.py my-exp-lr8e4 1 \
       -- --train_module.optim.lr=8e-4 --dataset.mix=<your-mix>
   ```
   Arguments: `<branch> <train_script> <run_name> <gpus> [-- <config overrides...>]`.
3. **Queue.** SLURM runs your job when a GPU is free; extras wait. Check with `squeue`,
   cancel with `scancel <jobid>`, tail logs in `logs/<run_name>-<jobid>.log`.
4. **Outputs.** Checkpoints are written to `${OLMO_CHECKPOINT_S3}/<run_name>` in S3. The
   evals team reads them from there.

The scheduler checks out your exact branch on the compute node per job, so groups never step
on each other. Data is read from S3 (pass `--dataset.mix_base_dir=$OLMO_DATA_S3/...`).

## Concurrency (on one 8×H100 node)

Total work per run is ~24–32 GPU-hours, so the same node can trade parallelism for
per-run speed. Pick the `<gpus>` argument to `submit.sh` accordingly:

| Concurrent runs | GPUs/run (`<gpus>`) | Approx time/run |
|---:|---:|---:|
| 1 | 8 | 3–4 h |
| 2 | 4 | 5–8 h |
| 4 | 2 | 10–15 h |
| 8 | 1 | 20–30 h |

We're targeting **2–4 concurrent runs** (`<gpus>` = 4 or 2). SLURM queues anything beyond the
node's 8 GPUs. Times are rough and scale sub-linearly with GPUs/run.

## Model recipes & caveats (370M / SFT)

Where these scripts come from and what to double-check before trusting the numbers.

**Architecture — canonical, safe.** `olmo2_370M` and `olmo2_190M` are official model-ladder
configs defined in `src/olmo_core/nn/transformer/config.py` (e.g. `olmo2_370M` = `d_model=1024`,
16 layers, 16 heads). These were not invented; the ladder there defines sizes from 1M → 32B.
Note the ladder only fixes the model *shape* — it does **not** prescribe LR / batch size / token
budget.

**Pre-training (`OLMo2-370M.py`) — derived from the official `OLMo2-1B.py`.** Same optimizer
(`SkipStepAdamW`), scheduler (`CosWithWarmup`, 2000 warmup), dtype/HSDP setup. Two deliberate
deltas from the 1B recipe, both heuristic scalings for the smaller model (not from an official
370M recipe):

| Setting | 1B (official) | 370M (ours) | Note |
|---|---|---|---|
| `lr` | `4e-4` | `8e-4` | heuristic — **verify with a short run** or fall back to `4e-4` |
| `GLOBAL_BATCH_SIZE` | `512 × seq` | `256 × seq` | heuristic |
| `MAX_DURATION` | `4e12` tokens | `4e12` tokens | inherited — **override per experiment** with `--trainer.max_duration=...`; 4T tokens is a full-scale budget |

**SFT (`sft/Olmo-2-370M-SFT.py`) — adapted from the official `sft/Olmo-2-7B-SFT.py`.** The recipe
(`lr=8e-5`, `weight_decay=0.0`, `betas=(0.9,0.95)`, `LinearWithWarmup` warmup 0.03, 3 epochs,
`dolma2` tokenizer, packed FSL dataset) is inherited **verbatim from the 7B script**. Our changes
are structural: decoupled from Beaker (runs via `train`/`dry_run`), `--gpus_per_node` is
configurable, and default `seq_len=4096`. ⚠️ `lr=8e-5` was tuned for 7B and is likely **too low**
for a 370M model — sweep it.

**No built-in LR table here.** The ladder training scripts (`ladder/*.py`) import
`olmo_core.model_ladder`, which is **absent from this checkout**, so they can't run and there is no
per-size LR formula to pull from in this repo.

**Verification status.** Scripts are syntax/lint-checked only — **not** dry-run validated yet.
Before a real run: (1) `dry_run` both scripts to confirm the configs build; (2) stage pre-tokenized
`dolma2` data in `$OLMO_DATA_S3`; (3) SFT additionally needs an `olmo2_370M` pre-training
checkpoint to load, so pre-train first.

## Files

| File | Purpose |
|------|---------|
| `submit.sh` | Thin wrapper each group runs to enqueue a run (`sbatch` under the hood). |
| `train-job.sbatch` | The SLURM job: clones the branch, sets up the env, launches `torchrun`. |
| `entrypoint.sh` | Resolves the torchrun rendezvous from the SLURM allocation and execs the run. |
| `env.example.sh` | Template for the per-user env (S3 buckets, repo URL, venv/image, WandB key). |
| `build_and_push_ecr.sh` | Builds the OLMo-core training image and pushes it to ECR (run on an x86_64 Linux Docker host). |

## What still needs deciding

These are gated on DevOps confirming how GPU compute is provisioned (the intern sandbox
boundary forbids launching GPU instances directly):

- **Compute channel:** self-managed SLURM on one GPU box, AWS ParallelCluster (autoscaling),
  or SageMaker HyperPod. Picks how the login/head node and GPU nodes are stood up.
- **Environment on the node:** prebuilt OLMo-core container in **ECR** (Option A in
  `train-job.sbatch`) vs a shared **virtualenv** (Option B). Container is recommended so the
  heavy deps (flash-attn, TransformerEngine) are baked in.
- **GPU architecture:** on H100 (`p5`) the stock OLMo Docker image works as-is; on A100/A10G
  add `--model.block.attention.backend=torch` to your submit overrides.
- **S3 access:** the compute nodes need an instance role (or profile) that can read
  `OLMO_DATA_S3` and write `OLMO_CHECKPOINT_S3`.
- **Region:** sandbox access is limited to `us-east-1` / `us-east-2` (us-east-2 = Ohio, closest).
