# Running P7 post-training on MIT ORCD (Engaging / SLURM)

Cluster port of the P7 SFT recipe. LoRA on a 1B model needs only **one L40S** on the
free `mit_normal_gpu` tier (up to 2 GPUs / 6h). All entrypoints `--resume auto`, so a
job that hits the wall-time resumes from its last checkpoint.

## Step 0 — SSH access (one time, on your Mac)
```bash
ssh-copy-id <KERBEROS>@orcd-login.mit.edu   # Kerberos password + Duo
ssh <KERBEROS>@orcd-login.mit.edu
```
Optional `~/.ssh/config` for fewer Duo prompts:
```
Host orcd
    HostName orcd-login.mit.edu
    User <KERBEROS>
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

## Step 1 — Copy the project up
From your Mac (whole `post-training/` folder is self-contained):
```bash
rsync -avP --exclude out --exclude '**/__pycache__' \
    ~/Documents/MericXing/MIT/Intern/AlphaAI/Training_Team/post-training orcd:~/
```

## Step 2 — One-time env (login node)
```bash
ssh orcd
cd ~/post-training
bash clusters/orcd/setup_orcd_env.sh     # conda env "p7post" + deps; uninstalls torchao
export HF_HOME=/orcd/pool/<yourpath>/hf_cache   # cache big downloads off your home quota
export WANDB_API_KEY=<your key>          # W&B logging is ON by default (report_to: wandb)
# optional: export WANDB_PROJECT=edullm-p7  WANDB_ENTITY=<team>
```
> W&B env vars propagate to jobs via `sbatch --export=ALL`. If you don't want W&B, add
> `--no_wandb` to the `CMD` (or set `WANDB_MODE=offline`).

> **Data** streams from the Hub (`hf_dataset: meric533/socrateach-sft` in each config) —
> no prepare step needed. Offline GPU node? On the **login node** run
> `python snapshot_hf_dataset.py` (writes `data/socrateach_sft_{train,val,test}.jsonl`),
> then pass `--data_dir data` so training reads the local copy.

## Step 3 — Smoke test on a GPU (recommended)
```bash
salloc -p mit_normal_gpu -G l40s:1 -c 16 --mem=64G -t 01:00:00
source ~/miniforge3/etc/profile.d/conda.sh && conda activate p7post
cd ~/post-training
python impl3_kl_reweighted_sft/train_kl_sft.py --variant a --temperature 2 \
    --config impl3_kl_reweighted_sft/config.yaml --train_total 400 --num_epochs 1
exit
```

## Step 4 — Submit the runs (batch)
**Impls 1 & 2 are already done — do NOT rerun them.** Reuse the saved Impl-2 model as
the vanilla baseline / Impl-3 variant-b reference; point `submit_sweep.sh impl3 b` at it.
Run these from `~/post-training` (paths are relative to it):
```bash
# Impl 3 (KL-reweighted, log-spaced checkpoints): variant a (base-surprise) all temperatures
bash clusters/orcd/submit_sweep.sh impl3 a
# variant b (forward-KL) needs the saved vanilla SFT (adjust the path to where it lives):
bash clusters/orcd/submit_sweep.sh impl3 b out/impl2-sft

# Impl 4 (SDFT): make self-distilled targets first (needs a local snapshot), then sweep
python snapshot_hf_dataset.py     # -> data/socrateach_sft_train.jsonl
python impl4_sdft/self_distill.py --mode rewrite --in_file data/socrateach_sft_train.jsonl \
    --out_file impl4_sdft/distilled/pedagogy_rewrite.jsonl --quality_gate
bash clusters/orcd/submit_sweep.sh impl4
```
Or submit any single command directly:
```bash
sbatch --export=ALL,CMD="python impl3_kl_reweighted_sft/train_kl_sft.py --variant a --temperature 2 --config impl3_kl_reweighted_sft/config.yaml --resume auto" clusters/orcd/run.sbatch
```

Monitor:
```bash
squeue -u $USER
tail -f logs/p7-*.out
```

## GPU / memory sizing (free tier)
- **1× L40S (44 GB)** per run. Gradient checkpointing ON + `per_device_batch=8`,
  `grad_accum=4` (effective 32) fits comfortably. OOM? `--per_device_batch 4
  --grad_accum 8` (same effective batch) or lower `--max_len`.
- A 30k-example epoch is ~1h on an L40S, so one 4h job finishes with headroom; the
  sweeps fan out as independent 1-GPU jobs and SLURM queues them.
- This is the operator-BYPASS path (direct SLURM). Your team's governed path is the
  GitHub → ORCD operator (`config/edullm/`); coordinate with the lead before large runs.
