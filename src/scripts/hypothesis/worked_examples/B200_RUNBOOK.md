# B200 weekend runbook — Worked Examples / Faded Scaffolds

**Do not buy capacity. Do not stop/terminate the instance. Do not train until Hiya says go.**

| | |
|---|---|
| Instance | `i-05de75630c4774cdd` (`ms-135m-b200-node`) |
| Account / region | `056956104102` / `us-east-1` |
| Reserved block | **GPU6** (backup **GPU7**), Sun Jul 26 **17:00** → Mon Jul 27 **05:00** CDT (13 h on sheet) |
| Hard kill | Mon Jul 27 **06:00 CDT** / 11:00 UTC — sync to S3 before then |
| Model | OLMo **370M** (`--model-factory olmo3_370M`) + Nathan Chinchilla ckpt |
| Budget | **50M tokens / arm**, arms: bare → complete → fade_ordered → fade_shuffled |
| S3 out | `s3://memorysplit-stephen-056956104102-us-east-1/runs/worked-examples/` |

## 0) Before 17:00 (laptop)

1. Put your name on the GPU sheet for **GPU6**, **17:00–05:00**, **13 h**.
2. Confirm AWS: `aws --profile sbsandbox sts get-caller-identity` → account `056956104102`.
3. Install **Session Manager plugin** (separate from AWS CLI). On this laptop:
   - Admin PowerShell: `.\src\scripts\hypothesis\worked_examples\install_ssm_plugin_windows.ps1`
   - Or MSI: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html
   - Verify: `session-manager-plugin`
4. Ask Nathan/channel for the **exact path/S3 URI** of the 370M Chinchilla OLMo-core checkpoint.
5. Pull this branch on the box later: `hypothesis/we-metamath-wandb-smoke`.

## 1) Connect (when go)

```bash
aws --profile sbsandbox ssm start-session \
  --target i-05de75630c4774cdd \
  --region us-east-1
# then:
sudo su - ubuntu
nvidia-smi   # confirm GPU6 free; shout in channel
export CUDA_VISIBLE_DEVICES=6
```

Use the DLAMI PyTorch env under `/opt` (system `python3` has no torch).

## 2) Stage (script)

```bash
# on box, after clone:
cd /mnt/nvme/we/code/OLMo-core   # or wherever you clone
bash src/scripts/hypothesis/worked_examples/prepare_b200.sh
# edit CKPT_URI / PACK source inside the script or pass env vars first
```

Creates:

```text
/mnt/nvme/we/pack/          # HF pack + tokenized + label_mask
/mnt/nvme/we/ckpt/370m/     # Nathan ckpt
/mnt/nvme/we/runs/<arm>/    # outputs
```

If `label_mask-00000.npy` missing, the prepare script re-runs `tokenize_arms.py`.

## 3) Dry-run config only (safe, no long train)

```bash
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/mnt/nvme/we/code/OLMo-core/src
# activate DLAMI torch env as per login banner / ls /opt

torchrun --standalone --nproc-per-node=1 \
  src/scripts/hypothesis/worked_examples/train_cpt_arm.py we-cpt-bare-b200-dry \
  --arm bare \
  --pack-dir /mnt/nvme/we/pack \
  --load-path /mnt/nvme/we/ckpt/370m \
  --token-budget 50000000 \
  --model-factory olmo3_370M \
  --run-tag b200 \
  --global-batch-size 65536 \
  --rank-microbatch-size 8192 \
  --save-folder /mnt/nvme/we/runs/bare \
  --dry-run
```

## 4) Train (ONLY when Hiya says go)

```bash
export CUDA_VISIBLE_DEVICES=6
export ALLOW_TRAIN=1
bash src/scripts/hypothesis/worked_examples/run_arms_b200.sh
```

(`run_arms_b200.sh` exits unless `ALLOW_TRAIN=1`. Syncs each arm to S3 after train.)

After all arms (or if time is short): holdout Pass@N via `holdout_passn.py` on a fixed subset; sync again.

## 5) Hard stop checklist (before Mon 06:00 CDT)

- [ ] `aws s3 sync` all `/mnt/nvme/we/runs`  
- [ ] W&B runs visible under `eduLLM/pretraining`  
- [ ] Channel note: GPU6 released  
- [ ] **Never** stop/terminate the instance  

## ORCD note

Issues #23–#26 remain the long-run 760M/200M path. This B200 job is a **50M / 370M pilot**.
