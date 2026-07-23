# Submitting Worked-Examples via eduLLM ORCD

**Additive only.** This folder does not change `main` or `config/edullm/policy.yaml`.

## What is already approved for `/submit-edullm-job`

Policy today allows these entrypoint profiles:

| Profile | Script | Data | Use for |
|---------|--------|------|---------|
| `generic-smoke` | `src/examples/llm/train.py` | `builtin://generic-smoke-v1` | Engineering smoke (W&B + ORCD path check) |
| `hypothesis-smoke` | `src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py` | skill-dag / curriculum manifests | Skill-DAG / curriculum only |

**MetaMath / worked-examples CPT is not yet a policy entrypoint.**  
Do not invent a new profile. Ask operators before changing `policy.py`.

### Recommended first submit (engineering)

Use the **generic-smoke** fixture so the queue + W&B `eduLLM/test` path is verified:

- W&B entity: `eduLLM` (fixed by policy)
- W&B project: `test`
- Metrics emitted by `src/examples/llm/train.py` + `WandBCallback` include
  `train/CE loss` (and related train/optim metrics such as throughput / LR groups)
- GPUs: 1× L40S, 30 minutes (fixture)

Then invoke `/submit-edullm-job` from a **clean**, **pushed** feature branch (not `main`).

## Scientific CPT (needs operator allowlist — do not invent policy)

| Item | Value |
|------|--------|
| Train script | `train_cpt_arm.py` (this folder) |
| Data (HF) | https://huggingface.co/datasets/hiyasvyas/worked-examples-metamath-v0 |
| Train shards | `tokenized/{arm}/shard-00000.npy` + `label_mask-00000.npy` |
| Eval | `eval/holdout_bare.jsonl` → `eval/pass_at_n`, `eval/pass_ratio_at_n` |
| Base ckpt | https://huggingface.co/allenai/OLMo-Ladder-760M-0.5xC (convert via HF→core) |
| Arms | 4 matched-token-budget CPT jobs |
| Fade | `label_mask` from `loss_start_char` (re-tokenize after pull) |
| Operator ask | **`OPERATOR_ALLOWLIST.md`** |
| Request drafts | `request_drafts/arm_*.json` (stamp SHA with `fill_sha.py`) |

Until operators add `worked-examples-cpt` on `main` and publish an `/orcd/pool/...` manifest, `/submit-edullm-job` will reject scientific Issues. Do **not** reuse `generic-smoke` for this study.

## Local W&B env (never commit secrets)

```bash
# On Engaging / operator machine — see src/scripts/orcd/wandb.env.example
export WANDB_API_KEY="$(cat "$HOME/.config/edullm/wandb.key")"
export WANDB_ENTITY="eduLLM"
export WANDB_PROJECT="test"
export WANDB_GROUP="worked-examples"
```

## Branch hygiene for submit gate

```bash
git status --porcelain   # must be empty
# branch != main
# HEAD must be pushed to edu-llm/OLMo-core
```
