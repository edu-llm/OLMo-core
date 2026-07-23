# Worked Examples + Faded Scaffolds (smoke / scaffold)

**Additive only.** Does not modify upstream OLMo-core training code.

**Primary source (current):** [`meta-math/MetaMathQA`](https://huggingface.co/datasets/meta-math/MetaMathQA)  
**Legacy source:** [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (`main`) via `build_from_gsm8k.py`  
**Regime:** CPT from an early OLMo checkpoint (run cards below)  
**Not used as treatment:** Dolma / week1 corpus (base pretrain only)

## Arms

| Arm | Description |
|-----|-------------|
| 1 `bare` | problem + final answer only |
| 2 `complete` | problem + full step-by-step solution |
| 3 `fade_ordered` | decreasing scaffold length; shown prefix = context, omitted suffix = train target |
| 4 `fade_shuffled` | same scaffold-length multiset as 3, random order within family |

Equal **token budget** across arms at train time (arm 1 cycles more short docs). Equal **family roster** (MetaMath: `original_question`), not equal tokens per pass.

## Quick start (MetaMathQA)

```bash
cd /path/to/OLMo-core

# 1) Build bank + 4 arms (GSM_* types by default; holdout by original_question family)
python src/scripts/hypothesis/worked_examples/build_from_metamath.py \
  --out-dir ./data/worked_examples_metamath_v0 \
  --max-train 10000 \
  --max-holdout 1000 \
  --types GSM

# Full-ish GSM slice (no caps):
#   --max-train 0 --max-holdout 0

# 2) Validate scaffolds / answers
python src/scripts/hypothesis/worked_examples/validate_pack.py \
  --pack-dir ./data/worked_examples_metamath_v0

# 3) Tokenize with dolma2 → uint32 .npy + label_mask per arm
python src/scripts/hypothesis/worked_examples/tokenize_arms.py \
  --pack-dir ./data/worked_examples_metamath_v0 \
  --tokenizer allenai/dolma2-tokenizer

# 4) Confirm eval JSONL
python src/scripts/hypothesis/worked_examples/export_eval.py \
  --pack-dir ./data/worked_examples_metamath_v0

# 5) CPT one arm (after HF→core convert of Ladder 760M-0.5xC)
# torchrun --standalone --nproc-per-node=1 \
#   src/scripts/hypothesis/worked_examples/train_cpt_arm.py RUN \
#   --arm fade_ordered --pack-dir ./data/worked_examples_metamath_v0 \
#   --load-path /path/to/converted-ckpt --token-budget 200000000
```

Windows: same commands in PowerShell.

## Outputs

```text
data/worked_examples_metamath_v0/
  bank/instances.jsonl
  meta/fade_schedule.json
  meta/splits.json
  arms/
    bare/docs.jsonl
    complete/docs.jsonl
    fade_ordered/docs.jsonl
    fade_shuffled/docs.jsonl
  eval/holdout_bare.jsonl
  tokenized/<arm>/shard-00000.npy
  tokenized/<arm>/label_mask-00000.npy
  reports/validation.json
```

## CPT note

Use `train_cpt_arm.py` with the **same** `--token-budget` and converted Ladder 760M-0.5xC init for every arm. Fade arms use `label_mask-00000.npy` (from `loss_start_char`). Holdout metrics: `holdout_passn.py` / `PassNEvalCallback` → `eval/pass_at_n`, `eval/pass_ratio_at_n`. See `run_cards/` and `OPERATOR_ALLOWLIST.md`.

## eduLLM / W&B / ORCD submit

See **`SUBMIT.md`**, **`run_smoke_wandb.md`**, **`OPERATOR_ALLOWLIST.md`**.

- Metrics go to W&B entity **`eduLLM`** (projects allowlisted in policy: `test`, `pretraining`, …).
- First engineering verify: `/submit-edullm-job` with **generic-smoke** fixture (Issue #20).
- Full MetaMath 4-arm CPT is **not** in policy yet — use `OPERATOR_ALLOWLIST.md` + `request_drafts/`.
- Never commit W&B keys; use `src/scripts/orcd/wandb.env.example`.
