# Submitting Worked-Examples via eduLLM ORCD (MIT)

**Additive only.** This folder does not change `main` unsupervised — the
`worked-examples-cpt` profile lives on branch
`hypothesis/we-metamath-wandb-smoke` (and must be the **Commit SHA** on the job
Issue so the operator runs that tree).

## MIT operator handoff (copy/paste)

```text
Please run Worked Examples + Faded Scaffolds (4 CPT arms) on ORCD.

Already filed (eduLLM jobs) — please execute / unstick:
  #23 bare       (status:ready)
  #24 complete   (status:ready)
  #25 fade_ordered (status:assigned)
  #26 fade_shuffled (status:assigned)
Allowlist ask: #21

Use commit on branch hypothesis/we-metamath-wandb-smoke (update Issue Commit SHA
to the tip that includes train_cpt_arm + worked-examples-cpt profile).

Profile (locked by policy on that commit):
  worked-examples-cpt
  script: src/scripts/hypothesis/worked_examples/train_cpt_arm.py
  2×H100, 360 min, torchrun --nproc-per-node=2
  model: olmo2_760M
  load-path: /orcd/pool/edullm/checkpoints/OLMo-Ladder-760M-0.5xC-core
  pack-dir: /orcd/pool/edullm/data/worked-examples-metamath-v0
  token-budget: 200000000 (matched across arms)
  W&B: entity eduLLM / project pretraining / group = study name

Before launch, operators must stage:
1) Convert HF → OLMo-core Ladder ckpt:
   python src/examples/huggingface/convert_checkpoint_from_hf.py \
     -i allenai/OLMo-Ladder-760M-0.5xC -m olmo2_760m -t dolma2 \
     -o /orcd/pool/edullm/checkpoints/OLMo-Ladder-760M-0.5xC-core
2) Pool copy of HF pack https://huggingface.co/datasets/hiyasvyas/worked-examples-metamath-v0
   For each arm bare|complete|fade_ordered|fade_shuffled:
     tokenized/<arm>/shard-00000.npy
     tokenized/<arm>/label_mask-00000.npy   # required for fade loss mask
   Plus eval/holdout_bare.jsonl
   If masks missing: run tokenize_arms.py --pack-dir <pool> --tokenizer dolma2
3) Manifest /orcd/pool/edullm/manifests/worked-examples-metamath-v0.json
   Replace the placeholder digest on Issues #23–#26 (currently all "b"s) with the
   real SHA-256 of that manifest.

Success metrics: finite train/PPL + eval/pass_at_n + eval/pass_ratio_at_n
(on unscaffolded holdout_bare; generation/Pass@N may be post-train if not wired live).

Do NOT use 4×H100 (policy max_gpu_count=2).
Do NOT use generic-smoke for this study.
Do NOT use the AWS 370M / B200 path for these Issues.
```

## What is locked in `worked-examples-cpt`

| Item | Value |
|------|--------|
| GPUs | **2× H100** |
| Token budget | **200M** / arm |
| Model | **olmo2_760M** (`--model-factory`) |
| Init | converted Ladder **760M-0.5xC** at pool path above |
| Pack | pool MetaMath WE pack + **label_mask** per arm |
| W&B | `eduLLM` / `pretraining` |

## Scientific Issues

| Arm | Issue |
|-----|--------|
| bare | #23 |
| complete | #24 |
| fade_ordered | #25 |
| fade_shuffled | #26 |

## Local W&B env (never commit secrets)

```bash
export WANDB_API_KEY="$(cat "$HOME/.config/edullm/wandb.key")"
export WANDB_ENTITY="eduLLM"
export WANDB_PROJECT="pretraining"
```

## Branch hygiene

```bash
git status --porcelain   # clean before asking for a new Commit SHA stamp
# branch = hypothesis/we-metamath-wandb-smoke (not main)
# HEAD pushed to edu-llm/OLMo-core
```
