# Implementations 1 & 2 — Prompting + SI-conditioned SFT

Impl 1 (prompting) and Impl 2 (SI-conditioned SFT) are documented and evaluated
**together** as one 2×2 factorial over `{Raw, SFT} × {no-SI, +SI}` (PRD §3):

|            | no-SI                    | +SI (canonical)                |
|------------|--------------------------|--------------------------------|
| **Raw**    | A — floor / control      | **B — Implementation 1**       |
| **SFT**    | C — should act normal    | **D — Implementation 2**       |

## Files
| File | What it does |
|---|---|
| `prompt_tutor.py`  | **Impl 1**: prompting-only tutor (interactive or batch first-turns). No training. |
| `train_sft.py`     | **Impl 2**: vanilla SI-conditioned SFT (LoRA), keeps ≥10 checkpoints. Thin wrapper over `common.sft_train`. |
| `generate_2x2.py`  | Produces the A/B/C/D model outputs for a held-out test set. |
| `config.yaml`      | Recipe defaults (PRD §2.6). |

## Run order (once data exists — it is blank now)
```bash
# 0. Prepare data (from repo root): see ../prepare_data.py. Produces data/socrateach_sft_{train,val,test}.jsonl
# 1. Impl 2 SFT
python train_sft.py --config config.yaml --output_dir out/impl2-sft
# 2. Impl 1 behavior (prompting only)
python prompt_tutor.py --interactive                 # or --problems your_problems.jsonl
# 3. 2×2 outputs for scoring (eval scoring itself is out of scope / blank)
python generate_2x2.py --adapter_dir out/impl2-sft --test_file data/socrateach_sft_test.jsonl
```

## Definition of done (PRD §4)
- **Impl 1**: prompted tutor beats an answer-supplying baseline on P5's rubric (CIs).
- **Impl 2**: D (SFT+SI) beats B (prompt-only) on pedagogy; C (SFT no-SI) still behaves
  like a normal assistant (SI-gating holds); old-task retention reported. If SFT does
  not clear prompting, stay at Impl 1.

Impl 2 is the **baseline** that Impls 3 & 4 must Pareto-beat (match pedagogy at lower
KL/forgetting). Its checkpoints feed the KL–forgetting curve (`../common/kl.py`).
