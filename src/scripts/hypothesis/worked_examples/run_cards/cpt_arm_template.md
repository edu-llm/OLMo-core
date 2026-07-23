# CPT run card template (send to cluster schools)

Fill in and send one card per arm.

```text
hypothesis: worked-examples-faded-scaffolds
branch: hypothesis/worked-examples-smoke
pack: <path>/worked_examples_metamath_v0
  # or HF: hiyasvyas/worked-examples-metamath-v0 (after upload)
source: meta-math/MetaMathQA  (GSM_* types; family = original_question)
arm: bare | complete | fade_ordered | fade_shuffled

model_script: <team 760M CPT launch — same as QA>
load_checkpoint: <same 760M-0.5xC base as QA; NOT QA-finetuned>
data_tokens: <pack>/tokenized/<arm>/shard-00000.npy

# Matched compute — SAME for all arms
max_tokens: <agree with training; keep modest vs pack size to limit memorization>
global_batch_tokens: <match team default>
save_folder: <shared>/runs/we-mm-<arm>-<date>
run_name: we-metamath-<arm>

notes:
- Do not mix unrelated Dolma filler into arm 1.
- Arm 1 may need more passes over its shard to hit max_tokens (shorter docs).
- Fade arms: mask loss before loss_start_char (scaffold = context).
- Eval: <pack>/eval/holdout_bare.jsonl  (unscaffolded; PassRatio@N)
```

Suggested smoke order on cluster:
1. `bare` 50–100 steps (path check)
2. all four arms at matched `max_tokens`
3. compare holdout curves
