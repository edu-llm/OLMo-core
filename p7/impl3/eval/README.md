# Evaluation — reused from the POC

These are the POC eval assets, ported for reuse on ORCD. Task = the Socratic math tutor
(`meric533/socrateach-sft`), so all three tracks apply exactly as in the POC:

| Track | What it measures | Location |
| --- | --- | --- |
| **KL axis** | New-task drift `KL(pi_0 ‖ pi)` from base | `../run_kl_curve.py` + `../common/kl.py` |
| **Math/logic retention** | OLD-task ability (forgetting probe), deterministic exact-match | `math_eval/` |
| **General instruction-following** | OLD-task ability (forgetting probe); IFEval (deterministic, paper) or MT-Bench judge | `general_eval/` |
| **Socratic pedagogy judge** | NEW-task tutoring quality | `llm_judge/` |

The KL–forgetting curve pairs the **KL axis** (drift) with **math retention**
(forgetting), colored by step; the **pedagogy judge** is the y-axis for the RL's-Razor
Pareto plots that Impls 3 & 4 must beat vs the vanilla Impl-2 baseline.

## End-to-end flow (per training run)

1. **Train.** Impl 1&2 are already done (reuse the saved Impl-2 checkpoints/data). For
   the NEW Impl 3&4 runs, `checkpoint_schedule: log` (set in their configs) yields
   `checkpoint-{1,2,3,4,8,16,32,64,128,256,512,...}` under `out/<run>/`.

2. **KL axis** — forward KL of each checkpoint vs the base, on held-out pedagogy prompts
   (use the dataset's `validation`/`test` split — Socratic dialogues):
   ```bash
   python ../run_kl_curve.py --base_model allenai/OLMo-2-0425-1B-Instruct \
       --ckpt c16=out/<run>/checkpoint-16 c64=out/<run>/checkpoint-64 ... \
       --pedagogy_file heldout_pedagogy.jsonl --out out/<run>/kl_by_checkpoint.json
   ```

3. **Math retention (forgetting probe) — fully deterministic** — generate base-vs-checkpoint
   outputs, then grade by exact match:
   ```bash
   python generate_eval.py --prompts math_eval/math_logic_prompts.jsonl \
       --adapter ../out/<run>/checkpoint-16 --out math_eval/results_c16.jsonl
   cd math_eval && python grade_math_logic.py results_c16.jsonl   # -> base vs sft accuracy
   ```
   MATH-500 (the `expr` items needing symbolic/LLM verification) has been **removed**, so the
   set is 45 exact-match items (GSM8K + AIME `int`, BBH-logical-deduction `mc`). No
   `needs_verify` spillover, no `--with-verify`, no subagents.

4. **General instruction-following (forgetting probe) — IFEval, deterministic.** Matches
   RL's Razor, which scores prior-task IF with **IFEval** (Zhou et al. 2023): rule-verifiable
   (e.g. "answer in all caps", "include keyword K exactly N times"), scored programmatically —
   **no LLM judge / subagents**, loggable to W&B per checkpoint.
   ```bash
   python generate_eval.py --prompts general_eval/ifeval_prompts.jsonl \
       --adapter ../out/<run>/checkpoint-16 --out general_eval/results_c16.jsonl
   cd general_eval && python grade_ifeval.py results_c16.jsonl   # -> strict/loose prompt & inst acc
   ```
   Use `sft.prompt_level_loose` as the retention number. (The old MT-Bench LLM-judge flow —
   `general_prompts.jsonl` + `judge_build.py` — is deprecated; it needed subagents.)

5. **Assemble `master_summary.json`** — one row per checkpoint (extra numeric keys are fine):
   `{"point": "c16", "step": 16, "kl_new": <kl_new_SI>, "acc": <retained math acc 0-1>}`.

6. **Plot + log to W&B**:
   ```bash
   python plot_kl_forgetting.py --summary out/<run>/master_summary.json
   # -> fig_kl_forgetting.png (forgetting vs KL) + fig_acc_vs_kl.png
   python wandb_eval.py --summary out/<run>/master_summary.json --project "$WANDB_PROJECT" \
       --run_name <run>-eval --figure out/<run>/figures/fig_kl_forgetting.png
   ```
   `wandb_eval.py` logs each checkpoint's metrics as `eval/*` curves over training step (plus
   `final/*` in the run summary and the plot image). It's schema-agnostic — reuse the same call
   for eval numbers other teams hand over later.

## Deterministic vs judge-based (what needs subagents)

- **Deterministic, log to W&B per checkpoint:** KL axis, math retention (exact match), and IFEval
  (`general_eval/grade_ifeval.py`). These are all the forgetting axes and cost only extra
  *generation* per checkpoint (KL is cached — see the runbook's cost note).
- **Judge-based (subagents), keep final-only or on a subset:** only the Socratic **pedagogy**
  quality (new-task y-axis; inherently about *how* it tutors, no deterministic metric).

The `*_build.py` scripts emit blind batches scored by an external LLM-as-judge (the POC used a
frontier model via subagents). Nothing here calls an API directly, so no key is required for the
deterministic parts (generation, exact-match grading, KL, plotting, W&B logging).
