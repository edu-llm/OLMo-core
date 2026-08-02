# Evaluation

Everything the Impl-3 sweep measures. The task is the Socratic math tutor
(`meric533/socrateach-sft`), so the axes are: how far the model moved (KL), what it lost
(math retention), and what it gained (pedagogy).

| Track | What it measures | Location |
| --- | --- | --- |
| **KL axis** | New-task drift `KL(pi_0 ‖ pi)` from base, in two SI conditions | `../common/kl.py`, `../run_kl_curve.py` |
| **Math retention** | OLD-task ability (forgetting probe), deterministic exact-match | `math_eval/` |
| **Pedagogy NLL** | NEW-task quality, cheap per-checkpoint proxy | `sweep_ckpt_eval.py` |
| **Socratic pedagogy judge** | NEW-task quality, the real metric | `llm_judge/` |

Two probes that used to live here are gone. **IFEval** was dropped because at 34 prompts it
never separated any two configurations, so it cost generation time and produced no signal.
**MATH-500** was dropped because its `expr` items need symbolic or LLM verification, which
would have put a subagent in the middle of an otherwise deterministic loop. Math is now 250
GSM8K items, exact-match only.

## The one command that matters

`sweep_ckpt_eval.py` scores every checkpoint of every run in one pass — KL in both SI
conditions, GSM8K in both prompt conditions, and pedagogy NLL — and appends to a JSONL that
`plot_figure3.py` turns into figures.

```bash
sbatch ../clusters/orcd/ckpt_sweep_eval.sbatch    # on ORCD, resumable
bash make_figures.sh                              # locally, from the committed jsonl
```

It is safe to re-run. Already-scored checkpoints are skipped, and each row is stamped with a
**measurement protocol** (KL context, math conditions, and a hash of the item ids). If the
protocol changes, the script aborts rather than appending rows that silently mean something
different from the ones already in the file.

## Per-run generation (the older, per-checkpoint-at-a-time path)

Still used by `impl3_h200.sbatch` / `t451_control.sbatch` for the final checkpoint of each run,
and by the hint A/B diagnostic.

```bash
# math, both prompt conditions
python generate_eval.py --prompts math_eval/math_logic_prompts.jsonl \
    --adapter ../out/<run>/checkpoint-923 --out math_eval/results_<run>.jsonl
python generate_eval.py --prompts math_eval/math_logic_prompts.jsonl --boxed_hint ...
python math_eval/score_results.py math_eval/results_*.jsonl   # accuracy, commit rate, acc|commit
```

The **boxing hint** matters more than it looks: "put your final answer in `\boxed{}`" collides
with the tutor persona's "never state the final answer yourself", so a tutor-tuned model
deflects instead of answering. Always read the hinted and bare numbers together — see
`../RESULTS_192CKPT.md`.

## Pedagogy judging (the only step needing subagents)

```bash
python gen_pedagogy.py --base_model ... --candidates base= impl2=... <run>=... \
    --n_dialogues 40 --out llm_judge/<batchdir>/test_results.jsonl
cd llm_judge && python build_batches.py <batchdir>/test_results.jsonl <batchdir> 2
# judge judge_batch_N.json -> judge_out_N.json with a frontier model, then:
cd <batchdir> && python ../aggregate.py
```

`base` and `impl2` are regenerated into every batch as in-batch anchors. The judge scores each
batch independently, so without them there is nothing to calibrate new runs against — and the
anchors are not free of noise either: two vanilla SFT runs scored 0.64 and 0.53 on the same
rubric, which bounds how small a difference is worth believing.

Nothing here calls an API directly, so no key is needed for the deterministic parts.
