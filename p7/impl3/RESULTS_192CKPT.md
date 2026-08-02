# Impl-3 results: the 192-checkpoint KL–forgetting sweep

Everything below comes from `out/ckpt_sweep_bare_hint250.jsonl` (194 rows) and the figures in
`out/figures/`. Both are committed, so the analysis can be re-derived without cluster access.

## What was run

16 training runs × 12 log-spaced checkpoints (steps 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 923)
= 192 scored checkpoints, plus two single-point anchors (the base model and the POC's
`checkpoint-923`).

| family | temperatures | what the reweighting does |
|---|---|---|
| `impl3-a` (base-surprise) | 0.5, 1, 2, 4, 8, 16, 32 | upweights tokens the base model finds surprising |
| `impl3-b` (forward-KL) | 0.5, 1, 2, 4, 8, 16, 32, 451 | upweights by forward KL against the vanilla SFT model |
| `SFT` (`impl2-rerun`) | — | vanilla SI-conditioned SFT, the baseline everything is judged against |

All share the POC recipe: `allenai/OLMo-2-0425-1B-Instruct`, LoRA r16, LR 2e-4, 1 epoch,
`meric533/socrateach-sft`. `T=451` is a deliberate T→∞ limit check.

Per checkpoint we measure forward KL in two conditions (with and without the pedagogy system
instruction), GSM8K accuracy over 250 items in two prompt conditions (with and without the
"put your final answer in `\boxed{}`" hint), the answer-commit rate, and held-out pedagogy NLL.

## The main result: the KL condition decides whether the relationship appears

The prior-task probes carry **no** system instruction, so KL has to be measured in that same
condition to predict forgetting. Measured with the SI present, the relationship largely
disappears:

| predictor of GSM8K accuracy | hinted | bare |
|---|---|---|
| KL measured **with** SI (linear R²) | 0.367 | 0.261 |
| KL measured **without** SI (linear R²) | **0.739** | **0.807** |
| pooled monotone fit, no-SI KL | **0.945** | 0.920 |
| pooled monotone fit, with-SI KL | 0.507 | 0.371 |

This is **not** a uniform improvement, and the exception is the interesting part. Within variant b
alone the with-SI KL predicts slightly *better* (0.836 vs 0.731). The no-SI KL wins by collapsing
both families onto a single curve — and that method-invariance, not correlation strength, is what
RL's Razor actually claims.

Variant a is the reason the conditions disagree. Its KL is 7–20× larger with the SI than without
(a-T1: 0.709 vs 0.036), whereas variant b differs by only 2.4–5×. Variant a learned a policy
*gated on the system instruction* — Socratic when instructed, near-base otherwise — so it retains
math despite a large with-SI KL. Plotting with-SI KL scatters those runs across the high-KL region
while their forgetting stays near zero.

Because of this, `plot_figure3.py` defaults to `--kl_key kl_ped_noSI`, and `make_figures.sh` emits
the with-SI figures alongside so the gap stays visible.

## Final checkpoints

| run | KL (SI) | KL (no SI) | ped NLL | hinted | bare | commit% |
|---|---|---|---|---|---|---|
| base | 0.000 | 0.000 | 1.416 | 0.664 | 0.656 | 98.8 |
| b-T0.5 | 0.176 | 0.073 | 0.959 | 0.620 | 0.612 | 100.0 |
| b-T1 | 0.239 | 0.096 | 0.924 | 0.612 | 0.600 | 100.0 |
| b-T2 | 0.306 | 0.118 | 0.901 | 0.548 | 0.588 | 98.4 |
| b-T4 | 0.337 | 0.136 | 0.881 | 0.360 | 0.536 | 98.0 |
| b-T8 | 0.528 | 0.146 | 0.864 | 0.284 | 0.500 | 95.2 |
| a-T8 | 0.662 | 0.067 | 0.966 | 0.652 | 0.616 | 99.2 |
| a-T16 | 0.722 | 0.098 | 0.896 | 0.516 | 0.612 | 99.6 |
| b-T32 | 0.712 | 0.150 | 0.862 | 0.212 | 0.504 | 92.4 |
| b-T451 | 0.759 | 0.150 | 0.862 | 0.212 | 0.480 | 89.6 |
| SFT | 0.761 | 0.150 | 0.862 | 0.212 | 0.456 | 90.4 |

Reading it:

- **`b-T0.5` is the headline.** It holds 62.0% hinted GSM8K against SFT's 21.2%, close to the base
  model's 66.4%, for 0.10 nats of new-task NLL (0.959 vs 0.862). Strict Pareto dominance comes back
  empty only because SFT sits exactly on the NLL floor of 0.862, so nothing can tie it *and* beat
  it — but trading a tenth of a nat for 41 points of retention is the result we were after.
- **`b-T451` reproduces SFT to within 0.002 on every column.** That is the reweighting correctly
  collapsing to vanilla at high temperature, and it is the main evidence the implementation is
  sound rather than accidentally doing nothing.
- **Pedagogy NLL is a proxy.** The LLM-judge batches for these runs are in `eval/llm_judge/extra/`
  (b-T0.5, b-T1) and `eval/llm_judge/extra_a/` (a-T16, a-T32), generated but not yet judged.

## The boxing hint is a confound worth knowing about

SFT scores 0.212 hinted but 0.456 bare. The hint ("put your final answer in `\boxed{}`") collides
with the tutor persona's "never state the final answer yourself", so a tutor-tuned model deflects
rather than answers — its commit rate falls to 90.4% while the base model stays at 98.8%. The
hinted number therefore mixes Socratic refusal with genuine skill loss; the bare number isolates
skill. Hinted is the default because the forgetting effect is far more visible there, but both are
plotted (`_bare` suffix) and neither should be quoted alone.

This also resolves an earlier apparent contradiction with the POC, where math dropped sharply after
vanilla SFT while our first runs showed it flat: the POC appended the hint and we initially did not.

## Reproducing

```bash
pip install -r requirements.txt          # versions are pinned

# 1. train (ORCD, 1x H200) — chains the per-checkpoint eval automatically
sbatch clusters/orcd/impl3_extra.sbatch
#    partial resubmit, with its own pedagogy dir so earlier batches survive:
RUNS="a:16 a:32" PED_DIR=extra_a CHAIN_EVAL=0 sbatch clusters/orcd/impl3_extra.sbatch

# 2. score every checkpoint (resumable; refuses to mix measurement protocols)
sbatch clusters/orcd/ckpt_sweep_eval.sbatch

# 3. figures — all four KL-condition x math-prompt combinations
bash eval/make_figures.sh
```

Training data is **not** vendored. It streams from
[`meric533/socrateach-sft`](https://huggingface.co/datasets/meric533/socrateach-sft); run
`python snapshot_hf_dataset.py` to materialise it locally for an offline GPU node.

## Infrastructure fixes made during this sweep

- **Resume was silently broken.** Transformers refuses to load `optimizer.pt` on torch < 2.6
  (CVE-2025-32434), and `rng_state.pth` additionally fails `weights_only` unpickling on numpy's
  MT19937 key. Every preempted run had been restarting from scratch. Both guards are now cleared
  for our own checkpoints in `common/sft_train.py::_allow_resume_from_our_own_checkpoints`, with
  torch left pinned so training numerics do not shift mid-sweep.
- **KL was being measured on the wrong context.** Pedagogy dialogues were passed whole, so several
  gold Socratic turns primed the model and the system instruction stopped mattering. They are now
  truncated before the first tutor turn (`common/kl.py::pedagogy_contexts`), matching the POC.
- **Base continuations are cached across checkpoints** (`common/kl.py::base_continuations`). KL
  samples from the base policy, so the continuation never depends on the checkpoint being scored;
  hoisting it out of the loop removed ~99% of sweep runtime.
- **The math set was underpowered.** 45 items could not resolve 5–15 point gaps; it is now 250
  GSM8K items. BBH logical-deduction was dropped after scoring at floor (6.7% against 14.3%
  chance) for a 1B model, which made it uninformative for measuring forgetting.
