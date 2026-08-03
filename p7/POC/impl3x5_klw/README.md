# Impl 3 × Impl 5 — James's loss weighting on self-distilled targets

## What this asks

Impl 3 and Impl 5 are both "stay closer to the base model" interventions, and they act on
different things:

| | what it changes | mechanism |
|---|---|---|
| **Impl 3** (James) | *how much each token counts* | per-token loss multiplier `m_t`, from how far that token pulls the policy off π₀ |
| **Impl 5** | *what the tokens are* | the tutor turns are rewritten by π₀ itself, so the targets start closer |

Neither has been run against the other. Three outcomes are all plausible, and they are
distinguishable:

- **Complementary** — the reweighting still finds distance to remove after distillation, and the
  combined arm lands below both on the KL–forgetting plane.
- **Redundant** — distillation already removed the distance low-`T` was suppressing, so the arms
  collapse onto D4 and the reweighting buys nothing.
- **Interfering** — on distilled targets the signal is small and mostly noise, so reweighting
  amplifies noise and costs new-task fit for nothing.

There is a specific reason to expect the *variants to split*. Variant **b** (forward-KL) applies
the same kind of pressure distillation does, so it is the one most likely to be redundant.
Variant **a** (base-surprise) works differently — James's §6.3 shows it avoids forgetting by
**gating on the system instruction** rather than by staying near base, and `a-T8` beats Impl 5's
D4 on forgetting (`math_hint` 0.652 vs 0.572) at the same KL while also outranking it on the
blind judge. Gating and distillation are orthogonal, so `aT8` is the arm where composition is
most likely to be real.

## The arms

Three conditions, picked as the top three on the +SI blind pedagogy judge (2026-08-03), where
James's arms took first, second and third and Impl 5's D4 came fourth — plus a control.

| arm | James's name | variant | T | why |
|---|---|---|---|---|
| `bT1` | `b-T1` | forward-KL | 1 | judge 0.913, 1st of 10 |
| `bT2` | `b-T2` | forward-KL | 2 | judge 0.910, 2nd |
| `aT8` | `a-T8` | base-surprise | 8 | judge 0.896, 3rd; the gated mechanism, and the arm to beat |
| `bT451` | `b-T451` | forward-KL | 451 | **control: must reproduce D4** |

`bT451` is not a fourth condition. It is the implementation check James's handoff recommends —
"T → ∞ recovers vanilla SFT exactly … a cheap and strong implementation check" — and here the
thing it must reproduce is **D4**. If it misses D4, no other arm's number means anything. It
runs on the fourth GPU, which is why it costs no wall clock.

## What every arm holds fixed

All four train on **D4's training file, byte for byte** — read straight out of
`impl5_ssd/runs/D4/`, never copied per arm, so two arms cannot train on files that differ. That
is the distilled pedagogy pool at realised δ = 0.368 label tokens, in A1's replay slot, in A1's
block positions.

Also fixed at D4's: `per_device_batch 8 × grad_accum 4`, seed 13, lr 2e-4 cosine, 1 epoch,
923 steps, LoRA r=16/α=32/dropout 0.05, the 24/8 block layout under a `SequentialSampler`, the
22-point checkpoint grid, and `gradient_checkpointing=True`.

**The batch shape is not a tuning knob**, and this is the one place where the obvious performance
win was declined on purpose. See `BUILD.md` §"GPU utilisation".

The only difference from D4 is the multiplier on pedagogy tokens. Replay tokens always get 1.0.

## Baselines, and what these numbers may not be compared to

The baseline for every arm here is **D4** (`impl5-D4`). Not `impl4-A1`, and **not James's
published gold-corpus numbers**: those were trained by his pipeline on gold targets, and the
temperatures are not transferable anyway because §4.1's softmax is normalised globally over the
pedagogy stream, and this stream is ~37% base-model text where both signals are systematically
smaller. `bT1` here is not `b-T1` there.

When comparing an arm to its gold-corpus twin, quote **`multiplier.ess`** from the precompute
(the fraction of pedagogy tokens effectively carrying gradient) rather than T. ESS is
corpus-comparable; T is not.

## Running it

```bash
# everything, saturating every GPU on the box
python run_klw.py

# the two gates, in order, before any GPU time
python acceptance_checks_klw.py --stage fast     # decides --loss_denom
python smoke_klw.py                              # end-to-end on a tiny model, CPU, ~1 min

# stages individually
python run_klw.py --stages fetch,mix,precompute
python run_klw.py --stages train --loss_denom global
python run_klw.py --stages bundle,fetch,math --adapters_from s3://…/checkpoints
```

`run_klw.py` needs two artefacts it cannot rebuild on a CPU, both from D4's run: the distilled
pool (`impl5_pool.tar.gz`, ~90 accelerator-minutes to regenerate) and `D4/ckpt-923`, which is
variant b's reference π_SFT. `--stages fetch` pulls both from S3.

## Files

| path | what |
|---|---|
| `klw/weighting.py` | IMPL3_HANDOFF §4.1's objective, reimplemented from the spec |
| `klw/trainer_klw.py` | the weighted `compute_loss` and the collator |
| `klw/config_klw.py` | the four arms; everything else imported from Impl 5 |
| `precompute_signal.py` | both variants in one sharded multi-GPU pass |
| `train_sft_klw.py` | one arm |
| `acceptance_checks_klw.py` | W1–W7 — the checks that catch silent wrongness |
| `smoke_klw.py` | end-to-end plumbing test on a tiny model |
| `run_klw.py` | the driver; all GPUs |
| `BUILD.md` | what was built, what it cost, and every deviation |
