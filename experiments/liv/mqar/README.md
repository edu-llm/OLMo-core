# MQAR calibration for the LIV study

Tuning MQAR difficulty so the endpoint can actually discriminate between arms. Standalone modules
— the older `probes/mqar_patch.py` works by rewriting `tasks.py` and `train_probe.py` in place;
this reuses its decoupled-axes design without the in-place edits.

| file | what it is |
|---|---|
| `mqar_data.py` | Generator (Zoology-faithful), calibrated constants, `degenerate_floor()` |
| `mqar_model.py` | Small LFM2-shaped hybrid. Uses the **same `ShortConv`** as the real arms |
| `mqar_calibrate.py` | The difficulty sweep. Baseline only, refuses to run under-budget |
| `mqar_positive_control.py` | "Can any config learn this at all?" — run this before any sweep |
| `mqar_data_test.py` | 43 tests pinning generator correctness |

Results: `mqar_calibration.json` (FarmShare 1670987), `mqar_positive_control.json` (1670928).

## Calibrated settings

```
vocab 256 · lr 3e-3 · attention at (1, 3) · 8000 steps × batch 64 = 512k examples
```

**Vocab 256, not Zoology's 8192.** At 8192 the best of 6 configs reached 0.214 and four sat exactly
at loss 8.32 = `ln(4096)`, the size of the value half. At 256, two configs reached 0.995 and 1.000.
An 8192-way softmax over 4 answers spends capacity on the output distribution, not the binding, at
this budget. Raise it only with a proportionally larger budget.

## Results

Capacity grid (length and pairs grow together):

| config | 1/D floor | success | per-seed |
|---|---:|---:|---|
| `N64_D4` | 0.250 | 80% | 0.27 0.99 1.00 1.00 1.00 |
| `N128_D8` | 0.125 | 100% | all 1.00 |
| `N256_D16` | 0.062 | 100% | all 1.00 |
| **`N512_D64`** | 0.016 | 20% | 0.05 0.09 0.20 0.56 0.98 |

Distance sweep — **D fixed at 8, so capacity is constant and distance is the only variable**:

| seq_len | success | median | × floor |
|---:|---:|---:|---:|
| 64 | 100% | 1.00 | 8.0× |
| 128 | 100% | 1.00 | 8.0× |
| 256 | 100% | 1.00 | 8.0× |
| **512** | **40%** | 0.16 | 1.3× |
| 1024 | 0% | 0.15 | 1.2× |

A clean monotone cliff between 256 and 1024, attributable purely to retention distance.

## Three things to know before reading any MQAR number here

**1. The chance baseline is 1/D, not 1/vocab, and it moves with the config.**

A model that learns *"the answer is one of the D values in this sequence"* without binding
anything scores exactly `1/D`. Six of twelve positive-control trials sat at 0.208–0.274 with losses
of 1.40–1.76 against `ln(4) = 1.386` — a fully-learned wrong algorithm, not partial recall. The
loss plateaus form a legible ladder:

| plateau | meaning | predicted | observed |
|---|---|---:|---:|
| `ln(vocab/2)` | "it's a value token" | 8.32 | 8.32 |
| `ln(D)` | "it's one of these D values" | 1.39 | 1.40–1.76 |
| 0 | actually bound | 0 | 0.000 |

At D=64 the floor is 0.016, so 0.10 there is real work; at D=4 it is *below* the degenerate
strategy. Always report against `degenerate_floor()`.

**2. Bimodality holds at low load and BREAKS at high load.**

The script's summary line says "strongly bimodal, success rate is the right endpoint." That is
true for 41/45 runs but **wrong for `N512_D64`**, whose seeds spread continuously: 0.05, 0.09,
0.20, 0.56, 0.98 (3.3× to 62.9× floor). With 64 pairs a model can bind some and not others, so
accuracy is graded, not binary. Collapsing that rung to "20% success" discards most of the signal
— an arm at 0.55 vs a baseline at 0.20 is a large real difference that a binary threshold reports
as 0-vs-0. **For high-load rungs report both success rate and median accuracy vs floor.**

**3. The script's recommended operating point is not the one to use.**

It picks `N512_D8` (40% success) by proximity to 50%. Prefer **`N512_D64`**: it is off-ceiling on
*both* axes at once, its graded scores carry more information per seed, and its 0.016 floor leaves
far more headroom to detect a difference. `N512_D8` is a good *secondary* — same length, 8× less
capacity load, so the pair separates capacity from distance.

## Two failures worth not repeating

**Under-budget run (job 1670963).** I fixed vocab/LR/attention in the script but resubmitted a
stale sbatch with `--steps 3000 --batch-size 32` — 96k examples against the control's 512k, a 5.3×
shortfall. `N64_D4`, which the control solved at 1.000, scored 0.24/0.25/0.25/0.26/0.93 with four
runs parked on the 1/D floor. **Under-training is indistinguishable from a too-hard task in the
output.** `mqar_calibrate.py` now owns the budget constants and refuses to run below them.

**Sweeping before a positive control (job 1670922).** The first sweep returned 0.000 everywhere
because vocab was 8192. A difficulty sweep whose easiest rung scores zero cannot separate "hard
task" from "broken setup." Always establish that *something* solves the easiest point first.

## Caveat on transferring these numbers

This is the **calibration model**: 4 layers, attention at (1,3), d=128. The real `L0` is 16 layers
with 6 attention layers at `[2,5,8,10,12,14]` and d=1024, so its cliff will sit somewhere else.
Note also that the cliff here is **not** a receptive-field limit — the attention layers are global,
so reach is not the binding constraint; what degrades is the difficulty of *finding* the recall
circuit as distractor count grows. **What transfers is the method and the 1/D floor, not the
operating point.** Re-check on real `L0` before using these settings in the study.
