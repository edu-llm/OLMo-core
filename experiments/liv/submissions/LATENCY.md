# The latency measurement, and what it is allowed to claim

This is Next Steps item ⓵ from `HANDOFF.md`: the one number the P1 study is missing.

The quality question is **answered and null** — `F-r128` and `G-grouped` match dense `L0` on
held-out cross-entropy at 1B tokens (−0.00383 nats, t = −1.437 against a 2.201 threshold) while
removing **15,728,640 parameters**. A null on quality only means something if the parameters
were buying something, so the claim is an efficiency claim, and it rests on a latency figure
that has never been measured end-to-end.

## What is already known, and why none of it is quotable as the headline

| source | card | what it timed | verdict |
|---|---|---|---|
| `probes/p1_launch_bench.py` | L40S | 7 linear layers, 40 MiB working set | **RETRACTED** — cache-resident, reported the wrong sign |
| `probes/p1_cache_check.py`, `probes/p1_scaled.py` | L40S | same layers at 40/160/320/960 MiB | valid, but a subgraph and not a model |
| this run | A100 | **the whole 390M model** | the headline |

The retracted number said low-rank was **8.2% slower**. It replicated to within 0.3% on a second
job, because it faithfully re-measured an unrepresentative configuration.

Two residency-scaled re-tests both flipped the sign. They agree in direction and differ in
magnitude, so the honest statement is a range across both rather than either one's best rung:

| working set | `F-r128` fused | `G-grouped` |
|---|---|---|
| 40 MiB (inside L2 — the artifact) | −3.7% to −1.8% | +20.2% to +20.4% |
| 160 MiB | +39.1% | +52.5% |
| 320 MiB | +40.4% | +54.4% |
| 960 MiB | +29.9% to +31.3% | +46.1% to +47.6% |

So past L2, `F-r128` is **+29.9% to +40.4%** and `G-grouped` is **+46.1% to +54.4%**. The spread
between the two jobs at the shared 960 MiB rung is ~1.4 points, which is the reproducibility of
the subgraph measurement and is worth knowing before reading a 1.8% end-to-end prediction off it.

**A writeup that still says low-rank is slower is citing a dead number.** `G-grouped` remains the
recommendation, but because grouped wins, not because low-rank loses.

Share-weighting the subgraph win by the ~5.4–5.9% of weights the gates hold predicts about
**+1.8%** end-to-end. That figure is arithmetic, not a measurement, and it is what this run
replaces. It is also the reason the harness reports p10/p90 alongside every median: a 1.8% effect
is only readable if the measurement's own spread is smaller than that, and if it is not, the
correct output is "cannot resolve" rather than a number.

## The two submissions

Both are `olmo-core-check` (1 h ceiling, 1 attempt, `checkpoint: null`), `team=scratch`,
`dataset_release=none` — the harness builds models from config and reads no corpus.

Both compile **routine → `run-approval-lead`** as of policy **v6**. The
`EXCEPTION_RATE_CEILING_USD_PER_HOUR = 20` that used to force every A100 run to an admin **has
been deleted**; `thresholds.automatic_below_cost_usd` is now $500 and the A100's worst case here
is $10.98, so this needs a lead and not an admin.

### 1. A100, the headline — $10.98 worst case

```bash
gh workflow run submit-run.yml --repo edu-llm/platform --ref main \
  -f repository=OLMo-core \
  -f commit_sha=<HEAD of edullm/liv-a100-latency> \
  -f workload_profile=olmo-core-check \
  -f compute_profile=gpu-8xa100 \
  -f team=scratch -f experiment=liv-p1-a100-gate-latency \
  -f dataset_release=none -f wandb_project=eduLLM \
  -f maximum_runtime_hours=0.5 -f maximum_attempts=1 \
  -f command='bash -lc '"'"'EDULLM_LAUNCH_CHECK=waived python .edullm/bench_gate_latency.py "$EDULLM_RUN_ID"'"'"''
```

**Seven of the eight cards sit idle, and that is the honest cost of the number.** Inference
latency is a per-card property, the harness uses one device, and the platform prices no
single-A100 shape — the `compute_profile` dropdown holds `gpu-8xa100` and nothing smaller with
an A100 in it. Paying for eight to measure one is unavoidable if the figure is to be measured on
the card the study ran on.

`EDULLM_LAUNCH_CHECK=waived` is required and is not a formality: the platform refuses one process
on a multi-GPU shape, and a benchmark is exactly the case the waiver exists for. The
`EDULLM_CHECKPOINT_CHECK` token is **not** needed — `olmo-core-check` declares `checkpoint: null`,
so nothing is promised to waive. Verified by compiling both ways.

Expect a **61-minute median queue wait**, worst observed 12.6 h (`config/capacity.yaml`, fourteen
nodes 2026-08-03→05). While it waits it sits in RUNNABLE writing nothing, which looks identical
to a shape that will never place.

### 2. L40S cross-check — $0.93 worst case, 11.8x cheaper

```bash
gh workflow run submit-run.yml --repo edu-llm/platform --ref main \
  -f repository=OLMo-core \
  -f commit_sha=<same> \
  -f workload_profile=olmo-core-check \
  -f compute_profile=gpu-1xl40s \
  -f team=scratch -f experiment=liv-p1-l40s-gate-latency \
  -f dataset_release=none -f wandb_project=eduLLM \
  -f maximum_runtime_hours=0.5 -f maximum_attempts=1 \
  -f command='bash -lc '"'"'python .edullm/bench_gate_latency.py "$EDULLM_RUN_ID"'"'"''
```

No waiver: one process on one device. Worth running **because it is nearly free and it is a
second host**, which is the cheapest available portability test — a harness that only ever runs
in one place has never been shown to be reproducible. It also re-measures the exact card the
retraction happened on, at full-model scale this time.

## The four receipts every row carries

A latency delta on its own is not checkable, and this project has already believed one for a day.

| field | what it is for |
|---|---|
| `working_set_mib` | weight bytes the timed region reads, from the built module |
| `achieved_gbs` | `working_set / elapsed` |
| `pct_of_hbm_peak` | against 1555 GB/s (A100-SXM4-**40**GB) or 864 (L40S) |
| `conv_path` | `fla` or `nn.Conv1d`, **asserted identical across arms** |

Above 100% of peak is impossible for an HBM-bound region and therefore proves cache residency.
`--fail-on-cache-resident` is **on by default** and exits non-zero, because this is the check
whose absence caused the retraction.

**Note what the bandwidth ratio would and would not have caught.** The retracted probe read
744.7 GB/s against an 864 GB/s peak — **86%, which is below 100 and would have passed.** What
catches it is the working set (40 MiB) against L2 (96 MiB), which is why both are printed on
every row. The full model is 744 MiB in bf16, more than 7x the L40S L2 and 18x the A100's 40 MiB,
so it is genuinely HBM-bound.

`conv_path` exists because `ShortConvConfig.use_fla` defaults to `True` and dispatches on
`use_fla and has_fla() and x.is_cuda`. `fla` is **absent from the research image** (verified
2026-08-05), so the default makes kernel selection a property of the environment rather than of
the declared config. Left alone across arms, the contrast can compare a fused kernel against
`nn.Conv1d` and attribute the difference to gate structure — and **the bias points toward the
hypothesis**, which is the direction that gets believed. The harness pins it to `false` on every
arm and asserts the realised path afterwards.

## The limit that must travel with the decode number

**Decode here is a `seq_len=1` forward, not an autoregressive step.** `ShortConv` implements no
conv-state cache and this config's attention runs without a KV cache, so a served decode would
reuse both and this does not.

That omitted traffic is **identical across the three arms** — they share attention geometry
exactly — so it enters both numerator and denominator and *dilutes* the ratio. Therefore the
decode delta reported here is an **UPPER BOUND** on the served-decode speedup, and the harness
prints that sentence in its own output rather than leaving it in a commit message.

The prefill rung carries no such caveat, which is why both regimes are reported. Prefill is also
the conservative one: it is compute-bound, so a smaller gate buys the least there.

## What this run cannot answer

- **Not a served-throughput number.** No batching scheduler, no cache, no continuous batching.
- **Not a training-speed number.** The 1B-token grid already measured that at 286,400 tok/s/node.
- **Not eager-vs-graph.** Both are eager, matching the study's `--no-compile-model`. Real serving
  would capture graphs, which cuts launch overhead — and low-rank *adds* one launch per gate, so
  graphs would help it more than they help dense.
