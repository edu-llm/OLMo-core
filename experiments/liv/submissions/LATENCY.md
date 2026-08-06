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
replaces.

## The constraint that dictates the whole design: the ceiling is 4.03%

`15,728,640 × 2 B = 31.5 MB` of the model's `780.3 MB`. So a **perfectly** weight-bandwidth-bound
decode step cannot improve by more than **4.03%**, prefill's ceiling is ~3.6%, and the predicted
value is ~1.8%.

**Ordinary GPU benchmarking noise is 5–20%.** Clock drift under sustained bf16 load is 4–10% on an
A100-SXM4; kernel-launch overhead at `seq_len=1` is worth 8–14%. A conventional benchmark here
does not return a small number with error bars — it returns a confound with a plausible sign.

An earlier version of this harness was exactly that, and it failed in three ways that all pointed
the same direction — **against the treatments**, which is the direction that would have read as
corroboration of the retracted −8.2%:

| what was wrong | worth | why it was invisible |
|---|---|---|
| **arm-at-a-time ordering** | 5–10% | `L0` measured first and coldest ⇒ artificially small baseline. p10/p90 comes back *tight* because drift is locally stable, certifying the run. |
| **eager-only decode** | 8–14% | the treatments *add* 40–50 kernel launches, so eager decode ranks arms by dispatch count |
| **a cache-residency ceiling** | — | **unfireable**: tripping it needs the step under 0.502 ms; the fastest shape is ~0.7 ms. Every row read `False` and printed "0 rows exceeded peak" |

The last one is the worst of the three, because it printed as a *passed check*.

## What the harness does instead

- **Interleaved randomized rounds.** All arms stay resident (4 × 780 MB against 40 GB), 75 s soak
  to reach steady clocks, then each round times every arm back-to-back in a **reshuffled** order.
  Ratios are formed **within** a round, so drift is common-mode and divides out.
- **Paired ratios with a percentile bootstrap CI**, not a difference of medians.
- **An A/A control arm.** `L0` built twice under two names. Its true effect is *exactly zero*, so
  the interval around it **is** the rig's resolution. If that interval is wider than 1.8%, the run
  **exits non-zero and refuses to report deltas** — two identical models differing by more than the
  effect being hunted means any delta is noise with a sign. This is the one guard that can fail for
  the right reason, and it is in the default arm list so a plain invocation measures its own floor.
- **Eager *and* CUDA-graphed decode**, with per-arm launch and copy-kernel counts, so a negative
  eager row can be attributed to dispatch rather than to gate structure. Graph replay is safe here
  precisely because a 744 MiB model against a 40 MiB L2 cannot be cache-resident.
- **A utilization FLOOR replacing the ceiling** — FLOPs for prefill, bandwidth for graphed decode.
  It fails in the direction the failure actually lies: a region that is neither compute- nor
  bandwidth-bound cannot show a saving. Plausible readings straddle it (graphed ≈70%, eager ≈20%
  against a 30% floor), which is what makes it a discriminating check rather than a decoration.
- **`logits_to_keep=1` for decode**, as real serving does. The full head is 3.37 TFLOP of
  arm-invariant work — ~23% of prefill — sitting in both numerator and denominator, diluting the
  delta toward zero. Prefill reports the delta **both ways** so the dilution is visible.

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

## The receipts every row carries

A latency delta on its own is not checkable, and this project has already believed one for a day.

| field | what it is for |
|---|---|
| `ratio_median` + `ci_low/high_pct` | the paired ratio and its bootstrap interval, not a bare delta |
| `pct_of_flops_peak` / `pct_of_hbm_peak` | against the utilization floor, per regime |
| `utilization_ok` | `False` ⇒ `LOW_UTIL` flag and a non-zero exit |
| `launch_count`, `copy_kernel_count` | per arm, so an eager gap is attributable to dispatch |
| `sm_clock_mhz_mean`, `temperature_c_max` | per arm; >2% clock spread is a refusal |
| `conv_path`, `gate_structure` | asserted identical / distinct across arms as appropriate |

**The bandwidth ratio is decode-only now.** In prefill at batch 4 × 4096 the logits tensor alone is
3.29 GB written — four times the entire weight footprint — and real traffic is 15–20 GB against
the 0.78 GB a weights-only ratio counts. Quoting it there understates by ~20×, so prefill prints
`n/a` and is judged on FLOPs instead.

**And note what the bandwidth ratio never would have caught.** The retracted probe read 744.7 GB/s
against an 864 GB/s peak — **86%, below 100, would have passed.** What catches *that* failure is
the working set (40 MiB) against L2 (96 MiB). At full-model scale neither check is the live one —
744 MiB against a 40 MiB L2 is 18.6× over, so cache residency is impossible — which is exactly why
the ceiling was replaced by a floor.

`conv_path` exists because `ShortConvConfig.use_fla` defaults to `True` and dispatches on
`use_fla and has_fla() and x.is_cuda`. `fla` is **absent from the research image** (verified
2026-08-05), so the default makes kernel selection a property of the environment rather than of
the declared config. Left alone across arms, the contrast can compare a fused kernel against
`nn.Conv1d` and attribute the difference to gate structure — and **the bias points toward the
hypothesis**, which is the direction that gets believed. The harness pins it to `false` on every
arm and asserts the realised path afterwards.

## The limits that must travel with the number

**Decode is a `seq_len=1` forward, not an autoregressive step.** `ShortConv` implements no
conv-state cache and this config's attention runs without a KV cache.

**A correction to what I wrote earlier: this is NOT an "upper bound" on served decode.** I claimed
that, reasoning that the omitted cache traffic is arm-invariant and therefore dilutes the ratio.
That much is true, but **added launch overhead pushes the opposite way and is larger**, so the
decode delta is neither an upper nor a lower bound. It is the delta for a cacheless `seq_len=1`
step, and that is the whole of what it is. A test fails if the tidier claim reappears.

**A negative result may belong to this implementation, not to gate structure.** The materializing
copies in `_GateProj` — a strided slice fed to `F.linear`, a transpose into `bmm` and a reshape out
— are costs of how the projections are *written*, not of low-rank or block-diagonal gating as
ideas. A rewrite avoiding them plausibly recovers most of the penalty. The launch and copy counts
are reported per arm so that distinction can be made rather than glossed over.

**`liv_arms.py` still carries the retracted −8.2% in its own arms table** as justification for
calling the latency claim dead. That line is stale and should be annotated before anyone cites it
as prior corroboration of a fresh negative.

Prefill is the conservative rung: it is compute-bound, so a smaller gate buys the least there, and
it carries none of the decode caveats.

## What this run cannot answer

- **Not a served-throughput number.** No batching scheduler, no cache, no continuous batching.
- **Not a training-speed number.** The 1B-token grid already measured that at 286,400 tok/s/node.
- **Not a statement about tuned kernels.** Both paths use the stock implementations, and the
  copy-avoiding rewrite noted above is untested.
- **Not `torch.compile`.** Graph *capture* is used for decode, which removes launch overhead, but
  no fusion pass runs. A compiled model could fuse the low-rank chain and change the ranking.
