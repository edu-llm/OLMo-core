# Pre-registration: Mamba and xLSTM comparison

Status: prepared before dispatch; no run has been submitted.

## Question

At matched parameter count, token budget, data order, attention schedule, and
training recipe, which recurrent architecture offers the best held-out
cross-entropy, steady-state training throughput, and peak-memory trade-off?

The control is `mamba-b3`. The treatments are `xlstm`,
`mamba3-siso-pd`, and `native-pd`.

## Frozen design

The machine-readable authority is `seeds.json`. There are four arms by five
replicates, ordered arm-major. Every replicate shares its data seed across all
four arms. Init seeds are not paired across arms because the parameter names,
shapes, and draw order differ.

Every cell consumes exactly 1,950,875,648 tokens: 3,721 steps at global batch
524,288. The exact models contain 390,142,976–390,169,664 parameters, giving
TPP 5.00007–5.00041. Sequence length is 4096 and every cell uses eight A100
GPUs, BF16 parameters, FP32 gradient reduction, compilation, the same
optimizer and schedule, and the same published dataset release.

Every arm has four identical attention layers at indices 3, 7, 11, and 15.
The other twelve layers are the named treatment. The xLSTM treatment is
specifically 10 mLSTM plus 2 sLSTM layers; it is not the earlier 8:4 control.

## Endpoints

Held-out corpus CE (`val_ce`) is the quality endpoint. Its token denominator
and shard count must match exactly across all cells before any arm difference
is computed.

Post-warmup total and per-device tokens/second are the speed endpoints.
Whole-run tokens/second is retained only for costing because startup,
compilation, dataset opening, and FSDP wrapping contaminate it.

Peak allocated and reserved memory are co-primary operational endpoints, but
only when `peak_memory_source` is `per_step_running_max`.

MFU is descriptive. It must not rank the arms unless every mixer FLOP counter
has first been audited to the same forward-plus-backward convention.
Wall-clock throughput does not depend on that convention and remains valid.

Inference is outside this training wave. The inherited KDA/GDN operator decode
probe is disabled because applying it to Mamba, PD-SSM, or xLSTM would measure
the wrong operator. No absence of a decode result may be interpreted as a
serving regression.

## Analysis

Report every arm's five cell values, mean, standard deviation, and its
arm-minus-control effect with an uncertainty interval. Use one shared-control
multiple-comparison procedure for the three CE contrasts. Report throughput
and memory ratios beside every CE contrast.

Do not infer equivalence from a non-significant CE difference. If CE is
unresolved, say so and report the interval as the bound the run actually
supports. The fallback recommendation is the fastest and leanest arm not shown
to be worse on CE; it must be labelled as that fallback, not as a quality win.

There is no post-hoc cell deletion rule. A failed cell may be rerun only under
the platform's recorded retry semantics; `attempts=1` is frozen for the first
submission, so any later replacement is a documented deviation.

## Preconditions

- One commit and one image must contain all four implementations.
- The 20 literal arm/data/init tuples in the run spec must agree with
  `seeds.json`.
- Exact parameter counts and the 10m/2s/4attention xLSTM role inventory must
  pass their tests.
- The target image must expose Mamba-3, native PD, mLSTM, and FlashRNN kernels
  on sm80 and carry the required license notice.
- `edullm check --json` must stand without `--force`.
- Any change to steps, batch, sequence length, corpus, world size, or seed
  schedule applies to all 20 cells and is recorded here before dispatch.

## Deviations

None. The experiment has not dispatched.
