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

Every cell consumes exactly 599,785,472 tokens: 1,144 steps at global batch
524,288. The exact models contain 390,142,976–390,169,664 parameters, giving
TPP 1.53724–1.53735. This matches the measured mixer-bakeoff Run 1 budget;
its original 1,907-step plan would have been TPP about 2.56, and the later
3,721-step Run 2 used TPP about 5. Sequence length is 4096 and every cell uses eight A100
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
- The 12 literal arm/data/init tuples in the run spec must agree with
  `seeds.json`.
- Exact parameter counts and the 10m/2s/4attention xLSTM role inventory must
  pass their tests.
- The target image must expose Mamba-3, native PD, mLSTM, and FlashRNN kernels
  on sm80 and carry the required license notice.
- `edullm check --json` must stand without `--force`.
- Any change to steps, batch, sequence length, corpus, world size, or seed
  schedule applies to all 12 cells and is recorded here before dispatch.

## Deviations

- Before dispatch, the prepared 3,721-step / TPP≈5 budget copied from
  mixer-bakeoff Run 2 was replaced with measured Run 1's 1,144-step budget.
  All cells moved together; the arm-major design, data release, sequence
  length, batch sizes, LR, hardware, and seeds are unchanged.
- Before dispatch, replicates per arm were cut from five to three (20 cells to
  12), matching mixer-bakeoff Run 1's three-seed shape and saving 40% of the
  node-hours. Data seeds 240028/250035 and their init seeds stay reserved and
  unissued. This widens the 95% interval on a per-arm CE mean by about 2x,
  because both the sample size and the t degrees of freedom fall; Run 1 could
  not resolve CE at three seeds with two mixer slots, and this design's
  twelve mixer layers per arm are what is being relied on for a larger effect.
  Throughput and peak memory are budget- and replicate-robust and are the
  endpoints this wave is expected to settle.
- Before dispatch, the decode probe was enabled by dropping `--no-decode-probe`,
  adding decode latency and recurrent-state bytes as secondary endpoints. This
  follows mixer-bakeoff Run 2, which added the measurement because training
  throughput does not predict serving cost. It cannot bias the primary
  throughput endpoint: it runs on rank zero inside `summarise()` after the timed
  loop and outside the steady-state window, drives one operator on one device
  with no collectives, and records failures as a reason rather than raising. Its
  cost is 216 single-token passes per cell. Unlike Run 2's two-slot arms, whose
  six attention layers dominated their decode footprint, all twelve recurrent
  layers here carry the arm's operator, so the measurement separates the arms.
