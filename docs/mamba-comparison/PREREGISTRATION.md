# Pre-registration: Mamba and xLSTM comparison

Status: prepared before dispatch; no run has been submitted.

## Question

At matched parameter count, token budget, data order, attention schedule, and
training recipe, which recurrent architecture offers the best held-out
cross-entropy, steady-state training throughput, and peak-memory trade-off?

The control is `mamba-b3`. The treatments are `xlstm`,
`mamba3-pd`, `native-pd`, `gdn`, `kda`, `kda-hh-r2`, and `kda-gconv`.

## Frozen design

The machine-readable authority is `seeds.json`. There are eight arms by three
replicates, ordered arm-major. Every replicate shares its data seed across all
eight arms. Init seeds are not paired across arms because the parameter names,
shapes, and draw order differ.

Every cell consumes exactly 599,785,472 tokens: 1,144 steps at global batch
524,288. The exact models contain 390,094,784–390,170,432 parameters, giving
TPP 1.53723–1.53754. This matches the measured mixer-bakeoff Run 1 budget;
its original 1,907-step plan would have been TPP about 2.56, and the later
3,721-step Run 2 used TPP about 5. Sequence length is 4096 and every cell uses eight A100
GPUs, BF16 parameters, FP32 gradient reduction, compilation, the same
optimizer and schedule, and the same published dataset release.

Every arm has four identical attention layers at indices 3, 7, 11, and 15.
The other twelve layers are the named treatment. The xLSTM treatment is
specifically 10 mLSTM plus 2 sLSTM layers; it is not the earlier 8:4 control.
The `gdn` treatment is twelve frozen measured GatedDeltaNet2 layers; it is a
peer arm, not a second control, and its mixer stays exactly as the mixer-bakeoff
measured it.

The three KDA treatments are one family. `kda` is the shipped Kimi Delta
Attention operator and is the arm the other two are read against: `kda-hh-r2`
moves the number of Householder factors from one to two and allows negative
eigenvalues, and `kda-gconv` moves the three short convolutions from plain to
LIV-style depthwise-gated. Each differs from `kda` in one mechanism, so a
`kda-hh-r2 − kda` or `kda-gconv − kda` difference is attributable; a
`kda-hh-r2 − kda-gconv` difference is not and will not be reported as one.

Weight decay is uniform across the arms. Every arm exempts the timescale
parameters it has and names none it lacks: `mamba3-pd` and `native-pd`
exempt `A_log`, `dt_bias`, and `D`; `mamba-b3`, `gdn`, and the three KDA arms
exempt `A_log` and `dt_bias`, having no `D`; `xlstm` exempts nothing beyond the
embeddings, because neither of its recurrences carries such a parameter. The
earlier recorded asymmetry — `GatedDeltaNet2` alone decaying its timescales
because its mixer did not tag them — is closed, so it is no longer a confound
for the `gdn` contrast.

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

Decode latency and recurrent-state bytes are secondary endpoints; see the
deviation below that enabled the probe.

## Analysis

Report every arm's three cell values, mean, standard deviation, and its
arm-minus-control effect with an uncertainty interval. Use one shared-control
multiple-comparison procedure for the seven CE contrasts against `mamba-b3`.
Report throughput and memory ratios beside every CE contrast.

The two within-family KDA contrasts (`kda-hh-r2 − kda` and `kda-gconv − kda`)
are reported separately from that procedure and against `kda`, because their
control is not the wave's control. They are not a licence to run a third
comparison: `kda-hh-r2` against `kda-gconv` varies two mechanisms at once and is
uninterpretable.

Do not infer equivalence from a non-significant CE difference. If CE is
unresolved, say so and report the interval as the bound the run actually
supports. The fallback recommendation is the fastest and leanest arm not shown
to be worse on CE; it must be labelled as that fallback, not as a quality win.

There is no post-hoc cell deletion rule. A failed cell may be rerun only under
the platform's recorded retry semantics; `attempts=1` is frozen for the first
submission, so any later replacement is a documented deviation.

## Preconditions

- One commit and one image must contain all eight implementations.
- The 24 literal arm/data/init tuples in the run spec must agree with
  `seeds.json`.
- Exact parameter counts and the 10m/2s/4attention xLSTM role inventory must
  pass their tests.
- The target image must expose Mamba-3, native PD, mLSTM, FlashRNN, and `fla`'s
  `chunk_gdn2` and KDA kernels on sm80, must build the in-tree Householder
  kernel, and must carry the required license notice.
- `edullm check --json` must stand without `--force`.
- Any change to steps, batch, sequence length, corpus, world size, or seed
  schedule applies to all 24 cells and is recorded here before dispatch.

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
  The endpoint has two halves and one arm reports only one of them: the
  recurrent-state footprint is arithmetic on the geometry and is recorded for
  all eight, while the latency half needs a fused recurrent kernel and
  `KimiDeltaHouseholder` has none, so `kda-hh-r2`'s three cells carry
  `decode_fast_path_taken: false` and a `decode_basis` naming the missing
  kernel. That is a stated absence in the record and not a null: the analysis
  reads it as "not measured" and never as a fast decode, and no substitute
  number is written. Its state bytes remain comparable with every other arm's.
- Before dispatch, `gdn` was promoted from a throughput-diagnostic key beside
  the wave to a fifth arm inside it, taking the wave from 12 cells to 15. It is
  a peer treatment, not a second control; the control is still `mamba-b3`. The
  arm was appended rather than inserted, so indices 0–11 are unchanged and
  `--fanout-size 12` still reproduces the four-arm study exactly. Its mixer,
  geometry, step budget, batch, sequence length, corpus, and the three data
  seeds are the ones the other arms already use, and its five init seeds were
  already issued and reserved in the ledger, so no seed was reissued. The
  parameter band widens from 390,142,976–390,169,664 to
  390,119,360–390,169,664 and TPP from 1.53724–1.53735 to 1.53724–1.53745,
  both still inside the frozen ±195,068 tolerance. This adds a fourth CE
  contrast to the shared-control procedure. `gdn`'s untagged decay timescales
  were a confound specific to that contrast; the mixer tags them now and the
  weight-decay policy is uniform, so that confound is closed.
- Before dispatch, three Kimi Delta Attention arms — `kda`, `kda-hh-r2`, and
  `kda-gconv` — were appended, taking the wave from 15 cells to 24. They were
  appended and not inserted, so indices 0–14 are unchanged and `--fanout-size
  12` and `15` still reproduce the four- and five-arm studies exactly. Their
  mixers were ported from `edullm/mixer-bakeoff` and are verified there; their
  geometry, step budget, batch, sequence length, corpus, and the three data
  seeds are the ones the other arms already use, and their fifteen init seeds
  continue the existing schedule, so no seed was reissued. The parameter band
  widens from 390,119,360–390,169,664 to 390,094,784–390,169,664 and TPP from
  1.53724–1.53745 to 1.53724–1.53754, both still well inside the frozen
  ±195,068 tolerance. This takes the shared-control procedure from four CE
  contrasts to seven, and adds the two within-family contrasts described under
  Analysis. Both smoke specs were extended from five cells to eight in the same
  change, so every arm is rehearsed once functionally and once for throughput
  before the wave is submitted; the KDA cells are appended there too, leaving
  the five earlier smoke cells at their original indices.
- Before dispatch, `.edullm/train_core6_arm.py` gained a kernel preflight for
  the three KDA arms, matching the ones the other five already had. It is a
  precondition check and not a change to the model, the data, the budget or any
  endpoint: a host that cannot run the arm now refuses in the first seconds
  with a sentence naming the package, where before it would have been priced,
  admitted, given a machine and then died inside the first step.
- Before dispatch, `native-pd`'s scan chunk moved from 128 to 64 on a
  measurement: `paper_backward` at 2.98–3.00 ms per layer-step against
  2.59–2.72 ms, forwards level, about 0.3 ms per layer-step across twelve
  layers. This is a throughput change to one arm and not a capacity one — the
  chunk blocks the scan and shapes no weight, so the arm's 390,142,976
  parameters and its solved FFN widths are unchanged and its cells stay
  comparable with the four-arm study already described off them. It does mean
  `native-pd`'s throughput endpoint is not comparable with any figure measured
  on the earlier chunk.
- After the 24-cell wave and before the rerun, four arms moved from the
  reordered post-norm shell to the pre-norm one, and the two PD arms gained a
  head-wise gated output RMSNorm. `mamba3-siso-pd` was renamed `mamba3-pd` in
  the same change; it is the same arm at the same index, and the completed wave
  logged it under the old name.

  The shell moved for `mamba-b3`, `xlstm`, `mamba3-pd`, and `native-pd`, which
  are the four arms that normalize nothing at their mixer input. `gdn` and the
  three KDA arms stay post-norm because they L2-normalize `q` and `k` and gate a
  normalized output, so the shell cannot reach them. The evidence is an `xlstm`
  A/B: 132 non-finite parameter gradients under post-norm against none under
  pre-norm, then 3.5582 against 3.7124 held-out CE on an identical seed. The
  block type moves no parameter and measured within 0.5% on throughput.

  The output norm is head-wise over `d_state`, applied before the gate, which is
  the shape `fla`'s `FusedRMSNormGated` already gave `gdn` and the KDA arms and
  `norm_before_gate` gave `mamba-b3`. `mamba3-pd` had the option and had it off.
  `native-pd` had no normalization anywhere: as published it computes
  `out_proj(y · silu(gate))`, and the readout entering `out_proj` measured
  0.000, 0.030, and 3.973 at input ×1, ×10, and ×100 — superlinear, and
  arithmetically dead at initialization — against 0.075, 1.112, and 12.705 with
  the norm. This is a capacity change of 768 weights an arm, one gain a
  head-width across twelve layers. The parameter band widens from
  390,094,784–390,169,664 to 390,094,784–390,170,432 and TPP narrows from
  1.53724–1.53754 to 1.53723–1.53754, both still well inside the frozen
  ±195,068 tolerance.

  These are architecture changes to four of eight arms, so their cells are not
  comparable with the completed wave's and are reported as a second wave rather
  than pooled with it. The control is still `mamba-b3`, itself changed, so no
  contrast in the first wave carries over. Budget, batch, sequence length,
  corpus, seeds, and the four attention layers are untouched.
