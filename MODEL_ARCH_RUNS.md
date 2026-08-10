# Mamba comparison

Everything needed for the eight-arm comparison lives on
`edu-llm/OLMo-core` branch `edullm/mamba-comparison`. A push to that
`edullm/**` branch builds an image for the pushed commit; local or merely
staged changes are not part of that image. None of the commands in this
document dispatches a GPU job unless `edullm submit` is used explicitly.

The fan-out structure follows `edullm/mixer-bakeoff` at remote commit
`092f2c2bd582c4daa9b3bbfae0effce76b0f833a`: one image, one entrypoint, literal
arm/data/init arrays, arm-major ordering, matched token streams, a fixed common
step count, and machine-readable seeds. The per-cell budget deliberately matches
the bakeoff's measured Run 1 rather than that commit's later Run 2: 1,144 steps
and TPP about 1.54. The source of truth is
`docs/mamba-comparison/seeds.json`; the platform command is
`.edullm/run-comparison.yaml`.

## Frozen architectures

Every arm has 16 layers, `d_model=1024`, a tied 100,352-token embedding/LM
head, sequence length 4096, and identical PyTorch fused-SDPA GQA layers at
indices 3, 7, 11, and 15.

- `mamba-b3`: twelve Mamba-3 SISO layers with rotation block size 3,
  `d_state=192`, and the exact `official_fast` SSD backend.
- `xlstm`: `[mLSTM, mLSTM, mLSTM, attention, mLSTM, mLSTM, sLSTM, attention]`
  repeated twice, giving exactly 10 mLSTM, 2 sLSTM, and 4 attention layers.
- `mamba3-siso-pd`: twelve native SISO PD-SSM layers with the Mamba-3
  projection, normalization, and discretization improvements.
- `native-pd`: twelve published native Flash PD-SSM layers.
- `gdn`: twelve frozen measured GatedDeltaNet2 layers, on `fla`'s `chunk_gdn2`
  from the pinned FLA/fla-core 0.5.1.
- `kda`: twelve shipped Kimi Delta Attention layers, on `fla`'s KDA kernels from
  the same pin. One delta factor per token and plain SiLU short convolutions.
- `kda-hh-r2`: twelve Kimi Delta Attention layers with two Householder
  (DeltaProduct) factors per token and negative eigenvalues allowed. This is an
  in-tree kernel, not an `fla` one.
- `kda-gconv`: twelve Kimi Delta Attention layers whose three short convolutions
  are LIV-style depthwise-gated instead of plain.

`native-pd` uses a scan chunk of 64. It was 128 until a measurement put
`paper_backward` at 2.98–3.00 ms per layer-step there against 2.59–2.72 ms at
64, with the forwards level. A chunk size blocks the scan and shapes no weight,
so the arm's parameter count and FFN widths did not move.

Attention FFNs remain fixed at width 4608. A deterministic `/32` solver changes
only recurrent-layer FFN widths. Exact totals are:

- `mamba-b3`: 390,153,344 parameters.
- `xlstm`: 390,143,056 parameters.
- `mamba3-siso-pd`: 390,169,664 parameters.
- `native-pd`: 390,142,976 parameters.
- `gdn`: 390,119,360 parameters.
- `kda`: 390,119,360 parameters.
- `kda-hh-r2`: 390,119,360 parameters.
- `kda-gconv`: 390,094,784 parameters.

All are within 40,768 parameters (0.0104%) of the 390,135,552 target and inside
the frozen ±195,068 tolerance.

Every arm after the first is a peer treatment, not a control; the control
remains `mamba-b3`. `gdn`'s mixer is the frozen mixer-bakeoff GDN2 and must not
be optimized further, so it also serves as the contemporaneous speed reference
the smokes already use.

The three KDA arms are one family and their contrasts are pairwise against
`kda`. `kda-hh-r2` moves the number of delta factors and the eigenvalue sign
constraint; `kda-gconv` moves the convolution gating and nothing else, at a cost
of 6,144 parameters per layer — about 0.14% of the layer, which is what keeps
that contrast a mechanism rather than a capacity difference. A difference
between `kda-hh-r2` and `kda-gconv` varies two things at once and is not
attributable.

Weight decay is uniform across the wave: every arm exempts the timescale
parameters it actually has, and no arm names one it does not. `mamba3-siso-pd`
and `native-pd` exempt `A_log`, `dt_bias`, and `D`; `mamba-b3`, `gdn`, and all
three KDA arms exempt `A_log` and `dt_bias`, having no `D`; `xlstm` exempts
nothing beyond the embeddings, because neither of its recurrences carries such a
parameter. An unmatched pattern is fatal rather than inert — the optimizer is
built with `strict=True` — so these rows are asserted against the mixers' own
`_no_weight_decay` tags rather than maintained by hand.

## Matched budget and wave

There are 24 cells: eight arms by three replicates. Data seeds
210007/220014/230021 repeat across all arms, so replicate `r` sees the same
token order in every arm. Init seeds remain arm-specific because the tensor
inventories differ. Seeds 240028/250035 stay reserved in the ledger, so a later
wave can add depth without reissuing a seed.

Each cell runs 1,144 steps at a 524,288-token global batch:

`1,144 × 524,288 = 599,785,472 tokens`.

That is TPP 1.53724–1.53754 across the eight exact parameter totals. This is the
measured Run 1 budget; the bakeoff's original 1,907-step plan would have been
TPP about 2.56, while its later Run 2 used 3,721 steps and TPP about 5. The step
count, corpus release, sequence length, global batch, DP world size, and data
seed must remain identical across all cells; changing one cell alone breaks
the paired token-stream contract.

Cells are arm-major:

- indices 0–2: `mamba-b3` (control);
- 3–5: `xlstm`;
- 6–8: `mamba3-siso-pd`;
- 9–11: `native-pd`;
- 12–14: `gdn`;
- 15–17: `kda`;
- 18–20: `kda-hh-r2`;
- 21–23: `kda-gconv`.

Arm-major order is deliberate: truncation loses whole arms instead of reducing
every arm below three replicates.

Every arm after the fourth was appended rather than inserted, so each earlier
prefix still runs exactly the study it always did, off the same spec and the
same seeds: `--fanout-size 12` is the four-arm study, `--fanout-size 15` the
five-arm one, and `--fanout-size 24` the whole wave.

## Platform preparation

First run the non-dispatching check against the comparison spec:

```bash
edullm check --json \
  --experiment mamba-comparison \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-comparison.yaml \
  --fanout-size 24 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX \
  --attempts 1
```

To run only the original four arms, keep every other argument identical and use
`--fanout-size 12`; for the five-arm study use `--fanout-size 15`. The arms and
seeds of those cells do not move.

Read the JSON on stdout without combining stderr into it. Exit 0 stands, exit
1 is a refusal on the merits, exit 2 means the command is wrong, and only exit
3 is retryable. Match refusal `code`, not its prose.

Only after the intended files are committed, that exact commit is pushed and
built, and the check has no refusals would the corresponding `edullm submit`
command be appropriate. Do not use `--force`. The platform takes a commit, not
this working tree.

The check output is the sole authority for the current runtime bound, cost,
approval class, image state, and capacity. This document deliberately does not
quote those values; reviewed platform configuration changes independently of
this branch.

The rank-0 JSON reports held-out CE, post-warmup throughput, per-device
throughput, real per-step peak allocated/reserved memory, FLOPs/token, and MFU
when the device peak is known. The comparison spec also leaves the decode probe
on, so each cell records decode latency and recurrent-state bytes as secondary
endpoints; it runs on rank zero after the timed loop, records failures as a
reason rather than raising, and cannot move the throughput figure. See the
deviation recorded in `docs/mamba-comparison/PREREGISTRATION.md`.

## Smoke tests before the full wave

Use two different smoke tests; ten steps cannot answer the throughput question.

### Functional smoke: 10 steps

`.edullm/run-smoke.yaml` runs one seed of each of the eight arms for ten steps.
It checks the image, strict CUDA backend selection, forward/backward, optimizer,
distributed collectives, and checkpoint writing. It explicitly skips the full
held-out pass, so `val_ce` is null.

Both smoke specs cover all eight arms, in the wave's order, one cell each:
`mamba-b3`, `xlstm`, `mamba3-siso-pd`, `native-pd`, `gdn`, `kda`, `kda-hh-r2`,
`kda-gconv`. The three KDA arms were appended, so indices 0–4 are the five cells
these specs ran before and `--fanout-size 5` still reproduces exactly that
submission. They were added because nine of the wave's 24 cells would otherwise
have reached a machine unrehearsed, on three arms that have never run on this
platform and a study whose `--attempts` is 1; a missing KDA kernel or an
out-of-memory gated convolution would first have surfaced in the wave itself.
`test_smoke_fanout_seeds_match_the_frozen_arm_table` now asserts each spec's arm
list equals the wave's, so an arm added to the wave and not to the smokes is a
red test rather than a discovery on a billed machine.

```bash
edullm check --json \
  --experiment mamba-comparison-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-smoke.yaml \
  --compute gpu-8xa100 \
  --attempts 1 \
  --fanout-size 8 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Ten steps are sufficient only for a pass/fail smoke. The audited throughput
reporter excludes steps 1–50, so `throughput_tok_s_steady` is intentionally
null in this run. Do not rank arms using its whole-run or last-step rate.

### Throughput smoke: 100 steps

`.edullm/run-throughput-smoke.yaml` uses those same eight cells for 100 steps.
After the fixed 50-step compile/allocator warmup, it reports 50 measured steps.
Its save interval is 101, so periodic checkpoint dispatch cannot contaminate a
timed step; the post-train hook still writes the final checkpoint.
The contemporaneous `gdn` cell is the only valid speed parity baseline for these
12-recurrent/4-attention models; the older mixer-bakeoff used different layer
roles and cannot supply this denominator.

```bash
edullm check --json \
  --experiment mamba-comparison-throughput-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-throughput-smoke.yaml \
  --compute gpu-8xa100 \
  --attempts 1 \
  --fanout-size 8 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Rank this smoke only by `throughput_tok_s_steady` and
`throughput_tok_s_steady_per_device`. It is useful for finding a grossly slow
arm before the 24-cell run, but one seed and 50 measured steps are not a final
performance estimate, and it says nothing at all about the three arms it does
not run.

### One A100

The code supports a one-rank A100 functional smoke. At the last local
`edullm check`, the reviewed catalog exposed no `gpu-1xa100` profile and the
available A100 target was the eight-card `gpu-8xa100` node. Treat that as an
observed catalog result, not a permanent platform promise: inspect the current
check output before running. Do not reserve an eight-card node and silently use
only one card.

On a separately controlled single A100 with the built image, dataset reader
credentials, and the platform-provided `EDULLM_*` environment, use one process
and reduce the global batch so each optimizer step still has eight B2
microbatches:

```bash
python -m torch.distributed.run --nproc-per-node=1 --standalone \
  .edullm/train_core6_arm.py single-a100-smoke \
  --arm mamba-b3 --data-seed 210007 --init-seed 110007 \
  --sequence-length 4096 --steps 10 --warmup-steps 1 \
  --global-batch-size 65536 --rank-microbatch-size 8192 \
  --save-interval 10 --save-folder "$EDULLM_CHECKPOINT_DIR" \
  --param-dtype bfloat16 --skip-heldout-eval --no-decode-probe
```

Substitute the arm and matching init seed from
`docs/mamba-comparison/seeds.json`. Use 100 steps instead of 10 for throughput.
This direct one-A100 path is configuration-valid and targets sm80, but it is
outside the recorded submission path and remains unverified on real A100
hardware until that smoke actually runs. An `edullm run` or `edullm shell`
session is likewise exploratory and does not produce a citable platform run.

## Image and dataset contract

One sm80 image contains Mamba-3, both native PD kernels, `mlstm-kernels==2.0.4`,
`xlstm==2.0.5`, `flashrnn==1.0.6`, and `flash-linear-attention==0.5.1` for `gdn`,
`kda`, and `kda-gconv`. `kda-hh-r2` needs no additional package: its recurrence
is the in-tree kernel in `olmo_core.nn.attention.kda_householder`, which has no
`fla` counterpart. The NXAI license notice and confirmed
organizational research approval are embedded in the image. The command text
also names `bfloat16` explicitly; a dtype set only in Python is invisible to
the platform precision guard.

Dataset reads use the `edullm-dataset-design` and `edullm-datasets` contracts.
The image pins `edullm-data` commit
`38bf831a6c3f445e394784018441fd59288b876c`; its asserted live registry is
`eval-results/v1`, `pretrain-tokens/v1`, `sft-conversations/v1`,
`token-order/v1`, and `tokenizer/v1`. The comparison selects
`reservoir-dolma2-v1` through the platform and resolves its paths, dtype,
tokenizer, and held-out split through that pinned reader. No object path or
hand-written manifest appears in the runner.
