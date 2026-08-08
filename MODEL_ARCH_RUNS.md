# Mamba comparison

Everything needed for the four-arm comparison lives on
`edu-llm/OLMo-core` branch `edullm/mamba-comparison`. A push to that
`edullm/**` branch builds an image for the pushed commit; local or merely
staged changes are not part of that image. None of the commands in this
document dispatches a GPU job unless `edullm submit` is used explicitly.

The fan-out follows `edullm/mixer-bakeoff` at remote commit
`092f2c2bd582c4daa9b3bbfae0effce76b0f833a`: one image, one entrypoint, literal
arm/data/init arrays, arm-major ordering, matched token streams, a fixed step
count, and machine-readable seeds. The source of truth is
`docs/mamba-comparison/seeds.json`; the platform command is
`.edullm/run-comparison.yaml`.

## Frozen architectures

Every arm has 16 layers, `d_model=1024`, a tied 100,352-token embedding/LM
head, sequence length 4096, and identical PyTorch fused-SDPA GQA layers at
indices 3, 7, 11, and 15.

- `mamba-b3`: twelve Mamba-3 SISO layers with rotation block size 3.
- `xlstm`: `[mLSTM, mLSTM, mLSTM, attention, mLSTM, mLSTM, sLSTM, attention]`
  repeated twice, giving exactly 10 mLSTM, 2 sLSTM, and 4 attention layers.
- `mamba3-siso-pd`: twelve native SISO PD-SSM layers with the Mamba-3
  projection, normalization, and discretization improvements.
- `native-pd`: twelve published native Flash PD-SSM layers.

Attention FFNs remain fixed at width 4608. A deterministic `/32` solver changes
only recurrent-layer FFN widths. Exact totals are:

- `mamba-b3`: 390,148,736 parameters.
- `xlstm`: 390,143,056 parameters.
- `mamba3-siso-pd`: 390,169,664 parameters.
- `native-pd`: 390,142,976 parameters.

All are within 34,112 parameters (0.0088%) of the 390,135,552 target and inside
the frozen ±195,068 tolerance.

## Matched budget and wave

There are 20 cells: four arms by five replicates. Data seeds
210007/220014/230021/240028/250035 repeat across all arms, so replicate `r`
sees the same token order in every arm. Init seeds remain arm-specific because
the tensor inventories differ.

Each cell runs 3,721 steps at a 524,288-token global batch:

`3,721 × 524,288 = 1,950,875,648 tokens`.

That is TPP 5.00007–5.00041 across the four exact parameter totals. The step
count, corpus release, sequence length, global batch, DP world size, and data
seed must remain identical across all cells; changing one cell alone breaks
the paired token-stream contract.

Cells are arm-major:

- indices 0–4: `mamba-b3` (control);
- 5–9: `xlstm`;
- 10–14: `mamba3-siso-pd`;
- 15–19: `native-pd`.

Arm-major order is deliberate: truncation loses whole arms instead of reducing
every arm below five replicates.

## Platform preparation

First run the non-dispatching check against the comparison spec:

```bash
edullm check --json \
  --experiment mamba-comparison \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-comparison.yaml \
  --fanout-size 20 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX \
  --attempts 1
```

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
when the device peak is known. The inherited KDA/GDN decode microbenchmark is
disabled because it cannot measure these four operators honestly; it records
that omission rather than publishing a mismatched serving number.

## Smoke tests before the full wave

Use two different smoke tests; ten steps cannot answer the throughput question.

### Functional smoke: 10 steps

`.edullm/run-smoke.yaml` runs one seed of all four experiment arms plus a
parameter-matched GDN diagnostic control for ten steps. It checks the image,
strict CUDA backend selection, forward/backward, optimizer, distributed
collectives, and checkpoint writing. It explicitly skips the full held-out
pass, so `val_ce` is null. GDN is smoke-only and does not change the frozen
four-arm science wave.

```bash
edullm check --json \
  --experiment mamba-comparison-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-smoke.yaml \
  --compute gpu-8xa100 \
  --attempts 1 \
  --fanout-size 5 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Ten steps are sufficient only for a pass/fail smoke. The audited throughput
reporter excludes steps 1–50, so `throughput_tok_s_steady` is intentionally
null in this run. Do not rank arms using its whole-run or last-step rate.

### Throughput smoke: 100 steps

`.edullm/run-throughput-smoke.yaml` uses the same five cells for 100 steps.
After the fixed 50-step compile/allocator warmup, it reports 50 measured steps.
Its save interval is 101, so periodic checkpoint dispatch cannot contaminate a
timed step; the post-train hook still writes the final checkpoint.
The contemporaneous GDN cell is the only valid parity baseline for these
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
  --fanout-size 5 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Rank this smoke only by `throughput_tok_s_steady` and
`throughput_tok_s_steady_per_device`. It is useful for finding a grossly slow
arm before the 20-cell run, but one seed and 50 measured steps are not a final
performance estimate.

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
`xlstm==2.0.5`, and `flashrnn==1.0.6`. The NXAI license notice and confirmed
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
