# Mamba comparison

Everything needed for the four-arm comparison now lives on the local branch
`edullm/mamba-comparison`. Nothing in this preparation pushes a branch or
submits a GPU job.

The fan-out follows `edullm/mixer-bakeoff` at remote commit
`092f2c2bd582c4daa9b3bbfae0effce76b0f833a`: one image, one entrypoint, literal
arm/data/init arrays, arm-major ordering, matched token streams, a fixed step
count, and machine-readable seeds. The source of truth is
`docs/mamba-comparison/seeds.json`; the platform command is
`.edullm/run-comparison.yaml`.

## Frozen architectures

Every arm has 16 layers, `d_model=1024`, a tied 100,352-token embedding/LM
head, sequence length 4096, and identical FlashAttention-2 GQA layers at
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

Only after the branch is committed, pushed, built, and the check has no
refusals would the corresponding `edullm submit` command be appropriate. Do
not use `--force`. The platform takes a commit, not this working tree.

The rank-0 JSON reports held-out CE, post-warmup throughput, per-device
throughput, real per-step peak allocated/reserved memory, FLOPs/token, and MFU
when the device peak is known. The inherited KDA/GDN decode microbenchmark is
disabled because it cannot measure these four operators honestly; it records
that omission rather than publishing a mismatched serving number.

## Image and dataset contract

One sm80 image contains Mamba-3, both native PD kernels, `mlstm-kernels==2.0.4`,
`xlstm==2.0.5`, and `flashrnn==1.0.6`. The NXAI license notice and confirmed
organizational research approval are embedded in the image.

Dataset reads use the `edullm-dataset-design` and `edullm-datasets` contracts.
The image pins `edullm-data` commit
`38bf831a6c3f445e394784018441fd59288b876c`; its asserted live registry is
`eval-results/v1`, `pretrain-tokens/v1`, `sft-conversations/v1`,
`token-order/v1`, and `tokenizer/v1`. The comparison selects
`reservoir-dolma2-v1` through the platform and resolves its paths, dtype,
tokenizer, and held-out split through that pinned reader. No object path or
hand-written manifest appears in the runner.
