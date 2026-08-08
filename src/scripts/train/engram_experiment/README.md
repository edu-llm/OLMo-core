# Engram 400M MoE experiment

This directory defines an iso-token comparison of three reordered-norm MoE designs:

- **Base MoE** is the control: 12 layers, model width 384, 6 attention heads, and 64
  experts with hidden size 336.
- **Engram MoE** implements Engram ([arXiv:2601.07372](https://arxiv.org/abs/2601.07372)).
  It reduces the ordinary MoE to 51 experts and adds order-2/order-3 Engram memory to
  layers 2 and 6.
- **Lngram MoE** implements Lngram
  ([arXiv:2605.24869](https://arxiv.org/abs/2605.24869)). It likewise uses 51 experts and
  adds order-2/order-3, 4-bit latent routing with memory dimension 61 to layers 2 and 6.

Both memory arms use the papers' dilation-3, kernel-4 causal convolution. Lngram starts as an
exact no-op through zero tables/readout biases, uses separate routing and gate-query RMSNorms,
and accumulates gate similarity in float32.

The arm modules print these values from `TransformerConfig` during config-only inspection.
The constants in each arm pin the same values as a regression check.

| Arm | Total parameters | Active parameters |
| --- | ---: | ---: |
| Base MoE control | 392,373,120 | 113,681,280 |
| Engram MoE | 392,372,864 | 114,414,208 |
| Lngram MoE | 392,198,016 | 122,942,208 |

The maximum total-parameter spread is 175,104 parameters, about 0.045% of the control and
therefore **<= 1%**. The memory modules hold about 60.4M Engram parameters and 60.3M
Lngram parameters, within the intended 60–75M conditional-memory budget. Lngram's active
count is 9,260,928 (8.15%) above the control. This is a
projection-driven active-parameter bump, not an assertion that the whole latent table is
active: accounting includes Lngram's dense normalization, discretization, shared key/value
readout, and convolution parameters, plus only one selected table row per route and order.

Engram uses a committed compression map generated from the pinned Dolma2 tokenizer revision
`5292e5d6c0f40b67cc765fe41bec991cf4345b5c`. It applies NFKC, NFD, accent stripping,
lowercasing, and whitespace normalization, collapsing 100,278 tokenizer entries into 62,347
canonical IDs. The 74 padded matrix rows remain distinct, for 62,421 compressed IDs total.
The binary artifact is verified by SHA-256 before model construction.

## Lngram CUDA acceleration

Lngram keeps the exact PyTorch lookup and table-gradient implementation as its portable
reference path. On CUDA hosts where Triton is available, order-2/order-3 four-bit routing
automatically dispatches the counterfactual `grad_z` calculation to a fused kernel. The kernel
computes forced-bit table differences and upstream dot products without materializing full
counterfactual retrieval tensors. It uses bfloat16/float16 loads with float32 accumulation.

CPU runs, missing Triton, unsupported shapes or dtypes, and higher-order gradients continue to
use the reference implementation. FSDP2 ownership, activation checkpointing, and the three AWS
run specs are unchanged.

Benchmark the isolated backward at the production shape on one CUDA device with:

```bash
python src/scripts/benchmark_lngram_counterfactual.py
```

## Sealed schedule and pipeline laws

All arms consume exactly the same target schedule:

- Target: 10,000,000,000 tokens (10B), sequence length 2,048.
- Global batch: 524,288 tokens; rank microbatch: 16,384 tokens on 8 ranks.
- Duration: 19,074 optimizer steps, `ceil(10,000,000,000 / 524,288)`.
- One 8-GPU node per arm, FSDP2 only. Parameters are bfloat16 and reductions are float32;
  TP, CP, PP, and EP are disabled for this experiment.
- AdamW uses learning rate `4e-4`, weight decay `0.1`, betas `(0.9, 0.95)`, fused updates,
  cosine decay, 1,000 warmup steps, gradient norm 1.0, and z-loss multiplier `1e-5`.
  Embeddings have zero weight decay. Engram and Lngram lookup tables use five times the base
  learning rate and zero weight decay; dense memory parameters use the backbone optimizer.
- Compilation and selected-op activation checkpointing are enabled. Initialization seed
  12536 and data seed 34521 are common to all arms.
- There are **no evals**. This is a controlled pretraining comparison, not an evaluation
  pipeline.

Config construction is deliberately local-only: importing an arm or running it without the
literal `train` command prints accounting but performs no registry lookup, data read, or
training setup. The explicit `train` path resolves the corpus, prepares distributed state,
builds the native dataset and trainer, and then trains. A run is valid only if all three arms
use the unchanged scripts and specs from the same commit and the same registered corpus
release.

This architecture is revision `memory-paper-fidelity-v1`. It changes tokenizer addressing,
memory-layer placement, convolution behavior, Lngram parameters, initialization, and optimizer
groups. It therefore requires fresh run IDs/checkpoint prefixes for every arm and must not resume
from checkpoints produced by the earlier experiment revision. A persistent revision digest is
stored atomically with model state, so strict checkpoint loading rejects missing or mismatched
revisions even when a fallback `load_path` is supplied.

## Corpus contract

Runtime requires the exact **sealed registry identity**:

- dataset ID `pretrain/regmix-10b`
- version `v1`
- tokenizer `tokenizer/dolma2-bpe`
- padded vocabulary size 100,352

The manifest must explicitly declare headerless `uint32` tokens (`header_bytes = 0`) in
native little-endian order, and the host must itself be little-endian. The canonical object
layout is
`s3://edullm-data/pretrain/regmix-10b/v1/tokens/<source>/train-*.u32le.bin`, with held-out
objects named `val-*.u32le.bin`, `tokens/manifest.json`, and metadata in `v1/dataset.json`.

That layout is **DOCUMENTATION ONLY**. Registry resolution owns the concrete shard list and
storage adapter. The experiment code has no hardcoded run paths, no downloads, and no
fallback to `latest`, another tokenizer, inferred dtype, inferred byte order, or inferred
header size. `EDULLM_DATASET_ID`, `EDULLM_DATASET_VERSION`, and
`EDULLM_DATASET_TOKENIZER` must repeat the sealed values before a manifest is accepted.

## Submission contract

Replace each angle-bracketed value before invoking the command. Each
`<*-experiment-slug>` is a distinct experiment name. `<registered-regmix-release>` is the
platform release that resolves to the sealed `pretrain/regmix-10b`, `v1`,
`tokenizer/dolma2-bpe` identity above; it is not a shard path. Run each check first and only
submit its unchanged matching command after the check stands.

These are **three separate single-node submissions**, one independent node for Base, one for
Engram, and one for Lngram. They are not replicas in a three-node distributed job.

The platform runs a commit, never the working tree. Before checking any arm, commit every
experiment file, verify the tree is clean, and push the commit on an `edullm/<name>` branch
(this scaffold uses `edullm/engram-lngram-moe-400m`). Image publication is triggered from that
pushed branch; submitting before the commit is available remotely can launch an image that does
not contain these scripts or specs.

### Base control node

```bash
edullm check --json --spec .edullm/run-base.yaml --experiment <base-experiment-slug> --dataset <registered-regmix-release> --compute gpu-8xa100
edullm submit --spec .edullm/run-base.yaml --experiment <base-experiment-slug> --dataset <registered-regmix-release> --compute gpu-8xa100
```

### Engram node

```bash
edullm check --json --spec .edullm/run-engram.yaml --experiment <engram-experiment-slug> --dataset <registered-regmix-release> --compute gpu-8xa100
edullm submit --spec .edullm/run-engram.yaml --experiment <engram-experiment-slug> --dataset <registered-regmix-release> --compute gpu-8xa100
```

### Lngram node

```bash
edullm check --json --spec .edullm/run-lngram.yaml --experiment <lngram-experiment-slug> --dataset <registered-regmix-release> --compute gpu-8xa100
edullm submit --spec .edullm/run-lngram.yaml --experiment <lngram-experiment-slug> --dataset <registered-regmix-release> --compute gpu-8xa100
```

Each spec names bfloat16 in its command and launches exactly eight local processes. The
platform supplies `EDULLM_RUN_ID`, `EDULLM_CHECKPOINT_DIR`, and the three sealed dataset
identity variables; do not replace those with machine-specific values in a script.

## Checkpoint, resume, and observability behavior

`EDULLM_CHECKPOINT_DIR` is the per-run checkpoint prefix and is passed through the spec as
`--save-folder`; no storage prefix is embedded here or in an arm. Saving never overwrites the
folder. Checkpoints are written asynchronously every 1,000 steps and valid checkpoints are
retained.

Before resume, rank zero removes only a `step<N>` directory that the canonical checkpointer
identifies as torn, then all ranks synchronize. Complete checkpoints and unrelated
directories are preserved. The trainer then uses `maybe_load_checkpoint()` to resume the
latest valid state before fitting. Resume is supported only within the same
`experiment_revision` and immutable run commit; cross-revision checkpoints are intentionally
incompatible.

ConfigSaver receives the complete serialized experiment config, including the resolved
dataset identity and shard list, before checkpoint loading. W&B is opt-in: it is absent
unless `EDULLM_WANDB_PROJECT` is set; when enabled, that value selects the project and
`EDULLM_RUN_ID` selects the run name. The experiment does not invent a group or a local run
directory.

## Deferred distributed work

The single-node FSDP2 comparison does not claim support for the following. Keep these as
explicit implementation TODOs before changing the parallelism or serving topology:

- TODO: EP/TP table sharding with a distributed table-storage interface.
- TODO: inference host-offload prefetch for latent and explicit memory tables.
- TODO: CP/PP routing and sequence-boundary halos so n-gram routes cannot cross missing
  sequence context.
