# Mixer architecture and optimization review

Status: read-only architecture audits complete; optimization recommendations
below are not implemented. The comparison budget was separately corrected,
before dispatch, to the measured mixer-bakeoff Run 1 budget, and one
run-correctness defect found during the crash audit was fixed.

## Run-correctness defects found and fixed

### Fixed: weight decay was applied to every recurrence timescale parameter

The mixers tag `A_log`, `dt_bias`, and `D` with `_no_weight_decay`, but that tag
is inert on its own: `OptimConfig.build_groups` reads only `group_overrides`.
The runner that every spec dispatches exempted `embeddings.weight` alone, so
AdamW's 0.01 decayed the decay rates of all three state-space arms for the whole
run while every reported field still looked correct.

Copying the ledger's four-pattern list would have replaced a silent wrong number
with a hard crash. `TransformerTrainModule` builds the optimizer with
`strict=True`, and `_expand_param_globs` raises `OLMoConfigurationError` for a
pattern that matches nothing. Measured against the built models:

| Arm | Tagged parameters | Unmatched under one shared list |
| --- | ---: | --- |
| mamba-b3 | 24 (`A_log`, `dt_bias`) | `*.D` |
| xlstm | 0 | `*.A_log`, `*.dt_bias`, `*.D` |
| mamba3-siso-pd | 36 (`A_log`, `dt_bias`, `D`) | none |
| native-pd | 36 (`A_log`, `dt_bias`, `D`) | none |
| gdn | 0 | `*.D` |

The exemption is now per arm and derived from those tags. GDN2 stays empty on
purpose: it has `A_log` and `dt_bias` but does not tag them, and it is the frozen
bake-off control, so exempting them would change the baseline.

### Fixed: BF16 rounded the native Flash-PD decay away entirely

The paper mixer cast the complex diagonal to the activation dtype before the
scan. Just below 1.0 bfloat16 spaces its values about 1.95e-3 apart, so the
intended per-token decay exp(-5e-4) = 0.99950012 rounded to exactly 1.0. Over a
4096-token context the arm applied an attenuation of 1.0 rather than the
intended exp(-2.048) = 0.129, so it trained with no long-horizon decay at all.

The diagonal now crosses the scan boundary in FP32 while the payload stays
BF16, which is the mixed ABI the Mamba3-SISO mixer already used. The forward
kernels are templated on the two dtypes separately. FP32 accumulation, complex
semantics, collision routing, the Appendix-C surrogate gradients, and the
three-launch forward with five-launch backward are all unchanged, and no public
signature moved. The cost is one scan-boundary buffer at double width, which is
what the sibling arm already pays.

### Not a code defect: local CUDA test failures

Two separate environment effects have masqueraded as defects here.

Running CUDA tests inside the restricted sandbox fails with "Found no NVIDIA
driver", because the sandbox hides the GPU.

Running the whole `flash_pd_native` directory in one process reports roughly a
hundred failures both before and after any change. `host_sync_test.py`
deliberately fires an asynchronous device-side assertion to prove that
rejecting out-of-range indices needs no host readback, and a device-side assert
poisons the CUDA context for the rest of the process. Run that directory file
by file; per file it is 170 passed with one skip, and the skip wants a 16 GiB
device.

### Fixed: decode geometry read a registry that is not in this tree

`_decode_geometry` imported `olmo_core.nn.transformer.core6_arms`, a module that
left with the previous bake-off. Every committed spec passes
`--no-decode-probe` and the probe records failures rather than raising, so this
could not break the wave; it would instead have reported a missing decode
measurement the moment anyone enabled the probe, against the old study's
geometry of two mixer slots in sixteen layers.

The probe now reads the frozen ledger, exactly as the sLSTM prewarm contract
already does, and reports this study's real geometry: twelve recurrent layers
and four attention layers. No reference to the old registry remains in any code
path, and a test asserts the module is absent so the import cannot return.

Per-head decode state resolves only for arms whose mixer declares a head
dimension: `gdn` at 16 x 64 x 64 per layer and `mamba-b3` at 16 x 96 x 64.
`xlstm`, `native-pd`, and `mamba3-siso-pd` declare none, so those fields are
null with a stated reason rather than a plausible guess, because that number
sizes a serving fleet.

The probe remains off by default and no run spec changed.

## Executive conclusions

1. `mamba3-siso-pd` and `native-pd` do **not** execute Mamba-b3, SO(3),
   quaternion, Rodrigues, or a rotation-prefix scan. Their only rotation-like
   operation is an independent complex phase on each diagonal state value
   (`magnitude * exp(i * phase)`), equivalent to an abelian per-coordinate
   complex-plane rotation. It is not the non-commutative b=3 mechanism.
2. `mamba-b3` is the only active arm that intentionally executes
   `rotation_block_size=3` with the quaternion/Rodrigues path.
3. The stable RTX 5050 layer benchmark shows that both PD implementations are
   presently much slower than GDN2 at `B=2, T=1024, D=1024`. This is a real
   local finding, but it does not reproduce the Flash PD paper's A100,
   long-sequence benchmark.
4. Do not route Mamba3-SISO-PD through the paper scan. Its Python wrapper is
   already thin, its backward is already shared, and its separate forward
   preserves a fused three-launch trapezoidal path.
5. The highest-value next action for both PD arms is an A100 profile at the
   exact production shape. Plausible changes exist, but several superficially
   easy ones either leave the hot path unchanged or alter the model.

## Active comparison shell

All active comparison arms use 16 layers, `d_model=1024`, sequence length 4096,
a tied 100,352-token embedding/LM head, and the repeated pattern:

`[treatment, treatment, treatment, global attention] x 4`

Global attention is at zero-based layers 3, 7, 11, and 15. It is GQA with 16
query heads, 8 KV heads, head dimension 64, RoPE theta 500,000, and the PyTorch
SDPA backend. Attention FFNs remain width 4608; treatment-layer FFNs are solved
on a `/32` grid to keep every model near 390,135,552 parameters.

### Mamba-b3

`[Mamba3-b3, Mamba3-b3, Mamba3-b3, global] x 4`

- 12 Mamba-3 SISO layers.
- `d_state=96`, one B/C group, MIMO rank 1.
- `rotation_block_size=3`, `rotation_scan_impl="quaternion"`.
- High-occupancy `simple_gla` scan selected; fused input projections enabled.
- Recurrent FFNs: seven at 4800, five at 4768.

### xLSTM

`[mLSTM, mLSTM, mLSTM, global, mLSTM, mLSTM, sLSTM, global] x 2`

- 10 mLSTM layers using
  `mlstm-kernels==2.0.4:chunkwise--triton_xl_chunk`, chunk 256, BF16.
- 2 sLSTM layers at indices 6 and 14 using
  `flashrnn==1.0.6`, `cuda_fused`, BF16, fixed rank batch 2.
- Recurrent FFNs: eight at 4672, four at 4640.

### Mamba3-SISO-PD

`[Mamba3-SISO-PD, Mamba3-SISO-PD, Mamba3-SISO-PD, global] x 4`

- 12 native collision-capable PD layers.
- 16 heads, state 64, dictionary size 16, chunk 64.
- Forced `GENERAL_SCATTER`, native CUDA backend.
- Mamba-3 additions: exponential-trapezoidal beta/gamma input, complex
  diagonal phase, BCNorm, B/C biases, and short-convolution removal.
- No b=3 rotation and no MIMO.
- Recurrent FFNs: eleven at 2752, one at 2720.

### Native Flash-PD

`[Flash-PD, Flash-PD, Flash-PD, global] x 4`

- 12 paper-style native PD layers.
- 16 heads, state 64, dictionary size 16, chunk 128.
- Forced `GENERAL_SCATTER`, native CUDA backend.
- Mamba-style depthwise causal convolution of width 4.
- No Mamba-3 trapezoid and no b=3 rotation.
- Recurrent FFNs: six at 2432, six at 2400.

### GDN2 diagnostic

`[GDN2, GDN2, GDN2, global] x 4`

- 12 `GatedDeltaNet2` layers using `fla.ops.gdn2.chunk_gdn2`.
- 16 heads, head dimension 64, `expand_v=1`, negative eigenvalues disabled,
  short-convolution width 4.
- One recurrent FFN at 3808 and eleven at 3776.
- This is a frozen throughput diagnostic, not a fifth science arm.

## Default OLMo-core linear-attention hybrid

The canonical shipped model is OLMo-Hybrid-7B:

`[GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, global attention] x 8`

- 32 layers: 24 original `GatedDeltaNet` layers and 8 attention layers.
- It starts from OLMo3-7B and reduces `d_model` from 4096 to 3840 and attention
  heads from 32 to 30 for parameter/throughput matching.
- GDN key head dimension is 96, value expansion is 2, and negative
  eigenvalues are enabled.
- GDN uses FLA `chunk_gated_delta_rule`; attention uses FlashAttention-2.
- Although the inherited OLMo3 attention config carries the repeating
  `[4096, 4096, 4096, full]` SWA schedule, the hybrid's attention layers occur
  exactly at indices `3 mod 4`. Therefore all eight selected attention layers
  are global; the three SWA positions in each cycle are replaced by GDN.

The dense OLMo3-370M reference is not a linear-attention hybrid. It is 16
attention layers with `[SWA, SWA, SWA, global] x 4`.

## Stable local layer benchmark

Source hash was unchanged across the complete run. Measurements are one
forward+backward mixer layer on an RTX 5050 Laptop GPU, BF16,
`B=2, T=1024, D=1024`, after 20 warmup iterations and over 50 measured
iterations. They are not full-model or A100 results.

| Mixer | Median ms | Tokens/s |
| --- | ---: | ---: |
| xLSTM mLSTM | 9.015 | 227,183 |
| GDN2 | 12.188 | 168,041 |
| xLSTM sLSTM | 13.777 | 148,649 |
| Native Flash-PD | 19.082 | 107,328 |
| Mamba-b3 official | 19.859 | 103,128 |
| Mamba-b3 simple-GLA | 19.974 | 102,535 |
| Mamba3-SISO-PD | 20.164 | 101,566 |

At this shape native PD is 36.1% slower than GDN2 and Mamba3-SISO-PD is 39.6%
slower. The official Mamba-b3 path beat simple-GLA by 0.6% locally, so
simple-GLA must not be promoted from older microbench evidence.

## Why this does not reproduce the Flash PD paper

The paper reports Flash PD up to 4% faster than Mamba2 at sequence length 5120
on an A100 80GB. Its kernel analysis uses batch 32/state 128 for forward speed,
batch 192/state 128 for bandwidth, finds chunk 128 best, and reports about 82%
of A100 peak memory bandwidth.

The current measurements and implementation differ:

- RTX 5050 sm120 rather than A100 sm80.
- Sequence 1024 locally; production sequence 4096.
- Rank batch 2 and state 64.
- Full mixer projections/norms/convolution are timed, not only recurrence.
- The authors' production kernel has not been released; this CUDA
  implementation was reconstructed from the paper.
- Training includes a five-launch custom surrogate backward.
- Production forces collision-capable `GENERAL_SCATTER`; the paper's Appendix E
  describes a gather-oriented hot path.

The paper result is therefore a target, not proof that this implementation is
already optimized.

## Ranked optimization review

### P0: measurement gates before more PD edits

1. Run the existing chunk sweep `{32, 64, 128}` on A100 at
   `B=2, T=4096, H=16`, separately for native PD and Mamba3-SISO-PD.
2. Profile complete forward+backward with Nsight Systems, then use Nsight
   Compute on phase A, backward phase C, dictionary-gradient, and
   selector-gradient kernels.
3. Benchmark naturally initialized routes and explicit collision stress
   separately. A collision-only benchmark measures the worst case rather than
   the average trained route distribution.
4. Rank only by steady post-compile time; do not include extension build,
   preflight, FSDP wrapping, or the first 50 steps.

### P1: Mamba3-SISO-PD candidates

1. **Reduce pre-scan layout traffic, if profiling confirms it.** The wrapper
   currently permutes/casts/contiguously materializes diagonal, value, beta,
   gamma, and selector tensors before the custom op. A safe fix must preserve
   the mixed FP32-diagonal/BF16-payload ABI and raw-pointer router layout.
2. **Profile the five-launch backward.** Selector softmax-Jacobian
   post-processing is the most plausible fusion candidate, but it must preserve
   every Proposition-2 dictionary/router/input gradient.
3. **Profile the two scatter/gather stages inside phase A.** They may be
   algebraically necessary for trapezoidal affine composition; do not merge
   them from inspection alone.
4. **Do not drop the FP32 diagonal.** The near-unit long-horizon decay was
   previously lost when cast to BF16. Any reduced-precision alternative needs
   a 4096-step decay and all-gradient red-first test.
5. **Do not use AUTO mode as a speed fix.** Production already bypasses its
   Python bijection proof. Device-side proof would only help a path that is not
   currently hot.
6. **Do not force permutation-gather.** Random/learned column maps can collide;
   requiring bijection changes the model family.

### P1: native Flash-PD candidates

1. **Fuse the post-convolution input projections.** Native PD currently issues
   separate B, C, selector, dt, and phase linear projections. Packing compatible
   outputs into one GEMM is a plausible launch reduction and can retain
   checkpoint conversion compatibility.
2. **Benchmark the causal-convolution implementation.** Compare the current
   Conv1d/pad/transpose path against the pinned fused causal-conv kernel before
   replacing it.
3. **Tune chunk size on A100.** The paper found 128 best, while the local SISO
   contract selected 64 for a different recurrence. Keep per-arm winners.
4. **Do not remove all FP32 conversions blindly.** Native scan diagonal/payload
   follow the BF16 input dtype; dictionary and selector logits are promoted for
   surrogate-gradient calculations. Test those gradients before changing them.
5. The same scatter, backward-fusion, AUTO-mode, and gather cautions as the
   SISO arm apply.

### P1: Mamba-b3

1. The active comparison arm now uses `simple_gla + quaternion`, explicitly to
   test its higher-occupancy chunk-tiled grid on A100.
2. Validate that choice on an 8xA100 whole-model smoke before the full wave.
   Existing parity tests make it scientifically valid, while the RTX 5050 result
   still has `official_fast` ahead by 0.6%.
3. Keep `official_fast` reachable as the named fallback. It pads state 96 to 128
   because its state width is power-of-two, wasting 25% of those lanes.
4. Preserve Rodrigues, quaternion prefix, BCNorm ordering, and selective FP32
   prefix accumulation. Those are intentional b=3 semantics/stability work.

### P1: xLSTM

1. mLSTM is already the fastest measured mixer. Profile its packed projection
   `einsum` and layout conversions before replacing them with `Linear`; both may
   lower to the same GEMM.
2. The padded mLSTM backend is unused at 4096/256, but removing its construction
   is not a steady-state throughput optimization.
3. sLSTM remains slower than GDN2 locally, but only two layers use it. Preserve
   FlashRNN `cuda_fused`, BF16 pointers, exact-shape prewarm, and per-forward
   parameter conversion unless an FSDP-safe cache invalidation test exists.
4. Sweep mLSTM chunk size on A100 only after projection/layout profiling.

### Frozen GDN2 and shared attention

1. Do not optimize GDN2 internally; it is the frozen diagnostic.
2. The four shared attention layers currently use PyTorch SDPA, while the
   canonical OLMo hybrid uses FlashAttention-2. Any backend experiment must
   change every arm together and pass image-build and numerical gates.
3. PyTorch SDPA may already select its fused flash kernel on A100. Profile the
   selected backend before assuming a source-level FlashAttention package wins.

## Mixer-bakeoff setup alignment

The branch contains two different bakeoff budgets:

- Measured Run 1: 1,144 steps, 599,785,472 tokens/cell, TPP about 1.54.
- Run 1's original unexecuted plan: 1,907 steps, TPP about 2.56.
- Branch-head Run 2: 3,721 steps, 1,950,875,648 tokens/cell, TPP about 5.

The current four-arm wave now uses the measured Run 1 per-cell budget:

- 1,144 steps.
- 114 warmup steps.
- Save interval 572.
- Global batch 524,288 tokens.
- 599,785,472 tokens/cell.
- TPP 1.53724–1.53735 across exact arm parameter counts.

It retains the stronger Run 2-style wave structure: four arms by five
replicates, arm-major, with five shared data seeds. It is therefore
**budget-aligned to Run 1**, not a copy of Run 1's six-arm/three-seed study.

Other contracts remain aligned: sequence length 4096, rank microbatch 8192,
LR 1.4e-3, 10% warmup, eight A100s, BF16 compute, FP32 reduction, attempts 1,
and dataset release `reservoir-dolma2-v1`.

The architecture necessarily differs from the old bakeoff. Run 1 used six
global-attention layers at 2, 5, 8, 10, 12, and 14 and changed only treatment
slots 6 and 11. The current study tests complete 3:1 hybrids with twelve
treatment layers and four global-attention layers.

## Dataset and reader contract used

Both required dataset skills were read:

- `edullm-dataset-design` for interpreting the frozen dataset design.
- `edullm-datasets` for the existing published release and reader audit.

The target image pins `edullm-data` commit
`38bf831a6c3f445e394784018441fd59288b876c`. Its asserted live registry is:

- `eval-results/v1`
- `pretrain-tokens/v1`
- `sft-conversations/v1`
- `token-order/v1`
- `tokenizer/v1`

`reservoir-dolma2-v1` is read as `pretrain-tokens/v1`; paths, uint32 dtype,
tokenizer dependency, and held-out split come from the pinned
`dataset_paths()` reader. No private object path or hand-written manifest is
used.

## Wave design and GPU count

The wave is one image, one entrypoint, and literal arm/data/init arrays selected
by `AWS_BATCH_JOB_ARRAY_INDEX`. That is the correct bake-off pattern and the
index arrays are machine-checked against `docs/mamba-comparison/seeds.json`.

Every cell launches `--nproc-per-node=8` on one `gpu-8xa100` node:

| Wave | Cells | GPUs per cell | Maximum concurrent GPUs |
| --- | ---: | ---: | ---: |
| Functional smoke | 5 | 8 | 40 |
| Throughput smoke | 5 | 8 | 40 |
| Full comparison | 20 | 8 | 160 |

The `nodes: 1` field in a check describes one cell, not the fan-out. Actual
concurrency is set by the platform queue, so cells may start in groups.

Arm-major ordering is deliberate: a truncated fan-out loses whole arms rather
than reducing every arm below five replicates.

## Required gates for future code changes

Every implementation change should begin with a failing test and include:

- Exact forward and every-gradient parity against the existing reference.
- Collision and bijective route cases.
- Chunk tails for 32, 64, and 128.
- FP32/BF16 mixed-ABI parity and 4096-step decay preservation.
- Static launch-count and no-host-readback contracts.
- A source-hashed layer benchmark.
- An 8xA100 whole-model throughput smoke before changing a default.

