# 07 — Latency, Kernels, Memory, and Edge Deployment

**Scope.** This document addresses the experiment's third core claim — that the proposed
architecture changes improve **latency and memory on real hardware**. FLOP formulas cannot
answer this, so everything here is either (a) a verified property of a real kernel / runtime /
tool, or (b) an explicit roofline calculation with stated assumptions.

**This file supersedes the earlier PARTIAL version** of `07_latency_kernels.md` (whose §1 and §2
are preserved and extended below; its four `[NOT RESEARCHED]` sections are now filled in).

**Epistemic conventions used throughout.**

| Tag | Meaning |
|---|---|
| **[VERIFIED]** | Read directly from source code, official docs, or a spec sheet. URL given. |
| **[CALC]** | My own arithmetic. Every step shown so it can be checked. |
| **[INFERENCE]** | A reasoned conclusion from verified facts. Labeled so it is not mistaken for fact. |
| **[UNKNOWN]** | Could not verify. Flagged rather than guessed. |

**Architecture under test (given).** LFM2-style. Gated short causal depthwise conv block ("LIV"):
`in_proj: d -> 3d`, chunked as `(B, C, x)`; `y = B * x`; `z = DWConv_causal_k(y)` with `k=3`,
depthwise (`groups=d`), no bias; `out = out_proj(C * z)`. **No activation anywhere in the conv
path.** RMSNorm outside the block. Interleaved with GQA attention layers.

| | geometry (a) | geometry (b) |
|---|---|---|
| `d` | 1024 | 2048 |
| layers | 16 (10 conv / 6 GQA) | 16 (10 conv / 6 GQA) |
| q / kv heads, head_dim | 16 / 8, 64 | 32 / 8, 64 |
| SwiGLU `ff` | 4608 | 8192 |
| vocab (tied) | 65536 | 65536 |
| params | ~354 M | ~1.170 B |

**Three proposed changes.**
1. **Low-rank factorized gates** — the two gate projections become `d -> r -> d`, `r=128`;
   value and out projections stay full width.
2. **Cross-layer KV sharing (CLA)** — pairing attention layers.
3. **Multi-branch dilated conv** — four 3-tap branches, dilations 1/2/4/7, token-dependent
   softmax router.

---

## 0. Established results carried in (independently re-verified)

I **re-derived all of the carried-in numbers from scratch** and they reproduce exactly. The
arithmetic is reproduced here so this document stands alone.

### 0.1 Parameter counts [CALC — confirms ~354 M / ~1.170 B]

Formulas used (bf16, tied embeddings, no linear biases, RMSNorm weight `d` each):

```
conv block = d*3d (in_proj) + d*d (out_proj) + d*k (conv weight, k=3)
GQA block  = d*(hq*hd) + 2*d*(hkv*hd) + (hq*hd)*d
SwiGLU     = 2*d*ff (gate+up) + ff*d (down)
embedding  = vocab*d           (tied, counted once)
norms      = 16*2*d + d
```

| component | (a) d=1024 | (b) d=2048 |
|---|---|---|
| conv block (each) | 4.20 M | 16.78 M |
| GQA block (each) | 3.15 M | 10.49 M |
| SwiGLU (each) | 14.16 M | 50.33 M |
| embedding (tied) | 67.11 M | 134.22 M |
| **non-embedding** | **287.4 M** | **1036.1 M** |
| **total** | **354.5 M** | **1170.3 M** |

### 0.2 Weight bytes read per decode token [CALC — confirms]

bf16 = 2 B/param; assumes every weight is touched once per decode step (true for a dense model
at batch=1; the tied embedding is read once as the LM head).

```
(a) 354.48e6 * 2 =  708.96e6 B =  708.96 MB   ✓ matches "708.9 MB/token"
(b) 1170.34e6 * 2 = 2340.68e6 B =   2.3407 GB ✓ matches "2.341 GB"
```

### 0.3 KV bytes per token [CALC — confirms 12,288 B]

6 GQA layers, `hkv=8`, `hd=64`, both K and V, bf16:

```
KV/token = 6 * 8 * 64 * 2 (K and V) * 2 B = 12,288 B = 12.0 KiB   ✓
```

**Scale-invariant** across (a) and (b) — `hkv`, `hd`, and attention-layer count are identical in
both. That is the crux of §8.

All-16-GQA control: `16 * 8 * 64 * 2 * 2 = 32,768 B = 32 KiB`.

### 0.4 Crossover: context at which KV read == weight read [CALC — confirms]

Solve `T * KV_per_token = weight_bytes`:

```
(a)                    T =  708.96e6 / 12288 =  57,696   ✓ (stated 57,690)
(b)                    T = 2340.68e6 / 12288 = 190,485   ✓ (stated 190,474)
(b) all-16-GQA control T = 2340.68e6 / 32768 =  71,432   ✓ (stated 71,428)
```

### 0.5 KV share of decode traffic [CALC — confirms]

`share = T*KV / (T*KV + weight_bytes)`:

| context T | (a) 354 M | (b) 1.17 B |
|---|---|---|
| 4,096 | **6.63 %** ✓ (6.6%) | **2.11 %** ✓ (2.1%) |
| 32,768 | **36.22 %** ✓ (36.2%) | **14.68 %** ✓ (14.7%) |

### 0.6 The key insight, restated — and its limit

**The 350 M model is the better testbed for any KV claim.** KV bytes/token is identical (12 KiB)
in both geometries while weight bytes shrink 3.30x from (b) to (a), so the KV term is
`2340.7/709.0 = 3.30x` more prominent in (a), and the crossover drops 190 K → 58 K tokens.
Confirmed.

> ⚠️ **Note the direction of this insight.** It makes KV effects *easier to see*. It does **not**
> make them *large* at trainable context lengths. §8 bounds this and the honest answer is
> uncomfortable.

### 0.7 Constraints carried in (not revisited)

- **[VERIFIED, carried in]** Dao-AILab `causal-conv1d`: README states kernel size **2, 3, 4
  only**; `causal_conv1d_fn(x, weight, bias=None, activation=None)` has **no dilation argument**.
  → dilation unsupported; widths 5/9/15 out of range.
  <https://github.com/Dao-AILab/causal-conv1d>
- **[VERIFIED, carried in]** CLA saves cache **capacity**, not **bandwidth**: *"no direct effect
  on the memory bandwidth consumed by the attention mechanism in each decoding step."*
  → **Proposal 2 has ~0 decode-latency effect at every context length.** Decisive; not revisited.

---

## 1. Kernels for depthwise causal convolution

### 1.1 `fla.modules.convolution.causal_conv1d` — exact signature [VERIFIED]

Source read directly from `main` of `fla-org/flash-linear-attention`.

**Structural fact first:** `fla/modules/convolution.py` is now only a **re-export shim** (42
lines, pure re-exports). The real code moved to the `fla/modules/conv/` **package**.

- Shim: <https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/convolution.py>
- Real entry point: <https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/conv/causal_conv1d.py>

So `from fla.modules.convolution import causal_conv1d` — exactly what the local OLMo-core wrapper
does — still works, but you are calling `fla.modules.conv.causal_conv1d.causal_conv1d`.

**The signature [VERIFIED — transcribed from source]:**

```python
@input_guard(no_guard_contiguous=["x"])
def causal_conv1d(
    x: torch.Tensor,                            # [B, T, D]
    weight: torch.Tensor | None = None,         # [D, W]
    bias: torch.Tensor | None = None,           # [D]
    residual: torch.Tensor | None = None,       # [B, T, D]
    initial_state: torch.Tensor | None = None,  # [N, D, W]
    output_final_state: bool | None = False,
    activation: str | None = None,              # 'swish'/'silu' or None
    backend: str | None = 'triton',             # 'triton' | 'cuda' | 'mix'
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    **kwargs,
):
```

**Q: Does it support dilation? — [VERIFIED] NO.**
There is **no `dilation` parameter** in the signature and none is threaded into the kernels.
Confirmed by reading `fla/modules/conv/triton/ops.py` and `.../triton/kernels.py`; the forward
kernel's tap loop is:

```python
o_w = tl.arange(0, BW) + W - BW
for i_w in tl.static_range(-W + 1, 1):
    o_x = o_t + i_w          # <-- stride-1 offsets ONLY; no dilation factor
```

A dilated version needs `o_t + i_w * dilation`. **So `fla` gives you nothing for proposal 3
either — both candidate off-the-shelf kernels are dilation-free.**

**Q: What kernel widths does it allow? — [VERIFIED] Arbitrary `W`.** This is the single most
useful finding in this section. Unlike Dao-AILab's package (hard-limited to W ∈ {2,3,4}), `fla`'s
Triton backend computes:

```python
# fla/modules/conv/triton/ops.py
W  = weight.shape[1]
BW = triton.next_power_of_2(W)
```

`W` is a `tl.constexpr` and the tap loop is `tl.static_range(-W + 1, 1)`. There is **no width
table, no `if W == 2/3/4` dispatch, and no upper-bound assert on `W`** anywhere in `ops.py` or
`kernels.py`. Width is a compile-time-specialized unrolled loop.

→ **[INFERENCE, high confidence] `fla`'s Triton backend will compile and run at W=15 (BW=16) or
W=22 (BW=32).** This is the escape hatch for proposal 3 (§1.7). Caveats: this is inferred from
reading code, **not documented**, and **not covered by fla's tests for large W** as far as I could
tell — validate numerically against `F.conv1d` before trusting it. Register pressure also grows
linearly with the unrolled `W`, so large `W` will eventually spill.

**Q: What does it RETURN? — [VERIFIED] a tuple. Your belief is correct.**
Docstring: *"Returns: Tuple of (output, final_state). If `output_final_state` is `False`, the
final state is `None`."* All branches return 2 values (`triton` → `y, final_state`; `cp_context`
→ `output, None`; `cuda`/`mix` → the underlying tuple).

⚠️ **State-shape convention differs from Dao-AILab.** `fla`'s state is `[N, D, W]` — the **full
W-wide window**. `causal_conv1d_update_states` allocates `torch.empty(N, D, W)` and
`ShortConvolution.state_size` returns `hidden_size * kernel_size`. Dao-AILab's convention is
`W-1`. **Do not mix them.** The local OLMo-core wrapper does `return output[0]`, indexing element
0 out of the tuple and **discarding the state**.

**Q: How does the fused `activation` argument work? — [VERIFIED]**
It is a **string**, not a callable, passed as a Triton `constexpr` so it is compiled into the
kernel — genuinely fused, no extra launch, no extra HBM round-trip. In `kernels.py`:

```python
ACTIVATION: tl.constexpr,
...
if ACTIVATION == 'swish' or ACTIVATION == 'silu':
    b_y = b_y * tl.sigmoid(b_y)
```

Only `'swish'`, `'silu'`, `None`. The CUDA backend explicitly raises:
`if activation not in [None, "silu", "swish"]: raise NotImplementedError(...)`.

**Q: Is `activation=None` supported (LFM2 needs it)? — [VERIFIED] YES, and it is the default.**
Three independent confirmations: (1) the signature declares `activation: str | None = None`;
(2) the Triton activation is behind `if ACTIVATION == 'swish' or ACTIVATION == 'silu'`, so `None`
simply skips it — no separate code path; (3) the CUDA backend's validator explicitly allows `None`.

> ### ⚠️ ACTIONABLE CORRECTNESS BUG — the wrappers default to SiLU, LFM2 needs none
>
> `/Users/ericwu/Developer/Capstone_LLM/OLMo-core/src/olmo_core/nn/convolution.py` line 27:
> ```python
> activation: Literal["silu", "swish"] | None = "silu",
> ```
> and `fla`'s own `ShortConvolution.__init__` has the identical default
> (`activation: str | None = 'silu'`, with `assert activation in ['silu','swish']`).
>
> **LFM2's conv path has no activation.** Instantiating either class without explicitly passing
> `activation=None` silently trains a **different architecture** — a SiLU fused into the conv
> output. The local wrapper's type (`Literal[...] | None`) does permit `None`, so the fix is one
> word per construction site:
> ```python
> CausalConv1d(hidden_size=d, kernel_size=3, bias=False, activation=None)   # LFM2-correct
> ```
> Grep every construction site before any training run. This is **not** a latency issue — it is a
> correctness issue that would invalidate the entire comparison.

### 1.2 `fla` backend options [VERIFIED]

| `backend` | Implementation | Notes |
|---|---|---|
| `'triton'` (**default**) | `fla/modules/conv/triton/` — own fwd + bwd + update kernels | Pure Triton, no external dep. **Arbitrary W.** Supports `initial_state`, `output_final_state`, `residual`, `cu_seqlens` (varlen). |
| `'cuda'` | Thin wrapper over **Dao-AILab `causal_conv1d`** | `from causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_cuda`; warns and falls back if absent. **Inherits W ∈ {2,3,4}.** |
| `'mix'` | `fast_causal_conv1d_fn` — Dao-AILab fwd + fla Triton bwd | **Asserts `output_final_state is False`, `initial_states is None`, `residual is None`.** Training-only fast path. |

Three further **[VERIFIED]** benchmarking gotchas from the source:

1. **Environment-variable override.** `ShortConvolution.__init__` does
   `self.backend = os.environ.get('FLA_CONV_BACKEND', backend)`. A stray `FLA_CONV_BACKEND` will
   silently change which kernel you benchmark. **Unset it explicitly in the harness and log the
   resolved `self.backend`.**
2. **Silent backend downgrade.** `ShortConvolution.forward` downgrades `'cuda'` → `'triton'` when
   a `cache` is provided (with a `warnings.warn`). A prefill-with-cache and a
   prefill-without-cache benchmark may not be running the same kernel. Log per call.
3. **Decode dispatch is implicit and shape-driven.** `forward` checks `if B * T == N` and routes
   to `self.step()` → `causal_conv1d_update`. "Decode" is selected by tensor shape, not a flag —
   so an accidentally-shaped input silently changes kernels.

### 1.3 Version pinning [VERIFIED]

Both `fla-core` and `flash-linear-attention` are at **0.5.2** on PyPI, history
`0.3.1/0.3.2, 0.4.0, 0.4.1, 0.4.2, 0.5.0, 0.5.1, 0.5.2`.
<https://pypi.org/pypi/fla-core/json> · <https://pypi.org/pypi/flash-linear-attention/json>

⚠️ **All source quotes in §1.1–1.2 are from `main`, not a release tag.** The
`fla/modules/convolution.py` → `fla/modules/conv/` package split is exactly the refactor that
breaks pinned code; the local wrapper survives only via the shim. **Pin `fla-core==0.5.2`** and
re-read the source on any bump. (OLMo-core separately requires
`flash-linear-attention >= 0.4.1` for `chunk_kda`.)

### 1.4 Where the local codebase wraps this [VERIFIED]

- `/Users/ericwu/Developer/Capstone_LLM/OLMo-core/src/olmo_core/nn/convolution.py` —
  `class CausalConv1d(nn.Conv1d)`, constructed as `nn.Conv1d(in_channels=hidden_size,
  out_channels=hidden_size, kernel_size=k, groups=hidden_size, bias=bias, padding=k-1)`. Weight
  is therefore stored in `nn.Conv1d` layout `[D,1,W]` and squeezed to `[D,W]` at call time
  (`weight.squeeze(1)`). It **returns `output[0]`, discarding the conv state** — so this wrapper
  is **prefill/training-only and cannot do incremental decode at all.** Threading
  `initial_state`/`output_final_state` through is a prerequisite for any decode benchmark.
- `.../nn/attention/flash_linear_attn_api.py` line 130 — `dispatch_causal_conv1d(x, weight, bias,
  activation, backend='triton', cu_seqlens)`. Exposes neither `residual`, `initial_state`,
  `output_final_state`, nor `dilation` (there is none to expose). All four need adding.
- `CausalConv1d.apply_cp()` implements Ulysses-style **channel-parallel** CP by slicing channels
  rather than sharding params — irrelevant to single-GPU latency, but explains the `cp_enabled`
  branches.

### 1.5 PyTorch native `F.conv1d` with `groups=d`

**[VERIFIED] It is the functional reference,** and Dao-AILab's README defines the equivalence:
`causal_conv1d_fn` == `F.conv1d(x, weight.unsqueeze(1), bias, padding=width-1, groups=dim)[..., :seqlen]`
at dilation 1. **`F.conv1d` does accept a `dilation` argument** — making PyTorch native the
**only** option here that can express proposal 3 without new kernel code.

**[CALC] Why it is bandwidth-bound, not compute-bound.** Per output element of our op:
- FLOPs: `2*W` = **6 FLOP** at W=3.
- Bytes: at best 1 read + 1 write amortized (the sliding window overlaps and is cache-served) =
  **4 B** in bf16.
- **Arithmetic intensity ≈ 6 / 4 = 1.5 FLOP/B.**

An A100's bf16 ridge point is ≈ 312 TFLOP/s ÷ 1.55 TB/s ≈ **200 FLOP/B**. The op sits
**~130x below the ridge point.** It is bandwidth-bound by two orders of magnitude and *no kernel,
however good, can change that.*

Three consequences that shape the whole experiment:
1. The op's floor is one read + one write of the activation tensor. Any implementation near that
   is optimal; **the only remaining lever is fusing it with its neighbours** so the tensor never
   round-trips to HBM (§1.6, §2).
2. **FLOP accounting will wildly understate the conv's cost.** `torch.utils.flop_counter` reports
   ~6 FLOP/element for an op whose runtime is set entirely by bytes. **Never report conv FLOPs as
   a proxy for conv latency** — this is exactly the trap the source document warns about.
3. Low occupancy is secondary: with `groups=d`, each group has one channel, so the reduction
   dimension a conv kernel would parallelize over has length 1, leaving a thin, launch-bound
   kernel.

**[UNKNOWN — flagged, resolvable empirically in one afternoon]** In **eager** mode I could not
verify which dispatch path PyTorch takes for `conv1d` with `groups=d` on CUDA: (i) cuDNN grouped
conv, (ii) unsqueeze-to-4D onto the special-cased
`aten/src/ATen/native/cuda/DepthwiseConv2d.cu` kernel, or (iii) a generic implicit-GEMM path.
**There is no `DepthwiseConv1d.cu` in ATen**; 1-D convs are generally unsqueezed to 2-D. This
matters because the 2-D depthwise kernel is well optimized whereas cuDNN grouped conv historically
is not, and a *dilated* variant is more likely to fall off any fast path. **Resolve this by
measurement, not literature:** run the conv under `torch.profiler`/`nsys` and read the kernel name
off the trace. `cudnn::...grouped...` = slow path;
`at::native::conv_depthwise2d_forward_kernel` = fast path.

(Under `torch.compile(mode="max-autotune")` this is **resolved** — Inductor has a purpose-built
depthwise-conv1d Triton template at dilation 1; see §1.6. The eager-mode question only matters for
the eager baseline you compare against.)

### 1.6 torch.compile on depthwise causal conv — and the highest-leverage trick in this document

**[VERIFIED — and the answer is better than I expected, with one fatal catch.]** I read
Inductor's convolution lowering directly:
<https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/conv.py>

**Inductor ships a dedicated depthwise-conv1d Triton template.** From the source, verbatim:

```python
# =============================================================================
# Depthwise conv1d (groups == in_channels == out_channels)
# Uses direct element-wise multiply-accumulate instead of implicit GEMM.
# Channels-last (NLC) layout with 3D tiling: BLOCK_N x BLOCK_L x BLOCK_C.
# =============================================================================
```

and it is selected by exactly our case:

```python
is_depthwise = groups > 1 and in_chan == 1 and out_chan == groups
if is_depthwise and ndim == 1:
    depthwise_configs = V.choices.get_depthwise_conv_configs(device_type)
    ...  depthwise_conv1d_template.maybe_append_choice(...)
```

Note *"direct element-wise multiply-accumulate instead of implicit GEMM"* — the PyTorch developers
independently arrived at the same conclusion as §1.5: for depthwise conv, implicit GEMM is the
wrong algorithm and a pointwise MAC formulation is right.

**⚠️ TWO GATES, and both matter to this experiment:**

1. **`max_autotune` is required.** The Triton template path is guarded by
   `use_triton_template(layout)`, which requires
   `config.max_autotune or config.max_autotune_gemm` (verified in
   `torch/_inductor/utils.py::use_triton_template`). **Under a plain `torch.compile(model)` you get
   the ATen/cuDNN extern kernel, not the Triton depthwise template.** You must compile with
   `mode="max-autotune"` (or set `torch._inductor.config.max_autotune=True`). Backend selection is
   further filtered by `config.max_autotune_conv_backends` (`_use_conv_autotune_backend("TRITON")`).
2. **`is_ones(dilation)` — the template is skipped for any dilation ≠ 1.** Verbatim from the
   choice-gating condition, with the source's own comment:
   ```python
   torch._inductor.utils._use_conv_autotune_backend("TRITON")
   and use_triton_template(layout)
   # templates only support these:
   and is_ones(dilation)
   and not transposed
   and is_zeros(output_padding)
   ```
   → **Proposal 3's dilated convs fall back to the ATen/cuDNN extern kernel** even under
   `max-autotune`. **This is the third independent confirmation that dilation has no fast path
   anywhere** (Dao-AILab: no dilation arg; `fla` Triton: stride-1 offsets only; Inductor: template
   gated on `is_ones(dilation)`). The finding is now robust across all three candidate stacks.

**And the fusion question — still the one that matters.** Even with the Triton depthwise template
available at dilation 1, it is a **template**, so Inductor can fuse a pointwise **epilogue** onto
it but generally not a **prologue**. The gate multiply `y = B*x` *precedes* the conv, so it is the
harder direction; expect it to remain a separate kernel with a full HBM round-trip of a `[B,T,d]`
tensor. When the template is *not* selected (no `max_autotune`, or any dilation), the conv is an
**extern kernel** and Inductor cannot fuse into it at all.

**The practical trick.** At W=3 there is no need to call a conv op at all. Write the conv as an
explicit unrolled sum of causally shifted multiplies:

```python
z = w0 * y + w1 * shift(y, 1) + w2 * shift(y, 2)      # causal shifts
```

This is **pure pointwise + indexing** — precisely what Inductor fuses best. It should fuse with
the preceding `y = B*x` and potentially the following `C*z` into **one** Triton kernel,
eliminating two full HBM round-trips of a `[B,T,d]` tensor. At W=3 the redundant reads are cheap
and largely cache-served.

**[INFERENCE — verify by reading the generated code]** Run with `TORCH_LOGS=output_code` and count
kernels. One fused pointwise kernel instead of three is a real win for **~an hour of work, no
Triton required.** I would test this before writing any custom kernel. It also **generalizes to
dilation for free** — `shift(y,2)`, `shift(y,4)`, `shift(y,7)` are just different offsets (§1.7).

**[UNKNOWN — narrowed]** Whether Inductor's `depthwise_conv1d_template` supports *prologue* fusion
(folding `y = B*x` into the template's input load) I could not confirm. Prologue fusion has been
landing incrementally in Inductor for mm templates; whether it applies to the conv templates is
unverified. **This is exactly why the unrolled-shift form is the recommended path** — it sidesteps
the question entirely by making the whole block pointwise, where fusion is guaranteed rather than
hoped for.

**Also note the layout requirement:** the depthwise template is documented as
*"Channels-last (NLC) layout"*. Our activations are `[B, T, D]`, which **is** NLC (channel-last) —
so the natural LFM2 layout is already the favourable one. But `nn.Conv1d` expects `[B, D, T]`
(NCL), so the OLMo-core wrapper's use of `nn.Conv1d` implies transposes somewhere. Check whether
transposes are appearing in the profile; `fla`'s `causal_conv1d` takes `[B, T, D]` directly and
avoids them, which is another reason to prefer it over `F.conv1d`.

### 1.7 A dilated multi-branch conv: what it would take

| Option | Dilation? | W=15? | Verdict |
|---|---|---|---|
| Dao-AILab `causal-conv1d` | **No** [VERIFIED, carried in] | **No** (2/3/4) | Unusable for P3 |
| `fla` backend `'cuda'`/`'mix'` | No (wraps the above) | No | Unusable |
| `fla` backend `'triton'` | **No** [VERIFIED §1.1] | **Yes** [VERIFIED — `next_power_of_2(W)`, no cap] | **Usable via masked-dense** |
| `F.conv1d(groups=d, dilation=k)` | **Yes** | Yes | Works; slow; no fused state/decode path |
| Inductor `depthwise_conv1d_template` | **No** — gated `is_ones(dilation)` [VERIFIED §1.6] | Yes | Needs `max-autotune`; dilation falls back to extern |
| Unrolled shifts + `torch.compile` | **Yes** (offsets are free) | Yes | **Most promising** |
| Hand-written Triton | Yes | Yes | Last resort |

**Dilation has no fast path in any of the three stacks** — Dao-AILab (no arg), `fla` Triton
(stride-1 offsets), Inductor (template gated on `is_ones(dilation)`). Triple-confirmed.

**The masked-dense reformulation — recommended.** Four 3-tap branches at dilations 1/2/4/7 have
tap offsets:

```
dilation 1: {0, 1,  2}
dilation 2: {0, 2,  4}
dilation 4: {0, 4,  8}
dilation 7: {0, 7, 14}
union = {0,1,2,4,7,8,14}  ->  7 distinct offsets
max offset = 14           ->  receptive field = 15 taps
```

**[CALC]** All four branches live inside a **dense causal window of W=15**. Since `fla`'s Triton
kernel takes arbitrary `W` and specializes on it (`BW = next_power_of_2(15) = 16`), implement each
branch as a **W=15 dense conv with a fixed zero mask** on the unused taps — reusing the
**existing, tested, differentiable kernel with its own backward pass and decode state**, writing
**zero** new CUDA/Triton code.

**Is the masking wasteful? Essentially no — and this is the key point.** The op is bandwidth-bound
at ~1.5 FLOP/B (§1.5). Going 3 → 15 taps multiplies *FLOPs* by 5 but leaves the *bytes* — the
activation read and write, which set the runtime — **unchanged**. Extra weight bytes at d=2048 are
`2048*15*2 = 61,440 B` per branch vs `2048*3*2 = 12,288 B`: an absolute increase of **~49 KB per
conv layer**, which is noise against a **2.34 GB/token** weight read. You pay 5x FLOPs on an op
that is 130x from being compute-bound. **[CALC + INFERENCE]**

Caveats before committing:
- The masked taps must be **structurally zero and stay zero** — mask the gradient, or the
  optimizer populates them and you have silently trained a dense W=15 conv (a different,
  *stronger* model — a confound, not a lucky break).
- **Numerically validate `fla` Triton at W=15 against `F.conv1d`,** fwd *and* bwd. Arbitrary-W is
  inferred from source, not documented or upstream-tested.
- Watch for register spills — the tap loop unrolls 15x via `tl.static_range`.

**Experimental-design consequence, worth escalating.** A **single dense W=15** conv is strictly
more expressive than the 4-branch masked version at nearly identical cost. If the multi-branch
structure's only justification is multi-scale receptive field, **the honest baseline to beat is
dense W=15, not W=3.** A router over four masked branches must earn its keep against that. This
falls out of the kernel analysis but is an architecture-claim issue.

**The router is a separate, unfused cost — and at batch=1 it is catastrophic.** The
token-dependent softmax router adds per conv layer: a `d -> 4` projection (trivial), a softmax
over 4 (trivial), and a **weighted sum of four `[B,T,d]` tensors** (not trivial — reads 4 full
activation tensors, writes 1). Unfused that is 5 HBM round-trips on top of 4 conv calls. At
batch=1 decode it is also **4 extra launches per conv layer x 10 conv layers = 40 extra launches
per token**; at ~5–10 µs each that is **200–400 µs/token of pure launch overhead** **[CALC]** —
compare a *whole* weight-limited decode step of ~1.5–2.6 ms (§2.5). CUDA graphs (§3) are
therefore **not optional** for proposal 3; they are the only thing that makes it measurable at
batch=1. Better still, fuse all four branches into one unrolled-shift expression and let Inductor
emit a single kernel, with the router weights as four more pointwise multiplicands — the same
trick as §1.6, again avoiding a hand-written kernel.

**If you must write Triton anyway.** Adding dilation to a copy of `fla`'s
`causal_conv1d_fwd_kernel` is a small offset change (`o_x = o_t + i_w * DILATION`) plus
initial-state indexing. **The forward is easy; the backward is the real work** — `fla` ships a
hand-written `causal_conv1d_bwd_kernel` with three boundary-handling code paths, plus
`causal_conv1d_update` (decode) and `causal_conv1d_update_states` (state materialization).
Reimplementing all of that *correctly*, with varlen and gradient checks, is realistically
**2–4 person-weeks** for a strong grad student — spent on infrastructure, not the research
question. **[INFERENCE]** Given both reformulations above avoid it, a custom Triton dilated conv
should be **out of scope**.

### 1.8 Conclusion for §1

**(a) Plain k=3 LIV — fully served off the shelf. No kernel work.** `fla`'s Triton backend gives
fwd + bwd + decode-update + conv state + varlen, with `activation=None` supported (and default).
Two required one-line actions: **pass `activation=None` explicitly** (wrappers default to
`'silu'` — §1.1) and **thread `initial_state`/`output_final_state` through the OLMo-core wrapper**
so decode is possible at all (§1.4). Optionally try the unrolled-shift + `torch.compile` form
(§1.6) to fuse away the gate multiply's HBM round-trip.

**(b) Dilated multi-branch LIV — nothing off the shelf supports dilation, but you do not need
it.** Both `causal-conv1d` and `fla` are dilation-free [VERIFIED]. Two reformulations give the
architecture with **no new kernel**: **masked dense W=15** through `fla`'s arbitrary-W Triton
kernel (recommended — keeps the tested backward and decode state; free in bandwidth terms), or
**unrolled shifted multiplies under `torch.compile`** (expresses arbitrary dilations, may fuse the
whole block). Writing a real Triton dilated depthwise causal conv with a correct backward is
**2–4 person-weeks** and is **not justified**.

---

## 2. Fused low-rank gates — roofline math

### 2.1 The question

Proposal 1 replaces two `d x d` gate projections with `d -> r -> d` pairs (`r=128`). Value and
out projections stay full width. Is fusing `[d->r matmul, r->d matmul, elementwise multiply,
depthwise conv]` realistic, and does the weight-byte saving survive to become a latency saving?

### 2.2 Weight bytes and FLOPs — every step [CALC]

Assumptions stated: bf16 (2 B/param), no biases, the **two gate projections only** (`B` and `C`
paths; the `x`/value path and `out_proj` stay dense), 10 conv layers per model.

**Parameter counts, d=2048, r=128:**

```
dense gates      = 2 * (d * d)         = 2 * 2048 * 2048        = 8,388,608 params
factorized gates = 2 * (d*r + r*d)     = 2 * (2048*128 + 128*2048)
                                       = 2 * (262,144 + 262,144) = 1,048,576 params
```

**Bytes (bf16):**

```
dense      = 8,388,608 * 2 =  16,777,216 B = 16.777 MB   per conv layer
factorized = 1,048,576 * 2 =   2,097,152 B =  2.097 MB   per conv layer
```

**The reduction factor — exactly 8x, and here is why in closed form:**

```
ratio = 2*d*d / (2*(d*r + r*d)) = d*d / (2*d*r) = d / (2r) = 2048 / 256 = 8.0x  ✓
```

So the promised ~8x is exact, and it is **`d/(2r)`** — worth noting because it means the factor
**halves when d halves**: at geometry (a), `d/(2r) = 1024/256 = 4.0x`, not 8x. **The 8x figure is
specific to the 1.2 B model.**

**Per-layer and whole-model saving:**

```
saving/layer     = 16.777 - 2.097 = 14.680 MB
x 10 conv layers = 146.80 MB
```

| | (b) d=2048 | (a) d=1024 |
|---|---|---|
| gate reduction factor `d/(2r)` | **8.0x** | **4.0x** |
| saving per conv layer | 14.680 MB | 3.146 MB |
| saving x 10 layers | **146.80 MB** | **31.46 MB** |
| model weight read/token | 2340.7 MB | 709.0 MB |
| **saving as share of model weight read** | **6.27 %** | **4.44 %** |
| new weight read/token | 2193.9 MB | 677.5 MB |
| **roofline decode speedup ceiling** | **1.0669x (6.3 %)** | **1.0464x (4.4 %)** |

**This is the headline result of §2: an 8x reduction in gate weights buys at most a 6.3 % decode
speedup**, because the gates are only 6.27 % of the model's weight bytes. The 8x number is real
but it applies to a small slice. Report both figures together or the 8x is misleading.

### 2.3 (i) Decode, batch=1, seqlen=1 — memory-bound [CALC]

At batch=1, T=1, each projection is a **GEMV**: one pass over the weight matrix, `2*P` FLOPs for
`P` params.

| d=2048 | FLOPs | weight bytes | arithmetic intensity |
|---|---|---|---|
| dense gates | `2 * 8,388,608` = 16.78 MFLOP | 16.777 MB | **1.00 FLOP/B** |
| factorized gates | `2 * 1,048,576` = 2.10 MFLOP | 2.097 MB | **1.00 FLOP/B** |

**Both are at AI = 1.00 FLOP/B**, versus an A100 bf16 ridge point of ~200 FLOP/B — i.e. **200x
below it.** This is the defining fact of batch=1 decode: *every* projection is a pure
bandwidth-streaming operation, cost ≈ weight bytes read, and FLOPs are irrelevant. Factorization
does not change the *intensity* at all; it only reduces the *bytes*. Which is exactly why the
saving is real — and exactly why it is capped at 6.27 %.

### 2.4 (ii) Prefill, seqlen=4096 — compute-bound [CALC]

At T=4096 the same weights are reused across 4096 tokens, so weight bytes are amortized and
FLOPs scale with T:

| d=2048, T=4096 | FLOPs | weight bytes | arithmetic intensity |
|---|---|---|---|
| dense gates | `2*4096*8,388,608` = **68.72 GFLOP** | 16.777 MB | **4096 FLOP/B** |
| factorized gates | `2*4096*1,048,576` = **8.59 GFLOP** | 2.097 MB | **4096 FLOP/B** |

AI = 4096 FLOP/B ≫ 200 FLOP/B ridge point → **firmly compute-bound.** Here the **8x FLOP
reduction is the operative saving** (68.72 → 8.59 GFLOP for the gate work), and weight bytes are
irrelevant. Note the *activation* traffic is unchanged — the `[B,T,d]` and `[B,T,r]` intermediates
still move — so the realized prefill speedup will be well under 8x on the gate work, and the gate
work is again only a slice of total prefill.

**So proposal 1 has two different stories and they must not be conflated:**
- **Decode:** saves *bytes*; ceiling **6.27 %** end-to-end (d=2048) / **4.44 %** (d=1024).
- **Prefill:** saves *FLOPs* 8x on the gate slice; end-to-end effect diluted by everything else.

### 2.5 Kernel-launch overhead and the breakeven condition [CALC]

Factorizing turns 1 GEMV into 2 per gate ⇒ **2 extra launches per conv layer**
⇒ `2 * 10 = 20 extra kernel launches per decode token`.

Time saved by the byte reduction, at several achievable bandwidths (d=2048):

| achievable BW | saved/layer | **saved/token (x10)** |
|---|---|---|
| 1000 GB/s | 14.68 µs | **146.8 µs** |
| 1300 GB/s (A100 realistic) | 11.29 µs | **112.9 µs** |
| 1555 GB/s (A100 peak) | 9.44 µs | **94.4 µs** |
| 3350 GB/s (H100 SXM peak) | 4.38 µs | **43.8 µs** |

Cost of the extra launches:

| per-launch cost | extra overhead/token |
|---|---|
| 2 µs | 40 µs |
| 5 µs | 100 µs |
| 10 µs | **200 µs** |

**Breakeven per-launch cost (overhead == saving):**

```
breakeven = saved_per_token / 20 extra launches
  at 1555 GB/s: 94.4 µs / 20 = 4.72 µs per launch
  at 3350 GB/s: 43.8 µs / 20 = 2.19 µs per launch
```

> ### ⚠️ THE DECISIVE ANSWER TO YOUR QUESTION
> **Yes — at batch=1 decode WITHOUT CUDA graphs, the extra launches plausibly eat the entire
> saving, and on an H100 they eat more than all of it.**
>
> Your own stated launch-overhead range is **5–10 µs**. The breakeven is **4.72 µs** on an A100
> and **2.19 µs** on an H100. **Both are below the bottom of that range.** At 10 µs/launch on an
> A100 the overhead is 200 µs against a 94 µs saving — proposal 1 makes decode **~106 µs/token
> slower**, i.e. a *negative* result of roughly the same magnitude as the hoped-for win.
>
> The situation gets **worse on faster hardware**, which is counterintuitive but follows directly:
> the byte saving shrinks with more bandwidth while launch overhead is fixed.
>
> **Therefore: CUDA graphs are mandatory, not optional, for measuring proposal 1 at batch=1.**
> Capturing the decode step into a CUDA graph collapses per-launch CPU cost to near zero and
> restores the ~6.27 % roofline ceiling as the achievable target. **A batch=1 decode benchmark of
> proposal 1 without CUDA graphs is not a measurement of the architecture — it is a measurement of
> the Python/dispatch overhead, and it will report the wrong sign.** This single methodological
> point determines whether proposal 1 looks like a 6 % win or a 5 % regression.

Secondary mitigations, in order of preference: (1) CUDA graphs / `torch.compile(mode=
"reduce-overhead")`; (2) **concatenate the two gate `d->r` projections into one `d -> 2r` matmul**
and the two `r->d` into one — halving the extra launches from 20 to 10 and doubling the breakeven
to ~9.4 µs, which is a pure win and should be done regardless; (3) larger batch, which amortizes
launches but also destroys the memory-bound regime the claim depends on.

### 2.6 Is the four-stage fusion realistic?

**[INFERENCE, with the reasoning made explicit.]**

**`torch.compile`:** the two matmuls will almost certainly go to **cuBLAS extern kernels**, not
Triton templates, under default settings — and Inductor cannot fuse *into* an extern kernel. So
out of the box you get: `mm` → `mm` → fused-pointwise(`*`) → conv. **Epilogue** fusion (folding
the elementwise multiply onto the end of the second matmul) is the realistic best case and
requires the Triton mm template, i.e. `max_autotune`. **Fusing the two matmuls into one kernel is
not something Inductor does.**

**Why fusing the two GEMMs is genuinely hard, not just unimplemented:** the second matmul
(`r -> d`) needs the **complete** `r`-vector before it can produce any output. In a single kernel
with a grid over the output's `d` dimension, every block needs the whole intermediate — that is a
**grid-wide barrier**, and Triton has no clean grid barrier. The ways around it are a persistent
kernel with cooperative-groups sync, or a single-block kernel (which at batch=1 leaves the GPU
almost entirely idle — one block on 100+ SMs). At batch=1 the intermediate is only `r=128`
elements, so it *would* fit in registers/SRAM; the blocker is the synchronization, not capacity.

**CUTLASS — [VERIFIED], and it confirms both the opportunity and the fatal catch.**
`examples/13_two_tensor_op_fusion` does exactly this: *"the mainloops of the two GEMMs/Convs run
back to back in a single kernel"*, with the 1st GEMM's accumulator held in the register file as the
2nd GEMM's activation input, which *"saves a round trip to memory for the activation matrix."*
<https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion>

The documented constraints are:
- `thread_block_tile_N = problem_N` — *"ensures that each threadblock loads the entire
  weight/filter matrix in addition to its own input activation tile"*, so the 2nd op's input tile
  depends only on the 1st op's output tile and the work is *"fully block-resident."*
- `warp_tile_N = thread_block_tile_N` — makes it *"fully register-file-resident"*; this one **can
  be relaxed** by staging the intermediate in shared memory (hence the shipped `_rf` and `_shmem`
  variants for f16/s8 on sm75/sm80).
- *"the same number of threadblocks are used across 2 GEMMs"*, which fixes a common threadblock
  tile M.

So the enabling condition is precisely **that the intermediate be narrow** — which is *exactly* the
low-rank case, `r=128`. CUTLASS is the right tool in principle.

> **But the catch is decisive for this experiment.** Because the grid is shared across both GEMMs
> and tiled over **M**, at **M=1 (batch=1 decode) there is essentially one threadblock** — i.e.
> **one SM out of 100+ doing all the work**, with the other ~99 % of the GPU idle. That is
> catastrophically slower than two separate bandwidth-saturating GEMV kernels, which spread the
> weight streaming across all SMs. **CUTLASS B2B fusion is structurally the wrong shape for
> batch=1 decode.** And the round trip it saves is the *intermediate activation* — which at
> batch=1 is 128 elements (256 B). It saves 256 B of traffic while forfeiting ~99 % of the machine.

**Conclusion for §2.6: do not attempt the four-stage fusion.** The intermediate is 128 elements
(256 B at batch=1), so the HBM round-trip you would eliminate is **negligible**; the dominant
traffic is the weight matrices, which must be read regardless. Worse, the one production-grade tool
for it (CUTLASS B2B) is structurally unsuited to M=1. The fusion would be weeks of CUTLASS/Triton
work to remove a cost that is already ~zero, and would likely be *slower*. **The launch-count
problem (§2.5) is the real issue, and CUDA graphs solve it completely, for free, in one line.**
Concatenating the gate pairs into `d -> 2r` (mitigation 2) captures the remaining easy win.

### 2.7 Verdict on proposal 1

- The **8x gate weight reduction is real and exactly `d/(2r)`** — but 8x only at d=2048; it is
  **4x at d=1024**.
- End-to-end decode ceiling: **6.27 %** (1.2 B) / **4.44 %** (350 M). Real, small, and
  **measurable only with CUDA graphs**.
- Without CUDA graphs at batch=1, the **20 extra launches/token plausibly exceed the entire
  saving** (breakeven 4.72 µs vs a 5–10 µs launch cost) and the measurement will report the wrong
  sign.
- Prefill sees an **8x FLOP reduction on the gate slice**, diluted end-to-end.
- **Deep fusion is not worth it.** Concatenate the gate projections; use CUDA graphs; report the
  roofline ceiling alongside the measurement.

---

## 8. Sanity-check of the crossover claim from the opposite direction

### 8.1 The question, posed precisely

Given KV is only **6.63 %** (350 M) / **2.11 %** (1.2 B) of decode traffic at 4 K, and **36.22 %**
/ **14.68 %** at 32 K: **what is the minimum context length at which a mostly-LIV topology shows a
≥10 % end-to-end decode-latency advantage over an all-GQA control of the same parameter count?**

### 8.2 Setup and assumptions [CALC]

- Both models are **parameter-matched**, so both read the same weight bytes `W` per token. The
  *only* difference is KV traffic. (This is the correct comparison and it is stricter than
  comparing against a differently-sized model.)
- LIV topology: 6 attention layers ⇒ `KV_LIV = 12,288 B/token`.
- All-16-GQA control: 16 attention layers ⇒ `KV_GQA = 32,768 B/token`.
- **Saving per token of context:** `ΔKV = 32,768 − 12,288 = 20,480 B = 20 KiB`.
- Decode is bandwidth-bound (established: AI ≈ 1 FLOP/B, 200x below ridge), so
  **latency ∝ bytes moved**.

The control's traffic is the denominator, since we want the advantage *relative to the baseline
being beaten*:

```
saving_frac(T) = ΔKV * T / (W + KV_GQA * T)
```

Setting `saving_frac(T) = f` and solving:

```
ΔKV*T = f*W + f*KV_GQA*T
T * (ΔKV − f*KV_GQA) = f*W
T = f*W / (ΔKV − f*KV_GQA)
```

### 8.3 Results

**Geometry (a), 350 M, `W = 708.96e6` B:**

```
f = 0.10:  T = 0.10 * 708.96e6 / (20480 − 0.10*32768)
             = 70.896e6 / (20480 − 3276.8)
             = 70.896e6 / 17203.2
             = 4,121 tokens
```

**Geometry (b), 1.17 B, `W = 2340.68e6` B:**

```
f = 0.10:  T = 0.10 * 2340.68e6 / 17203.2 = 234.068e6 / 17203.2 = 13,606 tokens
```

Full curves:

| target advantage `f` | **(a) 350 M** | **(b) 1.17 B** |
|---|---|---|
| 5 % | **1,881** | 6,211 |
| **10 %** | **4,121** | **13,606** |
| 20 % | 10,182 | 33,615 |
| 30 % | 19,971 | 65,937 |
| 50 % | 86,543 | 285,728 |
| **ceiling (T→∞)** | **62.5 %** | **62.5 %** |

Advantage at fixed contexts:

| T | (a) 350 M | (b) 1.17 B |
|---|---|---|
| 512 | 1.44 % | 0.44 % |
| 1,024 | 2.82 % | 0.88 % |
| 2,048 | 5.40 % | 1.74 % |
| **4,096** | **9.95 %** | 3.39 % |
| 8,192 | 17.17 % | 6.43 % |
| **16,384** | **26.93 %** | **11.66 %** |
| 32,768 | 37.64 % | 19.65 % |
| 131,072 | 53.64 % | 40.45 % |

The hard ceiling is `ΔKV / KV_GQA = 20480/32768 = 62.5 %` — the topology removes 10 of 16
attention layers, so it can never save more than 10/16 of the KV traffic, and the weight floor
keeps it strictly below that.

### 8.4 What this bounds — and it is the most important number in this document

**The minimum context for a ≥10 % end-to-end decode advantage is T ≈ 4,121 tokens on the 350 M
model, and T ≈ 13,606 tokens on the 1.2 B model.**

Reading that plainly:

1. **At the 350 M scale, a 4 K training context lands almost exactly on the 10 % threshold
   (9.95 % at T=4096).** This is *just barely* a defensible claim — and it is uncomfortably
   marginal. A 10 % effect requires excellent methodology (CUDA graphs, locked clocks, p50 over
   many samples) to resolve at all, and any measurement noise or unfused overhead will swamp it.
   **8 K context gives 17.2 %, which is comfortably measurable.** If the budget allows a 8 K
   context at 350 M, take it — that single choice moves this claim from marginal to solid.
2. **At the 1.2 B scale, 4 K gives only 3.39 %.** That is **not measurable** as an end-to-end
   claim in a student setting. The 1.2 B model needs **13.6 K context** just to reach 10 %.
   → **This independently and quantitatively confirms the carried-in insight: use the 350 M
   model.** The factor is `13606/4121 = 3.30x`, matching the `2340.7/709.0 = 3.30x` weight-byte
   ratio exactly, as it must.
3. **This advantage belongs to the TOPOLOGY, not to any of the three proposals.** It is the
   conv-vs-attention layer ratio (10/16 of attention layers removed) doing the work. It is
   available for free from the LFM2-style architecture the experiment starts from. None of P1, P2,
   or P3 contributes to it. **Do not let this win be attributed to the proposals** — that would be
   the single most likely way for this experiment to overclaim.
4. Corollary for the affordable-training-context question you asked: **at a 2 K context nothing
   here is claimable (5.40 % at 350 M, 1.74 % at 1.2 B).** Below ~2 K the topology advantage is
   inside the noise floor and should be reported as "no measurable decode-latency difference,
   as predicted."

### 8.5 Consistency check against the carried-in numbers

The earlier partial file estimated "roughly T ≈ 4,200 tokens" for the 10 % threshold at 350 M.
My independent derivation gives **4,121**. Agreement to ~2 %, and the small difference is
attributable to rounding in the weight-byte figure. **The carried-in claim is confirmed.**

---

## 4. Memory accounting — report the five buckets separately

### 4.1 Why "peak memory" as a single number is not a result

Reporting one peak-memory figure conflates five quantities with completely different scaling
behaviour, and the proposals affect them differently. **Report these five separately or the
memory claim is uninterpretable:**

1. **Weight bytes** — fixed, `params x bytes/param`. Proposal 1 reduces this.
2. **KV cache bytes** — scales with `batch x context`. Proposal 2 reduces this (and *only* this).
3. **Conv state bytes** — scales with `batch`, not context. Proposal 3 increases this.
4. **Activation peak** — scales with `batch x context`; transient, allocator-dependent.
5. **Allocator overhead / fragmentation** — the gap between `allocated` and `reserved`. Not a
   property of the model at all.

### 4.2 Analytic cross-check targets [CALC]

Compute these first, then check the measured numbers against them. A measurement that disagrees
with the analytic value by more than a few percent means the measurement is wrong (or is silently
including something else).

| quantity | formula | (a) 350 M, d=1024 | (b) 1.2 B, d=2048 |
|---|---|---|---|
| weights bf16 (inference) | `P * 2` | **709.0 MB** | **2340.7 MB** |
| weights fp32 + AdamW m,v | `P*4 + P*8` | 1417.9 + 2835.8 MB | 4681.4 + 9362.7 MB |
| **conv state, k=3, bs=1** | `10 * d * k * 2` | **61.4 kB** | **122.9 kB** |
| **conv state, k=15, bs=1** | `10 * d * 15 * 2` | **307.2 kB** | **614.4 kB** |
| KV cache, T=4096, bs=1 | `T * 12288` | 50.3 MB | 50.3 MB |
| KV cache, T=32768, bs=1 | `T * 12288` | 402.7 MB | 402.7 MB |

Two observations that matter:

**(i) Conv state is negligible — and this kills a possible framing of proposal 3.** At k=3 the
conv state is **61–123 kB**, i.e. ~0.01 % of weights and ~0.2 % of a 4 K KV cache. Even the
masked-dense k=15 variant is **307–614 kB**. So proposal 3's 5x state increase is
**5x of nothing.** Do not present "conv state grows 5x" as a cost, and do not present the LIV
topology's small conv state as a memory *win* against KV either — at 4 K, KV (50.3 MB) is
**~400–800x** larger than conv state. **[CALC]** The recurrent-state story is not where the memory
argument lives.

**(ii) KV cache bytes are identical across the two geometries** (50.3 MB at 4 K for both) because
`hkv`, `hd`, and attention-layer count are shared. Same scale-invariance as §0.3. So the 350 M
model has a *proportionally much larger* KV footprint relative to its weights — 50.3/709.0 = 7.1 %
vs 50.3/2340.7 = 2.1 % — reconfirming it as the better KV testbed.

### 4.3 ⚠️ The activation peak is probably the LOGITS, not the transformer [CALC]

This is the most common way a memory measurement on a small model with a large vocabulary goes
wrong. With `vocab = 65536` and prefill `T = 4096`:

```
logits [T, vocab] bf16 = 4096 * 65536 * 2 = 536,870,912 B = 536.9 MB
logits [T, vocab] fp32 = 4096 * 65536 * 4 =              = 1073.7 MB
```

Compare the largest *internal* activation tensors:

| tensor | (a) 350 M | (b) 1.2 B |
|---|---|---|
| `[T, 3d]` (in_proj output) | 25.2 MB | 50.3 MB |
| `[T, ff]` (SwiGLU intermediate) | 37.7 MB | 67.1 MB |
| **`[T, vocab]` logits bf16** | **536.9 MB** | **536.9 MB** |
| **`[T, vocab]` logits fp32** | **1073.7 MB** | **1073.7 MB** |

**The fp32 logits tensor (1073.7 MB) is larger than the entire 350 M model's weights (709.0 MB),
and it is 21x larger than the biggest internal activation.** If you measure "peak activation
memory" during a prefill forward pass without being careful, **you will mostly be measuring the
logits**, which none of the three proposals affect at all. A change in the conv blocks would be
completely invisible underneath it.

**Mitigations:** measure activation peak with the LM head excluded (hook the last hidden state), or
compute logits in chunks, or subtract the analytically-known logits contribution. Whichever you
choose, **state it explicitly** — this single tensor can flip a memory conclusion.

### 4.4 `max_memory_allocated` vs `max_memory_reserved`

**[VERIFIED — PyTorch memory-management semantics]**
- `torch.cuda.memory_allocated()` / `max_memory_allocated()` — bytes in **live tensors**. This is
  the number that corresponds to your analytic model. **Report this as the primary figure.**
- `torch.cuda.memory_reserved()` / `max_memory_reserved()` — bytes the **caching allocator holds
  from the driver**, including free-but-cached blocks. Always ≥ allocated, and it **overstates
  real need** because the allocator does not return freed blocks to the driver by default.
- `torch.cuda.reset_peak_memory_stats()` — **must be called immediately before each measured
  region**, or the peak is contaminated by warmup, compilation, and autotuning allocations.
- `torch.cuda.memory_stats()` — the detailed dict (`allocated_bytes.all.peak`,
  `reserved_bytes.all.peak`, `num_alloc_retries`, `num_ooms`, and the `inactive_split_bytes` keys
  that reveal fragmentation).

Docs: <https://pytorch.org/docs/stable/notes/cuda.html#memory-management> ·
<https://pytorch.org/docs/stable/cuda.html#memory-management>

**Report `reserved` too, but as an operational footnote, not the result.** The `reserved − allocated`
gap is your fragmentation/caching overhead — bucket 5. It is a property of the allocator and the
allocation *order*, not of the architecture, so a difference in `reserved` between two model
variants is **not evidence about the architecture**.

### 4.5 Snapshot-based attribution — how to actually split the buckets

Analytic formulas tell you what the buckets *should* be; the snapshot tool tells you what they
*are*, with stack traces:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run prefill + a few decode steps ...
torch.cuda.memory._dump_snapshot("snap.pickle")
torch.cuda.memory._record_memory_history(enabled=None)   # stop
```

Drag `snap.pickle` onto <https://pytorch.org/memory_viz> for a flamegraph-style timeline with
allocation stack traces. This is how you confirm that the 537 MB block is the logits and not
something you care about. Background: the PyTorch "Understanding GPU Memory" blog series
(<https://pytorch.org/blog/understanding-gpu-memory-1/>).

**A clean bucket-isolation protocol [INFERENCE — this is my recommended procedure]:**

```
1. Fresh process. Load weights only.        -> read max_memory_allocated()  = BUCKET 1 (weights)
   Cross-check against P * bytes_per_param. Must match within ~1%.
2. reset_peak_memory_stats(). Allocate the KV cache + conv state explicitly (or run one prefill
   and subtract).                            -> delta = BUCKETS 2 + 3
   Cross-check against T*12288 and 10*d*k*2. The conv state should be ~0.2% of KV; if it is not,
   you have a bug.
3. reset_peak_memory_stats(). Run prefill.   -> peak delta = BUCKET 4 (activations)
   Subtract the analytic logits term, or exclude the LM head. STATE WHICH.
4. reserved - allocated at every step        -> BUCKET 5 (allocator overhead)
```

### 4.6 Pitfalls that will corrupt these numbers

1. **Allocator caching** — freed tensors stay in the pool, so a second measurement in the same
   process sees no allocation. **Measure each configuration in a fresh process**, or the second
   variant you test will look artificially cheap.
2. **Reserved overstating** — never headline `max_memory_reserved`. (§4.4)
3. **Fragmentation** — a variant with more, smaller tensors (e.g. proposal 1's extra `[*, r]`
   intermediates, proposal 3's four branch tensors) can raise `reserved` without raising
   `allocated`. That is an allocator artifact, **not** a memory regression. Setting
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` substantially reduces this class of noise and
   makes cross-variant comparison fairer. Consider it mandatory for this experiment.
4. **Transient peaks** — peak memory is set by one instant, often inside a fused op or an
   autograd boundary, and is sensitive to allocation *order*. Peak is therefore **noisier and less
   reproducible than steady-state**. Report steady-state per-bucket numbers as the primary result
   and peak as secondary.
5. **Contaminated baseline** — `torch.compile` compilation, cuBLAS workspace creation, and Triton
   autotuning all allocate. **Warm up fully, then `reset_peak_memory_stats()`, then measure.**
6. **`max_memory_allocated` is per-device** — pass the right device, and note it does not capture
   host/pinned memory or the CUDA context (~300–600 MB), which is why `nvidia-smi` always shows
   more than PyTorch reports. Do not use `nvidia-smi` for model memory accounting.

---

## 7. Edge decode roofline — and the bound on what a conv-vs-attention change can achieve

### 7.1 Method [CALC]

Decode at batch=1 is bandwidth-bound (§2.3: AI ≈ 1 FLOP/B, ~200x below the ridge point), so:

```
tokens/s  <=  achievable_bandwidth / bytes_read_per_token
bytes_read_per_token = weight_bytes + context_length * KV_bytes_per_token
```

Assumptions, stated: (i) **theoretical peak** bandwidth is used below — real kernels achieve
roughly **70–85 %** of it, so **divide all figures by ~1.25–1.4** for a realistic expectation;
(ii) weights fully quantized at the stated precision, KV cache left at bf16 (the common
configuration); (iii) batch=1; (iv) **no compute or launch overhead** — these are pure upper
bounds and no implementation will reach them.

⚠️ **Bandwidth figures below are commonly-cited nominal values used to establish orders of
magnitude.** Per-device spec-sheet citations are consolidated in §7.5 pending verification;
treat the *shape* of the table as the finding, not the third significant figure.

### 7.2 Weight-limited decode ceiling (T→0), tokens/s [CALC]

| device | GB/s | 350M int4 | 350M int8 | 350M bf16 | 1.2B int4 | 1.2B int8 | 1.2B bf16 |
|---|---|---|---|---|---|---|---|
| Phone LPDDR5 (low end) | 51.2 | 289 | 144 | 72 | 87 | 44 | **22** |
| Phone LPDDR5X-8533 | 68.3 | 385 | 193 | 96 | 117 | 58 | **29** |
| Laptop DDR5-5600 dual | 89.6 | 506 | 253 | 126 | 153 | 77 | 38 |
| Apple M-series base | ~100 | 564 | 282 | 141 | 171 | 85 | 43 |
| Apple M4 base | ~120 | 677 | 339 | 169 | 205 | 103 | 51 |
| Apple M1/M2 Pro | ~200 | 1128 | 564 | 282 | 342 | 171 | 85 |
| Apple M3/M4 Pro | ~273 | 1540 | 770 | 385 | 467 | 233 | 117 |
| Apple M1/M2 Max | ~400 | 2257 | 1128 | 564 | 684 | 342 | 171 |
| Apple M3/M4 Max | ~546 | 3081 | 1540 | 770 | 933 | 467 | 233 |
| RTX 4090 | 1008 | 5687 | 2844 | 1422 | 1723 | 861 | 431 |
| A100-80GB | 1935 | 10917 | 5459 | 2729 | 3307 | 1653 | **827** |

Sanity read: a 1.2 B model at int4 on a phone gives ~87–117 tok/s upper bound, so ~60–95 tok/s
realistically — consistent with observed llama.cpp behaviour for ~1 B models on flagship phones.
The table is not obviously wrong.

### 7.3 THE BOUND: how much can a conv-vs-attention change possibly matter? [CALC]

This is the question that actually constrains the experiment's claims. Since latency ∝ bytes, the
**maximum possible** benefit of *any* change to the attention/conv ratio is the **share of decode
traffic that is KV**. Nothing about the change can exceed it. This ratio is **independent of
device bandwidth** — it is a pure byte ratio, so **one table covers every device in §7.2**.

**KV share of decode traffic (= absolute ceiling on any KV-side change):**

| context | 350M int4 | 350M int8 | 350M bf16 | 1.2B int4 | 1.2B int8 | 1.2B bf16 |
|---|---|---|---|---|---|---|
| 4 K | 22.1 % | 12.4 % | **6.6 %** | 7.9 % | 4.1 % | **2.1 %** |
| 32 K | 69.4 % | 53.2 % | **36.2 %** | 40.8 % | 25.6 % | **14.7 %** |
| 128 K | 90.1 % | 82.0 % | 69.4 % | 73.4 % | 57.9 % | 40.8 % |

(The bf16 columns reproduce §0.5 exactly — 6.6 %, 36.2 %, 2.1 %, 14.7 % — confirming the method.)

**Realistic advantage of the LIV topology vs a parameter-matched all-16-GQA control** (the §8
formula, `ΔKV*T / (W + KV_GQA*T)`, applied at each precision):

| model / precision | T=4 K | T=32 K | T=128 K | T for 10 % |
|---|---|---|---|---|
| 350 M int4 | **26.9 %** | 53.6 % | 60.0 % | **1,030** |
| 350 M int8 | **17.2 %** | 47.0 % | 57.7 % | **2,061** |
| 350 M bf16 | 9.9 % | 37.6 % | 53.6 % | 4,121 |
| 1.2 B int4 | 11.7 % | 40.5 % | 55.0 % | 3,402 |
| 1.2 B int8 | 6.4 % | 29.9 % | 49.1 % | 6,803 |
| 1.2 B bf16 | 3.4 % | 19.7 % | 40.5 % | 13,606 |

### 7.4 What this table actually tells you — three consequences

**(1) Quantization is the biggest amplifier of the topology's advantage, and it is free.**
Quantizing weights shrinks the denominator without touching KV, so the KV share rises sharply.
At 350 M / 4 K the topology advantage goes **9.9 % (bf16) → 17.2 % (int8) → 26.9 % (int4)** —
and the context needed for a 10 % win falls **4,121 → 2,061 → 1,030 tokens.** **[CALC]**

> **This is the single most useful actionable finding in §7.** If the experiment wants to
> demonstrate a decode-latency advantage at an affordable context length, **quantizing the weights
> to int8/int4 for the inference benchmark is a 2.7x lever on the effect size** — far cheaper than
> training at a longer context. It is also a legitimate and realistic deployment configuration, not
> a trick. A 350 M int4 model shows a **26.9 % advantage at 4 K** — comfortably measurable — where
> the bf16 version shows a marginal 9.9 %.

**Important caveat:** if the KV cache is *also* quantized to int8, the effect partly reverses,
because KV bytes halve too. At int4 weights + int8 KV, the 350 M advantage at 4 K drops from
26.9 % back to **17.2 %**, and T-for-10 % rises from 1,030 to 2,061. **[CALC]** So the protocol
must state the KV cache dtype explicitly — it moves the headline number by a factor of ~1.6.

**(2) The 1.2 B model at bf16 and 4 K is not measurable — 3.4 %.** This is the third independent
confirmation to use the 350 M model. **[CALC]**

**(3) The ceiling is real and it is not 100 %.** Even at 128 K and int4, the topology advantage
caps at ~60 % (asymptote `ΔKV/KV_GQA = 62.5 %`, §8.3). Removing 10 of 16 attention layers cannot
save more than 10/16 of KV traffic, and the weight floor keeps it below that. Any claim above
~60 % for this change is arithmetically impossible.

### 7.5 Honest caveats on the edge numbers

- These are **weight-streaming upper bounds** ignoring compute, launch overhead, prefill, and
  memory-system inefficiency. Real throughput is typically **60–80 %** of these figures.
- **Unified-memory bandwidth is shared.** On Apple Silicon and phone SoCs the published figure is
  the *system* bandwidth, shared with the OS, display, and other apps; a single compute cluster
  often cannot saturate it (documented for M1 Max, where one CPU cluster cannot reach the
  published 400 GB/s). Treat the Apple rows as generous.
- **Thermal throttling on phones is decisive and is not in this model.** Sustained decode on a
  phone throttles within tens of seconds. Any phone measurement must report sustained, not burst,
  throughput — and this alone makes phone latency claims hard to defend (§5).
- The `T→0` columns in §7.2 ignore KV entirely; at 32 K the 350 M bf16 model's real traffic is
  ~57 % higher than the weight-only figure, so divide accordingly.

---

## 3. Rigorous GPU latency measurement methodology

> **Framing.** §2.5 and §8 establish that the effects being chased are **6.3 %** (proposal 1
> roofline ceiling) and **~10 %** (topology at 4 K). Effects that small are *methodology-limited,
> not architecture-limited.* Everything in this section exists because a sloppy harness has a
> larger error bar than the effect being measured, and will report the wrong sign (§2.5).

### 3.1 The non-negotiable core protocol

**Order matters.** Each step below fixes a specific failure mode:

```
1. nvidia-smi -pm 1                     # persistence mode: keep driver/clocks initialized
2. nvidia-smi -lgc <f>,<f>              # LOCK graphics clocks (pick from --query-supported-clocks)
   nvidia-smi -lmc <f>,<f>              # lock memory clocks where supported
3. Warm up >= 20-50 iterations          # cuBLAS handles, Triton autotune, torch.compile, clock ramp
4. torch.cuda.synchronize()             # before starting the timer
5. Capture the decode step in a CUDA graph  <-- MANDATORY for this experiment (see 3.3)
6. Time N >= 100 replays with CUDA events (or a graph replay loop + one sync)
7. Report p50 AND p95 (not mean); report N; report the CV
8. nvidia-smi --query-gpu=clocks_throttle_reasons.active  # PROVE no throttling occurred
9. nvidia-smi -rgc / -rmc               # reset clocks when done
```

**Why locked clocks are not optional here.** An unlocked GPU boosts and then thermally/power
throttles. Boost-vs-throttled clock ranges routinely differ by **20–30 %** — which is **3–5x the
6.3 % effect** from proposal 1. If variant A is measured on a cool GPU and variant B on a hot one,
the result is pure thermal noise with an architecture label on it. Interleave the variants (ABABAB,
not AAABBB) **in addition** to locking clocks, so any residual drift cancels.

**Step 8 is the step everyone skips and it is the one that makes the result defensible.** Query
`clocks_throttle_reasons.active` after each measured block; if it is anything other than
`Not Active` / `GpuIdle`, discard the block. Without this you cannot claim the clocks were actually
locked, only that you asked for them to be.

### 3.2 CUDA events vs wall clock

- **CUDA events** (`torch.cuda.Event(enable_timing=True)`, `start.record()` / `end.record()`, then
  `torch.cuda.synchronize()` and `start.elapsed_time(end)` → **milliseconds**) measure GPU-side
  elapsed time on the stream. Preferred for *kernel/region* timing; they do not include Python or
  launch-queue time, which is a feature when you want device time and a **trap** when launch
  overhead is the thing you are studying.
- **Wall clock** (`time.perf_counter()` around a block, with `torch.cuda.synchronize()` **before
  starting and before stopping**) measures the end-to-end user-visible latency, **including** CPU
  dispatch and launch overhead.

> **For this experiment you need BOTH, and the gap between them is itself a result.** §2.5 shows
> that proposal 1's fate at batch=1 hinges entirely on launch overhead. CUDA events would hide the
> 20 extra launches; wall clock exposes them. **Report wall-clock end-to-end latency as the headline
> and CUDA-event device time as the diagnostic**; a large gap means you are overhead-bound and must
> use CUDA graphs before drawing any architectural conclusion.

`torch.utils.benchmark.Timer` handles warmup and synchronization correctly and reports robust
statistics — a good default for microbenchmarks (e.g. the conv op alone). It is less suited to a
full generate loop, where you want explicit control.
<https://pytorch.org/tutorials/recipes/recipes/benchmark.html>

### 3.3 CUDA graphs — mandatory, not optional, for this experiment

Established in §2.5: at batch=1 the **20 extra kernel launches/token** from proposal 1 cost
40–200 µs against a **94 µs** byte saving, so the breakeven per-launch cost (**4.72 µs** on A100,
**2.19 µs** on H100) sits *below* the typical 5–10 µs launch cost. **Without CUDA graphs, proposal 1
measures as a regression.** With them, per-launch CPU cost collapses to ~the cost of one graph
launch for the whole step, and the 6.27 % roofline ceiling becomes the achievable target.

Two routes:
- `torch.compile(model, mode="reduce-overhead")` — enables CUDA graphs automatically. Lowest effort;
  try this first.
- Manual `torch.cuda.CUDAGraph()` capture, or `torch.cuda.make_graphed_callables`.

**Capture requirements that shape the benchmark design:**
- **Static shapes.** The decode step must have fixed shapes — so a KV cache must be
  **pre-allocated at max context** and updated in place, not grown by concatenation. (This also
  makes memory accounting cleaner, §4.)
- **Static input/output addresses.** Copy new tokens *into* the same buffers; never rebind.
- **No CPU synchronization or `.item()`/`.cpu()` inside the captured region** — including sampling
  logic and stopping criteria. Move all of that outside the graph.
- Capture uses its **own memory pool**; warm up before capturing, and capture on a side stream.

**Consequence for §1.7 / proposal 3:** the router's 40 extra launches/token (200–400 µs) are also
entirely a graph-solvable cost. Proposal 3 is *unmeasurable* at batch=1 without graphs.

### 3.4 Statistics

- **Report p50 and p95, not the mean.** The mean is dominated by rare long tails (allocator
  growth, a driver hiccup, an interrupt) that are not properties of the architecture.
- **N ≥ 100 measured iterations** after warmup, and **report N**.
- **Report the coefficient of variation (CV).** [CALC] With per-iteration CV `c`, the standard error
  of the mean over `n` samples is `c/√n`; to resolve effect `e` at ~3σ you need `n > (3c/e)²`:

  | CV | effect 6.3 % | effect 10 % | effect 26.9 % |
  |---|---|---|---|
  | 1 % | n > 1 | n > 1 | n > 1 |
  | 2 % | n > 1 | n > 1 | n > 1 |
  | 5 % | n > 6 | n > 3 | n > 1 |

  **The encouraging conclusion:** with a properly locked, graph-captured harness (CV ~1–2 %), even
  the 6.3 % effect is resolvable with very few samples. **Sample count is not the binding
  constraint — harness quality is.** If you find you need thousands of iterations, your CV is too
  high and the fix is the harness (locking, graphs, interleaving), not more samples. Conversely, if
  CV is 5–10 % because clocks are unlocked, no amount of averaging rescues a 6 % claim, because the
  error is *systematic drift*, not zero-mean noise.
- **Do not silently discard outliers.** State the rule (e.g. "discarded iterations where
  `clocks_throttle_reasons.active != Not Active`", or a fixed trimmed-mean rule) and report how
  many were dropped.

### 3.5 Separating prefill from decode

Three approaches, with the pitfalls that matter:

| method | how | pitfall |
|---|---|---|
| **(a) Prefill-only** | Generate exactly 1 token from a `T`-token prompt; TTFT ≈ prefill | Includes one decode step and sampling; subtract or accept the small bias. |
| **(b) TTFT subtraction** | `TPOT = (total − TTFT) / (n_out − 1)` | Averages over a *growing* context, so it does not measure decode at a *fixed* context. Fine for serving-style reporting, wrong for a roofline check. |
| **(c) Explicit forward passes (recommended)** | Pre-fill a KV cache to exactly `T`, then time single-token forwards at that fixed `T` | Requires the pre-allocated static cache anyway (§3.3). |

> **Use (c) for every claim in this document.** All the arithmetic in §0, §2, §7, §8 is
> *per-token at a fixed context `T`*. Method (b) blends contexts and will not match the analytic
> prediction, making the analytic cross-check — your main defence against a broken harness —
> impossible. Method (c) is also the only one that lets you sweep `T` cleanly to plot the §8 curve,
> which is the single most convincing figure this experiment can produce.

**Metric definitions (serving convention):**
- **TTFT** (time to first token) — request arrival → first output token. Dominated by **prefill**.
- **TPOT** (time per output token) — mean of subsequent inter-token times. Dominated by **decode**.
- **ITL** (inter-token latency) — the per-token *distribution*; TPOT is roughly its mean. Report
  ITL p50/p95 rather than TPOT alone.
- MLPerf Inference defines server-scenario latency constraints per benchmark and requires
  token-by-token streaming for LLM tests, with TTFT and TPOT bounded separately.
  <https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>

**[UNKNOWN]** The exact numeric TTFT/TPOT targets per MLPerf round (e.g. the Llama-2-70B server
constraints) are version-specific and I did not verify a specific round's values. Cite the rules
document for the *definitions*, and do not quote target numbers without checking the round.

⚠️ **A note on MLPerf's applicability.** MLPerf targets are *serving SLOs* for large models at
throughput; this experiment is a single-model, batch=1, fixed-context latency study. **Borrow
MLPerf's definitions and its discipline (locked clocks, reported percentiles, immutable harness);
do not claim MLPerf compliance.** A student project cannot meet MLPerf's submission requirements
and should not imply otherwise.

### 3.6 Profiling tools and what each is for

| tool | use it for | do NOT use it for |
|---|---|---|
| `torch.profiler` | Which ops dominate; CPU-vs-GPU gaps; op-level shapes; Chrome trace | Precise latency of a whole step (profiling overhead) |
| `nsys` | Timeline: launch gaps, sync points, whether you are overhead-bound | Per-kernel counters |
| `ncu` | **Bytes moved**, occupancy, cache hit rates — the roofline verification | **Latency of any kind** (it serializes kernels) |
| `torch.utils.flop_counter` | Analytic FLOP cross-check | Anything about latency (§1.5) |

**`torch.profiler`** — use the `schedule(wait=, warmup=, active=)` API so you profile only steady
state, `record_shapes=True` to distinguish prefill from decode kernels, `profile_memory=True` for
§4, then `key_averages().table(sort_by="cuda_time_total")` and `export_chrome_trace()`.
The **first thing to check** is the gap between total CPU time and total CUDA time — if CPU ≫ GPU
you are launch-bound and §2.5 applies.

**`nsys`** — annotate regions and capture only the steady-state window:

```bash
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi -o decode_prof \
  python decode_bench.py
```
with `torch.cuda.nvtx.range_push("decode_step")` / `range_pop()` in the loop and
`torch.cuda.profiler.start()/stop()` bracketing the measured region. **The specific thing to look
for: whitespace between kernels on the CUDA timeline.** Visible gaps at batch=1 = launch-bound =
apply CUDA graphs before believing any architectural result.

**`ncu`** — this is how you *verify the roofline claims* rather than assert them. The verified
invocation and metric names (carried in from prior verification against Nsight Compute
13.3 / v2026.2.1 docs and NVIDIA-shipped `.section` files):

```bash
ncu --nvtx --nvtx-include "kv_read/" \
    --replay-mode application --cache-control none --clock-control none \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_hit_rate.pct,gpu__time_duration.sum \
    --csv --print-summary per-nvtx python decode_bench.py
```

**Confirmed metric names** [VERIFIED, carried in]: `dram__bytes_read.sum`,
`dram__bytes_write.sum`, `dram__sectors_read.sum`, `dram__sectors_write.sum`,
`lts__t_sectors_srcunit_tex_op_read.sum`, `lts__t_sector_hit_rate.pct` (**singular `sector`** — the
plural silently fails), `l1tex__t_sector_hit_rate.pct`, `gpu__time_duration.sum` (units **ns**).
For occupancy add `sm__warps_active.avg.pct_of_peak_sustained_active`, and for the memory-vs-compute
split `sm__throughput.avg.pct_of_peak_sustained_elapsed` alongside
`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` — on a memory-bound op the DRAM figure
should be high and the SM figure low, which is the direct empirical confirmation of §1.5 and §2.3.

**Five ncu traps that would invalidate the measurement** [VERIFIED, carried in]:
1. **`--cache-control` defaults to `all`**, flushing all GPU caches before every replay iteration →
   *cold-cache* traffic that **overstates DRAM reads**, which could fake or erase an L2-reuse
   effect. Use `--replay-mode application --cache-control none`; lock clocks externally instead.
2. **Collect hit rates and byte counts in separate runs** — with `--cache-control none` and
   multi-pass collection, ratio metrics can be wrong when numerator and denominator land in
   different passes.
3. **Prefer `dram__bytes_read.sum + dram__bytes_write.sum` over `dram__bytes.sum`** — NVIDIA's
   shipped `MemoryWorkloadAnalysis.section` does not gate the latter for all recent compute
   capabilities.
4. **Do not hard-code 32 B for `dram__sectors_*` on HBM.** Ground truth is
   `dram__bytes_read.sum ÷ dram__sectors_read.sum` measured on your GPU; using the byte metrics
   avoids the assumption entirely.
5. **Never read latency from an ncu run** — it serializes execution via per-device lock files. And
   the **Memory Chart shows instructions/requests, not bytes**; bytes live only in the tables
   (`--metrics group:memory__dram_table`). Do not screenshot the chart and call it traffic.

Also: `--metrics` without `--set` collects **no set** (`basic` is the default only when none of
`--set`/`--section`/`--metrics` is given). Working nvprof→ncu metric mapping table:
<https://archive.docs.nvidia.com/nsight-compute/2024.1/NsightComputeCli/index.html#metric-comparison>

### 3.7 The measurement that is a genuine contribution

**[VERIFIED reasoning, carried in and endorsed.]** The CLA paper *asserts analytically* that
cross-layer sharing does not reduce read bandwidth and **never measured it**. With the ncu recipe
above you can show, directly:

```
dram__bytes_write.sum   -> HALVES with CLA        (fewer distinct KV banks written)
dram__bytes_read.sum    -> stays FLAT with CLA    (consumers re-read the shared bank)
```

That is a **new, cheap, publishable measurement** which converts proposal 2 from a predicted null
into a *demonstrated mechanism*. It is the best possible outcome for a proposal whose latency
effect is ~0 (§0.7). **Label it "capacity + write traffic", never "KV cache traffic."**

### 3.8 Analytic cross-check — the harness's own unit test

Every latency number in this document has a closed-form prediction. **Check them against each
other; a mismatch means the harness is broken, not that the theory is wrong.**

```
predicted_decode_ms = bytes_per_token / achievable_bandwidth
bytes_per_token     = weight_bytes + T * 12288
```

At 1.2 B / bf16 (§2.5): 1.505 ms/token at 1555 GB/s theoretical, 1.801 ms at a realistic
1300 GB/s → **~555–665 tok/s**. If the harness reports 200 tok/s, you are overhead-bound (go to
§3.3); if it reports 900 tok/s, the model is not reading all its weights (check for a dtype or
tied-embedding accounting error). **Confirm the achieved bandwidth with
`dram__bytes_read.sum ÷ gpu__time_duration.sum` and compare to the device's STREAM/babelstream
number** — this closes the loop between the analytic model and the hardware, and is what makes the
roofline claims defensible rather than assumed.

---

## 5. Edge / CPU reality — and a blunt verdict on scope

### 5.1 llama.cpp / GGUF: LFM2 is fully and natively supported [VERIFIED]

I read the llama.cpp source directly. **Stock LFM2 support is genuinely first-class**, which is
both good news and — as §5.2 shows — the source of the problem.

**Architecture enums** (`src/llama-arch.cpp`):
```cpp
{ LLM_ARCH_LFM2,             "lfm2"             },
{ LLM_ARCH_LFM2MOE,          "lfm2moe"          },
```
So there is an **LFM2 MoE variant too**, with its own model class
(`llama_model_lfm2` / `llama_model_lfm2moe`, `src/llama-model.cpp:278-281`).
<https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp>

**Tensor names** — exactly the three you expected:
```cpp
{ LLM_TENSOR_SHORTCONV_CONV,    "blk.%d.shortconv.conv" },
{ LLM_TENSOR_SHORTCONV_INPROJ,  "blk.%d.shortconv.in_proj" },
{ LLM_TENSOR_SHORTCONV_OUTPROJ, "blk.%d.shortconv.out_proj" },
```
plus a hyperparameter key `{ LLM_KV_SHORTCONV_L_CACHE, "%s.shortconv.l_cache" }` (the conv window
length) and a name-fix entry `{ LLM_TENSOR_OUTPUT_NORM_LFM2, "token_embd_norm" } // fix for wrong
tensor name`.

**Op mapping — this is the decisive line:**
```cpp
{LLM_TENSOR_SHORTCONV_CONV,    {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_CONV}},
{LLM_TENSOR_SHORTCONV_INPROJ,  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_SHORTCONV_OUTPROJ, {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
```

The conv uses **`GGML_OP_SSM_CONV`** — the Mamba selective-scan conv op — **not** a generic conv.

**Graph construction** (`src/models/lfm2.cpp`, `build_shortconv_block`) matches your architecture
description **exactly**, which is a strong independent confirmation the spec is right:

```cpp
auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
constexpr auto n_chunks = 3;                       // in_proj d -> 3d, chunked 3 ways
auto * b = ggml_view_3d(..., 0 * chunk_size ...);   // B
auto * c = ggml_view_3d(..., 1 * chunk_size ...);   // C
auto * x = ggml_view_3d(..., 2 * chunk_size ...);   // x
auto * bx = ggml_transpose(ctx0, ggml_mul(ctx0, b, x));      // y = B * x
...
auto * conv_out = ggml_ssm_conv(ctx0, bx, conv_kernel);      // z = DWConv_causal(y)
auto * y = ggml_mul(ctx0, c, conv_out);                      // C * z
y = build_lora_mm(model.layers[il].shortconv.out_proj, y);   // out_proj(C * z)
```

**Note: no activation anywhere in that path** — confirming the "NO activation" spec from a second,
independent implementation. And note the `(B, C, x)` chunk *order* is B=0, C=1, x=2, which matters
if you write a GGUF converter.

**Conv state during decode — a rolling window via concat + view + copy** [VERIFIED]:
```cpp
const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;     // state is l_cache - 1 wide
...
bx = ggml_concat(ctx0, conv, bx, 0);                          // prepend saved state
// last d_conv columns is a new conv state
auto * new_conv = ggml_view_3d(ctx0, bx, conv->ne[0], ...);
ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_conv, ggml_view_1d(ctx0, conv_state, ...)));
```
So the state is `l_cache - 1` wide (**the Dao-AILab `W-1` convention, not `fla`'s `W`** — a third
convention to keep straight, cf. §1.1). It is managed through llama.cpp's **hybrid recurrent
memory** (`mem_hybrid_ctx`, `get_recr()`, `build_rs`), i.e. LFM2 is handled as a genuine
**hybrid recurrent + attention** model with separate recurrent and attention memory pools. There is
also a non-causal branch (symmetric padding for a centered window), so the implementation is more
general than pure decoding.

### 5.2 The critical constraint: `ggml_ssm_conv` has NO dilation parameter [VERIFIED]

From `ggml/include/ggml.h`:
```c
GGML_API struct ggml_tensor * ggml_ssm_conv(
        struct ggml_context * ctx,
        struct ggml_tensor  * sx,
        struct ggml_tensor  * c);
```

**Two tensors. No stride, no padding, no dilation, no groups.** It is a fixed-purpose fused op.

By contrast, ggml's *general* conv ops do take dilation:
```c
GGML_API struct ggml_tensor * ggml_conv_1d   (ctx, a, b, int s0, int p0, int d0);  // d0 = dilation
// depthwise
// TODO: this is very likely wrong for some cases! - needs more testing
GGML_API struct ggml_tensor * ggml_conv_1d_dw(ctx, a, b, int s0, int p0, int d0);
GGML_API struct ggml_tensor * ggml_conv_1d_dw_ph(ctx, a, b, int s0, int d0);
```

⚠️ **Note the upstream comment on the depthwise op: `// TODO: this is very likely wrong for some
cases! - needs more testing`.** That is ggml's own maintainers flagging `ggml_conv_1d_dw` as not
fully trusted. So the *only* dilation-capable depthwise 1-D conv in ggml is the one carrying a
correctness warning, and it is **not** the op LFM2 uses, is **not** on the fused SSM path, and
would not have the CUDA/Metal/NEON optimization that `ggml_ssm_conv` has.

**This makes dilation a fourth independent confirmation of the §1.7 finding:** Dao-AILab, `fla`,
Inductor, **and ggml** all lack a trustworthy dilated depthwise causal conv fast path.

### 5.3 Cost of running a NON-STANDARD variant, per runtime

This is the question that actually decides scope. **[INFERENCE — estimates are mine, reasoned from
the verified facts above; treat as order-of-magnitude, not precise.]**

| runtime | P1 low-rank gates | P2 cross-layer KV | P3 dilated multi-branch + router |
|---|---|---|---|
| **llama.cpp** | **Converter + graph edit.** Split `in_proj` into extra tensors, add 2 `ggml_mul_mat` calls in `build_shortconv_block`, edit `convert_hf_to_gguf.py`. All existing ops. **~1-2 weeks.** | **Graph + memory-layout edit.** Must make one layer's attention read another's cache — cuts across `llama-kv-cache` / hybrid memory. Invasive. **~3-6 weeks, upstream would resist.** | **New op or slow path.** `ggml_ssm_conv` cannot express it. Either write a dilated variant of `ggml_ssm_conv` for **every backend you want (CPU/NEON, CUDA, Metal, Vulkan)**, or fall back to the TODO-flagged `ggml_conv_1d_dw` and lose all fusion. Plus a router (4 branches + softmax + weighted sum) in the graph. **~6-12 weeks for multi-backend.** |
| **ExecuTorch** | Cheap — just more `aten.mm`. **~1 week** once a working export exists. | Stateful cache aliasing across layers is awkward in `torch.export`. **~3-5 weeks.** | Dilated depthwise conv is a standard ATen op, so it *exports*; performance on XNNPACK/QNN is the question. **~3-6 weeks.** |
| **ONNX Runtime** | Cheap — extra `MatMul`/`Gemm`. **~1 week.** | Requires expressing shared KV as graph I/O; doable but ugly. **~3-5 weeks.** | ONNX `Conv` supports `dilations` and `group`, so it exports. Mobile EP (NNAPI/CoreML) coverage for grouped+dilated Conv is the risk. **~3-6 weeks + real risk of CPU fallback.** |
| **Core ML** | Cheap. **~1 week.** | Needs `ct.StateType` stateful models (iOS 18 / coremltools 8+). **~3-5 weeks.** | Depthwise + dilated conv is expressible; **ANE support for dilated depthwise is the open question** — likely CPU/GPU fallback, losing the entire point. **~4-8 weeks, uncertain payoff.** |
| **MLX** | Cheap — `mx` matmuls. **~1 week.** | Moderate. **~2-4 weeks.** | `mx.conv1d` exposes `groups`/`dilation`, so expressible in Python quickly; kernel quality unknown. **~2-4 weeks.** |
| **Qualcomm QNN / Hexagon** | Cheap if a working graph exists. **~1-2 weeks.** | Hard. **~4-8 weeks.** | **Custom Hexagon op = specialist work.** Realistically **~8-16 weeks** and needs HVX experience. |
| **Liquid LEAP SDK** | **Effectively impossible.** A vendor SDK for *Liquid's own* checkpoints; not a general compiler for arbitrary custom architectures. | Impossible | Impossible |

**[UNKNOWN — flagged]** I did not verify: ExecuTorch's LFM2 support status, ONNX Runtime mobile EP
op-support tables for grouped+dilated Conv, whether ANE supports dilated depthwise conv, whether
`mlx-lm` ships an `lfm2.py`, whether LFM2 is in Qualcomm AI Hub, and LEAP's exact custom-model
policy. **The table's *relative ordering* is the robust part; the absolute weeks are estimates.**
A delegated research pass on these was in flight but did not return in time — treat §5.3 as
reasoned estimate, not verified fact, and do not cite the numbers as established.

### 5.4 ⚠️ BLUNT VERDICT: restrict to GPU-only measurement

**A student project cannot honestly claim edge latency wins for these proposals. Recommend
GPU-only as the defensible scope.** The reasoning, in order of force:

1. **The stock architecture is supported everywhere; your *variants* are supported nowhere.**
   This is the trap. It is tempting to reason "LFM2 runs in llama.cpp, therefore I can measure my
   LFM2 variant on a phone." **You cannot.** `GGML_OP_SSM_CONV` takes two tensors and has no
   dilation [VERIFIED §5.2]. P3 requires a new op on **every backend** you want numbers from.
2. **The cheapest honest edge result costs more than the research.** P1 is the cheapest variant
   (~1-2 weeks of converter + graph work in llama.cpp) and it buys a **6.27 % roofline ceiling**
   (§2.2) — and on CPU/mobile you cannot lock clocks, thermal throttling is uncontrolled, and
   run-to-run variance routinely exceeds 10 %. **You would spend weeks to produce a number whose
   error bar is larger than the effect.** That is not a measurement.
3. **Quantization interacts with the claim and multiplies the work.** Edge deployment means int4/int8
   (§7.4), so every variant needs a quantization recipe *and* a quality check to rule out that the
   low-rank gates or the router quantize worse. That is a second research project.
4. **Two of three proposals cannot show an edge latency win even in principle.** P2 is ~0 by
   construction [VERIFIED §0.7]. P3 *increases* op count and launch overhead (§1.7) and loses its
   fused kernel on every backend. Only P1 has a positive story, capped at 6.27 %.

**What to claim instead — and this is still a real contribution:**
- **Measure on GPU** with the §3 protocol. That is where locked clocks, CUDA graphs, and ncu
  counters make a 6 % effect defensible.
- **Report edge as an analytic roofline projection** (§7) — clearly labeled as a projection from
  published bandwidths, with the caveats in §7.5. This is honest, costs days not months, and is
  strictly more informative than one throttled phone number.
- **State the deployment relevance qualitatively**, grounded in the verified fact that LFM2's
  *topology* is edge-supported today (§5.1) while noting your variants would require the
  per-runtime work in §5.3. Frame that as **future work with a costed estimate** — which is a
  better contribution than a bad measurement.
- If you want *one* real edge datapoint, measure the **unmodified** LFM2-350M GGUF in llama.cpp to
  validate the §7.2 roofline. That is a day of work, requires zero custom ops, and calibrates the
  projection. **Do that; do not port the variants.**

---

## 6. Energy measurement

### 6.1 ⚠️ CITATION CORRECTION — arXiv 2002.05651 is not what you think it is

You asked me to cite **arXiv 2002.05651** for NVML power-measurement pitfalls. **[VERIFIED] It is
not about that.** The paper is:

> **"Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning"** —
> Peter Henderson, Jieru Hu, Joshua Romoff, Emma Brunskill, Dan Jurafsky, Joelle Pineau.
> arXiv:2002.05651 (cs.CY, cs.LG), submitted 31 Jan 2020, revised 29 Nov 2022.
> Published in **JMLR vol. 21, paper 20-312**. <https://arxiv.org/abs/2002.05651>

It presents `experiment-impact-tracker` — *"a simple interface for tracking realtime energy
consumption and carbon emissions"* plus standardized reporting appendices, and an energy-efficiency
leaderboard for RL. **It is the right citation for "you should report energy systematically and
here is a tool," and the wrong citation for "NVML power sampling is unreliable at short
timescales."** Cite it for the *reporting-standard* claim only.

**[UNKNOWN — do not fabricate]** I did **not** verify a specific paper documenting NVML's
sampling/averaging artifacts. I have a recollection of work showing NVML on V100/A100 exhibits a
power-reading averaging window that makes short-kernel energy attribution wrong, but **I could not
confirm a citation within this session and will not invent one.** If you need that claim, either
find the citation yourself or **drop the claim** — the recommendation in §6.4 does not depend on it.

**[UNKNOWN]** I also could not retrieve the NVML docs' detailed entries for
`nvmlDeviceGetPowerUsage` and `nvmlDeviceGetTotalEnergyConsumption` (the docs page truncated before
reaching them). The signatures are confirmed —
`nvmlDeviceGetPowerUsage(nvmlDevice_t, unsigned int* power)` and
`nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t, unsigned long long* energy)` — but **the units
(mW / mJ), the accuracy figure, the architecture requirement, and the averaging behaviour are
unverified here.** Read the untruncated page or PDF before quoting any of those.
<https://docs.nvidia.com/deploy/nvml-api/>

### 6.2 Why instantaneous power sampling is the wrong instrument here

**[INFERENCE, from first principles rather than from a contested citation]** The structural
argument does not need the missing paper:

- A decode step for the 1.2 B model takes **~1.5–2.6 ms** (§2.5). A *kernel* takes tens of µs.
- `nvidia-smi --query-gpu=power.draw` and `nvmlDeviceGetPowerUsage` return a **single scalar** that
  is some internal average over an unspecified window. Even under a generous assumption of a ~1 ms
  update period, that is the same order as an entire decode step and ~100x a single kernel.
- Therefore **per-kernel or per-layer energy attribution via power sampling is not possible**, and
  integrating a sparsely-sampled power signal over a short workload has an error bar far exceeding
  the **6.3 %** effect being chased.

**The correct instrument is a cumulative energy counter over a long workload**, not a power
sampler: `nvmlDeviceGetTotalEnergyConsumption` returns a monotonic accumulator, so taking a
**difference across a long window** cancels the sampling-window problem entirely. This is the one
methodological choice that makes GPU energy defensible at all.

### 6.3 Platform-by-platform caveats

| platform | mechanism | caveats |
|---|---|---|
| **NVIDIA GPU** | NVML: `nvmlDeviceGetPowerUsage` (instantaneous), **`nvmlDeviceGetTotalEnergyConsumption`** (cumulative accumulator) | Use the **cumulative** one. GPU-only: **excludes CPU, DRAM, PSU losses, and the whole host**. Consumer cards may not implement the energy counter. |
| **Intel/AMD CPU** | RAPL via `/sys/class/powercap/intel-rapl/`, or `perf stat -e power/energy-pkg/` | RAPL is a **model** on some generations, not a direct measurement. DRAM domain not always present. **Root/permission-gated** since the PLATYPUS side-channel mitigation (CVE-2020-8694) restricted unprivileged RAPL access; `perf` needs `perf_event_paranoid` lowered. |
| **macOS / Apple Silicon** | `sudo powermetrics --samplers cpu_power,gpu_power -i 1000` | Requires **root**. Reports CPU/GPU package power. **ANE power reporting is limited/absent** — so you cannot attribute NPU energy. Unified-memory power is not separable. |
| **NVIDIA Jetson** | `tegratstats`, INA3221 rails under `/sys/bus/i2c/.../in_power*` | **Not all boards have onboard power monitors** — several Orin-class dev kits lack per-rail INA sensors. Verify your specific board before promising numbers. |
| **External meter** | Wall-plug analyzer (e.g. Yokogawa WT310, as used in SPEC/MLPerf-style setups) | **The only measurement that captures true system energy** (host + GPU + PSU losses). MLPerf Power specifies a calibrated analyzer and a formal methodology. Requires hardware you probably do not have. <https://github.com/mlcommons/power-dev> |

**Higher-level tools:** `zeus` (<https://github.com/ml-energy/zeus>), CodeCarbon,
`experiment-impact-tracker` (the 2002.05651 tool), carbontracker, `nvitop`. **[UNKNOWN]** I did not
verify what specifically `zeus` does about NVML's limitations; it is the most actively maintained of
these and worth reading if you keep energy in scope.

⚠️ **Scope honesty:** NVML/RAPL/powermetrics all measure **components**, not the **system**. A
"J/token" figure from NVML alone omits the host CPU, DRAM, fans, and PSU inefficiency, which for a
small model at batch=1 can be a **large fraction** of true system energy (the GPU is not near full
utilization in a memory-bound batch=1 decode). **Any energy number must state exactly what it
includes.**

### 6.4 ⚠️ BLUNT RECOMMENDATION: cut energy from scope

**Cut it.** Reasons, in order of force:

1. **The effect is too small for the instrument.** Energy ≈ power x time. Proposal 1's ceiling is a
   **6.27 %** time reduction (§2.2) at essentially unchanged power (same op types, slightly fewer
   bytes). So the expected energy effect is **~6 %, with an instrument whose systematic error you
   cannot even characterize** (§6.1 unknowns). P2 is ~0. P3 likely *increases* energy via extra
   launches. **There is no proposal here with an energy story worth measuring.**
2. **Energy adds no information over latency in this experiment.** At fixed power, energy/token is
   a monotone restatement of latency/token. You would be reporting the same finding twice with a
   worse instrument. Energy only becomes independently informative when variants differ in
   *utilization* (e.g. sparsity, mixed precision) — not the case here.
3. **A defensible protocol costs real time.** Locked clocks, long fixed workloads, repeated trials,
   host-power accounting, and an explicit statement of exclusions — days of work, to produce a
   number that restates the latency result.
4. **It is the easiest place to accidentally overclaim,** and reviewers know it. A poorly-scoped
   "our model is 6 % more energy efficient" invites exactly the criticism the rest of this careful
   document is designed to avoid.

**If a supervisor insists on keeping it, the minimum defensible protocol:**

```
1. Lock clocks (nvidia-smi -pm 1; -lgc; -lmc)          # else power/freq confounds everything
2. Fixed, long workload: generate a FIXED token count taking >= 60 s (not a fixed wall time)
3. E = nvmlDeviceGetTotalEnergyConsumption(after) - (before)   # CUMULATIVE counter, never a
                                                              # power sampler
4. >= 5 repeats, interleave variants (ABABAB), report median and spread
5. Report J/token = E / tokens_generated
6. State explicitly: "GPU-only via NVML; EXCLUDES host CPU, DRAM, and PSU losses;
   measured with locked clocks; N=..., spread=..."
7. Verify clocks_throttle_reasons.active == Not Active for every trial
```

Never report per-kernel or per-layer energy (§6.2). Never report a single trial. Never omit the
exclusions sentence.

---
