# Mamba-3 b=2 versus b=3

Two questions, answered against code rather than recollection:

1. How close is this repository's `rotation_block_size=2` arm to published Mamba-3?
2. What exactly changes when it becomes `rotation_block_size=3`?

Everything below was checked against three sources:

- **The paper.** *Mamba-3: Improved Sequence Modeling using State Space Principles*,
  [arXiv:2603.15569](https://arxiv.org/abs/2603.15569) (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu).
  Sections 3.1–3.4 and Appendices C–D.
- **The reference implementation.** `mamba_ssm/modules/mamba3.py` at commit
  `e9594ce1c732d97440f0332fdc43170a2294dbfa`, which is the revision `pyproject.toml` pins under the
  `mamba3` extra. Where the paper is silent, this is the tiebreaker.
- **This repository**, at `src/olmo_core/nn/mamba3/`.

---

## Summary

There are **two** Mamba-3 architectures in this repository, and only one of them is faithful.

| | `Mamba3Config.mamba3_olmo3_370M` | the `mamba-b3` arm in `.edullm/model_arch_tests.py` |
| --- | --- | --- |
| Provenance | July, the original 370M ablation | August, written after a fidelity audit |
| Departures from published SISO beyond `b` | seven | three |
| Used by | `src/scripts/train/OLMo3/OLMo3-370M-mamba3.py` | `.edullm/train_core6_arm.py` |

The August arm is faithful modulo three deviations, all of which are forced or deliberate and all of
which are documented below. The July preset is not, and the seven-item gap it carries is large enough
that its b=2 numbers should not be read as "Mamba-3 b=2".

The b=2 → b=3 change itself is **one config field**. Everything else that differs between the two
arms is a mechanical consequence of that field: SO(3) needs three angles per block where SO(2) needs
one, and SO(3) does not commute, so the cumulative rotation stops being a `cumsum` and becomes a real
prefix product. The quaternion machinery exists only to make that prefix product cheap; it computes
the same rotation a 3×3 matrix scan would.

---

## Part 1 — Is the b=2 arm faithful Mamba-3?

### 1.1 The reference, stated precisely

The three innovations of §3:

- **Exponential-trapezoidal discretization** (§3.1, Prop. 1). A three-term recurrence
  `h_t = α_t h_{t-1} + β_t B_{t-1} x_{t-1} + γ_t B_t x_t` with `α_t = exp(Δ_t A_t)`,
  `β_t = (1-λ_t) Δ_t exp(Δ_t A_t)`, `γ_t = λ_t Δ_t`, and `λ_t ∈ [0,1]` data-dependent.
- **Complex-valued state via the RoPE trick** (§3.2, Props. 2–4). The transition is a block-diagonal
  matrix of 2×2 rotations `R(Δ_t θ_t[i])`, applied not to the state but as a *cumulative* product on
  `B` and `C`.
- **MIMO** (§3.3). Rank `R`; `R = 1` is SISO, which is the paper's default for like-for-like
  comparison.

The architecture (§3.4 and Appendix D): Llama-style **pre-norm**, expand factor **2**, `d_state`
**128**, head dimension **64**, RMSNorm on `B`/`C` ("BCNorm"), and learnable **head-specific,
channel-wise biases on `B` and `C` applied *after* BCNorm**. No short causal convolution. In hybrids
the post-gate RMSNorm is retained (Table 4). Remark 1: `A_t` is data-dependent by default.

Details the paper leaves out and the reference module fixes:

| | `mamba_ssm/modules/mamba3.py` |
| --- | --- |
| `A` parameterization | `A = -heavy_tail(dd_A)`, clamped to `≤ -1e-4`. **No static `A_log` parameter exists.** |
| Rotation angle | `cumsum(tanh(Angles) · π · Δ_t)`, per head via `Δ_t`, computed inside the kernel |
| `rope_fraction` | **0.5** by default — only half of `d_state` is rotated; `assert rope_fraction in [0.5, 1.0]` |
| Angle projection | one group-shared vector of `d_state·rope_fraction/2` angles, then broadcast to heads |
| `B_bias`, `C_bias` | shape `(nheads, mimo_rank, d_state)`, **initialized to 1** |
| `D` | per-head skip, initialized to 1, `_no_weight_decay` |
| `dt_bias` | inverse-softplus of `Uniform[0.001, 0.1]`, `_no_weight_decay` |
| Output norm (hybrid) | `RMSNormGated(norm_before_gate=True, group_size=headdim)` |

### 1.2 The audit

Verdicts are for the **August `mamba-b3` arm at b=2** — that is, the same mixer config with
`rotation_block_size=2`.

| # | Item | Reference | Here | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Exponential-trapezoidal recurrence | Prop. 1 coefficients | `α=exp(Δ A)`, `γ=λΔ`, `β=(1-λ)Δα`, `λ=sigmoid(lam_proj(x))` per head per token | **match** |
| 2 | Short causal convolution | absent | absent — no `Conv1d` anywhere in the mixer | **match** |
| 3 | Expand factor | 2 | `n_heads=32 × head_dim=64 = 2048 = 2 · d_model` | **match** |
| 4 | Head dimension | 64 | 64 | **match** |
| 5 | MIMO rank | 1 (SISO default) | `mimo_rank=1` | **match** |
| 6 | `n_groups` | 1 (MVA: `B`,`C` shared across heads) | 1 | **match** |
| 7 | BCNorm | RMSNorm on `B`,`C` before the rotation | `_rms_norm(Bm, bc_norm_b)` at `mixer.py:736` | **match** |
| 8 | `B`/`C` bias placement | after BCNorm, init 1 | `bc_bias=False` + `bc_bias_after_norm=True`, `torch.ones(...)` at `mixer.py:551` | **match** |
| 9 | `D` skip | per-head, init 1, no weight decay | `d_skip=True`, `mixer.py:526` | **match** |
| 10 | Output norm ordering | norm-before-gate, grouped by head dim | `norm_before_gate=True`; `_rms_norm` reduces the last axis, which is `head_dim` | **match** |
| 11 | Block norm placement | pre-norm | `_PRE_NORM_ARMS` selects the pre-norm block | **match** |
| 12 | `dt` | `softplus(dt_proj(x) + dt_bias)`, `dt_bias` from `Uniform[0.001,0.1]` | same | **match** |
| 13 | Weight-decay exemptions | `A_log`, `dt_bias`, `D` | `("*.A_log", "*.dt_bias", "*.D")` | **match** |
| 14 | Rotation angle formula | `tanh(θ) · π · Δ_t` | `torch.tanh(theta) * math.pi * rotation_dt`, `mixer.py:854` | **match** |
| 15 | Rotated fraction of state | `rope_fraction=0.5` | `rope_fraction=0.5` | **match** |
| 16 | Angle bound `theta_max` | none | `None` — the faithful path *forbids* it (`mixer.py:442`) | **match** |
| 17 | Token-dependent `A` | `A = -heavy_tail(A_proj(x))`, clamped `≤ -1e-4`, **no static baseline** | `A = -exp(A_log + a_proj(x))`, floored at `0.05` — keeps a per-head learned baseline | **deviation A** |
| 18 | Rotation timescale | per head (`Δ_t` is per head) | `group_mean` — `Δ_t` averaged over each group's heads | **deviation B** |
| 19 | `d_state` | 128 | 192 | **deviation C** |
| 20 | Rotation execution site | inside the Triton kernel | `B`/`C` preprocess in PyTorch, kernel run with `Angles=0` | implementation, math-neutral |
| 21 | Input projections | one packed `in_proj` | eight separate `nn.Linear` | implementation, math-neutral |
| 22 | Backbone | pure Mamba-3 stack, 2K context, Llama-3.1 tokenizer, FineWeb-Edu | 3:1 hybrid with RoPE attention, 4096 context, dolma2 tokenizer, Dolma | recipe, not architecture |

### 1.3 The three deviations, and why

**A — `A` keeps a static per-head baseline.** The reference has no `A_log`; `A` is entirely a
projection of the token. Here it is `-exp(A_log + a_proj(x))`, so `a_proj` modulates a learned
per-head baseline rather than producing `A` outright, and the shaping function is `exp` rather than
the reference's heavy-tail. With `a_proj` at zero this reduces exactly to the historical static
scalar `A`, which is why it was written that way: it made `dynamic_a` an additive feature that could
not disturb the arms that did not use it. The floor is also different — `0.05` here against the
reference's `1e-4` — which forbids the accumulator heads that a near-zero `A` produces. This is a
real difference in the decay parameterization and it is *not* forced by anything.

**B — the rotation timescale is group-shared, not per head.** The reference scales each head's
rotation by that head's `Δ_t`. Doing so makes the rotation head-specific, which forces `B`/`C` to be
broadcast to heads before the scan, which destroys GQA and multiplies the prefix product by
`n_heads` — 276.8 ms per layer against 14.7 ms group-shared on the rotation alone, at batch 2 and
sequence 4096. `group_mean` averages `Δ_t` over each group's heads, so the rotation still advances
with the timestep but at group granularity. A side effect: the post-BCNorm `B`/`C` bias becomes
group-shared rather than head-specific, since a head-specific additive bias cannot exist without
head-specific `B`/`C`.

Both forms were then run end to end on eight A100s, three seeds each, and the deviation is cheap
where it counts. Whole-model, the group-shared form is **1.375× the throughput at 68% of the peak
memory** (30,442 against 22,133 tok/s/device, 16.43 against 24.17 GiB), and it costs **0.0028 nats**
of held-out cross-entropy — 3.4176 against 3.4148, well inside a seed spread of 0.0126. That is the
measured basis for the default; the 18.8× above is the rotation kernel in isolation and overstates
what it does to a whole step.

**C — `d_state` is 192, not 128.** 128 is not divisible by 3, so it cannot express `b=3` at all
(`admissible_block_sizes(128) == (2, 4, 8)`). 192 is the smallest value admitting `b ∈ {2,3,4}`, which
is the entire reason it was chosen: it lets the baseline and the treatment share one state size so
that `rotation_block_size` really is the only field that moves. The cost is that the official kernel
zero-pads 192 up to 256, wasting a quarter of the state lanes. No power of two is divisible by three,
so this is unavoidable for any `b=3` configuration rather than a property of this particular number.

**All three are shared by both arms.** They are fidelity gaps against the paper. They are not
confounders for a b=2 versus b=3 comparison.

### 1.4 The July preset is a different animal

`Mamba3Config.mamba3_olmo3_370M` — what `OLMo3-370M-mamba3.py` builds today — departs from published
SISO in seven further ways: expand 1 instead of 2 (3.77M mixer parameters against the published
6.69M, with the difference reallocated into the FFN, where it cannot buy cross-token state capacity);
static `A` only; `B`/`C` bias before BCNorm and initialized to zero instead of after and to one; no
`D` skip; post-gate norm instead of norm-before-gate; the OLMo-2 reordered-norm block instead of
pre-norm; and a group-shared unbounded angle with a `theta_max` clamp instead of `tanh(θ)·π·Δ_t` over
half the state.

That preset is left untouched — other scripts and tests depend on it — and the new comparison uses a
new one.

### 1.5 What the difference was worth, measured

This is not a stylistic preference. The August eight-arm wave ran both versions of the arm on eight
A100s, three seeds each, 600M tokens, everything else held:

| `mamba-b3` | Held-out CE | Seed range | Steps gradient-clipped | Median grad norm |
| --- | --- | --- | --- | --- |
| Pre-faithful (the seven departures) | 4.5418 | 0.965 | 43–75% | 0.87–2.62 |
| Faithful (`b=3` the only deviation) | 3.4176 | 0.0126 | 0.0% | 0.287 |

The same architecture, the same recipe, the same data: **1.12 nats**, and last place of eight became
first by 0.02 nats over the best delta-rule arm. The arm was never capacity-limited; it was
optimization-limited by the departures, and the clipping rate is the mechanism — the seed that
clipped least trained best.

Two things follow for this comparison. A `b=2` number from the July preset would have been a
measurement of that pathology rather than of SO(2), which is why the ablation is built on the
faithful preset. And the faithful arm's seed spread is small enough to resolve a real `b` effect,
where the pre-faithful arm's was not.

One caveat on the "first of eight" claim, from the study's own preregistration: only two of the four
re-shelled arms were actually rerun, so the final ranked table mixes two shells and `mamba-b3`'s win
is not a clean contrast against the arms it beats. It is a clean contrast against its own earlier
self, which is the part that matters here.

---

## Part 2 — The exact change from b=2 to b=3

### 2.1 The config field

```python
Mamba3MixerConfig(..., rotation_block_size=3)   # was 2
```

Declared at `src/olmo_core/nn/mamba3/mixer.py:1273` and validated in `_validate_dims`
(`mixer.py:201`: `d_state` must be divisible by it). That is the entire user-facing change.
Everything in the rest of this section is what the code does *in response* to it.

### 2.2 What the field forces, mechanically

**The transition group.** `b=2` makes the block-diagonal transition a product of 2×2 rotations —
SO(2), which is abelian. `b=3` makes it SO(3), which is not, and which contains the alternating group
A₅. A₅ is non-solvable, so its word problem is NC¹-complete (Barrington). That is the point of the
change: a diagonal or abelian-block SSM is confined to TC⁰ and provably cannot track a non-solvable
group's state, regardless of depth or width.

**The angle count.** `angles_per_block = b(b-1)/2` (`mixer.py:360`): one angle for SO(2), three for
SO(3). With `d_state=192` and `rope_fraction=0.5`, both arms rotate exactly **96 of 192 state
channels**; b=2 does it as 48 planes needing 48 angles, b=3 as 32 blocks needing 96 angles. So
`theta_proj` widens from 48 to 96 outputs.

**The cumulative rotation.** This is the substantive one. Because `R(a)R(b) = R(a+b)` in SO(2), the
b=2 cumulative rotation is a **cumulative sum of scalars**:

```python
# src/olmo_core/nn/mamba3/mamba3_ssd_fast.py:1056
if block_size == 2:
    cumulative = torch.cumsum(angles.squeeze(-1) if angles.dim() == 5 else angles, dim=1)
    return _rotate_bc(B_in, cumulative), _rotate_bc(C_in, cumulative)
```

In SO(3) it is not, so `Q_t = R_t R_{t-1} ... R_1` has to be an actual **non-commutative prefix
product** over the sequence. The same two-line branch appears in all four scan backends, which is the
whole of the b=2/b=3 divergence in the scan:

| Backend | File | b=2 branch | b≥3 branch |
| --- | --- | --- | --- |
| fast / official-kernel | `mamba3_ssd_fast.py:1056` | `cumsum` | quaternion prefix product |
| official | `mamba3_ssd_official.py:106` | `cumsum` | matrix prefix product |
| chunked | `mamba3_ssd_chunked.py:222` | `cumsum` | matrix prefix product |
| reference | `mamba3_ssd_api.py:416` | `cumsum` | matrix prefix product |

Note what the table also says: **the scalar-decay scan itself never changes.** The rotation is applied
as a `B`/`C` preprocess and the SSD kernel is handed `Angles=0` either way
(`mamba3_ssd_official.py:26`), so b=3 reuses the official SISO Triton kernel unmodified. The kernel
sees nothing but a per-head scalar decay at both block sizes.

### 2.3 The quaternion implementation

A naive `b=3` carries a 3×3 matrix through the prefix scan: nine values per step, 27 multiplies and
18 adds per composition. A unit quaternion carries four, at 16 multiplies and 12 adds. That is the
whole motivation — it is a **representation of the same rotation**, not a different rotation.

Measured on B200 at shape `(32, 4096, 1, 64, 3)`, fp32, forward + backward:

| Scan | Time | Speedup |
| --- | --- | --- |
| chunked (chunk 32) | 206.5 ms | 1.00× |
| `associative_scan`, 9-leaf matrix | 100.1 ms | 2.06× |
| **quaternion, 4-leaf** | **29.2 ms** | **7.07×** |

The pieces, all in `src/olmo_core/nn/mamba3/mamba3_ssd_fast.py`:

- **`_angles_to_quaternion`** (line ~600) builds a unit quaternion per step straight from the three
  angles, never from a matrix. Axis convention `v = (-θ₃, θ₂, -θ₁)`, `q = (cos(φ/2), sin(φ/2)/φ · v)`
  in `(w,x,y,z)` order with `φ = ‖θ‖`. The convention was not derived — it was pinned by a test
  asserting `_quaternion_to_matrix(_angles_to_quaternion(θ)) == fast_block_rotations(θ, 3)`.
- **Small-angle handling.** `φ` is clamped before the `sqrt` (which has infinite derivative at zero)
  and both half-angle coefficients fall back to their Taylor series below `_SMALL_ANGLE_SQ = 1e-6`.
  This is the init regime — `theta_proj` starts at `std·0.1` — so getting it wrong is NaN gradients at
  step 1, not a slow drift.
- **`_quaternion_pointwise_combine`** (line 678) composes as `b ⊗ a`, newest-on-the-left, matching
  `Q_t = R_t R_{t-1} ... R_1`. Four separate scalar leaves rather than one width-4 tensor, because
  that is what `associative_scan`'s `pointwise` mode requires.
- **`_QuaternionPrefix`** (line 719) supplies a hand-derived linear-memory backward. This is not
  optional: the prototype autograd for `associative_scan` saves one full scan level per token and
  asks for 512 GiB at the production shape. The closed form used instead is
  `a_t = p_t/|p_t|² ⊗ Σ_{k≥t}(conj(p_k) ⊗ g_k)`, so one reverse `cumsum` replaces the O(T)-copy
  backward, and `dq_t = a_t ⊗ conj(p_{t-1})`.
- **`_quaternion_rotate`** (line 824) applies the rotation to 3-vectors directly, so the production
  `B`/`C` path never materializes a 3×3 matrix at all.

**The double cover does not need handling.** `q` and `−q` are the same rotation, and a Hamilton
product chain can walk to either sheet across a 4096-step scan. Every entry of `_quaternion_to_matrix`
is quadratic in `(w,x,y,z)`, so `R(q) = R(−q)` and the sign is annihilated on the way out. There is no
canonicalization, hemisphere flip, or sign-tracking logic anywhere in the path — deliberately.

The quaternion path is `b=3` only (`quaternion_cumulative_block_rotation` raises for anything else,
line 812). `b=4` would need the same treatment written out separately, which is one of several reasons
it was dropped; the others are that it buys no extra hardness, since `A₅ ⊂ SO(3) ⊆ SO(b)` already, and
that it has no closed-form exponential and falls back to `matrix_exp`.

### 2.4 Parameter and shape deltas

At the 370M geometry (`d_model=1024`, 16 layers of which 12 are Mamba, `d_state=192`,
`rope_fraction=0.5`, `n_groups=1`):

| | b=2 | b=3 |
| --- | --- | --- |
| `angles_per_block` | 1 | 3 |
| `n_rotation_blocks` | 96 | 64 |
| `n_rotated_blocks` | 48 | 32 |
| rotated state channels | **96** | **96** |
| `theta_proj` out features | 48 | 96 |
| `theta_proj` parameters per layer | 49,152 | 98,304 |
| extra parameters, whole model | — | **+589,824** |

`+589,824` is **0.16%** of a 371M non-embedding model. Under the Chinchilla-style loss scaling
`ΔCE ≈ −0.076 · Δln N`, that predicts about **1.2 × 10⁻⁴ nats** — three orders of magnitude below the
seed-to-seed spread observed on these arms. It is also irreducible: SO(3) needs three angles per block
and there is no version of the treatment that does not add them.

### 2.5 What does not change

Listed so it does not have to be re-derived: `d_state`, the number of rotated state channels, every
input and output projection except `theta_proj`, `A_log`, `a_proj`, `dt_proj`, `lam_proj`, `dt_bias`,
`D`, both BCNorm weights, both post-BCNorm biases, the output norm, the trapezoidal coefficients, the
scalar-decay SSD scan and its Triton kernel, the block pattern, the attention layers, the FFN, the
tokenizer, and the optimizer.

---

## Part 3 — Confounder ledger for a b=2 vs b=3 comparison

What has to be true for a difference in loss to be attributable to `b`:

| Factor | Status | How it is held |
| --- | --- | --- |
| Architecture, every field but `b` | controlled | one shared preset; the script asserts a single-field config diff |
| Parameter count | +0.16% on b=3 | irreducible; predicted effect ~1.2e-4 nats. `--param-match ffn` re-solves the FFN width to equalize exactly, as the paper itself does for MIMO |
| Learning rate | controlled | pinned to one value for both arms. **Do not let it be auto-derived** — the ladder formula reads the parameter count, so the 0.16% gap silently gives the arms different LRs |
| Data and data order | controlled | same mixture, same `--token-budget`, same data seed |
| Initialization seed | controlled | same seed; multiple seeds per arm when budget allows |
| Optimizer, schedule, precision, batch | controlled | one recipe, one code path |
| Rotation timescale | controlled | `group_mean` on both arms |
| Scan implementation | **differs by construction** | b=2 `cumsum` vs b=3 quaternion prefix product. Not a confounder: it is the treatment. The two compute the same mathematical object at their own block size, and both are exact |
| Throughput | **differs, and is an endpoint** | report it; do not read a loss-at-fixed-tokens comparison as a loss-at-fixed-wall-clock one |
| Kernel eligibility | controlled | both arms reach the same official SISO Triton kernel with `Angles=0` |

The table above is not a promise; most of it is checked. `OLMo3-370M-mamba3-b-ablation.py verify`
flattens both arms' configs and refuses the run if they differ in anything but the treatment, and
`test_the_two_arms_train_identically_apart_from_the_treatment` does the same against the whole
experiment config -- optimizer, schedule, seed, token budget and trainer included. On the current
default the entire diff is:

```
model.block.mamba3.sequence_mixer.rotation_block_size: 2 -> 3
trainer.callbacks.mamba3_sentinel.expected_rotation_block_size: 2 -> 3
```

the second being the first restated so the sentinel can refuse a run that trains as the other arm.

Two things that are **not** controlled, and should be stated in any write-up rather than papered
over:

- **`d_state=192` is off-reference for both arms.** The comparison is internally valid; it is not a
  reproduction of the paper's b=2 number.
- **The b=2 arm pays for the rotation outside the kernel too.** Published Mamba-3 gets its per-head
  rotation free inside the Triton kernel, which this repository cannot use (the kernel's angle
  parameterization cannot express an unconstrained group-shared Θ, `mamba3_ssd_official.py:32-44`).
  So the measured b=3-over-b=2 slowdown is the honest cost *within this implementation*, and an
  upper bound on the cost against a kernel-native b=2.

---

## Running it

The comparison lives in `src/scripts/train/OLMo3/OLMo3-370M-mamba3-b-ablation.py`, over the ladder's
own recipe: the dolma2 source mixture on S3, sequence length 4096, global batch 786,432 tokens.

```bash
# The wave, and the proof that it is a one-field difference. No network, no GPU.
python src/scripts/train/OLMo3/OLMo3-370M-mamba3-b-ablation.py plan --scale 190M

# One cell.
torchrun --standalone --nproc-per-node=8 \
    src/scripts/train/OLMo3/OLMo3-370M-mamba3-b-ablation.py train mamba3-190m-b2-r0 \
    --scale 190M --arm b2 --replicate 0 --save-folder s3://<bucket>/mamba3-190m-b2-r0
```

### Which scale

| | `--scale 370M` | `--scale 190M` |
| --- | --- | --- |
| Preset | `mamba3_faithful_olmo3_370M` | `mamba3_faithful_olmo3_190M` |
| Shell | `d_model` 1024, 16 layers | `d_model` 768, 12 layers |
| Non-embedding parameters | 371,445,632 | 190,293,192 |
| Token budget | 10.00B (1.35x Chinchilla, the ladder's) | 3.81B (20/parameter, Chinchilla) |
| Steps | 12,715 | 4,839 |
| `b=3` parameter surcharge | +589,824 (0.159%) | +331,776 (0.174%) |
| Tokens per parameter | 26.9 | 20.0 |
| Estimated `b=3` cell, eight A100s | ~12.5 h | ~3.7 h |

The architecture is identical between them — the preset test asserts it field by field — so this is
a budget decision, not a scientific one, and both fit inside the runtime bound of the eight-A100
node this organization provisions.

The runtime figures are rescaled from a real measurement rather than guessed: the August wave's
faithful `mamba-b3` rerun ran at **30,442 tok/s/device** on that shape at sequence 4096, with about
1.6 h of fixed overhead per cell. These arms compute 0.954× and 0.489× that cell's FLOPs per token.
`b=2` replaces a non-commutative prefix product with a `cumsum`, so it can only be faster than
`b=3`, which makes those the binding numbers. A throughput smoke is still the honest confirmation.

370M is the experiment as specified and it is affordable, so prefer it. 190M is the cheaper option —
about a seventh of the work — and it is the better one if you want more seeds for the same money, or
a second point to check that a result at one scale survives at another.

Two runs is the minimum experiment. `--replicates 3` is better and costs three times as much. The
faithful arm's measured seed spread is **0.0126 nats of range across three seeds**, which is tight
enough that three seeds would resolve a `b` effect down to a few hundredths of a nat. Do not size
this off the one-nat spread the *pre-faithful* arm showed: that was 43–75% of steps being
gradient-clipped, and the fidelity fixes took it to 0%.

`.edullm/run-b-ablation.yaml` is the platform spec. Run `edullm check --json` against it before
submitting and read `cost` and `approval_class` out of that, not out of this document.

## Provenance

The b=3 work was done 24–28 July 2026; the fidelity audit and the faithful rewrite followed on 7–11
August. Load-bearing commits on `edullm/mamba-comparison`: `19f1a6d` (Mamba-3, xLSTM, Mamba-PD
updates), `74b6015` (mamba-b3 fixes), `9cf41bc` (mamba-b3 speed fix), `0152ffd` (pd-ssm and mamba-pd
fixes).

Two claims that circulate in the chat logs and are **wrong**, recorded here so they stop being
repeated: quaternions were never found to be unstable — they were never tested for stability at the
time that was said, and they are the fast path, not a rejected one. What was actually rejected on
stability grounds is `b=4`, which is a property of that arm and not of the representation.

The term "recursive tracking" appears nowhere in the design history. The project's own vocabulary is
**state tracking**, TC⁰/NC¹, and the A₅ word problem.
