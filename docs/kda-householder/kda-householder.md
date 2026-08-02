# Householder delta-products with a per-channel forget gate

**A linear-attention recurrence combining KDA's per-channel decay with DeltaProduct's
R Householder factors: implementation, verification, and what the evidence supports.**

---

## Abstract

A linear-attention layer whose state transition is diagonal is limited in the group
structure it can track; replacing the single rank-one delta update per token with a
product of R such updates enlarges the reachable transitions, and is the motivation for
DeltaProduct. Kimi Delta Attention (KDA) supplies a *per-channel* forget gate but only
one delta factor per token; DeltaProduct supplies R factors but its available
implementations require a *per-head* scalar gate. We implement the combination — R
ordered delta factors applied after a single per-channel decay — as a fused Triton
forward and backward kernel in OLMo-core, and we measure what R buys.

On the S5 word problem raising R from 1 to 4 in a 3-layer, 1.0M–2.2M-parameter model
raises mean token accuracy by +55.7pp at evaluation length 128 (n=8 seeds, paired,
p=8.7e-7). The same change produces no detectable effect on Z/2, S3, or S4 at any of
seven evaluation lengths (28/28 cells not significant, minimum p=0.087), even though S3
and S4 are far harder than Z/2 in absolute terms — so the effect does not simply track
task difficulty. That interaction is significant at all seven lengths and survives Holm
correction over the whole 109-test family at six of them. Reparameterizing accuracy as an
*effective horizon* — the number of leading positions a model tracks before falling to
chance — shows the effect compactly: on S5, R=1 reaches 40.5 tokens against a training
range of 3–40, i.e. it does not extrapolate at all, while R=4 reaches 120.3 tokens
(Δ = +79.8, 95% CI [+64.0, +95.6]).

We deliberately do **not** claim that group *solvability* is the discriminator. S5 is the
only non-solvable group we tested and also the largest, so solvability, cardinality, and
DeltaProduct's `(R+1)`-permutation arity bound are perfectly confounded in our design. The
case that separates them is A5 — non-solvable, |A5| = 60 — which we did not run and which
DeltaProduct reports extrapolating at `n_h = 2`, like solvable S4 and unlike S5. That
result favours the arity account over the solvability account.

**The mechanism claim is supported; the utility claim is not.** R cannot be varied at
fixed parameter count, and the probe grid contains no purpose-built parameter-matched
control — R=4 has 2.21× the mixer parameters of R=1. The one strictly parameter-matched
comparison in the project is a 12-layer, 52–72M-non-embedding-parameter language model
trained on 1.04B tokens of FineWeb-Edu, with a third arm widening R=1 to match R=4's
parameter count. There, at matched capacity, **R=4 is significantly worse than
width-matched R=1**: +0.036 nats (95% CI [+0.020, +0.052], p=0.0056, n=4 paired seeds,
consistent in 4/4 seeds and at all four evaluation lengths), while costing 1.62× the
wall-clock time. Extra parameters help (−0.053 nats for width alone); spending them on R
helps less. So on natural text at this scale the mechanism is not what limits next-token
prediction, and the honest reading is not the "capacity" verdict of the project's own
pre-registered rule but something stronger: R actively underperforms the matched
alternative.

One further caution. The seven probe evaluation lengths are not seven independent
findings — per-seed effects have a two-block correlation structure worth ≈1.7–4.4
effective tests, so "significant at all seven lengths" is one result seen from several
angles.

The kernel is correct where it has been checked and fast enough to train with, but the
headline "406×" compares it to a deliberately naive reference we also wrote, not to a
production kernel, and no comparison against a production chunked kernel was ever made.
On novelty: DeltaProduct already reports that S5 requires four Householder factors, so
**the S5 finding is a replication** on a per-channel-gated variant; what is new here is
the per-channel-gated kernel at R>1, the joint depth × R grid, and the parameter-matched
language-model comparison.

---

## 1. What this document claims, and at which level

This project produced three kinds of evidence. They are not interchangeable, and the
main way a write-up like this can mislead is by letting one borrow credibility from
another. Throughout, results are tagged:

| Tag | Meaning | Where |
|---|---|---|
| **[V]** | Verification — the kernel computes the intended recurrence | [§4](#4-verification-what-is-and-is-not-established) |
| **[P]** | Probe — synthetic state-tracking tasks, 1.0–3.0M parameters | [§5](#5-probe-results), [§6](#6-the-effective-horizon-view) |
| **[LM]** | Language model — 52–72M non-embedding parameters, FineWeb-Edu | [§7](#7-the-language-model-result) |

A **[P]** result is evidence about a mechanism in a regime built to expose it. It is not
evidence that the mechanism pays off in a language model, and this document never treats
it as such. The **[LM]** evidence is the only natural-data, parameter-matched test here,
and it is the one that should govern whether to adopt R>1.

The short version of how they relate: **[V]** says the operator is implemented correctly
where it has been checked; **[P]** says R>1 confers a real and specific capability;
**[LM]** says that capability does not pay for its parameters at this scale. All three
can be true simultaneously, and this document argues they are.

---

## 2. The recurrence

KDA updates one state matrix per head with a per-channel (diagonal) forget gate and a
single delta-rule write per token. DeltaProduct replaces the single write with R ordered
writes, so the token-to-token transition becomes a product of R Householder-like
factors. The combination implemented here applies the decay once, then R writes:

```
S_{t,0} = Diag(alpha_t) . S_{t-1}
S_{t,j} = S_{t,j-1} + beta_{t,j} k_{t,j} ( v_{t,j}^T - k_{t,j}^T S_{t,j-1} ),   j = 1..R
S_t     = S_{t,R}
```

with the readout taken after the final factor. Each factor contributes an erase of the
form `(I - beta k k^T)`, whose eigenvalue along `k` is `1 - beta ||k||^2`. With
`beta = 2*sigmoid(.)` in `(0, 2)` and L2-normalized keys, that eigenvalue can reach
`-1`: the factor is a genuine reflection rather than a contraction. This is the property
the state-tracking theory turns on ([§8](#8-relation-to-prior-work)), and it is enabled
by `allow_neg_eigval=True` (`recurrent.py:1257-1259`).

Measured at initialization (d_model 512, 8 heads, R=2, 16,384 samples), the flag does
exactly what is claimed: with `allow_neg_eigval=True`, `beta ∈ [0.302, 1.750]`, so
`1 − beta` reaches −0.750 and reflections are reachable; with `False`,
`beta ∈ [0.151, 0.875]` — strictly below 1, so the eigenvalue never leaves `(0,1)` and the
mechanism is unavailable. **The code default is `False`** in both the config
(`recurrent.py:1480`) and the constructor (`:1102`), so the mechanism is opt-in. All 98
published probe runs and all 13 language-model runs pass `allow_neg_eigval=True`
explicitly, verified in the result-generating sources — so every result reported here was
produced with reflections available.

The same measurement substantiates a design decision. The backward recomputes the forward
rather than inverting it, because inversion requires `1/(||k||² − 1/beta)`, which with
normalized keys is exactly singular at `beta = 1`. That point is interior and genuinely
hit: **67 of 16,384 sampled betas fall within 1e-3 of 1.0** under
`allow_neg_eigval=True`. (Under `False` the singularity is unreachable — but so is the
mechanism.)

Two implementation notes that matter for interpreting the results:

- Keys are L2-normalized in the **module** (`recurrent.py:1300`), not in the operator.
  The kernel itself never normalizes. So the negative-eigenvalue guarantee is a property
  of the module-level composition, and the correct statement of the condition is about
  `beta*||k||^2`, not `beta` alone.
- `beta` is computed once for all R factors from a single projection
  (`recurrent.py:1257-1259`). The R factors therefore share a beta parameterization;
  per-factor beta regimes are not expressible through the existing flag.

**Availability.** The two capabilities exist separately in `flash-linear-attention` and
are mutually exclusive there. Verified directly against the installed `fla` 0.5.1 source:

| Operator | Gate requirement | R |
|---|---|---|
| `fla.ops.kda.chunk` | `assert g.shape == (B, T, HV, K)` — per-**channel** (`chunk.py:411`) | 1 only |
| `fla.ops.gated_delta_product.chunk` | `assert g.shape == (B, T, H)` — per-**head** scalar (`chunk.py:344`) | R > 1 |

The same per-head assertion appears in that module's `chunk_ref.py:36` and `naive.py:21`,
so it is a property of the formulation, not of one kernel. Obtaining a per-channel gate
together with R > 1 therefore required new kernel work.

This is a statement about **available implementations**, verified by inspection, and not a
claim of conceptual novelty — see [§8](#8-relation-to-prior-work).

Source: `src/olmo_core/nn/attention/kda_householder.py` (Triton forward and backward),
`kda_householder_torch.py` (differentiable reference), `recurrent.py`
(`KimiDeltaHouseholder` and config). All in commit `6b75c06`.

---

## 3. Experimental setup

**Probe tasks [P].** Group word problems: the input at each position selects a
generator, and the target at position *t* is the index of the accumulated product of the
first *t+1* generators. Verified properties of the four tasks used (audit script
`scripts/`-adjacent, output in `data/probe_task_spec.tsv`):

| Task | Group | \|G\| = out_vocab | Chance | Solvable | Verified |
|---|---|---|---|---|---|
| `parity` | Z/2 | 2 | 50% | yes | closure, associativity |
| `s3_words` | S3 | 6 | 16.67% | yes | equals full S3, contains odd generator |
| `s4_words` | S4 | 24 | 4.17% | yes | equals full S4, contains odd generator |
| `s5_words` | S5 | 120 | 0.833% | **no** | equals full S5, contains odd generator |

Each was checked to be the full symmetric group (not the alternating subgroup — an odd
generator is present in every case, so `s5_words` is genuinely S5 and not A5), closed,
associative, with all inverses present, and with realized label distributions close to
uniform. Labels are near-uniform at every evaluation length and there is no
position-dependent leakage beyond the first few tokens.

**Metric.** Mean token accuracy over **all** positions of the evaluation sequence. The
group-word tasks mark no position with `-100`, so all `64 x L` tokens enter the
denominator (verified: 0/131072 ignored). This matters for interpretation and is the
reason for [§6](#6-the-effective-horizon-view). Because the denominator is large, within-run
binomial noise is negligible (≈0.05pp at L=2048, p≈0.03), so **the study is seed-limited,
not sample-limited**: the operative noise floor is the between-seed standard deviation,
which is two to three orders of magnitude larger. Measured (`data/probe_noise.tsv`), it is
0.18pp for S5/R=1 at length 2048 and 3.0pp at length 128, and reaches 22.8pp for
S5/R=4 at 1 layer, length 128 — so per-cell precision varies enormously across the grid
and every comparison in this document is reported with its own interval rather than a
shared error bar.

**Training.** 2000 steps, sequence lengths drawn uniformly from **[3, 40]**, batch 64,
AdamW with one-cycle LR 1e-3, `d_model=256`, 4 heads x 64, `allow_neg_eigval=True`
passed explicitly, Triton backend. Evaluation at lengths 40, 64, 128, 256, 512, 1024,
2048 — so all but the first are **extrapolation**, by factors of 1.6× to 51×.

Evaluation data are drawn from a generator seeded `seed*100003 + length`, so at a fixed
seed **both arms see identical evaluation data** (the paired tests are legitimate) while
each length sees a different draw. Train/eval overlap at L=40 was checked directly:
0 collisions.

**Parameters.** Because R widens `w_k`, `w_v`, `w_b` and the k/v convolutions, R is
confounded with capacity by construction:

| | R=1 | R=2 | R=4 |
|---|---|---|---|
| Mixer params/layer | 332,356 | 466,500 | 734,788 |
| Mixer ratio vs R=1 | 1.000× | 1.404× | **2.211×** |
| Total (S5, 3 layers) | 1,029,324 | 1,431,756 | 2,236,620 |

All counts reproduce analytically from the code (`data/probe_arm_spec.tsv`). The
R-scaling is concentrated in `w_k` and `w_v` (+196,608 each per layer from R=1 to R=4).

---

## 4. Verification: what is and is not established

**[V]** The previous project record described a "six-level verification chain" with
results ranging from bit-exact to 7.1e-15. The chain exists and its individual
measurements were reproduced, but its *coverage* is narrower than the summary implied,
and the narrowing is documented in the implementation's own commit message.

The commit message of `6b75c06` states the limitation this way:

> The R=2 path has NO independent verification. The torch reference is a transcription
> of `probes/naive_kda_householder.py`, so those two are one oracle; the only external
> anchor (`fla.ops.kda.chunk_kda`) runs at R=1.

That is the right worry — the torch reference and the naive oracle are two transcriptions
of one derivation, so their 1e-15 agreement is a consistency check, not independent
verification — but it is **too pessimistic**, and re-measurement corrects it in the
project's favour. An external R>1 anchor does exist and does agree. When the per-channel
gate `g` is held constant along the key dimension, the recurrence reduces exactly to
`fla`'s `gated_delta_product`, and the docstring at `kda_householder.py:689-693` claims
agreement to float64 ulp. Re-run (`data/verification_chain.tsv`):

| Check | R | Result |
|---|---|---|
| vs `fla.ops.kda.naive.naive_recurrent_kda` | 1 | `max|diff| = 0.0` (bit-exact) |
| vs `fla...naive_recurrent_gated_delta_product`, fp64 | 2 | `1.78e-15` |
| vs `fla...naive_recurrent_gated_delta_product`, fp64 | 3 | `3.55e-15` |

The same comparison in float32 reports 2.9e-6 and 1.4e-6, which a 1e-10 threshold calls a
failure; one float32 ulp at these output magnitudes is ~1.8e-6, so the float32 "failure"
is round-off, not disagreement. **So R=2 and R=3 are externally anchored after all**, in
the constant-along-K slice of the gate's parameter space. What remains unverified
externally is the genuinely *per-channel* gate at R>1 — precisely the novel combination —
because no external implementation of it exists. That is a narrower and more accurate
statement than either "verified" or "no independent verification."

Levels 2–4 reproduce, with two of the recorded figures slightly optimistic:

| Level | Claimed | Re-measured | Note |
|---|---|---|---|
| 2 — fp64 `gradcheck` | PASS | **PASS** (R=1,2,3, varlen) | `gradgradcheck` also passes at R=1,2 on the torch backend; no tension with `once_differentiable`, which binds the Triton path only. But no `gradgradcheck` existed in the repo — it was written for this audit. |
| 3 — manual backward vs autograd | 3.6e-15 | **4.4e-15** | worst of six gradients |
| 4 — emulator vs manual, 7 cases | 7.1e-15 | **1.42e-14** | ragged K/V, initial state, varlen |
| 5 — "Triton 44/44 on L40S" | 44/44 | **unsubstantiated** | no artifact produces 44 of anything |
| 6 — GPU acceptance, 6/6 | 6/6 | **not re-run** | GPU unavailable during this audit; see below |

Levels 1–4 above were re-measured on CPU in fp64, which is where they are meaningful.
**Levels 5 and 6 require a GPU and were not re-run for this document** — the Slurm GPU
queue was saturated throughout by unrelated jobs. They are therefore reported as
*previously claimed, not re-verified here*, and the two static weaknesses noted below
(an absolute-tolerance acceptance criterion that cannot test `dg`, and a bf16-blind
determinism check) apply to level 6 as recorded.

The O(R²) re-walk restructuring is **bit-identical** to stored tiles at R=1,2,3,4,8 —
confirmed, zero extra error. One presentational point: the level-1 artifact's own
top-level exit status is `RESULT: FAIL`, because its secondary float32 comparison trips a
1e-10 threshold on what is float32 ulp noise; the chain summary omitted this, and a reader
running the script would see a failure.

Two claims that sound stronger than they are. **"Bit-identical determinism"** is checked
with `torch.equal` on gradients that have been cast to bf16, whose 1 ULP is ~0.4% — four
orders coarser than the ~1e-7 nondeterminism it purports to exclude; it would report
`True` even for `atomic_add`. And **"R=1 ≡ KDA"** is exact in parameter count and
`state_dict` structure (1,190,984 both, identical keys and shapes) but *computationally*
equal only to a bf16 budget — necessarily, since KDA fuses the gate in-kernel while the
Householder path computes it eagerly in fp32 (`recurrent.py:775-786` vs `:1261-1268`).

Three further qualifications, each traceable to source:

- **The fp64 bar cannot reach the production kernel.** `kda_householder.py:737-739`
  rejects float32 on the Triton backend and the forward accumulates in `tl.float32`
  unconditionally, so an fp64 oracle can only validate `backend="torch"`. The Triton
  kernel is comparable at bf16, where the test constant is `ATOL = RTOL = 2e-2`
  (`kda_householder_test.py:30-31`). A deliberately injected dropped-cross-term bug
  produces median relative error ~3.5e-3 and therefore passes that gate for most seeds.
  This is the principal false-pass pathway.
- **The gate can report green having verified nothing.** `kda_householder_test.py:34-58`
  loads its oracle from `probes/` outside the package and calls `pytest.skip()` when it
  is absent; a skipped suite exits 0. The fallback path resolves on a workstation that
  has the sibling directory and fails on a fresh checkout — i.e. exactly on a GPU node.
  Any future use of this suite as a gate must assert **zero skips**, not merely a zero
  exit status.
- **The two backends differ in their differentiable set.** Triton marks the final state
  non-differentiable (`kda_householder.py:639-640`) and the backward drops `dht`
  (`:654`), while the torch backend's final state *is* differentiable. Gradients
  therefore cannot flow through the carried state on the Triton path, so truncated-BPTT
  handoff and any consumer of the final state's cotangent are unsupported there.
  Relatedly, the kernel is wrapped `once_differentiable`, so second-order consumers
  (gradient penalties, Hessian-vector products) now raise instead of silently
  receiving zeros — a fix, but also a limitation to state plainly.

**The test count is "278 passed" — and also 982 skipped.** The log that produced the
figure (`/scratch/users/ericrcwu/kda/tests-1660766.log`) reads verbatim:
`278 passed, 982 skipped, 22 warnings in 39.74s`. Three qualifications follow:

- **78% of the collected suite skipped** (982 of 1,260), and the command used bare `-q`
  without `-rs`, so no skip reasons were recorded. A test that skips is not a test that
  passes. The bulk of the skips are `attention_test.py`'s flash-attention and multi-GPU
  variants, which have nothing to do with this kernel.
- The run covered **all** of `src/test/nn/attention/`, where the KDA-specific files are
  only 140 of 1,260 collected tests (11%). Reconciling from measured CPU baselines, the
  KDA files contribute **139** of the 278 passes (50 kernel + 89 module) and the other
  ~139 come from `attention_test.py` and `ring_test.py`. Quoting 278 for this kernel
  roughly doubles the real figure; the defensible number is **139**.
- The companion "52/52" figure is unverifiable: `kda_householder_test.py` collects 50
  tests at `6b75c06`, 50 in the exact tree that produced the 278, and 64 at HEAD. No state
  anywhere collects 52. The four session-2 tests credited with taking it "48 → 52" are all
  already present at `6b75c06`.

Honest phrasing: *"278 of 1,260 tests in `src/test/nn/attention/` passed with 982 skipped
(mostly unrelated flash-attention and multi-GPU variants); the KDA-specific contribution is
139."*

A specific silent-green risk was checked and came back **clean**: with `KDA_PROBES_DIR`
set, no test skipped for a missing oracle, and the `pytest.skip()` at
`kda_householder_test.py:48` never fired. One new defect surfaced instead:
`test_kda_householder_r1_matches_fla_chunk_kda` (`kda_householder_test.py:299`) carries no
`@requires_gpu`/`@requires_fla` decorator while hard-coding `cuda`, so on a CPU host it
**errors rather than skipping**.

**The mutation-testing negative control is not 18/18, and the artifact implements 8, not
18.** The project recorded "18/18 hypothesised bugs caught" over "151 cases." The only
reproducible artifact, `probes/audit_exp2_mutation.py`, implements **eight** mutations
(M1–M8) and runs them against the Python emulator, not the Triton kernel; the 18-mutation
and 151-case figures exist only in prose. Re-running it over three gate regimes
(`data/mutation_coverage.tsv`) gives 8/8 in a synthetic regime — but in **both realistic
regimes, mutation M1 (`dg` computed after the decay instead of before) passes silently**:

| Regime | `dg` reference magnitude | M1 error | Verdict at ATOL=2e-2 |
|---|---|---|---|
| `accept` (naive `-rand()` gate) | 1.75e-1 | 5.54e-2 | caught |
| `real` (production `A_log = log U(1,16)`) | 1.56e-3 | 1.54e-3 | **passes** |
| `negeig` (reflection regime) | 3.20e-3 | 3.15e-3 | **passes** |

The mechanism is the one the project itself documented for a different test: production
gating makes decay strong enough that `|dg|` collapses to ~1e-3, below any flat 2e-2
threshold, so a mutation that corrupts `dg` *completely* (relative error ≈ 99%) is
invisible to an absolute-tolerance check. The correct summary is that mutation coverage is
**7/8 in the production gate regime**, and that `dg` requires a per-gradient *relative*
error test — which the repository does now contain
(`test_kda_householder_backward_relative_error`), but which the mutation harness does not
use.

Two related tolerance findings. The determinism assertion is weaker than it looks: `dq`,
`dk`, `dv`, and `dbeta` are returned in bf16, whose epsilon is 7.8e-3, so `torch.equal`
on them cannot see reduction-order differences below ~0.4% relative — only `dg`, returned
in fp32, is a genuine determinism probe. And `torch.sum` over the `NV` axis is repeatable
for a fixed shape but **not** partition-invariant (max difference 9.8e-4 at NV=16), so
retuning `BV`/`BK` changes the bits.

**What can be claimed.** The R=1 reduction is bit-exact against `fla`'s KDA. R=2 and R=3
agree with `fla`'s `gated_delta_product` to fp64 ulp in the constant-along-K gate slice.
The genuinely per-channel gate at R>1 has no external reference and is verified only
against two transcriptions of one derivation plus a Triton implementation, with 7/8
mutation coverage in the production regime. That is a meaningful and unusually
well-documented state of affairs — but it is not the same as "verified," and this document
does not use that word for the per-channel R>1 path.

**Performance — the "406×" is against our own naive reference.** The previously reported
figures (L40S, forward+backward) are internally consistent, and their arithmetic checks
out:

| Config | Triton | torch reference | Ratio |
|---|---|---|---|
| B4 T2048 R4 | 36.1 ms | 5,018 ms | 139.0× |
| B2 T8192 R4 | 137.7 ms | 55,846 ms | **405.6×** |

The projection "a 2020-step run at T=8192/R=4 goes from 31.3 hours to 5 minutes" is also
arithmetically right (2020 × 55.846 s = 31.3 h; 2020 × 0.1377 s = 4.6 min, so "5 minutes"
is rounded up).

But the baseline is `kda_householder_torch.py`, a Python `for t in range(T)` /
`for j in range(R)` loop that builds `T*(R+1)` autograd nodes per batch-head with every
state update out-of-place and retained. Its own docstring (`kda_householder_torch.py:27-33`)
says it is "slow and memory-hungry **by design**." **So this measures our fused kernel
against our own teaching reference, not against the state of the art.**

Two further problems with the benchmark as recorded. First, `probes/bench_bwd.py` runs
`iters=3` with **no warmup**, so Triton's JIT compilation is amortized into three
iterations. Second, and decisively, the same benchmark table that yields "406×" also shows
the Triton kernel **losing**: at B4/T512/R1 it takes 291.2 ms against the reference's
272.5 ms — 7% *slower*. A speedup figure quoted from a table that contains a slowdown at
another shape needs that context.

The honest phrasing is that the Triton kernel is what makes the operator trainable at all,
turning ~56 s/iteration into ~0.14 s/iteration at T=8192. It is *not* evidence of
competitiveness with a production chunked kernel — indeed the module's own docstring
(`kda_householder.py:66-70`) states it is "expected to be materially slower than fla's
chunked kernels." **No comparison against `fla`'s `chunk_kda` was ever made**, at R=1 or
otherwise. That is the single most valuable missing measurement, because it is the number
a practitioner would actually want.

The kernel is also strongly occupancy-limited: the grid is `cdiv(V,BV) * B * H` programs,
so the probe shape (B=2, H=8, V=64) launches 64–128 programs on the L40S's 142 SMs —
nearly empty. Per-token throughput improves ~6× from batching alone (B=1→16 at T=2048
costs 2.70× the time for 16× the work), so raising batch size matters more than adding
GPUs. Note also that sequence length, not batch, is the cost driver: at constant total
tokens, T=1024→8192 costs ~3.14× more because the kernel is sequential over time.

**Known cost.** The backward's transient workspace is dominated by `hs`, which is
`O(B*T*H*K*V)` fp32 and is **not** chunked over time, plus `NV=8` partial-gradient buffers
for the four gradients that reduce over V (`dq`, `dg`, `dbeta`, and — easy to miss —
`dk`). Measured totals (`data/limitations.tsv`):

| Shape | `hs` | Partial buffers + temps | Total transient |
|---|---|---|---|
| B1 T8192 H8 K64 V64 R4 | 1.00 GiB | 0.88 GiB | **1.88 GiB** |
| B2 T8192 H8 K64 V64 R4 | 2.00 GiB | 1.77 GiB | **3.77 GiB** |
| B4 T2048 H8 K64 V64 R4 | 1.00 GiB | 0.88 GiB | **1.88 GiB** |
| B4 T8192 H8 K64 V64 R4 | 4.00 GiB | 3.54 GiB | **7.54 GiB** |

Note `hs` is independent of `R` and of `NV`, so raising `BV` does not help. The forward
fits and the backward can then OOM; keep the micro-batch at 4 or below and reach the
effective batch by accumulation. Against the torch backend it replaces, the Triton
workspace is smaller at the same shape (3.77 vs 5.65 GiB measured at B2/T8192/R4, a 0.67×
ratio) because the torch path retains activations scaling as `O(B*T*R*H*K*V)` — but the
`NV` partial buffers are unique to the Triton path, so the advantage is not uniform.

---

## 5. Probe results

**[P]** Every number in this section was recomputed from the 98 raw per-run JSON files
by `scripts/verify_claims.py`; 134 of 138 previously recorded numbers reproduced exactly,
and the four that did not are itemized in [§9](#9-corrections-to-the-project-record).

### 5.0 Backend equivalence

All 98 confirmatory runs use the Triton backend. Its adoption rests on an n=8 comparison
against the gradcheck-validated torch backend on S5 at R=4 (`data/probe_backend_equiv.tsv`,
all 48 per-seed cells reproduced exactly): pooled difference **−1.24pp, 95% CI
[−3.18, +0.70]**, sign test 5/8 negative (p=0.73), every length not significant.

One structural caveat the mean hides: the seeds do not scatter around zero but form two
clusters, five at −2 to −4pp and three at +1.2 to +2pp — the signature of runs landing in
different optimization basins with numerical differences deciding which. An *individual*
triton-vs-torch run can therefore differ by ~4pp, so only seed-averaged quantities may be
compared across backends. The two backends are bit-identical at length 40 across all
8 seeds; at length 64 they agree for 7 of 8 seeds and differ by −0.29pp on seed 4.

[§5.0a](#50a-runs-are-not-bit-reproducible-and-this-bounds-what-counts-as-an-effect)
supplies the missing context for that clustering: **same-backend, same-seed reruns are
themselves not reproducible**, differing by up to 10.4pp. So the ~4pp spread between
backends is not evidence of a kernel-fidelity gap at all — it is within the variability a
single backend shows against itself.

### 5.0a Runs are not bit-reproducible, and this bounds what counts as an effect

A free replication exists in the artifacts and was not previously exploited: the earlier
`p12` grid and the final `all_night` grid ran **the same 18 configurations** — same R,
task, seed, step count, code, and backend — differing only in the `--eval-lengths` list,
which cannot affect training or the evaluation bank. Comparing the 90 shared cells:

| | |
|---|---|
| Cells bit-exact | 23 / 90 |
| Configurations bit-exact at every shared length | **0 / 18** |
| Mean absolute difference | 1.38pp |
| Worst cell | **10.42pp** (R=4, parity, seed 2, length 256) |

Both grids ran on identical L40S hardware, so this is intrinsic software nondeterminism,
not hardware drift. **Same-seed, same-backend reproduction is therefore not exact and this
document does not claim it.** The consequence for interpretation is important and is
applied throughout:

- The S5 R-effect (+8 to +56pp) is 19–30× this noise floor and survives comfortably.
- The solvable-task differences (1–4pp on parity/S3/S4) are **within** it. They are
  correctly reported as null, and must not be read as small real effects in either
  direction.

### 5.1 R improves S5 monotonically

Mean token accuracy on S5, 3 layers, n=8 seeds (`data/probe_s5_table.tsv`):

| R | @40 | @64 | @128 | @256 | @512 | @1024 | @2048 |
|---|---|---|---|---|---|---|---|
| 1 | 91.93% | 64.34% | 32.87% | 16.81% | 8.66% | 4.84% | 2.85% |
| 2 | 99.37% | 86.53% | 46.75% | 23.67% | 12.23% | 6.48% | 3.68% |
| 4 | **100.00%** | **99.96%** | **88.60%** | **48.12%** | **24.51%** | **12.79%** | **6.78%** |

Monotonicity in R holds in **all 56 per-seed triples**, not merely in the means. Both
R=2 and R=4 beat R=1 at every length (14/14 significant uncorrected).

### 5.2 The solvability dissociation

The interaction `(R4-R1 on S5) - (R4-R1 on parity)`, paired within seed, n=8
(`data/probe_interaction.tsv`):

| Length | S5 effect | parity effect | Interaction | 95% CI | p | Holm (109 tests) |
|---|---|---|---|---|---|---|
| 40 | +8.07 | +0.00 | +8.07 | [+4.36, +11.78] | 1.3e-3 | not rejected |
| 64 | +35.63 | −0.00 | +35.63 | [+31.14, +40.12] | 3.0e-7 | **rejected** |
| 128 | +55.74 | −2.42 | **+58.15** | [+48.17, +68.14] | 2.5e-6 | **rejected** |
| 256 | +31.31 | −2.63 | +33.95 | [+25.21, +42.69] | 3.7e-5 | **rejected** |
| 512 | +15.85 | −2.10 | +17.95 | [+10.83, +25.06] | 5.6e-4 | **rejected** |
| 1024 | +7.96 | −0.79 | +8.75 | [+5.21, +12.29] | 6.3e-4 | **rejected** |
| 2048 | +3.93 | −1.15 | +5.09 | [+3.78, +6.40] | 3.7e-5 | **rejected** |

The length-40 cell was omitted from the earlier record while the prose claimed all
seven; it is in fact significant uncorrected (+8.07pp) and is the one cell that does not
survive global Holm correction. Every parity cell is not significant.

### 5.3 The difficulty control

The dissociation above is equally consistent with "R helps *hard* tasks," since parity
is trivial. S3 and S4 separate those hypotheses: as word problems they are much harder
than parity, yet R buys them nothing. **This rules out difficulty; it does not establish
solvability** — see the confounding discussed in
[§6.1](#61-what-the-horizon-view-shows), which is why this section is titled a difficulty
control rather than, as in the original record, "the solvability control." Full results
across all 28 task × length cells (`data/probe_solvability.tsv`):

| Task | Solvable | Acc @2048 at R=1 | R4−R1 @128 | R4−R1 @2048 | Cells significant |
|---|---|---|---|---|---|
| Z/2 (parity) | yes | 54.55% | −2.42 ns | −1.15 ns | 0 / 7 |
| S3 | yes | 21.96% | +4.57 ns | +0.78 ns | 0 / 7 |
| S4 | yes | 13.05% | +4.77 ns | −0.99 ns | 0 / 7 |
| **S5** | **no** | **2.85%** | **+55.74 SIG** | **+3.93 SIG** | **7 / 7** |

The universal quantifier holds as stated: **every** control cell is not significant
(minimum p = 0.087, n=8 for parity and n=5 for S3/S4) and **every** S5 cell is
significant (maximum p = 9.7e-6, n=8). S3 and S4 are 13–22% at length 2048 against
parity's 55%, so they are genuinely harder in absolute terms, and R still buys them
nothing. **The effect therefore does not track task difficulty.** What it does track —
solvability, group order, or the `(R+1)` arity bound — is not identified by this design.

Four caveats, each of which narrows the claim:

- **In raw percentage points, parity is informative at almost no length.** It is pinned at
  100.00% for both arms at lengths 40–64 (ceiling) and within ~5pp of its 50% chance level
  from 256 onward (floor). The measurable band is roughly 128–512 for S3 and 128–2048 for
  S4, so **S3 and S4 carry this argument, not parity** (`data/probe_chance_levels.tsv`).
  [§6](#6-the-effective-horizon-view) is what rescues the solvable arms from the
  floor objection, by showing they extrapolate to horizons of 128–190 tokens.
- **S3/S4 are n=5**, and this project twice produced confident effects at small n that
  vanished at n=8 ([§9](#9-corrections-to-the-project-record), item 6 of
  [§10](#10-limitations)). At n=5 the minimum detectable effect on S3 is ~8.2pp at
  length 128, against an observed +4.57pp — so the controls could hide effects up to
  roughly 2.3× their point estimates. The defensible claim is **"no large effect,"** not
  "no effect." Note also that every solvable point estimate (1–4pp) sits inside the
  run-to-run nondeterminism floor of up to 10.4pp established in
  [§5.0a](#50a-runs-are-not-bit-reproducible-and-this-bounds-what-counts-as-an-effect),
  which is an independent reason not to read a direction into them.
- **The interaction subtracts effects measured on incommensurable scales.** Parity has
  50pp of dynamic range above chance; S5 has 99.17pp. Renormalizing each task by its own
  headroom leaves the interaction significant at every length (+61.04 at length 128), so
  the claim survives — but the headline figure "+58.15pp" is scale-dependent and somewhat
  inflated by this asymmetry.
- **The "no arity ladder" observation should be cited at a length where it is visible.**
  The claim that S3 ≈ S4 despite a 4× cardinality difference was originally cited at
  length 40, where both sit at 100.00% and no ladder could be observed. It does hold in
  the measurable band: S3 60.00% vs S4 61.37% at length 256.

### 5.4 Depth and R are not substitutes

`R4−R1` on S5 by depth, n=5, with the `(L=1) − (L=4)` substitution contrast
(`data/probe_depth_r.tsv`). All seven lengths are significant uncorrected; three of seven
survive global Holm:

| Length | @L=1 | @L=2 | @L=4 | (L1)−(L4) | p | Holm |
|---|---|---|---|---|---|---|
| 40 | +61.69 | +24.95 | +3.37 | +58.32 | 4.8e-6 | **rejected** |
| 64 | +75.97 | +46.28 | +25.64 | +50.32 | 5.8e-5 | **rejected** |
| 128 | +68.10 | +52.80 | +48.48 | +19.62 | 4.1e-4 | **rejected** |
| 256 | +36.80 | +29.34 | +27.09 | +9.71 | 1.6e-3 | not rejected |
| 512 | +18.59 | +14.48 | +13.42 | +5.17 | 3.7e-3 | not rejected |
| 1024 | +9.23 | +7.11 | +6.76 | +2.47 | 2.5e-3 | not rejected |
| 2048 | +4.70 | +3.59 | +3.31 | +1.39 | 3.4e-3 | not rejected |

R's benefit is larger when the model is shallow, at every length in raw percentage points.
But "partial substitutes" is the wrong mechanism, and two checks show why.

First, under headroom normalization (dividing each effect by the accuracy range still
available to it) the substitution contrast is **not significant at lengths 40, 64, or
128**, and appears only from 256 onward. Second, the asymmetry is not that depth and R
trade off but that **R=4 is nearly insensitive to depth while R=1 is not**: at length 128,
R=4 goes 80.50% → 86.04% from 1 to 4 layers, while R=1 goes 12.40% → 37.56%. The large
+58.32pp contrast at length 40 is therefore mostly a statement about how badly the
1-layer R=1 model fails, not about substitution.

The accurate claim is that **R largely saturates the task, so depth adds little on top of
it, while depth alone only partially closes the gap.** Depth does not replace R for
long-range state tracking: at length 128 R still buys +48pp even at 4 layers.
[§6.2](#62-depth-r-and-an-incidental-capacity-control) makes this quantitative.

### 5.5 Per-channel versus per-head gating is unresolved

A secondary question was whether KDA's per-channel gate beats the per-head scalar gate of
Gated DeltaNet (GDN) at this scale. The two architectures are closely matched in size
(1,029,324 vs 1,030,872 parameters). Recomputed cleanly at n=8
(`data/probe_kda_vs_gdn.tsv`), KDA − GDN on S5 is +7.84 / +7.10 / +3.46 / +1.99 / +0.88 pp
at lengths 40–512, significant uncorrected at all five (pooled +4.25, p=6.2e-3), i.e. in
KDA's favour. On parity at length 512 it is −7.43pp *against* KDA, and on `mod_arith` at
length 40 −1.30pp against KDA (n=3).

Two reasons this is reported as **unresolved** rather than as a win. First, none of the 18
tests in this family survives Holm correction over the 109-test family. Second, the
previously recorded value for this comparison ("+2.01pp ns") was produced by a
file-loading collision that silently substituted a 2-layer model for one seed of an
8-seed 3-layer comparison; across the 216 possible resolutions of that collision the
length-128 estimate spans [−4.59, +6.12] — see
[§9](#9-corrections-to-the-project-record), item 1. A quantity that moves that much under
a bookkeeping change should not be reported as a finding in either direction. The running-
product tasks also structurally disfavour a per-channel gate, since their optimal policy
is "never forget," so per-channel flexibility is mostly a liability here.

---

## 6. The effective-horizon view

**[P]** The percentage-point tables above are hard to compare across lengths, because
the metric averages over all positions while training stops at 40. An arm scoring 6.78%
at length 2048 has not "almost failed" — it has solved a prefix and fallen to chance
(0.833%) afterwards. As length grows, any fixed prefix advantage shrinks like 1/L, which
is why the S5 effect appears to decay from +55.7pp to +3.9pp.

Model the model as correct on the first *h* positions and at chance thereafter:

```
acc(L) = h/L + (1 - h/L) * chance      =>      h = L (acc - chance) / (1 - chance)
```

*h* is an **effective horizon** in tokens. It is a descriptive reparameterization, not a
fitted mechanism, and its adequacy is testable: if one scalar *h* reproduces a run's
accuracy at all seven lengths, the seven numbers are seven views of one quantity.

**It does — for S5.** The within-run coefficient of variation of *h* across the seven
lengths is **4.9%** on S5 but **17–18%** on parity and S4, where *h* also drifts upward
with length (`scripts/effective_horizon.py`, `data/probe_effective_horizon.tsv`).
So the one-parameter description is good on the non-solvable task and only rough on the
solvable ones; the reparameterization is claimed for S5 and used descriptively elsewhere.
Ceiling-pinned cells (accuracy > 99.9%) are excluded, since there *h* is capped at *L*.

| Task | \|G\| | Solvable | n | h(R=1) | h(R=4) | Δh | 95% CI | Within-run CV |
|---|---|---|---|---|---|---|---|---|
| `s5_words` | 120 | **no** | 8 | **40.5** | **120.3** | **+79.8** | [+64.0, +95.6] | 4.9% |
| `parity` | 2 | yes | 8 | 162.9 | 138.2 | −24.6 | [−70.0, +20.7] | 18.3% |
| `s3_words` | 6 | yes | 5 | 128.4 | 144.4 | +16.1 | [−20.3, +52.4] | 8.8% |
| `s4_words` | 24 | yes | 5 | 159.0 | 157.9 | −1.0 | [−24.2, +22.2] | 17.0% |

### 6.0 A prefix correction, and what it does to the headline

The horizon model above is one way to handle the all-positions metric. A second, more
direct check is to strip the in-distribution prefix outright. Since training covers lengths
3–40 and every position is scored, positions 1–40 of a length-2048 evaluation are
in-distribution. Solving
`acc_reported = (40/L)·acc(L=40) + ((L−40)/L)·acc_tail` for the tail gives accuracy on
*extrapolated positions only* (`scripts/prefix_correction.py`):

| Task | R | tail @256 | @512 | @1024 | @2048 |
|---|---|---|---|---|---|
| `s5_words` | 1 | 2.90% (3.5× chance) | 1.60% (1.9×) | 1.30% (1.6×) | **1.07% (1.28×)** |
| `s5_words` | 2 | 9.65% (11.6×) | 4.85% (5.8×) | 2.71% (3.3×) | 1.77% (2.1×) |
| `s5_words` | 4 | **38.5% (46.2×)** | 18.1% (21.7×) | 9.25% (11.1×) | **4.92% (5.91×)** |

This materially qualifies the headline in one direction and strengthens it in another.
**R=1's apparent "3.4× chance" at length 2048 is largely a prefix artifact** — on
extrapolated positions alone it is at 1.28× chance, i.e. essentially failing. Any claim
resting on R=1 being meaningfully above chance at long lengths is invalid. But R=4's
advantage is *not* an artifact: it stands at 5.91× chance on extrapolated positions at
2048 and 46× at 256. The R effect is real and, on the corrected metric, proportionally
larger than the raw percentage points suggest.

For the solvable tasks the tail correction leaves the null intact (parity 1.05× vs 1.07×
at R=4 vs R=1; S3 1.27× vs 1.22×; S4 2.47× vs 2.72×) — small differences in both
directions, consistent with the nondeterminism floor of [§5.0a](#50a-runs-are-not-bit-reproducible-and-this-bounds-what-counts-as-an-effect).

### 6.1 What the horizon view shows

Three things become visible that the pp tables obscure:

1. **h(R=1) on S5 is 40.5 tokens against a training maximum of 40.** Without extra
   Householder factors the model does not extrapolate one token past its training range
   on the non-solvable group. R=4 reaches 120.3 — three times the training range. This
   is a cleaner statement of the headline than any single percentage-point figure.
2. **The solvable tasks are not at a floor.** Their horizons are 128–163 tokens, i.e.
   3–4× the training range, so their null Δh is informative rather than vacuous: these
   models demonstrably do extrapolate, and R changes nothing about how far. This is the
   answer to the natural objection that parity and S3 sit near chance at long lengths —
   they do, on the raw metric, but only because the metric divides by L. Caveat: *h* is a
   worse summary on the solvable tasks (within-run CV 17–18%, drifting upward with length)
   than on S5 (4.9%), so treat these three horizons as approximate.
3. **Solvability and group order are perfectly confounded in this design, and the
   published literature suggests solvability is the *wrong* discriminator.** S5 is both
   the only non-solvable group tested and the largest, and h(R=1) is ordered by |G|
   (parity 163 > S4 159 > S3 128 > S5 40). These data cannot separate solvability from a
   group-size or arity effect.

   The decisive case is **A5**, which is non-solvable but has |A5| = 60, and it was not
   run. DeltaProduct did run it, and its result cuts against the solvability reading:
   S3 needs `n_h = 2`, **S5 needs `n_h = 4`, but A5 and S4 both extrapolate at `n_h = 2`**
   — and the authors flag this explicitly as *"Unexpectedly, S4 and A5 can extrapolate
   robustly using only n_h = 2 despite the theorem suggesting 3 and 4"* (Siems et al.,
   2025, §5). Their explanation is that both are isomorphic to subgroups of SO(3,ℝ) —
   S4 to the rotation group of the cube and A5 to that of the dodecahedron — so they need
   only two Householder factors and keys of rank 3, which they support with betas ≈ 2 and
   a PCA showing three components carry >95% of key-space variance.

   **If a non-solvable group (A5) behaves like the solvable ones, then "R helps
   non-solvable groups" is not the right generalization of our data.** Every group we
   tested is consistent with *both* the solvability story and DeltaProduct's arity bound,
   so our experiment cannot distinguish them — and the one group that would has already
   been reported to favour the arity account. §5.3 is therefore relabelled a
   **difficulty control**, and solvability is presented as a hypothesis consistent with
   our data rather than as an established discriminator. Running A5 is the single most
   important missing experiment.

### 6.2 Depth, R, and an incidental capacity control

Applying the same reparameterization to the depth grid (`scripts/horizon_depth.py`):

| Layers | n | h(R=1) | h(R=4) | Δh | 95% CI | ratio |
|---|---|---|---|---|---|---|
| 1 | 5 | 14.8 | 105.6 | +90.8 | [+65.6, +116.0] | 7.11× |
| 2 | 5 | 30.1 | 103.3 | +73.2 | [+30.4, +116.0] | 3.43× |
| 3 | 8 | 40.5 | 120.3 | +79.8 | [+64.0, +95.6] | 2.97× |
| 4 | 5 | 45.9 | 113.9 | +67.9 | [+51.4, +84.4] | 2.48× |

This qualifies §5.4's "partial substitutes" framing. The *ratio* falls monotonically
(7.11× → 2.48×), so "R matters more when shallow" is real. But Δh barely moves
(+90.8 → +67.9, with overlapping intervals): on an absolute-horizon scale the benefit of
R is roughly constant in depth, and the apparent substitution is mostly h(R=1) rising
from 14.8 to 45.9 while h(R=4) stays flat near 105–120. **Four layers of R=1 (h=45.9) do
not reach one layer of R=4 (h=105.6).** Note also that h(R=1, L=1) = 14.8 is *below* the
training maximum: a 1-layer R=1 model fails on S5 even in distribution.

That last comparison is the only near-parameter-matched contrast the probe data contain,
and it runs against a capacity explanation (`scripts/param_confound.py`):

| Arm | Params | h |
|---|---|---|
| R=1, 4 layers | 1,361,936 | 45.9 |
| R=4, 1 layer | 766,532 | **105.6** |

The R=4 model has **43.7% fewer parameters and a 2.3× longer horizon**. A pure-capacity
account predicts the opposite ordering. The comparison was checked per-seed rather than
only in means: it holds on **5 of 5 seeds, at all seven lengths, in raw percentage points
as well as in horizon units**. Across all eight (R, depth) cells, horizon separates
perfectly by R with no overlap between the R=1 and R=4 groups.

This is cross-design rather than a purpose-built matched control — R and depth vary
together, and the theory itself predicts depth should help — so it weakens the capacity
confound without removing it. It is nonetheless the strongest anti-capacity evidence the
probe data contain, and it was not exploited in the original analysis.

---

## 7. The language-model result

**[LM]** This is the only parameter-matched, natural-data test in the project, and it is
the most important section of this document. **It is not a null.** Read on the endpoint
that actually has power, it is a significant result *against* R=4.

### 7.1 Design

Three arms, 12 layers, `d_model` 512, 8 heads × 64, SwiGLU MLP, tied embeddings, pure
recurrent (**no interleaved full attention** — `train_lm.py:160-176` builds a
Householder-KDA mixer for every block). Trained on 1.042B tokens of FineWeb-Edu
(`sample-10BT`, GPT-2 tokenizer) at sequence length 2048 — 31,800 steps × micro-batch 4
× accumulation 4 × 2048, i.e. ~20 tokens per non-embedding parameter for `hh1`.

R cannot be varied at fixed parameter count, so a third arm widens R=1 to match `hh4`:

| Arm | R | `d_model` | Non-embed params | n | Wall-clock/run |
|---|---|---|---|---|---|
| `hh1` | 1 | 512 | 52.1M | 5 | 24,455 s |
| `hh4` | 4 | 512 | 71.2M (+37%) | 4 | **42,930 s** |
| `hh4_r1wide` | 1 | 616 | 71.7M (within 0.7% of `hh4`) | 4 | 26,479 s |

The decision rule was fixed before seeing data: `hh4` beats **both** ⇒ the effect is R;
`hh4` beats `hh1` but ties `r1wide` ⇒ the effect is capacity and R contributes nothing;
`hh4` ≈ `hh1` ⇒ no effect at this scale.

### 7.2 Which endpoint has power

The pre-registered **primary** endpoint was degradation, `loss(L) − loss(2048)`, chosen
because val loss was judged "underpowered by design." That judgement was inverted by the
data, and understanding why is the methodological lesson of this study.

Evaluation draws windows with `FlatWindowLoader(..., seed=seed*7919 + L)`
(`train_lm.py:191`). The seed depends only on the run seed and the length — **not on the
arm** — and the source comments the intent: *"Identical across arms so the comparison is
paired"* (`train_lm.py:59`). So all arms score the *same* 32 windows at a fixed seed and
length, and evaluation-sampling noise **cancels** in every arm contrast. Pairing shrinks
the contrast standard deviation ~8× on val loss, which is why n=4 suffices to reach
p=0.0056. Only 32 windows per length is a real limitation of the *absolute* loss
estimates, but it is not what limits the *comparisons*.

The degradation endpoint therefore did not fail for lack of power — its minimum detectable
effect at n=4 is 0.010–0.018 nats, and the observed arm differences are ≤0.003. **It
failed because the phenomenon it was built to measure does not exist in this model.** The
predicted degradation was "1–5 nats"; the measured pooled change from 2048 to 16384 is
**−0.039 nats, negative in 9 of 13 runs.** Three reasons, all structural:

1. **There is no positional encoding anywhere** in the model — no RoPE, no ALiBi, no
   learned positions — so there is no extrapolation cliff to mitigate. The 1–5 nat
   prediction was imported from softmax-attention transformers, where that cliff is the
   dominant effect.
2. The recurrent state is fixed-size and O(1) in length.
3. The window-mean metric is **biased negative by cold-start dilution**: every window
   begins from an empty state, and that fixed prefix penalty is averaged over 8× more
   tokens at 16384 than at 2048, so measured loss falls with length by construction.

All nine degradation contrasts are not significant (|difference| ≤ 0.003 nats).

The power arithmetic makes the reversal explicit (`data/lm_power.tsv`). To reach 80% power
on its own observed effect, each endpoint would need:

| Endpoint | Observed effect | Paired sd | Seeds needed | Significant at n=4? |
|---|---|---|---|---|
| Val loss, `hh4` − `r1wide` @2048 | +0.0357 | 0.0100 | **3** | yes |
| Val loss, `r1wide` − `hh1` @2048 | −0.0530 | 0.0134 | **3** | yes |
| Degradation, `hh4` − `r1wide` @8192 | +0.0004 | 0.0086 | **3,670** | no |
| Degradation, `hh4` − `hh1` @16384 | −0.0007 | 0.0059 | **551** | no |

**The pre-registered primary endpoint measured nothing; the endpoint dismissed as
underpowered is the one that resolved.** The lesson is not "we needed more seeds" — at
n=4 the degradation contrast could already have detected 0.010–0.018 nats, and the arms
differ by ≤0.003. The endpoint was **vacuous rather than underpowered**: it was designed
to measure a length-degradation cliff that this architecture structurally does not have.
Choosing an endpoint on the strength of an effect-size prediction imported from a
different architecture family is the error worth carrying forward.

### 7.3 Result

Val loss at 2048 (nats/token, lower is better), per seed:

| Arm | s0 | s1 | s2 | s3 | Mean |
|---|---|---|---|---|---|
| `hh1` | 3.4620 | 3.3536 | 3.4973 | 3.5793 | 3.4730 |
| `hh4` | 3.4331 | 3.3471 | 3.4739 | 3.5687 | 3.4557 |
| `hh4_r1wide` | **3.3914** | **3.3004** | **3.4471** | **3.5411** | **3.4200** |

Paired contrasts on the 4 common seeds (`data/lm_contrasts.tsv`):

| Contrast | Isolates | Δ nats @2048 | 95% CI | p | dz |
|---|---|---|---|---|---|
| `hh4` − `hh1` | R **+** 37% capacity (confounded) | −0.0173 | [−0.0340, −0.0006] | 0.046 | −1.65 |
| `r1wide` − `hh1` | capacity alone, no R | **−0.0530** | [−0.0743, −0.0318] | 0.0042 | −3.97 |
| **`hh4` − `r1wide`** | **R at matched capacity** | **+0.0357** | **[+0.0198, +0.0516]** | **0.0056** | **+3.57** |

The ordering `r1wide` < `hh4` < `hh1` is consistent in **4 of 4 seeds**, holds at all
four evaluation lengths, and holds on final training loss (`hh4` − `r1wide` = +0.0378,
p=0.013).

So: adding parameters helps (−0.053 nats). Adding parameters *as R* helps less
(−0.017 nats). And at matched capacity, **R=4 is significantly worse than a width-matched
R=1 model** (+0.036 nats, p=0.0056).

The cost side sharpens this. Per run, `hh4` took 42,930 s against `r1wide`'s 26,479 s and
`hh1`'s 24,455 s — so R=4 is **1.62× the wall-clock of the arm that beats it** and 1.76×
the cheapest arm. Ranked by loss per unit of compute, R=4 is last on both axes at this
scale: it is worse per parameter *and* worse per second.

Training loss tells the same story as validation loss (`hh4` − `r1wide` = +0.0378,
p=0.013), and train sits 0.065–0.082 nats below val across all runs, so nothing here is an
overfitting artifact — at ~20 tokens per parameter these models are undertrained, not
overtrained.

### 7.4 What the rule says, and what it does not

The outcome fell **outside the pre-registered decision space**: the rule anticipated
`hh4` tying `r1wide`, not losing to it. The nearest branch is the second — the gain over
`hh1` is capacity, and R contributes nothing — but the data say something stronger and
worth stating plainly: at this scale, on natural text, the 37% parameter budget is better
spent on width than on Householder factors, and spending it on R is actively harmful
relative to the matched alternative.

Three limits on how far this generalizes:

1. **Scale.** 52–72M non-embedding parameters, 1.04B tokens. This says nothing about
   whether R>1 pays at 1B+ parameters or on longer training horizons.
2. **n=4.** Significant with a large paired effect size, but four seeds. The tight
   intervals come from the paired design cancelling shared eval noise; between-seed sd on
   raw val loss is ~0.08–0.10 nats, far larger than the effects, so an *unpaired*
   comparison at this n would have detected nothing.
3. **Natural text may simply not contain much non-solvable state tracking.** The probe
   result says R extends the horizon on S5; the LM result says that capability is not
   what limits next-token prediction on FineWeb-Edu at this scale. Both can be true.
4. **The "parameter-matched" control is matched on non-embedding parameters only.**
   `r1wide` matches `hh4` to +0.63% non-embedding, but reaching `d_model` 616 from 512
   also enlarges the *tied* embedding and output head by **+20.3% (+5.23M parameters)**,
   so totals differ by +5.9% (102.6M vs 97.0M). The defensible claim is therefore "width,
   including a larger tied head, beats R at equal mixer capacity" — not "mixer width beats
   R." A cleanly matched control would hold the embedding fixed and spend the budget in
   the MLP or mixer width alone. This qualifies the headline LM finding and was not noted
   in the original record.
5. **Two runs are missing, both for infrastructure reasons**, so `hh4` and `r1wide` are
   n=4 while `hh1` is n=5. `hh4-s4` hung after step 17,000 and hit its wall clock despite
   being on pace to finish; `hh4_r1wide-s4` died in 16 s on a node where CUDA was
   unavailable and `fla` fell back to CPU. Neither failure is related to the arms, so the
   missing data are missing at random with respect to the comparison.

### 7.5 Reconciling this with the probe results

The two bodies of evidence do not contradict each other, and the honest synthesis is
narrower than either taken alone.

The probes show that R>1 extends how far the S5 word problem can be tracked — an effect
the difficulty control ties to something other than task hardness, and which the horizon
analysis shows is large (40.5 → 120.3 tokens). That is a real capability difference in a
setting constructed to require exactly that capability, though which structural property
of S5 drives it is not identified here ([§6.1](#61-what-the-horizon-view-shows)).

The LM result shows that on natural text, at ~52–72M parameters, this capability is not
the binding constraint: the same parameters spent on width do more good, and R does
measurable harm at matched capacity. The probe tasks were chosen precisely *because* they
isolate the mechanism, which also makes them unrepresentative of natural text by
construction.

A reader deciding whether to use R>1 in a real model should weight the LM result more
heavily, because it is parameter-matched, on natural data, and at a scale two orders of
magnitude larger. A reader asking whether the Householder mechanism does what the theory
says should weight the probes. **This document's position is that the mechanism claim is
supported and the utility claim is not.** The one experiment designed to separate
mechanism from capacity found against the mechanism's usefulness — and it, not the probe
sweep, is the result that should govern adoption.

It is worth being precise about what would change this verdict, since "no benefit at 52M
on FineWeb-Edu" is a narrow claim. The LM study did not test the thing the probes say R is
good at: it trained at sequence length 2048 on general web text and evaluated
next-token loss, an objective in which long-range non-solvable state tracking is rare and,
where present, largely substitutable by local statistics. A fair test of the mechanism at
scale would need a corpus or task whose loss is actually gated by long-range algorithmic
structure, and a control that holds the embedding fixed. Until such a test exists, the
correct summary is not "R>1 does not work" but **"R>1 does what the theory says, and the
one place we checked whether that pays for itself, it did not."**

---

## 8. Relation to prior work

Full bibliography with per-entry confidence flags in `data/bibliography.tsv`. The
load-bearing claims below were checked against primary sources.

**Components.** The per-channel forget gate is Kimi Delta Attention (Kimi Team, 2025,
arXiv:2510.26692), which replaces the per-head scalar decay of Gated DeltaNet (Yang,
Kautz, Hatamizadeh, ICLR 2025, arXiv:2412.06464) with a diagonal `Diag(α_t)` over key
channels. The R Householder factors are DeltaProduct (Siems, Carstensen, Zela, Hutter,
Pontil, Grazzi, NeurIPS 2025, arXiv:2502.10297), which generalizes DeltaNet's single delta
step to a product of R generalized Householder matrices. Allowing `beta ∈ (0,2)` follows
Grazzi, Siems, Zela, Franke, Hutter, Pontil (ICLR 2025, arXiv:2411.12537), who show that
admitting negative eigenvalues is what unlocks state tracking in linear RNNs, extending
Sarrof, Veitsman, Hahn (NeurIPS 2024) from the diagonal case.

**Neither the pairing nor its motivation is new.** RWKV-7 already pairs a per-channel
vector decay with a Householder-like rank-1 term at R=1, so per-channel gating alongside a
delta factor is prior art. DeltaProduct's own Appendix B.4 (Eq. 30) writes a per-channel
`diag(w_{i,j})` *inside* the Householder product with a supporting theorem, though it is
theory-only — unnamed, untrained, and flagged as destabilizing. Closest in practice is
Erase-then-Delta Attention (arXiv:2606.26560, 2026), which trains a per-channel gate with
two rank-1 factors per token at 2.5B and 25B-A2.8B using the same interleave-into-a-
doubled-sequence trick to reuse KDA's chunkwise kernels; its second factor is an
asymmetric erase address rather than a full write pair, and it is fixed at R=2.

**So the contribution is an implementation and an empirical one, not a conceptual one.**
As of July 2026 no public library exposes a general, R-configurable per-channel-gated
delta-product with R symmetric `(k_r, v_r, β_r)` write pairs: in `flash-linear-attention`
0.5.1, `fla.ops.kda` asserts a per-channel gate `(B, T, HV, K)` at R=1 only, while
`fla.ops.gated_delta_product` supports R>1 but asserts a per-head scalar gate `(B, T, H)`
across its chunked, reference, and naive implementations alike ([§2](#2-the-recurrence)).
Per-channel-gate + Householder is not novel; **only R>1 under a shared per-channel gate
is.**

**The S5 result is a replication.** DeltaProduct evaluates exactly S3, S4, A5, and S5, and
reports that S3 needs `n_h = 2` while S5 needs `n_h = 4`. Our "R=4 helps S5" is therefore a
replication on a per-channel-gated variant, not a new finding, and this document claims it
as such.

**Two things are not in that work.** First, DeltaProduct sweeps R at one layer and depth at
R=1 as separate one-dimensional slices; there is no joint grid, so the depth × R
interaction reported in [§6.2](#62-depth-r-and-an-incidental-capacity-control) has no
counterpart there. Second, nothing published tests per-channel against per-head gating on
state tracking — Kimi Linear contains no gate-granularity ablation, and its own
state-tracking probe is saturated for both variants — which is why
[§5.5](#55-per-channel-versus-per-head-gating-is-unresolved) is reported even though it is
unresolved.

**Theory, and why we do not lean on it.** The classical dichotomy is that the word problem
of a finite non-solvable group is NC¹-complete (Barrington, 1989, JCSS 38(1):150–164),
while solvable groups' word problems lie in ACC⁰ (Barrington & Thérien, 1988, JACM
35(4):941–952); S5 is the standard non-solvable exemplar. Since log-precision transformers
and diagonal state-space models are confined to uniform TC⁰ (Merrill & Sabharwal, TACL
2023; Merrill, Petty, Sabharwal, ICML 2024), they cannot solve S5 unless TC⁰ = NC¹. Note
the direction: NC¹-completeness holds *for* non-solvable groups; the converse would need
ACC⁰ ≠ NC¹, which is open.

That frame is background, not our explanation. DeltaProduct never invokes solvability — it
frames its ladder entirely as the `(R+1)`-permutation arity bound — and its A5 result
actively cuts against a solvability reading, as discussed in
[§6.1](#61-what-the-horizon-view-shows). We therefore treat solvability as a hypothesis
consistent with our data rather than as an established discriminator.

---

## 9. Corrections to the project record

Every item here is a place where this document says something different from — usually
weaker than — the earlier project handoff. Each was found by recomputation from raw
artifacts.

**1. "KDA's per-channel gate is not better than GDN's per-head gate: S5 +2.01pp ns" —
wrong, and the clean data reverse it.** `probes/analyze8.py:10` globs `results/*.json`
**unsorted** and keys records on `(mixer, task, seed)`. Nine keys collide because
`depth<N>-*` and `p46-*` files share a mixer, task, and seed with the intended flat
family, so last-write-wins silently loaded `depth2-kda-s5_words-s2.json` — a **2-layer,
696,712-parameter** model — as seed 2 of an 8-seed **3-layer** comparison. Recomputed
cleanly, KDA − GDN on S5 is **+7.84 / +7.10 / +3.46 / +1.99 / +0.88 pp, significant at
all five lengths in KDA's favour** (pooled +4.25, p=6.2e-3). The claimed "+2.01" matches
the contaminated length-128 cell. The result is also unstable: the 216 possible
collision resolutions span [−4.59, +6.12] at length 128. Note that under Holm correction
over the 109-test family, 0 of the 18 KDA-vs-GDN tests survive, so the honest summary is
that this comparison is **unresolved**, not that KDA wins.

**2. "Theorem 1 predicts S5 solvable at 4 layers — confirmed" fails three ways.** First,
the depth-ladder figures (1/2/4/6 layers = 39.0 / 72.7 / 97.6 / 99.0%) are accuracies **at
length 40**, the training maximum; the same runs are at 12.4 / 24.5 / 38.0 / 40.8% by
length 128, with the effective horizon saturating near 49 tokens by 6 layers. So the claim
says nothing about length generalization (`scripts/depth_ladder_check.py`). They also come
from a different mixer (`kda`, not `kda_hh`) at n=3, and the 6-layer figure recomputes to
98.95%.

Second, the paraphrase is unfaithful. The actual Theorem 1 (Siems et al., 2025) gives
**three** routes, not two: *"(i) one layer with n_h = n−1 … (ii) 3 layers with n_h > 1
(iii) 4 layers with n_h = 1."* The record's "R ≥ 4" should be `n_h = n−1 = 4` exactly, and
route (ii) is dropped entirely.

Third — and decisively — **the construction's central ingredient is absent from the probe
model.** Routes (ii) and (iii) require *"that the MLP at the second last layer computes a
lookup-table of size 2m × (n!)^{2m}"*. `probes/model.py` has no FFN at all; its own
docstring says the model is *"free of an FFN, so that measured differences are
attributable to the mixer rather than to depth or MLP capacity."* A 4-layer R=1 model
without an MLP cannot instantiate route (iii), so the depth ladder cannot confirm it. This
document therefore drops the "Theorem 1 confirmed" claim rather than restating it.

**2a. The depth ladder also conflicts with the published result it is compared to.**
DeltaProduct reports that for S5 at `n_h = 1`, even 10 layers proved insufficient, whereas
our ladder reaches 97.6% at 4 layers. The setups differ (training length 40 vs 128, head
dim 64 vs 32, and our figure is in-distribution), but the discrepancy should be named
rather than left for a reader to find.

**3. "Both backends are exactly identical at lengths 40/64" — false at length 64**, and
contradicted by the handoff's own table three lines above the claim: seed 4 differs by
−0.293pp. Length 40 is genuinely identical across all 8 seeds. The backend-equivalence
conclusion itself is unaffected (pooled −1.24pp, 95% CI [−3.18, +0.70], every length not
significant, all 48 per-seed cells reproduced exactly).

**4. The language-model run described as "IN FLIGHT" is complete**, with 13 of 15
result files present, and its results were never reported in the project record.

**4a. The language-model result is not a null, and the "+0.0053 nats" figure is not a
measurement from this run.** The handoff describes "the one strictly parameter-matched LM
pair" as "+0.0053 nats — i.e. a clean pre-registered null," and builds its central
interpretation ("the LM null means the effect is capacity") on it. The measured
parameter-matched contrast is **+0.0357 nats, 95% CI [+0.0198, +0.0516], p=0.0056,
dz=+3.57** — about 6.7× larger and statistically significant **against** R=4, consistent
in 4/4 seeds, at all four evaluation lengths, and on final training loss. The
"+0.0053 nats" appears to be an a-priori effect-size estimate used in the power analysis,
not an observation. Treating it as a result inverted the study's conclusion.

**4b. The pre-registered primary endpoint was vacuous, and the secondary endpoint
resolved.** Degradation, `loss(L) − loss(2048)`, was predicted to be 1–5 nats and measured
−0.02 to −0.05 nats — two orders of magnitude smaller and of the opposite sign, with all
nine contrasts not significant. This was *not* a power failure: at n=4 the contrast could
have detected 0.010–0.018 nats and the arms differ by ≤0.003. The endpoint simply measured
a phenomenon the architecture does not exhibit — there is no positional encoding anywhere
in the model, the state is fixed-size, and the window-mean metric is biased negative by
cold-start dilution. The 1–5 nat prediction was imported from softmax-attention models.
Meanwhile val loss, dismissed as "underpowered by design," is well powered precisely
because eval windows are seeded `seed*7919 + L` (`train_lm.py:191`), independent of the
arm, so sampling noise cancels in every arm contrast — the source says as much
(`train_lm.py:59`), and pairing shrinks the contrast sd about 8×.

**5. The language-model `tokens_seen` field under-reports by 4×.**
`lm/train_lm.py:349` computes `steps * batch * seq_len`, omitting `args.accum`, while the
training loop does run 4 accumulation micro-steps per optimizer step. The true budget is
31800 × 4 × 4 × 2048 = **1.042B tokens** (~20 per parameter), not the 260.5M the JSONs
report. The run received its intended budget; only the field is wrong.

**6. The language-model model is pure recurrent.** `train_lm.py:160-176` builds a
Householder-KDA mixer for all 12 blocks, with no interleaved full attention. The earlier
concern that "3 of 12 layers are full attention absorb exactly the long-range work being
measured" applies to a different, unbuilt 370M proposal — not to the run that happened.
This makes the null cleaner, not dirtier.

**7. "SIG at all seven lengths" is not seven independent findings.** Per-seed effects
correlate strongly across lengths (mean |r| = 0.585 for the S5 contrast). The
eigenvalue spectrum of the 7×7 correlation matrix is 4.95, 1.92, 0.08, 0.02, 0.02,
0.005, 0.000 — two components carry 98% of the variance, in a clean two-block split
between in-distribution lengths (40–64) and extrapolation lengths (128–2048). The
effective number of independent tests is **1.74** (participation ratio) to **4.40**
(Cheverud–Nyholt). The seven lengths cannot be used as evidence of robustness.

**8. Nothing in the original analysis corrected for multiplicity.** 109 unique
hypothesis tests were run across the grids. Under Holm over that family: the interaction
survives at 6 of 7 lengths and the S5 R-effect at 6 of 7 (the length-40 cells drop); the
depth substitution survives at only 3 of 7; the KDA-vs-GDN family survives at 0 of 18.
No null result is affected, so the difficulty control and backend equivalence are
untouched.

**9. A significance rule in the analysis code is indefensible but never fires.**
`analyze_conf.py:60-61` returns verdict "SIG" with `dz = inf` whenever the paired
differences have zero variance and a nonzero mean. Of 130 tests, 7 have zero variance —
and in all 7 the mean is also exactly zero, so the branch is never taken. No headline
claim depends on it. Separately, the hardcoded `T_CRIT` table is correct to <5e-4 for
all 13 entries it contains but silently falls back to 2.0 for missing degrees of freedom
(12 and 13), which would be anti-conservative if reached; it is not reached.

**9a. "278 tests pass on GPU, 0 failed" omits 982 skips and annexes unrelated tests.** The
source log reads `278 passed, 982 skipped`. The run covered all of
`src/test/nn/attention/`, of which the KDA files are 140 of 1,260 collected; the
KDA-specific contribution is **139** passes. "52/52" matches no collectable state (50 at
`6b75c06`, 64 at HEAD), and "44/44" corresponds to no artifact. Separately,
`test_kda_householder_r1_matches_fla_chunk_kda` has no device decorator and hard-codes
`cuda`, so it errors rather than skips on CPU — a defect already present in the tree that
produced the 278.

**9b. Mutation coverage is 8 mutations, not 18, and 7/8 in production gate regimes.** The
"18/18 across 151 cases" figure has no reproducible artifact; the harness that exists tests
eight mutations against the Python emulator rather than the Triton kernel, and one of them
survives the acceptance threshold whenever the production gate collapses `|dg|` to ~1e-3.

**9c. The recorded verification tolerances are slightly better than measured** (3.6e-15 vs
4.4e-15 at level 3; 7.1e-15 vs 1.42e-14 at level 4), and the level-1 artifact exits
`RESULT: FAIL` on a float32 ulp comparison that the chain summary does not mention.

**9d. The R>1 path is better verified than the commit message claims.** Against its own
"no independent verification" note, R=2 and R=3 do match `fla`'s
`naive_recurrent_gated_delta_product` to fp64 ulp (1.78e-15, 3.55e-15) in the
constant-along-K gate slice. See [§4](#4-verification-what-is-and-is-not-established).

**10. The vendored probe harness is not the code that produced the results.** The
result-generating `train_probe.py` on FarmShare is 6,313 bytes and hardcodes
`allow_neg_eigval=True`; the vendored copy is 30,235 bytes and requires a
`--beta-regime` flag with no default, so the published runs' command lines would now
fail with an argparse error. `model.py` differs by nearly 5×. The reproduction path
documented in [`README.md`](README.md) therefore points at the FarmShare copies.

**11. The mechanism was on.** One risk worth recording as *cleared*: the KDA classes
default `allow_neg_eigval=False`, which by the project's own account "voids the
mechanism." The result-producing `train_probe.py` passes `allow_neg_eigval=True`
explicitly for all three mixers, so `beta ∈ (0,2)` and reflections were available in
every published run.

---

## 10. Limitations

Ordered by how much they should change a reader's conclusions.

1. **No purpose-built parameter-matched probe control.** R=4 has 2.21× the mixer
   parameters of R=1. The probe grid cannot separate mechanism from capacity by design.
   The handoff's defence — that R=4 buys nothing on parity despite the extra parameters —
   is weak, because parity needs no capacity. The genuine mitigations are the incidental
   cross-design comparison in [§6.2](#62-depth-r-and-an-incidental-capacity-control)
   (fewer parameters, longer horizon) and the difficulty control, which no
   capacity account explains. Neither is a substitute for a matched arm.
2. **Solvability, group order, and the arity bound are perfectly confounded — and the
   literature favours the account we did not test.** S5 is the only non-solvable group
   tested and also the largest (|G| = 120 against 24, 6, 2). The separating case is A5
   (non-solvable, |G| = 60), which we did not run and which DeltaProduct reports
   extrapolating at `n_h = 2` like solvable S4, attributing this to A5 ≅ the rotation
   group of the dodecahedron ⊂ SO(3). Our data rule out *difficulty* as the explanation;
   they do not establish solvability, and the best available external evidence points to
   the `(R+1)` arity bound instead.
3. **The only parameter-matched test finds against R.** At matched capacity R=4 is
   significantly *worse* than width-matched R=1 (+0.036 nats, p=0.0056, n=4) and costs
   1.62× the time. This is the result that should govern adoption; the probe sweep should
   not. Its own limits are n=4 and a single scale (52–72M non-embedding parameters,
   1.04B tokens) — see [§7](#7-the-language-model-result).
4. **The per-channel gate at R>1 has no external reference.** R=1 is bit-exact against
   `fla`'s KDA and R=2/R=3 match `fla`'s `gated_delta_product` to fp64 ulp when the gate
   is constant along K — but the novel combination itself is verified only against two
   transcriptions of one derivation. A shared misreading of the intended recurrence would
   pass every level of the chain.
5. **Seven lengths are ≈2 findings**, and the depth-substitution result survives global
   Holm at only 3 of 7 lengths and is not significant at lengths 40–128 under headroom
   normalization.
6. **Runs are not bit-reproducible.** Re-running a published configuration at the same
   seed and backend reproduces 0/18 configurations exactly, with a mean absolute difference
   of 1.38pp and a worst case of 10.42pp. Every solvable-task point estimate (1–4pp) lies
   inside that floor, which is an independent reason to treat those cells as null rather
   than as small effects.
7. **The published runs were produced by a less-instrumented harness than the one
   vendored in the repository.** The result-generating `train_probe.py` has no arm table,
   no beta-regime flag, no parameter-matching helper, no eval-bank checksum, and no
   source-revision field; the vendored `model.py` has an FFN and a parameter ledger that
   the result-generating copy lacks entirely. The run records carry only `seed`,
   `backend`, `num_householder`, `task`, `steps`, `train_range`, and `n_params` — so
   several governance features one might assume were in force were absent. One integer
   also seeds initialization, curriculum, and task instances together, so "n=8 seeds"
   varies initialization and data jointly, and the evaluation bank moves with the seed.
   Arms are paired on the same seed, so the contrasts remain valid.
8. **The two results ranked highest by the original record are n=5**, a regime this
   project independently proved unreliable: an n=3 "+8.92pp" effect collapsed to
   +2.01pp ns at n=8, and an n=2 "systematic" backend gap proved to be noise.
9. **Probes are 1.0–3.0M parameters on algorithmic tasks with out-of-distribution
   evaluation by construction.** They are built to expose a mechanism, not to predict
   value on natural text.
10. **The bf16 test gate (ATOL=RTOL=2e-2) is too loose for `dg`.** It passes an injected
   cross-term bug for ~90% of seeds, and mutation coverage is 7/8 rather than 8/8 in both
   realistic gate regimes because a mutation that corrupts `dg` by ~99% relative lands
   under a flat absolute threshold. The GPU test suite can also skip its oracle-dependent
   tests and still exit 0. Additionally, the determinism check is blind below ~0.4%
   relative on the four gradients returned in bf16.
11. **Engineering limits:** the backward's `hs` workspace is not chunked over time and
   can OOM above micro-batch 4; the Triton path does not propagate gradients through the
   carried state; second-order autograd raises.
12. **The "406×" is against our own naive reference**, not a production kernel, and the
    benchmark that produced it ran 3 iterations with no warmup. The same table shows the
    Triton kernel 7% *slower* than the reference at B4/T512/R1. **No comparison against
    `fla`'s chunked kernels was ever made** — the module docstring itself expects to be
    "materially slower" than them. This is the most valuable missing measurement.
13. **"278 tests pass" is 278 passed and 982 skipped**, across a directory that is 89%
    unrelated tests; the KDA-specific figure is 139. "52/52" and "44/44" have no artifact
    that produces those numbers.
14. **Reported chain tolerances were mildly optimistic** (level 3: 4.4e-15 measured vs
    3.6e-15 claimed; level 4: 1.42e-14 vs 7.1e-15), and the determinism check is blind
    below ~0.4% relative because it compares bf16-cast tensors.
15. **The headline probe result is a replication, and the mechanism pairing is prior art.**
    DeltaProduct already reports S5 requiring four Householder factors, and per-channel
    gating alongside a delta factor already exists in RWKV-7 and (at R=2, trained at
    scale) in Erase-then-Delta Attention. What is new here is R>1 under a shared
    per-channel gate, the joint depth × R grid, and the parameter-matched LM comparison.
    See [§8](#8-relation-to-prior-work).

---

## 11. Reproduction

All computation was performed on Stanford FarmShare (NVIDIA L40S, sm_89). The analysis
scripts in `scripts/` are CPU-only, depend only on the standard library, use no RNG, and
sort every glob, so rerunning regenerates `data/` byte-for-byte.

**Anchor the probe harness at git `93b60d7`, not at HEAD.** The harness vendored in the
repository is a later evolution and **cannot run the original command lines** — it rejects
`--seed` as ambiguous against `--seed-init`/`--seed-data` and exits 2. (It fails loudly,
so there is no silent-green risk, but it does not reproduce the published runs.) The
result-generating files on FarmShare at `/scratch/users/ericrcwu/kda/probes/` are
md5-identical to `93b60d7`; retrieve them with `git show 93b60d7:./train_probe.py`.

Caveats on what "reproduction" can mean here, established in
[§5.0a](#50a-runs-are-not-bit-reproducible-and-this-bounds-what-counts-as-an-effect):
re-running a published configuration at the same seed and backend does **not** reproduce
its accuracy bit-exactly (0/18 configurations, up to 10.4pp). Seed-averaged quantities and
paired contrasts are the only stable units. The run records also omit `allow_neg_eigval`,
`n_layers`, model geometry, learning rate, batch size, source revision, and library
versions; `n_layers` survives only in the filename (`-L<n>-`), and any analysis that fails
to parse it will silently pool the 1- and 2-layer depth runs into the 3-layer cells.
