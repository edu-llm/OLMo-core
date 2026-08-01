# LIV experiment probes

Cheap measurements that gate the expensive work. Each one exists to kill a claim before we spend
GPU-days on it. Run from this directory.

Venv: `../.venv-spectra/bin/python` (torch 2.13.0, transformers 5.14.1). Note `OLMo-core/.venv` has
**no** `transformers`, and this Mac has **no CUDA** — that is why `p1_launch_bench.py` is a FarmShare job.

| script | status | what it decides |
|---|---|---|
| `l40s_breakeven.py` | ✅ run | Analytic break-even for P1's extra kernel launches, per GPU. **Corrected the widely-quoted 4.72 µs figure, which was computed at d=2048 instead of the frozen d=1024.** Now superseded by direct measurement. |
| `spectra_v2.py` | ✅ run → `spectra_v2_results.json` | Whether LIV gate projections are preferentially low-rank, in the metric that governs output error. **Answer: no** (see below). |
| `p1_launch_bench.py` | ✅ run (FarmShare 1670883) → `p1_bench_results.json` | Whether P1's latency claim is real on metal. **Answer: no — it is 8.2% slower even in the best case.** |
| `p1_verify.py` | ✅ run (FarmShare 1670884) → `p1_verify_results.json` | Replication (3 trials, ≤0.3% spread), profiler-measured kernel counts, and the iso-byte controls that identify *why*. |
| `structure_energy.py` | ✅ run → `structure_energy_results.json` | Which cheap structure to prefer, now that latency no longer decides. **Answer: low-rank, by 80 points.** |

## 🔴 Headline: P1's decode-latency claim is dead

L40S, CUDA-graphed, 3 trials, spread ≤0.3%, kernel counts measured (not assumed):

| arm | kernels | MiB/tok | graphed | vs dense | achieved BW |
|---|---:|---:|---:|---|---:|
| `dense` (stock LIV) | 20 | 40.0 | 56.2 µs | — | 695 GB/s |
| `lowrank_fused` r=128 | 30 | 10.0 | 60.8 µs | **−8.2%** | 161 GB/s |
| `lowrank_sep` r=128 | 40 | 10.0 | 76.5 µs | −36.0% | 128 GB/s |
| **`grouped` g=4** | 20 | 10.0 | **47.6 µs** | **+15.3%** | 205 GB/s |

**Even fused gates + CUDA graphs — the configuration the analytic model predicted would win — is
8.2% slower than stock LIV while reading 4× fewer bytes.**

**Why, from the iso-byte control** (bytes held constant, only structure varies): at 40 MiB, `dense`
56.2 µs vs `lowrank_fused r=512` 90.0 µs ⇒ **~3.4 µs per extra kernel even under graphs**. Dense
reaches 80% of L40S peak bandwidth; the factorized version reaches 19%. **Skinny GEMVs cannot saturate
the memory system, so the bytes they save buy far less time than roofline predicts.** The analytic
model blamed launch overhead; the real cause is GEMV inefficiency.

**Generalizable lesson: for any decode-time factorization, run an iso-byte control before trusting a
roofline estimate.**

## `grouped` wins latency but loses quality — do not promote it

`grouped g=4` and `lowrank r=128` cost *identically* (0.25× params, 10 MiB/token), so the choice is
pure quality. Activation-weighted retained energy on the released checkpoint:

| structure | params | retained |
|---|---:|---:|
| `lowrank r=128` | 0.25× | **0.929** |
| `grouped g=4` | 0.25× | **0.130** |
| *random mask, 25% density* — null | 0.25× | *0.130* |
| `grouped g=2` | 0.50× | 0.336 |

Block structure buys **nothing** over random sparsity at equal density, and channel ordering doesn't
rescue it (random permutations `[0.125, 0.133]` vs 0.130 for identity) — the deficit is structural.

⚠️ **Caveat:** this measures approximation of *trained dense* weights, which favors low-rank by
construction — Eckart-Young makes rank-r truncation optimal, while the block mask is not optimized at
all. A from-scratch grouped layer is not the block-diagonal part of a trained dense one (cf. GaLore's
`W=BA` collapsing from scratch, 142.53 vs 15.56 ppl at 1B, at more generous rank fractions). A strong
prior, not a verdict.

## Results so far

**`l40s_breakeven.py`** — at the frozen d=1024, r=128, 0.75× achievable bandwidth:

| card | saved/token | break-even/launch |
|---|---:|---:|
| A100-40GB | 27.0 µs | 1.35 µs |
| H100-80GB | 12.5 µs | 0.63 µs |
| L40S (FarmShare) | 48.5 µs | 2.43 µs |

All below a typical 5-10 µs launch cost, so **P1 needs both mitigations** — fused `d→2r` gates
(20 launches → 10, doubling break-even) *and* CUDA graphs. Fused+graphed on L40S breaks even at
**4.85 µs**. Break-even scales as 1/bandwidth, so the slower L40S is the **most forgiving** card
available: FarmShare is a better venue for this benchmark than an H100 would be. Also `r=512` saves
**zero** bytes at d=1024 (`2dr ≥ d²` once `r ≥ d/2`) — keep it as a quality datapoint only.

**`spectra_v2.py`** — mean over all 10 LIV layers, 32,768 calibration tokens, `rank(Σ_x) = 1024`:

| tensor | plain eff. rank | activation-aware | E@128 aware |
|---|---:|---:|---:|
| B (pre-gate) | 790.1 | 505.9 | 0.926 |
| C (post-gate) | 770.9 | 480.7 | 0.931 |
| x (value stream) — *control* | 790.5 | 507.8 | 0.925 |
| `out_proj` — *control* | 778.5 | 609.2 | 0.787 |
| random Gaussian — *null* | 824.2 | 646.3 | 0.760 |

**P1's premise stays falsified.** Activation-aware rank drops ~36%, but *identically for the value
stream* (gates 493.3 vs value 507.8) — all three read the same `x`, so the collapse belongs to the input
distribution, not to gates. Keep the framing *"gates **tolerate** low rank"*, not *"gates are low-rank"*.

**But feasibility improved a lot:** rank-128 retains **92.6%** of activation-weighted energy against
only **45.8%** of plain Frobenius energy. The defensible claim is *"rank-128 discards 7% of the energy
that actually reaches the output — we test what that costs."*

## Two traps these scripts encode

1. **Calibration tokens must be ≫ d.** A first pass used 568 tokens for a 1024×1024 covariance, so
   `Σ_x` was rank-deficient by construction and reported a spurious **3.0× collapse to 267** — which
   read as strong support for P1. Always report `rank(Σ_x)`. Convergence is still rising at 32k
   (L0: 573.5 → 600.2 → 608.0 at 8k/16k/32k), so current numbers are a mild *under*estimate.
2. **Never benchmark P1 without CUDA graphs.** Un-graphed, the measurement reports the wrong *sign* —
   it measures dispatch overhead, not the architecture.
