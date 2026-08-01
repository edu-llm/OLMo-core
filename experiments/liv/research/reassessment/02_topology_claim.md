# Reassessment 02 — Adjudicating the TOPOLOGY claim

**Author:** reassessment team member 02. **Started:** 2026-08-01.
**Assignment:** re-derive from scratch the claim that "mostly-LIV vs param-matched all-GQA saves
20 KiB/token, hitting a 10% decode-traffic win at T ≈ 4,121"; sanity-check the framing against the
P1 bytes≠time lesson; verify the arm builder; design the minimum experiment; adversarial
novelty check.

**Labelling convention used throughout:** every claim is tagged
`MEASURED` (I ran something and observed it), `INFERRED` (derived by arithmetic/logic from
something measured or from a primary source), or `ASSUMED` (a modelling choice I made, stated
so it can be attacked).

**Status: IN PROGRESS — written incrementally, sections appended as completed.**

---

## §0. Sources consulted (running log)

- `/Users/ericwu/Developer/Capstone_LLM/HANDOFF.md` — read in full. MEASURED (file content).

- `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/01_lfm2_architecture.md`
  §3.2 (SwiGLU width transform), §6.1–6.4 (param formula + 6-checkpoint verification), §7.1–7.4
  (cache accounting), §11 (full config table). MEASURED (file content).
- `https://huggingface.co/LiquidAI/LFM2-350M/raw/main/config.json` — fetched live 2026-08-01.
  MEASURED. Confirms: `hidden_size` 1024, 16 layers, 16 q heads, 8 kv heads, vocab 65,536,
  `block_ff_dim` 6656, `conv_L_cache` 3, `conv_bias` false, attention at 2/5/8/10/12/14,
  `torch_dtype` **bfloat16**, `block_auto_adjust_ff_dim` true, `block_multiple_of` 256,
  `rope_theta` 1e6, `norm_eps` 1e-5. Independent of the repo docs — they agree.
- `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/crossover.py`,
  `proposals.py` — read + run. MEASURED.
- `/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer/src/olmo_core/nn/transformer/liv_arms.py`
  — worktree EXISTS, branch `agent/claude-01/liv-short-conv-mixer` at `c2aac8e` on top of
  `83e4dce` on top of `f17824e`. Working tree clean. File is 396 lines. MEASURED.

**Execution constraint honoured:** per a mid-task correction, NOTHING was executed on the local
Mac after the first two reads. All arithmetic below is either done by hand in this document or
run on the FarmShare **login node** (pure CPU) over the pre-authenticated control socket.
(Exception, disclosed: `crossover.py` and `proposals.py` were run locally *before* the
correction arrived — they are 60-line integer-arithmetic scripts with no imports.)

---

## §1. FIRST AND LARGEST FINDING — `crossover.py` is the WRONG MODEL

`crossover.py` and `proposals.py` are both hard-coded to **d=2048, F=8192, H=32** — that is the
**LFM2-1.2B** geometry, not the frozen 350M study scale. Line 7 of `crossover.py`:
`d, L, n_liv, n_gqa = 2048, 16, 10, 6` and the docstring says so explicitly
("Geometry: LFM2-1.2B-like").

MEASURED output of `crossover.py`:

```
TOTAL           1,170,272,256  (1.170B)      <- the 1.2B model
weight bytes/decode-token   2.341 GB
KV bytes/token  6 GQA         12,288 B (12 KiB)
KV bytes/token 16 GQA         32,768 B (32 KiB)
CROSSOVER: 6-GQA hybrid T = 190,474 ; 16-GQA control T = 71,428
T=4,096 -> KV is 2.1% of traffic ; T=32,768 -> 14.7%
```

**None of the HANDOFF's headline numbers appear in this script's output.** The HANDOFF quotes
6.6% @4K / 36.2% @32K / crossover T=57,690 — the script prints 2.1% / 14.7% / 190,474. The
HANDOFF numbers are the *350M* numbers (§2 below reproduces them); the script is the *1.2B*
numbers. So:

> **DISCREPANCY 1 (documentation, not arithmetic).** `crossover.py` does **not** compute the
> numbers the HANDOFF cites, and nothing in the HANDOFF says so. Anyone re-running the "runnable"
> script that `00_my_arithmetic_check.md` advertises will get numbers that contradict the
> HANDOFF by 3×. **Correction: `crossover.py` must be re-parameterised to d=1024/F=4608/H=16, or
> renamed `crossover_1p2b.py` and a 350M sibling added.** The arithmetic in it is *correct for
> 1.2B* — I re-derived 1,170,272,256 vs the checkpoint's 1,170,340,608 and the 68,352 gap is the
> script omitting the 16×2d layer norms (65,536), the final norm (2,048) and the 6×2×head_dim
> qk-norms (768) — i.e. the script is a *mixer/MLP/embedding* ledger, ~0.006% light. Fine for
> traffic ratios, wrong as a parameter ledger.

One consequence worth flagging: `crossover.py`'s `12,288 B` KV/token is numerically identical at
1.2B and 350M — which is exactly the "scale-invariance" the HANDOFF leans on, and is why this
error was invisible. See §2.4 for how far that invariance actually extends (less far than claimed).

---

## §2. RE-DERIVATION FROM SCRATCH — the topology arithmetic

Script: `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/reassessment/topology_math.py`
(stdlib only, no torch). **Executed on the FarmShare login node**
(`/scratch/users/ericrcwu/liv/topology_math.py`, `python3` 3.12.3, CPU). MEASURED = the outputs
below are that script's stdout.

### 2.1 Geometry, from the live config (not from any doc)

| field | value | source |
|---|---:|---|
| `hidden_size` d | 1,024 | HF config.json, fetched 2026-08-01 |
| `num_hidden_layers` L | 16 | ditto |
| `num_attention_heads` H | 16 | ditto |
| `num_key_value_heads` G | 8 | ditto |
| head_dim (derived H·hd=d) | 64 | INFERRED |
| `vocab_size` V, tied | 65,536 | ditto |
| `conv_L_cache` k | 3 | ditto |
| `block_ff_dim` → effective F | 6656 → **4,608** | `256·ceil(int(⅔·6656)/256)` = 256·18. INFERRED, matches paper Table 1 |
| attention indices | 2,5,8,10,12,14 → 6 attn / 10 LIV | ditto |
| `torch_dtype` | **bfloat16** | ditto |

### 2.2 Parameter ledger — MEASURED, exact

```
LIV mixer   4d² + kd            =   4,197,376   (4.003 d²)
GQA mixer   2d² + 2d(G·hd) + 2hd =  3,145,856   (3.000 d²)
MLP/layer   3dF                 =  14,155,776
embeddings  V·d (tied)          =  67,108,864
L0 TOTAL                        = 354,483,968     <-- exactly HF's safetensors total
A16-P (16 GQA, F solved = 4,820)= 354,388,992     <-- exactly liv_arms.py's declared value
delta = -94,976 = -0.0268% of L0
```

I independently re-solved the A16-P SwiGLU width by brute force over the multiple-of-4 grid
(4000..5600) minimising |params − L0|: the optimum is **4,820**, matching the constant hard-coded
at `liv_arms.py:126`. **The arm builder's width is not guessed and not drifted. VERIFIED.**

Side note worth carrying: at d=1024 the LIV mixer is only **1.334×** the GQA mixer, not the
1.60× that holds at d=2048, because GQA's coefficient is `2 + 2·(G·hd/d)` = 3.0d² at d=1024
(KV width = d/2) vs 2.5d² at d=2048 (KV width = d/4). Any prose that quotes "1.6×" while
discussing the 350M study is wrong. INFERRED.

### 2.3 Decode traffic — every headline number CONFIRMED

**ASSUMED dtype: bf16 for weights AND KV.** Justification: the released config says
`torch_dtype: bfloat16`, and this is a *training-scale* argument about models we will train in
bf16. See §2.5 for why this assumption is the weakest link.

| quantity | HANDOFF claim | my re-derivation | verdict |
|---|---:|---:|---|
| KV bytes/token, L0 (6 attn) | 12 KiB | **12,288 B = 12.00 KiB** | CONFIRMED |
| KV bytes/token, A16-P (16 attn) | 32 KiB (implied) | **32,768 B = 32.00 KiB** | CONFIRMED |
| ΔKV saving | 20 KiB/token | **20,480 B = 20.00 KiB** | CONFIRMED |
| weight bytes/token, L0 | (not stated) | **708,967,936 B** | matches `07`'s `708.96e6` |
| weight bytes/token, A16-P | (not stated) | **708,777,984 B** (−189,952) | new |
| KV read == weight read (L0) | T = 57,690 | **T = 57,696.0** | CONFIRMED (docs rounded) |
| KV share of decode traffic @ 4K | 6.6% | **6.63%** | CONFIRMED |
| KV share @ 32K | 36.2% | **36.22%** | CONFIRMED |

Full KV-share curve for L0 (MEASURED): 1.74% @1K, 3.43% @2K, **6.63% @4K**, 12.43% @8K,
22.12% @16K, **36.22% @32K**.

### 2.4 The T ≈ 4,121 threshold — CONFIRMED, with a caveat about which formula

The docs solve `f = ΔKV·T / (W + KV_A16·T)` → `T = f·W/(ΔKV − f·KV_A16)`, with `W = W_L0`:

```
T = 0.10 × 708,967,936 / (20,480 − 0.10×32,768) = 70,896,793.6 / 17,203.2 = 4,121.14
```

**MEASURED: 4,121.14. The claim T ≈ 4,121 is CONFIRMED to 4 significant figures.**

But three different quantities are all called "a 10% win", and they give three different T:

| definition | T |
|---|---:|
| (i) docs' formula: ΔKV·T is 10% of A16-P traffic, both arms assigned W_L0 | **4,121.1** |
| (ii) **ratio A16-P/L0 = 1.10** using each arm's own weight bytes | **3,692.6** |
| (iii) L0 saves 10% of A16-P traffic, each arm's own weights (ratio = 1/0.9) | **4,131.1** |

The task brief asked specifically for "**the T at which the decode-traffic ratio A16-P/L0 hits
1.10**". That is definition (ii), and the answer is **T = 3,693, not 4,121** — an 11.6% difference.
The two differ because a 10%-of-baseline *saving* is a 11.1% *ratio*, and because A16-P is 189,952
weight-bytes lighter (a second-order effect, ~10 tokens' worth).

> **DISCREPANCY 2 (definitional, not an error).** 4,121 is right for "L0 saves 10% of the
> all-GQA arm's traffic." It is NOT the T where the ratio reaches 1.10 (that is 3,693). Both
> are defensible; the write-up must pick one and name it. I recommend definition (iii)/"L0
> saves f of A16-P's traffic" because it is what a reader hears, and state T=4,131 with each
> arm's own weights. **The difference is immaterial to the conclusion** — all three land
> between 3.7K and 4.2K, i.e. "right at a 4K training context."

Traffic ratio A16-P/L0 as a function of cache occupancy T (MEASURED):

| T | ratio | L0 saves |
|---:|---:|---:|
| 512 | 1.0144 | 1.42% |
| 1,024 | 1.0288 | 2.80% |
| 2,048 | 1.0569 | 5.38% |
| **4,096** | **1.1102** | **9.93%** |
| 8,192 | 1.2070 | 17.15% |
| 16,384 | 1.3684 | 26.92% |
| 32,768 | 1.6035 | 37.64% |
| ∞ | 2.6667 | 62.50% (hard ceiling = 10/16) |

`07_latency_kernels.md`'s table (9.95% @4K, 17.17% @8K, 26.93% @16K, 37.64% @32K, ceiling
62.5%) reproduces to within 0.02pp everywhere — the residual is that doc assigning both arms
`W_L0`. **The `07` table is sound.**

### 2.5 "Scale-invariant 12 KiB" — TRUE but MISLABELLED, and the label matters

MEASURED across the family:

| model | d | n_attn | KV/token | T where KV read == weight read |
|---|---:|---:|---:|---:|
| LFM2-350M | 1024 | 6 | **12,288 B** | 57,696 |
| LFM2-700M | 1536 | 6 | **12,288 B** | 120,848 |
| LFM2-1.2B | 2048 | 6 | **12,288 B** | 190,485 |
| LFM2-2.6B | 2048 | 8 | **16,384 B** | 313,632 |
| LFM2-8B-A1B | 2048 | 6 | **12,288 B** | 1,357,410 |
| LFM2-24B-A2B | 2048 | 10 | **20,480 B** | — |

> **DISCREPANCY 3 (wording, and it is load-bearing for Key Decision #1).**
> `KV/token = n_attn · 2 · G · hd · b` **contains no `d`**. The constancy at 12,288 B is not
> "scale invariance" — it is (a) independence from `d_model`, times (b) Liquid's coincidence of
> holding `G·hd = 512` and `n_attn = 6` across four checkpoints. It **breaks at 2.6B (16 KiB)
> and 24B-A2B (20 KiB)** the moment `n_attn` changes.
> **Correct statement: "KV bytes/token depends only on `n_attn·G·hd`, not on `d_model` — so
> shrinking `d` from 2048 to 1024 makes cache effects 3.3× more visible without changing the
> cache at all."** The HANDOFF says "~2.5× more visible"; the true factor is the weight-byte
> ratio **W_1.2B/W_350M = 2,340,681,216/708,967,936 = 3.30×** (MEASURED), which is also exactly
> the ratio of the 10% thresholds (13,606/4,121 = 3.30). **Correction: 2.5× → 3.30×.**
> The direction of Key Decision #1 is unaffected and in fact *strengthened*.

### 2.6 The dtype assumption is the biggest soft spot — and the task brief was right to ask

`T = f·W/(ΔKV − f·KV_A16)` is **invariant under a uniform rescaling of W and KV** (numerator
and denominator both scale). It is **not** invariant when only weights are quantized, which is
exactly what the measured ONNX build did (q4 weights, fp16/fp32 KV). MEASURED sensitivity:

| weight dtype | KV dtype | W bytes | T(10%) |
|---|---|---:|---:|
| bf16 | bf16 | 708,967,936 | **4,121** |
| bf16 | fp32 | 708,967,936 | 2,061 |
| int8 | bf16 | 354,483,968 | 2,061 |
| **q4 (+ scales, ~4.5 bit)** | **bf16** | 199,397,232 | **1,159** |
| q4 (+ scales) | fp32 | 199,397,232 | 580 |
| q4 (raw 4-bit) | bf16 | 177,241,984 | 1,030 |
| int8 | int8 | 354,483,968 | 4,121 |

**This is good news for the claim, not bad.** On the *deployment* configuration Liquid actually
ships (q4 weights, unquantized KV — the ONNX build measured at 40.3 tok/s), the 10% threshold
falls to **T ≈ 1,159**, and at a 4K context the topology advantage is far larger than 9.93%.
The bf16 number is the **conservative** end of the range.

> **Recommendation: report the threshold as a RANGE across deployment dtypes
> (T ≈ 1,030–4,121, i.e. ~1K at q4-weight edge deployment to ~4K in bf16 training precision),
> not as a single number.** A single "4,121" invites the objection "but nobody deploys a 350M
> model in bf16," and the honest answer makes the claim stronger. Also state the KV dtype
> explicitly: if a reviewer assumes fp32 KV (some runtimes default to it), T halves again.

---

## §3. VERIFYING THE ARM BUILDER — and a real bug in the FLOP counter

### 3.1 What I could verify

Worktree `/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer`
**EXISTS**, on branch `agent/claude-01/liv-short-conv-mixer`, HEAD `c2aac8e` ("Add declarative arm
builder for the LIV study") on `83e4dce` ("Add ShortConv…") on upstream `f17824e`. Working tree
clean. MEASURED.

`src/olmo_core/nn/transformer/liv_arms.py` is 396 lines, 11 arms, and reads exactly as the HANDOFF
describes. I did NOT run it in the worktree (no local execution). Instead I scp'd it to FarmShare
and ran `arm_report()` against the `olmo_core` checkout at
`/scratch/users/ericrcwu/kda/olmo` (torch 2.11.0+cu128, login node, CPU/meta only). That checkout
already carries `short_conv.py` at **md5 `10429898813677b3f3581dce9e62b552`, byte-identical to the
worktree's** — so the mixer under test is the same code. MEASURED.

**MEASURED output of `arm_report()` on FarmShare:**

```
arm                 params    vs L0      flops@4K    vs L0     flops@32K    vs L0
---------------------------------------------------------------------------------
L0             354,483,968   1.000x 2,260,822,528   1.000x 4,374,751,744   1.000x
A16-P          354,388,992   1.000x 2,931,443,712   1.297x 8,568,588,288   1.959x
F-r128         338,755,328   0.956x 2,229,365,248   0.986x 4,343,294,464   0.993x
F-r256         343,998,208   0.970x 2,239,851,008   0.991x 4,353,780,224   0.995x
G-grouped      338,755,328   0.956x 2,229,365,248   0.986x 4,343,294,464   0.993x
N-narrow       338,804,528   0.956x 2,182,119,904   0.965x 4,296,049,120   0.982x
W-k5           354,504,448   1.000x 2,260,863,488   1.000x 4,374,792,704   1.000x
W-k9           354,545,408   1.000x 2,260,945,408   1.000x 4,374,874,624   1.000x
W-k15          354,606,848   1.000x 2,261,068,288   1.000x 4,374,997,504   1.000x
A-fewer3       357,638,528   1.009x 2,078,392,576   0.919x 3,135,357,184   0.717x
Q-mqa          348,978,944   0.984x 2,227,792,384   0.985x 4,341,721,600   0.992x
```

| claim | verdict |
|---|---|
| `L0` = **354,483,968** | **VERIFIED** — matches HF safetensors AND my independent ledger |
| `A16-P` = 354,388,992, param-matched to 0.027% | **VERIFIED** |
| `A16-P` FLOP ratio **1.297× @4K** | **VERIFIED** (reproduced to the FLOP) |
| `A16-P` FLOP ratio **1.959× @32K** | **VERIFIED** (reproduced to the FLOP) |
| Every other row of the HANDOFF cost table | **VERIFIED** — all 6 rows match exactly |
| A16-P SwiGLU width 4,820 is *solved*, not guessed | **VERIFIED** by independent brute-force re-solve |

I reproduced the entire FLOP table **by hand in `topology_math.py`** without importing olmo_core,
using this reverse-engineered convention, and got `2,260,822,528 / 2,931,443,712 /
4,374,751,744 / 8,568,588,288` — **exact agreement on all four**. That is strong evidence I have
the counter's semantics right, which sets up the next finding.

### 3.2 What I could NOT verify

- **The 55 tests did not run.** The FarmShare checkout has `short_conv.py` but **not**
  `liv_arms.py`, and no `src/test/` for either; I imported `liv_arms.py` as a loose module. So the
  32 mixer tests + 23 arm tests (parity to 0.0 float64 vs `Lfm2ShortConv`, causality, receptive
  field, gradient flow, etc.) are **UNVERIFIED by me** — I am taking the HANDOFF's word.
- **Forward/backward for every arm** — not run (needs real device allocation; `arm_report` uses
  `init_device="meta"`).
- The parity-to-released-LFM2 claim — not re-run.

### 3.3 DISCREPANCY 4 — the ShortConv FLOP counter charges 2× params; every other module charges 6×

This is a real inconsistency in committed code, not a documentation nit.

`src/olmo_core/nn/attention/short_conv.py:361-376`:
```python
linear_flops = 2 * params          # <-- 2x
conv_flops   = 2 * self.kernel_size * d
gate_flops   = 2 * d
```
Every sibling uses **6×** with the comment *"6 FLOPs per parameter (2 ops * 3 for
forward+backward)"* — `feed_forward.py:206-209`, `lm_head.py:422-425`,
`attention/__init__.py:798-799`, `attention/__init__.py:1152-1153`. And the attention score term
uses **12×** ("2 matmuls * 2 ops each * 3 for forward+backward").

So `ShortConv.num_flops_per_token` returns a **forward-only** count while the rest of the model
returns **forward+backward**. Since `TransformerBlock.num_flops_per_token` simply adds
`self.attention.num_flops_per_token(seq_len) + self.feed_forward.num_flops_per_token(seq_len)`
(`block.py:226-229`), the totals mix conventions. The docstring says "FLOPs per token" without
stating which, so this is plausibly an oversight rather than a deliberate choice — and it is
undetectable by the committed tests, which only assert *ratios between arms that all contain the
same number of ShortConv layers*. `A16-P` has **zero** ShortConv layers, so it is precisely the
arm the bug distorts.

**Direction and magnitude — MEASURED by recomputing both conventions by hand:**

| T | convention | L0 | A16-P | ratio |
|---:|---|---:|---:|---:|
| 4,096 | **olmo as-committed** | 2,260,822,528 | 2,931,443,712 | **1.297×** |
| 4,096 | ShortConv fixed to 6× | 2,428,758,528 | 2,931,443,712 | **1.207×** |
| 4,096 | 6× + causal attn score | 2,277,763,584 | 2,528,790,528 | **1.110×** |
| 32,768 | **olmo as-committed** | 4,374,751,744 | 8,568,588,288 | **1.959×** |
| 32,768 | ShortConv fixed to 6× | 4,542,687,744 | 8,568,588,288 | **1.886×** |
| 32,768 | 6× + causal attn score | 3,334,728,192 | 5,347,362,816 | **1.604×** |

Two separate issues stack:

1. **The 2× bug undercounts L0 by 167,936,000 FLOPs/token (7.4% at 4K, 3.8% at 32K)**, all of it
   in the arm that *has* the conv layers. It therefore **inflates the reported compute gap**:
   1.297× should be **1.207×**, and 1.959× should be **1.886×**. The headline "param-matched but
   1.96× the FLOPs" is overstated by ~4 points. **The qualitative conclusion survives — the gap
   is still large — but the specific numbers in the HANDOFF, in `liv_arms.py`'s docstring, and in
   the test that asserts the gap are all wrong by this amount.**
2. **The attention score term is non-causal.** `12 · n_heads · head_dim · seq_len` prices the full
   T×T score matrix; a causal model computes half of it. This is a *widespread* convention in
   MFU-reporting code (PaLM-style), so it is defensible as-is — but combined with fix (1) it moves
   the 32K gap from 1.959× to **1.604×**. If anyone tries to compute-match arms using these
   numbers, the two conventions must be nailed down first.

> **RECOMMENDED CORRECTION.** Change `linear_flops = 2 * params` → `6 * params` and
> `conv_flops`/`gate_flops` to 6× in `short_conv.py:373-375`, then re-run
> `solve_*` and re-state the gap as **1.21× @4K / 1.89× @32K**. Add a test that asserts
> `ShortConv.num_flops_per_token(T) ≈ 6 × ShortConv.num_params` (within the conv/gate epsilon), so
> the convention is pinned. This is a ~4-line fix and it changes a number quoted in three
> documents. **Do it before any compute-matched arm is solved**, because a compute-matched
> `A16-P` solved against the buggy counter would be mis-sized by ~7%.

---

## §4. SANITY-CHECKING THE FRAMING — does the traffic argument survive P1's bytes≠time lesson?

### 4.1 What P1 actually proved

MEASURED (FarmShare job 1670884, L40S, CUDA-graphed, 3 trials, ≤0.3% spread), per HANDOFF:

| arm | MiB/tok | latency | achieved BW |
|---|---:|---:|---:|
| `dense` (stock LIV) | 40.0 | 56.2 µs | **695 GB/s** (80% of L40S peak) |
| `lowrank_fused` r=128 | 10.0 | 60.8 µs | **161 GB/s** |

**4× fewer bytes, 8.2% MORE time.** The roofline model predicted a 6.3% win and got the sign wrong.
The mechanism, per the iso-byte control, is that a skinny `d→r` GEMV cannot saturate the memory
system: bytes are a *necessary* but not *sufficient* proxy for time.

### 4.2 Does the same failure mode apply to the topology argument? — **Largely NO, and here is why**

This is the key analytic question of my assignment, and the answer is more favourable than the
P1 result superficially suggests. **The two situations differ in the one variable that caused P1's
failure: whether the byte reduction changes the SHAPE of the memory access.** INFERRED throughout
this subsection.

| | P1 (failed) | Topology L0 vs A16-P |
|---|---|---|
| what changed | one `d×d` GEMV → two `d×r` GEMVs | **10 layers' KV reads deleted entirely** |
| kernel count | 20 → 30 (measured) | **unchanged or fewer** (fewer attention kernels) |
| access pattern | narrow, latency-bound GEMV | **large contiguous KV-cache streaming read** |
| bytes removed | from a *weight* GEMV that was already 80% efficient | from a *cache* read that is a pure sequential stream |
| roofline validity | broke — 695→161 GB/s | should hold — nothing narrows |

The bytes L0 removes are `KV_A16 − KV_L0 = 20,480 B/token × T`. At T=4,096 that is **80 MiB per
decode step** of paged/contiguous cache reads. Those are the most bandwidth-friendly reads in the
whole model — FlashDecoding-style kernels routinely hit near-peak on them. Crucially, **L0 does
not make any remaining read narrower**; it deletes whole layers' worth. So the "skinny GEMV"
mechanism that killed P1 has no analogue here.

**BUT there are three real reasons the traffic win will still not convert 1:1 into latency:**

1. **L0 does not delete the layers — it replaces them with ShortConv layers that also cost time.**
   A16-P's 16 attention layers become L0's 6 attention + 10 ShortConv. Each ShortConv is
   `in_proj (d→3d)` + `depthwise conv` + `out_proj (d→d)` — that is **4d² = 4.20M weights read
   per layer per token** vs GQA's 3.00M. So at T→0, **L0 is the SLOWER arm on weight traffic**
   (L0's mixers total 10×4.197M + 6×3.146M = 60.85M vs A16-P's 16×3.146M = 50.33M). The
   parameter matching hides this by shrinking A16-P's FFN, but the *mixer* traffic asymmetry is
   real and the crossover at T≈4K is precisely the point where the KV saving overtakes it. The
   arithmetic in §2 already accounts for this via the total-parameter equality — I flag it because
   **it means the topology win is a near-cancellation of two larger effects**, which is exactly
   the regime where a 10% predicted effect can measure at 0%.
2. **Kernel-launch and per-layer fixed costs.** P1's iso-byte control MEASURED **~3.4 µs per extra
   kernel even under CUDA graphs** on L40S. L0 has 10 ShortConv layers; if the mixer's decode path
   costs even 2 extra kernels per layer over a fused attention decode, that is 20 kernels ×
   3.4 µs = **68 µs/token** — which at 350M bf16 (708.97 MB weight read ÷ 695 GB/s ≈ **1.02 ms**
   for the weight stream alone) is ~6.7% of the step, i.e. **the same order as the entire 9.93%
   effect.** INFERRED, and it is the single biggest threat to the claim.
3. **Batch=1 vs batched.** At batch 1 the KV read is `20,480·T` bytes; at batch B it is
   `B·20,480·T` while the weight read stays fixed at 709 MB. **So the topology advantage grows
   with batch size** — at B=8, T=4,096, the KV saving is 640 MiB against a 709 MB weight read,
   nearly a 2× total-traffic win. Conversely a batch-1 edge deployment is the *worst* case for
   this claim while being the case the "device" narrative rests on. This should be stated.

### 4.3 Is a traffic claim without a latency measurement defensible? — **Yes, if and only if it is labelled as one**

My verdict: **a decode-traffic claim is defensible, publishable, and honest — but ONLY under three
conditions**, all of which this project is in a position to satisfy:

1. **It must be named "decode memory traffic," never "decode latency" or "speedup."** These are
   different quantities and this project has its own measurement proving they diverge by more than
   100% of the effect size. Using "traffic" is not a hedge; it is the correct noun for what is
   computed.
2. **It must be reported alongside P1's counterexample.** The strongest version of this write-up
   says: *"We compute a 20 KiB/token traffic reduction and we explicitly decline to convert it to
   a latency claim, because in this same study a 4× byte reduction measured 8.2% slower."* That is
   a **better** result than an unqualified speedup claim — it demonstrates measurement discipline
   and it is unfalsifiable-by-hardware in the good sense (the arithmetic is exact; the latency is
   device-specific).
3. **It must state the arithmetic-intensity regime.** `07_latency_kernels.md:309-316` establishes
   decode AI ≈ 1 FLOP/B against an A100 ridge of ~200 FLOP/B — 200× below. The claim "in the
   bandwidth-bound regime, traffic bounds latency from below" is true and is the right formal
   statement: **traffic gives a CEILING on the achievable speedup, not a prediction.** L0's ceiling
   at 4K is 9.93%; the achieved number will be somewhere in [0%, 9.93%] and is likely near the
   bottom of that range for the reasons in §4.2.

> **Sharpest framing available: "The topology buys a 9.93% decode-traffic ceiling at 4K and
> 37.64% at 32K. We do not claim the latency; this study's own P1 measurement shows a 4× byte
> reduction can be 8.2% slower."** That sentence is true, quantified, and cannot be attacked.

### 4.4 What it would take to MEASURE the actual latency difference

**HARD BLOCKER, confirmed by reading the code: `ShortConv` has NO incremental decode path.**
MEASURED — I grepped `short_conv.py` for `conv_state` / `decode` / cache handling: the only cache
reference is `cache: Optional[BufferCache] = None` at line 450 with `del layer_idx, n_layers, cache
# Unused` at line 461. There is no `conv_state` buffer, no step function, no `init_kv_cache`
analogue. This matches the HANDOFF's "Remaining Phase 0 items". **L0 cannot be decoded today.**

Work required, in dependency order (all INFERRED effort estimates):

| # | item | effort | why it is load-bearing |
|---|---|---|---|
| 1 | `ShortConv` conv-state cache: `[B, d, k]` ring buffer + a step path | 1-2 days | without it there is no L0 decode at all |
| 2 | Correctness test for the step path vs prefill | 0.5 day | **the HANDOFF flags that HF's own LFM2 decode reportedly drops a tap via `conv_state.roll(-1)`** — do not copy it; test against prefill |
| 3 | Wire ShortConv into `generation_module.py`'s cache lifecycle | 0.5-1 day | the assertions were relaxed but nothing allocates conv state |
| 4 | CUDA-graphed batch-1 decode harness at fixed T (reuse `p1_verify.py`'s protocol) | 0.5 day | the P1 harness already exists and is validated to ≤0.3% spread |
| 5 | Run L0 vs A16-P at T ∈ {512, 1K, 2K, 4K, 8K, 16K, 32K}, both arms, locked clocks | ~2 GPU-h | the actual measurement |
| 6 | `ncu` bytes-moved verification that the arms move the predicted bytes | 0.5 day | closes the traffic→latency loop; this is what makes the result citable |

**Total: ~4-5 engineer-days + ~2 GPU-hours.** That is cheap. **The measurement does not require
trained models** — decode latency at fixed T is a function of shapes only, so both arms can be
randomly initialised. This is the single highest-value-per-hour item in the whole project and it
is *not* on the critical path of any training run.

> **RECOMMENDATION: do item 1-6 BEFORE committing any training compute.** If L0 measures 0%
> or negative against A16-P (entirely possible per §4.2), the topology story changes from "a
> traffic win" to "a traffic win that does not convert" — which is still a result, but it
> reshapes the whole framing, and you want to know that before you spend GPU-hours.

---

## §5. ADVERSARIAL NOVELTY CHECK — is the topology result already in the literature?

**Tooling note:** `WebSearch` was unavailable this session (HTTP 403, model-access error on the
search backend). I substituted the **arXiv API** (`export.arxiv.org/api/query`) and direct
`WebFetch` of arXiv abstract/HTML pages, plus the repo's own
`06_baselines_infra.md` §1.1/§2.1 which contains a 20-row primary-source table of published ratio
ablations with paper IDs, scales, and numbers. Every citation below traces to an arXiv ID.

### 5.1 The general claim IS well established — decisively so

**"A mostly-linear/conv hybrid matches or beats an all-attention model at equal parameters while
using far less KV cache" is one of the most reproduced findings in the 2024-25 architecture
literature.** Six independent labs, five scales, consistent direction:

| paper | arXiv | scale of ablation | ratios tested | result |
|---|---|---|---|---|
| **Mamba-2** | 2405.21060 Tbl 2 | 350M, 48L, 7B tokens | 0,1,2,3,4,5,6,7,9,11,15,24 of 48 (0-50%) | **best 6/48 = 12.5%**, ppl 8.26; *"around a 10% ratio of attention layers performs best"*; hybrid beats **both** pure endpoints |
| **Waleffe et al. (Mamba-2-Hybrid)** | 2406.07887 Fig 4 | 130M, confirmed at 840M | swept attention % | **~8% optimal**; *"minimized when roughly 8% of the layers are self-attention"* |
| **Jamba** | 2403.19887 Tbl 4/5 | 1.3B, 250B tokens; 7B, 50B | a:m 1:3 (25%) vs 1:7 (12.5%) vs pure | **tie between ratios**, both beat pure attention and pure Mamba |
| **MAD** | 2403.17844 Fig 4.2 | IsoFLOP 70M-7B | 0, 8.3, 25, 50, 100% | **25% compute-optimal across all IsoFLOP groups** |
| **Samba** | 2406.07522 Tbl 6 | 438M, 20B tokens | 0/1/2 full-attn layers | 0 full + SWA best; 1 full-attn **explodes at 16K** |
| **Falcon-H1** | 2507.22448 §2.1 | 1.2B, 60L, 70GT, near-constant params | **21 channel partitions**, α_A 1/8..6/8 | **α_A = 1/8**; *"having more attention channels significantly degrades the performance"* |
| **Hymba** | 2411.13676 Tbl 1/10 | 300M, 100B tokens | all-SWA → +global layers | 3 global layers suffice; all-SWA costs **>20.75 recall points** |
| **MiniMax-01** | 2501.08313 | production + scaling law | 1 softmax per 8 | 12.5% |

Plus production adoptions with no published ablation: Nemotron-H (7.7%), Granite 4.0 (~10%),
Qwen3-Next (25%), Kimi Linear (25%).

> **VERDICT ON THE GENERAL CLAIM: NOT NOVEL. Emphatically so.** If the write-up says
> "mostly-conv hybrid ≈ all-attention at equal params with less KV," a reviewer will cite four
> papers from 2024 that said it first, at larger scale, with more ratios. **Do not lead with this.**
> The KV-reduction half is even more established — Hymba alone advertises an **11.67× cache
> reduction** and 3.49× throughput vs Llama-3.2-3B (MEASURED from the abstract page).

### 5.2 BUT the specific claim has a real, narrow, defensible gap

The distinction the task brief asked me to draw is genuine, and it holds on three axes
simultaneously. All three must hold — any one alone is not enough.

**Axis 1 — the MIXER. Every published ratio ablation uses a large-state SSM or linear attention.
None uses a k=3 gated short conv.**

This matters mechanistically, not just taxonomically. Mamba-2 carries `d_state=128` per head of
recurrent state; a k=3 depthwise conv carries **2 tokens** of history per channel. The whole
"how much attention do you need" question is *about* how much the linear mixer can substitute for
attention, and these two mixers differ in state capacity by ~2 orders of magnitude. This is exactly
the hypothesis LFM2's own 37.5% attention fraction implies: **every SSM hybrid that ran a sweep
converged on 7-25%; LFM2, the only short-conv hybrid, ships 37.5%.** That gap is either (a)
necessary because a k=3 conv has almost no state, or (b) an artifact of hardware-first search.
**Nobody has distinguished (a) from (b), and that IS a publishable question.** VERIFIED from
`06_baselines_infra.md` §1.1's primary-source table.

**Axis 2 — the ENDPOINT. Recall is systematically missing from the ratio literature.**

The ratio ablations report perplexity and averaged downstream accuracy. Hymba is the *only* one
that reports a recall benchmark (and it is the one that found the biggest effect: 19.23 vs 39.98,
a **20.75-point** gap that would be invisible in a perplexity sweep — Mamba-2's whole 2-11 block
range spans **0.06 ppl**). So the literature has measured the ratio question with an instrument
that is 300× less sensitive than the one that shows the effect. **A ratio sweep with MQAR /
needle / passkey as primary endpoints is a methodological contribution independent of the mixer.**

**Axis 3 — LFM2 SPECIFICALLY published nothing. CONFIRMED against the actual paper.**

I fetched `https://arxiv.org/html/2511.23404v1` (LFM2 Technical Report, 2025-11-28) and checked
directly. MEASURED:

- **Conv:attention ratio ablation — ABSENT.** Only prose: the search *"repeatedly selects a
  minimal hybrid architecture where most blocks are inexpensive gated short convolution blocks."*
  Table 1 lists final attention-block counts (6 for 350M/700M/1.2B/8B-A1B, 8 for 2.6B) as
  hyperparameters, not as a sweep.
- **Kernel-width ablation — ABSENT.** The search space says *"gated short convolution blocks with
  varying kernel sizes"* but reports **no results**; Table 1 pins k=3 everywhere.
- **Recall/retrieval benchmarks — ABSENT.** No NIAH, passkey, MQAR, or RULER anywhere. Reported
  benchmarks (Tables 6-7) are MMLU, MMLU-Pro, GPQA, IFEval, IFBench, Multi-IF, GSM8K, GSMPlus,
  MATH 500/Lvl5, MMMLU, MGSM. Notably the paper *cites* the RNN/SSM recall-limitation literature
  (Wen et al. 2025; Arora et al. 2024b; Park et al. 2024) and then does not test for it.
- **Parameter-matched all-attention baseline — ABSENT as data.** Only the prose claim that selected
  hybrids *"match or exceed the aggregate quality of attention-heavier and mixed baselines at the
  same budget"* with *"lower peak RSS at long context (4K/32K)"* — **no numbers, no model cards,
  no table.** All published comparisons (Tables 2/3/6/7) are against *external released models*
  (Qwen3, Gemma 3, Llama 3.2, Granite 4.0, SmolLM3), not author-trained controls.

> **This is the single strongest sentence available to the write-up, and it is now verified against
> the primary source rather than inferred from a blog post: Liquid explicitly claims their hybrid
> matches attention-heavier baselines at the same budget, and publishes no data supporting it, no
> ratio sweep, no kernel-width sweep, and no retrieval benchmark — while citing the very papers
> that establish short-state mixers fail at retrieval.**

### 5.3 DISCREPANCY 5 — the repo contradicts itself about whether an LFM2 paper exists

`06_baselines_infra.md:118` states: *"No LFM2 paper exists (only a blog post…)"* and the §1.1
master table's LFM2 row says *"(no paper)"*. But `01_lfm2_architecture.md:7` correctly cites
**arXiv:2511.23404v1** and uses a `[PAPER]` tag 20+ times against it. **The paper exists** (I
fetched it). `06` is wrong and its claim is repeated in the §2.1 consensus table.

This is not cosmetic. **If the write-up says "no LFM2 paper exists" it will be desk-rejected on
a factual error, and the reviewer will assume the rest of the novelty analysis is equally
careless.** The *correct* and equally strong statement is: *"The LFM2 technical report exists and
reports no ratio ablation, no kernel-width ablation, and no retrieval benchmark; its
architecture-quality claims are prose without supporting tables."* **Correction required in
`06_baselines_infra.md` lines 118 and the §1.1 table row, and anywhere `docs/` repeats it.**

Secondary correction: `01_lfm2_architecture.md` §8.1 establishes (from the paper's own §2.1) that
LFM2's ratio is **NOT** the output of STAR — Liquid explicitly disowns STAR's perplexity+cache
proxies as *"not transfer[ring] reliably."* Any framing of the form "Liquid's search picked 37.5%,
we test whether the search was right" must not attribute it to STAR.

### 5.4 FINAL NOVELTY VERDICT

| framing | novel? | verdict |
|---|---|---|
| "Mostly-conv hybrid ≈ all-attention at equal params, less KV" | **NO** | Established by Mamba-2, Jamba, Waleffe, MAD, Falcon-H1, MiniMax. Do not claim. |
| "A hybrid saves KV cache vs all-attention" | **NO** | Arithmetic, plus Hymba's 11.67× is already published. |
| "The optimal attention fraction is ~10-25%" | **NO** | Six labs, converged. |
| **"The conv:attention ratio ablated on a *k=3 gated short-conv* mixer, with *recall* endpoints, with seeds"** | **YES, narrowly** | No published ratio ablation uses a short-conv mixer; only Hymba reports recall, and Hymba is a *parallel-head* architecture, not a layer-ratio one. |
| **"LFM2's shipped 37.5% is 1.5-5× higher than every published optimum, and Liquid published no data justifying it"** | **YES** | Verified against arXiv:2511.23404 directly. This is the sharpest true claim available. |
| "L0 vs A16-P saves 20 KiB/token / 9.93% decode traffic at 4K" | **NO as a finding, YES as a design constraint** | This is arithmetic, not a measurement. It belongs in the setup, never in the results. |

> **THE HARD ADVERSARIAL ANSWER:** **the two-arm L0-vs-A16-P comparison, on its own, is a
> re-measurement of published work and is NOT worth GPU-hours.** Its result is predictable from
> Mamba-2 Table 2 (hybrid beats all-attention at equal params) and its traffic advantage is
> arithmetic that needs no training at all. **What is worth the compute is the RATIO SWEEP with
> RECALL endpoints** — where the 37.5%-vs-10% discrepancy makes the outcome genuinely uncertain
> and where the short-conv mixer makes the answer non-transferable from the SSM literature.
> A16-P is then not the contribution; it is the **right endpoint of the sweep**, i.e. one of
> several x-axis points, and its value is calibration, not novelty.

---

## §6. THE MINIMUM EXPERIMENT — is param-matched defensible, and what does it cost?

### 6.1 Is a param-matched, compute-mismatched comparison scientifically defensible?

**My verdict: it is defensible, but ONLY as one half of a pair, and the current 2-arm plan is
NOT sufficient.** Reasoning:

The compute gap is **1.21× at 4K and 1.89× at 32K** (my corrected numbers from §3.3; the
committed builder says 1.297/1.959). That is not a rounding error. Consider the two ways it can
land:

| outcome | param-matched reading | what a reviewer says |
|---|---|---|
| **L0 wins** | "conv topology beats attention at equal params" | **Strong and safe.** L0 won while using 21-89% *less* compute. Compute-matching would only widen the gap. |
| **A16-P wins** | "attention beats conv at equal params" | **Uninterpretable.** A16-P also got 1.21-1.89× the FLOPs. Did the topology win, or did the compute? Cannot tell. |
| **tie** | "topologies are equivalent at equal params" | **Weak-to-uninterpretable.** A tie at 1.89× the compute is actually a *win for L0* on a compute basis — but the param-matched framing cannot say so. |

**So a param-matched comparison is a one-sided test: it can only cleanly establish "L0 wins."**
Given the literature (§5.1) predicts hybrids beat all-attention at equal params, the *most likely*
outcome is the one this design can interpret — but that is luck, not method, and it means the
design has no defence if the result goes the other way.

**The compute gap is also not symmetric in the way people assume.** At the 4K training context
the arms differ by 21% FLOPs, which at fixed token budget means A16-P *costs 21% more GPU-hours*.
So a param-matched pair at fixed tokens is neither iso-param-iso-compute nor iso-cost. There are
three legitimate matching axes and they are mutually exclusive:

1. **iso-params, iso-tokens** (the current plan) — A16-P gets 1.21× the FLOPs.
2. **iso-FLOPs/token, iso-tokens** — shrink A16-P's width/FFN until FLOPs match; it then has
   *fewer parameters* than L0.
3. **iso-total-compute** — give A16-P 1/1.21 = 0.83× the tokens at matched params.

> **RECOMMENDATION: report axes 1 and 3, i.e. THREE ARMS, not two.** Add `A16-C`: same
> parameter-matched all-attention model, trained on **0.83× the tokens** so total training FLOPs
> match L0. Then:
> - `L0` vs `A16-P` answers *"at equal capacity, which topology is better?"*
> - `L0` vs `A16-C` answers *"at equal compute budget, which topology is better?"*
>
> These two questions have genuinely different answers in the literature (MAD's 25% is
> compute-optimal while Mamba-2's 12.5% is loss-optimal at fixed params — `06`'s §2.2 identifies
> exactly this as one of the field's real disagreements), and **reporting both is itself a
> contribution** because nobody in the ratio literature has separated them cleanly.
> A16-C costs **less** than A16-P (fewer tokens), so the third arm is nearly free.
>
> I do **not** recommend axis 2 (shrink A16-P to match FLOPs). It breaks the param match, which is
> the more legible axis, and it requires re-solving against the FLOP counter — which has a bug
> (§3.3) that must be fixed first.

### 6.2 Endpoints — perplexity cannot resolve this, and the docs are right about that

MEASURED from the literature (`06_baselines_infra.md` §2.1/§2.2): Mamba-2's entire 2-to-11
attention-block range (4-23% attention) spans **0.06 ppl**; Jamba's 1:3 vs 1:7 are
**indistinguishable**. The same architectural axis produces a **20.75-point** recall gap in Hymba
and a **35.3-point** IMDB gap in Jamba. **The effect is 300-580× larger on recall than on
perplexity.** Any design that puts held-out CE first is instrumenting the wrong variable.

**Primary endpoints, in priority order:**

1. **MQAR at the calibrated operating point.** The harness is already built and calibrated
   (FarmShare jobs 1670928/1670987; vocab 256, `N512_D64`, 8000 steps × batch 64). ⚠️ Per HANDOFF
   the operating point was calibrated at 4 layers/d=128 and **must be re-swept on real L0** —
   budget that. Report **success rate AND median accuracy vs the 1/D floor**, never a bare
   threshold.
2. **Needle/passkey at and beyond training length** (4K trained → test 4K/8K/16K/32K). This is
   where Samba Tbl 6 found a full-attention layer *explodes* at 16K — so it discriminates, and it
   is the endpoint most relevant to the KV/topology story.
3. **AR-Hits sliced perplexity** (associative-recall-token-sliced CE) — a continuous metric with
   recall sensitivity; gives a 2-18× SNR gain over accuracy per `05_evaluation.md` §8.
4. Held-out CE — **reported, never a gate.**

**Seeds.** `05_evaluation.md` §8.2 gives `n = 2·7.84·σ²/δ²`. For a δ=2pt effect on the good
benchmarks that is 1-3 seeds; the doc then correctly says to **inflate ~2× at sub-1B scale**, and
this repo's own KDA study measured a +8.92pp effect at n=3 collapse to +2.01pp (n.s.) at n=8.
**Use n=5 for screening, n=8 fresh paired seeds for the confirmation stage.** For MQAR
specifically the bimodality at low load means seeds behave like Bernoulli trials — 5 seeds gives a
usable success rate, 3 does not.

### 6.3 Cost — MEASURED arithmetic, on the hardware that actually exists

The task brief asked for 8×A100 @ 40% MFU using 6ND. I computed both, and flag that **6ND is the
wrong formula here**:

- `6N = 6 × 354,483,968 = 2.127 GFLOP/token`
- the model's own `num_flops_per_token(4096) = 2.261 GFLOP/token` (as-committed), **2.429** with
  the ShortConv fix.

6ND is 6.3% low because it prices the tied 67.1M embedding as a matmul on the input side (it is a
lookup) and misses the attention score term. **Use `num_flops_per_token`.** MEASURED:

| budget | tokens | GPU-h (8×A100, 40% MFU, 6ND) | GPU-h (using true fpt) |
|---|---:|---:|---:|
| Chinchilla 20× | 7.09B | **4.20** | 4.46 |
| 40× | 14.18B | **8.39** | 8.92 |
| 30B fixed | 30.0B | **17.75** | **18.87** (L0) / **24.47** (A16-P) |

**These numbers are small enough to be suspicious, so I sanity-checked on the hardware that
actually exists.** FarmShare is **L40S sm_89** (362 TFLOP/s dense bf16, ~864 GB/s), not A100:

| config | 30B tokens, L0 @4K | wall-clock |
|---|---:|---:|
| 1× L40S @ 35% MFU | 148.7 GPU-h | 148.7 h (6.2 days) |
| 4× L40S @ 35% MFU | 37.2 GPU-h | 9.3 h |
| 8× L40S @ 35% MFU | 18.6 GPU-h | 2.3 h |

⚠️ **35% MFU on L40S at 350M is optimistic** — L40S has no NVLink, and a 350M model at d=1024 has
small GEMMs. **Assume 20-25% MFU and inflate by 1.5×** for planning. The 8×A100 figure at 40% MFU
is also optimistic at this scale for the same reason.

### 6.4 CONCRETE MINIMUM EXPERIMENT

**Stage 0 — free, do first (no training).** Fix the FLOP counter (§3.3, ~4 lines). Build the
ShortConv decode path and measure L0 vs A16-P decode latency (§4.4): ~4-5 engineer-days + **2
GPU-hours**. This can invalidate the whole traffic framing before a single token is trained.

**Stage 1 — the ratio sweep, which is the actual contribution (§5.4).** Not 2 arms. Five points
on the x-axis, all param-matched by solved SwiGLU width, all at 4K/30B tokens:

| arm | attention layers | attn fraction | notes |
|---|---:|---:|---|
| `A0` | 0 | 0% | pure-conv floor — the arm that shows recall collapse |
| `A2` | 2 (5, 10) | 12.5% | **the published SSM optimum** — the key comparison point |
| `A3` (=`A-fewer3`) | 3 (5, 10, 14) | 18.75% | already declared in `liv_arms.py` |
| `L0` | 6 (2,5,8,10,12,14) | **37.5%** | **LFM2 as shipped** |
| `A16-P` | 16 | 100% | param-matched all-attention ceiling |
| `A16-C` | 16 | 100% | **iso-compute**: same model, 0.83× tokens |

**Budget:**

- 6 arms × 5 seeds = **30 runs** for screening.
- 30B tokens/run (≈85× Chinchilla — justified because recall emerges late and because
  under-training is indistinguishable from a hard task, per the MQAR job-1670963 failure).
- L0-equivalent cost ≈ 18.9 GPU-h/run on 8×A100@40%; A16-P is 1.30× that (24.5). Mean ≈ **21
  GPU-h/run** across the arm mix.
- **Screening total: 30 × 21 = ~630 GPU-hours** (8×A100@40% MFU, ideal).
- **Confirmation: 2 arms × 8 fresh paired seeds × 21 = ~336 GPU-h.**
- **Grand total ≈ 970 GPU-hours ≈ 1,000 GPU-h.** With a realistic 25% MFU derate: **~1,550
  GPU-hours**. On 8×A100 that is ~8 wall-days of continuous 8-GPU occupancy.

**Cheaper honest alternative if that is over budget:** cut to 4 arms (`A0`, `A2`, `L0`, `A16-P`),
3 screening seeds, 15B tokens → **~250 GPU-h screening + 336 confirmation ≈ 590 GPU-h**. Losing
`A3` and `A16-C` costs the iso-compute axis, which weakens §6.1's recommendation — trade
deliberately, and if forced, **drop a seed before dropping the `A2` arm**: `A2` is the point where
the published optimum sits and is the whole reason the sweep is interesting.

### 6.5 What would show a difference, quantitatively

INFERRED prediction, stated in advance so the experiment is falsifiable:

| endpoint | expected L0 − A16-P | resolvable at n=5? |
|---|---|---|
| held-out CE | **≤ 0.01 nats** | **No** — needs n≈43 per this repo's KDA study |
| HellaSwag/PIQA avg | ≤ 1 pt | marginal (needs n≈3-10) |
| **MQAR `N512_D64` accuracy** | **10-40 pts** (A0 vs A16-P; L0 intermediate) | **Yes, easily** |
| **needle @ 4×train-length** | **20-60 pts** | **Yes** |
| decode traffic @4K | 9.93% (arithmetic, exact) | n/a — not measured, computed |

**If MQAR and needle both come back flat across `A0`→`A16-P`, the experiment has failed its
positive control and the results are uninterpretable** — the same failure mode as FarmShare job
1670922. Gate on `A0` showing a recall deficit before trusting any middle rung.

---

## §7. BOTTOM LINE — is the topology result the real contribution?

**Short answer: NO as stated, YES as a component.** The topology *traffic* claim is exact,
verified, and correct — and it is arithmetic, not a finding. It belongs in the setup section of
a paper, not the results section. What is worth compute is the **ratio sweep with recall
endpoints on a short-conv mixer**, of which `A16-P` is one x-axis point.

### Verification scorecard

| claim | verdict |
|---|---|
| KV = 12 KiB/token (L0), 32 KiB (A16-P), Δ = 20 KiB | **CONFIRMED exactly** |
| KV is 6.6% of decode traffic @4K, 36.2% @32K | **CONFIRMED** (6.63% / 36.22%) |
| KV read == weight read at T = 57,690 | **CONFIRMED** (57,696; docs rounded) |
| 10% decode-traffic win at T ≈ 4,121 | **CONFIRMED** for the docs' definition. The ratio-1.10 definition the brief asked for gives **3,693** |
| "KV bytes/token is scale-invariant" | **TRUE for d, FALSE in general** — breaks at 2.6B (16 KiB) and 24B (20 KiB) |
| "350M makes cache effects ~2.5× more visible" | **UNDERSTATED — the true factor is 3.30×** |
| `L0` = 354,483,968 | **CONFIRMED** vs HF safetensors and by independent ledger |
| A16-P = 354,388,992, matched to 0.027% | **CONFIRMED**; width 4,820 independently re-solved |
| A16-P FLOP ratios 1.297× @4K / 1.959× @32K | **Reproduced exactly — but both are WRONG** because `ShortConv.num_flops_per_token` uses a 2× (forward-only) multiplier where every sibling uses 6×. Corrected: **1.207× / 1.886×** |
| `crossover.py` computes the HANDOFF's numbers | **FALSE** — it is hard-coded to the 1.2B geometry |
| "No LFM2 paper exists" (`06_baselines_infra.md:118`) | **FALSE** — arXiv:2511.23404, fetched and read |

### Five corrections requiring edits

1. **`short_conv.py:373-375`** — `2 *` → `6 *`; re-state the compute gap as **1.21×/1.89×** in
   `liv_arms.py`'s docstring, the HANDOFF cost table, and `docs/liv-brainlift-experiment-design.md`.
   Add a test pinning the convention. **Do this before solving any compute-matched arm.**
2. **`crossover.py`** — re-parameterise to d=1024/F=4608/H=16 (or rename and add a 350M sibling).
3. **`06_baselines_infra.md:118` and §1.1 table** — remove "no LFM2 paper exists"; replace with
   "the technical report (arXiv:2511.23404) reports no ratio ablation, no kernel-width ablation,
   and no retrieval benchmark." This is a factual error that would sink credibility.
4. **HANDOFF Key Decision #1** — "~2.5× more visible" → **3.30×**; "scale-invariant" →
   "independent of d_model".
5. **Pick one definition of the 10% threshold** and name it. Recommend reporting a **range across
   deployment dtypes (T ≈ 1,030-4,121)** rather than a single bf16 number.

### The three sentences I would actually put in the paper

> *"L0 and A16-P are parameter-matched to 0.027% (354,483,968 vs 354,388,992), and at a 4K
> decode context L0 moves 9.93% less memory traffic — 12 KiB/token of KV cache against 32 KiB.
> We report this as traffic, not latency: elsewhere in this study a 4× byte reduction measured
> 8.2% slower on an L40S, so bytes bound latency from below but do not predict it. The arms are
> not compute-matched (A16-P uses 1.21× the FLOPs/token at 4K and 1.89× at 32K), so we also
> train an iso-compute variant."*

That paragraph is honest, quantified, and every number in it is verified above.
