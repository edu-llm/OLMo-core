# 07 — Claim 4 arithmetic verification (claims c, d, e)

Verification agent. All arithmetic executed on Stanford FarmShare login node `rice-03`
(`/scratch/users/ericrcwu/kda/venv/bin/python`). No code executed on the local Mac.
GPU reference for all bandwidth math: **NVIDIA L40S sm_89, 46 GB, L2 = 96.0 MiB, HBM peak ~864 GB/s**.

**STATUS: COMPLETE** (2026-08-01). No GPU jobs queued; login-node CPU arithmetic only.

Scripts written to FarmShare (persisted, re-runnable):
- `/scratch/users/ericrcwu/liv/verify/ledger.py` — parameter ledgers at d=1024 and d=2048
- `/scratch/users/ericrcwu/liv/verify/bw.py` — bandwidth unit derivation
- `/scratch/users/ericrcwu/liv/verify/bw2.py` — L2 vs HBM adjudication

Status legend: MEASURED (ran it / read from a stored artifact) / INFERRED (derived from stated
premises) / ASSUMED (premise I could not check).

---

## SUMMARY OF VERDICTS

| claim | verdict |
|---|---|
| (c) the "d=2048 contamination" — arithmetic | **CONFIRMED** (every correction reproduces exactly) |
| (c) …except the r=32 correction | 🔴 **REFUTED** — reassessment says 4.91%, true value is **5.55%** |
| (c) "error" vs "unlabelled scope" adjudication | **SPLIT** — §A.2 right about the memory note + HANDOFF; §C.1 right about the design doc |
| (d) "No LFM2 paper exists" is wrong | **CONFIRMED** (paper exists; contradiction in **3** places, not 1; gap claim survives with one caveat) |
| (e) 695 GB/s is GiB/s | **CONFIRMED** (unit bug located at `p1_verify.py:81` + `:87`) |
| (e) propagate "86.4% of peak"? | **REFUTED as a fix** — the whole comparison is moot (L2); delete rather than correct. Also: 86.3%, not 86.4% |

**Two new findings not in the reassessment:** (1) the reassessment's own r=32 correction is wrong
(4.91% → **5.55%**); (2) the repo now mixes **GiB/s** (`p1_verify.py`) and true **GB/s**
(`p1_scaled.py`, `p1_cache_check.py`) with identical `GB/s` labels, so any cross-job bandwidth
table built from both is silently corrupt — see §e.2.

---

## CLAIM (c) — the "d=2048 contamination"

### c.1 The frozen d=1024 ledger reproduces EXACTLY (MEASURED)

Geometry per `00_SYNTHESIS.md` §A.1: L=16, d=1024, V=65,536, **tied** embeddings, SwiGLU
ff=4,608, 10 LIV layers (4d² + kd, k=3), 6 GQA layers (hq=16, hkv=8, hd=64), RMSNorms +
per-head QK-norm of size head_dim.

```
$ ssh -S /tmp/farmshare-ericrcwu.sock ericrcwu@login.farmshare.stanford.edu \
    '/scratch/users/ericrcwu/kda/venv/bin/python /scratch/users/ericrcwu/liv/verify/ledger.py'

===== d=1024 / frozen L0 =====
   emb 67108864
   mlp 226492416
   liv 41973760
   gqa 18874368
   gqa_per 3145728
   norms 34560
   tot 354483968
  target 354483968 match: True
  FULL denom  : emb 18.93%  liv 11.84%  gqa 5.32%  mlp 63.89%
  NONEMB denom: liv 14.61%  gqa 6.57%  mlp 78.81%  (nonemb=287375104)
```

**The frozen total 354,483,968 reproduces to the parameter.** ✅ Independent of `00_SYNTHESIS.md`
— I rebuilt it from the geometry statement alone.

### c.2 The d=2048 / LFM2-1.2B ledger ALSO reproduces exactly (MEASURED)

```
===== d=2048 / LFM2-1.2B =====   (L=16, V=65536, ff=8192, 10 conv, 6 attn, hq=32, hkv=8, hd=64)
   emb 134217728
   mlp 805306368
   liv 167833600
   gqa 62914560
   gqa_per 10485760
   norms 68352
   tot 1170340608
  target 1170340608 match: True
  emb 11.47%  liv 14.34%  gqa 5.38%  mlp 68.81%
  gqa per-layer 10485760 = 2.500 d^2
```

**This settles the "is it really a d=2048 number?" question.** All three disputed shares
reproduce at d=2048 to the stated precision:

| disputed figure | d=2048 recompute | d=1024 recompute (full denom) | d=1024 (non-emb denom) |
|---|---:|---:|---:|
| "LIV mixer is 14.3% of the model" | **14.34%** ✅ | **11.84%** | 14.61% |
| "MLPs are 68.8%" | **68.81%** ✅ | **63.89%** | 78.81% |
| "GQA mixers 5.4%" | **5.38%** ✅ | 5.32% | 6.57% |
| "embeddings 11.5%" | **11.47%** ✅ | 18.93% | — |

**So the "these are d=2048 numbers" explanation is CORRECT, not itself an error.** All four
shares in the block quoted at `docs/liv-brainlift-experiment-design.md:498` reproduce at d=2048
to within 0.05pp and none of them reproduce at d=1024 under either denominator. The
`embeddings 11.5%` row is the tell — at d=1024 the tied embedding is 18.9% of the model, which
is 7.4pp off and unmistakable.

### c.3 Which denominator? (resolves the 11.8% vs 14.6% ambiguity)

The reassessment worried that "% of model" might mean non-embedding params, which would make the
d=1024 LIV share 14.6% and coincidentally near the 14.3% figure. **It does not.** Evidence:

1. The source table at `02_lowrank_gates.md:1084-1088` and `00_my_arithmetic_check.md:224-228`
   **lists `embeddings (tied) 134.218M 11.5%` as a row of the same table.** A table that assigns
   a percentage to the embeddings is by construction using the full-model denominator. The four
   shares sum to 11.5 + 68.8 + 5.4 + 14.3 = **100.0%**.
2. Under the non-embedding denominator at d=2048 the LIV share would be 16.20% and MLPs 77.72%
   (computed above) — neither matches.

**So the source docs unambiguously mean the FULL 354.5M denominator, and the correct frozen-
geometry figures are LIV = 11.84%, MLP = 63.89%.** The reassessment's 11.8% / 63.9% are right.
The 14.6% non-embedding coincidence is a red herring; report it only as a footnote.

### c.4 P1 rank sweep at d=1024 (MEASURED)

P1 factorizes the two gate projections: `2d² → 2·(2dr) = 4dr` per LIV layer, 10 layers.

```
  rank sweep at d=1024:
   r=  32 gate    1310720 saved   19660800  full%=5.546  nonemb%=6.842  new_total=334823168
   r=  64 gate    2621440 saved   18350080  full%=5.177  nonemb%=6.385  new_total=336133888
   r= 128 gate    5242880 saved   15728640  full%=4.437  nonemb%=5.473  new_total=338755328
   r= 256 gate   10485760 saved   10485760  full%=2.958  nonemb%=3.649  new_total=343998208
   r= 512 gate   20971520 saved          0  full%=0.000  nonemb%=0.000  new_total=354483968
```

vs. the same sweep at d=2048:

```
   r=  32  saved 81264640 = 6.944% of model
   r=  64  saved 78643200 = 6.720%
   r= 128  saved 73400320 = 6.272%
   r= 256  saved 62914560 = 5.376%
   r= 512  saved 41943040 = 3.584%
   r=1024  saved        0 = 0.000%
```

**Adjudication of the specific claims:**

| claim | verdict |
|---|---|
| P1 at r=128 saves **4.44%** at frozen d=1024 | ✅ **CONFIRMED** — 4.437% |
| 6.27% is the d=2048 value | ✅ **CONFIRMED** — 6.272% exactly |
| P1 at r=32 saves **4.91%** at d=1024 | ❌ **REFUTED — the reassessment's own correction is WRONG.** The true value is **5.55%** (19,660,800 / 354,483,968 = 5.546%). See below. |
| 6.94% is the d=2048 r=32 value | ✅ CONFIRMED — 6.944% |
| **r=512 saves exactly zero at d=1024** | ✅ **CONFIRMED** — `4dr = 4·1024·512 = 2,097,152 = 2d²` exactly. Saving is 0 params, bit-for-bit. The general condition is `r ≥ d/2`. |
| new total at r=128 = 338,755,328 | ✅ matches the committed arm builder per §A.2 |

🔴 **NEW ERROR FOUND, in the reassessment itself.** `00_SYNTHESIS.md` §A.2's correction table
gives "savings saturate (r=128 → 6.27%, r=32 → 6.94%)" → "**4.44% / 4.91%**". The 4.44% is right;
**the 4.91% is not.** It appears to have been produced by scaling 6.94% by the same ratio as
6.27→4.44 (0.7077 × 6.94 = 4.91) rather than recomputing. The correct r=32 figure at d=1024 is
**5.55%**. This matters slightly for the narrative: at d=1024 the *spread* across the rank sweep
is 1.11pp (4.44 → 5.55), noticeably wider than the 0.67pp the docs advertise at d=2048, so
"savings saturate" is *less* true at the frozen geometry than the doc claims — though the
absolute saving is smaller. Anyone doing the correction pass should write 5.55%, not 4.91%.

### c.5 The GQA 2.5d² vs 3d² question (MEASURED)

```
  GQA at 2.5d^2 vs 3d^2, d=1024:
   2.5d^2 per-layer = 2621440.0  x6 = 15728640.0
   3.0d^2 per-layer = 3145728    x6 = 18874368
   shortfall total  = 3145728    per-layer 524288
```

✅ **CONFIRMED.** GQA at the frozen geometry is **exactly 3.000 d² per layer** (my ledger prints
`gqa_per 3145728 = 3.000 d^2`), and using 2.5d² undercounts the total by **exactly 3,145,728** —
the number §A.1 states. Mechanism: the coefficient is `2 + 2·(hkv·hd/d)`. At d=2048 with
hkv·hd = 512 = d/4 → 2.5d². At d=1024 with hkv·hd = 512 = **d/2** → 3.0d². Same absolute KV
width, half the model width, so the coefficient rises.

### c.6 file:line INVENTORY — where each wrong number lives, with an ERROR / SCOPE judgement

Judgement rule I applied: **ERROR** = the number is asserted about the *frozen d=1024 design*, or
appears in a doc/section whose stated subject is the frozen design, with no scale qualifier.
**SCOPE-OK** = the number appears in a section explicitly about LFM2-1.2B / d=2048, correctly.
**SCOPE-UNLABELLED** = correct at d=2048, in a doc that is nominally about the frozen design, with
no qualifier — i.e. right number, missing label, actively misleading.

#### 🔴 ERROR — must change

| file:line | text | why |
|---|---|---|
| `/Users/ericwu/.claude/projects/-Users-ericwu-Developer-Capstone-LLM/memory/liv-experiment-key-numbers.md:30-31` | "the LIV mixer is only **14.3%** of the model (MLPs are **68.8%**). A 44% mixer cut … is a **6.27% model cut**." | The memory file is titled *LIV experiment key numbers* and describes the frozen program. **No scale qualifier anywhere in the bullet.** And it *self-contradicts* at line 150 of the same file: "the decode ceiling at our chosen 350M/d=1024 geometry is **4.44%**, not 6.27%." One file, two mutually exclusive claims, 120 lines apart. → **ERROR.** Replace with 11.8% / 63.9% / 4.44%. |
| `…/memory/liv-experiment-key-numbers.md:179` | "Sweep {128,256,512}; savings saturate (r=128 → **6.27%**, r=32 → **6.94%**)." | Same file, same defect. → **ERROR.** Replace with 4.44% / **5.55%** (not 4.91%). Also note the sweep set `{128,256,512}` contains **r=512, which saves exactly zero at d=1024** — the arm is a no-op and line 144 of the same file already says so. The sweep spec and the correction contradict each other. |
| `HANDOFF.md:81` | "Every number in the brainlift is exact … LIV mixer 16,783,360 at d=2048/k=3; **GQA 10,485,760 (2.5d²)**; factorized r=128 = 9,443,328; 12 KiB/token KV" | Borderline but I judge **ERROR**. The bullet *does* say "at d=2048/k=3" for the first item, so the scale is nominally present — but this bullet is in HANDOFF's **"Verified sound"** section, which is the standing summary of what is true *for the current program*, and it mixes a scale-invariant fact (12 KiB/token KV) with d=2048-only facts under one "every number is exact" header. The specific coefficient `2.5d²` is **false at the frozen geometry (3d²)** and is the one an implementer would reuse. → mark the whole bullet "(1.2B reference arithmetic; at the frozen d=1024 arm GQA is **3d² = 3,145,728/layer**)". |

#### 🟡 SCOPE-UNLABELLED — misleading, should be labelled

| file:line | text | judgement |
|---|---|---|
| `docs/liv-brainlift-experiment-design.md:498` | "embeddings 11.5% \| 10 LIV mixers **14.3%** \| 6 GQA mixers 5.4% \| 16 MLPs **68.8%**" | The block sits under §5.1 with the header sentence "**The mixer is only 14% of the model**". Eleven lines later (line 510) the doc *does* give the correct d=1024 4.44% in an explicit two-row table. So the doc contains its own correction — but the ledger block itself carries **no scale label** and is the part that gets quoted. → **SCOPE-UNLABELLED.** Add "(at d=2048/1.2B; at the frozen d=1024 arm: emb 18.9% / LIV 11.8% / GQA 5.3% / MLP 63.9%)". |
| `docs/liv-brainlift-experiment-design.md:661` | "Parameter savings saturate almost immediately (r=128 saves **6.27%**, r=32 saves **6.94%** …)" | §5.1 again, no qualifier here, 150 lines after the doc's own d=1024 correction table. → **SCOPE-UNLABELLED, trending ERROR.** Correct pair at frozen geometry: **4.44% / 5.55%**. |
| `docs/liv-brainlift-experiment-design.md:656` | "the gates are only ~8.1% of FLOPs (SwiGLU MLPs are **68.8%** of parameters)" | Same. → SCOPE-UNLABELLED. |
| `docs/liv-brainlift-experiment-design.md:113` | "GQA mixer = 2.5d² = **10.486M** \| 10,485,760 ✓" | Line 120 immediately says "The 2.5d² GQA coefficient holds *because* hkv=8 at d=2048 … which becomes **3.0d² at d=1024**. Restate for whichever scale is chosen." → **NOT AN ERROR.** Correctly labelled and correctly caveated seven lines later. This is the one place the repo gets it fully right. |

#### 🟢 SCOPE-OK — correct in context, leave alone

| file:line | note |
|---|---|
| `00_my_arithmetic_check.md:13, 22, 225, 227, 238, 301` | The doc's own subject is LFM2-1.2B (d=2048); line 225 sits inside a ledger whose total is `1,170.3M`, printed right there. Correct. |
| `01_lfm2_architecture.md:288, 608, 667, 672, 679, 685, 695, 699` | All explicitly d=2048; line 685-687 even derives the 3.0d² d=1024 case correctly. Best-labelled in the repo. |
| `02_lowrank_gates.md:46, 1084-1088, 1107-1109, 2035, 2601, 2754` | The `1084` table prints `total 1170.341M`; `2035` says "805M of 1170M" inline. Scale is on the page. Correct. |
| `06_baselines_infra.md:1791` | Table row keyed `2048` in the first column. Correct. |
| `06_baselines_infra.md:748` | "14.3%" here is `4/28 layers` for Mamba-2-Hybrid — an unrelated coincidence, not this number at all. **False positive of the grep.** |
| `07_latency_kernels.md:582, 623, 739` | These are the **model citizens**: they print both, e.g. "`saving as share of model weight read` \| **6.27 %** \| **4.44 %**" and "ceiling **6.27 %** end-to-end (d=2048) / **4.44 %** (d=1024)". Explicitly dual-scaled. Leave alone. |
| `07_latency_kernels.md:587, 604, 670, 1191, 1505, 1514, 1609` | Bare "6.27%" but inside the same document whose §-header table already established the dual scale. → mild SCOPE-UNLABELLED; low priority. |

#### The `4.72 µs` sibling (same error class, already fixed)

| file:line | status |
|---|---|
| `docs/liv-brainlift-experiment-design.md:526`, `:1439` | ✅ Already corrected in-place with an explicit dated `⚠️ CORRECTION (2026-07-31)` block. |
| `HANDOFF.md:216`, `…/memory/liv-experiment-key-numbers.md:144`, `…/memory/liv-experiment-frozen-decisions.md:39` | ✅ All three carry the correction. |
| `docs/liv-brainlift-experiment-design.md:618`, `07_latency_kernels.md:652, 660, 742, 1188` | 🟡 Still bare 4.72 µs. `:618` is inside the table the `:526` correction banner explicitly disclaims, so it is covered. The `07_latency_kernels.md` hits are in a d=2048-era doc. Low priority. |

**Conclusion on the 4.72 µs precedent:** the team *did* sweep this one properly — every
standing-summary document got the correction. That is exactly the treatment the 6.27% family
did **not** get, which is the real finding here: it is not that they didn't know, it is that
**they fixed one instance of the error class and not the other four.**

### c.7 ADJUDICATION — §A.2 "ERROR" vs §C.1 "mislabel, not an error"

§A.2: *"ERROR FOUND (systematic) … three other bullets still quote the superseded d=2048 figures
as if they applied to the frozen geometry."*
§C.1: *"6.27% vs 4.44% is NOT an error — it is a mislabel. 6.27% is correct at d=2048, 4.44% at
the frozen d=1024. Both HANDOFF and I are right; `docs/…:661` and the memory note quote the
d=2048 figure without saying so."*

**They are arguing about different files and both are half right. The correct verdict is SPLIT:**

- **§C.1 is right about the arithmetic and about the design doc.** 6.27% *is* exact at d=2048
  (I get 6.272%). Nobody miscalculated anything. And `docs/…:498/:661` sit in a document that
  states the d=1024 correction explicitly at `:510` — a reader of the whole document is not
  deceived. For the design doc, "mislabel" is the right word.
- **§A.2 is right about the memory note, and §C.1 is wrong to generalise from the design doc to
  it.** `liv-experiment-key-numbers.md` is not a d=2048 document with a missing label; it is the
  *frozen-program* summary and it asserts **both** "6.27% model cut" (line 31) and "4.44%, not
  6.27%" (line 150). A document that states P and ¬P is not mislabelled, it is **wrong** —
  whichever bullet a future agent reads first determines what it believes. §C.1's own framing
  ("both HANDOFF and I are right") cannot apply to a single file that contradicts itself.
- **A distinction neither section drew, and it is the one that decides the case:** the test is
  not *"is the number right at some d?"* but *"can a reader who reads only this passage recover
  the right number for the frozen design?"* By that test `docs/…:113`→`:120` passes (explicit
  caveat), `docs/…:498`→`:510` passes weakly (correction 12 lines down), `07_latency_kernels.md`
  passes (dual columns), and **`liv-experiment-key-numbers.md:30-31` and `:179` fail outright.**

**Practical resolution for the writeup:** do not use the word "error" for the class; say
*"the d=2048 figures propagated into the frozen-design summaries without a scale label, and in
the memory note they sit alongside the correction, contradicting it."* Then fix the three ERROR
rows and label the four SCOPE-UNLABELLED rows. **And fix the reassessment's own 4.91% → 5.55%.**

### 🔵 VERDICT (c): **CONFIRMED** — with one correction to the correction

Every substantive claim in the reassessment's table verifies: 4.44%, 11.8%, 63.9%, the 6.27%/
14.3%/68.8% figures are genuine d=2048 values (they reproduce there to 0.05pp), GQA is 3d² at the
frozen geometry, the 2.5d² ledger misses by exactly 3,145,728, r=512 saves exactly zero, and the
denominator is the full 354.5M. **One exception: the claimed r=32 correction of 4.91% is wrong;
it is 5.55%.** The "error vs unlabelled scope" dispute is **SPLIT** as adjudicated above.

---

## CLAIM (d) — "No LFM2 paper exists" vs arXiv:2511.23404

### d.1 The contradiction is real, and it is in TWO places in `06`, not one (MEASURED)

`Brainlifts/liv_experiment_research/06_baselines_infra.md:118`, verbatim:

> `- No LFM2 paper exists (only a blog post: https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models). There is **no published ratio ablation** for the conv:attention split, no published optimizer/LR/schedule, no published data mixture beyond "10T tokens", and no published intermediate checkpoints.`

The reassessment cites only this line. **I found two more instances it missed:**

| file:line | text |
|---|---|
| `06_baselines_infra.md:118` | "No LFM2 paper exists (only a blog post…)" — under header `### 0.3 What is *not* published about LFM2 [UNKNOWN — flag clearly]` |
| `06_baselines_infra.md:157` | master architecture table, row label **`| **LFM2** (no paper) | 2025 | 10 gated-short-conv + 6 GQA (16L) | …`** |
| `06_baselines_infra.md:1946` | risk register item 7: "**No LFM2 paper exists** — no published ratio ablation, recipe, or data mix. **Everything about LFM2 in this document comes from `config.json`, the HF model card, and the transformers implementation.**" |

Against `01_lfm2_architecture.md:7`:

> `- \`[PAPER]\` — stated in the LFM2 Technical Report, arXiv:2511.23404v1`

and `:14-15` listing both `https://arxiv.org/abs/2511.23404` and `https://arxiv.org/html/2511.23404v1`
as primary sources. `01` uses `[PAPER]` throughout and even reports having grepped the extracted
full text (`:26`, `:1097`).

✅ **The self-contradiction is CONFIRMED, and it is worse than reported: three sites in `06`, not
one.** Note the two docs have the *same research date* (2026-07-30) — so this is not one doc being
stale relative to the other; it is two parallel agents that never reconciled. `06`'s line 1946 is
the damaging one because it frames the non-existence of the paper as *the opportunity*: "That is
also the opportunity: the ablation Liquid never published is the contribution." That framing
survives (see d.3) but its stated premise does not.

Downstream: `docs/liv-brainlift-experiment-design.md:1457` cites arXiv:2511.23404 in its reference
list, so the design doc is on the right side. `05_evaluation.md:524` also cites it correctly. Only
`06` is wrong.

### d.2 The paper exists (MEASURED — fetched 2026-08-01)

`https://arxiv.org/abs/2511.23404` resolves.

- **Title:** *LFM2 Technical Report*
- **Submitted:** 28 November 2025 (v1), cs.LG / cs.AI, ~17.2 MB
- **Authors:** 33, including Alexander Amini, Ramin Hasani, Mathias Lechner, Daniela Rus,
  Maxime Labonne, Jimmy T.H. Smith, Rom N. Parnichkun, Tim Seyde
- **Subject:** yes, LFM2 — "a family of Liquid Foundation Models" for on-device use; hybrid
  backbone of gated short convolutions + a few GQA blocks; 350M/700M/1.2B/2.6B dense + 8.3B-A1B
  MoE; 32K context; 10–12T pretraining tokens.

**On the future-dated-ID caution in the brief:** taken literally, `2511` = **November 2025**, which
is 8 months in the past as of today. There is no future-dating problem with this ID and no reason to
suspect a typo. (The convention warning applies to the repo's `26xx` citations, e.g. the 2606.06467
paper cited in §C.2 — not to this one.)

### d.3 Does reading the paper UPGRADE the gap claim? — ablation inventory (MEASURED, with one caveat)

I fetched the HTML and enumerated the tables. Full table list (16 tables, 6 figures):

| # | caption / content | relevant? |
|---|---|---|
| 1 | "LFM2 model hyperparameters." — includes **`Attn. Blocks`** and **`Conv k`** as *columns* | ⚠️ final values only |
| 2 | Prefill/decode throughput, Galaxy S25 / Snapdragon 8 Elite, batch 1 | no |
| 3 | Prefill/decode throughput, AMD Ryzen HX 370 CPU, batch 1 | no |
| 4a/4b | SFT data mixture composition | no |
| 5 | Hyperparameters for direct alignment | no |
| 6 | "Performance of tiny language models (350M–2B)" — MMLU, MMLU-Pro, GPQA, IFEval, IFBench, Multi-IF, GSM8K, GSMPlus, MATH 500, MATH Lvl 5, MMMLU, MGSM | no recall |
| 7 | Same for "small language models (2B–8B)" | no recall |
| 8, 9 | VLM benchmarks (<2B; 2–4B) | no |
| 10 | LFM2-Audio dataset composition | no |
| 11 | VoiceBench | no |
| 12 | ASR word error rates | no |
| 13 | Cross-lingual retrieval NDCG@10, **LFM2-ColBERT-350M** | ⚠️ see caveat |
| 14 | Cross-lingual retrieval, GTE-ModernColBERT-v1 baseline | ⚠️ |
| 15 | **Per-task NanoBEIR breakdown** | ⚠️ |
| 16 | Per-language multilingual VLM scores (Appendix D) | no |

Figures 1–6: portfolio, architecture diagram, post-training pipeline, quality-vs-throughput Pareto
frontier, VL architecture, VL token-budget accuracy. **No ablation figure.**

**Answers to the three load-bearing questions:**

**(i) Attention:conv RATIO ablation — NO.** ✅ gap claim holds. §2.1 describes a
hardware-in-the-loop Pareto search over quality (internal 50+ eval suite) × latency × peak memory,
whose space includes "*Layout: interleaving patterns of local context blocks, global context
blocks, position-wise blocks, and overall block counts under fixed parameter budgets.*" The
**outcome is reported only as prose** — the search "repeatedly selects a minimal hybrid" — and
Table 1 lists the *final* counts (6/16 for 350M, 700M, 1.2B; 8/30 for 2.6B; 6/24 for 8B-A1B) as
hyperparameters. **No per-ratio numbers, no sweep, no plot.** One nuance the repo has not exploited:
Table 1 itself shows the attention *fraction* falling with depth (37.5% → 26.7% → 25.0%), which is
an unexplained inconsistency in Liquid's own family and is free ammunition for the ratio question.

**(ii) Kernel-WIDTH ablation — NO.** ✅ gap claim holds. The search space explicitly says "gated
short convolution blocks with **varying kernel sizes**" — so they searched it — but **no results are
reported**, and Table 1 pins **k=3 for every model in the family, at every scale, dense and MoE.**
This is the strongest form of the gap: they admit running the sweep and publish none of it.

**(iii) Retrieval / recall / needle / RULER benchmark — NO for the LLM, but ⚠️ WITH A CAVEAT the
reassessment does not state.** Tables 6–7 (the LLM tables) contain **zero** long-context or recall
benchmarks — no RULER, NIAH, passkey, LongBench, MQAR, phonebook — despite a 32K context claim.
**However, Tables 13–15 DO report retrieval numbers (NanoBEIR, NDCG@10, 13 tasks).** They are not a
counterexample, and here is exactly why, which the writeup must state pre-emptively or a reviewer
will:
- Tables 13–15 evaluate **LFM2-ColBERT-350M**, a *separate late-interaction retrieval encoder* —
  LFM2-350M backbone + 9 task-specific layers + a 1024→128 projection, 353M params, trained by MSE
  distillation from a cross-encoder teacher via PyLate, on a checkpoint continued to 25T tokens
  (vs ~11T for the released dense LLM). It is a different model on a different training path.
- NanoBEIR is **corpus-level document ranking**, not in-context recall. Documents are capped at
  **512 tokens** and queries at 32, against a 32,768 backbone. Retrieval happens over an offline
  index via MaxSim — nothing is recalled from a long prompt.
→ So: **"the LFM2 technical report reports no in-context recall / long-context retrieval benchmark
for its language models"** is TRUE and defensible. The unqualified "no retrieval benchmark" is
sloppy and attackable. **Use the qualified form.**

**(iv) The "matches attention-heavier baselines" claim — CONFIRMED as unsupported prose.** §2.1
states selected candidates "match or exceed the aggregate quality of attention-heavier and mixed
(conv+linear/SSM/conv) baselines at the same budget." **There is no table, no figure, no model card,
and no numbers behind it.** Every published comparison (Tables 2/3/6/7) is against *external
released models* — Qwen3, Gemma 3, Llama 3.2, Granite 4.0, SmolLM3 — not against author-trained
parameter-matched controls. This is the single sharpest true sentence available.

**A rhetorical strengthener nobody has written down:** §2.1 justifies keeping GQA blocks by citing
the recall-limitation literature for recurrent/linear models ("their limitations in
retrieval-intensive tasks", "long-range retrieval abilities"), and §2.2 asserts that once a few GQA
blocks exist "the inexpensive gated short convolution" suffices. **Liquid names retrieval as the
exact failure mode their architecture is designed around, cites the papers that establish it, and
then publishes no retrieval measurement of the resulting language model.** That is a much better
sentence than "no LFM2 paper exists."

### d.4 Honest limitation on my own verification (ASSUMED, flagged)

The arXiv HTML render truncates at §5.4 on every fetch I attempted (four different anchors:
`#S2`, `#S6`, `#S7`, `#S9.SS2`). My table inventory for §§6–9 and Appendices A–D comes from a
**secondary index (alphaxiv)**, cross-checked against the paper's own table-of-contents structure
and the in-body cross-reference to Table 16. This is the same wall `05_evaluation.md:2064` hit and
flagged as unresolved. **Confidence: high that Tables 1–9 are as described (read directly);
medium-high for Tables 10–16.** The residual risk is small and one-directional: if a hidden recall
table exists it would be in Appendix C, and it would weaken only sub-claim (iii), not (i), (ii),
or (iv) — the ratio and kernel-width gaps are established from §2.1 + Table 1, which I read
directly. **To close it fully, pull the PDF and grep it** (I did not, to avoid a local download
without permission).

### 🔵 VERDICT (d): **CONFIRMED**

The paper exists (arXiv:2511.23404, *LFM2 Technical Report*, 33 authors, 28 Nov 2025). The repo's
self-contradiction is real and sits in **three** places in `06_baselines_infra.md` (lines 118, 157,
1946), not the one the reassessment cited. The upgrade from inferred to verified holds on all four
sub-claims: **no ratio ablation, no kernel-width ablation (despite the search space naming it), no
long-context/in-context-recall benchmark for the LLMs, and the "matches attention-heavier baselines"
claim is prose with no supporting table.** One required qualification: Tables 13–15 do report
NanoBEIR retrieval, for a *different* model (LFM2-ColBERT-350M) doing *corpus retrieval at 512-token
documents* — so the recall gap must be stated as "no in-context recall benchmark for the LFM2
language models," never as a bare "no retrieval benchmark."

**Correction pass required at:** `06_baselines_infra.md:118`, `:157`, `:1946`.

---

## CLAIM (e) — 695 GB/s units bug

### e.1 The units: all four figures are GiB/s (MEASURED)

Raw measurements, read from the committed artifact
`Brainlifts/liv_experiment_research/probes/p1_verify_results.json` (job 1670884, L40S) — medians,
full precision:

```
$ ssh -S /tmp/farmshare-ericrcwu.sock ericrcwu@login.farmshare.stanford.edu \
    '/scratch/users/ericrcwu/kda/venv/bin/python /scratch/users/ericrcwu/liv/verify/bw.py'

arm                           bytes         us  GB/s(1e9) GiB/s(2^30)      doc
dense                      41943040     56.224      746.0       694.8      695
lowrank_fused r=128        10485760     60.832      172.4       160.5      161
lowrank_sep r=128          10485760     76.480      137.1       127.7      128
grouped g=4                10485760     47.600      220.3       205.2      205
grouped g=2                20971520     47.680      439.8       409.6     None
lowrank_fused r=512        41943040     90.016      466.0       434.0     None
```

**All four documented figures match the GiB/s column and none match the GB/s column.**
695↔694.8, 161↔160.5, 128↔127.7, 205↔205.2. ✅ **CONFIRMED.**

### e.2 The bug in the code, located exactly (MEASURED)

`/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/probes/p1_verify.py`,
lines 81 and 87:

```python
 81:        mb = bytes_per_token(arm, r, g) / 2**20          # -> MiB, not MB
 ...
 87:              f"{mb/statistics.median(trials)*1e6/1024:6.0f} GB/s")
```

The reassessment says "line 87 divides by 1024" — correct, and I can now state the mechanism
precisely. **It is a two-part error, not one.** Line 81 converts to **MiB** (`/2**20`). Line 87 then
does `MiB / µs × 1e6 / 1024` = `MiB/s / 1024` = **GiB/s**, and labels it `GB/s`. Neither step is
wrong on its own; the mismatch is that a *binary* prefix chain is given a *decimal* label. If the
author had intended GB/s the line needed `bytes/(t·1e-6)/1e9`, i.e. line 81's `2**20` and line 87's
`1024` both replaced.

Exact recomputation of the code path confirms it reproduces the printed output bit-for-bit:

```
code line 87 recompute: mb/median*1e6/1024  where mb = bytes/2**20
  dense                  ->   694.77   (rounds to 695)  == GiB/s? True
  lowrank_fused r=128    ->   160.53   (rounds to 161)  == GiB/s? True
  lowrank_sep r=128      ->   127.69   (rounds to 128)  == GiB/s? True
  grouped g=4            ->   205.16   (rounds to 205)  == GiB/s? True
  grouped g=2            ->   409.63   (rounds to 410)  == GiB/s? True
  lowrank_fused r=512    ->   433.95   (rounds to 434)  == GiB/s? True
```

**Scope of the bug — it is isolated to `p1_verify.py` (MEASURED).** I grepped every probe script:

| script | conversion | verdict |
|---|---|---|
| `probes/p1_verify.py:81,87` | `/2**20` then `/1024` | 🔴 **BUGGY** — the only source of 695/161/128/205 |
| `probes/p1_cache_check.py:180` | `per_stack[name] / (us*1e-6) / 1e9` | ✅ correct GB/s |
| `reassessment/p1_scaled.py:137` | `mib * 2**20 / t / 1e3` (t in µs) → B/s/1e9 | ✅ correct GB/s |
| `probes/p1_launch_bench.py:218` | only prints MiB/tok, no rate | ✅ n/a |

**Consequence for cross-document comparison, and this is a trap for the correction pass:** the
`GBs` field in `p1_scaled_results.json` and the `achieved_gbs` field in `cachechk_results.json` are
**true GB/s**, while the README's numbers are **GiB/s**. Anyone tabulating "dense 695 (old) vs 745
(new)" is comparing a GiB/s to a GB/s and will report a 7.2% improvement that does not exist —
`p1_scaled`'s 40 MiB rung measures 744.7 GB/s, and job 1670884's dense is 746.0 GB/s. **They agree
to 0.2%; they are the same measurement.** Do not let this become a second unit artifact.

### e.3 The "80% of peak" arithmetic (MEASURED)

```
L40S HBM spec peak = 864 GB/s (1e9)
  dense GB/s / peak  = 86.3%
  dense GiB/s / peak = 80.4%  <- the doc's 80% (unit-mismatched)
  fused GB/s / peak  = 20.0%
  fused GiB/s / peak = 18.6%  <- the doc's 19%
peak in GiB/s = 804.7
  dense GiB/s / peak_GiB = 86.3%  (self-consistent unit comparison)
```

✅ **CONFIRMED.** The doc took a GiB/s numerator over a GB/s denominator. Fixed either way — GB/s
over GB/s, or GiB/s over GiB/s — the answer is **86.3%** (the reassessment says 86.4% in §A.5 and
86.3% in §C.1; **86.3% is right**, 746.0/864 = 86.34%). The "19%" for the factorized arm becomes
20.0%.

**Where the "80% of peak" claim lives:**

| file:line | text |
|---|---|
| `Brainlifts/liv_experiment_research/probes/README.md:33` | "Dense reaches **80% of L40S peak bandwidth**; the factorized version reaches 19%. **Skinny GEMVs cannot saturate the memory system…**" |
| `Brainlifts/liv_experiment_research/probes/README.md:23-26` | the 695 / 161 / 128 / 205 table itself |
| `HANDOFF.md:208` | "Dense achieves 695 GB/s (80% of L40S peak, genuinely bandwidth-bound)" |
| `docs/liv-brainlift-experiment-design.md:576` | "Dense hits 695 GB/s (80% of L40S peak) — genuinely bandwidth-bound —" |
| `docs/liv-brainlift-experiment-design.md:41` | "skinny GEMVs achieve only 161 GB/s vs dense's 695" (summary table, row P1) |
| `…/memory/liv-experiment-key-numbers.md:128` | "695 GB/s (80% of L40S peak, bandwidth-bound); `lowrank_fused r=128` achieves only **161 GB/s**." |

### e.4 THE PART THAT MATTERS — are §A.5's and §A.5b's corrections consistent?

§A.5 says: units are wrong, the true figure is 86.4% of peak, and this is **self-harming** — fixing
it *strengthens* the bandwidth-bound argument.
§A.5b says: the benchmark's working set (10/20/40 MiB) fits entirely inside the L40S's **96.0 MiB
L2**, so the rate measured is an **L2** rate and comparing it to the **HBM** spec is meaningless.

**Are they consistent? Formally yes; rhetorically they annihilate each other. §A.5b strictly
dominates.**

- They are consistent as *statements*: §A.5 is about how a ratio was computed, §A.5b about whether
  the ratio means anything. Both are true simultaneously. The number 746.0 GB/s is a correct
  measurement of *something*; §A.5b identifies that something as L2, not HBM.
- But §A.5's **interpretive rider is false given §A.5b.** "86.4% of peak strengthens the
  bandwidth-bound argument" only holds if the numerator is an HBM rate. It is not. Correcting
  80% → 86.3% makes the *cache* look better saturated relative to a spec it never touched. It does
  not strengthen anything; it makes a meaningless ratio more precisely meaningless.

**Three independent confirmations that the 40 MiB run is not HBM-bound (MEASURED):**

1. **Working sets fit L2 with room to spare.** L2 = 100,663,296 B = 96.0 MiB (measured, job 1671407,
   `torch.cuda.get_device_properties`). 40 MiB = **41.7%** of L2; 20 MiB = 20.8%; 10 MiB = 10.4%.
   One CUDA-graph capture, 50 warmup iters, 300 timed replays, nothing evicting between them.
2. **`grouped g=2` vs `g=4`: identical 20 kernels, 2× the bytes, 0.17% time difference.**
   (47.680 vs 47.600 µs, both from the committed JSON.) A bandwidth-bound benchmark cannot produce
   that. This pair alone falsifies the roofline reading and it was sitting in the results file.
3. **The same dense arm re-measured out of cache runs at 462 GB/s, not 746** (job 1671420,
   `p1_scaled_results.json`):

```
dense achieved rate by working set:
    40 MiB      56.32 us  ->    744.7 GB/s ( 693.6 GiB/s)   =  86.2% of 864 GB/s HBM peak
   320 MiB     728.06 us  ->    460.9 GB/s ( 429.2 GiB/s)   =  53.3% of 864 GB/s HBM peak
   960 MiB    2180.06 us  ->    461.7 GB/s ( 430.0 GiB/s)   =  53.4% of 864 GB/s HBM peak
```

The 320 and 960 MiB rungs agree to **0.2%** with each other and are **1.61× slower** than the
in-cache rung. That gap *is* the L2/HBM boundary, measured. Also note the per-kernel fixed cost:
56.224/20 = **2.81 µs/kernel**, against ~0.52 µs of data movement per kernel at a nominal 4 TB/s L2
— the 40 MiB benchmark is dominated by per-kernel overhead, not data movement, exactly as (2)
implies.

**So which correction goes in the writeup? Neither as written. Do this:**

1. **DROP "86.4% (or 86.3%) of peak" entirely. Do not propagate it.** It is the right arithmetic on
   a quantity that should not be divided by the HBM spec at all. Propagating it converts a units
   bug into a *more confident* wrong claim — the worst possible outcome of a correction pass. This
   is a real risk: §A.5's own phrasing is "**Fix the units and raise the claim to 86%**," which if
   executed literally makes the paper worse.
2. **DO fix the unit labels** — in the README table, `HANDOFF.md:208`,
   `docs/…:41`, `docs/…:576`, and the memory note. Not because the ratio matters, but because
   `p1_scaled.py` and `p1_cache_check.py` emit true GB/s and mixing the two silently corrupts every
   future cross-job comparison (see e.2). Simplest correct fix: **relabel to GiB/s**, or better,
   **fix `p1_verify.py:81,87` to emit GB/s** and reprint as 746 / 172 / 137 / 220 so the whole repo
   is in one unit.
3. **The sentence that replaces the claim** — supported by the numbers above, and stronger than
   either version: *"At the 40 MiB working set the probe reports 746 GB/s, but the L40S has 96 MiB
   of L2 and the entire benchmark is cache-resident; the same arm scaled past L2 achieves 462 GB/s
   = 53% of HBM peak. The in-cache figure is not a roofline datapoint, and the `grouped g=2` vs
   `g=4` pair — 2× the bytes, 0.17% the time — shows the probe was never bandwidth-bound at all."*
4. **Kill the dependent sentence too.** README:33's "**Skinny GEMVs cannot saturate the memory
   system**" is downstream of the same confound and must go with it — at 960 MiB the factorized arm
   is **31.3% faster**, not slower. (Per the brief, the L2 artifact and the P1 sign reversal are
   already settled; I flag this only because it is the same *sentence* the units fix would touch,
   and a correction pass that fixes "80%→86%" while leaving that clause intact produces an
   internally incoherent paragraph.)

**One point in §A.5's favour, for completeness:** the units error genuinely *was* self-harming *at
the time it was made*, under the (then-unquestioned) assumption that the measurement was of HBM. So
§A.5 was not wrong to call it self-harming; it was reasoning correctly inside a frame that §A.5b
later demolished. The two sections were written in sequence, and §A.5's rider simply was not
retracted when §A.5b landed. **That is the actual defect to fix in `00_SYNTHESIS.md`: §A.5's "Fix
the units and raise the claim to 86%" should be struck and replaced with a pointer to §A.5b.**

### 🔵 VERDICT (e): **CONFIRMED on the units, REFUTED on the remedy**

The unit bug is real, reproduces exactly, and is localized to `p1_verify.py:81` + `:87` (a MiB
numerator with a decimal-prefix label). All four documented figures are GiB/s. The corrected
ratio is 86.3%, not 86.4%. **But the "86.4% of peak" correction must NOT be propagated** — the
40 MiB working set is 41.7% of the L40S's 96.0 MiB L2 and the measurement is of cache, so dividing
it by an HBM spec is meaningless in either unit. §A.5 and §A.5b are formally consistent but §A.5b
dominates; the correct action is to **fix the labels for cross-job comparability and delete the
"% of peak" claim**, replacing it with the measured out-of-cache 462 GB/s = 53% of HBM peak.

---

## CONSOLIDATED CORRECTION-PASS CHECKLIST (for the human)

**Claim (c) — d=2048 contamination**

| file:line | action |
|---|---|
| `…/memory/liv-experiment-key-numbers.md:30-31` | 14.3% → **11.8%**; 68.8% → **63.9%**; 6.27% → **4.44%** |
| `…/memory/liv-experiment-key-numbers.md:179` | 6.27%/6.94% → **4.44% / 5.55%**; drop `r=512` from the sweep set (saves exactly 0 at d=1024) |
| `HANDOFF.md:81` | GQA `2.5d²` → note **3d² = 3,145,728/layer at the frozen d=1024 arm** |
| `docs/liv-brainlift-experiment-design.md:498` | label the ledger block "(d=2048/1.2B)"; add the d=1024 row |
| `docs/liv-brainlift-experiment-design.md:661` | 6.27%/6.94% → **4.44% / 5.55%** |
| `docs/liv-brainlift-experiment-design.md:656` | label 68.8% as d=2048 |
| `00_SYNTHESIS.md` §A.2 table | 🔴 **the reassessment's own "4.91%" is wrong → 5.55%** |
| *no change* | `docs/…:113,120`; `01_lfm2_architecture.md` (all); `02_lowrank_gates.md` (all); `00_my_arithmetic_check.md` (all); `07_latency_kernels.md:582,623,739`; `06_baselines_infra.md:748` (false-positive grep hit) |

**Claim (d) — LFM2 paper**

| file:line | action |
|---|---|
| `06_baselines_infra.md:118` | replace with the qualified gap statement |
| `06_baselines_infra.md:157` | table row label `**LFM2** (no paper)` → cite arXiv:2511.23404 |
| `06_baselines_infra.md:1946` | risk item 7 — same |
| `05_evaluation.md:524, 2064` | the "§6–9 could not be retrieved" flag stands; the recall gap is now supported by the table inventory. Add the LFM2-ColBERT / NanoBEIR caveat. |

**Claim (e) — bandwidth units**

| file:line | action |
|---|---|
| `probes/p1_verify.py:81,87` | fix to emit true GB/s (`bytes/(t·1e-6)/1e9`) |
| `probes/README.md:23-26` | 695/161/128/205 → **746/172/137/220 GB/s** (or relabel GiB/s) |
| `probes/README.md:33` | **DELETE** "80% of peak" and "skinny GEMVs cannot saturate the memory system" — do not replace with 86% |
| `HANDOFF.md:208` | same |
| `docs/liv-brainlift-experiment-design.md:41, 576` | same |
| `…/memory/liv-experiment-key-numbers.md:128` | same |
| `00_SYNTHESIS.md` §A.5 | strike "Fix the units and raise the claim to 86%"; point to §A.5b |

