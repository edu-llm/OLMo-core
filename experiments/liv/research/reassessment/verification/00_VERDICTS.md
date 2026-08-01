# VERIFICATION of the reassessment team's claims

**Date:** 2026-08-01. **Author:** orchestrator of the verification team.
**Status:** COMPLETE — all 7 children reported; written incrementally throughout.

Scope: four claims from `../00_SYNTHESIS.md` that would change what the human does next.
NOT in scope (already settled, confirmed twice): the L2-cache artifact / P1 latency reversal, and the
`ShortConv.num_flops_per_token` 3× undercount.

Rating scale: **CONFIRMED** (primary source or re-derived arithmetic) / **REFUTED** / **UNCLEAR**.
Evidence class: MEASURED / INFERRED / ASSUMED.

| child file | claim |
|---|---|
| `01_claim1_p2_yoio.md` | C1a — does arXiv 2606.06467 exist and report retrieval for cross-layer KV sharing? |
| `02_claim1_p2_priorart.md` | C1b — is P2 anticipated by Hymba / Character.AI / Gemma 3n? What is left? |
| `03_claim2_p3_taps.md` | C2a — is the conv-tap read sound and does it support "k=3 nowhere near binding"? |
| `04_claim2_p3_tian.md` | C2b — does Tian et al. 2607.18413 Table 7 contain the reparameterization experiment? |
| `05_claim3_compute.md` | C3 — FarmShare's real walltime / concurrency / QOS limits |
| `06_claim4_stats.md` | C4a/b — CE margin vacuity; "5 of 9 gates fail-open" |
| `07_claim4_arith.md` | C4c/d/e — d=2048 contamination; arXiv 2511.23404; GiB/s vs GB/s |

---

## ORCHESTRATOR CHECK 0 — arXiv ID existence sweep (MEASURED, 2026-08-01)

Run by the orchestrator before any child reported, to settle the fabrication question directly.
`curl -sIL https://arxiv.org/abs/<id>`, reading the `citation_title` meta tag. **Literal IDs, no
typo-correction applied.** Note this project's notes use future-dated IDs (25xx = 2025, 26xx = 2026);
today is 2026-08-01, so June/July 2026 IDs are chronologically plausible.

| arXiv ID | HTTP | title returned | claimed as |
|---|---:|---|---|
| 2606.06467 | **200** | *You Only Index Once: Cross-Layer Sparse Attention with Shared Routing* | the paper that kills P2's headline |
| 2607.18413 | **200** | *Convolution for Large Language Models* | Tian et al., P3's steelman killer |
| 2511.23404 | **200** | *LFM2 Technical Report* | the LFM2 paper the repo says doesn't exist |
| 2606.03825 | **200** | *Dynamic Short Convolutions Improve Transformers* | Sieberling et al., width-flat-past-k=3 |
| 2411.13676 | **200** | *Hymba: A Hybrid-head Architecture for Small Language Models* | P2 prior art |
| 2406.16450 | **200** | *Building on Efficient Foundations: Effectively Training LLMs with Structured Feedforward Layers* | Wei et al., grouped-vs-lowrank prior art |

**Verdict: 6 / 6 resolve, with the exact titles claimed. NO FABRICATED CITATIONS at the ID level.**
This is a meaningful reliability signal for the prior team. What remains to be checked per-paper is
whether the *contents* (tables, signs, effect sizes) are transcribed correctly — that is the children's
job, and a real ID with a mis-transcribed table is still a fatal error.

---

## VERDICT TABLE

| # | prior-team claim | verdict | decisive evidence |
|---|---|---|---|
| **1a** | P2's headline gap is gone — 2606.06467 "ran a better version, opposite sign" | **PARTLY CONFIRMED / PARTLY REFUTED** | Paper real (`arxiv.org/html/2606.06467v1`), table transcribed 72/72 correct. **But its Transformer control uses RoPE 5e5 while both KV-sharing arms use RNoPE 1e4 + SWA-512 (App. C Table 8, §3.1), with no ablation separating them.** The +6.1 cannot be attributed to sharing. |
| **1b** | P2 anticipated 3× (Hymba / Character.AI / Gemma 3n) | **CONFIRMED as mechanism, REFUTED as measurement** | Hymba's "recall" is a 2-task average, its NIAH is an architecture comparison **not a sharing ablation**; Character.AI is a blog with **zero numbers**; Gemma 3n publishes **no quality eval** and is 13-way/4-way, **not pairwise** (and `num_kv_shared_layers=15` is E4B-only; E2B is 10). |
| **2a** | Conv taps show k=3 "nowhere near binding, measured inside the gate" | **UNCLEAR (split): arithmetic CONFIRMED, "inside the gate" REFUTED** | Orientation settled by impulse test (idx 0 = oldest); **every statistic reproduces to the digit** on all 4 checkpoints via independent code. **But it is weights-only — `in_proj`/`out_proj` never touched, no token ever run**, so it cannot answer the gate objection (the owed activation-weighted "C3" is exactly that test). Headline 1.4% → **14.4% under AR(1) ρ=0.9**. "No sub-population" **REFUTED** (792/10240 >20%, 102 saturated, growing with scale). "Independent replication" **REFUTED** (same script lineage). Settle with C3+C5 ≈ **0.5 GPU-h**. |
| **2b** | P3's steelman already run + negative in Tian Table 7 | **REFUTED** | Table 7 adds **narrower** (k=1,k=2) branches to k=3 — **union of lags stays {0,1,2}**. P3 asks a **span** question (dilated → lags {0,1,2,4,7,8,14}). Also no per-branch norm ⇒ not RepVGG; n=1, no error bars; −0.49 ppl is 78% of the whole conv effect. |
| **3** | FarmShare is 48 h / 4 concurrent GPUs, not 6 h / 1 GPU | **CONFIRMED** (2 independent routes) | `MaxTime=2-00:00:00`; QOS `gpu` `MaxTRESPU=gres/gpu=4`, `MaxJobsPU=4`, **MaxWall empty**; 6×`gpu:4` nodes = 24 L40S. **Bonus: `--qos=long` gives 7 DAYS.** Sub-claim "`-c 8 --mem=48G` is rejected" is **REFUTED** — Slurm silently bumps to 14 CPUs. |
| **4a** | CE margin vacuous (+0.010 vs a 0.0030–0.0072 nat phenomenon) | **CONFIRMED on arithmetic, OVERSTATED in rhetoric** | Basin = `ln(8.32/8.26)` = **0.00724 nats**; margin is 1.38× wider. But "would declare all-conv non-inferior to a transformer" holds only vs Mamba-2's Tf++ row (0.00926); the relevant contrast is **0.0403 = 4× the margin**, which WOULD fire. §6.1 already demotes CE ⇒ low consequence. |
| **4b** | 5 of 9 gates are fail-open | **SUBSTANCE CONFIRMED, COUNT WRONG** | 9 rows but ~20 clauses; only **3** unambiguously fail-open + 1 half + 2 pass-only ⇒ **6 of 9 cannot stop anything**. Audit's own score line sums to **10 across 9**. n=2 "91%" **does not reproduce** (86.5%; the 9% was a *different effect size*). Ratio 1.1708 ✅, Fisher 1/6 ✅, FWER 43.12% ✅. |
| **4c** | 6.27%/14.3%/68.8% are d=2048 numbers | **CONFIRMED — with an error IN the correction** | Both ledgers rebuilt exactly (354,483,968 / 1,170,340,608). Frozen values **4.44% / 11.84% / 63.89%**; GQA is exactly 3.000d²; r=512 saves **exactly zero**. 🔴 **The reassessment's own "r=32 → 4.91%" is WRONG; it is 5.55%.** |
| **4d** | "No LFM2 paper exists" is wrong — 2511.23404 does | **CONFIRMED** | Paper real (Nov **2025**, not future-dated). Contradiction is in **three** places in `06_baselines_infra.md`. Gap claim survives: no ratio ablation, no width ablation, no in-context recall for the LLMs. ⚠️ **Must say "no in-context recall benchmark for the LFM2 *language models*"** — Tables 13–15 report NanoBEIR for LFM2-**ColBERT**. |
| **4e** | "695 GB/s" is GiB/s and describes L2 | **CONFIRMED on units; REFUTED as a fix** | All four figures match GiB/s exactly (694.8/160.5/127.7/205.2). Bug at `p1_verify.py:81,87`. **But do not propagate "86.4% of peak"** — the L2 finding makes any %-of-HBM-peak meaningless. Delete the sentence; report the out-of-cache **462 GB/s = 53%**. |

---

## Findings

### CLAIM 1a — "arXiv 2606.06467 kills P2's headline" → **PARTIALLY CONFIRMED**
Child: `01_claim1_p2_yoio.md`.

- **The paper is REAL.** `export.arxiv.org` API returns `totalResults=1`; `https://arxiv.org/html/2606.06467v1`
  is 265 KB of HTML. Title, authors (Yutao Sun, Yanqi Zhang, Li Dong, Jianyong Wang, Furu Wei), and
  date `2026-06-04` match the prior team exactly.
- **The transcription is perfect — 72/72 cells.** It is the paper's Table 3 (not "the RULER table"
  generically). All six row means re-add correctly: 64.433 / 62.675 / 62.925 / 46.167 / 52.325 / 53.092
  vs the stated 64.4 / 62.7 / 62.9 / 46.2 / 52.3 / 53.1.
- 🔴 **BUT THE +6.1 IS CONFOUNDED, and the prior team missed it.** Appendix C Table 8: the Transformer
  control uses **RoPE base 5×10⁵**, while *both* YOCO arms use **RNoPE base 1×10⁴**; §3.1 adds that the
  YOCO arms use **SWA window 512** in the self-decoder. Three variables move at once — sharing,
  positional encoding, and attention pattern — **with no ablation separating them.** RNoPE
  (arXiv 2501.18795) exists specifically to improve long-context retrieval, and the gain appears *only*
  at 32K, i.e. exactly past the 8K stage-1 training context. The paper itself says the "key practical
  difference lies in how positional information is handled."
- Two smaller mis-readings: the prior team attributed the gains to MK2/MK3 where the paper's prose says
  "especially MK1 and MK2"; and it quoted the Dense row's −1.7/+6.1 while labelling it CLSA (−1.5/+6.9).

**Consequence.** Sub-claim (a) — *"no cross-layer-KV-sharing paper reports needle/passkey retrieval"* —
is **genuinely FALSE and must be struck.** Sub-claim (b) — *"a better version was run and it lands with
the opposite sign"* — is **REFUTED as stated**: 2606.06467 does not isolate sharing, so it cannot answer
the capstone's question in either direction. The prior team's advice that a reviewer's question has "no
mechanism" answer is **overstated**; the correct answer is that the existing paper is confounded.

**What survives is better than the prior team concluded, and it is a gap of QUESTION not configuration:**
*no paper cleanly isolates KV sharing's retrieval effect holding positional encoding and attention
pattern fixed.* (MQAR-specifically, CLA-pairwise-across-conv, and 350M-scale are gaps of *configuration*
and are weak on their own. The full text has 0 occurrences of MQAR / passkey / associative recall.)

**Citation audit: 11/11 resolve (100%).** All nine 2026-era IDs in §1.5 plus both anchors — CLA
`2405.12981` (Brandon et al.) and `2410.14442` (arXiv comment literally reads "Accepted to NAACL2025
main conference"). DepthWeave-KV and CommonKV retrieval claims verified verbatim; the "Stochastic KV
Routing names no benchmark" caveat is correct. Methodological note for future checks: `curl -L` is
required on the export API or every ID falsely appears to 301-fail.

**Net reliability signal on the prior team: the errors are over-reading, not invention.**

---

### ORCHESTRATOR CHECK 1 — the LFM2 Technical Report is real, and its abstract carries a warning

`https://arxiv.org/abs/2511.23404` → **HTTP 200, *LFM2 Technical Report***, ~30 Liquid AI authors
(Hasani, Lechner, Labonne, Amini, …). So the repo line "No LFM2 paper exists (only a blog post)"
(`06_baselines_infra.md:118`) is **wrong on its face** — see child `07_claim4_arith.md` for the
file:line inventory and the ablation-table audit.

⚠️ **One phrase in the abstract is load-bearing for the capstone's residual contribution and nobody in
the project has flagged it** (MEASURED, quoted verbatim):

> "Using **hardware-in-the-loop architecture search** under edge latency and memory constraints, we
> obtain a compact hybrid backbone that combines gated short convolutions with a small number of
> grouped query attention blocks…"

The capstone's surviving framing is *"Liquid ships this architecture and published no ratio ablation,
no kernel-width ablation, and no recall benchmark."* An explicit statement that the topology was
chosen by **architecture search** means the ratio and the kernel width were almost certainly *searched
over internally* — so the honest claim is **"Liquid did not PUBLISH the ablation," not "nobody knows."**
That is still a real gap (an unpublished search is not a reproducible ablation, and the search was
under *edge-latency* objectives, not quality-at-fixed-params), but it is a weaker and more precise
claim than the one currently written, and a reviewer familiar with the report will make exactly this
objection. The abstract also confirms **32K context for all sizes** and **10–12T training tokens**.

---

### ORCHESTRATOR CHECK 2 — CLAIM 3 (compute budget) independently CONFIRMED from the live cluster

Run by the orchestrator in parallel with child `05_claim3_compute.md`, so this is a genuinely second
route to the same facts. **MEASURED, 2026-08-01**, raw output:

`scontrol show partition gpu`:
```
AllowQos=ALL   QoS=gpu   DefaultTime=02:00:00   MaxTime=2-00:00:00
MaxNodes=UNLIMITED   OverSubscribe=NO   TotalNodes=6   MaxMemPerCPU=4000
```
`sacctmgr show qos` (Name | MaxWall | MaxTRESPU | MaxJobsPU | MaxSubmitPU):
```
normal |        | cpu=512,gres/gpu=1 | 128 | 1024
gpu    | <none> | gres/gpu=4         |   4 |   32
```
`sinfo -N -p gpu`: **6 nodes (oat-01…06), each `gpu:4`, 64 CPUs, 256 GB → 24 L40S cluster-wide.**

| question | answer | binding constraint |
|---|---|---|
| Max walltime | **48 h** | partition `MaxTime=2-00:00:00`; QOS `gpu` `MaxWall` is **empty (no cap)** |
| Max concurrent GPUs per user | **4** | QOS `gpu` `MaxTRESPU=gres/gpu=4` |
| Max concurrent GPU *jobs* per user | **4** | `MaxJobsPU=4` — so 4×1-GPU **or** 1×4-GPU, not both |
| Multi-GPU single-node | **YES, up to 4** | every node is `gpu:4`; 4 is also the per-user ceiling |
| Queue depth | 32 submitted | `MaxSubmitPU=32` |
| `--mem=48G -c 8` | **REJECTED** | `MaxMemPerCPU=4000` ⇒ 8 CPUs cap at 32 G. Need **`-c 12`** for 48 G |

**CLAIM 3 = CONFIRMED**, by two independent routes. The plan's "1 GPU / 6 h" assumption is wrong in
both fields; the truth is **48 h × 4 concurrent L40S = up to 192 GPU-hours per 48-h window.** Note the
trap the prior team found is real and worth propagating: the `normal` QOS allows only **1** GPU, so the
4-GPU allowance exists only when the job actually lands on the `gpu` QOS.

---

### ORCHESTRATOR CHECK 3 — context that bounds how consequential the CE-margin claim can be

Read directly from `/Users/ericwu/Developer/Capstone_LLM/docs/liv-brainlift-experiment-design.md`
(MEASURED, verbatim). The **+0.010 nats** margin the risk audit attacks is real and appears at
**line 1133**:

> "The existing protocol's gate is CE non-inferiority at **+0.010 nats**. With paired seeds and
> `n = ceil(((1.645+0.842)·s_δ/m)²)`, that margin is reachable only if `s_δ ≲ 0.011` at n≥8."

But the section containing it is **§6.1, titled: "Held-out CE is almost certainly underpowered — do
not make it the primary endpoint"** (line ~1126), and it goes on to designate the primary endpoints as
recall composite / length extrapolation / AR-Hits sliced perplexity. Line 1136 already mandates the
conservative reading: *"If it is out of reach, say 'inconclusive' rather than 'non-inferior'."*

⚠️ **So the risk audit's phrasing "every perplexity gate in the plan is useless" is TRUE ON THE MERITS
but OVERSTATED IN CONSEQUENCE.** The plan does not rest on the CE gate; it already demoted CE and
pre-committed to calling an unreachable margin "inconclusive." The correct action is to **delete the
CE gate** (it can only mislead), not to conclude that the plan's decision structure is broken. See
child `06_claim4_stats.md` for the full gate inventory and the re-derived power arithmetic.

---

### CLAIM 1b — "P2 is anticipated 3× (Hymba / Character.AI / Gemma 3n)" → **CONFIRMED as a MECHANISM, REFUTED as a MEASUREMENT**
Child: `02_claim1_p2_priorart.md`. No hallucinations; the prior audit's citations hold. Two corrections.

**Hymba `2411.13676` — exists, matches, but retrieval is NOT ablated.** Table 1 row C→D verified to the
decimal: commonsense 44.56→**45.16**, recall 48.79→**48.04**, 2399.7→**2756.5** tok/s, cache
41.2→**39.4 MB** (only **−4.4%**). The HF config `nvidia/Hymba-1.5B-Base` reproduces the docs' block
exactly: `kv_reuse_group` is strictly adjacent, `global_attn_idx=[0,15,31]` excluded.
🔴 **Decisive correction: the "Recall" column is an unnamed 2-task average, and Hymba's NIAH (Fig 10) is
an architecture comparison vs Mamba2/Llama3 — NOT a sharing ablation.** So the memory note's *"only
Hymba reports recall"* is true only in the weakest sense and should not be used to concede the point.
**Topology distinction CONFIRMED:** Hymba is a *parallel* hybrid-head (attention ∥ SSM inside one
layer), so producer→consumer has **no intervening mixer**; LFM2 has a full conv block between them.

**RoPE decision CONFIRMED from released code** — `modeling_hymba.py`, 3 identical sites: producer
rotates K pre-handoff, consumer rotates only its own Q, consumer allocates no `k_proj`/`v_proj` and
never writes cache. HANDOFF's decision #4 is sound.

**Character.AI — an existence proof with ZERO evaluation.** Retrieved via `web.archive.org` snapshot of
`research.character.ai/optimizing-inference/`. Verbatim: *"We tie the KV cache across neighboring
attention layers… we find that sharing KV across layers does not regress quality."* **No numbers, no
benchmark, no table.** Their needle-in-a-haystack mention attaches to the *sliding-window* item, not to
sharing. Pure transformer, not a hybrid.

**Gemma 3n — the prior team's "UNVERIFIED (HTTP 401)" is now CLOSED, and the docs' number is wrong.**
Settled from `transformers/models/gemma3n/configuration_gemma3n.py:127` (`num_kv_shared_layers: int = 15`)
plus ungated mirrors: **E4B = 35 layers / 15 shared; E2B = 30 layers / 10 shared.** The docs cite 15 as
if universal — **it is E4B-only.** And it is **not pairwise at all**: layer 18 (sliding) feeds 12
consumers (13-way), layer 19 (full) feeds 3 (4-way). Google claims 2× prefill and publishes **no
quality or retrieval eval**. Second independent confirmation of post-rotary K sharing.

**Synthesis.** None of the three shares across an intervening non-attention mixer; none shares between
*full-attention* layers in a hybrid; **none publishes a sharing-isolated retrieval number.**
→ **"Anticipated 3×" is overstated as grounds to CUT P2.** The scoped claim survives these three.
What actually pressures P2 is (i) `2606.06467` — and per child 1 that paper is *confounded*, so it
pressures less than advertised — and (ii) the capacity-not-bandwidth arithmetic, which is P2's real
economic problem and was never in dispute.
**Build note if P2 is kept:** both reference implementations confirm post-rotary K and no consumer KV
projections, so the zero-training post-hoc version is unambiguous to implement.

---

### ORCHESTRATOR CHECK 4 — bonus: the Wei et al. prior art (flagged by the prior team as single-sourced)

`00_SYNTHESIS.md` §C.7 used **arXiv 2406.16450** to retire the grouped-vs-lowrank direction, and its own
meta-note listed this as one of three single-sourced findings needing re-check. Verified from the abs
page (MEASURED): **exists, *"Building on Efficient Foundations: Effectively Training LLMs with
Structured Feedforward Layers"*, Wei, Moalla, Pascanu, Gulcehre, "Accepted by NeurIPS2024".** Confirms
training-from-scratch perspective, scaling to 1.3B, three structured parameterizations from low-rank
and block-diagonal matrices, and the **"self-guided training"** regime introduced to fix "the poor
training dynamics that these approximations exhibit when used from initialization."

⚠️ **One scope caveat the synthesis should carry:** the paper explicitly targets **feedforward networks
(FFNs)** and distinguishes itself from "convolutional architectures." The capstone's structures sit on
the **gate projections of a gated conv block** — multiplicative, not an FFN. So Wei et al. is strong
prior art on *structure choice* (low-rank vs block-diagonal from scratch) but does **not** cover the
operator, which is what §C.7 itself concluded survives. The per-structure PPL gaps (0.4–0.8) and the
110M lower bound were not confirmable from the abstract page and remain **single-sourced**; the full
HTML/PDF would settle them. Low priority — it does not change any recommendation.

---

### ORCHESTRATOR CHECK 5 — the four LFM2 checkpoints behind the P3 tap claim really exist on disk

The prior team's P3 kill rests on tap reads of four released checkpoints, and `04_cheap_experiments.md`
claims replication across 350M/700M/1.2B/2.6B. Verified the artifacts are real (MEASURED,
`ls`/`du` on FarmShare `/scratch/users/ericrcwu/liv/ckpt/`):

```
model.safetensors  708,984,464 B (674M)  ← LFM2-350M, alongside its config.json
LFM2-700M   1.4G      LFM2-1.2B   2.2G      LFM2-2.6B   4.8G
```

**All four checkpoints are present.** The 350M safetensors is 708,984,464 B, consistent with
354,483,968 params × 2 B (bf16) + header — an independent corroboration of the frozen ledger.
So the tap numbers are computed over real weights, not invented. (Whether the *two* tap analyses were
methodologically independent, and whether a weights-only read can support the "inside the gate" claim,
is child `03_claim2_p3_taps.md`'s call.)

---

### CLAIM 2a — the P3 conv-tap read: **ARITHMETIC FULLY CONFIRMED; interpretation still under test**
Child: `03_claim2_p3_taps.md` (partial — sections 3–8 pending at time of writing).

**The load-bearing orientation risk is RETIRED.** The whole conclusion inverts if tap index 0 is the
*current* token rather than the oldest. The child settled it two ways: (1) reading
`modeling_lfm2.py` (`nn.Conv1d(..., padding=L_cache-1)` then left-slice `[..., :seqlen]`; decode path
`sum(conv_state * weight[:,0,:])` with `conv_state` newest-frame-last), and (2) a **decisive empirical
impulse test on FarmShare** — one-hot kernels through the real module with `x[t]=t`:

| one-hot weight index | output | implied lag |
|---|---|---|
| 0 | `[0,0,0,1,2,3]` | **t−2 (OLDEST)** |
| 1 | `[0,0,1,2,3,4]` | t−1 |
| 2 | `[0,1,2,3,4,5]` | **t (CURRENT)** |

Prefill and decode agree. **The prior team's orientation is correct.**

**Every reported statistic reproduces to the digit**, via an independently written script that does not
import theirs: pooled energy 4.26/29.62/66.12%; normalized medians 0.0143/0.1721/0.7439; ratio 0.0833;
boundary-argmax 0.0208; boundary-saturated 0.0246; layer-0/1 lag-1 medians 0.9905/0.9259; layer-15
current-token 97.62%; and the cross-scale series **4.26/5.24/5.34/4.78** across all four checkpoints,
which the child read itself. Also confirmed: `conv_bias: false`, so the 3 taps are the *entire* conv
parameterization, and the 2.6B topology really is 30 layers / 22 LIV / attn at {2,5,9,13,17,21,24,27}.

**So there are no transcription errors, no wrong tensors, no dtype bug. The dispute is entirely about
METHOD** — and there, the claim breaks in five places.

#### 🔴 "MEASURED INSIDE THE GATE" IS **REFUTED** — and it is the P1 `spectra_v2` failure mode repeating

The statistic is `w²` read off `conv.conv.weight`. **It never touches `in_proj` or `out_proj`, never
runs a token, and has no access to any activation.** The gates `B` and `C` multiply the conv's input
and output (`Bx = B*x`; `y = C*conv_out`) and are **input-dependent**. So "inside the gate" is true
*topologically* (the weight sits between two gates in the graph) and false *evidentially* (the measured
quantity reflects gate behaviour) — the two senses are equivocated.

**This is precisely the error the project already made and corrected once.** Plain weight spectra of
these same gates gave a misleading answer; only the activation-weighted version was right (rank-128
retains **45.8%** of plain Frobenius energy but **92.6%** of activation-weighted energy — a 2× swing
that inverted the conclusion). **The prior team knew:** `04_cheap_experiments.md` flags the owed
activation-weighted follow-up ("C3") three times, including *"Required before publishing C1."* Then
`06_p2_p3_verdict.md` §5.5 promotes the un-activation-weighted read into a *refutation of the gate
objection*. **"C3 is owed" and "the gate objection is answered" cannot both be true** — C3 is the very
thing that would answer it.
→ **Retract** the sentence "measured inside Liquid's actual double gate … this measurement removes it."

**Two quantitative bounds the child produced (both new):**
- 🟢 *In the prior team's favour, and stronger than anything they had:* weighting each channel by a
  throughput proxy (`‖B_c‖²‖x_c‖²‖C_c‖²‖out_proj[:,c]‖²`) shows the channels the gates care most about
  use **less** span — top-1% importance stratum has **6.6%** off-current energy vs **46%** for the
  bottom half. This substantially de-risks C3.
- 🔴 *Against:* `w²` is a per-tap variance decomposition **only if the conv input is white.** LM
  residual streams are strongly autocorrelated. Under AR(1) with ρ, the honest statistic is the
  leave-one-out variance drop. At **ρ=0.9** (routine for adjacent-position hidden states) the headline
  **1.4% becomes 14.4%** — a 10× change — and the fraction of channels where the oldest tap carries
  >10% of variance goes from 17% to **60%.** The headline is not robust to the one distributional
  assumption it silently makes.

#### Three further overstatements, each verified

- **"No sub-population anywhere wants a wider kernel" — REFUTED.** 792/10240 channels exceed 20%
  boundary energy; **102 exceed 80%** (pure lag-2 delay lines pinned at the window edge), concentrated
  in layers 0–1 — and the fraction **grows with scale** (4.9% → 8.4% of layer-0 channels, 350M→1.2B).
  Small (1–2% model-wide), but not zero, and it was missed by looking only at medians.
- **"1.4%" is the most favourable of three summary statistics.** Median 1.43% < pooled raw 4.26% <
  **mean 6.13%**; the distribution is right-skewed **4.3×**. Also, pooling delay-lines and vestigial
  layers with true mixers halves the key ratio: the mixer-only ratio is **0.184, not 0.083.**
- **"Independent replication across four checkpoints" — REFUTED as independence.** `tapread.py`,
  `tapfreq.py`, `tapread2.py` were written **within 2m40s** of each other, same session; `tapfreq.py`
  contains a **character-for-character copy** of `tapread.py`'s bf16 decoder and key selector. The
  350M 4.26% is identical across two documents because **it is the same computation reported twice.**
  Every shared assumption is shared *code* — had the orientation been wrong, all four would have been
  wrong identically. Call it a cross-scale consistency check, not a replication.
- **The random-init control is a strawman** (any symmetric init gives ~1/3 by construction). The
  *actual* LFM2 init is current-token identity — against which the trained model moved **toward**
  history, not away.
- **A real bug, direction conservative:** `np.argmax` credits ties to index 0 (the oldest tap). At
  **2.6B, 15.5% of channels have tied maxima**, so the reported "9.75% oldest-is-argmax" is **4.8×
  inflated** — true value **2.03%**. The 2.6B row of the cross-scale table is wrong, and the "2.6B
  proves it is depth-relative" reading is a tie artifact.

**CLAIM 2a VERDICT: UNCLEAR (split).** Orientation, arithmetic, and the qualitative shape ("decays
toward the boundary") are **CONFIRMED and robust across scale**. "Nowhere near binding" is
**directionally supported but the margin is far smaller than presented**. "Measured inside the gate"
is **REFUTED**. "No sub-population" is **REFUTED**. "Independent replication" is **REFUTED as
independence**.

**One more check the child ran, to its credit: was the pre-registered rule falsifiable?** Yes, better
than expected — a box-filter profile would have tripped the BINDING branch, so the rule was not
vacuous. **But** at a true desired-span correlation of r=0.7 a model leaves **11.8% of desired mass
outside the window and still decays monotonically.** So boundary-heaviness is the signature of a *flat*
truncated filter, not of truncation in general — the rule is sound but less discriminating than "5 for
5, by 16× margins" implies.

**What would settle it, and it is remarkably cheap:** **C3** (activation-weighted tap energy) ≈ **0.2
GPU-h**, and **C5** (causal tap zeroing — actually ablate the oldest tap and measure the loss change)
≈ **0.3 GPU-h**. Both fit trivially in one 48 h job. Neither answers whether a from-scratch k=15 would
help — only ~2 GPU-days of training does.

**Do not reverse the P3 deprioritization** — the direction of the evidence is right. But rest it on
**the two published width sweeps**, not on this read. Do not put "measured inside the gate" in a paper.

---

### CLAIM 2b — "P3's steelman was already run and lost in Tian et al. Table 7" → **REFUTED**
Child: `04_claim2_p3_tian.md`. The paper is real and quoted correctly; the *inference drawn from it* is
wrong.

**The paper and the table are exactly as transcribed (no hallucination).** `arxiv.org/abs/2607.18413`
and `arxiv.org/html/2607.18413v1` (169 KB) resolve. Tian, Shu, He, Zhang, Zhao, Xu, Chen, Wang, Chen,
Wang; PKU/Huawei/Tsinghua; 20 Jul 2026. Qwen3-1.7B from scratch on FineWeb-100B, WikiText-103 ppl.
Table 7 caption: *"Effect of convolutional reparameterization in Qwen3-1.7B."* Rows verbatim:
2.4795/12.79/1721.03 → 2.5029/13.28/1721.26 → 2.5048/13.28/1721.61. Body text confirms fusion:
*"merge the branches into an equivalent convolution for inference… One possible explanation is that the
branches mix local patterns over different spans, although this ablation does not isolate the cause."*

🔴 **But it is NOT the capstone's steelman.** Tian adds **narrower** branches (k=1, k=2) to a k=3
kernel — **the union of lags stays {0,1,2}**. P3's steelman adds **dilated** branches extending the
reachable lags to {0,1,2,4,7,8,14}. Tian isolates *over-parameterization at fixed span*; P3 asks a
*span* question. Different experiments. Further, it is **not even RepVGG-style**: Tian's branches carry
no per-branch norm (he tested norms and rejected them), so the k=1 case is literally one redundant
scalar per channel — the most degenerate possible instance of reparameterization.
**Note the prior team's own caveat (iii) said exactly this and flatly contradicts its own headline.**

**The negative result is also fragile.** Zero seeds and zero error bars anywhere in the paper
(grep-verified). The −0.49 ppl is **78% of the entire conv effect** (13.42→12.79), is *worse than
deleting the residual shortcut* (13.05), and lands within 0.01 of the paper's own degraded-init row
(13.27). The child reproduced the params column exactly as `(k+1)·4096·28`, proving it is a
*training-time* count **and** that each branch carries its own random-init bias — so an init-scale
confound is live, which is precisely the trap this project already documented for P1's rank sweep.

**The "gates don't save you" sub-argument is REFUTED too.** A pointwise *activation* on the conv output
is not a *multiplicative gate* around the block; Tian never trains a gated variant. If anything the
0.15–0.54 ppl swings from norms/activations show *sensitivity to the surround*, which weakly **supports**
the gate escape hatch rather than undercutting it.

**Sieberling `2606.03825` — REAL and exact to 2 dp.** Table 3a 18.42/18.17/18.08/18.10/18.09/18.10 →
marginals +0.25/+0.09/−0.02/+0.01/−0.01 confirmed; rank R=16→128 = 0.25 confirmed. And **ungated**:
verbatim `X = X + dynamicShortConv(X)`.
⇒ **Both width citations are ungated-residual. No width sweep in a gated mixer exists in the
literature** — the capstone's original gate caveat is intact and is a real gap.

**Net:** the P3 *cut* may still stand on other grounds (the LFM2 weight measurement + Sieberling), but
**"the steelman has already been run" must be downgraded to "an adjacent negative prior: n=1,
fixed-span, norm-free, ungated."** A reviewer with the paper open would catch the overclaim.

---

### CLAIM 4c — "6.27% / 14.3% / 68.8% are d=2048 numbers in a d=1024 design" → **CONFIRMED, with one correction TO the correction**
Child: `07_claim4_arith.md`. All arithmetic re-derived independently on FarmShare.

Confirmed exactly: P1 at r=128 saves **4.437%** at the frozen geometry; **6.272%** is the genuine
d=2048 value; the corrected mixer/MLP shares **11.8% / 63.9%** hold; GQA is **exactly 3.000 d²** per
layer at d=1024 and using 2.5d² undercuts the ledger by **exactly 3,145,728**; **r=512 saves exactly
zero** (`4dr = 4·1024·512 = 2,097,152 = 2d²`, bit-for-bit — the general condition is `r ≥ d/2`); the
percentage denominator is the full 354.5M including the tied embedding.

Nice mechanism the child pinned down for the GQA coefficient: it is `2 + 2·(hkv·hd/d)`. With
`hkv·hd = 512` fixed at every scale, that is `d/4` at d=2048 → **2.5d²**, but `d/2` at d=1024 →
**3.0d²**. Same absolute KV width, half the model width, so the coefficient *rises* — which is also
the reason KV bytes/token is scale-invariant.

🔴 **NEW ERROR, in the reassessment itself.** `00_SYNTHESIS.md` §A.2 corrects "r=32 → 6.94%" to
**4.91%**. That is wrong: the true r=32 figure at d=1024 is **5.55%** (19,660,800 / 354,483,968).
The 4.91% looks to have been produced by rescaling 6.94% by the 6.27→4.44 ratio (0.7077 × 6.94 = 4.91)
instead of recomputing. **Consequence for the narrative:** the rank-sweep spread at d=1024 is
**1.11 pp (4.44 → 5.55)**, *wider* than the 0.67 pp advertised at d=2048 — so "savings saturate" is
**less** true at the frozen geometry, not more. Write 5.55%.

**The "error vs unlabelled scope" dispute (§A.2 said ERROR, §C.1 said mislabel) is adjudicated SPLIT.**
The clearest genuine ERROR: the memory note
`~/.claude/projects/.../memory/liv-experiment-key-numbers.md:30-31` states 14.3% / 68.8% / 6.27% with
**no scale qualifier**, in a file about the frozen program — and **self-contradicts at line 150 of the
same file** ("the decode ceiling at our chosen 350M/d=1024 geometry is 4.44%, not 6.27%"). One file,
two mutually exclusive claims, 120 lines apart. Other hits are correct-at-d=2048 but unlabelled.
The child's file carries the full file:line inventory for the correction pass.

---

### ORCHESTRATOR CHECK 6 — "5 of 9 gates are fail-open": the SUBSTANCE holds, the COUNT does not
(Complements child `06_claim4_stats.md`, which owns the power arithmetic.)

I read the risk audit's own gate table (`07_risk_audit.md:439-457`) and tallied its per-row verdicts:

| category | gates | n |
|---|---|---:|
| REAL | G2, G7 | 2 |
| REAL but unevaluable | G1 | 1 |
| PASS-ONLY | G4, G8 | 2 |
| FAIL-OPEN (unambiguous) | G3, G6, G9 | 3 |
| "half REAL / half FAIL-OPEN" | G5 | 1 |
| **total rows** | | **9** ✅ |

**The audit's own summary line reads "2 REAL (G2, G7), 1 REAL-but-unevaluable (G1), 5 FAIL-OPEN, 2
PASS-ONLY" — which sums to 10 across 9 gates.** Only **3** rows are marked unambiguously fail-open;
a 4th (G5) is marked half. To reach 5 you must also count G5 in full *and* re-classify a PASS-ONLY row.
Elsewhere the same document says "all six gates" and "rewrite all six" (`:485`, `:773`), a third count.

**Adjudication: the claim is directionally CONFIRMED but the headline number is not reproducible.**
The defensible statement is *"3 gates are unambiguously fail-open, a 4th is half fail-open, and 2 more
are pass-only — so 6 of 9 gates cannot stop anything."* That is arguably a **worse** finding than "5 of
9 fail-open", so the fix strengthens the audit. The substantive point — the gates are non-inferiority
criteria confirmed by failing to reject, and the free fix is **CI-upper-bound vs an explicit Δ** — is
sound and should be adopted regardless.

Worth flagging for the human as the genuinely useful observation in this section: **G2 — the one gate
that ever killed anything — was a microbenchmark with a pre-committed numeric threshold.** Every gate
attached to a *training* result is soft. (Sharp irony now that G2's kill has itself been reversed by
the L2-cache finding: the plan's single functioning gate fired on an artifact.)

---

### ORCHESTRATOR CHECK 7 — the fail-open POWER arithmetic re-derived independently → **CONFIRMED**

All computed on FarmShare (no scipy there, so the t-tests are 300–400k-draw Monte Carlo against exact
t critical values). MEASURED:

| audit's claim | my re-derivation | verdict |
|---|---|---|
| MQAR `N512_D64` seed spread σ ≈ **39.3 pp** | **sample SD = 39.30 pp** (pop SD 35.15, mean 37.6) on 0.05/0.09/0.20/0.56/0.98 | ✅ exact |
| n=2 paired rejects iff the two differences agree within **1.171 : 1** | t=(d₁+d₂)/\|d₁−d₂\| > 12.7062 ⇒ ratio < **1.1708** | ✅ exact |
| Fisher 2-vs-2 gives **p = 1/6**, can never reach α=0.05 | most extreme one-sided p = 1/C(4,2) = **0.16667** | ✅ exact |
| family-wise error over 11 comparisons = **43%** | 1 − 0.95¹¹ = **0.4312** | ✅ exact |
| "at n=5, ~1 in 4 catastrophic 60-pp regressions pass" | paired **two-sided**: miss = **0.274** | ✅ reproduces |
| "at n=2, 91% pass" | paired two-sided **0.865**; unpaired two-sided **0.904** | ✅ substantively |

⚠️ Minor methodological inconsistency worth knowing: the audit's two power figures are not from one
assumption set — n=5's "1 in 4" matches **paired two-sided**, while n=2's "91%" matches **unpaired**
two-sided (paired gives 86.5%). Under a **one-sided** test — which is the correct test for a
non-inferiority gate — the misses are materially lower (n=2: 0.734, n=5: **0.128**). So the audit's
n=5 figure is roughly 2× pessimistic against the test it should be using.

**This does not rescue the gates.** Even at the most favourable assumption, **n=2 lets ~73% of a
catastrophic 60-pp regression through**, and the plan's 12-arm screen runs at n=2. The recommendation
(CI-upper-bound vs an explicit Δ; drop the 2-seed row) stands, and it is free.

---

### CLAIM 4a — "the CE margin is vacuous" → **CONFIRMED on the arithmetic, OVERSTATED in rhetoric**
Child: `06_claim4_stats.md` (hand-rolled t-CDF validated against published tables to 5 sf; non-central
t power by numerical integration — no scipy on FarmShare).

**The arithmetic holds.** The margin is real (`design doc:1133`, +0.010 nats). Traced to its primary
source — Mamba-2 (arXiv 2405.21060) Table 2, 350M/48 layers/7B tokens/Pile — the basin is
`ln(8.32/8.26) = 0.00724 nats`. **Margin exceeds basin by 1.38×**, so the acceptance region does
contain the ratio-sweep phenomenon. A sibling audit reaches 0.00598 nats by a different route; both
< 0.010, so the conclusion is robust to the baseline choice.

🔴 **But the dramatic sentence is false as stated.** "It would declare an all-conv model non-inferior to
a transformer" is true **only** against Mamba-2's Transformer++ row (0.00926 nats, clears by 7%) — and
that row is *worse* than the best hybrid, so it is arguably the wrong comparison. Against the relevant
contrast, **pure-SSM vs best-hybrid = 0.0403 nats = 4× the margin**, which the gate **would** catch.
So the acceptance region contains the **basin**, not the **table**.

**And the consequence is low.** §6.1 is titled *"Held-out CE is almost certainly underpowered — do not
make it the primary endpoint"*, and line 1136 already pre-commits to reporting "inconclusive."
→ Correct action: **delete the CE gate** (it can only mislead). Do **not** conclude the decision
structure is broken.

**s_δ / KDA cross-reference CONFIRMED and transferable-with-care.** Both KDA numbers are real and
in `KDA/HANDOFF.md`; back-solving gives `s_δ ≈ 0.0112–0.0124` against a requirement of `≲0.0114`
(the child reproduced the doc's "≲0.011" exactly as `0.010·√8/2.48647 = 0.011375`). **The requirement
is straddled** — so the planned Phase-2 pilot would spend GPU-days re-deriving what is on disk.

### CLAIM 4b — "5 of 9 gates are fail-open" → **SUBSTANCE CONFIRMED, DENOMINATOR WRONG**

Child 6 independently reached my Check-6 conclusion. The gates are `design doc:1373-1383` — **9 table
rows**, but ~**20 separately-decidable clauses**, and the audit's own §3.1 says **six** are
non-inferiority. "5 of 9" counts cells; "six criteria" counts clauses; both appear in one document.
Also **the audit's G-numbering is off by one vs document order** (it places Phase 0b in table order;
the doc lists it at line 1378, after Phase 2) — so "G2 is the one that fired" is **G4** in document
order. Anyone cross-referencing must know this.

Quantitative sub-claims (child's numbers, matching mine): n=2 ratio **1.170850** ✅ exact; Fisher 2v2
**p=1/6** ✅ exact; FWER **43.12%** ✅ exact; n=5 "1 in 4 pass" ✅ (27.5% two-sided). **n=2 "91%" does
NOT reproduce** — 86.5% paired two-sided (90.3% only if unpaired). The child also flags that the audit
silently assumes ρ=0 correlation between paired arms when setting `s_δ = σ`; at ρ=0.5 the misses fall.

**Verdict unchanged in direction:** the majority of training-attached gates cannot fail, and the free
CI-upper-bound-vs-Δ fix should be adopted. Just do not quote "5 of 9" or "91%".

**Additional detail from the child's final pass:**

🟢 **A finding FAVOURABLE to the plan that the risk audit missed.** The **parent protocol**
(`docs/liv-kda-gqa-sub500m-experiment.md:386,396`) *already states the gate in the CI-upper-bound
form* the audit proposes as its headline fix: `U95(CE_K2 − CE_L0-P) <= +0.010 nats`, and likewise for
recall (`LCB ±2.0`, lines 387/397). **The fail-open defect was introduced in TRANSCRIPTION into the
LIV design doc, not designed in.** So the fix is not new methodology — it is restoring what the parent
document already said. That makes it near-zero-effort and uncontroversial.

🔴 **The audit's "0.0030–0.0072 nats" is mislabelled.** It is not an interval estimate of the basin;
it is *the same 0.06 ppl* expressed at two assumed baselines (8.33 and 20.0). The single correct value
at the actual Mamba-2 baseline is **0.00724 nats**. Quote that.

🔴 **Where the "91%" came from.** The child traced it: 9% is the correct *power* for a **one-seed-SD**
effect (it reproduces 9.06%), which was then transposed onto the **60-pp** scenario. Two different
effect sizes were conflated. The 60-pp figure is 86.5%.

⚠️ **Two cautions for the synthesis, both reducing the strength of the "cut the pilot" advice:**
1. **The two KDA inversions are NOT independent** — the same `s_δ` is stated twice in different forms.
   So "the pilot only confirms what is already on disk" **overreaches**. The *vacuity* conclusion
   survives the scale-transfer objection; the *"already measured, unreachable"* conclusion does not.
2. Synthesis row 16 silently drops the audit's own §2.8 caveat that **the 39.3-pp variance argument
   holds only if MQAR is trained from scratch.** If MQAR is an *eval* on pretrained arms, the power
   picture changes materially — and the plan still never says which. **One sentence of clarification
   is worth more than any experiment here** (the prior team said this too, and it survives review).
3. Minor: the quoted `s_δ` range "0.0113–0.0126" should be **0.0112–0.0124**.

---

### CLAIM 4d — "'No LFM2 paper exists' is wrong; arXiv 2511.23404 does" → **CONFIRMED**
Child: `07_claim4_arith.md`.

The self-contradiction is real and **worse than reported: three sites in `06_baselines_infra.md`**
(`:118`, `:157`, `:1946`), not one. Line 1946 is a risk-register item asserting "Everything about LFM2
in this document comes from `config.json`, the HF model card, and the transformers implementation" —
i.e. the whole document was written as if no paper existed. Note `2511` = **November 2025**, so this
ID is not even future-dated; it was simply missed.

**Reading the paper UPGRADES the gap claim from inferred to verified — with one caveat the prior team
did not state.** Ablation inventory (MEASURED from the paper):
- **(i) attention:conv ratio ablation — NO.** ✅ gap holds. §2.1 describes an architecture search whose
  outcome is reported **only as prose** ("repeatedly selects a minimal hybrid"); Table 1 lists final
  counts as hyperparameters. No per-ratio numbers, no sweep, no plot.
  🟢 **Free ammunition nobody has used:** Table 1 shows the attention *fraction* **falling with scale
  (37.5% → 26.7% → 25.0%)** — an unexplained inconsistency inside Liquid's own family.
- **(ii) kernel-WIDTH ablation — NO.** ✅ gap holds, and it is stronger than assumed: the search space
  explicitly names gated short convolutions, yet **no width is ever swept** and Table 1 pins **k=3 for
  every model at every scale, dense and MoE.**
- **(iii) retrieval/recall — NO for the LLMs, ⚠️ BUT WITH A CAVEAT.** Tables 6–7 contain zero
  long-context or recall benchmarks (no RULER/NIAH/passkey/LongBench/MQAR) despite a 32K context claim.
  **However Tables 13–15 DO report retrieval numbers (NanoBEIR, NDCG@10, 13 tasks)** — for
  **LFM2-ColBERT-350M**, a *separate late-interaction retrieval encoder*, and NanoBEIR is corpus-level
  document ranking, not in-context recall.
  → **The claim must be stated precisely as "no in-context recall / long-context retrieval benchmark
  for the LFM2 LLMs."** Saying "LFM2 publishes no retrieval numbers" is refutable in one search and
  would badly damage credibility in review.
- **(iv) "matches attention-heavier baselines" — CONFIRMED as unsupported prose.** No supporting table.

⚠️ Confidence caveat the child states honestly: Tables 1–9 were read directly; Tables 10–16 came via a
secondary index (alphaxiv) cross-checked against the ToC. High confidence on (i)/(ii)/(iv) — which are
the load-bearing ones — medium-high on the Table 13–15 detail.
**Combine with ORCHESTRATOR CHECK 1: the abstract's "hardware-in-the-loop architecture search" means
the honest framing is "Liquid did not PUBLISH the ablation," not "nobody knows."**

### CLAIM 4e — "'695 GB/s' is GiB/s" → **CONFIRMED as a unit bug; REFUTED as a fix worth making**

All four documented figures match the GiB/s column and none match GB/s: **695↔694.8, 161↔160.5,
128↔127.7, 205↔205.2.** The bug is isolated to `p1_verify.py` (the child grepped every probe script).

**But do NOT propagate the "86.4% of peak" correction.** The already-settled L2-cache finding means the
benchmark never reached HBM, so comparing *any* achieved rate to the 864 GB/s HBM spec is meaningless —
correcting 80% → 86.4% would just replace a wrong number with a differently-wrong number that sounds
more authoritative. **Delete the "% of peak" claim rather than fix it**, and report the out-of-cache
462 GB/s from the scaled re-run instead. (The two corrections in `00_SYNTHESIS.md` — "it's GiB/s, raise
to 86.4%" in §A.5 and "it was measuring L2" in §A.5b — are mutually inconsistent as written; §A.5b
supersedes.)

**Child's added detail:** the bug is two-part, `p1_verify.py:81` (`/2**20`) and `:87` (`×1e6/1024`),
and recomputing that expression reproduces the printed output bit-for-bit. It is isolated —
`p1_cache_check.py:180` and `p1_scaled.py:137` emit true GB/s.
🔴 **This creates a NEW trap for the correction pass:** the repo now mixes GiB/s and GB/s under
identical labels, so "695 old vs 745 new" *looks* like a 7.2% improvement when `p1_scaled`'s 40 MiB
rung (744.7 GB/s) and job 1670884's dense (746.0 GB/s) are **the same measurement to 0.2%.** Fix the
labels for cross-job comparability, then delete the "% of peak" sentence *and* the dependent "skinny
GEMVs cannot saturate the memory system" clause. Corrected ratio is 86.3%, not 86.4% — moot anyway.

---

### CLAIM 3 (child's full treatment) — **CONFIRMED on walltime + concurrency; one sub-claim REFUTED**
Child: `05_claim3_compute.md`. All literal cluster output; no GPU job enqueued (`sbatch --test-only`).

Agrees with my ORCHESTRATOR CHECK 2 on every figure, and adds three things that matter:

🟢 **A `--qos=long` exists that nobody in the project knew about: MaxWall = 7 DAYS** (it overrides the
partition limit via `PartitionTimeLimit`). This is a materially larger envelope than even the prior
team's corrected claim.

🔴 **Sub-claim REFUTED: `-c 8 --mem=48G` is NOT rejected.** `MaxMemPerCPU=4000` causes Slurm to
**silently auto-bump the request to 14 CPUs**, not to deny it. So the recipe in HANDOFF and the probe
docstrings *does* run — the prior team's "the printed recipe cannot run as written" is wrong. (Worth
knowing anyway: the job quietly consumes 14 CPUs of your allocation, not 8.)

Also: **preemption is OFF**; multi-node GPU is effectively barred (2 nodes × 4 GPUs = 8 > the cap of 4);
hardware confirmed as 24 L40S cluster-wide (48 GB, CC 8.9). **"6 h / 1 GPU" appears NOWHERE in the
Slurm config** — the nearest real numbers are `DefaultTime=02:00:00` (what you get if you omit `-t`)
and QOS `normal`'s `gres/gpu=1`. That is almost certainly the origin of the myth.

**FEASIBILITY DELTA** (6ND; L40S bf16 dense peak 181 TFLOP/s; 25% MFU realistic / 40% optimistic; one
48 h × 4-GPU window = 192 L40S-h. Reality check from this cluster's own history: the KDA gated-delta-net
arm achieved ~10% MFU, so 25% is generous):

| stage | L40S-h @25% | calendar days (realistic ~3 GPU) | status |
|---|---:|---:|---|
| **(d) cheap replacement** — 4 arms × 350M × 2B × 8 seeds | **834** | **~12 d** | ✅ **NEWLY FEASIBLE.** One run = 26.1 h — impossible in 6 h, comfortable in 48 h with 22 h headroom. **No checkpoint-resume needed.** |
| (a) rank screen 24 × 150M × 10B | 1,324 | ~18 d | ✅ feasible, slow |
| (b) confirm 5 × 350M × 20B | 1,304 | ~18 d | ✅ feasible, slow |
| (c) headline 3 × 750M × 50B | 4,139 | **~57 d** | ❌ **STILL INFEASIBLE.** No submit form on this cluster fits one run; needs checkpoint-resume across ~3 seven-day jobs, ~2 months saturated. |
| **full (a)+(b)+(c)** | **6,767** | **~94 d** (70.5 perfect; 58.8 even at 40% MFU) | ❌ out of reach |

For scale: the same program on 8×A100 @40% is **12.8 days / ~2,450 A100-h**. FarmShare's 4×L40S is
**~5.5× slower in calendar time.**
⚠️ Multi-GPU is not free: total GPU-hours rise ~**1.18×** (PCIe all-reduce) while wall-clock falls —
use it to fit a run inside a window, not to save budget.

**So the relaxation is real and it changes exactly one decision: the cheap 4-arm × 8-seed study is now
comfortably executable on FarmShare with no checkpointing and no SB-AWS.** The 8-day headline program
remains unhostable. This vindicates the prior team's *recommendation* while correcting its premise.

**Child's final pass — three corrections and one new trap, all MEASURED against real jobs:**

1. 🔴 **"`-c 8 --mem=48G` is rejected" is REFUTED with a completed job as proof.** Job **1671018_0**
   requested `cpu=8,mem=48G` and was **allocated `cpu=14`** — Slurm auto-bumps, it does not deny. The
   real defect is *silent over-allocation* against your 4-job quota, not failure. Also **my own
   `-c 12` suggestion was wrong**: 49152/4000 → 13 → rounds to **14** under `ThreadsPerCore=2`.
   **Corrected submit line:**
   `sbatch -p gpu --gres=gpu:1 -c 14 --mem=48G -t 48:00:00`  (add `--qos=long -t 7-00:00:00` for >48 h)
2. **`--qos=long` = 7 days is real and live-proven** (job 1664922 ran 4-00:55 against a 5-day limit).
   Used by only 11 of 5,524 GPU jobs in 30 days — effectively an unused resource.
3. 🔴 **NEW TRAP nobody flagged: `MaxSubmitPU=32` counts ARRAY ELEMENTS, and `%K` does not exempt
   them.** So **`--array=0-47%4` is rejected outright.** This breaks any 24-or-32-run campaign script
   written the obvious way — directly relevant to the (a) rank screen and the (d) 32-run study.
4. **"KDA already ran 20-hour jobs" CONFIRMED** (job 1662404_13, 20:00:20 against a 20 h limit).
   **4 concurrent is real, not nominal:** four array elements started in the same second and sustained
   **2.99 GPUs over 43.8 h**. Median 1-GPU queue wait **1.2 min** (n=570). `PreemptMode=OFF`.
5. **Origin of the myth identified:** "6 h" appears nowhere in Slurm; it traces to an unverified
   assertion at `docs/liv-brainlift-experiment-design.md:1401`. The "1 GPU" half almost certainly comes
   from `-p normal`, whose QOS caps `gres/gpu=1` (and PriorityTier 1 vs 5). **Never use `-p normal` for
   GPU work.**
6. Spec correction worth keeping: **L40S bf16 dense is 181 TFLOP/s**; the 362 datasheet figure is
   *with sparsity*. Gap to the 8×A100 program is **3.4× at best, 7.3× realistic**.
7. One extra feasibility nuance: **(d) at full 350M is 26.1 h/run and fits a plain 48 h 1-GPU job with
   no resume — the 6 h belief had forced a retreat to 20–50M models.** (b) fits only as
   `--gres=gpu:4 --qos=long` (76.7 h); at 4 GPU/48 h/40% MFU it is exactly 48.0 h, i.e. unsafe.

---

## BOTTOM LINE

### Reliability of the prior team
**No fabrication.** 6/6 arXiv IDs I checked resolve with the exact claimed titles; the child-level
citation audit ran 11/11. Every checkpoint, script and job referenced is physically on disk. Where the
prior team is wrong, it is wrong by **over-reading a real source**, never by inventing one. Its
arithmetic reproduces essentially everywhere (the one exception: its own r=32 correction, 4.91% → 5.55%).

### The pattern in what broke
Four of the prior team's strongest-sounding claims fail the same way: **a real primary source is
stretched one step past what it supports.**
- 2606.06467 is real and correctly transcribed — but changes 3 variables at once, so it cannot carry
  "opposite sign."
- Tian Table 7 is real and correctly quoted — but adds *narrower* branches, so it is not the *span*
  question P3 asks.
- The conv-tap read is arithmetically perfect — but is weights-only, so it cannot carry "inside the gate."
- Hymba/CAI/Gemma really do share KV — but none publishes a sharing-isolated retrieval number, so they
  cannot carry "anticipated, therefore cut."

**This is the same failure mode the project already diagnosed twice** (the P1 spectra episode; the
L2-cache episode): a measurement that is *correct in its own frame* gets promoted to answer a question
its frame cannot reach. It has now happened a third time, in the document written to catch it.

### Net effect on what to do next
- **P2 is more alive than the reassessment concluded.** Its residual contribution is now sharper and is
  a gap of *question*: **no one isolates KV sharing's retrieval effect with positional encoding and
  attention pattern held fixed.** Still weigh against the capacity-not-bandwidth economics — that was
  never disputed and remains P2's real problem.
- **P3 stays deprioritized, but on the published width sweeps, not on the tap read.** Two ~0.2–0.3
  GPU-h probes (C3, C5) would settle the interpretive gap outright.
- **Compute is the biggest actionable change.** 48 h × 4 GPUs (7 days via `--qos=long`) makes the
  350M × 2B × 8-seed × 4-arm study (~834 L40S-h, ~12 days, **no checkpointing**) genuinely runnable —
  the 6 h myth had forced a retreat to 20–50M toy models. The 750M headline stage remains out of reach.
- **The statistical fixes are free and should just be done** — and the CI-upper-bound form already
  exists in the parent protocol, so it is restoration, not new methodology.

### Corrections the prior team's own output needs
1. `00_SYNTHESIS.md` §A.2: r=32 is **5.55%**, not 4.91%.
2. `00_SYNTHESIS.md` §A.5: strike "raise the claim to 86%" — §A.5b moots it. Report 462 GB/s / 53%.
3. `06_p2_p3_verdict.md` §5.5: retract "measured inside Liquid's actual double gate."
4. `06_p2_p3_verdict.md` §0.1: downgrade "the steelman has been run" to "an adjacent negative prior."
5. `06_p2_p3_verdict.md` §1.5: add the RNoPE/SWA confound; drop "opposite sign."
6. `07_risk_audit.md` §3: the score line sums to 10 across 9 gates; "5 of 9" → "6 of 9 cannot stop
   anything." Drop "91%" (it is 86.5%, and the 9% was a different effect size).
7. `04_cheap_experiments.md` §4.6: the four-checkpoint numbers are a cross-scale consistency check,
   not an independent replication; the 2.6B boundary-argmax row is a tie artifact (9.75% → 2.03%).
8. Anywhere: "no retrieval benchmark for LFM2" → "no **in-context recall** benchmark for the LFM2
   **language models**" (Tables 13–15 report NanoBEIR for LFM2-ColBERT).
9. `05_claim3_compute.md` supersedes: `-c 8 --mem=48G` is accepted (bumped to 14 CPUs), not rejected.
