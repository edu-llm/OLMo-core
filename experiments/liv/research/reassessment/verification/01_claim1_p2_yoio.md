# Verification 01 — Claim 1: P2's novelty gap "is now FALSE" because of arXiv 2606.06467

**Status: FINAL**
**Verifier:** independent verification agent, 2026-08-01
**Target:** `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/reassessment/06_p2_p3_verdict.md` §1.5 (lines 152-217)
**Constraint honored:** no code executed on the local Mac beyond `curl` fetches + text parsing of fetched
HTML. No FarmShare compute was needed — this was a literature-verification task end to end.

---

## 1. Existence verdict for arXiv 2606.06467 — **EXISTS, EXACTLY AS DESCRIBED**

| probe | URL | result |
|---|---|---|
| arXiv API | `http://export.arxiv.org/api/query?id_list=2606.06467` | **HTTP 200**, `totalResults=1`, one `<entry>` |
| HTML full text | `https://arxiv.org/html/2606.06467v1` | **HTTP 200**, 265,239 bytes, full LaTeXML render |

Metadata returned by the API (MEASURED, verbatim):

- **Title:** *You Only Index Once: Cross-Layer Sparse Attention with Shared Routing* ✓ matches
- **Authors:** Yutao Sun, Yanqi Zhang, Li Dong, Jianyong Wang, Furu Wei ✓ matches (prior team wrote
  "Sun, Zhang, Dong, Wang, Wei" — correct, in order)
- **Published:** `2026-06-04T17:54:04Z` ✓ matches "4 Jun 2026"
- **Categories:** cs.CL (primary), cs.AI, cs.LG
- **Affiliation:** not in the API metadata. Li Dong + Furu Wei are the MSRA/Microsoft Research
  foundation-model group and the paper is a direct YOCO follow-on (YOCO is Sun/Dong/Wei, MSRA), so
  MSRA is a well-founded **INFERENCE**, not a MEASURED fact from the paper HTML. Minor.

**The prior team did not fabricate this paper.** The citation is real, correctly dated, correctly
attributed, and correctly titled. That is a meaningful positive finding about their reliability.

---

## 2. Row-by-row table comparison — **PRIOR TEAM'S TRANSCRIPTION IS EXACT (72/72 cells)**

Source: Table 3 of the HTML (`id="S3.T3.3"`), caption *"RULER results at 16K and 32K context lengths.
TRM denotes the standard Transformer. CLSA maintains strong single-needle retrieval performance and
achieves the best average score at 32K, with gains mainly from the harder multi-needle settings."*

| ctx | model (paper's label) | S1 | S2 | S3 | MK1 | MK2 | MK3 | MQ | MV | QH | QS | CWE | FWE | Avg | match? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16K | TRM | 100.0 | 99.8 | 98.4 | 88.2 | 71.4 | 14.4 | 85.7 | 85.6 | 28.8 | 33.2 | 15.6 | 52.1 | 64.4 | ✓ |
| 16K | YOCO (Dense) | 100.0 | 99.8 | 96.4 | 69.4 | 91.6 | 61.2 | 45.8 | 49.3 | 30.8 | 31.4 | 9.4 | 67.0 | 62.7 | ✓ |
| 16K | YOCO (CLSA) | 100.0 | 100.0 | 98.4 | 70.4 | 92.4 | 58.4 | 53.0 | 47.2 | 31.2 | 32.7 | 9.8 | 61.6 | 62.9 | ✓ |
| 32K | TRM | 100.0 | 98.8 | 83.4 | 57.0 | 38.8 | 0.8 | 45.6 | 42.6 | 21.2 | 20.2 | 1.8 | 43.8 | 46.2 | ✓ |
| 32K | YOCO (Dense) | 100.0 | 90.2 | 74.8 | 53.2 | 84.0 | 43.6 | 27.0 | 29.0 | 30.6 | 30.6 | 4.6 | 60.3 | 52.3 | ✓ |
| 32K | YOCO (CLSA) | 100.0 | 93.6 | 83.2 | 58.4 | 88.8 | 38.0 | 31.6 | 29.8 | 29.2 | 29.2 | 5.1 | 50.2 | 53.1 | ✓ |

**Every one of the 72 numeric cells matches.** One cosmetic deviation: the prior team relabeled the
paper's `TRM` as "Transformer (no sharing)" and `YOCO (Dense)` as "YOCO (Dense, KV-sharing)". Those
glosses are substantively correct (see §3) but they are the prior team's words, not the paper's.

### 2.1 Independent re-addition of the Avg column — **CONSISTENT**

I recomputed each Avg as the unweighted mean of the 12 subtasks:

```
16K TRM           64.433  vs stated 64.4   (+0.03)
16K YOCO (Dense)  62.675  vs stated 62.7   (-0.03)
16K YOCO (CLSA)   62.925  vs stated 62.9   (+0.03)
32K TRM           46.167  vs stated 46.2   (-0.03)
32K YOCO (Dense)  52.325  vs stated 52.3   (+0.03)
32K YOCO (CLSA)   53.092  vs stated 53.1   (-0.01)
```

All six agree to within rounding of a 1-decimal display. **The table is internally consistent** — the
Avg column is a genuine mean of the printed subtasks, not a pasted-in number. No arithmetic laundering.

### 2.2 One derived number in the prior team's prose is slightly off

The prior team wrote "KV sharing is **−1.7 avg at 16K but +6.1 avg at 32K** (52.3 vs 46.2)". That is
the **YOCO (Dense)** row, and it is right: 62.7−64.4 = −1.7; 52.3−46.2 = +6.1. Fine. But their
narrative then credits the effect to "CLSA," whose deltas are −1.5 and **+6.9**. Immaterial to the
argument, but the two rows are conflated in their prose.

They also wrote the gains are "concentrated in hard multi-needle subtasks (MK2 38.8→84.0, MK3
0.8→43.6)." Those cells are correct. **However the paper's own sentence names "especially MK1 and
MK2"** — not MK3. Verbatim from the HTML: *"The improvement is mainly driven by stronger robustness
on the more difficult multi-needle settings, especially MK1 and MK2."* The prior team substituted MK3
for MK1. Since the prior team's MK3 numbers are *larger* deltas than MK1's, this is a substitution
that flatters their own argument, though the underlying cells are real.

---

## 3. Is it actually cross-layer KV sharing? — **YES, but YOCO-style, and the control is CONFOUNDED**

### 3.1 What is shared (MEASURED, from §2 and Fig. 1 of the HTML)

> "The **self-decoder first produces a shared KV cache, which is computed only once and then reused
> by all subsequent cross-decoder layers.** … a shared query-aware indexer jointly generates the
> routing queries and keys and computes a token-level sparse top-k index for each query token. This
> sparse index is also produced only once and is shared across [layers]."

Config (MEASURED, §3.1 + Appendix C Tables 7/8): 4B; hidden 2560; FFN 7680; 32 layers; 20 heads;
**4 KV heads**; head_dim 128; QK-norm on; no weight tying. **For the YOCO variants the 32 layers split
into 16 self-decoder + 16 cross-decoder layers.**

So: **one global KV bank produced by the 16-layer self-decoder, consumed by all 16 cross-decoder
layers.** This is *genuine* cross-layer KV sharing — the `YOCO (Dense)` row is a real KV-sharing model,
not a mislabeled baseline. The prior team's caveat #2 (line 202-205) is correct: this is YOCO-style
one-bank-to-all-uppers, **not CLA-style pairwise banks between adjacent attention layers.**

### 3.2 The +6.1 is NOT apples-to-apples — this is the finding the prior team missed

Matched: width, depth, head count, KV heads, head_dim, QK-norm, weight tying, and the training recipe
(dense stage 1: 8M tok/batch, seqlen 8192, LR 3e-4→3e-5, 2000 warmup, 125,000 updates; dense stage 2:
context 32,768, LR fixed 3e-5, 10,000 updates). Params are never stated numerically but geometry is
aligned, so ≈matched. Training tokens are matched. Good so far.

**NOT matched — Appendix C Table 8, verbatim:**

| | Transformer | YOCO (Dense) | YOCO (CLSA) |
|---|---|---|---|
| Positional encoding | **RoPE** | **RNoPE** | **RNoPE** |
| RoPE base | **5×10⁵** | **1×10⁴** | **1×10⁴** |
| Attention type | GQA | GQA | GQA+CLSA |

Plus, from §3.1: *"Both YOCO variants use **sliding window attention in the self-decoder with window
size 512**."*

And Appendix C states it plainly: *"**The key practical difference lies in how positional information
is handled**: the Transformer applies RoPE throughout, while YOCO (CLSA) uses an RNoPE setting that
restricts RoPE to the sliding-window self-decoder and **removes positional encoding from the global
cross-decoder attention path**."*

This is decisive. The YOCO arms differ from the baseline in **at least three** ways simultaneously:
1. **KV sharing** (the variable of interest),
2. **RNoPE vs RoPE** — NoPE on the global attention path,
3. **SWA(512) in the lower half** vs full attention throughout.

RNoPE (`arXiv 2501.18795`, "RoPE to NoPE and Back Again") **exists precisely because removing RoPE
from global-attention layers improves long-context retrieval.** It is a known length-generalization
intervention. And the baseline's RoPE base of 5×10⁵ vs the YOCO arms' 1×10⁴ is a further
length-extrapolation difference.

**Therefore the +6.1 at 32K cannot be attributed to KV sharing.** The single most likely explanation
for a gain that *appears only at 32K* (i.e. exactly where the stage-1 8K training context is exceeded
by 4×) and is *concentrated in multi-needle subtasks* is the **RNoPE/NoPE positional treatment**, which
is the textbook signature of that intervention. The paper runs **no ablation** separating sharing from
RNoPE from SWA — I searched the HTML; there is none. The paper also never claims sharing causes the
gain; that causal reading is the prior team's, not the authors'.

This is a **confounded three-way comparison being read as a one-way one.**

### 3.3 The paper's own framing is about efficiency, not a sharing-vs-retrieval study

The abstract's claims are 7.6× decoding speedup and 17.1× throughput at 128K. RULER is one of several
evaluation benchmarks (alongside ARC-C, BBH, GSM8K, HellaSwag, HumanEval, MMLU, DROP, WinoGrande).
The paper's contribution is **sharing the routing index**, not establishing whether KV sharing helps
or hurts retrieval. The RULER table is incidental to its thesis. It is real evidence, but it is
**not "the first paper that actually measured it"** in the sense of a controlled study, because the
control is not clean.

### 3.4 MQAR / passkey — **absent, confirmed by exhaustive string search**

Full-text counts in the HTML: `MQAR` → **0**, `passkey` → **0**, `Passkey` → **0**,
`associative recall` → **0**, `needle in a haystack` → **0**. (`Needle` → 2, both the RULER
subtask-group column header.) The prior team's caveat #1 is **CONFIRMED**.

---

## 4. VERDICT on "P2's headline gap no longer exists"

### **PARTIALLY CONFIRMED — the literal headline is dead; the prior team's causal reading is REFUTED.**

Split into two separable propositions:

**(a) "No cross-layer-KV-sharing paper reports needle/passkey/MQAR retrieval" → FALSE. CONFIRMED.**
2606.06467 reports all 12 RULER subtasks at two context lengths for a from-scratch KV-sharing model
against a non-sharing control. CommonKV reports RULER. DepthWeave-KV reports Needle-in-a-Haystack.
The claim as written in the design doc (line 923-927) is factually overtaken. **The prior team is
right on this and the capstone must stop making that claim.**

**(b) "The question has now been answered in the direction opposite to the capstone's worry" →
NOT SUPPORTED. REFUTED as stated.** The +6.1 rides on a confound (RNoPE + SWA + sharing, all changed
at once, no ablation). The prior team asserted this as MEASURED; it is at best a confounded
observation, and the most parsimonious explanation credits RNoPE rather than sharing. The prior team's
strongest sentence — *"a reviewer who knows 2606.06467 will ask why 350M/CLA-pairwise is expected to
behave differently from 4B/YOCO, and the honest answer is 'we don't have a mechanism'"* — **overstates
the threat.** The correct response to that reviewer is: *2606.06467's comparison also changes the
positional encoding and the attention pattern, so it does not isolate sharing.* That is a good answer
and a reviewer would accept it.

### What would settle it
An ablation holding RNoPE + SWA fixed and toggling only KV sharing. Not present in 2606.06467, and I
found no such ablation in any of the audited papers. **This gap is real and is arguably a better
framing for P2 than the original one.**

---

## 5. What actually remains of P2's contribution

Surviving sub-claims, ranked by how much a reviewer would credit them:

1. **[STRONGEST — a gap of *question*, not configuration]** *No paper isolates the effect of cross-layer
   KV sharing on retrieval with a clean control.* 2606.06467 confounds it with RNoPE and SWA. This
   survives fully and is **better than the original P2 pitch** because it is a causal-identification
   claim rather than a "nobody ran the benchmark" claim. The prior team did not notice this and
   consequently under-rated what is left.
2. **[MEDIUM]** *MQAR specifically is unreported for any cross-layer-sharing architecture.* MEASURED-
   confirmed for 2606.06467 (0 occurrences) and for all nine audited abstracts. But this is a gap of
   *configuration* — a reviewer will say RULER is the benchmark they want anyway, and RULER is covered.
   Weak on its own.
3. **[MEDIUM]** *CLA-style pairwise sharing across an intervening conv block in a sequentially
   interleaved hybrid is unreported.* Gap of configuration. Defensible only if paired with a mechanism
   for why topology should matter — and note that FusedKV (2512.03870) already reports that cross-layer
   sharing "typically underperforms within-layer methods like GQA," which is a partial mechanism the
   capstone could build on.
4. **[WEAKEST]** *Nobody has done it at 350M / in an LFM2-shaped hybrid.* Pure scale/configuration.
   A reviewer will not credit this alone.

**Recommendation:** re-pitch P2 around #1. It is the only one of the four that is a gap of question,
and it is directly enabled by the very paper the prior team thought killed P2.

---

## 6. Citation audit — **11/11 RESOLVE. Hit rate 100%.**

All fetched via `http://export.arxiv.org/api/query?id_list=<id>` (note: `-L` is required; without
following the 301 redirect every ID appears to fail — an early false negative in my own run).

| ID | Resolves? | Title | Date | Retrieval benchmark in abstract? | Prior team's claim accurate? |
|---|---|---|---|---|---|
| **2606.06467** | ✅ | You Only Index Once: Cross-Layer Sparse Attention with Shared Routing | 2026-06-04 | RULER (in body, 12 subtasks @16K/32K) | ✅ yes, table exact |
| **2607.06523** | ✅ | DepthWeave-KV: Token-Adaptive Cross-Layer Residual Factorization… | 2026-07-07 | **LongBench, Needle-in-a-Haystack, L-Eval** — all three named | ✅ exactly as claimed |
| **2508.16134** | ✅ | CommonKV: Compressing KV Cache with Cross-layer Parameter Sharing | 2025-08-22 | **LongBench and Ruler** | ✅ exactly as claimed |
| **2604.22782** | ✅ | Stochastic KV Routing: Enabling Adaptive Depth-Wise Cache Sharing | 2026-04-03 | **none named** | ✅ prior team's caveat #4 correct |
| **2604.13556** | ✅ | YOCO++: Enhancing YOCO with KV Residual Connections… | 2026-04-15 | none named (perf @50% compression only) | ✅ listed, no claim made |
| **2512.03870** | ✅ | Reconstructing KV Caches with Cross-layer Fusion (FusedKV) | 2025-12-03 | none named in abstract | ✅ listed, no claim made |
| **2410.15252** | ✅ | Lossless KV Cache Compression to 2% (**CLLA**) | 2024-10-20 | none named in abstract | ✅ listed, no claim made |
| **2503.18893** | ✅ | xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction | 2025-03-24 (ICML 2026) | "long-context tasks", unnamed | ✅ listed, no claim made |
| **2507.08045** | ✅ | Krul: Efficient State Restoration… Dynamic Cross-layer KV Sharing | 2025-07-10 | none named in abstract | ✅ listed, no claim made |
| **2510.11236** | ✅ | XQuant: Ultra-Low Bit KV Cache Quantization with Cross-Layer Compression | 2025-10-13 (EMNLP 2025) | TruthfulQA, LongBench | ✅ listed, no claim made |

### Baseline anchors

| ID | Resolves? | Title | Authors | Claimed as | Verdict |
|---|---|---|---|---|---|
| **2405.12981** | ✅ | Reducing Transformer Key-Value Cache Size with Cross-Layer Attention | Brandon, Mishra, Nrusimha, Panda, Ragan-Kelly | "CLA" | ✅ **correct** — this is the CLA paper; abstract confirms "sharing key and value heads between adjacent layers … Cross-Layer Attention (CLA)" |
| **2410.14442** | ✅ | A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference | You Wu, Haoyi Wu, Kewei Tu | "NAACL 2025 systematic study" | ✅ **correct** — arXiv comment field literally reads "Accepted to NAACL2025 main conference" |

**Assessment of the prior team's reliability:** on citations, high. Every ID resolves, every title
matches, both baseline anchors are right, the 72-cell table transcription is perfect, and the Avg
column is internally consistent. Their errors are **not** fabrication — they are (i) an unnoticed
experimental confound in the cited paper, (ii) MK1→MK3 substitution against the paper's own prose,
and (iii) a Dense/CLSA row conflation. Errors of over-reading, not of invention.

---

## 7. Bottom line for the capstone

- Delete the claim "no cross-layer-KV-sharing paper reports needle/passkey/MQAR retrieval." It is
  false as of Jun 2026 and citing it will damage credibility.
- **Do not** accept the prior team's conclusion that the question is answered. It is not. 2606.06467
  changed three variables at once.
- Re-pitch P2 as: *the first clean isolation of cross-layer KV sharing's effect on retrieval, holding
  positional encoding and attention pattern fixed.* Cite 2606.06467 as motivation-and-confound, which
  is a stronger related-work posture than pretending it does not exist.
