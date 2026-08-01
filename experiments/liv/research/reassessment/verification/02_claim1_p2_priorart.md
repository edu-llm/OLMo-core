# Claim 1 / P2 — "anticipated 3x" prior-art verification

**Verifier:** verification agent #2. **Date:** 2026-08-01.
**Constraint honored:** no code executed on the local Mac. All fetches were `curl` from the
FarmShare login node (CPU, no GPU jobs) or WebFetch; all "analysis" is reading JSON/Python
source. No training, no benchmarks.

**Claim on trial (HANDOFF.md:103):** P2 (cross-layer KV sharing in LFM2 — 3 KV banks serving the
6 GQA layers at `[2,5,8,10,12,14]`) is *"Anticipated 3×"* by **Hymba (arXiv 2411.13676)**,
**Character.AI**, and **Gemma 3n**.

**Evidence-grade legend:** MEASURED = read verbatim in a primary source I fetched this pass;
INFERRED = derived by logic/arithmetic from a MEASURED fact; ASSUMED = neither.

---

## BOTTOM LINE (full reasoning in §4-§5)

| | verdict |
|---|---|
| "P2's **mechanism** is anticipated 3×" | **CONFIRMED.** All three sources exist, all three genuinely do cross-layer KV sharing, and all three are correctly described in the docs. Hymba and Character.AI both cite Brandon et al. (CLA) by name. |
| "P2's **measurement** is anticipated 3×" | **REFUTED.** Zero of the three publishes a controlled retrieval ablation of the sharing. Hymba's row C→D is the only controlled ablation of any kind and its "Recall" column is an unnamed 2-task average; Character.AI publishes no numbers at all; Gemma 3n publishes no eval of sharing whatsoever. |
| Net on the HANDOFF phrase "Anticipated 3×" | **Directionally right, but it conflates the two.** It is fair as a *novelty* warning and misleading as an *evidence* claim. See §5 for the residual gap and whether it is worth GPU-days (my answer: the gap is real but thin, and it is **not** worth GPU-days — for a reason independent of these three sources). |

**No hallucinations found. The prior team's citation audit holds — I re-verified every load-bearing
number independently and found one understatement and one thing they could not read that I could.**

---

## 1. Hymba — arXiv 2411.13676 — **EXISTS, does what is claimed, retrieval NOT ablated**

**URLs:** `https://arxiv.org/abs/2411.13676` · `https://arxiv.org/html/2411.13676v1` ·
`https://huggingface.co/nvidia/Hymba-1.5B-Base/raw/main/config.json` ·
`https://huggingface.co/nvidia/Hymba-1.5B-Base/raw/main/modeling_hymba.py`
(note: `arxiv.org/html/2411.13676v2` returns **HTTP 404** — v1 is the HTML that exists)

### (a) Does it exist? — **YES. MEASURED.**
*Hymba: A Hybrid-head Architecture for Small Language Models.* Dong, Fu, Diao, Byeon, Chen,
Mahabaleshwarkar, Liu, Van Keirsbilck, Chen, Suhara, Lin, Kautz, Molchanov (NVIDIA).
Submitted **20 Nov 2024**, cs.CL, 20 pp, CC BY 4.0. Abstract verbatim contains:
> "incorporating **cross-layer key-value (KV) sharing** and partial sliding window attention,
> resulting in a compact cache size"

Headline: vs Llama-3.2-3B — *"1.32% higher average accuracy, an 11.67x cache size reduction, and
3.49x throughput."* (Note: that number bundles SSM heads + SWA + sharing against a *different,
larger* model. It is **not** a sharing ablation.)

### (b) Cross-layer KV sharing, of what form? — **YES, CLA-style, strictly adjacent. MEASURED.**
From `nvidia/Hymba-1.5B-Base/config.json`, fetched this pass, HTTP 200:

```
num_hidden_layers = 32,  hidden_size = 1600
num_attention_heads = 25,  num_key_value_heads = 5,  sliding_window = 1024
global_attn_idx  = [0, 15, 31]
kv_weight_reuse  = false
kv_reuse_group   = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],
                    [16,17,18],[19,20],[21,22],[23,24],[25,26],[27,28],[29,30]]
```
This **reproduces the docs' block exactly** (03_kv_sharing.md:895-903). Confirmed:
- pairing is **strictly adjacent** (CLA's own recommendation);
- the **3 global layers `[0,15,31]` are excluded from every group** — MEASURED from the config by
  set difference, *not* stated in the paper text. The paper's prose only says KV is shared
  *"between consecutive layers (e.g., every two layers share the same KV cache)"* and gives **no
  explicit statement that global layers are excluded.** So the docs' claim "Hymba explicitly
  excludes its 3 global layers" is **true of the shipped config but is NOT an explicit paper
  claim** — a fine distinction that matters, because it means Hymba never *argued* for excluding
  them, let alone ablated it. The docs already say this ("Hymba's config proves the choice but not
  the reason") and they are right.
- one **3-way** group `[16,17,18]` (sharing factor 3) absorbing the odd count around global layer 15.
- Citations: **Brandon et al. (CLA) [ref 11]** for the technique; **MiniCache [ref 23]** for
  "KV cache shares a high similarity between adjacent layers."

### (c) The "row C→D" ablation — **VERIFIED VERBATIM. MEASURED from arXiv HTML v1, Table 1.**

| Row | Commonsense % | **Recall %** | tok/s | Cache MB |
|---|---:|---:|---:|---:|
| Transformer (Llama) | 44.08 | 39.98 | 721.1 | 414.7 |
| SSM (Mamba) | 42.98 | 19.23 | 4720.8 | 1.9 |
| A. + Attention heads (sequential) | 44.07 | 45.16 | 776.3 | 156.3 |
| B. + Multi-head structure (parallel) | 45.19 | 49.90 | 876.7 | 148.2 |
| **C. + Local / global attention** | 44.56 | **48.79** | 2399.7 | 41.2 |
| **D. + KV cache sharing** | **45.16** | **48.04** | **2756.5** | **39.4** |
| E. + Meta tokens | 45.59 | 51.79 | 2695.8 | 40.0 |

Setup: **300M params / 100B tokens**; A100, 8k seq, batch 128, cache in FP16.
C→D: commonsense **+0.60**, recall **−0.75**, throughput **+14.9%**, cache **−4.4%** (41.2→39.4).
**The docs' numbers are exact to the decimal.** The prior reassessment's "row C→D verified" is
**CONFIRMED**.

Two things the docs get right and that a reviewer will notice:
- The cache saving is only **4.4%**, because row C already cut cache 3.8× with SWA. SWA and CLA
  are largely **substitutes** on the capacity axis. (INFERRED, trivially, from the table.)
- The paper's own gloss of row D is *"improves throughput by 1.15× while maintaining comparable
  recall accuracy and boosting commonsense accuracy by +0.60%"* — i.e. Hymba **frames the −0.75
  recall as noise** and does not investigate it.

### (d) ⚠️ THE DECISIVE QUESTION: is Hymba's recall number attributable to the SHARING ablation? — **NO. This is the finding that most matters for the capstone.**

- The **only** KV-sharing-specific number in the entire paper is Table 1 row D. Its "Recall" column
  is, per the caption, an **average over 2 tasks** — and **the caption never names them.**
  MEASURED. (Table 3 uses SWDE + SQuAD-C as the recall-intensive pair, so that is the likely
  correspondence, but it is **INFERRED, not stated**.) A 2-task, unnamed, 300M/100B average.
- Hymba **does** run **needle-in-a-haystack** — **Figure 10**, Hymba vs Mamba2 vs Llama3, all 1B,
  pretrained at 1k, finetuned at 4k, tested to 16k. **But it is an architecture comparison, not a
  KV-sharing ablation.** The KV-sharing row is not in it.
- **No passkey. No RULER. No MQAR. No sharing-specific retrieval ablation anywhere.** MEASURED
  (searched the HTML for all of these).

**So the project-memory note "only Hymba reports recall" is TRUE only in the weakest sense:**
Hymba reports a 2-task recall *average* on the sharing row, and separately reports needle on a
*non-sharing* comparison. **It does NOT report a retrieval benchmark isolating sharing's effect.**
The prior team's own §6 line ("the 'recall' column is a 2-task average the paper never names in
the table caption — so it is even weaker evidence than the docs concede") is **CONFIRMED
independently**, and it is the correct reading.

### (e) Topology: is Hymba's shape the same as LFM2's? — **NO. The distinction is CONFIRMED.**
- Hymba is a **parallel hybrid-head** model: the paper's own title and abstract describe a
  *"hybrid-head parallel architecture"* where **attention heads and SSM heads sit inside the same
  layer** processing the same input. MEASURED (abstract).
- Therefore Hymba's shared pair `[1,2]` = two **adjacent** hybrid layers. The producer's K/V is
  consumed by the very next block.
- LFM2 is **sequentially interleaved**: between attention layer 8 and attention layer 10 sits a
  *complete gated short-conv block* that rewrites the residual stream. **A Hymba producer/consumer
  pair has no intervening sequence mixer; an LFM2 pair does.** INFERRED, but the inference is
  immediate from the two architectures' definitions.
- Second structural difference, equally load-bearing: **Hymba shares between SWA-1024 layers and
  excludes its full/global layers. LFM2's 6 attention layers are all full attention.** P2 would
  share between exactly the layer type Hymba declined to share. MEASURED from the config.

**Verdict: the capstone's residual-novelty distinction is real and survives.** But see §5 —
"structurally different" is not the same as "expected to behave differently."

### (f) RoPE decision — **CONFIRMED FROM THE RELEASED CODE. MEASURED.**
`modeling_hymba.py` (2651 lines), fetched from HF. The pattern appears three times (three attention
implementations: lines ~644-663, ~1111-1129, ~1288-1307), identical each time:

```python
if self.config.rope:
    query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)   # consumer rotates its OWN Q

if self.reuse_kv:
    assert kv_last_layer is not None
    key_states, value_states = kv_last_layer          # consumer takes the ALREADY-ROTATED K
else:
    key_states  = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    ...
    if self.config.rope:
        _, key_states = apply_rotary_pos_emb(None, key_states, cos, sin)   # producer rotates K
```
Producer applies RoPE to K *before* the tensor is handed on (line 663 → `key_states_no_repeat` at
729 → returned at 775). Consumer applies RoPE only to its own Q (line 644). **HANDOFF's claim
"settled by Hymba's released implementation (producer rotates K, consumer rotates only its own Q)"
is exactly right. POST-rotary K sharing CONFIRMED.**

Two corroborating implementation facts also confirmed:
- **Consumer allocates no K/V projections at all** — line 522: `if not self.attn_only_wo_proj and
  not self.reuse_kv:` guards `k_proj`/`v_proj` construction. Q is unguarded.
- **Consumer does not write to cache** — lines 667, 690: every cache path guarded
  `... and not self.reuse_kv`. Confirms CLA's capacity-not-bandwidth property in a real hybrid.

---

## 2. Character.AI — **EXISTS, does what is claimed, ZERO evaluation published**

**URL (primary, archived — the live `research.character.ai` URL 301-redirects and the post is
gone):** `https://web.archive.org/web/20240726230425/https://research.character.ai/optimizing-inference/`
*"Optimizing AI Inference at Character.AI", Jun 20, 2024, 4 min read.*
(WebFetch is blocked from web.archive.org; I fetched it with `curl` on the FarmShare login node and
stripped tags. All quotes below are **verbatim from that fetch. MEASURED.**)

### The three KV techniques, verbatim
> "The key bottleneck of LLM inference throughput is the size of the cache of attention keys and
> values (KV). ... We use the following techniques to reduce KV cache size by **more than 20X
> without regressing quality**."

1. **Multi-Query Attention** — *"We adopt Multi-Query Attention (Shazeer, 2019) in all attention
   layers. This reduces KV cache size by 8X compared to the Grouped-Query Attention adopted in most
   open source models."*
2. **Hybrid Attention Horizons** — *"We interleave local attention (Beltagy et al., 2020) with
   global attention layers. ... We found that reducing attention horizon to 1024 on most attention
   layers does not have a significant impact on evaluation metrics, **including the long context
   needle-in-haystack benchmark**. In our production model, **only 1 out of every 6 layers uses
   global attention**."*
3. **Cross Layer KV-sharing** — the sentence the whole claim rests on:
   > "**We tie the KV cache across neighboring attention layers, which further reduces KV cache
   > size by a factor of 2-3x. For global attention layers, we tie the KV cache of multiple global
   > layers across blocks, since the global attention layers dominate the KV cache size under long
   > context use cases. Similar to a recent publication (Brandon et al., 2024), we find that
   > sharing KV across layers does not regress quality.**"

   Figure 1 caption: *"...curves indicate KV-sharing. **For global attention layers, we share KV
   across multiple non-adjacent layers.** This illustration depicts only a subset of the layers in
   the full model."*

### Verdict on Character.AI
- **Do they share across layers? YES.** MEASURED. Two distinct patterns: (i) **adjacent** local
  layers ("neighboring attention layers", 2-3× factor); (ii) **non-adjacent global** layers tied
  "across blocks."
- **What fraction?** 1-in-6 layers is global; sharing factor stated only as **"2-3x"**. Exact
  pattern, layer count, and model size are **not published**. ASSUMED beyond that.
- **Do they cite CLA? YES, by name** ("Brandon et al., 2024").
- **⚠️ Do they publish any quality or retrieval evaluation of the sharing? NO. ZERO NUMBERS.**
  MEASURED. The only evaluative words are *"does not regress quality"* and *"more than 20X without
  regressing quality"* — bare assertions with no table, no benchmark name, no delta. Note the
  needle-in-haystack mention belongs to item **2 (sliding window)**, **not** item 3 (KV sharing).
  A reviewer who reads carefully will see that Character.AI's one retrieval-flavoured claim is
  about *SWA*, and their sharing claim has no evaluation attached at all.
- **Topology:** a pure **transformer** (MQA + local/global). **No conv, no SSM, no hybrid.**
  MEASURED (nothing in the post mentions any non-attention mixer).

**Grade as prior art: strong as an existence proof (>20k QPS production, ~20% of Google Search
volume), weak as evidence — a blog post with no evaluation.** It also *contradicts* CLA's own
`DenseBack` ablation (+0.43 ppl for non-adjacent pairing) by asserting non-adjacent global sharing
is free. Nobody has reconciled that. The docs already flag this and are right to.

**Follow-up post** ("Part Deux", Nov 21, 2024,
`https://web.archive.org/web/20241126230517/https://research.character.ai/optimizing-ai-inference-at-character-ai-part-deux/`)
— not re-fetched this pass; the docs cite it for int8/serving, not for new sharing evidence.

---

## 3. Gemma 3n — **`num_kv_shared_layers=15` VERIFIED. The reassessment's "UNVERIFIED / HTTP 401" is now CLOSED.**

The prior reassessment (§8) could not read this because `google/gemma-3n-*` is gated. **I
reproduced the 401 and then got the value two other ways.**

### (a) Google's own HF configs — still gated. MEASURED.
```
google/gemma-3n-E2B      -> HTTP 401
google/gemma-3n-E4B      -> HTTP 401
google/gemma-3n-E2B-it   -> HTTP 401
google/gemma-3n-E4B-it   -> HTTP 401
google/gemma-3n-E4B-it-litert-preview -> HTTP 401
```

### (b) The transformers library source — **AUTHORITATIVE AND UNGATED. MEASURED.**
`https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma3n/configuration_gemma3n.py`
(478 lines). Line 127: `num_kv_shared_layers: int = 15`. Docstring, lines 52-56, **verbatim**:
> "num_kv_shared_layers (`int`, *optional*, defaults to 15): The number of layers that share KV
> cache values. During the forward pass, the last `num_kv_shared_layers` layers in the model
> "share" the KV values in that each local and global layer in this range uses the KV cache values
> computed for **the last local or global layer, respectively, before entering this range**. The
> value should be a multiple of the attention pattern size (see `layer_types` parameter)."

Also: `num_hidden_layers: int = 35`, `sliding_window: int = 512`, and `layer_types` defaults to
every 5th layer `"full_attention"`, rest `"sliding_attention"`.

### (c) The actual shipped configs, via an **ungated mirror**. MEASURED.
`https://huggingface.co/unsloth/gemma-3n-E4B-it/raw/main/config.json` (HTTP 200) and
`https://huggingface.co/unsloth/gemma-3n-E2B-it/raw/main/config.json` (HTTP 200):

| | E4B-it | E2B-it |
|---|---:|---:|
| `num_hidden_layers` | **35** | **30** |
| **`num_kv_shared_layers`** | **15** | **10** |
| `sliding_window` | 512 | 512 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 8 / 2 / 256 | 8 / 2 / 256 |
| `full_attention` layer indices | [4,9,14,19,24,29,34] | [4,9,14,19,24,29] |

**⚠️ Correction to the docs: `num_kv_shared_layers=15` is the E4B value AND the library default.
E2B ships `10`.** The docs cite a single number as if it were "Gemma 3n's"; it is model-specific.
(Caveat, stated honestly: the mirror is a third-party re-upload, not Google's own repo. But it
agrees exactly with the library default and the docstring semantics, so I grade the E4B value
**MEASURED** and the E2B value **MEASURED (mirror-sourced, single source)**.)

### (d) What the mechanism actually is. MEASURED from `modeling_gemma3n.py` lines 1186-1194, 1241-1259.
```python
first_kv_shared_layer_idx = self.config.num_hidden_layers - self.config.num_kv_shared_layers
self.is_kv_shared_layer  = layer_idx >= first_kv_shared_layer_idx > 0
prev_layers = config.layer_types[:first_kv_shared_layer_idx]
if self.is_kv_shared_layer:
    # last non-shared layer of the SAME TYPE before sharing starts
    self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
...
if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
```
Applying that code to the shipped configs (arithmetic, INFERRED but mechanical):

| model | sharing starts at | producer | consumers | fan-out |
|---|---:|---|---|---|
| **E4B** (35 L, 15 shared) | layer 20 | **18** (sliding) | 20,21,22,23,25,26,27,28,30,31,32,33 | **13-way** |
| | | **19** (full) | 24, 29, 34 | **4-way** |
| **E2B** (30 L, 10 shared) | layer 20 | **18** (sliding) | 20,21,22,23,25,26,27,28 | **9-way** |
| | | **19** (full) | 24, 29 | **3-way** |

So Gemma 3n is **NOT CLA-style pairwise at all.** It is a **one-producer-many-consumers** scheme:
**exactly two KV banks serve the entire top 43% of the model**, one for sliding layers and one for
full layers, both produced at the boundary (layers 18/19). Consumers allocate **no k_proj/v_proj
and no k_norm/v_norm** (line 1206: `if not self.is_kv_shared_layer:` guards all four) and **RoPE
is applied to K by the producer** (line 1249, `apply_rotary_pos_emb(key_states, ...)` inside the
`else:` non-shared branch) — a **second independent vote for post-rotary sharing**, matching Hymba.

Note also the comment at line 1240, which is a real implementation trap worth carrying:
> "We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember
> the full states in their Cache once we are past the sliding window - so we always use
> `shared_kv_states` instead"

### (e) Google's own description — **found, and it CONFIRMS my code-derived topology. MEASURED.**
`https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/`, verbatim:
> "The keys and values of **the middle layer** from local and global attention are directly shared
> with **all the top layers**."

That is precisely the layer-18/19 → layers-20-34 structure I derived from the code in (d), stated
independently by Google. **One producer per attention type, feeding the whole top of the stack.**
The claimed benefit, verbatim:
> "a notable **2x improvement on prefill performance** compared to Gemma 3 4B"

⚠️ **A nuance worth carrying into P2's cost model:** Google claims a **prefill** win, which the
docs assert CLA cannot claim (03_kv_sharing.md:184-185, "CLA saves nothing at prefill"). Both are
right for different reasons — the consumer skips `k_proj`/`v_proj` *compute*, which is real at
prefill, but the consumer still runs full attention over the shared bank, so the attention-score
FLOPs are untouched. With Gemma 3n's **13-way** fan-out the skipped projections are a large share
of the top of the stack; with P2's **2-way** pairing over 3 pairs they are 3 of 6 layers'
projections in a model where attention is a minority of blocks. **The 2× does not transfer to P2 —
it is a fan-out effect.** INFERRED.

### (f) Does Google publish any retrieval eval of the sharing? — **NO.**
The developer guide gives the 2× prefill number and **no quality, retrieval, or long-context
benchmark of the feature.** MEASURED. Separately,
`https://ai.google.dev/gemma/docs/gemma-3n` — **the phrase "KV cache sharing" does not appear on
that page at all.** It documents MatFormer and Per-Layer Embeddings (PLE) as the named
innovations, gives **no layer counts / head dims**, lists "32K token context" as a feature, and
reports **no evaluation scores of any kind**. There is no Gemma 3n technical report on arXiv that
I could find (2507.06261 is the *Gemini 2.5* report, not Gemma 3n — checked and excluded).

**Grade as prior art: the weakest of the three on evidence, the strongest on deployment.** Gemma 3n
ships a far more aggressive sharing scheme (13-way!) into production and upstream into
transformers, with **zero published ablation and zero published retrieval number**.

---

## 4. Per-source verdict table

| source | EXISTS? | says what HANDOFF claims? | shares across layers? | topology | **retrieval ablation of the sharing?** | evidence grade |
|---|---|---|---|---|---|---|
| **Hymba** 2411.13676 | ✅ MEASURED | ✅ yes, and cites CLA | ✅ adjacent pairs, 14 groups, globals excluded, factor 2 (one group 3) | **parallel** hybrid-head (attn ∥ SSM in one layer); shares between **SWA** layers | **NO.** Table 1 row D is a 2-task unnamed recall average at 300M/100B; NIAH (Fig 10) is a non-sharing architecture comparison | **Paper w/ 1 controlled ablation row, no retrieval isolation** |
| **Character.AI** blog | ✅ MEASURED (archived) | ✅ yes, and cites CLA | ✅ adjacent local **+ non-adjacent global**, factor "2-3x" | **pure transformer**, MQA + local/global 1:6 | **NO. No numbers at all.** ("does not regress quality", unquantified). Their NIAH mention is about SWA, not sharing | **Blog post, existence proof only** |
| **Gemma 3n** | ✅ MEASURED (lib source + ungated mirror + Google dev blog) | ✅ `=15` for E4B (**but E2B=10**) | ✅ **13-way + 4-way**, one producer per layer-type ("middle layer ... shared with all the top layers") | **pure transformer**, sliding/full 4:1 | **NO.** Only a **2× prefill** claim; no quality/retrieval number, and the feature is not even named on ai.google.dev | **Shipped config + 1 perf number, zero quality eval** |

**Every load-bearing citation in the docs resolves and says what is claimed. No hallucinations.
Two corrections found, both small and both in the docs' disfavour on precision (Gemma E2B=10, and
Hymba's global-exclusion is a config fact not a paper claim).**

---

## 5. Synthesis — is "anticipated 3×" right?

### Anticipated as a MECHANISM: **YES, decisively. Three independent times.**
Cross-layer KV sharing is not a research question in 2026. It is in a shipped NVIDIA hybrid
(Nov 2024), in a >20k-QPS production transformer citing CLA (Jun 2024), and in a Google model
upstreamed into `transformers` with a 13-way fan-out. **Any framing of P2 as "we propose to try
cross-layer KV sharing" is an overclaim a reviewer kills in one sentence.** HANDOFF's "Anticipated
3×" is doing correct and necessary work here.

### Anticipated as a MEASUREMENT: **NO. Zero of three.**
This is where the HANDOFF label misleads by compression. Precisely:
- **Nobody among the three ran a retrieval benchmark as a sharing ablation.** Hymba comes closest
  and its number is a 2-task *unnamed* average that moved **−0.75**, which the paper waves off as
  "comparable." Character.AI publishes no number. Google publishes no number and does not name the
  feature.
- **Nobody among the three shares across an intervening non-attention sequence mixer.** Hymba's
  pairs are adjacent in a *parallel* architecture (no block between producer and consumer);
  Character.AI and Gemma 3n are pure transformers, so their "intervening layers" are attention or
  MLP, never a conv/SSM. **CONFIRMED — the capstone's central structural distinction survives all
  three sources.**
- **Nobody among the three shares between full-attention layers *in a hybrid*.** Hymba shares
  between SWA and excludes globals; Character.AI does share globals but is not a hybrid.
- **Nobody is at 350M in an LFM2-shaped stack** with GQA-8 / head_dim 64.

### The residual gap that would survive a knowledgeable reviewer
> *CLA-style **pairwise** sharing between **full-attention** layers **separated by a complete gated
> short-conv block**, in a **sequentially interleaved** hybrid at 350M, evaluated with **retrieval
> endpoints**.*

That is genuinely unoccupied by these three. It is a gap of **configuration**, not of question.

### Is it worth GPU-days? **No — but the three sources are not why.**
Being honest in both directions, as instructed:
1. **These three do NOT kill P2.** If Hymba/Character.AI/Gemma 3n were the whole prior art, the
   scoped claim above would be defensible: "everyone deploys this, nobody has measured what it
   costs retrieval, and nobody has done it across a conv block." That is a legitimate, modest,
   negative-result-safe capstone contribution. **On this evidence alone, "anticipated 3×" is
   OVERSTATED as a reason to cut.**
2. **What actually kills it is a fourth source the docs never cite** — arXiv **2606.06467** (MSRA,
   4 Jun 2026), which the prior reassessment found and which ran **RULER × 12 subtasks × {16K,32K}
   on from-scratch 4B KV-sharing models against a matched non-sharing control**, i.e. the
   measurement gap, closed, at 12× the scale, in the *opposite* direction from the capstone's
   motivating worry (**+6.1 RULER avg at 32K**). I did not re-verify 2606.06467 this pass (out of
   my scope); if the parent wants the cut defended, **that** is the citation to re-verify, not
   these three.
3. **And the arithmetic kills it independently:** P2 saves capacity, not bandwidth (Hymba's own
   code confirms — consumers skip the cache *write*, not the *read*), so decode latency is ≈0% by
   construction. Hymba's own row D moved cache only **4.4%**.

**Recommended precise wording to replace "Anticipated 3×":**
> *"The mechanism is anticipated three times over (Hymba, Character.AI, Gemma 3n) — all three ship
> it, two cite CLA by name. What none of them provides is a controlled retrieval evaluation of the
> sharing itself: Hymba's only sharing-specific recall figure is an unnamed 2-task average that
> moved −0.75 and was dismissed as noise; Character.AI asserts 'does not regress quality' with no
> numbers; Google does not name the feature in its docs. So P2 is anticipated as a mechanism and
> unmeasured as a phenomenon. The residual gap — pairwise sharing between full-attention layers
> across an intervening conv block, with retrieval endpoints — is real but narrow, and it is
> closed at larger scale by arXiv 2606.06467."*

### If P2 is kept in any form
The **zero-training** version is the one I would defend, and both Hymba's and Gemma 3n's code make
it cheap and unambiguous to build: apply CLA-style pairing **post-hoc** to the released LFM2-350M
weights (copy the producer's **post-rotary** K and its V, drop the consumer's `k_proj`/`v_proj`)
and measure the recall cliff on the existing passkey/BABILong harness. **Both reference
implementations independently confirm post-rotary K sharing and no consumer KV projections**, so
there is no design ambiguity left to resolve and no pretraining run required.

---

## 6. Loose ends / what would settle what

- **`google/gemma-3n-*` remains HTTP 401.** Settled instead by (i) the transformers library default
  + docstring and (ii) an ungated third-party mirror that agrees with it. **What would fully
  settle it:** an authenticated HF token with the Gemma license accepted, reading
  `google/gemma-3n-E4B-it/config.json` directly. I consider the value established well enough to
  cite; the honest citation is *to the transformers source*, not to Google's repo.
- **`arxiv.org/html/2411.13676v2` is HTTP 404.** v1 HTML exists and is what I read. Any quote
  should cite v1.
- **Hymba's "Recall" 2-task identity is INFERRED** (SWDE + SQuAD-C from Table 3), not stated in the
  Table 1 caption. Settling it requires reading the Appendix table-by-table; it does not change any
  conclusion, because the number is weak either way.
- **Character.AI "Part Deux"** not re-fetched; cited by the docs for int8/serving, not sharing.
- **arXiv 2606.06467** — the actual P2-killer — was **not** re-verified by me (out of scope). It is
  the single highest-value re-check if the cut needs defending.
- **CLA 2405.12981's `DenseBack` +0.43 ppl vs Character.AI's "non-adjacent is free" vs Gemma 3n's
  13-way fan-out** is a live, unreconciled three-way contradiction in the literature. Nobody has
  run the controlled comparison. If the human wants a *cheap* P2-shaped contribution, this — not
  "CLA in a hybrid" — is the question with no owner.

**Status: COMPLETE.**
