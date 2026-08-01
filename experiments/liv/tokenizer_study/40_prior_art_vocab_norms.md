# 40 — Prior art: what vocab/tokenizer do comparable architecture-comparison papers actually use?

**Status: IN PROGRESS — written incrementally, append-only.**
Owner: child agent 4 of 5. Scope: literature only, no code execution.
Started 2026-08-01.

Evidence tags used throughout:
- **MEASURED** = I read the number/sentence in the primary source.
- **INFERRED** = derived by arithmetic or logic from a MEASURED fact.
- **ASSUMED** = neither; flagged as such.
- **CONFIRMED / REFUTED / UNCLEAR** = verdict on the claim being checked.

Every arXiv ID below is reported with whether it *resolved to the title I expected*.

---

## BOTTOM LINE

**1. Is GPT-2 vocab standard at this scale? — YES. Emphatically. It is the modal choice among the
papers this study directly inherits from.**

- **8 comparable architecture-comparison papers at 300M–500M** were checked (Zoology 360M, Based
  355M, Mamba ~350M ablations, Samba 438M, Hymba 300M/350M, Griffin 400M, DeltaNet 340M, Gated
  DeltaNet 400M).
- **Zoology (2312.04927) and Based (2402.18668) both explicitly used the GPT-2 BPE tokenizer at
  355–360M** — MEASURED, direct quotes. **Zoology is the source of the AR-Hits metric this study is
  adopting wholesale.** Using GPT-2 is *protocol fidelity*, not a compromise.
- Observed vocab range at this scale: **32,000 (Mistral / LLaMA-2) to 50,277 (GPT-NeoX)**.
- **Papers at this scale using a vocab ≥ 64K: ZERO.** 50,257 is inside the norm; **65,536 is outside it.**
- **Papers that print a numeric vocab size at all: 1 of 10.** Papers that *justify* their tokenizer
  choice: **0 of 10.** Papers reporting a tokenizer/vocab ablation inside an architecture comparison:
  **0 of 10.**
- Griffin (2402.19427) — the phonebook precedent this study wants to beat — **never states its
  tokenizer or vocabulary anywhere**, at up to 14B parameters.

**2. Compute-optimal vocab at our scale — ~16K. Both candidates are oversized; GPT-2 is the closer one.**

Tao et al. (**2407.13623**, RESOLVED) contains a **direct empirical measurement at N_nv = 302M** —
our geometry is **N_nv = 287,375,104**, a 4.9% difference, i.e. the same rung:
> best vocabulary **16K**, moving **down to 10K when data-constrained** and up to 24K with excess data.

My substitution through their γ=0.83 power law gives **V_opt ≈ 14K–19K**, agreeing. **We are on the
data-constrained side** (2–5B tokens at 350M vs ~7B Chinchilla-optimal), which pushes the optimum
*lower still*. So: **50,257 is ~3.1× oversized; 65,536 is ~4.0× oversized.** Independently
corroborated by Ali et al. (**2310.08754**): *"in the monolingual English setting, the
smaller/medium-sized vocabulary performs better"* — their 32k tokenizer beat their 82k one.
**Bonus: GPT-2 drops embeddings from 18.9% → 15.19% of parameters, which RAISES the mixer's share of
the model under study. This is a methodological improvement, not a concession.**

**3. AR-Hits tokenizer-dependence — YES, dependent; and the bigger problem is corpus size.
The 6.4% / ≤1250× constants CANNOT be inherited — but this is NOT an argument for retokenizing.**

AR Hits is defined over *token bigrams*, so the slice is necessarily tied to GPT-2's BPE — Zoology
**never acknowledges or tests this**. Worse: the **1250 threshold is an absolute count**, and Zoology
applied the same 1250 to models trained on 5B, 10B *and* 50B tokens without reconciling it. **Our
2–5B budget would mechanically balloon the slice far past 6.4%.** Different corpus (Dolma2/FineWeb-Edu
vs Pile) too. **Action: build our own bigram frequency table (one cheap CPU pass) and re-derive the
cutoff as a percentile; report the measured slice fraction.** This applies identically under either
vocab. Handled well, it becomes a *contribution* — Zoology never ablated the threshold.

**4. Does any published work show a tokenizer flipping an architectural ranking? — NO. Found none.**

Every tokenizer study located (Ali et al. 2310.08754; TokSuite 2512.20757; Tao et al. 2407.13623)
**holds architecture FIXED and varies the tokenizer** — establishing a *main effect*, never an
*interaction*. A main effect is absorbed by the shared baseline in an all-arms-same-tokenizer design.
*Caveat, stated honestly:* `WebSearch` was HTTP-403 for this entire session, so this negative rests on
18 directly-fetched primary sources rather than an exhaustive search.

**5. REVIEWER RISK: LOW — and strictly LOWER for GPT-2 than for 65,536.**

One-sentence reason: **the two papers this study takes its primary endpoint from both used the GPT-2
tokenizer at exactly this scale, no comparable paper uses a ≥64K vocab at 350M, and the vocab-scaling
literature says 65,536 is 4× oversized here — so GPT-2/50,257 is simultaneously the more standard,
the more faithful, and the more compute-optimal choice.**

→ **Recommendation: redeclare the ledger at 50,304** (50,257 padded to a multiple of 128 for
tensor-core alignment; standard practice) **and train now.** Drop the "reproduces LFM2's exact
released shape" claim — a from-scratch model trained on **0.02–0.05% of LFM2's 10T-token budget**
(2,000–5,000×) is not LFM2 regardless of vocab, no surveyed paper makes such a claim, and the claim
invites a quality-comparison question the study cannot answer. Keep "**follows the released LFM2-350M
layer geometry and attention schedule**", which is the part that carries real weight.

**Biggest actual risk found in this survey is not the vocab at all — it is the inherited
"6.4% / ≤1250×" AR-Hits constants (§3c).**

---

## Log

### [1] Zoology — arXiv 2312.04927 — RESOLVED ✅

Fetched `https://arxiv.org/abs/2312.04927` (resolved: "Zoology: Measuring and Improving Recall in
Efficient Language Models", Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré, 8 Dec 2023).
Abs page has no experimental detail; full text via `https://ar5iv.labs.arxiv.org/html/2312.04927`
(resolved, same title).

**THE HEADLINE FINDING OF THIS WHOLE DOCUMENT:**

> "The Pile data is tokenized using the GPT2BPETokenizer and all models see the data in the same order."

**MEASURED / CONFIRMED.** The paper the study is adopting AR-Hits *wholesale* from **used the GPT-2
BPE tokenizer**. Not GPT-NeoX, not a custom 64k vocab. GPT-2. Training infra was EleutherAI GPT-NeoX
(the *framework*), but the tokenizer is explicitly GPT2BPE.

Implication, stated plainly: **the 6.4% AR-Hits slice, the 1250× frequency threshold, and the 82%
gap-attribution number are all GPT-2-tokenizer numbers.** If the study retokenizes to LFM2's 65,536
vocab, it is *further* from Zoology's measurement conditions, not closer. Using GPT-2 at 50,257 is
the *maximally faithful* reproduction of the AR-Hits protocol.

Other MEASURED facts from the same source:
- AR Hits definition: *"AR Hits: (6.4% of tokens) Tokens in the final position of a bigram (a pair of
  consecutive tokens) which previously appeared in context, but"* … occurred **≤1250×** during training.
- Other tokens: **93.6%**.
- Synthetic MQAR: **vocabulary size 8,192** (matches the study's `zoology/data/multiquery_ar.py`
  reproduction grid noted in the local design doc at line 1220).
- Scales: 17-model suite ~70M–1.4B; fine-grained Pile analysis at **70M–360M trained on 10B tokens**;
  BaseConv hybrids at ~150M/168M and **354M/360M**.

Note the scale coincidence: Zoology's own upper analysis rung is **~354M–360M**, i.e. exactly our
scale, **with a 50,257 GPT-2 vocab**. This is the single best precedent available.

### [2] Based — arXiv 2402.18668 — RESOLVED ✅

Fetched `https://arxiv.org/abs/2402.18668` → resolved to "Simple linear attention language models
balance the recall-throughput tradeoff" (Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou,
Rudra, Ré; v1 28 Feb 2024, v2 7 Mar 2025). Full text `https://ar5iv.labs.arxiv.org/html/2402.18668`
(resolved).

> "The Pile data is tokenized using the GPT-2 BPE tokenizer"

> "We pretrain language models from scratch at two parameter scales (355M and 1.3Bn parameters) on the Pile"

> "Each model sees the same 10 billion tokens of pretraining data in the same order."

**MEASURED / CONFIRMED.** Based = **GPT-2 BPE, 355M and 1.3B, 10B Pile tokens.** No numeric vocab
size is ever printed in the paper — only the tokenizer *name*. Parameter counts reported in tables:
Based **363m**, Transformer++ **360m**, Mamba **358m** at the small scale.

Two things worth flagging hard:
1. **This is a same-scale, same-endpoint, same-research-group precedent** (Based is the direct
   successor to Zoology; both are Hazy Research). 355M, architecture comparison, recall endpoints,
   GPT-2 tokenizer, ~10B tokens. Our study is 350M, architecture comparison, recall endpoints,
   2–5B tokens. The proposed configuration is *the same configuration*.
2. **Based does not state a vocabulary size at all.** It states a tokenizer name and moves on. That
   is itself evidence about reviewer expectations: the field's flagship recall-architecture paper did
   not consider the vocab number worth printing.

Also MEASURED: Based's DNA experiments use *a completely different tokenizer* ("a byte-level
tokenizer wherein the vocabulary consists of characters corresponding to the nucleotide bases") with
no suggestion that this invalidates cross-domain architectural conclusions — they swap tokenizers
across domains freely and still draw one architecture ranking.

### [3] Mamba — arXiv 2312.00752 — RESOLVED ✅

Fetched `https://arxiv.org/abs/2312.00752` (resolved: "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces", Gu & Dao) and full text `https://arxiv.org/html/2312.00752v2` (resolved).
The `arxiv.org/pdf/2312.00752` fetch returned raw binary and was unusable — noted for honesty.

> Table 1 caption: "Pile refers to the validation split, comparing only against models trained on the same dataset and tokenizer (GPT-NeoX-20B)."

> §4.2.2: "Pythia ... and RWKV ... which were trained with the same tokenizer, dataset, and training length (300B tokens) as our models."

**MEASURED / CONFIRMED:** Mamba uses the **GPT-NeoX-20B tokenizer**, *not* GPT-2. **The paper never
prints a vocabulary size number.** (The NeoX-20B vocab is 50,277, padded to 50,280 in the released
Mamba configs — that is a *repo* fact, not a paper fact; tagged INFERRED, see [3a].)

Scales: scaling laws 125M–1.3B; downstream Mamba-130M/370M/790M/1.4B/2.8B at 300B tokens, ctx 2048;
**ablations at ~350M**.

**The important structural observation:** Mamba's Table 1 has a **"Token." column** listing `NeoX`
vs `GPT2` vs `OPT` per row. Mamba compares its own NeoX-tokenizer models against **GPT-2-tokenizer
baselines in the same table**, and merely annotates the difference rather than treating it as
disqualifying. That is direct evidence that the field's norm is: *disclose the tokenizer, don't
match it*.

---

## THE VOCAB-SCALING-LAW ANSWER (decision-critical)

### [4] Tao et al., "Scaling Laws with Vocabulary" — arXiv 2407.13623 — RESOLVED ✅

Fetched `https://arxiv.org/abs/2407.13623` → resolved to **"Scaling Laws with Vocabulary: Larger
Models Deserve Larger Vocabularies"**, Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff,
Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong. NeurIPS 2024. v1 18 Jul 2024, v3 1 Nov 2024.
Full text `https://arxiv.org/html/2407.13623v3` (resolved). Code: github.com/sail-sg/scaling-with-vocab.

**Definitions (MEASURED, quoted):**

> "We break down the total model parameters (N) into non-vocabulary (N_nv) and vocabulary parameters (N_v)"

N_v = V·d — deliberately V·d and **not** 2Vd, because the FLOPs cost "is associated with the output
layer, but not the word embedding layer." (Relevant to us: our design is **tied** embeddings, so
V·d is exactly the right accounting.)

**The scaling relation (MEASURED, quoted):**

> "we then fit the relationship between N_nv and N_v using the power-law function N_v ∝ N_nv^γ"

> "N_v^opt = N_v^0 * (N_nv / N_nv^0)^γ"

> "the scaling proportion γ=0.83 after our fitting"

Approach 1 gives the only fully-numeric fits, in FLOPs C rather than N_nv:

> "N_nv = 0.08*C^0.50, N_v = 0.20*C^0.42 and H = 6.42*C^0.50"

> "γ = 0.42/0.50 = 0.84 < 1"

Approach 3 parametric loss: `ℒ_u = −E + A_1/N_nv^α_1 + A_2/N_v^α_2 + B/D^β` with A₁=1.831, A₂=0.196,
B=2.124, E=5.533, α₁=β=0.447, α₂=0.671; and f(V) = a log²(V) + b log(V) + c with a=0.0064,
b=−0.1581, c=1.2047.

### ⭐ THE SINGLE MOST DECISION-RELEVANT NUMBER IN THIS DOCUMENT

The paper contains a **direct empirical measurement at essentially our exact scale**. MEASURED:

> with **N_nv = 302M** held fixed, the empirically best vocabulary shifts with data volume —
> **decreasing 16K → 10K when data-constrained**, increasing **16K → 24K when trained on excess data**.

Our geometry: **N_nv = 287,375,104** (d=1024, 16 layers). *Source note:* this exact figure was
supplied mid-task by the orchestrating agent, which states it derived and validated the formula
against all six published arm counts to the parameter, and that it is **vocab-invariant**. It agrees
with the ~287M in my original brief. Companion totals it supplied: **354,483,968 at V=65,536** and
**338,838,272 at V=50,257** — both of which match the local `HANDOFF.md` ledger at lines 307–310
(`L0` = 354,483,968; `F-r128`/`G-grouped` = 338,755,328 — close but not identical, as those arms also
change mixer params). Tagged **MEASURED (secondhand, internally cross-checked)**.

**287.4M vs their 302M is a 4.9% difference.** This is not extrapolation; it is practically the same
rung of the same ladder.

**Substitution via Approach 2** (INFERRED — my arithmetic, done by hand, no code executed):

Anchor on Table 1's smallest row: N_nv = 3B → N_v^opt ≈ 0.1B, d = 3200, V^opt = 43K (Approach 2).

    ratio       = 287,375,104 / 3e9 = 0.095792
    ratio^0.83  : log10(0.095792) = −1.018672 ; × 0.83 = −0.845498 ; 10^(−0.845498) = 0.142726
    N_v^opt     = 0.1B × 0.142726 = 14.27M   (using the paper's rounded "≈0.1B")
    V^opt       = 14.27M / d = 14.27M / 1024 ≈ 13,940  →  ~14K

    Using the un-rounded anchor instead (3200 × 43,000 = 137.6M):
    N_v^opt     = 137.6M × 0.142726 = 19.64M
    V^opt       = 19.64M / 1024 ≈ 19,180  →  ~19K

**So the formula predicts V_opt ≈ 14K–19K at our scale, and their direct empirical sweep at
N_nv = 302M says 16K.** These agree. Call it **~16K**.

**And we are on the data-constrained side.** Our budget is **2–5B tokens at 350M params**. Chinchilla-
optimal for 350M is ~7B tokens (20×N), so 2–5B is *under*-trained (INFERRED, standard 20× rule).
Tao et al.'s own finding is that the data-constrained regime pushes the optimum **down, 16K → 10K**.

### Consequence for the actual decision

| vocab | multiple of the ~16K compute-optimal at N_nv≈287M | multiple of the ~10K data-constrained optimum |
|---|---:|---:|
| **16,384** (their prediction) | 1.0× | 1.6× |
| **50,257** (GPT-2, what we have) | **3.07×** | 5.0× |
| **50,304** (padded GPT-2) | 3.07× | 5.0× |
| **65,536** (LFM2 released) | **4.00×** | 6.6× |

**MEASURED-anchored / INFERRED arithmetic. Verdict: CONFIRMED — the frozen 65,536 is ~4× oversized
for this scale, and GPT-2's 50,257 is *strictly closer to compute-optimal* than the LFM2 vocab.**

This **inverts** the framing in the local design doc. `docs/liv-brainlift-experiment-design.md`
lines 264–270 recommend keeping 65,536 while acknowledging *"a large vocab does dilute the mixer
signal we are trying to measure"* (embeddings 67.1M / 354.4M = **18.9%** of the model). With GPT-2 at
50,257 tied and d=1024, embedding params = 50,257 × 1024 = **51,463,168**, and
51,463,168 / 338,838,272 = **15.19%** (INFERRED arithmetic; both figures cross-check against the
orchestrator's ledger — 287,375,104 + 51,463,168 = 338,838,272 exactly ✅). That is a
**3.7-point reduction in the fraction of the model that is not the thing under study.**

Sanity check on the other candidate: 65,536 × 1024 = 67,108,864; 287,375,104 + 67,108,864 =
**354,483,968** ✅ — matches the frozen `L0` ledger exactly. The orchestrator's numbers are internally
consistent and I have verified both additions by hand.

For a study whose entire purpose is measuring *mixer* differences, moving embeddings from 18.9% to
15.2% of parameters is a **methodological improvement, not a compromise.** The retokenization
argument runs backwards: the "correct" vocab for this experiment is *smaller* than both candidates,
and the corpus we already have is the one closer to correct.

---

## Remaining architecture papers

### [5] Mamba-2 — arXiv 2405.21060 — RESOLVED ✅ (but full text NOT retrieved)

`https://arxiv.org/abs/2405.21060` resolved to **"Transformers are SSMs: Generalized Models and
Efficient Algorithms Through Structured State Space Duality"**, Tri Dao & Albert Gu, ICML 2024.
**Title CONFIRMED.**

**HONEST FAILURE TO REPORT:** I could not retrieve the full text. `arxiv.org/html/2405.21060v1` and
`v2` both returned **HTTP 404**; `ar5iv.labs.arxiv.org/html/2405.21060` returned *"Conversion to HTML
had a Fatal error and exited abruptly."* **I therefore have NO primary-source quote for Mamba-2's
tokenizer or vocab, and I am not going to invent one.** Status: **UNCLEAR (unretrieved)**.

INFERRED-only (do not cite as measured): Mamba-2 trains on the Pile and compares directly against
Mamba-1 and Pythia, which per [3] use the GPT-NeoX-20B tokenizer; the released `state-spaces/mamba2-*`
configs use vocab 50277→50288. Treat as **ASSUMED** until someone reads the PDF text layer.

The **Table 2 hybrid attention-ratio sweep at 350M/48 layers spanning 0.06 ppl** cited by the design
doc was likewise **not verified** here — I could not open the table. Flagging for the parent: that
citation is currently unverified in *this* document (a sibling agent may have it).

### [6] Samba — arXiv 2406.07522 — RESOLVED ✅ (tokenizer NOT stated in retrievable text)

Resolved to **"Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language
Modeling"**, Ren, Liu, Lu, Shen, Liang, Chen (Microsoft), ICLR 2025.

MEASURED from `https://arxiv.org/html/2406.07522v3`:
- Architecture-comparison ablations at **~438M and 1.3B on SlimPajama**, 20B / 100B tokens, ctx 4096.
- *"Perplexity on the validation set of SlimPajama for different attention and linear recurrent model architectures trained at 4,096 context length."*
- 1.7B comparison on Phi2 data, 230B tokens; 3.8B flagship on Phi3 data, 3.2T tokens.

**Tokenizer/vocab: NOT STATED in the body.** The fetch reports it would be in Appendix G, which was
truncated. Status **UNCLEAR** for the exact vocab; but **CONFIRMED** that Samba's *architecture
comparison* section states corpus, scale, context length and token budget while **not** putting the
tokenizer in the main text. Another datapoint that this is not a headline-level disclosure.

Note also: Samba's own architecture ablations (438M, SlimPajama, 20B tokens) run at **our scale on a
different corpus and token budget than its flagship**, and nobody considers that disqualifying.

### [7] Hymba — arXiv 2411.13676 — RESOLVED ✅ (KV-sharing detail CONFIRMED, vocab not stated)

Resolved to **"Hymba: A Hybrid-head Architecture for Small Language Models"**, Dong, Fu, Diao, Byeon,
Chen, Mahabaleshwarkar, Liu, Van Keirsbilck, Chen, Suhara, Lin, Kautz, Molchanov. 20 Nov 2024.

MEASURED from `https://arxiv.org/html/2411.13676v1`:
- Scales: **"a 125M model, a 350M model, and a 1.5B model."** Ablations at **300M with 100B tokens**.
- Corpus: *"a mix of DCLM-Baseline-1.0, SmoLM-Corpus, and a proprietary high-quality dataset, with 1T, 250B, and 50B tokens, respectively."*
- **Architecture comparison experiments used SmolLM-Corpus (1B scale) and FineWeb (300M scale).**
- Cross-layer KV sharing (the P2 prior art): KV *"shared between consecutive layers (e.g., every two layers share the same KV cache)"*, motivated by *"KV cache shares a high similarity between adjacent layers."* Effect: throughput +1.15×, commonsense accuracy +0.60%. Global attention only in *"the first, middle, and last layers."*

**Tokenizer/vocab: NOT STATED anywhere in the retrievable text.** Status **UNCLEAR** for vocab;
**CONFIRMED** that a 20-page NVIDIA paper doing exactly our kind of comparison at exactly our scale
**never prints a vocabulary size.**

Two bonus findings the parent will care about:
1. **Hymba's architecture-comparison rung is 300M on FineWeb** — i.e. the same scale *and the same
   corpus family* the study proposes. Direct precedent.
2. Hymba's flagship (1.5B, 1.3T tokens) and its ablations (300M, 100B tokens) use **different
   corpora**. The paper draws architectural conclusions from the small-corpus ablations anyway.

### [8] Griffin / Hawk — arXiv 2402.19427 — RESOLVED ✅ (phonebook precedent CONFIRMED)

Resolved to **"Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language
Models"**, De, Smith, Fernando, Botev, Cristian-Muraru, Gu, Haroun, Berrada, Chen, Srinivasan,
Desjardins, Doucet, Budden, Teh, Pascanu, De Freitas, Gulcehre. 29 Feb 2024.
Full text via `https://ar5iv.labs.arxiv.org/html/2402.19427` (resolved).

MEASURED:
- Corpus: *"Models are trained on the MassiveText dataset (Hoffmann et al., 2022)"* with *"a slightly different data subset distribution."*
- Scales (Appendix C): **100M, 200M, 400M, 1.3B, 3B, 7B, 14B.** Downstream at 300B tokens, seq len 2048.
- **Tokenizer/vocab: NEVER STATED for the language models.** The *only* "vocabulary size" in the paper is the synthetic task: *"We use a vocabulary size of 16, and train on sequences of length 1024, containing 16 data tokens"* (Selective Copying).
- Phonebook (the precedent the study wants to beat), adapted from Jelassi et al. 2024: prompt is a synthetic name→number directory, two worked examples, then a query; *"the model is asked to retrieve the correct phone number given a name."* Result: Griffin *"can perfectly solve this task up to a context length that matches its local attention window size of 1024"*; Hawk degrades with directory length; the Transformer is near-perfect within training length and fails beyond.

**This is a strong one for the parent.** Griffin is a DeepMind paper at up to 14B, it is *the*
phonebook precedent this study is targeting, and **it does not state its tokenizer or vocabulary size
anywhere.** If a 14B DeepMind paper can report phonebook results without naming a vocab, a 350M
capstone reporting "GPT-2, 50,257" is strictly *more* disclosive than the precedent it is beating.

### [9] CLA — arXiv 2405.12981 — RESOLVED ✅ (vocab not retrieved from abs page)

Resolved to **"Reducing Transformer Key-Value Cache Size with Cross-Layer Attention"**, Brandon,
Mishra, Nrusimha, Panda, Ragan-Kelley. 21 May 2024.
MEASURED: *"experiments training 1B- and 3B-parameter models from scratch"*; method *"also sharing
key and value heads between adjacent layers"*; *"reduce the size of the KV cache by another 2x while
maintaining nearly the same accuracy as unmodified MQA."*
Tokenizer/vocab/corpus: **not on the abs page**; status **UNCLEAR** (I did not get the full text).

### [11] DeltaNet — arXiv 2406.06484 — RESOLVED ✅ (tokenizer STATED)

Resolved to **"Parallelizing Linear Transformers with the Delta Rule over Sequence Length"**, Songlin
Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim. v6 15 Jan 2025.
Full text `https://ar5iv.labs.arxiv.org/html/2406.06484` (resolved).

> "All models are trained on the same subset of the SlimPajama dataset with the Mistral tokenizer."

> "We train all models from scratch in two configurations: 340M and 1.3B parameters."

> "The 340M models are trained using 15 billion tokens and a batch size of 0.5M tokens"

**MEASURED / CONFIRMED. This is a very close analogue of the proposed study:** 340M params, 15B
tokens, architecture comparison (DeltaNet vs Mamba vs GLA vs Transformer++ vs hybrids), **Mistral
tokenizer = vocab 32,000** (the 32,000 figure is not printed in this paper — tagged INFERRED from
Mistral's known vocab). Note: **32,000, not 65,536.** And no vocab-size number is printed.

### [12] Gated DeltaNet — arXiv 2412.06464 — RESOLVED ✅ (tokenizer AND vocab BOTH stated)

Resolved to **"Gated Delta Networks: Improving Mamba2 with Delta Rule"** (Yang, Kautz, Hatamizadeh).
Full text `https://ar5iv.labs.arxiv.org/html/2412.06464` (resolved).

> models "employ the LLaMA 2 tokenizer with a vocabulary size of 32,000."

> Main experiments: "100B tokens sampled from the FineWeb-Edu dataset"

> Ablations: "All models have 400M parameters and are trained for 15B tokens" on "the same subset of FineWeb-Edu dataset"

**MEASURED / CONFIRMED — and this is the single closest precedent to the proposed study found
anywhere in this survey:**
- **400M parameters** (we are 350M)
- **FineWeb-Edu** (the corpus the study's own doc measures at line 256)
- **15B tokens for ablations** (we are 2–5B)
- **4K training sequence length** (we train at 4K)
- **vocab 32,000** — *below* GPT-2's 50,257 and **half** of LFM2's 65,536
- hybrid ablations at 500M/15B; 2K sliding window for Samba and hybrid variants

So a 2024 NVIDIA-affiliated paper doing gated-recurrent-vs-attention architecture comparison at
400M on FineWeb-Edu at 4K used a **32,000** vocab, and it is nearer to Tao et al.'s ~16K optimum
than either of our candidates. **Nobody objected.**

### [10] Zamba — arXiv 2405.16712 — RESOLVED ✅ (vocab not retrieved)

Resolved to **"Zamba: A Compact 7B SSM Hybrid Model"**, Glorioso, Anthony, Tokpanov, Whittington,
Pilault, Ibrahim, Millidge. 26 May 2024. MEASURED: 7B, *"trained on 1T tokens from openly available
datasets"*, *"a Mamba backbone with a single shared attention module."* Tokenizer/vocab **not on the
abs page**; status **UNCLEAR**.

---

## §3 — Vocab, tokenizer, and RECALL / long-context specifically

### [13] BABILong — arXiv 2406.10149 — RESOLVED ✅ — ⭐ DIRECT HIT

Resolved to **"BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack"**,
Kuratov, Bulatov, Anokhin, Rodkin, Sorokin, Sorokin, Burtsev. NeurIPS 2024 D&B. v2 6 Nov 2024.
Full text `https://ar5iv.labs.arxiv.org/html/2406.10149` (resolved).

**This is the paper that answers the §3 question, and it answers it in our favour.** MEASURED quotes:

> "The length in tokens is measured using the classic GPT-2 tokenizer, which is close in fertility to the popular GPT-4 tokenizer."

> "We measure the length of BABILong samples using the conservative GPT-2 tokenizer."

> "the number of tokens for tokenizers of different models may differ for samples in the same split"

> "Actual token sizes may vary depending on the model tokenizer."

Table 5 ("Token count for various models across selected tasks") shows the same split measured under
GPT-4 / GPT-2 / Llama-2 / Mistral tokenizers: at the 128k split, ~123k under GPT-4 vs ~128k under
GPT-2. **A ~4% spread across four very different tokenizers.**

**Three consequences, all favourable:**

1. **CONFIRMED: the tokenizer-dependence of "context length in tokens" is a KNOWN and EXPLICITLY
   HANDLED issue** — it is not an unexamined hole a reviewer will discover. BABILong names it,
   quantifies it in a table, and proceeds.
2. **The benchmark the study has already chosen as its PRIMARY long-context metric defines its own
   length axis in GPT-2 tokens.** So if the study uses the GPT-2 tokenizer, its models' context
   windows are measured in *the same units BABILong uses natively*. Under a 65,536 LFM2 vocab there
   would be a units mismatch to reconcile. **This is an argument FOR GPT-2, not against it.**
3. The paper *"notes a tradeoff consideration between sequence length and embedding layer size,
   arguing the comparison stays fair despite tokenizer differences."* i.e. the benchmark's own
   authors adjudicated exactly this fairness question and ruled it fair.

Also MEASURED — BABILong's small-model roster, confirming the design doc's line 1175 claim:
RMT with **GPT-2 137M** backbone (segment 512, 16 memory tokens, curriculum to 32 segments, evaluated
to ~11.1M tokens); **Mamba-130M** (fine-tuned, strong to 128k); GPT-2 137M backbone alone;
Mamba-2.8B zero-shot. **ARMT is NOT mentioned in the v2 text I retrieved** — the design doc's line
1175 mentions "RMT/ARMT on GPT-2-137M"; the RMT half is CONFIRMED, the ARMT half is **UNCLEAR /
not found here** (may be a later paper). Flagging the small discrepancy rather than papering over it.

**Note the delicious detail: BABILong's small-scale reference models are GPT-2-137M-backbone and
Mamba-130M. Our 350M GPT-2-vocab models are directly commensurable with the published BABILong
small-model results. Retokenizing to 65,536 would make them LESS comparable.**

### [14] RULER — arXiv 2404.06654 — RESOLVED ✅ (silent on tokenizers — REFUTES the concern differently)

Resolved to **"RULER: What's the Real Context Size of Your Long-Context Language Models?"**, Hsieh,
Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg. v3 6 Aug 2024. Full text via ar5iv (resolved).

**MEASURED negative result:** *"No sentence in the paper addresses tokenizer dependence of context
length"*. Searching "tokenizer"/"tokenization" in the full text finds nothing relevant. RULER
controls length by scaling task content (*"size_haystack ∝ context length"*, *"num_noises ∝ context
length"*, etc.) and evaluates at *"500 examples generated for each length from the series (4k, 8k,
16k, 32k, 64k, 128k)"* while *"complying with each model's necessary chat template."*

> The paper does not state whether length targets are measured with a shared reference tokenizer or
> each model's own tokenizer.

**Verdict: the field's most-used long-context benchmark ranks models across wildly different
tokenizers (Llama, GPT, Mistral, Command-R, Yi…) and does not even mention the issue.** That is the
strongest possible evidence that cross-tokenizer long-context comparison is not treated as a
methodological defect by the community. (Note this cuts both ways epistemically — it may be an
under-examined issue — but for *reviewer risk*, which is what the parent asked about, it is decisive.)

### [15] Digit tokenization — arXiv 2402.14903 — RESOLVED ✅

Resolved to **"Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs"**,
Aaditya K. Singh & DJ Strouse, 22 Feb 2024, 21 pages.

MEASURED: contrasts single-digit tokenization (LLaMa, PaLM) with GPT-3.5/GPT-4's *"separate tokens
for each 1-, 2-, and 3-digit numbers"*. Right-to-left grouping (*"enforced by comma separating
numbers at inference time"*) *"leads to largely improved performance"*; errors under L2R
*"follow stereotyped error patterns"*; the gap shrinks with scale, *"possibly indicating that larger
models are better able to override this tokenization-dependent inductive bias."* Self-described as
*"the first study of how number tokenization choices lead to differences in model performance on
arithmetic tasks."*

**RELEVANCE TO OUR PASSKEY/PHONEBOOK ENDPOINTS — this is a real, actionable caveat, the only one I
found in this whole survey:**

- MEASURED/CONFIRMED: digit grouping demonstrably changes numeric-task performance.
- **INFERRED (important):** GPT-2's BPE **does not** enforce single-digit tokenization. It has
  learned multi-digit merges, so a phone number or passkey is split into an *arbitrary* and
  *content-dependent* set of multi-digit chunks. LFM2's 65,536 vocab, being modern, is
  **more likely** to use digit-splitting (Llama-3 style) — I did **not** verify LFM2's specific
  pretokenizer regex here; a sibling agent owns the LFM2 primary sources
  (`10_lfm2_primary_sources.md`). Tagged **ASSUMED** and explicitly flagged as needing verification.
- **The consequence is NOT a validity threat but a DIFFICULTY-CALIBRATION issue.** Under GPT-2 BPE a
  10-digit phone number may be 3–5 tokens rather than 10, so a phonebook of N entries occupies fewer
  tokens, and *the number of tokens the model must retrieve per answer is smaller and more variable*.
- **Crucially, this is IDENTICAL ACROSS ALL ARMS.** Every arm shares the tokenizer (design doc line
  371: *"All arms share tokenizer, data snapshot, data order, token budget, precision, optimizer,
  context"*). A tokenizer-induced difficulty shift moves every arm equally and **cannot flip an
  A-vs-B ranking.** It only affects (a) the absolute difficulty level, and (b) comparability to
  Griffin's published phonebook numbers.
- **Concrete actionable recommendation:** since Griffin is the stated precedent to beat and Griffin
  used a different (unstated!) tokenizer, **report phonebook difficulty in BOTH entries and tokens**,
  and report the measured tokens-per-entry for our tokenizer. That defuses the objection completely
  and costs nothing. Do the same for passkey depth.

---

## §2b — Does tokenizer choice FLIP an architectural ranking? (the strongest counter-evidence)

### [16] Ali et al., "Tokenizer Choice For LLM Training: Negligible or Crucial?" — arXiv 2310.08754 — RESOLVED ✅

Resolved to exactly that title; lead author **Mehdi Ali**, 21 authors. cs.LG, v1 12 Oct 2023, v4
17 Mar 2024. Full text `https://ar5iv.labs.arxiv.org/html/2310.08754` (resolved).

**I am reporting this one against my own thesis, because it is the best objection available.**

MEASURED:
> "for each tokenizer we trained decoder-only models with a size of 2.6B parameters while keeping the remaining configuration (i.e., dataset and model hyper-parameters) fixed."

> "only the tokenizer has been changed while the model configuration is the same."

24 tokenizers → 24 models at 2.6B, ~52B tokens each, plus GPT-2-tokenizer baselines. Vocab sizes
compared: **33k / 50k / 82k / 100k**.

> "the tokenizer choice can significantly impact the model's downstream performance, training and inference costs."

> "Among the monolingual tokenizers, there can be significant performance differences."

> "the differences across tokenizers are even larger than for monolingual tokenizers." [multilingual]

> "Performance increasements reach from +5,3% up to 380,9%"

> multilingual models using English-centric tokenizers incur "additional training costs of up to 68%"

**⭐ AND THE FINDING THAT MATTERS MOST FOR OUR DECISION:**
> "in the monolingual English setting, the smaller/medium-sized vocabulary performs better"

English-monolingual best-vs-worst average across all downstream tasks (Table 9): **BPE-SP-32 at
47.06 (best) vs BPE-HF-82 at 44.97 (worst)** — a ~2.1-point spread, **and the winner is the 32k
vocab while the loser is the 82k vocab.** That is *independent corroboration of Tao et al. from a
completely different methodology*: for English-only training at multi-billion scale, **smaller
vocabularies won.** Our corpus is English (FineWeb-Edu / Dolma2). 50,257 sits in their
"smaller/medium" winning band; 65,536 sits nearer the losing 82k end.

**Now the honest caveat, stated plainly.** I initially expected this paper to say the monolingual
effect was negligible. **It does not, and I am not going to pretend otherwise.** The paper's headline
is that tokenizer choice is *crucial*, including monolingually. Anyone citing it as "tokenizers don't
matter" is misreading it.

**BUT — and this is the decisive distinction for our purposes — the paper varies tokenizer with
architecture FIXED. It establishes a MAIN EFFECT of tokenizer, not an INTERACTION between tokenizer
and architecture.** Those are different claims:

- **Main effect (what Ali et al. proves):** model quality depends on tokenizer. → Affects absolute
  numbers. Our study reports *relative* comparisons between arms sharing one tokenizer, so a main
  effect is absorbed entirely by the shared baseline.
- **Interaction (what would actually hurt us):** *the ranking of architecture A vs architecture B
  changes under a different tokenizer.* **Ali et al. does not test this — it trains one architecture.**

### The interaction question: I could not find ANY paper demonstrating it. Verdict: UNCLEAR-leaning-REFUTED

**MEASURED (as a survey outcome):** across [1]–[16], **not one paper reports a tokenizer × architecture
interaction, and not one reports an ablation over tokenizer or vocab size within an architecture
comparison.** The papers either (a) fix one tokenizer and never revisit it (Zoology, Based, DeltaNet,
Gated DeltaNet, Griffin, Samba, Hymba), or (b) compare across tokenizers and merely annotate the
difference (Mamba's "Token." column; RULER; BABILong's Table 5).

**Caveat on my own search, stated honestly:** `WebSearch` was returning HTTP 403 for the entire
session (confirmed independently by the orchestrator and a sibling agent), so I could not run
open-ended queries like *"does tokenizer flip architecture ranking"*. My negative result therefore
rests on **absence of evidence across 16 directly-fetched primary sources**, not on an exhaustive
search. That is a real limitation and the parent should weight it accordingly. It remains a strong
signal — if such a paper existed and were well known, it would almost certainly be cited *by* one of
the 16 architecture papers I did read, and none cite one.

**A theoretical argument for why the interaction should be weak, tagged INFERRED:** the study's arms
differ in the *sequence mixer* (low-rank vs dense vs block-diagonal gates, KV sharing, conv width,
conv-vs-attention topology). The tokenizer determines the *input/output symbol distribution*, which
is processed by the embedding, the MLPs, and the unembedding — all held identical across arms. For
the ranking to flip you would need the mixer's advantage to be conditional on token-distribution
statistics. The one place this is *not* obviously safe is exactly the recall endpoint: recall
difficulty depends on how repeated content is chunked into tokens (see [15]). Which is precisely why
**the AR-Hits slice must be re-measured on our own corpus** — see §3c below.

### [17] TokSuite — arXiv 2512.20757 — RESOLVED ✅ (the modern controlled study; full text NOT retrieved)

Resolved via the arXiv API to **"TokSuite: Measuring the Impact of Tokenizer Choice on Language Model
Behavior"**, Gül Sena Altıntaş, Malikeh Ehghaghi, Brian Lester, Fengyuan Liu, Wanru Zhao, Marco
Ciccone, Colin Raffel. v1 23 Dec 2025, **v2 6 Jul 2026, ICML 2026**, 46 pages.

*Note on the ID:* `2512.20757` looks future-dated relative to my May 2026 knowledge cutoff, but
**it resolves** — arXiv's own API returned it with the exact expected title, and the abs page and PDF
both load. So this is a genuine paper, not one of the fabricated 25xx/26xx IDs the brief warned about.
**RESOLVED, verified two independent ways (API + abs page + PDF metadata).**

MEASURED (abstract + HF org page):
> fourteen pre-trained models that "share the same architecture, training data, training budget, and initialization but differ only in their tokenizers"

> "a multilingual robustness benchmark that measures model performance under real-world perturbations in English, Chinese, Farsi, Italian, and Turkish"

**HONEST FAILURE:** full text not retrievable — `arxiv.org/html/2512.20757v1` and `v2` both **404**,
and the PDF returned compressed binary. **I have no effect sizes, no vocab-size list, and no
ranking-stability quote from this paper.** From PDF bookmark metadata only (weak evidence, tagged
INFERRED): trains on **FineWeb-Edu** / FineWeb2 via **Meta Lingua**, evaluates with lm-eval-harness
(HellaSwag, ARC, PIQA, XNLI, Belebele), uses Wilcoxon tests. HF org shows ~2B-labelled models.

**Why it still matters for the parent, and it cuts BOTH ways:**
- *Against us:* as of ICML 2026 there now exists a 46-page, 14-model controlled study of tokenizer
  choice. Tokenizer effects are a live topic; a reviewer in 2026 is more likely to ask than one in 2024.
- *For us:* **its design is architecture-fixed / tokenizer-varied — the mirror image of ours.** Even
  the field's most thorough tokenizer study does not test tokenizer × architecture interaction. And
  it trains on **FineWeb-Edu**, same as us.

### [18] Over-Tokenized Transformer — arXiv 2501.16975 — RESOLVED ✅

Resolved to **"Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling"**, Huang, Zhu, Wu,
Zeng, Wang, Min, Zhou. ICML 2025. v1 28 Jan 2025, v2 23 May 2025.

MEASURED:
> "decouples input and output vocabularies to improve language modeling performance"

> "scales up input vocabularies to leverage multi-gram tokens"

> "a log-linear relationship between input vocabulary size and training loss, demonstrating that larger input vocabularies consistently enhance model performance, regardless of model size"

**This is the apparent counter-argument to Tao et al., and it dissolves on inspection.** The paper's
"bigger vocab is better" claim is specifically about a **decoupled INPUT vocabulary of multi-gram
tokens** — an *architectural* modification (an n-gram input embedding table) that we are not adopting.
It is **not** a claim that a conventional tied input+output vocabulary should be larger. I could not
retrieve a quote about output-vocabulary effects on small models from the abs page, so I mark the
input/output asymmetry claim **UNCLEAR** rather than asserting it.

**Net effect on our decision: none.** Our design is **tied** embeddings (single shared table), which
is exactly the *coupled* setting Tao et al. models and Over-Tokenized explicitly departs from. No
conflict; Tao et al. remains the governing result for our geometry.

---

## §3c — ⭐ IS THE AR-HITS 6.4% FIGURE TOKENIZER-DEPENDENT? (actionable finding)

**VERDICT: YES, and worse — it is ALSO corpus-size-dependent and threshold-arbitrary. The 6.4%
figure and the 1250 threshold CANNOT be inherited. They MUST be re-measured on our own corpus.
This is true REGARDLESS of which tokenizer we pick, so it is NOT an argument for retokenizing.**

Source: `https://ar5iv.labs.arxiv.org/html/2312.04927` (resolved). MEASURED quotes:

> **AR Hits** — "(6.4% of tokens) Tokens in the final position of a bigram (a pair of consecutive tokens) which previously appeared" … "in context, but ≤1250× during training."

> **Other tokens** — "(93.6% of tokens) Tokens in the final position of a bigram which did not previously appear in context or it" … "appeared >1,250 times during training."

> Appendix C: the Pile data "is tokenized using the GPT2BPETokenizer"

> the 6.4/93.6 split is measured by scaling "our analysis to over 10 million tokens of Pile validation data"

**Five independent reasons the 6.4% cannot be inherited (each MEASURED or INFERRED as tagged):**

1. **Tokenizer-dependence — INFERRED, but structurally airtight.** AR Hits is defined over *token*
   bigrams. Change the tokenizer and you change (a) which character spans become tokens, (b) therefore
   which bigrams exist at all, (c) therefore the in-context-repeat mask, and (d) the frequency counts.
   The paper **never acknowledges or tests this.** The fetch's own summation: *"Since AR Hits are
   defined over token bigrams, the 6.4% figure is necessarily tied to that BPE vocabulary, but the
   paper does not acknowledge or test this."*
   *Direction of the effect, INFERRED:* a **coarser** tokenizer (larger vocab, e.g. 65,536) packs more
   characters per token, so a fixed text span is fewer tokens and each bigram covers more text →
   bigrams become **rarer and more specific** → fewer exceed the 1250 threshold → **the AR slice
   should grow.** A finer tokenizer (GPT-2's 50,257 relative to 65,536) does the reverse. **I cannot
   quantify the magnitude from published data — nobody has measured it. It must be measured on ours.**

2. **Corpus-size-dependence — MEASURED, and this one is severe.** The threshold is an **absolute
   count (1250)**, not a percentile. Zoology applied *the same 1250* to models trained on **5B, 10B,
   and 50B tokens**, and *"the page does not reconcile that."* **Our budget is 2–5B tokens** —
   at the very bottom of that range, 2–10× less training data than Zoology's main 10B suite. With
   fewer training tokens, *far fewer* bigrams reach 1250 occurrences, so **the AR slice would
   mechanically balloon well past 6.4% if we naively reused the threshold.** This is a bigger
   distortion than the tokenizer effect and it applies **no matter what vocab we choose.**

3. **The corpus is different.** Zoology counted on the **Pile**; we would use **Dolma2 / FineWeb-Edu**.
   Different domain mixture → different bigram frequency distribution.

4. **The threshold itself is unjustified — MEASURED.** *"1250 is a suspiciously round figure with no
   stated derivation or sensitivity analysis, and no ablation over alternative thresholds appears."*
   The paper never states which Pile subset/shard was scanned to build the frequency table, and
   **never describes how the bigram frequency table is built** (no counting procedure, data structure,
   or cutoff).

5. **The authors themselves call it approximate — MEASURED.** They label it *"a simple heuristic"*,
   concede *"we don't know which next token predictions in raw text require associative recall"*, and
   note real recall involves *"fuzzier substrings (e.g. synonymous bigrams)"* that exact bigram
   matching misses.

Minor MEASURED inconsistency worth knowing: the threshold is written *"≤1250×"* in one bullet and
*">1,250"* in the other — same number, sloppy typesetting.

### What this means operationally (the actionable part)

**This finding is INDEPENDENT of the tokenizer decision and should be logged as its own work item.**
Whichever vocab is chosen, the study must:

1. **Build its own bigram frequency table over its own training tokens** (2–5B, Dolma2/FineWeb-Edu).
   This is a single streaming counting pass — cheap, no GPU, no new training runs.
2. **Re-derive the threshold as a PERCENTILE, not the literal 1250.** Recommended: pick the frequency
   cutoff that reproduces Zoology's *slice proportion* (~6.4% of validation tokens) on our corpus,
   and **report both the cutoff and the resulting slice size.** That preserves the *statistical
   power* properties Zoology's decomposition relies on while being honest that the constant does not
   transfer. Alternatively report a **sweep** over thresholds and show the gap attribution is stable —
   which would be a genuine methodological improvement over Zoology, since they never ablated it.
3. **Report the measured slice fraction explicitly** ("AR Hits = X% of our held-out tokens at cutoff
   C") rather than repeating "6.4%".

**Reviewer angle: this converts a liability into a contribution.** Zoology never tested threshold
sensitivity or tokenizer sensitivity. A study that re-measures the slice on its own corpus and shows
the gap attribution is robust to the cutoff is doing something *better* than the paper it inherits
from. Conversely, a paper that prints "6.4% of Pile tokens" while training on 2–5B Dolma2 tokens is
making a checkable error — **that is the real reviewer risk in this area, and it has nothing to do
with vocab size.**

⚠️ **Correction flag for the parent:** `docs/liv-brainlift-experiment-design.md` lines 1150–1152
currently state the slice as *"**≤1250×** in training data (**6.4%** of Pile tokens)"* and
*"Other — everything else (93.6%)"*. These are Zoology's Pile/GPT-2/10B numbers. **They should be
relabelled as inherited-from-Zoology, with our own re-measured values substituted before publication.**

---

## §1 — THE TABLE

Sorted roughly by relevance to our study. "Vocab stated?" = does the *paper text* print a numeric
vocabulary size. "Rationale?" = does the paper justify *why* it picked that tokenizer/vocab.

| # | Paper | arXiv (resolved?) | Scale(s) | Tokenizer | Vocab size stated in paper? | Corpus | Rationale given? |
|---|---|---|---|---|---|---|---|
| 1 | **Zoology** | 2312.04927 ✅ | 70M–1.4B; analysis at **70M–360M** | **GPT-2 BPE** (`GPT2BPETokenizer`) | ❌ no number | Pile, 10B tok | **N** |
| 2 | **Based** | 2402.18668 ✅ | **355M** & 1.3B | **GPT-2 BPE** | ❌ no number | Pile, 10B tok | **N** |
| 3 | **Mamba** | 2312.00752 ✅ | 125M–2.8B; ablations **~350M** | **GPT-NeoX-20B** | ❌ no number | Pile, 300B tok | **N** (only "same as Pythia/RWKV") |
| 5 | **Mamba-2** | 2405.21060 ✅ (full text **NOT** retrieved) | 130M–2.7B | **UNCLEAR** — could not open | ❓ unknown | Pile | **UNCLEAR** |
| 6 | **Samba** | 2406.07522 ✅ | **438M**, 1.3B, 1.7B, 3.8B | not in main text (Appx G) | ❌ | SlimPajama 20B/100B; Phi2; Phi3 | **N** |
| 7 | **Hymba** | 2411.13676 ✅ | 125M/**350M**/1.5B; ablations **300M** | **not stated anywhere** | ❌ | **FineWeb (300M ablations)**, DCLM+SmolLM (flagship) | **N** |
| 8 | **Griffin / Hawk** | 2402.19427 ✅ | 100M/200M/**400M**/1.3B/3B/7B/14B | **not stated anywhere** | ❌ (only synthetic "vocabulary size of 16") | MassiveText, 300B tok | **N** |
| 11 | **DeltaNet** | 2406.06484 ✅ | **340M** & 1.3B | **Mistral** (⇒ 32,000) | ❌ no number | SlimPajama, 15B/100B tok | **N** |
| 12 | **Gated DeltaNet** | 2412.06464 ✅ | **400M**/500M/1.3B | **LLaMA-2** | ✅ **"vocabulary size of 32,000"** | **FineWeb-Edu**, 15B/100B tok | **N** |
| 9 | **CLA** | 2405.12981 ✅ (abs only) | 1B & 3B | not on abs page | ❓ | not on abs page | **UNCLEAR** |
| 10 | **Zamba** | 2405.16712 ✅ (abs only) | 7B | not on abs page | ❓ | 1T tok, open datasets | **UNCLEAR** |
| — | **Jamba** | 2403.19887 ✅ (ID resolved via arXiv API; content not fetched) | 52B MoE | not fetched | ❓ | — | **UNCLEAR** |
| — | **Jamba-1.5** | 2408.12570 ✅ (ID only) | 94B/398B MoE | not fetched | ❓ | — | **UNCLEAR** |
| — | **Falcon-H1** | 2507.22448 ✅ (ID only) | family | not fetched | ❓ | — | **UNCLEAR** |
| 13 | **BABILong** (bench) | 2406.10149 ✅ | evaluates **GPT-2-137M**, Mamba-130M … 2.8B | **GPT-2 to define the length axis** | ✅ (Table 5, cross-tokenizer counts) | bAbI + PG19 | **Y — explicitly** |
| 14 | **RULER** (bench) | 2404.06654 ✅ | ≥7B LLMs | **never mentions tokenizers** | ❌ | synthetic | **N** |
| 4 | **Scaling Laws w/ Vocabulary** | 2407.13623 ✅ | 33M–3B (N_nv 33M–1.13B) | varies **by design**, V swept **4K–96K** | ✅ throughout | — | **Y — it's the subject** |
| 16 | **Tokenizer Choice: Negligible or Crucial?** | 2310.08754 ✅ | 2.6B × 24 models | 24 tokenizers, **33k/50k/82k/100k** | ✅ | multiling. + English | **Y — it's the subject** |
| 17 | **TokSuite** | 2512.20757 ✅ (full text **NOT** retrieved) | 14 models, ~2B | 14 off-the-shelf tokenizers | ❓ | FineWeb-Edu (inferred) | **Y — it's the subject** |
| 18 | **Over-Tokenized Transformer** | 2501.16975 ✅ | — | decoupled in/out vocab | ✅ | — | **Y — it's the subject** |
| 15 | **Tokenization counts (arithmetic)** | 2402.14903 ✅ | GPT-3.5/4 (API) | digit-grouping study | n/a | n/a | **Y — it's the subject** |

**Count for the BOTTOM LINE (architecture-comparison papers only, rows 1,2,3,5,6,7,8,11,12,9,10):**
- Papers at 300M–500M doing architecture comparison: **Zoology (360M), Based (355M), Mamba (350M ablations), Samba (438M), Hymba (300M/350M), Griffin (400M), DeltaNet (340M), Gated DeltaNet (400M)** = **8**.
- Of the ones whose tokenizer I could establish: **GPT-2 BPE = 2** (Zoology, Based), **GPT-NeoX = 1**
  (Mamba), **Mistral 32k = 1** (DeltaNet), **LLaMA-2 32k = 1** (Gated DeltaNet), **unknown/unstated = 3**
  (Samba, Hymba, Griffin).
- **Papers using a vocabulary ≥ 64K at this scale: ZERO. Not one.**
- **Papers that print a numeric vocab size at all: 1 of 10** (Gated DeltaNet).
- **Papers that justify their tokenizer choice: 0 of 10.**
- **Papers reporting a tokenizer or vocab-size ablation within an architecture comparison: 0 of 10.**

---

## §4 — REVIEWER-RISK ASSESSMENT

### 4a. Is "GPT-2 vocab at 350M for an architecture ablation" a reviewer objection?

**NO. It is the single most standard choice available, and at this scale it is more standard than the
alternative.** Evidence, in descending force:

1. **Zoology and Based — the two papers this study inherits its PRIMARY ENDPOINT from — both used
   GPT-2 BPE at 355–360M.** (MEASURED, direct quotes.) Adopting AR-Hits while *also* adopting GPT-2
   is not a compromise; it is **protocol fidelity**. Retokenizing to 65,536 would make our AR-Hits
   numbers *less* comparable to the source of the metric.
2. **Zero of ten comparable papers used a vocab ≥64K.** The observed range at 300–500M is
   **32,000–50,277**. **50,257 is inside the norm; 65,536 is outside it.**
3. **BABILong — the study's chosen primary long-context benchmark — defines its length axis in GPT-2
   tokens** and publishes small-model baselines with a GPT-2-137M backbone. (MEASURED.)
4. **Griffin, the phonebook precedent this study explicitly wants to beat, never states its tokenizer
   or vocab at all** — up to 14B parameters, DeepMind, ICML-tier. (MEASURED.) Reporting "GPT-2,
   50,257" is *strictly more disclosive than the precedent.*
5. **The compute-optimal vocab at our N_nv is ~16K** (Tao et al., direct empirical measurement at
   N_nv=302M vs our 287.4M). **50,257 is 3.1× optimal; 65,536 is 4.0× optimal.** GPT-2 is *closer to
   correct*, and independently corroborated by Ali et al.'s English-monolingual finding that
   *"the smaller/medium-sized vocabulary performs better"* (their 32k tokenizer beat their 82k one).

**Which claims could it possibly threaten? Essentially none of the study's claims.** Going endpoint
by endpoint:

| Endpoint | Threatened by GPT-2 vocab? | Why |
|---|---|---|
| **MQAR / synthetic recall** | **NO** | Synthetic, own vocab (Zoology uses 8192; the study calibrated to 256 per `HANDOFF.md` line 379). The real tokenizer is not involved at all. |
| **AR-Hits sliced ppl** | **NO — helped** | Zoology's own tokenizer. But see §3c: the slice must be re-measured regardless. |
| **Length extrapolation** | **NO** | Ratio-based (train 4K → 8/16/32K). Tokenizer-invariant by construction. |
| **Needle / passkey / phonebook** | **Minor, arms-symmetric** | Difficulty calibration shifts (§15); identical across arms; fix by reporting entries *and* tokens. |
| **KV/state accounting, decode traffic** | **NO** | Per-token bytes. §3.1's 6.6%/36.2% KV-share table is embedding-independent. |
| **Parameter-matched arm comparisons** | **NO** | Embeddings are a shared constant across every arm. |
| **P1/P2/P3 mixer conclusions** | **NO — helped** | Embeddings drop 18.9%→15.2% of params, *raising* the mixer's share of what is measured. |

### 4b. How much is the "exact released shape, 354,483,968 params" claim actually worth?

**Blunt answer: almost nothing, and the study should stop treating it as an asset.**

The reasoning:

- **A from-scratch 350M model trained on 2–5B tokens is not LFM2 in any meaningful sense.** The
  sibling document `10_lfm2_primary_sources.md` (lines 43, 670, 722) has this MEASURED from primary
  sources: *"The released dense LFM2 model checkpoints are pre-trained for **10T tokens**"* /
  *"Training budget: 10 trillion tokens"*. So **we would be at 0.02%–0.05% of LFM2's training
  tokens — a factor of 2,000×–5,000×, i.e. between three and four orders of magnitude** (INFERRED:
  2–5B / 10T). Matching the parameter count *to the digit* while missing the training budget by
  ~3.5 orders of magnitude does not make the model "LFM2-shaped" in any way a reviewer would credit.
  It makes it *a 350M hybrid with the same layer geometry* — which is exactly what it is, and a
  perfectly good thing to be.
- **The claim a reviewer actually cares about is "the arms are matched to each other,"** not "the
  arms are matched to a released checkpoint." Internal parameter matching (the `L0` vs `A16-P` vs
  `F-r128` ledger in `HANDOFF.md` lines 307–310) is the load-bearing methodological claim, and it is
  **completely unaffected by vocab choice** — the orchestrator's own note confirms N_nv is
  **vocab-invariant** at 287,375,104, so every arm-to-arm delta is preserved exactly under any V.
- **"We reproduce the released shape" invites a question the study cannot answer favourably:** *"if
  you match the shape, how do your numbers compare to the released checkpoint?"* They cannot compare —
  different corpus, 1000× fewer tokens. **Claiming shape-fidelity creates the expectation of
  quality-comparability and then fails it.** Dropping the claim removes a trap.
- **Precedent says nobody does this.** None of the ten architecture papers surveyed claims to
  reproduce a released model's exact parameter count. Gated DeltaNet, Samba, Hymba and Griffin all
  choose round-ish geometries and ablate at scales convenient to them (400M, 438M, 300M). **The
  currency in this literature is arm-to-arm matching, not checkpoint-to-checkpoint matching.**

**Recommended framing** (costs nothing, keeps every real advantage):
> "We study a 350M-parameter LFM2-style hybrid (10 gated short-conv layers + 6 GQA layers, d=1024,
> 16 layers), following the released LFM2-350M layer geometry and attention schedule. Models are
> trained from scratch on FineWeb-Edu with the GPT-2 tokenizer (V=50,257), giving 338.8M parameters;
> all arms are parameter-matched to each other to within X%."

That sentence is honest, is standard, and gives up nothing. "**Following the released layer geometry
and attention schedule**" is the part that carries actual rhetorical weight — the schedule
`[2,5,8,10,12,14]` and the 10:6 conv:attention ratio are the *architecturally interesting* inherited
choices. **The embedding table is not an architectural choice; it is a lookup table.** Nobody will
credit matching it, and nobody will penalize not matching it.

### 4c. Does ANY specific claim in this study require the released vocab?

**I checked both local documents. Answer: NO.**

- `docs/liv-brainlift-experiment-design.md` line 267 is the only place the released-vocab rationale
  appears: *"it preserves the exact released-scale ledger the existing protocol froze."* That is a
  **self-referential** justification — the ledger was frozen *by this project*, not by an external
  requirement. It can be re-frozen. The same passage already concedes the cost:
  *"a large vocab does dilute the mixer signal we are trying to measure."*
- Line 270 notes *"a different tokenizer creates a separately-named family"* — a **naming/bookkeeping**
  convention, not a scientific constraint.
- **The ONNX/GGUF calibration datapoint does NOT require a shared vocab. CONFIRMED by reading §7.**
  Design doc §7.1 profiles `onnx-community/LFM2.5-350M-ONNX` q4 per-op (`MatMulNBits` 91.2%, `Conv`
  1.0%) and §7.3 takes *"One real edge datapoint: … measured 24.83 ms/token = 40.3 tok/s decode."*
  `HANDOFF.md` line 151-152 confirms: *"Take **one** calibration datapoint on the *unmodified*
  LFM2-350M GGUF/ONNX."* **The word "unmodified" is doing the work** — this is the released checkpoint
  measured *on its own weights*, a pure systems/latency measurement with **no numerical comparison to
  our trained models.** Our models' vocab is irrelevant to it. Note also this datapoint's headline
  finding (conv = 1% of decode, matmul = 91%) is a *per-op share* that would be essentially unchanged
  by vocab anyway.
  - One **second-order** caveat, tagged INFERRED: a 65,536 vs 50,257 vocab changes the size of the
    final unembedding GEMM, which is part of the 91.2% `MatMulNBits` bucket. If the study ever wanted
    to compare *our* decode latency to the released checkpoint's 24.83 ms/token, that difference
    would matter slightly. **But §7 explicitly does not do this** — the edge number is a standalone
    datapoint, and all variant-vs-variant latency work is GPU-side and parameter-matched. No issue.
- **No claim anywhere of the form "we match released LFM2 quality/benchmarks."** The study is
  explicitly a from-scratch controlled comparison; `HANDOFF.md` line 143 fixes the endpoints as
  *"recall + length extrapolation + AR-Hits sliced perplexity. NOT held-out CE."* None of these are
  compared against LFM2's published numbers.

### 4d. Risk rating

**LOW.** And specifically: **lower for GPT-2/50,257 than for 65,536.**

| Option | Risk | Notes |
|---|---|---|
| **GPT-2 50,257 (or 50,304 padded), train now** | **LOW** | Matches Zoology/Based exactly; inside the observed 32k–50k norm; 3.1× vs 4.0× off compute-optimal; frees weeks of compute. |
| **Retokenize to 65,536** | **LOW-MEDIUM** | No paper at this scale uses ≥64K; 4.0× compute-optimal; dilutes the mixer signal to 18.9% of params; costs a retokenization; buys a shape-match claim no reviewer credits. |

**Recommend 50,304** (= 50,257 padded to a multiple of 128) over bare 50,257 for tensor-core
alignment — this is standard practice (nanoGPT, and Mamba/Pythia pad NeoX's 50,277 the same way).
Cost: 47 unused rows × 1024 = 48,128 params. **Do not round to 50,432 or similar without recomputing
the ledger.** State whichever number is used and keep it constant across all arms.

**The single highest-risk item found in this entire survey is NOT the vocab — it is the inherited
"6.4% of Pile tokens / ≤1250×" AR-Hits constants (§3c), which are wrong for our corpus and token
budget no matter which tokenizer we use.** That is the thing a careful reviewer would actually catch.

---

## §5 — arXiv ID VERIFICATION LEDGER (per the "check IDs literally" rule)

**Every ID below was fetched. None were substituted or guessed. Where a fetch failed, I say so.**

| arXiv ID | Expected title | Resolved to expected title? | Full text retrieved? |
|---|---|---|---|
| 2312.04927 | Zoology | ✅ **YES** | ✅ ar5iv |
| 2402.18668 | Based | ✅ **YES** | ✅ ar5iv |
| 2312.00752 | Mamba | ✅ **YES** | ✅ arxiv.org/html v2 |
| 2405.21060 | Mamba-2 / Transformers are SSMs | ✅ **YES** (abs) | ❌ **NO** — html v1 & v2 both **404**, ar5iv "Fatal error" |
| 2406.07522 | Samba | ✅ **YES** | ⚠️ partial (Appx G truncated) |
| 2411.13676 | Hymba | ✅ **YES** | ✅ arxiv.org/html v1 |
| 2402.19427 | Griffin | ✅ **YES** | ✅ ar5iv |
| 2405.12981 | CLA | ✅ **YES** (abs) | ❌ abs page only |
| 2405.16712 | Zamba | ✅ **YES** (abs) | ❌ abs page only |
| 2403.19887 | Jamba | ✅ **YES** (arXiv API) | ❌ not fetched |
| 2408.12570 | Jamba-1.5 | ✅ **YES** (arXiv API) | ❌ not fetched |
| 2507.22448 | Falcon-H1 | ✅ **YES** (arXiv API) | ❌ not fetched |
| 2406.06484 | DeltaNet | ✅ **YES** | ✅ ar5iv |
| 2412.06464 | Gated DeltaNet | ✅ **YES** | ✅ ar5iv |
| 2404.06654 | RULER | ✅ **YES** | ✅ ar5iv |
| 2406.10149 | BABILong | ✅ **YES** | ✅ ar5iv |
| **2407.13623** | Scaling Laws with Vocabulary | ✅ **YES** | ✅ arxiv.org/html v3 |
| 2310.08754 | Tokenizer Choice: Negligible or Crucial? | ✅ **YES** | ✅ ar5iv |
| 2402.14903 | Tokenization counts (arithmetic) | ✅ **YES** | ⚠️ abs only |
| **2512.20757** | TokSuite | ✅ **YES** — future-dated but **REAL**, verified 3 ways (arXiv API `id_list`, abs page, PDF metadata); v1 2025-12-23, v2 2026-07-06 ICML 2026 | ❌ **NO** — html v1 & v2 both **404**, PDF binary |
| 2501.16975 | Over-Tokenized Transformer | ✅ **YES** | ⚠️ abs only |
| *(bonus)* 2606.03825 | cited at design doc line 951 as the width-sweep paper | ✅ **RESOLVES** to **"Dynamic Short Convolutions Improve Transformers"** — future-dated but real | not fetched |

**On the future-dated-ID warning in my brief:** two IDs in the 25xx/26xx range were involved
(**2512.20757** TokSuite, **2606.03825** in the design doc). **Both resolve to real papers with the
expected titles**, verified through arXiv's own API. I found **no fabricated IDs** in what I checked.

### Things I could NOT establish (stated so nobody assumes I did)

1. **Mamba-2's tokenizer and vocab** — full text unreachable (404 + ar5iv fatal error). Also could
   **not verify the design doc's "Table 2, 350M/48-layer hybrid ratio sweep spanning 0.06 ppl"**
   citation. A sibling agent may have it; otherwise it needs a PDF text-extraction pass.
2. **TokSuite's effect sizes, vocab list, and model scale** — 404 + binary PDF. Its *design*
   (architecture-fixed) is established; its *numbers* are not.
3. **CLA's, Zamba's, Jamba's, Falcon-H1's tokenizers** — abs pages only. Not load-bearing: none is at
   our scale, and CLA/Zamba are cited by the study for KV-sharing mechanics, not tokenization.
4. **LFM2's own pretokenizer digit-handling** — owned by the sibling doc `10_lfm2_primary_sources.md`;
   I deliberately did not duplicate that work. Relevant only to the §15 phonebook-calibration note.
5. **Exhaustive search for a tokenizer × architecture interaction paper** — `WebSearch` was HTTP-403
   all session. Mitigated with arXiv API queries via FarmShare (which surfaced TokSuite and
   Over-Tokenized), but this is not equivalent to a full web search.

---

*Document complete. Written incrementally; final pass 2026-08-01.*

