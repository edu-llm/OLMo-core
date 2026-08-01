# Evaluation Methodology for Associative Recall and Long-Context Ability in Efficient/Hybrid LMs

**Purpose.** Support an experiment design that compares a *mostly-LIV* hybrid (many gated
short-convolution layers + a few GQA attention layers) against (a) all-attention and
(b) recurrent-hybrid baselines at 100M–1B params. The evaluations below are chosen to
**expose retrieval failures that average perplexity hides**.

**Status legend.** ✅ verified from primary source · ⚠️ partially verified / inferred ·
❓ unknown, flagged.

**Document map**
1. MQAR / Zoology / Based / JRT — the core recall diagnostic (§1)
2. Phonebook, passkey, needle-in-a-haystack (§2)
3. RULER and the long-context benchmark landscape (§3)
4. Entity/state tracking + expressivity theory (§4)
5. Instruction persistence (§5)
6. Small-scale eval validity (§6)
7. Harness tooling (§7)
8. Statistical rigor and comparison protocol (§8)
9. **Recommended evaluation protocol** — tiered suite (§9)

---

## 1. MQAR — Multi-Query Associative Recall (Zoology)

**Primary sources**
- Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. *Zoology: Measuring and
  Improving Recall in Efficient Language Models.* arXiv:2312.04927 (ICLR 2024).
  <https://arxiv.org/abs/2312.04927>
- Code: <https://github.com/HazyResearch/zoology> (Apache-2.0; paper is CC0)
- Generator: `zoology/data/multiquery_ar.py`
  <https://github.com/HazyResearch/zoology/blob/main/zoology/data/multiquery_ar.py>
- Reproduction configs: `zoology/experiments/paper_configs/iclr24_zoology_figure2/configs.py`

### 1.1 Why MQAR exists (the motivating measurement)

Zoology's framing result is not the synthetic — it is the **perplexity decomposition on real
data**, and this is the single most reusable idea in the paper for our experiment. ✅

They define **AR Hits** by a purely mechanical heuristic (§3.1, verbatim structure):

> An AR Hit is the last token of an n-gram repeated in context … some common n-grams (e.g.
> "of the") could have been memorized during training, so we factor in the frequency with
> which an n-gram appeared in the training data.

Concretely, the Pile validation tokens are sliced into two groups:

| Slice | Definition | Share of tokens |
|---|---|---|
| **AR Hits** | Final token of a **bigram** that previously appeared **in the same context**, and appeared **≤ 1250×** in the *training* data | **6.4 %** |
| **Other** | Final token of a bigram not previously in context, or seen > 1250× in training | 93.6 % |

Gap attribution formula (verbatim from the paper):

```
% of gap due to AR   =   Δ log(φ_AR) · |T_AR|
                        ─────────────────────
                          Δ log(φ) · |T|
```

where φ is perplexity and T the token set. Interpretation given by the authors: "the fraction
of the overall gap that would close if a model matched attention on the AR slice."

**Headline numbers (Table 1, 10B tokens of Pile, log-ppl with NLL in parens):** ✅

| Model | Params (M) | Overall | AR Hits | Other | % of gap from AR |
|---|---|---|---|---|---|
| Attention | 125 | 11.01 (2.40) | **2.16 (0.77)** | 12.45 (2.52) | — |
| Long Conv | 128 | 16.98 (2.83) | 25.62 (3.24) | 16.46 (2.80) | 40.1 % |
| H3 | 168 | 12.06 (2.49) | 6.75 (1.91) | 12.60 (2.53) | 88.4 % |
| Hyena | 158 | 11.60 (2.45) | 5.00 (1.61) | 12.28 (2.51) | 100.0 % |
| RWKV | 169 | 11.64 (2.45) | 5.70 (1.74) | 12.29 (2.51) | 100.0 % |
| Attention | 360 | 9.44 (2.25) | **1.98 (0.69)** | 10.62 (2.36) | — |
| Long Conv | 360 | 13.13 (2.57) | 13.27 (2.59) | 13.12 (2.57) | 40.5 % |
| H3 | 357 | 10.38 (2.34) | 4.81 (1.57) | 11.00 (2.40) | 65.8 % |
| Hyena | 358 | 10.07 (2.31) | 3.83 (1.34) | 10.75 (2.38) | 98.2 % |
| RWKV | 351 | 9.79 (2.28) | 3.82 (1.34) | 10.51 (2.35) | 100.0 % |

Key claims: ✅
- Gated convolutions trail attention by **up to 2.1 perplexity points** on the Pile.
- **82 %** of the average gap is explained by the AR slice, which is only **6.4 %** of tokens.
- **On "other" tokens there is essentially no gap.** This is the crux: *average perplexity
  hides the retrieval deficit almost perfectly.*
- A **70M attention** model beats a **1.4B Hyena** (20× larger) on AR-slice perplexity
  (2.41 vs 3.43 ppl) — 1.4B models trained for 50B tokens (Table 5).
- The AR gap persists at 7B scale (RWKV vs Llama-2, Appendix G), and RWKV degrades sharply
  as the *number of queries per example* increases while attention is flat.

> **Actionable for our experiment:** this AR-Hit slicing is *cheap* (it's just a re-weighted
> eval-loss pass over held-out text with a bigram-repeat mask + a training-frequency table)
> and it is exactly the "expose what average perplexity hides" instrument requested. It is
> a **Tier-1** diagnostic. See §7.4 for implementation notes and §9.

### 1.2 The MQAR synthetic — exact construction

Prior AR synthetics (Fu et al. H3, Poli et al. Hyena, Olsson et al. induction heads) used
**one query per sequence, at a fixed position, with vocab ≤ 40 tokens** (i.e. vocab smaller
than model dimension). Zoology's diagnosis is that this is why gated convs appeared to "solve"
AR. MQAR fixes three things at once: **multiple queries**, **varying positions/gaps**, and
**vocab ≫ d**. ✅

Task form (verbatim from the paper's example; the sequence is a single autoregressive stream):

```
A 4  B 3  C 6  F 1  E 2   →   A ?  C ?  F ?  E ?  B ?
└──── key–value pairs ────┘   └──────── queries ────────┘
                              (correct outputs: 4, 6, 1, 2, 3)
```

**Algorithm 1 (MQAR Synthetic Procedure), verbatim from Appendix E:** ✅

> **Input:** Vocabulary C, Sequence length N, Power-law parameter α, Number of Key-Value Pairs D
> 1. Let the first half of C be keys K and the second half be values V.
> 2. Pair each key token k ∈ K with a random value token v ∈ V.
> 3. Sub-select D random key-value pairs to include in the sequence.
> 4. Place the D key-value pairs at the start of the sequence (i.e. consuming the first 2D positions).
> 5. Place a second occurrence of each d ∈ D at a distance from the first occurrence in the
>    sequence. The distance for each d ∈ D is selected at random from the positions [2D..N],
>    where the probability of choosing each position follows the power law distribution
>    specified by α.
> 6. Output the synthetic sequence.

**The two difficulty knobs (verbatim intent, Appendix E.1):** ✅
1. **Size of the key-value map** — `D` = number of unique KV pairs per example. Stresses
   *storage capacity* (how much must be held in state).
2. **Number of gaps in the data** — the number of unique *token-interaction distances*
   required. Controlled jointly by `N`, `D`, `α`. Holding α and D fixed while increasing N
   increases the number of distinct gaps. Stresses *input-dependence of the mixing*.

**Gap distribution.** Gaps follow a power law, motivated by measured Pile statistics
(Appendix D: most AR hits are within ~100 token positions of the prior bigram occurrence,
with a long tail). ✅ The implementation is:

```python
space = (input_seq_len - context_size) // 2
p = power_a * np.arange(1, space + 1) ** (power_a - 1)
p = p / p.sum()
```

with `power_a = 0.01` as the default used in all paper configs. Note `power_a = 1.0` gives a
**uniform** distribution (documented in the docstring); `power_a = 0.01` is a strongly
front-loaded power law. ⚠️ Naming: the paper calls it α; the code calls it `power_a`, and
older configs use `train_power_a` / `test_power_a`.

**Vocabulary split.** Verified from code: keys are drawn from `[1, vocab_size//2)` and values
from `[vocab_size//2, vocab_size)`, i.e. **disjoint key and value vocabularies**, each key
appearing exactly once per example (sampled without replacement). ✅

**Label masking.** Only the token immediately after a query key is supervised; everything else
is `-100` (ignored by loss and metrics). From the code docstring: ✅

```
        Key   Val  Key  Val            Query                         Query
Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0
Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100
```

**`random_non_queries`** (default `True` in the config class): the filler `0` positions are
replaced with **uniformly random vocab tokens**, making the task harder / more language-like.
⚠️ Important reproduction detail: **all the paper configs set `random_non_queries=False`**
(zeros kept as filler), while the `MQARConfig` class default is `True`. Report which you use.

**Assertions in the generator (hard constraints):** `input_seq_len % 2 == 0`;
`vocab_size > input_seq_len`; `num_kv_pairs*2*num_passes + num_kv_pairs*2 <= input_seq_len`. ✅

**`num_passes`** (added post-paper): repeats the KV block `num_passes` times, implementing the
**JRT "just read twice"** protocol (arXiv:2407.05483). ✅

### 1.3 Scoring metric

Next-token **accuracy on the supervised (query-answer) positions only** — i.e. exact-match
argmax accuracy over the value vocabulary, averaged over all queries in all test examples.
Zoology reports **maximum test accuracy across the learning-rate sweep** for each
architecture × (d_model, N, D) cell. ✅

The repo's `DataSegment.slices` carries `{num_kv_pairs, input_seq_len, num_passes}` so the
trainer can report accuracy **broken down by difficulty cell** — use this; per-cell accuracy is
the informative output, not the aggregate. ✅

⚠️ **Methodological caution:** "max over LR sweep" is a best-of-k statistic and is biased
upward; it is also *architecture-favoring in proportion to sweep size*. For our experiment,
report either (a) max over an identical LR grid for every architecture (Zoology's protocol,
defensible only if the grid is identical), or better (b) mean ± CI over seeds at the per-arch
best LR. See §8.

### 1.4 Experimental grid used in the paper (reproduction recipe)

From `iclr24_zoology_figure2/configs.py` and Appendix E.2: ✅

| Setting | Value |
|---|---|
| Vocab size | **8,192** |
| (seq_len, num_kv_pairs) | **(64, 4), (128, 8), (256, 16), (512, 64)** |
| d_model swept | **64, 128, 256, 512** |
| Layers | **exactly 2** (sequence mixer + MLP each, LayerNorm interleaved) |
| Attention heads | **1** |
| Train / test examples | **100,000 / 3,000** |
| Epochs | **64** |
| LR sweep | `np.logspace(-4, -2, 4)` → 1e-4, ~4.6e-4, ~2.2e-3, 1e-2 |
| Optimizer | AdamW, weight decay 0.1, linear warmup 10 % |
| Batch size | 512 (≤128 len), 256 (256), 128 (512), 64 (1024) — Appendix E.2 states 64/16/8 by len-or-d thresholds ⚠️ discrepancy between text and configs; configs are authoritative |
| Positional embeddings | learned, **only** for attention variants; none for pure conv |
| `state_mixer` | `torch.nn.Identity` in the figure-2 configs (⚠️ so those runs have **no MLP**, contradicting E.2 — configs are authoritative) |
| `power_a` | 0.01 (train and test) |
| `random_non_queries` | **False** in paper configs |

Based paper's harder grid (`arxiv24_based_figure2/configs.py`) — **train/test mismatch is
deliberate**, to test generalization to more KV pairs and longer sequences: ✅

- **Train:** (64, 4), (128, 8), (256, 16), (256, 32), (256, 64) — 100k / 20k examples
- **Test:** (64, 4), (64, 8), (64, 16), (128, 32), (256, 64), (512, 128), (1024, 256) — 1k each
- d_model: attention {64, 128}; Based {48, 64, 128} × feature_dim {8, 16, 24}
- 2 layers, 1 head, `max_position_embeddings=0`, conv mixer = `BaseConv` kernel_size 3
  (`implicit_long_conv=True`) interleaved with the linear-attention mixer via
  `zoology.mixers.hybrid.Hybrid`.

> **Note the hybrid pattern in Based's own configs:** `Hybrid(configs=[conv_mixer, attn_mixer])`
> alternates a **short conv (kernel 3)** with attention/linear-attention across the 2 layers.
> This is architecturally *very close to our mostly-LIV hybrid* and means **zoology is an
> almost drop-in testbed for our architecture** — see §7.4.

**Compute cost.** ✅ Tiny. Each cell is a 2-layer model with d ≤ 512 on ≤ 1024-token sequences,
100k examples × 64 epochs. On one A100/H100 a single run is **minutes**; the full figure-2
sweep is 4 (len,kv) × 4 d_model × 4 LR × 7 mixers = **448 runs**, parallelizable one-per-GPU
via Ray (`-p` flag). Estimate **single-digit GPU-hours** for a focused sweep (one architecture
family, 4 d_model × 4 LR × 4 cells = 64 runs). This is the cheapest high-information eval
available to us. ⚠️ Exact wall-clock not stated in the paper — flagged.

**Usability at < 1B scale.** ✅ **Ideal.** MQAR is *designed* for 2-layer, d ≤ 512 models
trained from scratch on the synthetic. It is not a downstream benchmark and has no floor
problem. Two ways to use it:
- **(a) Standalone probe** — train the *architecture* (not the pretrained LM) on MQAR and
  read off the d_model needed to reach > 0.9 accuracy at each (N, D). This is what gives the
  scaling claim. Use this for architecture screening.
- **(b) In-context probe on the pretrained LM** — format MQAR as tokens in the real
  vocabulary and evaluate zero-shot. ⚠️ Zoology does *not* do this in the main paper; they use
  a "controlled semi-synthetic dataset" for open-source models (Appendix G). Expect near-zero
  for 100M-scale base models unless you few-shot or include a small amount of such data in
  pretraining. Flag as a design decision.

### 1.5 The precise dimension result — **this is the headline theory**

**Claim 1 (Gated-convolutions and attention), verbatim:** ✅

> Gated-convolution models with two layers require model dimension to scale **at least
> linearly in sequence length** in order to solve associative recall, while attention models
> can solve it with **near-constant dimensionality**. … In the top row of Fig. 2, attention
> solves MQAR perfectly at all sequence lengths using a constant model dimension of **64**.
> In contrast, MQAR does not achieve accuracy > 0.9 unless **d ≥ N**.

So the empirical result is **d ≳ N** (model dimension at least the sequence length) for
gated convolutions, versus **d = 64 constant** for attention. Note this is stated in terms of
**sequence length N** (equivalently, number of distinct token-interaction distances), not
directly in terms of D. ⚠️ **Get this right — it is easy to misquote.** The paper is careful:
- **N (sequence length / number of gaps)** drives the requirement for **gated convolutions
  with input-independent filters** (Claim 1, Appendix F item 1).
- **D (number of unique KV pairs)** drives the requirement for **gated recurrences**
  (Appendix F item 2: "gated recurrent models use larger dimensionality than attention to
  solve the task as the number of unique key-value pairs to store (D) increases").

**Formal statements, verbatim:** ✅

- **Proposition 4.3 (Attention).** "Given an input u ∈ {0,1}^{N×3c}, Attention (even without
  using soft-max) solves MQAR for u using **O(c²) parameters, O(Nc² + N²c) time complexity
  and O(1) layers**." → parameters **independent of sequence length**; cost is quadratic in N.

- **Theorem 4.4 (Data-Independent Filters).** "Given an input u ∈ {0,1}^{N×O(log c)} to MQAR
  (where we assume that distinct tokens are embedded into distinct vectors in {0,1}^{O(log c)}),
  there exists a BaseConv operator that solves MQAR for u using **Õ(N log c) parameters** as
  well as time complexity and **Õ(1) layers**." → the *upper bound*: parameters grow with N.

- **Theorem 4.5 (Input-Dependent Filters).** "Given an input u ∈ {0,1}^{N×c} to MQAR (where
  we assume that the tokens are embedded as one-hot encoding in {0,1}^c and there exists at
  most **t distinct interaction distances**), there exists a BaseConv operator that uses
  **input-dependent kernels** to solve the above case of MQAR using **O(t · N c) parameters
  and O(1) layers**." → input-dependence buys **constant depth** (and, empirically, near-constant
  width — Claim 2).

- **Theorem 4.2 (Equivalency to Arithmetic Circuits).** BaseConv can simulate any arithmetic
  circuit of size s and depth Δ with poly-log blowup. This is the machinery that makes
  BaseConv a *canonical* gated convolution: results about BaseConv transfer to H3, Hyena,
  RWKV-v4, RetNet. ✅ **This is why "BaseConv" results apply to our LIV short-conv layers**
  — but see the caveat below.

- **Gated recurrence lower bound (Appendix H.6, referenced at p.27):** "the gated recurrence
  requires **Ω(N) bits** to solve MQAR for d ≤ N for model dimension d." ✅

⚠️ **Caveat on transferring to our architecture.** Theorem 4.4 is an **upper bound**
(there *exists* a solution with Õ(N log c) params); Claim 1 is an **empirical** finding at
2 layers with a specific training protocol. The paper does **not** prove a width lower bound
of Ω(N) for arbitrary-depth gated convolutions. The clean *lower* bounds are (i) the Ω(N)-bit
**state-size** bound for recurrences (Zoology H.6, and Based Thm 3.1 below), and (ii) the
depth lower bounds in Based (Thm 3.2, 3.4). Do not overclaim "gated convs provably need
d = Ω(N)". ✅ verified by reading the theorem statements.

**Claim 2 (Input-dependent filters), verbatim:** ✅

> Using input-dependent filters in gated-convolution models can close some of the gap to
> attention. … BaseConv with **programmatic** input-dependent filters achieves near-constant
> scaling in model dimension and BaseConv with **autocorrelation** input-dependent filters
> achieves improved scaling over BaseConv with input-independent filters.

**Sparse-attention finding (Appendix F item 3) — directly relevant to a mostly-LIV hybrid:** ✅

> **sliding window attention** helps close the MQAR gap when the token-interaction distances
> are relatively **short**, and **degrades on longer distances**. Meanwhile, **blocked window
> attention struggles** to close the gap (intuitively tokens earlier in a block are able to
> interact with few other tokens in the sequence).

And from the intro: inserting an input-dependent operator (a shift-convolution keyed on
bigram positions, or sparse attention placed only on repeated-bigram positions) at
**< 10 % of layers** suffices to beat the Transformer baseline on Pile LM and closes
**> 80 %** of the MQAR perplexity gap; input-dependent sparse attention patterns close
**97.4 %** of the gap while keeping sub-quadratic scaling. ✅

> **This is the strongest prior for our experiment:** a *few* attention layers among many
> gated-conv layers is *predicted* to recover most of the recall gap — Zoology essentially
> ran the ablation at < 10 % of layers. Our contribution is therefore not "does it work" but
> "how few, placed where, and with what width/state budget" — which means the eval must be
> **sensitive enough to resolve the residual 2.6 % of the gap**, not just the 82 %. That
> demands the recall-sliced perplexity (§1.1) plus MQAR at high (N, D), not average ppl.

### 1.6 Recall–throughput tradeoff (Based, arXiv:2402.18668)

**Primary source:** Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou, Rudra, Ré.
*Simple linear attention language models balance the recall-throughput tradeoff.*
arXiv:2402.18668 (ICML 2024). <https://arxiv.org/abs/2402.18668> ·
Code: <https://github.com/HazyResearch/based>

**The tradeoff, stated precisely.** ✅ The controlling variable is **recurrent state size in
bytes during generation** (for attention this is the KV-cache; for linear attention it is
`num_heads × feature_dim × head_dim`; for sliding-window it is window × d). Findings:
- **Within** an architecture, increasing recurrent state size almost always improves recall.
- **Across** architectures, at a *fixed* state size, recall differs substantially — so state
  size is necessary but not sufficient.
- Efficient alternatives (H3, Mamba, RWKV) keep a fixed-size state and "struggle at recall."
- Attention solves recall perfectly but its state grows with sequence length.

**Theorem 3.1, verbatim:** ✅

> "Any recurrent model depending causally on input u ∈ {0,1}^{N×d} requires **Ω(N)-bits** in
> state size to solve MQAR."

with the footnote that "each token from the vocabulary has the natural binary [encoding]".
The authors add: "This result suggests that the tradeoff observed in Figure 2 is
**fundamental, not an artifact of architectural quirks**." ✅

**Theorem 3.2, verbatim:** ✅

> "Given an input sequence u ∈ {0,1}^{3N×d}, where N and d denote the sequence length and head
> dimension, respectively, a **data-independent BaseConv** model needs **log(2d)-layers** to
> solve MQAR for d = log₂(c), where c denotes the vocabulary size."

**Remark 3.3, verbatim:** "For a class of input encodings that generalizes one-hot encodings
termed as p-hot encodings (Definition F.22), **input-dependent BaseConv** needs at least
**⌊log(2p)⌋-layers** to solve MQAR where d = p·√c." ✅

**Combined depth lower bound, verbatim:** "Theorem 3.2 and Theorem 3.4 imply that we need
**Ω(max(log log c, log log N))** many BaseConv layers to solve MQAR" — and Theorem F.30 shows
this is tight: **O(max(log log c, log log N))** layers suffice in certain settings. ✅

Also: linear attention can be simulated by BaseConv with only a poly-log blowup in layers
(Prop. F.8), "pointing to the relative efficiency of linear attention over gated-convolution
architectures." ✅

**Based's architecture and eval protocol (highly reusable for us):** ✅
- BASED = **linear attention (Taylor-exp feature map) + tiny sliding-window softmax attention**
  (window **64 or 128** tokens). Window sizes of 64–128 "recover 90.8 % of full [attention]".
- Trained from scratch at **360M and 1.3B** params on the **Pile**, GPT-2 BPE, **10B / 30B /
  50B tokens**, "each model sees the same tokens of pretraining data in the same order."
- Reported metrics per model: **Pile ppl overall / AR-slice / Other-slice** (the Zoology
  slicing), plus **zero-shot recall-intensive tasks**: **SWDE** and **FDA** (information
  extraction from semi-structured web / FDA documents) and **SQuAD** (QA), plus the
  **LM-Eval-Harness commonsense average** used by Mamba (Gu & Dao).
- Headline: matches Mamba in perplexity, beats sub-quadratic baselines on real recall tasks by
  **6.22** points (abstract) / **10.36** points (intro, at 1.3B/50B) ⚠️ two different numbers
  appear for different aggregations — cite carefully. **24×** higher generation throughput
  than FlashAttention-2 at 1024 generated tokens, 1.3B params, single H100.
- Example Table-1 row: Transformer++ 360M/10B → Pile All 8.39, AR 1.87, Other 9.42,
  SWDE 57.97, FDA 58.00, SQuAD 27.18, LM-Evals 44.08; at 360M/30B → All 7.68, AR 1.80,
  Other 8.40, SWDE 70.75, FDA 63.79, SQuAD 25.07, LM-Evals 44.75. ✅

> **Actionable:** SWDE / FDA / SQuAD-as-extraction at **360M** produce scores in the
> **25–70 %** range — i.e. **well off both floor and ceiling at exactly our scale**. This
> makes them the best-evidenced *real-text* recall evals for a sub-1B comparison. And note
> **LM-Evals barely move** (44.08 → 44.75) while SWDE moves **+12.8** points on the same
> checkpoints: standard commonsense evals are *nearly blind* to recall ability. That is the
> empirical justification for this whole document.

**Set-disjointness / state-size probe.** Based trains on sequences of length **256** with a
range of set sizes and evaluates at length **1,024** — an explicit length/hardness
extrapolation protocol. ✅

### 1.7 "Just read twice" (JRT, arXiv:2407.05483)

**Primary source:** Arora, Timalsina, Singhal, Spector, Eyuboglu, Zhao, Rao, Rudra, Ré.
*Just read twice: closing the recall gap for recurrent language models.* arXiv:2407.05483.
<https://arxiv.org/abs/2407.05483>

**Core theoretical reduction.** ✅ "the hardness of information recall reduces to the hardness
of a problem called **set disjointness (SD)**, a quintessential problem in communication
complexity that requires a streaming algorithm (e.g., recurrent model) to decide whether
inputted sets are disjoint" (Theorem G.11). The consequence they emphasize is **order
dependence**: "the recurrent memory required to solve SD changes with set order, i.e. whether
the smaller set appears first in-context." The one-way communication complexity is
Θ(min(|A|,|B|)) — so a recurrent model can be *much* cheaper if the smaller set comes first,
and this is **information-theoretic, not a training artifact** (Theorem G.15).

**Two fixes.** ✅
1. **JRT-Prompt** — repeat the context multiple times in the prompt ("effectively showing the
   model all data orders"). Gives **+11.0 ± 1.3** points averaged across **16 recurrent LMs**
   and **6 recall-intensive ICL tasks**, with **11.9×** higher prefill throughput than
   FlashAttention-2 (length 32k, batch 16, H100).
2. **JRT-RNN** — non-causal **Prefix Linear Attention (PLA)**: encode the prompt region
   non-causally, decode causally. Reaches **99 %** of Transformer quality at 360M/30B and
   **96 %** at 1.3B/50B, **19.2×** faster prefill than FA2. Also **+13.7 / +6.9** points over
   the recurrent baseline at the two scales.

**Motivating measurement.** ✅ "a **2.8B Mamba** LM trained on **300B tokens** of the Pile
underperforms a **1.3B Transformer** (2.2× smaller) trained on **50B tokens** (6× fewer
tokens) by **5 points**, averaged across a suite of recall-intensive ICL tasks." → *another
demonstration that matched perplexity hides a large recall gap.*

**Actionable protocol for us — the JRT diagnostic pair.** ✅ This yields a beautiful,
nearly-free **causal-order ablation** that isolates *storage* failure from *retrieval*
failure:
- Present the same recall task in **[Document, Question]** vs **[Question, Document]** order.
  A model with adequate state but poor retrieval does fine on both; a state-limited model
  collapses on **[D, Q]** (must remember everything) and recovers on **[Q, D]** (can filter).
- Present the context **once vs twice** (`num_passes=1` vs `2` in `multiquery_ar`, already
  implemented). Large gain from reading twice ⇒ the bottleneck is *what to store*, not
  *capacity to compute*.
- Both are implemented in zoology already; both are Tier-1 cheap.

### 1.8 Summary card — MQAR

| Field | Value |
|---|---|
| **Construction** | Disjoint key/value vocab halves; D unique KV pairs in first 2D positions; each key re-queried once at a power-law-distributed gap; filler = zeros or random tokens |
| **Difficulty knobs** | `input_seq_len` N, `num_kv_pairs` D, `vocab_size` (must be > N), `power_a` α, `random_non_queries`, `num_passes` |
| **Scoring** | Exact-match next-token accuracy at query positions only (labels −100 elsewhere); report per-(N,D) cell |
| **Official impl** | <https://github.com/HazyResearch/zoology> — `zoology/data/multiquery_ar.py` |
| **Compute** | Minutes/run on 1 GPU; ~64 runs for a focused per-arch sweep; single-digit GPU-hours |
| **< 1B usability** | ✅ Excellent as an architecture probe (trained from scratch, 2 layers, d ≤ 512). ⚠️ As a zero-shot probe on a pretrained sub-1B LM, expect floor effects |
| **Key result to cite** | Gated convs need **d ≳ N**; attention solves it at **d = 64** constant (Claim 1). Any causal recurrent model needs **Ω(N) bits** of state (Based Thm 3.1) |

---

## 2. Phonebook, Passkey, and Needle-in-a-Haystack

These are the "does it retrieve at all" family. All three are *lexical, single-hop, low-distractor*
retrieval — which makes them cheap and interpretable, but also **easy to saturate** and easy to
pass for the wrong reasons. Order them by difficulty: NIAH < passkey < phonebook < RULER MK/MV.

### 2.1 Phonebook lookup — **the single best fit for our comparison**

**Origin (verified).** The phonebook lookup task as used in efficient-LM papers comes from
Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State
Space Models at Copying.* arXiv:2402.01032 (ICML 2024). <https://arxiv.org/abs/2402.01032> ✅
(I verified the construction directly in the paper, §4.3.)

**Exact construction, verbatim:** ✅

> "We generate the phone-book by randomly sampling **L** names and their associated phone
> number. One line of this phone-book looks like **"John Powell: 609-323-7777"**. Our prompt
> to the model consists of the phone-book, **two few-shot examples** and a question asking for
> the phone number of a randomly sampled name from the phone-book."

So the full prompt is: `[L lines of "Name: NNN-NNN-NNNN"]` + 2 few-shot Q/A demonstrations +
`"<Name>:"`-style query. Phone numbers are **10 digits in NNN-NNN-NNNN format**. ✅

**Scoring.** **String-level (exact-match) accuracy** on the generated number. From §4.1:
"we use the string-level accuracy in all the experiments except in Figure 7c where we consider
question answering and thus report the F1 score." Evaluation is over **10 batches of size 64
= 640 examples** per configuration. ✅

**Difficulty knob.** **L**, the number of phonebook entries. The x-axis of Figure 1c is
"Number of entries in phone-book." ✅

**The headline result — and why this task is perfect for us:** ✅

> "even the **smallest transformer (410M parameters)** outperforms the **largest GSSMs (2.8B
> parameters)** when the phone-book size is long enough (**L ≥ 70**). This shows that in
> retrieval tasks which require access to the whole context, GSSMs struggle to store the
> relevant information in their fixed-size state."

Models compared: **Pythia** (transformer) vs **Mamba** (GSSM), both trained on **the Pile with
the same tokenizer**, at matched sizes — and Mamba had *slightly lower training perplexity* at
each size. This is exactly the "matched perplexity, unmatched retrieval" design we need. ✅

> **This is the cleanest existence proof that the effect we care about is detectable at
> 410M params with a ~70-entry phonebook (i.e. ~1-2k tokens of context).** No long-context
> training required. **Tier 1.**

**Copying theorems from the same paper (the theory behind phonebook):** ✅
- **Theorem 2.3.** "For all n, there exists a depth-2 transformer T of dimension
  **O(n log D)** s.t. for all 2n ≤ L ≤ D^n, and for any copy distribution D_L,
  err(T) < p_{n-gram}(D_L)." (D = vocab size, L = sequence length; p_{n-gram} = probability of
  a repeated n-gram.) The mechanism is an **n-gram-based induction head**.
- **Lemma 2.4.** For the uniform copy distribution, p_{n-gram}(D_L) < L² D^{−n} — decays
  exponentially in n.
- **Corollary 2.5.** A depth-2 transformer of dimension **O(log(L/ε) log D)** copies with
  error < ε — i.e. **parameters only logarithmic in sequence length**.
- **Theorem 2.7 (GSSM lower bound).** "Fix some GSSM H over state space S. Then, for all L,
  for the uniform copy distribution D_L, the model H has error err(H) > **1 − |S| / D^L**."
- **Corollary 2.8.** "every GSSM H with state space S s.t. **mem(S) < L log(D) − 1** has error
  > 1/2 for the uniform copy distribution." → **you cannot copy more than your state holds.**
- **Remark 2.9.** Transformers are near-optimal in *input-dependent* memory for copying:
  Õ(L) bits, matching the lower bound.

**Usage in other hybrid-architecture reports — CORRECTION to a common assumption.** ✅
A targeted full-text search of Jamba, Hymba, Zamba, Zamba2, and MEGALODON for
"phonebook"/"phone book"/"phone-book" found **none of them use the phonebook task**. The
correct picture:

| Report | Retrieval eval actually used |
|---|---|
| **RecurrentGemma / Griffin** (2402.19427) | ✅ **Uses phonebook, explicitly citing Jelassi et al.**: "a phone number lookup task designed to test both copying and retrieval capabilities… a synthetic phonebook containing names and numbers." 7B Hawk / 7B Griffin / 6B MQA-Transformer, 300B tokens. **Result: Hawk (pure recurrent) fails as phonebook length grows; MQA Transformer solves up to training length then breaks; Griffin solves perfectly up to its 1024-token local-attention window and extrapolates somewhat beyond.** ⚠️ figure-only, no numeric scores. |
| **Jamba** (2403.19887) | Kamradt-style **needle-in-a-haystack** (qualitative heatmap to 256K). Jamba-1.5 (2408.12570) reports **RULER** and is 1st on NVIDIA's leaderboard (96.0 avg). |
| **Hymba** (2411.13676) | Generic **needle-in-a-haystack** (1B scale; pretrain 1k / finetune 4k / test to 16k). |
| **Zamba** (2405.16712) | ❌ No needle/passkey/phonebook at all — only PIQA/ARC/MMLU/GSM8k/HumanEval. |
| **Zamba2** (2411.15242) | **Passkey retrieval.** Zamba2-7B: NTK-aware RoPE rescaling → ~17,000 tokens without retraining. Zamba2-2.7B (no positional embeddings) fine-tuned with a doubling curriculum (4096→65536, doubling every 100 steps) → accurate passkey retrieval to **65,536** tokens. |
| **MEGALODON** (2404.08801) | ❌ None; long-context eval is PPL on 2M-token book concatenations + SCROLLS QA. |
| **Liquid AI LFM / LFM2** | **RULER, not phonebook.** The original LFM blog reports RULER with the **85.6 % effective-context threshold** (e.g. **LFM-3B: 94.4 / 93.5 / 91.8 / 89.5 at 4k / 8k / 16k / 32k**). ⚠️ LFM2 blog + HF card mention only "50+ internal evaluations"; the **LFM2 Technical Report (arXiv:2511.23404)** shows no phonebook/needle/passkey in accessible sections, but **§6–9 and Appendix C ("Evaluation Details") could not be retrieved** — ❓ **flagged as unresolved.** |

> **Griffin's phonebook result is the single most relevant published datapoint for our
> architecture.** Griffin = recurrent blocks + *local* (sliding-window) attention, and it
> **solves phonebook perfectly up to the window size and no further**. Our mostly-LIV hybrid
> replaces the window with a few *full* GQA layers, which should remove that ceiling. **The
> phonebook length sweep is therefore the direct test of whether our few full-attention layers
> buy unbounded-range retrieval, and it is exactly the axis on which Griffin's design fails.**
> Make L span well past any local receptive field.

**Implementation.** ❓ No canonical pip package located. The task is ~30 lines of Python
(sample L names from a name list, sample 10-digit numbers, format, add 2 few-shot examples,
generate, exact-match). Recommend writing it ourselves — it is the cheapest high-signal eval in
this document. **Compute: trivial** (640 generations of ~12 tokens each per L value).

**Usability at < 1B.** ✅ **Excellent, and directly evidenced** — Jelassi et al. run it at
**410M** and the effect is already large there. Use L ∈ {5, 10, 20, 40, 70, 100, 200} to
produce a *degradation curve*, not a single number.

### 2.2 Passkey retrieval (Landmark Attention)

**Primary source:** Mohtashami & Jaggi. *Landmark Attention: Random-Access Infinite Context
Length for Transformers.* arXiv:2305.16300 (NeurIPS 2023). <https://arxiv.org/abs/2305.16300>

**Exact prompt template (verbatim from Figure 3a):** ✅

```
There is an important info hidden inside a lot of irrelevant text. Find it and
memorize them. I will quiz you about the important information there.

<prefix filler by continuously repeating:
The grass is green. The sky is blue. The sun is yellow. Here we go. There
and back again.>

The pass key is <PASS KEY>. Remember it.
<PASS KEY> is the pass key.

<suffix filler>

What is the pass key? The pass key is
```

**Exact parameters (verbatim from the Figure 3 caption):** ✅
- The pass key is "**a random number between 1 and 50000**".
- "Results are averaged over **50 random generation** of the pass key, which each time is
  located at a **random position** in the full-length prompt."
- "The space before and after the pass key is filled accordingly by the **suffix and prefix
  filler**" — i.e. **total prompt length is held fixed** and the depth is randomized by
  splitting filler between prefix and suffix. This is the correct way to decouple length
  from depth.
- **Scoring:** accuracy of "generating the correct pass key **as the first integer within
  the first 100 generated tokens**." ✅ (Note: this is a *lenient* extraction rule — it does
  not require the model to produce clean formatting. Good for base models.)
- Inference detail in that paper: segments of length 250 tokens, top-k = 4 landmark blocks.

**The filler sentence** "The grass is green. The sky is blue. The sun is yellow. Here we go.
There and back again." is the canonical passkey noise text, and **RULER explicitly reuses it**
(RULER footnote 3: "Following Mohtashami & Jaggi (2023), we use 'The grass is green. The sky is
blue. The sun is yellow. Here we go. There and back again.' as noise sentences."). ✅

**Official implementation.** <https://github.com/epfml/landmark-attention> (Apache-2.0). ⚠️ The
exact passkey-eval script is not named in the README; best candidates are `llama/run_test.py`
and `lm_benchmark/eval_cmd_generator.py`. **Flagged.**

**Reuse — verified in detail.** ✅
- **LongLoRA** (2309.12307): reuses the **identical** template verbatim (same instruction, same
  filler repeated M times before / N times after, key sentence e.g. "The pass key is 12362.
  Remember it. 12362 is the pass key.", same query). **5-digit** passkey, resampled per test,
  **10 trials per length**, sweep ~1k–34k. Baseline Llama2-7B degrades sharply past 4k.
- **YaRN** (2309.00071): "The passkey retrieval task as defined in Mohtashami and Jaggi (2023)."
  **5-digit** passkey; 50 iterations for 32k ablations, 10 iterations for 64k/128k.
  **Important scoring convention:** they define the "passkey context" as the largest tested
  window where accuracy stays **≥ 80 %**, and "passkey accuracy" as the average over all sizes
  at/below it. ✅ Also a directly relevant methodological observation from that paper: Code
  Llama 13B reaches **99.4 %** passkey at 128k **even though its perplexity degrades past
  100k**, leading the authors to argue "**perplexity may not be a great indicator**" of
  retrieval ability. *That is independent support for this entire document's premise.*
- Landmark itself is the ancestor of RULER's `type_haystack: noise` filler. ✅
- ❌ **CORRECTION: Mamba (2312.00752) does NOT use passkey, needle, or phonebook.** Its
  long-context evidence is **induction heads** (train @ 256, extrapolate to 2²⁰), **selective
  copying**, Great-Apes DNA classification, and DNA/audio perplexity scaling. ✅ Do not cite
  Mamba as a passkey source.
- **Zamba2** does use passkey (see the phonebook table above). ✅

**Compute.** Negligible: 50 generations × (a handful of lengths). Seconds to minutes.

**Usability at < 1B.** ⚠️ **Use with care.** Two problems: (1) the filler is *extremely*
repetitive and low-entropy, so the passkey sentence is trivially out-of-distribution — a model
can succeed by "find the only surprising token," which is not the retrieval skill we care
about; (2) LLaMA-7B already succeeds up to ~2k. At 100–350M with 2–4k context, expect either
near-ceiling at short lengths or a sharp cliff, with little resolution in between. **Keep it
as a Tier-1 sanity check / regression test, not as a discriminator.**

**Variance note.** With only 50 trials, the binomial standard error at p = 0.5 is
√(0.25/50) = **7.1 points**. That is far too noisy for architecture comparison. **Raise to
≥ 500 trials** (SE ≈ 2.2 pts) if you intend to compare on it. ✅ (own calculation)

### 2.3 Needle-in-a-Haystack (Kamradt)

**Source.** Greg Kamradt, *LLMTest_NeedleInAHaystack* (2023) — a repo/blog artifact, **not a
paper**. <https://github.com/gkamradt/LLMTest_NeedleInAHaystack> · pip: `needlehaystack`

**License:** MIT (attribution requested). **pip:** `pip install needlehaystack`
(CLI: `niah run/validate/reconstruct/demo`).

**Construction — v1 (recovered from commit `73ffdd4`, 2023-11-28; the current README is a v2
rewrite that no longer states these numbers).** ✅ All strings verified from source:
- **Needle (exact):** `"\nThe best thing to do in San Francisco is eat a sandwich and sit in
  Dolores Park on a sunny day.\n"`
- **Retrieval question (exact):** `"What is the most fun thing to do in San Francisco?"`
- **Haystack:** the repo's `PaulGrahamEssays/` directory (~48 `.txt` files), concatenated
  repeatedly until the target token count is reached (`read_context_files`), then trimmed and
  spliced (`insert_needle`): reserve `final_context_length_buffer` (**default 200 tokens**) for
  system/question/reply → `insertion_point = len(tokens_context) * (depth_percent/100)` → walk
  **backward to the nearest sentence boundary** (a `'.'` token) → splice (or append at the very
  end if `depth_percent == 100`).
- **Default grid:** `context_lengths` 1,000 → 200,000 in **35** linear intervals ×
  `document_depth_percent` 0 → 100 in **35** intervals (linear or sigmoid) = **1,225 cells**,
  each costing 1 generation + 1 judge call.
- **Scoring (exact rubric from `evaluate_response`):** LangChain
  `load_evaluator("labeled_score_string", …)` with judge `ChatOpenAI(model="gpt-4",
  temperature=0)`, comparing `prediction=response` against `reference=needle`:
  **1** = "completely unrelated to the reference"; **3** = "minor relevance but does not align";
  **5** = "moderate relevance but contains inaccuracies"; **7** = "aligns with the reference but
  has minor omissions"; **10** = "completely accurate and aligns perfectly." (Judge instructed
  to "Only respond with a numberical score" — typo in source.)

**v2 (current main).** Now a framework: `niah` CLI + YAML configs. Built-in tasks: `single`
(**exact-match — no judge**), `multi` (fractional), `uuid`, `uuid_chain` (returns e.g.
`{"hops_correct": 3, "chain_length": 5}` → 0.6). Multi-needle spacing:
`depth_percent_interval = (100 − depth_percent) / N`. v2 explicitly fixed a **v1 bug where each
needle's reported depth was off by however much earlier needles had inflated the token count** —
⚠️ so **published v1 multi-needle depth numbers are unreliable**. Providers: OpenAI, Anthropic,
Cohere, FakeProvider. `niah demo --fake` runs offline in ~1 s. ✅

> **For our use: set task=`single` (exact match) and skip the judge entirely.** `niah demo`
> with gpt-4o-mini costs ~$0.01; a v1-style 1,225-cell GPT-4-judged sweep costs orders more.

**Criticisms — this is the important part.** ✅
1. **RULER (2404.06654)** is a direct response. Verbatim from its conclusion: "**Despite
   achieving perfect results in the widely used needle-in-a-haystack test, almost all models
   fail to maintain their performance in other tasks of RULER as we increase input length.**"
   RULER's three criteria implicitly enumerate NIAH's defects: retrieval must be
   "(1) agnostic to the type of the 'needle' and the 'haystack', (2) strong enough to disregard
   **hard distractors**, and (3) of **high recall when multiple items** need to be retrieved."
   Vanilla NIAH tests none of these. ✅
2. **Out-of-distribution needle.** RULER deliberately swaps Kamradt's sentence for
   "the special magic number for XXX is: YYY" "due to its extendability" (footnote 2) —
   acknowledging the needle/haystack distributional mismatch makes detection trivial. ✅
3. **BABILong (2406.10149)** notes the same: "The brevity and similarity of the task sentences
   also enable the model [to] distinguish them from seemingly close background text." ✅
4. **NoLiMa (arXiv:2502.05167)** ✅ verified. Code:
   <https://github.com/adobe-research/NoLiMa> (**Adobe Research License, non-commercial**).
   Argument: models "can exploit existing **literal matches** between the needle and haystack to
   simplify the task." They build a needle set where "questions and needles have **minimal
   lexical overlap**," forcing **latent associations**. Result: of **13 LLMs claiming ≥128K
   context, 11 fall under half their ≤1K baseline at 32K**; **GPT-4o drops 99.3 % → 69.7 %**.
   Reasoning models/CoT do not rescue it. ⚠️ Smallest model tested is **Gemma-3-4B** (already
   collapses); **no sub-1B baseline published.**
5. **HELMET (2410.02694)** ✅ verified: synthetic NIAH-style tasks "do not reliably predict
   downstream performance"; NIAH is **saturated** ("most models max out NIAH") yet open-source
   models "significantly lag behind closed ones" on full-context reasoning/instruction-following
   at the same lengths — a gap NIAH cannot see.
6. **Judge noise + cost.** Scoring is a paid GPT-4 call per cell; the 1–10 rubric is
   unreliable at the boundary; results are not reproducible across judge versions. ✅
7. **Counting-Stars (2403.11802)** ✅ verified. Code:
   <https://github.com/nick7nlp/Counting-Stars>. Evidence "stars" at **uniform intervals**
   (not one needle at one depth). Templates: `"The little penguin counted {number1} ★"` (search,
   CS-S) and a reasoning variant (CS-R) where a wrong count is stated then corrected.
   **M = 32 stars** used throughout (scalable to 1024). Haystacks: *The Story of the Stone*
   (ZH) / Paul Graham essays (EN). **Metric: P@N** — search P@32 (1 if `{number1}` present);
   reasoning P@32★ (1 correct-only, 0.5 both, 0.25 wrong-only, 0 neither); plus F1@32/F1@M.
   4K–128K in 4K steps, 32 samples/run. **The authors themselves concede the metric "seems too
   simple."**
8. **Ada-LEval (2404.06480, NAACL 2024)** ✅ exists; code
   <https://github.com/open-compass/Ada-LEval>. Critiques L-Eval/LongBench for entangling
   "samples of varying lengths (from 2k to 32k+)" and omitting 100k+. Introduces
   length-adaptable **TSort** and **BestAnswer**. ⚠️ Internal task mechanics not recoverable
   from the abstract page. **Flagged.**

**Multi-needle variants.** ✅ RULER's **MK-NIAH / MV-NIAH / MQ-NIAH** are the rigorous version
and supersede the LangChain multi-needle blog experiments. Notably **RULER's MQ-NIAH is
explicitly MQAR**: verbatim, "This is the same **multi-query associative recall** task setup
used by Arora et al. (2024)." ✅ *That single sentence links our entire eval stack: MQAR at
2-layer synthetic scale ≡ RULER MQ-NIAH at LM scale.*

Additional verified variants:
- **LangChain multi-needle** (Gola, 2024-03-13,
  <https://www.langchain.com/blog/multi-needle-in-a-haystack>): pizza-ingredient needles
  (`" Figs are one of the secret ingredients needed to build the perfect pizza. "`), needle
  counts **{1, 3, 10}**, 6 length intervals ~1k→120k, 3 replicates, GPT-4 only. **Findings
  highly relevant to us:** accuracy falls monotonically as needle count rises 1→10 and as
  context grows; GPT-4 "consistently retrieves needles towards the end **while ignoring needles
  at the beginning**" (10-needle/24.8k trace returned only the last four); front-of-document
  failure begins **~25K with multiple needles vs ~73K single-needle**; and "**retrieval may set
  an upper bound on reasoning performance.**" ✅ *The recency bias here is the same failure mode
  a fixed-state recurrent model shows — worth distinguishing carefully in our analysis.*
- **Needle Threading (2411.05000, ICLR 2025)** ✅ verified; <https://needle-threading.github.io/>.
  Haystack = string-serialized JSON of random **32-char UUID** key–value pairs. **Seven
  variants:** Single Needle; Multiple Needles (2–25 keys, random or clustered); Conditional
  Needles (retrieve all values whose key matches a pattern); **Threading** (chain built by
  `K_{j_k} ← V_{j_{k-1}}`; given only the first key, return the final value after n hops);
  Multi-Threading; Branched Threading. 12 haystack sizes **1.2k–630k**; needle counts
  {1,2,3,4,5,10,15,20,25}; thread lengths to 25. Scoring: greedy decode → LLM-normalized parse
  → **exact match**; "effective context length" = the **75 %-accuracy contour, reported in
  characters** (deliberately tokenizer-independent — "token counts from different tokenizers
  should not be directly compared"). Finding: many models are "remarkably **threadsafe**" up to
  25 concurrent threads, but effective context is far below advertised. ⚠️ Cost warning from the
  authors: "a single task run on the priciest models could cost **hundreds of dollars**."
  **The Threading task is an excellent cheap design template for a multi-hop probe at our
  scale — it is pure UUID chasing with exact-match scoring and no natural-language competence
  required.**

**Compute.** Cheap in GPU terms (a few hundred long-context generations) but **has a real
dollar cost** if using the GPT-4 judge. At < 1B scale with short needles, you can replace the
judge with **exact substring match** on the magic number — do this; it removes cost and noise.

**Usability at < 1B.** ⚠️ Only in the RULER-style "magic number" form with substring scoring,
and only at lengths ≤ the model's training context. Vanilla Kamradt NIAH with a GPT-4 judge is
**not** appropriate for sub-1B base models (they will not produce judgeable prose).

### 2.4 Summary card — lexical retrieval family

| Task | Construction knob | Metric | Official impl | Cost | < 1B usability |
|---|---|---|---|---|---|
| **Phonebook** | L entries | exact-match string acc, 640 ex. | ❓ none found; ~30 LOC | trivial | ✅ **best**; effect verified at 410M, L≥70 |
| **Passkey** | total length, random depth | first integer in 100 tokens | Landmark repo | trivial | ⚠️ sanity check only; 50 trials too noisy → use ≥500 |
| **NIAH (Kamradt)** | length × depth grid | GPT-4 judge 1–10 | `needlehaystack` pip | $ judge cost | ⚠️ replace judge w/ substring; superseded by RULER |
| **RULER MQ/MV/MK-NIAH** | #keys/#values/#queries | `string_match_all` | NVIDIA/RULER | see §3.1 | ⚠️ needs short-length reconfig |

---

## 3. RULER and the long-context benchmark landscape

### 3.1 RULER (arXiv:2404.06654, COLM 2024)

**Primary source:** Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg (NVIDIA).
*RULER: What's the Real Context Size of Your Long-Context Language Models?*
<https://arxiv.org/abs/2404.06654> · Repo: <https://github.com/NVIDIA/RULER>
(**Apache-2.0** — verified from the license header in `scripts/synthetic.yaml`) ✅

**Four categories, 13 tasks.** ✅ Verbatim: "RULER comprises tasks across four categories:
**retrieval, multi-hop tracing, aggregation, and question answering**." The 13 tasks and their
**exact config args** (verified from `scripts/synthetic.yaml` in the repo) are:

| # | Task | `task` | Exact args |
|---|---|---|---|
| 1 | `niah_single_1` | niah | haystack=**noise**, key=words, value=numbers, k=1,v=1,q=1 |
| 2 | `niah_single_2` | niah | haystack=**essay**, key=words, value=numbers, 1/1/1 |
| 3 | `niah_single_3` | niah | haystack=essay, key=words, value=**uuids**, 1/1/1 |
| 4 | `niah_multikey_1` | niah | haystack=essay, **num_needle_k=4**, v=1, q=1 |
| 5 | `niah_multikey_2` | niah | haystack=**needle** (haystack *is* distractor needles), k=1,v=1,q=1 |
| 6 | `niah_multikey_3` | niah | haystack=**needle**, key=**uuids**, value=**uuids** |
| 7 | `niah_multivalue` | niah | haystack=essay, k=1, **num_needle_v=4**, q=1 |
| 8 | `niah_multiquery` | niah | haystack=essay, k=1, v=1, **num_needle_q=4** ← **= MQAR** |
| 9 | `vt` | variable_tracking | haystack=noise, **num_chains=1, num_hops=4** |
| 10 | `cwe` | common_words_extraction | **freq_cw=30, freq_ucw=3, num_cw=10** |
| 11 | `fwe` | freq_words_extraction | **alpha=2.0** (Zeta) |
| 12 | `qa_1` | qa | dataset=**squad** |
| 13 | `qa_2` | qa | dataset=**hotpotqa** |

**Needle template.** ✅ "the **special magic number for XXX is: YYY**" (RULER's replacement for
Kamradt's SF sentence). Values are **numbers (7 digits)** or **UUIDs (32 digits)**. Haystack is
either **repeated noise sentences** (the Landmark filler) or **Paul Graham essays** (downloaded
from Kamradt's repo).

**Variable Tracking (VT) construction, verbatim:** ✅ "a variable X1 is initialized with a
value V, followed by a **linear chain of variable name binding statements** (e.g., X2 = X1,
X3 = X2, ...), which are inserted at various positions of the input. The objective is to
**return all variable names pointing to the same value V**." Difficulty = more hops or more
chains. Example from Table 2 (2 chains, 2 hops): `VAR X1 = 12345 ... VAR Y1 = 54321 ...
VAR X2 = X1 ... VAR Y2 = Y1 ... VAR X3 = X2 ... VAR Y3 = Y2 ...` → *"Find all variables that are
assigned the value 12345." → "X1 X2 X3"*. The shipped config uses **1 chain, 4 hops**.

> **VT is the closest thing in RULER to an entity/state-tracking probe** — it is a coreference
> chain. But note it is a *chain of aliases*, not a mutable state, so it does **not** test the
> NC¹-hard state tracking of §4. It's a multi-hop *retrieval* task.

**Aggregation (CWE/FWE) construction, verbatim:** ✅ "In the **common word extraction** task
(CWE), words are sampled from **discrete uniform distributions**, with the number of common
words fixed while the number of uncommon words **increases with the sequence length**. In the
**frequent words extraction** task (FWE), words are sampled from **Zeta distribution**... A
model needs to return the **top-K frequent words**. In CWE, K equals to the number of common
words. In FWE, we set **K to 3**, as increasing K leads to poor performance even at small
context sizes for most models." The Zeta law (footnote 4): frequency of the k-th ranked word is
**k^{−α} / ζ(α)**, with the **top-ranked word set to noise**. ✅

> **FWE's K=3 admission is a warning shot: "increasing K leads to poor performance even at small
> context sizes for most models."** Aggregation tasks floor easily. Expect CWE/FWE ≈ 0 for a
> sub-1B model.

**QA construction, verbatim:** ✅ "we insert the **golden paragraphs** (i.e., the paragraphs
that contain answers) into paragraphs **randomly sampled from the same dataset**... the question
serves as the query, the golden paragraphs are the 'needles', and the distracting paragraphs
form the 'haystack'."

**Scoring — exact functions from `scripts/eval/synthetic/constants.py`:** ✅

```python
def string_match_part(preds, refs):   # used for QA
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
                 for pred, ref in zip(preds, refs)]) / len(preds) * 100

def string_match_all(preds, refs):    # used for niah, VT, CWE, FWE
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
                 for pred, ref in zip(preds, refs)]) / len(preds) * 100
```

i.e. **recall-based substring matching, case-insensitive** — `string_match_all` is *partial
credit* (fraction of required references present), `string_match_part` is *any-match*. ✅ No LLM
judge anywhere. This is a major practical advantage over NIAH. From the paper: "we append the
task input with an **answer prefix** and check the presence of the target output with
**recall-based accuracy**" — done specifically "to prevent the model from refusing to answer a
query or generating explanations." ✅ **That design choice makes RULER unusually base-model-friendly.**

**Effective context length — exact definition:** ✅ Verbatim: "To determine the maximum context
size a model can effectively handle, we grade each model with a fixed threshold... **We use the
performance of Llama2-7b model at the 4K context length as the threshold.** We report... the
**maximum length exceeding the threshold** as the 'effective length'."

**The threshold number is 85.6.** ✅ Verified twice: the paper's Table 3 row reads
`Llama2 (7B) | 4K | - | 85.6`, and the repo README states: "only half of them can effectively
handle sequence length of 32K by exceeding a qualitative threshold, **Llama-2-7b performance at
4K (85.6 %)**." Each score is the **average accuracy over the 13 tasks**.
⚠️ **Note the ambiguity:** 85.6 is the **chat** model. Appendix reports both
`Llama2-7B (base)` and `Llama2-7B (chat)` thresholds per-task (e.g. 79.4 base vs 96.9 chat in
one table) — so "85.6" is a *specific* aggregate. If we invent our own threshold, say so
explicitly; do not claim "effective context length" without the Llama2 reference point.

**Ranking metrics.** `wAvg.(inc)` and `wAvg.(dec)`: weighted averages across context sizes with
weight increasing/decreasing linearly with length. ✅

**Run configuration.** ✅ **500 examples per task per length** (`NUM_SAMPLES=500` in
`scripts/config_tasks.sh`), lengths **{4K, 8K, 16K, 32K, 64K, 128K}**, greedy decoding,
BFloat16, **8× A100**, served via **vLLM**. 17 models evaluated.

**Compute cost.** ✅ 13 tasks × 500 examples × 6 lengths = **39,000 generations** per model, at
an average context of ~42K tokens ⇒ order **1.6 billion prefill tokens** per model. That is
**substantial** — the paper used 8×A100 with vLLM. **For our purposes, reconfigured to
{1K, 2K, 4K} with 200 examples this drops to 13 × 200 × 3 = 7,800 generations at ≤4K, which is
minutes-to-an-hour on one GPU.** ✅ (own estimate)

**Can it be reconfigured to short lengths?** ✅ **Yes** — `SEQ_LENGTHS` is a plain bash array in
`scripts/run.sh` and data is generated on the fly given a tokenizer, so 1K/2K/4K work. But:
⚠️ several tasks have length-coupled construction (`num_ucw ∝ context length`,
`size_haystack ∝ context length`, `num_document ∝ context length`), so at 1K the tasks become
*qualitatively different and much easier* — CWE with few uncommon words, QA with 1-2 documents.
**Any sub-4K RULER numbers are non-comparable to published RULER and must be labeled
"RULER-short (reconfigured), not comparable to the leaderboard."** Also: does the model need
vLLM/TRT-LLM? The pipeline supports HF models but the deprecated path expects
`MODEL_DIR` HF folders + `scripts/data/template.py` chat templates; newer `rulerv1-ns` /
`rulerv2-ns` branches are the maintained pipelines. ✅

**RULER at sub-1B scale — what the evidence says.** This is the decisive question and the
answer is **mostly discouraging**:

1. **Smallest model on the official leaderboard: EXAONE-4.0-1.2B (1.2B)** — 4K: **87.0**,
   8K: 86.7, 16K: 88.8, 32K: 81.1, 64K: 77.4, Avg 84.2, effective length 32K. ✅ (from the repo
   README, results reported by the EXAONE authors, arXiv:2507.11407.) So a **heavily
   instruction-tuned 1.2B model clears the 85.6 threshold at 4K** — barely. Everything else on
   the leaderboard is ≥ 3.8B.
2. **Non-Transformer architectures fail early.** Verbatim from RULER §6: "We evaluate the
   effective context length for two models with non-Transformer architectures: **RWKV-v5** and
   **Mamba-2.8B-slimpj**. We find that **both models demonstrate significant degradation when
   extending context size to 8K**, and both **underperform the Transformer baseline Llama2-7B by
   large margins up till the length of 4K**." ✅ **A 2.8B Mamba is already below the floor at
   4K.** A 350M recurrent-hybrid will be at/near zero on most RULER tasks.
3. **Model size matters a lot.** Verbatim: "the **34B model is significantly better than the
   6B model** on RULER for both performance at length of 4K and the relative degradation,
   suggesting the benefit of scaling model sizes for better long-context modeling." ✅
4. **BABILong's direct comparison (2406.10149) is the killer datum:** verbatim — "**BABILong can
   detect differences in models behavior starting from lengths as small as 2K tokens, while
   RULER requires lengths of at least 128K tokens to show significant differentiation** from
   [the] relatively short MMLU benchmark." ✅ Their measured MMLU correlations:
   BABILong-0K vs MMLU = **0.928**; RULER(≤128K) vs MMLU = **0.435**; RULER(64K) = **0.455**.

❓ **Unknown / flagged:** I did not find published RULER scores for Llama-3.2-1B,
Qwen2.5-0.5B, SmolLM2, RWKV-7, Falcon-Mamba, LFM2-350M/700M, Zamba2-1.2B, or Hymba-1.5B. The
official leaderboard's smallest entry is EXAONE-4.0-1.2B. **If you need a sub-1B RULER
reference point, you will likely have to produce it yourself.**

> **VERDICT on RULER for our experiment.** ❌ **Do not make RULER a primary metric.** At
> 100M–1B with 2–8K training context, expect **floor effects on 9 of 13 tasks** (all multikey
> variants, VT with 4 hops, CWE, FWE, both QA tasks) and the aggregate will be dominated by
> noise near zero. Two legitimate uses: **(a) Tier-3, one run at the largest scale, reporting
> only the sub-tasks that are off the floor** (likely `niah_single_1/2`, and `niah_multiquery`
> at low q); **(b) steal the task *generators*** — `niah_multiquery` is literally MQAR and
> `vt` is a nice multi-hop probe — and run them at 1–4K with our own difficulty ladder.
> Report as "RULER-derived tasks," not "RULER."

### 3.2 The rest of the landscape — viability at small scale

| Benchmark | What it is | < 1B verdict |
|---|---|---|
| **BABILong** (2406.10149) | bAbI QA1–QA20 facts hidden in **PG19** book text; lengths **0K → 10M**; `booydar/babilong` + HF `RMT-team/babilong`; leaderboard on HF Spaces | ✅ **BEST long-context option at small scale.** Has a **0K** setting (no distractor text) and discriminates from **2K**. Crucially: "fine-tuning of small scale models (**ARMT and RMT, 137M** and **Mamba, 130M**) shows that the tasks are solvable" — **explicit sub-200M results exist.** |
| **HELMET** (2410.02694) | 7 categories incl. synthetic recall, RAG, passage re-ranking, long-doc QA, many-shot ICL, summarization, citation | ⚠️ Designed for frontier models; argues NIAH is insufficient; ❓ **I did not verify its base-model claim this pass — flagged.** Expect floor at < 1B. |
| **LongBench / LongBench-v2** (2308.14508 / 2412.15204) | Realistic bilingual (EN/ZH) tasks; v2 is **4-way multiple choice** | ❌ v2 chance = 25 %; a sub-1B base model will sit at chance. ❓ exact floors unverified. |
| **∞Bench / InfiniteBench** (2402.13718) | Avg length **>100K tokens** | ❌ Hopeless at < 1B. |
| **LOFT** (2406.13121) | Retrieval/RAG/SQL/many-shot at 32K–1M corpus | ❌ Hopeless at < 1B. |
| **Michelangelo** (2409.12640) | Latent Structure Queries: **Latent List**, **MRCR**, **IDK** | ⚠️ Latent List is a Python-list-operations task — *in principle* small-scale-adaptable, but requires code competence a 350M model won't have. MRCR requires multi-turn coreference over long dialogues. ❌ at our scale. ❓ constructions unverified this pass. |
| **ZeroSCROLLS / SCROLLS / L-Eval / Ada-LEval / LV-Eval / NoLiMa / LongProc** | Realistic long-doc suites (RULER §2 cites these) | ❌ All require generative competence beyond sub-1B base models. NoLiMa is interesting *methodologically* (latent needles) but ❓ unverified. |

**Recommendation for §3.** Use **BABILong at 0K/1K/2K/4K on QA1–QA5** as the long-context
benchmark (Tier 2), **RULER-derived generators at 1–4K** as controllable synthetics (Tier 1–2),
and **skip everything else** at this scale. Reserve full RULER for a Tier-3 single run if the
largest model reaches ≥ 1B with ≥ 8K context.

---

## 4. Entity tracking, state tracking, and the expressivity theory

This section supplies the **principled reason to expect specific failures**. The short version:

> **Retrieval capacity and sequential state tracking are two different limitations with two
> different fixes.** A few full-attention layers provably fix the *retrieval* limitation
> (Wen et al. Thm 5.6 — literally "adding one Transformer layer"). They provably do **not** fix
> the *sequential state tracking* limitation, because attention is itself in TC⁰
> (Merrill & Sabharwal) and so are SSMs (Merrill et al. 2404.08819). Design the eval suite so
> these two axes are measured separately, or you will misattribute a failure.

### 4.1 The boxes task — Entity Tracking in Language Models

**Primary source:** Kim & Schuster. *Entity Tracking in Language Models.* ACL 2023,
arXiv:2305.02363. <https://arxiv.org/abs/2305.02363> · Code/data:
<https://github.com/sebschu/entity-tracking-lms> ⚠️ (repo URL stated in the paper as "Code and
data are available at https://github.com/…" — line truncated in my extraction; **verify the
exact org/repo before citing**).

**Exact construction (verbatim example from the paper's abstract figure):** ✅

> **Q:** Box 1 contains the book. Box 2 contains the apple. Box 4 contains the brain. Move the
> book into Box 2. Put the bell into Box 4. Move the bell and the brain into Box 5.
> Box 2 contains ____
> **A:** the apple and the book

**Parameters, verbatim from §3.3:** ✅

> "Given a natural language description of the initial state of the world followed by
> **0–12 state-changing operations**, the content of each box at the end of the description must
> be correctly identified. To evaluate this, we created a test example for **each box after each
> operation**. This corresponds to **n × (NumOps + 1)** examples per scenario (**91 exs. in our
> datasets**). Each example is formulated in the style of a **cloze test**… ending in
> **Box N contains __**."

- **n** = number of boxes, **m** = max objects per box (**m = 3**). 91 examples per scenario ⇒
  n = 7, NumOps = 12 (7 × 13 = 91). ✅
- Operations: place in / take out of / **move from one box to another**; plus a harder variant
  with "**Move contents of Box N to Box M**" which "does not explicitly mention" the objects —
  requiring the model to already know the box state. ✅ *This is the genuinely stateful
  operation.*
- **Prompt structure:** (a) general instruction, (b) **two** format demonstrations, (c) initial
  state + operations, (d) the incomplete `Box N contains ___`. ✅
- **Scoring:** exact-set-match on the box contents (a set of objects), with **95 % confidence
  intervals** reported. ✅ Baseline: "randomly outputs 0 to m = 3 objects from the set of
  objects that appeared in the same clauses as the box in question" — a deliberately **strong**
  baseline, "much stronger than a fully random baseline." ✅ *Adopt this baseline design; a naive
  random baseline would flatter our models.*
- **Controls (Desideratum 3):** they compute a "**signature**" of each initial state (e.g.
  `2111111` = first box has 2 objects, rest 1 each) and ensure no train/eval example shares an
  initial description modulo signature; plus a **lexically disjoint** variant to prove the model
  isn't pattern-matching surface forms. ✅ *This is the right way to build a controlled probe and
  we should copy it.*

**Key finding.** ✅ Models pretrained on **code** do non-trivial entity tracking; models trained
on text alone do not. The model table separates this explicitly by a `Code?` column:
GPT-3 `davinci` (175B, ✗ code), `davinci-instruct-beta`, `text-davinci-001` (✗) vs GPT-3.5
`code-davinci-002`, `text-davinci-002`, `text-davinci-003` (✓ code). Behavioral results:
`text-davinci-003` "consistently outperformed" others but "the accuracy of this model also
decreases as the [number of operations increases]"; **Flan-T5 "seemed to ignore the
operations"** and **GPT-3 davinci "primarily repeated the initial state"** with a steep decrease.
**Flan-T5-base is 250M** — i.e. the paper contains direct evidence that **a 250M model fails
this task by copying the initial state.** ✅

> **Implication for us: the boxes task at 100M–1B is a floor.** A sub-1B model trained on a few
> B tokens of web text will "primarily repeat the initial state." That is *not useless* — the
> **"ignores operations / repeats initial state" failure signature is itself a measurable,
> gradable outcome** (report % of predictions equal to the initial state). But do not expect
> above-baseline accuracy. **Tier 3 at best, or Tier 2 as a *trained* probe.**

**Follow-up:** Prakash et al. *Fine-Tuning Enhances Existing Mechanisms: Entity Tracking* (ICLR
2024, arXiv:2402.14811) — circuit analysis showing fine-tuning *sharpens an existing* entity
tracking circuit rather than creating a new one. ⚠️ **Not independently verified this pass.
Flagged.**

### 4.2 bAbI — and a serious eval-validity warning

**Primary source:** Weston et al. *Towards AI-Complete Question Answering: A Set of Prerequisite
Toy Tasks.* arXiv:1502.05698. **20 tasks**, **1k or 10k** training examples per task, per-task
accuracy with a conventional **95 % "pass" threshold**. ✅

The state/entity-tracking-relevant tasks, with fact counts from BABILong's Table 1: ✅

| Task | Name | Facts/task | Relevant facts | LLM acc. at 0K (no distractors) |
|---|---|---|---|---|
| QA1 | Single supporting fact | 2–10 | 1 | **99** |
| QA2 | Two supporting facts | 2–68 | 2 | **64** |
| QA3 | Three supporting facts | 4–320 | 3 | **38** |
| QA4 | Two arg relations | 2 | 1 | 55 |
| QA5 | Three arg relations | 2–126 | 1 | 80 |
| QA6 | Yes/no questions | 2–26 | 1 | 91 |
| QA7 | **Counting** | 2–52 | 1–10 | **28** |
| QA8 | Lists/sets | 2–50 | 1–8 | 77 |
| QA9 | Simple negation | 2–10 | 1 | 89 |
| QA10 | Indefinite knowledge | 2–10 | 1 | 80 |

(the last column is median accuracy across the models BABILong tested, **with no background
text at all** — so these are *ceiling* numbers.) ✅ Note **QA7 counting = 28 %** and
**QA3 = 38 %** even with zero distractors: these are hard for real LLMs.

⚠️ **THE VALIDITY TRAP.** bAbI was designed to be *trained on*. A tiny model trained on bAbI's
own 10k split reaches >95 % on most tasks — memorizing templates, not tracking state. **Any bAbI
number from a model that saw bAbI-format data in training is meaningless as an architecture
comparison.** Two legitimate protocols: (a) **zero/few-shot** on a pretrained LM (expect floor
at <1B), or (b) **train all architectures identically on bAbI from scratch** and compare
sample-efficiency + length generalization — which is a clean, cheap, controlled comparison and
is what BABILong's own RMT/Mamba experiments do. **Choose (b) and say so explicitly.** ✅
(Corollary: bAbI is *immune to pretraining contamination* precisely because it is generated —
BABILong makes this point: "Generated benchmarks, such as bAbI and BABILong, are immune to this
type of contamination.")

**BABILong's small-model evidence (repeating from §3.2 because it is the load-bearing datum):**
✅ verbatim — "fine-tuning of small scale models (**ARMT and RMT, 137M** and **Mamba, 130M**)
shows that the tasks are **solvable**." Setup: RMT with a **GPT-2 (137M)** backbone, trained
per-task, **segment size 512, memory size 16**; ARMT used **10 memory tokens**. Finetuned Mamba
achieved the **best overall** results; ARMT scaled to **50M tokens**. Also: "**RAG methods do not
help**" (retrieval by chunks, top-5 to top-20). ✅

> **This is the strongest evidence in the whole document that a long-range eval can produce real
> signal at ~130M params.** It is protocol (b): train on the task, compare architectures on
> sample efficiency and length extrapolation. **Make BABILong-QA1–QA5 a Tier-2 trained probe.**

**Other entity-state benchmarks.** ⚠️ Named in the brief; **I did not verify these this pass** —
ProPara (process paragraphs / entity state change), TRACIE (temporal), SCONE
(Alchemy/Scene/Tangrams — Long, Potts et al.; note Kim & Schuster explicitly contrast their
boxes design against **Alchemy**, saying their setup "has a benefit of requiring fewer additional
reasoning abilities. The beaker domain in Alchemy requires the model to count and perform simple
arithmetic" ✅), neural process networks / recipes, TextWorld. **Flagged.** Of these, **SCONE
Alchemy** is the closest cheap synthetic — but per Kim & Schuster it *confounds state tracking
with arithmetic*, so prefer the boxes design.

### 4.3 The expressivity theory — what is provably impossible

#### 4.3.1 Transformers: log-precision ⊆ uniform TC⁰

Merrill & Sabharwal, *The Parallelism Tradeoff: Limitations of Log-Precision Transformers*
(arXiv:2207.00729, TACL): log-precision transformers can be simulated by **uniform TC⁰**
circuits (constant-depth, polynomial-size, unbounded-fan-in threshold circuits). ✅ (verified
via its role as the foundational premise in 2404.08819, which states "One theoretical weakness of
transformers is that they cannot express certain kinds of sequential computation and state
tracking (Merrill & Sabharwal, 2023a)" and follows "the proof structure of Merrill &
Sabharwal"). Related: *A Logic for Expressing Log-Precision Transformers* (2210.02671) and
*The Expressive Power of Transformers with Chain of Thought* (2310.07923) — CoT adds serial
computation and escapes TC⁰. ⚠️ exact statements of the latter two not independently pulled.
**Flagged.**

**Consequence:** unless **TC⁰ = NC¹** (widely believed false), transformers cannot solve
**NC¹-hard** problems in constant depth. From 2404.08819 verbatim, the NC¹-hard set includes:
"**simulating finite automata (NC¹-complete), evaluating boolean formulas (NC¹-complete),
determining graph [connectivity] (NC¹-complete)**." ✅

#### 4.3.2 SSMs are *also* in TC⁰ — "The Illusion of State"

**Primary source:** Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models.*
ICML 2024, arXiv:2404.08819. <https://arxiv.org/abs/2404.08819> · Code: <http://jpetty.org/ssm-illusion>

**Abstract, verbatim (the key claim):** ✅

> "But do SSMs truly have an advantage (over transformers) in expressive power for state
> tracking? **Surprisingly, the answer is no.** Our analysis reveals that the expressive power
> of **S4, Mamba, and related SSMs is limited very similarly to transformers (within TC⁰)**,
> meaning these SSMs **cannot solve simple state-tracking problems like permutation composition**
> and consequently are **provably unable to accurately track chess moves with certain notation,
> evaluate code, or track entities in a long narrative**."

**Theorems, verbatim:** ✅
- **Theorem 4.2 (Non-gated SSM).** "Let M be a log-precision generalized linear SSM such that,
  for any i, Āᵢ = Ā, B̄ᵢ = B̄, Cᵢ = C, Dᵢ = D. Then there exists an **L-uniform TC⁰ circuit
  family** that computes M's convolutional form." → **Corollary 4.3:** holds for **S4**.
- **Theorem 4.4 (Diagonal SSM).** "Let M be a log-precision generalized linear SSM where for
  1 ≤ i ≤ n: (1) the transition matrix **Āᵢ is diagonal**, denoted diag(āᵢ); (2) each of āᵢ, B̄ᵢ,
  Cᵢ and Dᵢ can be computed in L-uniform TC⁰ as a function of xᵢ. Then there exists an
  **L-uniform TC⁰ circuit family** that computes M's convolutional form." → **Corollary 4.5:**
  holds for **S6 (used by Mamba)**, because "S6's transition matrix Āᵢ is defined as exp(δᵢA) for
  a fixed diagonal A. The set of diagonal matrices is closed under scalar multiplication and
  matrix exponentiation."
- **Theorem 4.6** extends this to **simultaneously diagonalizable** SSMs.

> ⚠️ **The load-bearing architectural condition is DIAGONAL (elementwise) TRANSITIONS.** The
> proof works because diagonal-matrix products reduce to elementwise products, which are in TC⁰.
> **Gated short convolutions are even weaker** — a depthwise conv is *literally* an elementwise
> (per-channel) linear operator with a fixed finite kernel, so the same TC⁰ containment applies a
> fortiori. ⚠️ **This is my inference, not a stated theorem in 2404.08819. Flagged as
> inference** — but note the Zoology/Based line independently proves BaseConv results, and
> BaseConv canonically simulates gated convolutions (Zoology Thm 4.2).

**What they can and cannot do.** ✅
- **CANNOT (assuming TC⁰ ≠ NC¹):** the **S5 word problem** (composition of permutations of 5
  elements) and anything NC¹-hard. **Definition 3.1 (Word problem):** given a finite monoid M
  and a sequence of elements, compute the product. Barrington: "the word problem of every finite
  **non-solvable** group is NC¹-complete." S5 and **A5** (the alternating group on 5 elements,
  a subgroup of S5) are non-solvable. **Proposition 3.3: "S5 can be reduced to chess state
  tracking"** → **Corollary 3.4: "The chess state-tracking problem is NC¹-[hard]."** ✅
- **CAN:** **parity** — note verbatim, the word problem for parity is "**multiplication modulo
  2**", which is in TC⁰. ⚠️ **So parity is NOT an example of what SSMs provably cannot do.**
  Getting this wrong is a common error. (Whether a *particular* architecture *learns* parity is
  a separate empirical matter — see Grazzi et al. below.)

**Empirical protocol (directly reusable, and cheap).** ✅
- Groups tested: **A5** (non-solvable, NC¹-complete), **A4 × Z5** (solvable, in TC⁰), and one
  more commutative control. Each element gets a unique token.
- Models: transformer (TC⁰ baseline), RNN (true recurrence), **S4**, **Mamba**, and **IDS4**
  (input-dependent S4, à la Liquid S4 — note the relevance to LIV!).
- Metric: **full-sequence accuracy**; the reported quantity is **"the minimum depth with 90 %
  test accuracy as a function of input sequence length."** ✅ *This "minimum depth vs length"
  curve is an elegant, cheap, and rigorous architecture-comparison instrument.*
- Training detail: "We always include **all 3600 pairwise sequences of length 2** in the training
  data along with the training split of length-n sequences."
- **Results, verbatim:** "single-layer **RNN and IDS4** models learn the word problem for
  arbitrarily long sequences for all three groups. In contrast, **transformer, S4, and Mamba
  models require depth monotonically increasing in sequence length**." Also: "Transformers, S4,
  and Mamba require **greater depth even for A4 × Z5**, which can be theoretically expressed by
  TC⁰ circuits" — i.e. *even TC⁰-expressible tasks are not efficiently learned*. And a nuance
  worth quoting: "**S4 and Mamba appear empirically better than transformer at approximate state
  tracking** on the non-commutative tasks."
- **Theorem 5.2 / IDS4:** an **input-dependent** SSM (non-diagonal / Liquid-S4-style) **can both
  express and learn** the S5 word problem. ✅ **This is the escape hatch, and it is an
  input-dependence result — the same lever Zoology's Theorem 4.5 identifies for recall.**

> **Two independent literatures converge on the same conclusion: INPUT-DEPENDENCE of the
> sequence-mixing operator is the decisive property** — for recall (Zoology Thm 4.5 / Claim 2)
> *and* for state tracking (Illusion Thm 5.2). A **gated** short conv is input-dependent only in
> its *gate* (elementwise), not in its *mixing weights* (the kernel is fixed). **That predicts
> our mostly-LIV model inherits the gated-convolution limitation on both axes except where the
> attention layers intervene.** ⚠️ inference, clearly flagged.

#### 4.3.3 Finite-precision SSMs = star-free regular languages

**Primary source:** Sarrof, Veitsman, Hahn. *The Expressive Capacity of State Space Models: A
Formal Language Perspective.* NeurIPS 2024, arXiv:2405.17394.

**Abstract, verbatim:** ✅

> "We find that **SSMs and transformers have overlapping but distinct strengths.** In **star-free
> state tracking, SSMs implement length-generalizing solutions to problems that transformers
> struggle to represent exactly.** They can also model **bounded hierarchical structure with
> optimal memory even without simulating a stack.** On the other hand, we **identify a design
> choice in current SSMs that limits their expressive power.**"

⚠️ The brief's phrasing "express exactly the star-free regular languages" is **approximately but
not exactly** what the abstract claims — the abstract is a comparative statement about star-free
*state tracking* and length generalization, plus a limiting design choice. **I did not extract
the formal theorem.** For context, Angluin, Chiang & Yang (2310.13897, cited by 2404.08819) show
**masked hard-attention transformers recognize exactly the star-free languages** — so the
star-free class is the natural meeting point for both families. **Flagged: verify the precise
SSM theorem statement before citing a specific equality.**

#### 4.3.4 Shortcuts to automata — the length-generalization prediction

Liu, Ash, Goel, Krishnamurthy, Zhang. *Transformers Learn Shortcuts to Automata*
(arXiv:2210.10749). Cited by 2404.08819 in the decisive sentence: ✅ verbatim — SSMs
"**can only solve simple state-tracking problems for which shallow shortcuts exist** (Liu et al.,
2023)."

**Prediction:** a constant-depth model trained on sequences of length ≤ n learns a **shortcut**
valid up to n, and this shortcut **does not extend** to longer sequences. ⇒ **Any state-tracking
eval must test length extrapolation, or it will be passed by a shortcut.** ✅ Concretely: train
at length L, test at 2L, 4L, 8L. This is the single most important design rule in this section.

#### 4.3.5 RNNs are not Transformers (Yet) — and "add one attention layer" is a *theorem*

**Primary source:** Wen, Dang, Lyu. *RNNs are not Transformers (Yet): The Key Bottleneck on
In-context Retrieval.* arXiv:2402.18510. <https://arxiv.org/abs/2402.18510>

**The four retrieval primitives, verbatim definitions:** ✅
- **Def 4.2 (Index):** "given a sequence of tokens with length n and a query token i ∈ [n],
  requires the model to output the **type of the i-th token**."
- **Def 4.3 (Associative Recall):** "given a sequence of tokens with length n… and a query token
  q ∈ [n], requires the model to output the **next token of q** in the sequence."
- **Def 4.4 (c-gram Retrieval):** given a query (c−1)-gram that prefixes a c-gram in the
  sequence, "output the **last token of that c-gram**." (The multi-token-key generalization of
  AR — "studied empirically, but not theoretically in Jelassi et al. (2024).")
- **Def 4.5 (Counting):** "given a sequence…, a query token q, and a query number t ∈ ℕ,
  output 0 or 1 to indicate whether the **number of occurrences of q** is greater than t."

**Theorem 4.6, verbatim (the central lower bound):** ✅

> "For task T ∈ {**Index, AR, c-gram retrieval, Counting**}, there exist **constant-size
> Transformers** that can solve T. On the other hand, **any RNN with o(n)-bit memory cannot
> solve T of size n with any length of CoT** for large enough n."

With the crucial interpretive note, verbatim: "**the maximal context length that RNNs can
effectively retrieve from is linear in its state size.**" ✅ And an empirical citation directly
on point: "in **Waleffe et al. (2024)**, both pretrained **Mamba and Mamba-2 7B** models are
shown to have significantly worse **Phonebook-retrieval** capabilities on **1K context length**
than Transformers with the same size and trained on the same data." ✅
*⇒ Phonebook at 1K context discriminates even at 7B. Strong support for §2.1.*

**Theorem 4.7 (retrieval is needed even for non-retrieval tasks):** ✅ "There exist constant-size
Transformers that can solve **IsTree** with CoT of length O(n). On the other hand, **any RNN with
o(n)-bit memory cannot solve IsTree with any length of CoT**." IsTree tokenization:
`{<s>, u₁, ∼, v₁, …, u_m, ∼, v_m}`. Their experiments train **0.5M / 1M / 2M** parameter models
on graphs of **16 / 32 / 64** nodes — ✅ **note the scale: half-a-million-parameter models. This
is by far the cheapest architecture probe in this document.**

**Theorem 5.6 — the hybrid result, and the theoretical charter for our architecture:** ✅

> **Definition 5.5 (Hybrid RNN):** "a model that consists of an RNN with transition and output
> function t, o and **one Transformer layer** f, the output of the RNN is used as the input of the
> Transformer layer…"
>
> **Theorem 5.6.** "For task T ∈ {**Index, AR, c-gram retrieval, Counting, IsTree**}, there exist
> **constant-size hybrid Linear RNNs** that can solve T. For T other than IsTree, **no CoT is
> required**, and for IsTree, O(n log n) steps of CoT is required."

And from the contributions list, verbatim: "**adding one Transformer layer at the end of the RNN
architecture is sufficient to close the representation gap.**" ✅

> **This is the theorem that justifies the mostly-LIV design — and it also sets the bar for what
> the experiment must show.** "A few attention layers fix retrieval" is *already a theorem and
> already demonstrated* (Zoology's <10 %-of-layers result closes >80 % of the MQAR gap; Based
> recovers 90.8 % with a 64–128 token window; Wen et al. Thm 5.6). **So the novel claim cannot be
> "hybrids retrieve better than pure recurrent."** It must be a *quantitative placement/budget*
> claim, and the evals must resolve **where the residual gap is** — hence: MQAR at high (N,D),
> AR-sliced perplexity, phonebook length sweeps past the conv receptive field, and length
> extrapolation.

#### 4.3.6 Copying: the state-size bound (recap of §2.1)

Jelassi et al. 2402.01032 (verbatim, §2.3): ✅ **Theorem 2.7** "Fix some GSSM H over state space
S. Then, for all L, for the uniform copy distribution D_L, the model H has error
**err(H) > 1 − |S| / D^L**." **Corollary 2.8:** "every GSSM H with state space S s.t.
**mem(S) < L log(D) − 1** has error > 1/2." Versus **Theorem 2.3 / Corollary 2.5:** a **depth-2**
transformer of dimension **O(log(L/ε) log D)** copies with error < ε. **Remark 2.9:**
transformers are near-optimal in input-dependent memory for copying (Õ(L)).

#### 4.3.7 Fixes for linear RNNs (negative eigenvalues, delta rule)

- **DeltaNet** / *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*
  (arXiv:2406.06484): a delta-rule (rank-1 corrective) update, non-diagonal transition, hardware-
  efficient. ⚠️ **not independently verified this pass. Flagged.**
- **Grazzi et al.**, *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*
  (arXiv:2411.12537): allowing the eigenvalue range **[−1, 1]** instead of **[0, 1]** enables
  **parity** and richer state tracking. ⚠️ **not independently verified this pass. Flagged** —
  but note this is *consistent* with §4.3.2: parity is in TC⁰, so this is a *learnability/
  parameterization* fix, not a complexity-class escape. **Do not describe it as escaping TC⁰.**
- **"Understanding the Skill Gap in Recurrent Language Models"** (Eyuboglu/Arora et al.)
  ❓ **I could not confirm this paper's existence/arXiv id in this pass. Flagged.**
  Its reported thesis (separating "cannot store" from "cannot retrieve" failure modes) is,
  however, exactly the distinction that JRT's data-order experiment operationalizes (§1.7) —
  use that regardless.

### 4.4 SYNTHESIS — what a mostly-LIV hybrid is predicted to fail

Predictions for **many gated short-conv layers + a few full GQA layers**, versus all-attention
and versus recurrent hybrids:

| Axis | Prediction for mostly-LIV | vs all-attention | vs recurrent hybrid | Diagnostic |
|---|---|---|---|---|
| **(i) Associative recall / retrieval capacity** | ⚠️ **Mostly rescued** by the attention layers (Wen Thm 5.6; Zoology <10 % of layers → >80 % of gap). **Residual risk:** the *number* and *placement* of attention layers, and whether the conv layers can present keys/values to them in a usable form. | Attention: **d = 64 constant** (Zoology Prop 4.3, Claim 1) | Recurrent hybrid is bounded by **Ω(N)-bit state** (Based Thm 3.1); a mostly-LIV model with **full** attention has O(L) KV cache, so it should **beat** the recurrent hybrid at large N,D | **MQAR** sweep over (N, D) reading off required d; **AR-sliced ppl**; **phonebook L-sweep** |
| **(ii) Copying** | ✅ Should succeed **if** the attention layers are full-range — a depth-2 transformer suffices (Thm 2.3). ⚠️ Fails if attention is windowed. | Near-optimal (Remark 2.9) | Fails beyond state size (Cor 2.8) | **Verbatim copy** of length-L strings; **phonebook** |
| **(iii) Inherently-sequential state tracking** (permutation composition, S5/A5, chess, code eval, narrative entity tracking) | ❌ **Predicted to FAIL, and the attention layers do NOT help** — attention is in TC⁰ too. Required depth grows with sequence length. | ❌ Also fails (TC⁰) | ❌ Also fails if diagonal/elementwise transitions (Illusion Cor 4.3/4.5). ✅ **Would succeed** only with non-diagonal input-dependent transitions (IDS4 Thm 5.2, DeltaNet-style) | **A5 word problem**: min-depth-for-90 %-acc vs length; **boxes** with the implicit "move contents" op; **length extrapolation** |
| **(iv) Counting** | ⚠️ Attention can count (Wen's COUNT mechanism, Thm 4.6) but only within the attention layers' range; conv layers count only within the kernel. | ✅ Constant-size solution | ❌ Fails with o(n) memory (Thm 4.6) | **RULER FWE/CWE**; **bAbI QA7** (LLM ceiling only 28 %!) |
| **(v) Parity** | ⚠️ In TC⁰ so *expressible*, but **empirically hard to learn** for both families; sensitive to eigenvalue range (Grazzi). | learnable-with-depth | depends on parameterization | **parity** with length extrapolation |

**Two additional predictions specific to SHORT convolutions** (⚠️ inference, flagged):
1. **Fixed receptive field.** A depthwise conv of kernel k over ℓ conv layers gives a receptive
   field of **O(ℓ·k)** tokens. If our model has, say, kernel 3–4 and ~20 conv layers, the
   conv-only receptive field is **~60–80 tokens**. **Every token interaction beyond that must
   route through one of the few attention layers.** ⇒ Predict a **sharp accuracy cliff in the
   phonebook/MQAR gap-distribution once the required interaction distance exceeds the conv
   receptive field**, and heavy dependence on **where** the attention layers sit in the stack
   (an attention layer at layer 2 cannot use features the conv stack has not yet assembled; one
   at the last layer cannot feed its retrieved content into further conv processing).
   **This is the most interesting and most testable prediction of the whole design.** Design the
   eval to sweep interaction distance explicitly: use MQAR with **`power_a` varied (0.01 → 1.0,
   i.e. front-loaded → uniform gaps)** to move mass past the receptive field.
2. **Zoology's own sparse-attention finding is the warning:** "sliding window attention helps
   close the MQAR gap when the token-interaction distances are relatively **short**, and
   **degrades on longer distances**." Griffin's phonebook result is the same story at 7B (solves
   up to its 1024-token local window, then fails). **Our few attention layers must be full-range
   for the design to differ from Griffin.**

---

## 5. Instruction persistence — "did it forget the system prompt?"

### 5.1 What exists

**IFEval** — Zhou et al., *Instruction-Following Eval for Large Language Models*,
arXiv:2311.07911. <https://arxiv.org/abs/2311.07911> ·
Repo: `google-research/google-research/tree/master/instruction_following_eval` ✅

- **Core idea:** "**verifiable instructions**" — instructions checkable "using a simple,
  interpretable, [deterministic program]", e.g. "write at least 25 sentences." ✅
- **Scale:** "We identified **25 types** of those verifiable instructions and constructed around
  500 prompts" — precisely **541 prompts**, each containing one or more verifiable
  instructions. ✅
- **Four metrics, verbatim:** ✅
  1. **Prompt-level strict-accuracy:** "The percentage of prompts that **all** verifiable
     instructions in each prompt are followed."
  2. **Inst-level strict-accuracy:** "The percentage of **verifiable instructions** that are
     followed."
  3. **Prompt-level loose-accuracy** and 4. **Inst-level loose-accuracy** — same, under the
     *loose* criterion.
- **The "loose" criterion (worth copying).** ✅ Each response is scored under **8
  transformations** — the identity, plus: "(1) Remove commonly seen font modifiers in the
  markdown syntax, especially '*' and '**'. (2) Remove the **first line** of the response, so
  that we skip intros like 'Sure, here it is:'. (3) Remove the **last line**… like 'Hope it
  helps.'" — and every pairwise and triple combination. An instruction counts as followed if
  *any* transformation passes. The authors note this "reduces false negatives [but] is likely to
  introduce false positives." ✅ *For base models this leniency is essential; use loose scoring
  and report both.*
- Reference scores: **GPT-4** 76.89 / 83.57 / 79.30 / 85.37; **PaLM 2 S** 43.07 / 55.76 /
  46.95 / 59.11. ✅

**Does IFEval work at < 1B?** ✅ **Only for instruction-tuned models, and then weakly.**
Measured (HF model cards, `lighteval`):
- **SmolLM2-135M-Instruct: IFEval 29.9** (SmolLM-135M-Instruct: 17.2)
- **SmolLM2-360M-Instruct: IFEval 41.0** (Qwen2.5-0.5B-Instruct: 31.6; SmolLM-360M-Instruct: 19.8)

So a **135M instruct** model scores ~30 and a **360M instruct** ~41. ✅ **That is real signal.**
But these are *post-SFT+DPO* models. ⚠️ **For a base model trained on a few B tokens, IFEval will
floor** — a base LM does not follow "write at least 25 sentences" at all. ❓ I found no published
base-model IFEval below 1B. **Flagged.**

**"Lost in the Middle"** — Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang. TACL 2024,
arXiv:2307.03172. <https://arxiv.org/abs/2307.03172> ✅
- **Multi-document QA:** built from **NaturalQuestions-Open**, using "the 2655 queries where the
  annotated long answer [is a paragraph]"; the input has **1 document containing the answer +
  k−1 "distractor" documents** from Wikipedia that do not contain it. Context length is varied by
  changing **k ∈ {10, 20, 30}** documents; **the position of the gold document is swept across
  all k positions.** Baselines: **closed-book** (no documents) and **oracle** (gold only). ✅
- **Finding:** a **U-shaped curve** — "performance is highest when relevant information occurs at
  the [beginning or end]"; models "do not robustly make use of information in the middle of long
  contexts." Also: encoder-decoder models are relatively robust to position *only within* their
  training sequence length. ✅
- **The key-value retrieval synthetic — a cheap, reusable, fully-verifiable task.** ✅ Exact
  prompt (verbatim from Figure 6):

```
Extract the value corresponding to the specified key in the JSON object below.

JSON data:
{"2a8d601d-1d69-4e64-9f90-8ad825a74195": "bb3ba2a5-7de8-434b-a86e-a88bb9fa7289",
 "a54e2eed-e625-4570-9f74-3624e77d6684": "d1ff29be-4e2a-4208-a182-0cea716be3d4",
 ...}

Key: "9f4a92b9-5f69-4725-ba1e-403f08dea695"
Corresponding value:
```
  - Keys and values are **unique randomly-generated 128-bit UUIDs**. **k−1 distractor pairs.**
  - **k ∈ {75, 140, 300}** pairs, **500 examples each**.
  - **Scoring:** "we measure accuracy by evaluating whether the **correct value appears in the
    predicted output**" — substring containment, no judge. ✅
  - Design rationale, verbatim: they deliberately remove "as much natural language semantics as
    possible (using **random UUIDs** instead), since language features may present potential
    confounders." ✅
  - **Finding:** even this "only requires identifying exact match within the input context, [yet]
    not all models achieve high performance" — GPT-3.5-Turbo and MPT-30B-Instruct are worst when
    the pair is **in the middle**. ✅
  - ⚠️ Cost note from the paper: evaluating GPT-4 on the full multi-doc QA + KV experiments "would
    cost upwards of **$6000**" — irrelevant for local models, but indicative of scale.

> **This KV task is essentially MQAR with UUID tokens and a position sweep.** It is the natural
> bridge between MQAR (§1) and a position/depth analysis. ⚠️ At < 1B, UUIDs tokenize into many
> tokens each and 75 pairs already blows past a 2k context — **scale k down to {5, 10, 20, 40}
> and/or shorten the keys to 8 hex chars.** Then it is a fine Tier-1 probe.

**Other candidates.** ⚠️ Named in the brief / plausible; **not verified this pass. Flagged:**
FollowBench (2310.20410, multi-level constraints), Multi-IF (multi-turn), SIFo (sequential
instruction following), InfoBench, CFBench, ComplexBench, SysBench, "persona drift", LongProc.
❌ **CORRECTION: LongForm (2304.08460) is an instruction-tuning DATASET for long-form
generation, not an instruction-persistence eval.** ⚠️ (stated with moderate confidence from the
title/abstract framing; **verify before citing**.)

### 5.2 Verdict: **no standard eval exists for instruction persistence at small scale**

✅ **Stated plainly:** I found **no standardized benchmark** that measures "did the model forget an
instruction stated far earlier in the context," and certainly none validated below 1B. IFEval has
no distance axis. Lost-in-the-Middle has a distance axis but measures *fact* retrieval, not
*instruction* compliance. HELMET has an instruction-following category but targets frontier
models. **The needle-is-an-instruction design does not exist as a standard.** ❓/✅

### 5.3 PROPOSED TASK: **Persistent Instruction Probe (PIP)**

A synthetic, programmatically verifiable, distance-controlled instruction-persistence task
designed for **100M–1B base models**. (This is my design; ⚠️ novel, not from the literature.)

**Core construction.**

```
<INSTRUCTION>
<FILLER: d tokens of held-out natural text>
<QUERY>
```

with three instruction types of escalating difficulty, all trivially verifiable, all chosen so
that **even a tiny base model can comply at d = 0**:

| Type | Instruction text | Query | Verifier |
|---|---|---|---|
| **T1 — Fixed-token suffix** | `Rule: every answer must end with the word ZEBRA.` | `Q: What color is the sky?\nA:` | does the generation contain `ZEBRA`? |
| **T2 — Format constraint** | `Rule: answer using exactly one uppercase letter.` | `Q: Which is larger, a cat or a dog? Answer A for cat, B for dog.\nA:` | is the first non-space token in `{A,…,Z}` and length 1? |
| **T3 — Mapping / indirection** | `Rule: when you see the token QQQ, output the number 7.` | `… QQQ` | is the next token `7`? |

**T3 is the important one** — it is a *rule* that must be *retrieved and applied*, i.e. it is
formally an associative-recall problem wearing instruction-following clothes. It connects this
probe directly to §1 and makes it interpretable under the theory in §4.

**Difficulty knobs.**
1. **d = instruction→compliance distance**, the primary axis:
   `d ∈ {0, 16, 64, 128, 256, 512, 1024, 2048}` tokens of filler. **Crucially, sweep d past the
   gated-conv receptive field** (~ℓ·k tokens; see §4.4) — the predicted cliff is the headline
   measurement.
2. **Filler type:** (a) low-entropy repeated sentence (the Landmark filler) vs (b) real held-out
   web text vs (c) **text containing decoy rules** (e.g. `Rule: every answer must end with the
   word WALRUS.`). Condition (c) is the *hard distractor* condition and is what RULER's MK-NIAH
   showed matters most.
3. **Number of competing instructions:** 1 vs 2 vs 4 rules, only one of which is queried
   (multi-key), or all of which must be honored (multi-value).
4. **Instruction position:** at the very start (system-prompt analogue) vs at a random depth.

**Scoring.** Programmatic, three numbers per (type, d) cell:
- **Compliance rate** — verifier passes.
- **Base rate** *(no-instruction control)* — same query with the instruction **removed**; measures
  how often the constraint is satisfied by chance. **Report compliance − base rate.**
- **Adjacent-compliance ceiling** *(d = 0 control)* — the same instruction immediately before the
  query. **This is the crucial control**: it separates *"the model cannot follow this instruction
  at all"* (low even at d = 0 ⇒ the task is out of the model's reach, discard the cell) from
  *"the model forgot"* (high at d = 0, decaying with d ⇒ **exactly the effect we want to
  measure**).
- **Headline metric: normalized persistence** `P(d) = (acc(d) − base) / (acc(0) − base)`, so
  P(0) = 1 by construction and P(d) is a decay curve. Fit a **half-life d₅₀** (the distance at
  which P = 0.5) and compare architectures on **d₅₀**, which is a single interpretable scalar
  with an obvious mechanistic reading.

**Why this suits 100M–1B base models.** T1/T2 need no semantic competence — only copying a
literal token or emitting one character. A 135M base model can do this at d = 0 (that is an
induction-head-level capability, present very early in training). ⚠️ **Pilot the d = 0 cell
first**; if the adjacent-compliance ceiling is < ~0.5 for a given type, drop that type.

**Item count and noise.** With binary scoring at p ≈ 0.5, the per-cell SE is
√(p(1−p)/n). For **n = 400** items, SE = 2.5 pts; for **n = 1000**, SE = 1.6 pts. Since
architectures are compared on **identical items**, use the **paired** analysis of §8 — the paired
SE is typically 2–3× smaller. **Recommendation: n = 500 items per (type, d) cell**, 3 types × 8
distances = 24 cells × 500 = 12,000 generations of ≤ 8 tokens each. **Cost: minutes on one
GPU.** ✅ (own calculation)

**Caveat.** ⚠️ For **base** (non-SFT) models, the instruction will be followed only if the format
is *in-distribution* for pretraining. **Prepend 2 few-shot demonstrations of the rule being
honored** (as Jelassi et al. do for phonebook and Kim & Schuster do for boxes) — otherwise you
measure "can it do zero-shot instruction following" (answer: no) rather than persistence.

---

## 6. Small-scale evaluation validity

### 6.1 What actually gives signal at 100M–1B

Verified numbers. **SmolLM2 model cards** (`lighteval`, zero-shot unless noted): ✅

| Benchmark | Chance | **135M** (2T tok) | **360M** (4T tok) | Qwen2.5-0.5B | Verdict @100–350M | Verdict @~1B |
|---|---|---|---|---|---|---|
| HellaSwag | 25 | **42.1** | **54.5** | 51.2 | ✅ SIGNAL | ✅ SIGNAL |
| ARC (avg) | 25 | **43.9** | **53.0** | 45.4 | ✅ SIGNAL | ✅ SIGNAL |
| PIQA | 50 | **68.4** | **71.7** | 69.9 | ✅ SIGNAL | ✅ SIGNAL |
| MMLU (cloze) | 25 | 31.5 | 35.8 | 33.7 | ⚠️ WEAK (barely above chance) | ⚠️ WEAK |
| CommonsenseQA | 20 | **33.9** | **38.0** | 31.6 | ✅ SIGNAL | ✅ SIGNAL |
| **TriviaQA** | 0 | **4.1** | **16.9** | 4.3 | ⚠️ WEAK at 135M, ✅ **SIGNAL at 360M** | ✅ SIGNAL |
| **WinoGrande** | 50 | **51.3** | **52.5** | 54.1 | ❌ **CHANCE** | ⚠️ WEAK |
| OpenBookQA | 25 | 34.6 | 37.4 | 37.4 | ⚠️ WEAK | ⚠️ WEAK |
| GSM8K (5-shot) | 0 | 1.4 | 3.2 | 33.4* | ❌ FLOOR | ❌ FLOOR |

\* Qwen2.5-0.5B's GSM8K 33.4 is an outlier reflecting heavy math data, not scale.

**The two most important entries:**
- ❌ **WinoGrande is at chance (51.3 at 135M, 52.5 at 360M vs 50 chance).** Do **not** use it for
  ablation decisions below ~1B. This matches Madaan et al.'s general finding that some benchmarks
  sit at chance for a long time.
- ✅ **TriviaQA rises 4.1 → 16.9 from 135M→360M** — a 4× relative move. It is a **closed-book
  recall** task and therefore *thematically* aligned with our question, but note it measures
  *parametric* recall (memorized facts), **not in-context recall.** Do not confuse the two.

**Pythia (arXiv:2304.01373)** ✅ — the 8-model suite (70M…12B), **300B tokens** each, GPT-NeoX
tokenizer, all evals run with **lm-evaluation-harness** on "eight common language modeling
benchmarks" (Appendix G), Apache-2.0, 154 checkpoints/model. It is the ideal **reference suite**
for our scales. ⚠️ I did not extract Pythia's per-benchmark table this pass; the paper's
qualitative claim is that Pythia matches OPT/BLOOM at equal params/tokens. **Flagged — pull
Appendix G tables if exact Pythia numbers are needed.** ❓ I did **not** verify a specific Pythia
statement that named benchmarks "stay at chance below scale X"; do not attribute that to Pythia
without checking.

❓ **Unverified this pass (flagged):** exact tables from OLMo 1/2, TinyLlama, MobileLLM,
Cerebras-GPT. The SmolLM2 + Qwen2.5 numbers above are sufficient to make the design decisions.

**Independent corroboration from Based (§1.6)** — the single most decision-relevant datum in this
whole section: at **360M**, going 10B→30B tokens moved **LM-Evals 44.08 → 44.75 (+0.67)** while
**SWDE moved 57.97 → 70.75 (+12.8)**. ✅ **Commonsense-benchmark averages are nearly blind to
recall improvements at our scale.** Do not use them as the primary metric.

### 6.2 Eval noise — the numbers you need to size seed counts

**Primary source:** Madaan, Singh, Namboodiri, et al. *Quantifying Variance in Evaluation
Benchmarks.* arXiv:2406.10229. <https://arxiv.org/abs/2406.10229> ✅

**Their definitions:** ✅
- **Seed variance** `E(S,M)` = std-dev of the metric across **10 identically-configured 7B models
  differing only in init seed** (deterministic data order held fixed), **averaged over 21
  checkpoint timesteps** (10B…210B tokens).
- **95 % CI** = bootstrapped per-model CI; also the analytic form, verbatim:
  **CI_analytic(M) = 1.96 · √( S_M(1−S_M) / N )** where N = number of test instances.
  "bootstrapped and Analytic CIs converge when the number of bootstrap samples is large."
- **Monotonicity** = Kendall rank correlation between the score sequence over training and a
  monotone array.

**Table 1 (7B seed models) — the reference table for expected noise:** ✅

| Benchmark | Size | Chance | µ | **Seed σ** | 95 % CI | mon_disc | mon_cont |
|---|---|---|---|---|---|---|---|
| AGIEval | 2546 | 20 | 23.44 | 0.77 | 1.63 | 0.37 | 0.29 |
| ARC-C | 1165 | 25 | 39.71 | 0.80 | 2.74 | 0.88 | 0.91 |
| BigBench-Hard | 6511 | 0 | 29.10 | 0.87 | 1.07 | 0.77 | — |
| **COPA** | **100** | 50 | 78.80 | **2.15** | **8.30** | 0.56 | 0.90 |
| GSM8k | 1319 | 0 | 4.10 | 0.41 | 0.87 | 0.74 | 0.30 |
| **HellaSwag** | 10042 | 25 | 70.08 | **0.21** | **0.93** | **0.99** | **0.99** |
| HumanEval | 164 | 0 | 11.89 | 1.11 | 3.98 | 0.79 | 0.98 |
| MATH | 5000 | 0 | 1.52 | 0.23 | 0.28 | 0.52 | — |
| **MMLU** | 14042 | 25 | **25.86** | 0.57 | 0.72 | **0.09** | **0.15** |
| **MMLU-Cloze** | 14042 | 25 | **37.47** | **0.22** | 0.79 | **0.95** | **0.96** |
| NaturalQuestions | 3610 | 0 | 16.43 | 0.60 | 1.04 | 0.91 | — |
| PIQA | 1838 | 50 | 76.93 | 0.41 | 1.99 | 0.87 | 0.93 |
| SIQA | 1954 | 33 | 46.69 | 0.55 | 2.21 | 0.66 | 0.81 |
| TriviaQA | 11313 | 0 | 42.69 | 0.45 | 0.83 | 0.99 | — |

**Four decisive lessons:** ✅
1. **COPA is unusable** (only **100** items ⇒ seed σ = 2.15, 95 % CI = **8.30**). ❌ **Drop COPA.**
   Same for HumanEval (164 items).
2. **MMLU in MC format is at chance even at 7B/210B tokens (25.86 vs 25 chance) with
   monotonicity 0.09** — i.e. it does not even *increase* during training. **Reformulated as
   cloze, the same benchmark jumps to 37.47 with monotonicity 0.95 and seed σ drops 0.57→0.22.**
   ✅ **This single result is the strongest argument for CF/cloze formulation at small scale**, and
   verbatim from the conclusion: "**simple changes, such as framing choice tasks (like MMLU) as
   completion tasks, can often reduce variance.**"
3. **HellaSwag is the gold standard for low-noise ablation**: seed σ = **0.21**, monotonicity
   **0.99**. ✅
4. **Seed variance is generally well below the 95 % CI**, "though the ratio of the two is quite
   variable." ⚠️ So the CI computed from a single run *overestimates* the noise relevant to
   comparing two training runs — a useful, conservative fact.

**Continuous metrics — Table 2 SNR (SNR = µ/σ):** ✅ Using "probability mass of the predicted
answer for choice benchmarks and NLL of the correct answer for generation benchmarks":

| Benchmark | Disc SNR | **Cont SNR** | Gain |
|---|---|---|---|
| HellaSwag | 608.23 | **1921.15** | 3.2× |
| PIQA | 198.98 | **1641.14** | 8.2× |
| MMLU-Cloze | 302.73 | **678.42** | 2.2× |
| COPA | 38.63 | **662.41** | 17× |
| ARC-C | 45.89 | **381.64** | 8.3× |
| MMLU | 52.45 | **347.57** | 6.6× |
| AGIEval | 25.20 | **254.93** | 10× |
| HumanEval | 6.79 | **124.08** | 18× |
| GSM8k | 7.88 | **15.24** | 1.9× |

Verbatim: "the **SNR is considerably higher for continuous metrics for all benchmarks**,
suggesting that they may be better when comparing models in the sense that they are less
confounded by noise… along with accurate comparisons between two models that have performances
lying within the confidence interval for the discrete metric." ✅

> **⇒ MANDATE FOR OUR EXPERIMENT: report continuous metrics (answer probability mass / NLL /
> bits-per-byte), not just accuracy, on every ablation.** This is a 2–18× effective noise
> reduction *for free* — cheaper than any number of extra seeds.

They also tried **item analysis and item response theory** and found them **ineffective** at
reducing variance. ✅ Don't bother.

**Adding Error Bars to Evals** — Miller (Anthropic), arXiv:2411.00640. ✅ Five recommendations,
verbatim:
> 1. Computing standard errors of the mean using the **Central Limit Theorem**
> 2. When questions are drawn in related groups, computing **clustered standard errors**
> 3. Reducing variance by **resampling answers** and by **analyzing next-token probabilities**
> 4. When two models are being compared, conducting statistical inference on the
>    **question-level paired differences**, rather than the population-level summary statistics
> 5. Using **power analysis** to determine whether an eval (or a random subsample) is capable of
>    testing a hypothesis of interest

Formulas (verbatim): ✅
- Clustered SE: **SE_clustered = √( SE²_CLT + 2 Σ (s_{i,c} − s̄)(s_{j,c} − s̄) )** — "a kind of
  'sliding scale'" between perfectly correlated and uncorrelated clusters. Real-world effect is
  "far from trivial (**up to 3×**)". ⚠️ **Directly applies to us:** MQAR/RULER/phonebook items
  generated from the same template or the same haystack **are clustered**; naive SEs will be
  up to 3× too small.
- Unpaired: **µ̂_{A−B} = µ̂_A − µ̂_B**, **SE_{A−B} = √(SE²_A + SE²_B)**,
  **CI₉₅ = µ̂_{A−B} ± 1.96·SE_{A−B}**, **z = µ̂_{A−B}/SE_{A−B}**.
- Paired: define per-question difference **s_{A−B,i} = s_{A,i} − s_{B,i}** and analyze its mean.
- **Resampling:** K=2 reduces total variance by **1/3**; K=4 by **1/2**; K=6 further. ✅
- **Power (Eq. 9), verbatim:**
  **n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²**
  Worked example from the paper: with σ²_A = σ²_B = 0, ω² = 1/9, δ = 0.03, β = 0.20, α = 0.05:
  **n = (1.96 + 0.84)²(1/9)/(0.03)² ≈ 969** ⇒ "**new evals should contain at least 1,000
  questions** in order to have good signaling [power]." ✅

**⚠️ Unverified this pass (flagged):** Biderman et al. *Lessons from the Trenches*
(arXiv:2405.14782) and Heineman et al. *Signal and Noise* (AI2, 2025 — arXiv id not confirmed).
The latter is the most directly on-point work for choosing benchmarks by SNR for small-scale
ablation decisions; **strongly recommend chasing it down before finalizing the suite.** ❓

### 6.3 OLMES — the formulation standard

**Primary source:** Gu, Tafjord, Kuehl, Haddad, Dodge, Hajishirzi. *OLMES: A Standard for
Language Model Evaluations.* arXiv:2406.08446 (AI2). ✅

- **Motivation, verbatim:** existing practice "**disadvantage[s] smaller base models that require
  the unnatural 'cloze' formulation of multiple-choice questions**" — i.e. the MC-vs-cloze choice
  systematically biases small-model comparisons. ✅
- **The two formulations:** **MCF** (present labeled choices, score the label token) vs
  **CF** (score each answer string as a continuation). CF exists "because the MCF format is not
  natural for the pure language modeling task"; GPT-3 "found that it was possible to elicit much
  better performance using a 'cloze' completion version." ✅
- **Standard:** **5 in-context examples, manually curated per task** ("Restricting to 5 in-context
  examples helps limit [context]"; "5 shots generally does not provide meaningful [gains beyond]").
  Developed on **15 base models from 1B to 70B**, explicitly including **Pythia-1B, OLMo-1B,
  TinyLlama-1.1B, StableLM2-1.6B**. ✅
- **The 10 tasks, with split / #choices / #instances / CF normalization (Table 2):** ✅

| Task | Split | #C | # instances (of total) | CF norm |
|---|---|---|---|---|
| ARC-Challenge | Test | 4 | 1172 | **pmi** |
| ARC-Easy | Test | 4 | 1000 (2376) | char |
| BoolQ | Val | 2 | 1000 (3270) | none |
| CommonsenseQA | Val | 5 | 1221 | **pmi** |
| HellaSwag | Val | 4 | 1000 (10042) | char |
| MMLU | Test | 4 | 14042 | char |
| OpenBookQA | Test | 4 | 500 | **pmi** |
| PIQA | Val | 2 | 1000 (1838) | char |
| SocialIQA | Val | 3 | 1000 (1954) | char |
| WinoGrande | Val | 2 | 1267 | none |

  where CF normalization is per-**char**acter length, **pmi** (point-wise mutual information), or
  **none**. ✅ *These normalization choices are exactly the "acc vs acc_norm" ambiguity that makes
  cross-paper numbers incomparable; OLMES pins them down per task.*
- ⚠️ Note **OpenBookQA has only 500 instances** and **WinoGrande 1267** — per §6.2's power
  arithmetic, 500 items gives SE ≈ 2.2 pts at p = 0.5. Combined with WinoGrande being at chance
  below 1B, both are weak choices for us.
- **Recommendation:** adopt OLMES formatting + CF/cloze + its normalizations, and **report both
  CF and MCF** for the finalists.

### 6.4 Perplexity done right — Paloma

**Primary source:** Magnusson, Bhagia, Hofmann, Soldaini, et al. *Paloma: A Benchmark for
Evaluating Language Model Fit.* arXiv:2312.10523 (AI2). ✅
Code: <https://github.com/allenai/OLMo-Eval/tree/main/paloma>

- **Scale:** **546 domains** from **16 sources** (⚠️ **not** 585/18 — the brief's numbers are
  wrong; verified: "PALOMA is derived from **16 sources** further divided into **546 domains**").
  ~123.7M tokens total. ✅
- **The five guidelines, verbatim:** ✅
  - **G1 DECONTAMINATION** — "Remove pretraining data that leaks evaluation data."
  - **G2 TRAINING ORDER** — "keep the training data order the same to control differences from
    recency effects."
  - **G3 SUBSAMPLING** — "Subsample size poses a tradeoff between inference cost and variance.
    **Size subsamples to tolerate variance equally for each domain.**"
  - **G4 VOCABULARY** — "Vocabulary determines the event space… **Normalizing likelihood by a
    segmentation intrinsic to the text (e.g., bytes) partially addresses this, but fixing the
    vocabulary is preferable.**"
  - **G5 EVALUATION FORMAT** — "Use a consistent implementation of perplexity… regarding
    engineering details such as the handling [of] maximum sequence lengths."
- **Implementations:** G1 uses a **Bloom filter** for exact paragraph-level (newline-separated)
  overlap, **ignoring paragraphs < 13 unicode-segmented tokens** and punctuation/emoji-only
  paragraphs, **no decontamination on code sources**, and removes the **whole pretraining
  document** if any paragraph is contaminated. G2 fixes tokenization, max sequence length, and
  seed, with order-invariant dataloading. G3 targets **1M tokens per source** and **100k tokens
  per domain** (chosen by extrapolating an observed inverse relationship between subsample size
  and perplexity variance). G4 fixes vocab to **GPT-NeoX-20B + 3 special tokens**, falling back to
  **bits-per-byte** when vocabularies must differ. G5 follows The Pile's format: **documents are
  evaluated individually, not packed into concatenated max-length inputs; documents longer than
  max length are split into disjoint inputs.** ✅
- **Bits-per-byte formula, verbatim:** **BPB = (1/B)·log₂(e^{−ℓ}) = −ℓ / (B·ln 2)** where ℓ is
  the log-likelihood over documents and **B is the count of UTF-8 encoded bytes**. ✅
- ✅ They also note "few vocabulary types account for most of the loss measured in perplexity" —
  i.e. aggregate perplexity is dominated by frequent strings, **the same phenomenon that lets the
  AR-hit deficit hide in the average** (§1.1). Paloma proposes *average likelihood per vocabulary
  type* as an alternative. **Consider adding this alongside AR-hit slicing.**

> **For our experiment:** if all architectures share one tokenizer (they should — hold it fixed),
> **fix the vocabulary and report plain per-token NLL**; use BPB only when comparing against
> external models. **Report per-domain, not just aggregate** — and add the AR-hit slice.

---

## 7. Harness tooling

### 7.1 lm-evaluation-harness (EleutherAI) — **and a major find**

**Repo:** <https://github.com/EleutherAI/lm-evaluation-harness> · **License: MIT** ·
~13.5k stars ✅ (verified via GitHub API).

**Major find: RULER, BABILong, LongBench, Paloma, IFEval, and `squad_completion` are ALL native
harness tasks.** ✅ Verified via the GitHub API — `lm_eval/tasks/` contains 220 task
directories, including `ruler`, `babilong`, `longbench`, `longbench2`, `paloma`, `pile`,
`pile_10k`, `ifeval`, `squad_completion`, `squadv2`, `triviaqa`, `lambada`, `lambada_cloze`.
**This eliminates most of the integration work assumed by the brief.**

**RULER in the harness** (`lm_eval/tasks/ruler/`) — all 13 tasks present as individual YAMLs
(`niah_single_1..3`, `niah_multikey_1..3`, `niah_multiquery`, `niah_multivalue`, `vt`, `cwe`,
`fwe`, `qa_squad`, `qa_hotpot`) plus the `ruler` group and a `longcxt` tag. ✅ From its README,
verbatim:
> "1. A tokenizer is required for data processing… 2. **The default maximum sequence length is
> 4096.** For calculating metrics of different max seq lengths, specify additional lengths using
> the metadata parameter: `--metadata='{"max_seq_lengths":[4096,8192,16384,32768,65536,131072]}'`
> … 3. To prevent truncation of longer sequences, we recommend setting the max_length parameter in
> model_args: `--model_args=pretrained=...,max_length=32768`"

> ✅ **This resolves §3.1's open question decisively: the harness's RULER defaults to 4096 and
> lengths are a free parameter, so `--metadata='{"max_seq_lengths":[1024,2048,4096]}'` gives
> RULER-short in one command, on plain HF models, with no vLLM/TRT-LLM and no NVIDIA pipeline.**
> Remember the §3.1 caveat: sub-4K numbers are **not** comparable to the published leaderboard.

**BABILong in the harness** (`lm_eval/tasks/babilong/`) — all 20 tasks (`babilong_qa1..qa20`),
a `babilong` group, and a **`babilong_longctx` group covering exactly QA1–QA5**. ✅ From
`_babilong_common_yaml`:
```yaml
dataset_path: RMT-team/babilong-1k-samples
output_type: generate_until
num_fewshot: 2
generation_kwargs: {do_sample: false, temperature: 0.0, max_gen_toks: 16, until: []}
metric_list: [{metric: acc, aggregation: mean, higher_is_better: true}]
```
✅ **Note: `num_fewshot: 2`, greedy, `max_gen_toks: 16`** — cheap and base-model-friendly. And the
per-task YAML ships a **curated instruction + 2 hand-written few-shot examples**, e.g. for QA1:
> "I will give you context with the facts about positions of different persons hidden in some
> random text and a question. You need to answer the question based only on the information from
> the facts. If a person was in different locations, use the latest location to answer the
> question.\nAlways return your answer in the following format:\nThe most recent location of
> 'person' is 'location'. Do not write anything else after that."
with shots `"Charlie went to the hallway. Judith come back to the kitchen. Charlie travelled to
balcony." / "Where is Charlie?" → "The most recent location of Charlie is balcony."` ✅
The dataset is **`RMT-team/babilong-1k-samples`** — i.e. the **1k-token split**, which is exactly
the small-scale-friendly setting. ✅ (Other splits available on the Hub for other lengths.)

**Request types.** `loglikelihood` (score a continuation given a context — the CF/cloze path),
`loglikelihood_rolling` (whole-document perplexity — the Paloma path), `generate_until`
(free generation with stop sequences — the RULER/BABILong/phonebook path). ✅ (verified by the
`output_type` fields above and standard harness design.)

**`acc` vs `acc_norm`.** `acc_norm` divides the continuation log-likelihood by the continuation's
**byte length**, correcting the bias toward short answers. ⚠️ **This is the single largest source
of cross-paper incomparability at small scale** — OLMES pins normalization per task
(char / pmi / none; §6.3). **Report both, and state which.**

**Standard errors.** The harness reports a stderr per task by default. ⚠️ It is (for accuracy
metrics) the **analytic binomial/CLT SE over items**, i.e. exactly
`1.96·√(p(1−p)/N)`-style — **it does NOT capture seed variance, and it does NOT cluster.**
Per §6.2, our synthetic items *are* clustered, so harness stderrs will be **optimistically small
by up to 3×**. **Do not use harness stderr as your only error bar.** ✅ (inference from §6.2's
explicit warning + harness design)

**Pitfalls (consolidated).** ✅/⚠️
1. **Version drift changes scores.** Task YAMLs are versioned (`metadata: version:`) precisely
   because they change. **Pin the harness commit hash and record it in every result.**
2. **Prompt-format sensitivity.** OLMES documents that even `"Question:"` vs `"Q:"` and
   `"A."` vs `"(A)"` vs `"<mc>A</mc>"` vary across papers *and within a single paper*. ✅
3. **Tokenizer/BOS handling** and chat-template application differ by model backend; for base
   models, do **not** apply a chat template.
4. **Batch-size nondeterminism** — `--batch_size auto` can change results slightly via padding/
   numerics. **Fix the batch size for comparisons.**
5. **`--limit` subsampling** is convenient but shrinks N and inflates SE; per §6.2's power
   arithmetic you want ~1000 items.
6. See Biderman et al., *Lessons from the Trenches on Reproducible Evaluation of Language Models*
   (arXiv:2405.14782) — by the harness authors. ⚠️ **not read this pass; flagged.**

**Custom tasks** (needed for MQAR-in-vocab, phonebook, and PIP): register a YAML with
`output_type: generate_until` + a `!function utils.process_results` callback, exactly as
`babilong`/`ruler` do. The RULER task dir is the best template — it has `prepare_niah.py`,
`essays.py`, and per-task `*_utils.py`. ✅

### 7.2 OLMES / oe-eval / Catwalk (AI2)

- **OLMES** paper arXiv:2406.08446 (§6.3). Public code: **<https://github.com/allenai/olmes>**
  ⚠️ (repo name inferred from the paper/org convention — the paper says the standard is
  "documented, practical, open"; **verify the exact URL**). Also `allenai/oe-eval-internal` is
  referenced in AI2 work. ❓ **flagged.**
- **Catwalk** / `allenai/OLMo-Eval` — the latter definitely hosts Paloma at
  <https://github.com/allenai/OLMo-Eval/tree/main/paloma>. ✅
- **Value for us:** adopt OLMES's *formulation decisions* (5-shot curated, CF+MCF, per-task
  normalization) even if you run them through lm-eval-harness. That is the cheapest way to make
  our downstream numbers defensible.

### 7.3 HELM (Stanford CRFM)

<https://github.com/stanford-crfm/helm>. Scenario/adapter/metric architecture; HELM Classic and
HELM Lite. ⚠️ **Expensive** and oriented toward instruction-following API models. ❌ **Not
recommended at our scale** — the cost/benefit is poor versus lm-eval-harness + OLMES formatting.
❓ exact compute cost not verified.

### 7.4 Implementing the synthetic tasks — concrete plan

| Task | Source | Effort |
|---|---|---|
| **MQAR (architecture probe)** | `HazyResearch/zoology` | ✅ Drop-in. `pip install -e .` (skip `[extra]` to avoid `mamba_ssm`/`causal-conv1d` build pain — the README warns "The `mamba_ssm, conv1d` installs are often problematic"). Add our LIV block as a `sequence_mixer` module and use `zoology.mixers.hybrid.Hybrid` to interleave conv+attention exactly as the Based configs do. W&B logging is built in. |
| **AR-hit sliced perplexity** | Not packaged | ⚠️ **Write it (~100 LOC).** Needs: (a) an eval-loss pass returning **per-token** NLL, (b) a bigram-repeat mask over each eval sequence, (c) a **training-corpus bigram frequency table** with the ≤1250× threshold. (c) is the only real work — a counting pass over the training set. **Highest value-per-line of code in the whole suite.** |
| **RULER (short)** | lm-eval-harness `ruler` | ✅ Drop-in with `--metadata='{"max_seq_lengths":[1024,2048,4096]}'`. |
| **BABILong** | lm-eval-harness `babilong` / `babilong_longctx` + HF `RMT-team/babilong-1k-samples` | ✅ Drop-in, 2-shot, greedy. |
| **Phonebook** | ❓ none found | ⚠️ Write it (~30 LOC): names + 10-digit `NNN-NNN-NNNN`, 2 few-shot examples, exact match, sweep L. |
| **Passkey** | `epfml/landmark-attention` (Apache-2.0) | ⚠️ Template is 6 lines; easier to reimplement. **Raise trials to ≥500.** |
| **NIAH** | `needlehaystack` pip (MIT) | ⚠️ Use **v2 task=`single` (exact match)**; skip the GPT-4 judge. |
| **Lost-in-Middle KV** | <https://github.com/nelson-liu/lost-in-the-middle> ⚠️ (URL inferred; **verify**) | ⚠️ Trivial to reimplement; shorten UUIDs and use k ∈ {5,10,20,40} for 2k contexts. |
| **PIP (instruction persistence)** | novel (§5.3) | ⚠️ Write it (~80 LOC). |
| **A5 word problem** | <http://jpetty.org/ssm-illusion> | ✅ Cheap; the "min depth for 90 % acc vs length" protocol. |
| **IsTree** | Wen et al. 2402.18510 | ⚠️ ~0.5M-param models — the cheapest probe available. |
| **Paloma** | lm-eval-harness `paloma` + `allenai/OLMo-Eval` | ✅ Available; **fix vocab, report per-domain**. |
| **HELMET** | <https://github.com/princeton-nlp/HELMET> | ❌ Skip at our scale. |

### 7.5 Perplexity across tokenizers — the formula

If tokenizers differ, per-token perplexity is **not comparable** (different event spaces —
Paloma G4). Use **bits per byte**: ✅

**BPB = −ℓ / (B · ln 2)**  where ℓ = total log-likelihood of the corpus, B = number of **UTF-8
bytes**. Equivalently **BPB = (L_tok / L_bytes) · (NLL_per_token / ln 2)**.

Chunking, per Paloma G5 (verbatim): evaluate **documents individually, rather than packed into
concatenated maximum sequence length inputs. Documents longer than maximum sequence length are
split into disjoint inputs.** ✅ ⚠️ **Note this is *disjoint* chunking, not sliding-window.**
Sliding-window (stride) evaluation gives systematically *lower* perplexity because every token
gets more context — so **the context/stride policy must be identical across architectures or the
comparison is invalid.** This matters acutely for us: an all-attention model and a
short-conv-heavy model have different effective context, and a generous stride would flatter the
attention model. **Fix the policy, state it, and report both disjoint and strided if in doubt.**

---

## 8. Statistical rigor and the comparison protocol

### 8.1 What to report

Following Miller (2411.00640) and Madaan et al. (2406.10229): ✅
1. **Paired, item-level comparison.** Both architectures see **identical eval items**; compute
   per-item differences `s_{A−B,i}` and test their mean. This is strictly more powerful than
   comparing two summary scores.
2. **Clustered standard errors** wherever items share a generator/template/haystack — which is
   *every synthetic task in this document*. Naive SEs can be **3× too small**.
3. **Continuous metrics alongside accuracy** (answer probability mass / NLL / BPB) — a **2–18×**
   SNR gain (§6.2, Table 2).
4. **Multiple seeds** for the *training runs*, not just bootstrap over items. Bootstrap-over-items
   answers "would another sample of questions change the ranking?"; seeds answer "would another
   initialization change the ranking?" **You need both, and they are different numbers.**
5. **Explicit power analysis** stating the minimum detectable effect.
6. **A pinned harness commit and full formulation disclosure** (shots, normalization, prompt).

### 8.2 Power arithmetic — how many seeds?

Standard two-sample formula: **n = 2(z_{α/2} + z_β)² σ² / δ²** per arm, with α = 0.05
(z = 1.96), 80 % power (z_β = 0.84), so **(1.96 + 0.84)² = 7.84**.

Using Madaan et al.'s measured **7B seed σ** as the best available estimates (⚠️ our σ at
100M–1B is likely **larger**; treat these as optimistic lower bounds):

| Benchmark | seed σ | seeds for **δ = 2 pts** | seeds for **δ = 1 pt** |
|---|---|---|---|
| HellaSwag | 0.21 | 2·7.84·0.21²/4 = **0.17 → 1** | **0.7 → 1** |
| MMLU-Cloze | 0.22 | **0.19 → 1** | **0.76 → 1** |
| PIQA | 0.41 | **0.66 → 1** | 2·7.84·0.41²/1 = **2.6 → 3** |
| TriviaQA | 0.45 | **0.79 → 1** | **3.2 → 4** |
| ARC-C | 0.80 | 2·7.84·0.64/4 = **2.5 → 3** | **10.0 → 10** |
| MMLU (MC) | 0.57 | **1.3 → 2** | **5.1 → 6** |
| **COPA** | 2.15 | 2·7.84·4.62/4 = **18.1 → 19** | **72.5 → 73** |

**Readings:** ✅ (own arithmetic)
- To detect a **2-point** difference on HellaSwag / MMLU-Cloze / PIQA / TriviaQA, **1–3 seeds
  suffice.**
- To detect a **1-point** difference, you need **3–10 seeds** on the good benchmarks and
  **73 on COPA** — which is another way of saying **COPA cannot be used.**
- ⚠️ **These σ are from 7B models at 210B tokens.** At 100M–1B, seed variance is typically
  *larger* (scores are closer to chance, where the benchmark is less stable). **Inflate every
  seed count by ~2× as a planning margin, and measure your own σ from a 3-seed pilot before
  committing.** This pilot is mandatory, not optional.

**Item-count power (Miller Eq. 9):** ✅ **n = (z_{α/2}+z_β)²(ω² + σ²_A/K_A + σ²_B/K_B)/δ²**.
With the paper's illustrative ω² = 1/9, deterministic scoring (σ²_A = σ²_B = 0), δ = 0.03:
**n ≈ 969** ⇒ **design every synthetic task with ≥ 1000 items** if you want to resolve 3-point
differences. For our synthetics this is nearly free.

**Variance reduction by resampling:** K = 2 → variance ×2/3; K = 4 → ×1/2. ✅ Only helps for
*stochastic* decoding; with greedy decoding (which RULER/BABILong use) σ²_within = 0 and
resampling buys nothing. **Use greedy and spend the budget on items and seeds instead.**

### 8.3 Compute-matched vs param-matched vs token-matched

**All three, and say which.** ⚠️ A reviewer will attack whichever one you omit, because a
mostly-LIV hybrid and an all-attention model **cannot be simultaneously matched on all axes**:
short convs are cheaper per parameter than attention, so equal-params ⇒ unequal FLOPs, and equal
FLOPs ⇒ the hybrid gets more parameters.

**The Efficiency Misnomer** — Dehghani, Arnab, Beyer, Vaswani, Tay (ICLR 2022, arXiv:2110.12894).
✅ Verbatim: "researchers and practitioners often **assume that these metrics are correlated with
each other and report only few of them.** We … **demonstrate how incomplete reporting of cost
indicators can lead to partial conclusions and a blurred or incomplete picture**." **⇒ Report
parameters, FLOPs, training wall-clock, inference throughput, AND peak KV/state memory.** For our
architecture the last one is the whole point of the design, so it must be a first-class axis
(this is exactly Based's "state size in bytes" x-axis, §1.6).

**Recommended primary protocol: compute-matched (isoFLOP), with param-matched as a secondary
panel.** Chinchilla (Hoffmann et al., arXiv:2203.15556) gives the isoFLOP machinery: fix a set of
FLOP budgets, sweep (params, tokens) within each budget, and read off the minimum of each
isoFLOP curve. ⚠️ I did not re-verify Chinchilla's three approaches this pass — **flagged**, but
the method is standard.

⚠️ **Vocabulary is a confound in param-matched comparisons.** "Scaling Laws with Vocabulary"
(arXiv:2407.13623) argues the compute-optimal vocabulary size grows with model size; at 100M–1B
the embedding matrix is a *large fraction* of total parameters, so a param-matched comparison is
sensitive to vocab choice. **Fix the tokenizer and vocab across all arms, and report
non-embedding parameters separately** (Pythia does exactly this: models "marked as 'equivalent'
have the same architecture and number of **non-embedding** parameters" ✅).

### 8.4 Single points are not enough — you need scaling curves

**Language Models Scale Reliably with Over-Training and on Downstream Tasks** — Gadre et al.,
arXiv:2403.08540, <https://github.com/mlfoundations/scaling>. ✅ Verbatim:

> "we create a testbed of **104 models with 0.011B to 6.9B parameters** trained with various
> numbers of tokens on three data distributions. First, we fit scaling laws that extrapolate in
> both the amount of over-training and the number of model parameters. This enables us to predict
> the validation loss of a **1.4B parameter, 900B token run (i.e., 32× over-trained)** and a
> **6.9B parameter, 138B token run**—each from experiments that take **300× less compute**.
> Second, we relate the perplexity of a language model to its downstream task performance by
> proposing a **power law**. We use this law to predict **top-1 error averaged over downstream
> tasks** for the two aforementioned models, using experiments that take **20× less compute**."

**Three things this buys our design:** ✅
1. **It legitimizes small-scale extrapolation** — a scaling-curve argument from ≤1B models is a
   recognized methodology, not a limitation to apologize for.
2. **The over-training axis matters.** Models are trained well past Chinchilla-optimal in
   practice; an architecture comparison at Chinchilla-optimal may not hold at 32× over-trained.
   **Sweep the token/param ratio, not just the size.**
3. **The loss→downstream power law** is the principled way to connect our perplexity measurements
   to task accuracy — and it is also the tool that reveals **when an architecture breaks the
   relationship**. ⭐ **This is the sharpest framing available for our result:** if the mostly-LIV
   hybrid sits *on* the loss→downstream curve for commonsense tasks but *off* it for
   recall-intensive tasks, that is a clean, quantitative statement of "perplexity hides the
   retrieval failure" — much stronger than a table of raw scores. **Design the experiment to
   produce that plot.**

**Kaplan et al. (arXiv:2001.08361)** famously found loss is only weakly sensitive to
architectural shape at fixed parameter count. ⚠️ not re-verified; the implication stands: **a
single-point win is easily an artifact of hyperparameter luck.** ⇒ **at minimum 3 model sizes ×
3 seeds per architecture**, reporting the *curve*.

### 8.5 What a reviewer will demand — checklist

1. **Matched training data, order, and tokenizer** across all arms (Based: "Each model sees the
   same tokens of pretraining data in the same order" ✅; Paloma G2).
2. **Per-arm LR tuning**, disclosed. Zoology sweeps `np.logspace(-4,-2,4)` for *every*
   architecture — an *identical* grid is the defensible minimum. ⚠️ Never let one arm get more
   tuning; and note that Zoology's "max over sweep" is upward-biased (§1.3).
3. **≥ 3 model sizes** → a scaling curve, not a point (§8.4).
4. **≥ 3 seeds** at the primary size, with a measured-σ pilot (§8.2).
5. **Paired, clustered CIs** and a stated minimum detectable effect (§8.1–8.2).
6. **Multiple cost metrics** — params (total and non-embedding), FLOPs, wall-clock, throughput,
   state/KV memory (§8.3).
7. **Both a metric that should be insensitive** (commonsense average, aggregate ppl) **and metrics
   that should be sensitive** (AR-slice ppl, MQAR, phonebook). The *dissociation* is the result.
8. **Negative controls:** a task where theory predicts **no** difference (§4.4: sequential state
   tracking — all arms should fail) and one where it predicts a **large** difference (phonebook at
   large L). Passing both directions is what makes the claim credible.
9. **Pinned harness commit + full prompt/normalization disclosure** (§7.1).
10. **Length extrapolation** on every synthetic (§4.3.4 — otherwise a shortcut passes the test).

---

## 9. Recommended evaluation protocol

Design rules driving the tiering:
- **Average perplexity is nearly blind to the effect** (Zoology: 82 % of the gap in 6.4 % of
  tokens; Based: LM-Evals +0.67 while SWDE +12.8). **⇒ never decide on aggregate ppl alone.**
- **Retrieval and sequential state tracking are separate axes with separate fixes.** (§4.4)
- **Every synthetic must sweep interaction distance past the conv receptive field** and **test
  length extrapolation**, or a shortcut passes. (§4.3.4, §4.4)
- **Continuous metrics + paired clustered CIs** are a 2–18× free noise reduction. (§6.2)

### TIER 1 — cheap diagnostics, run on EVERY ablation
Target: **< 30 GPU-min per checkpoint.** All are programmatically scored, no judge.

| # | Eval | Config | Metric | Expected σ | Seeds for δ |
|---|---|---|---|---|---|
| 1.1 | **AR-hit sliced eval loss** ⭐ | held-out in-domain text; bigram-repeat mask; ≤1250× train-freq threshold | NLL on **AR-hit** vs **other** slice + % gap attributable | very low (continuous, millions of tokens) | **1 seed** detects <0.05 nats |
| 1.2 | **Aggregate + per-domain ppl** | Paloma G1–G5 discipline; fixed vocab; **disjoint** chunking | NLL / BPB per domain | very low | 1 |
| 1.3 | **MQAR** (zoology, arch probe) | vocab 8192; (N,D) ∈ {(64,4),(128,8),(256,16),(512,64)}; d_model {64,128,256,512}; LR grid `logspace(-4,-2,4)`; 2 layers; `random_non_queries=True` | per-cell query accuracy; **read off min d for >0.9** | low (10k+ query positions) | 2–3 |
| 1.4 | **MQAR gap-distance sweep** ⭐ | fix (N,D); sweep `power_a` 0.01 → 1.0 | accuracy vs mean interaction distance — **look for the conv-receptive-field cliff** | low | 2–3 |
| 1.5 | **JRT order + double-read** ⭐ | `num_passes` 1 vs 2; and [D,Q] vs [Q,D] | Δaccuracy — **separates "can't store" from "can't retrieve"** | low | 2–3 |
| 1.6 | **Phonebook** ⭐ | L ∈ {5,10,20,40,70,100,200}; 10-digit `NNN-NNN-NNNN`; 2 few-shot; ≥640 items/L | exact-match acc vs L | ~2 pts @ n=640 | 2–3 |
| 1.7 | **HellaSwag + PIQA + ARC-e/c** (CF/cloze, OLMES formatting) | 5-shot curated; report acc **and** answer prob. mass | acc + continuous | HSwag 0.21, PIQA 0.41, ARC-C 0.80 | 1–3 for δ=2 |
| 1.8 | **Passkey** (sanity) | fixed total length, random depth; **≥500 trials** | first-integer-in-100-tokens acc | 2.2 pts @ n=500 | 1 (regression check only) |

### TIER 2 — run on FINALISTS (2–4 surviving configs)
Target: **a few GPU-hours per model.**

| # | Eval | Config | Metric | Notes |
|---|---|---|---|---|
| 2.1 | **BABILong QA1–QA5** ⭐ | lm-eval `babilong_longctx`, `RMT-team/babilong-1k-samples`, 2-shot greedy; also the **0K** (no-distractor) condition | per-task acc | ✅ **Best long-context eval at this scale** — discriminates from **2K**; 130M–137M models solve it when trained on it |
| 2.2 | **RULER-short** | lm-eval `ruler` with `--metadata='{"max_seq_lengths":[1024,2048,4096]}'`; ≥200 ex/task | `string_match_all` / `string_match_part` per task | ⚠️ label **"reconfigured, not leaderboard-comparable"**; expect floors on CWE/FWE/VT/QA — **report per-task, never the 13-task average** |
| 2.3 | **Lost-in-Middle KV retrieval** | shortened UUIDs, k ∈ {5,10,20,40}; **sweep gold position across all k** | substring-match acc vs position | gives the **U-shape / positional bias** picture |
| 2.4 | **PIP — instruction persistence** ⭐ | §5.3: T1/T2/T3 × d ∈ {0,16,64,128,256,512,1024,2048}; 500 items/cell; decoy-rule condition | **normalized persistence P(d)** and **half-life d₅₀** | novel; the d=0 and no-instruction controls are mandatory |
| 2.5 | **A5 word problem** | Illusion protocol; min depth for 90 % acc vs length; A5 + A4×Z5 controls | min-depth curve | ⭐ **negative control** — theory says *all* arms fail; if one passes, something is wrong with the setup |
| 2.6 | **Verbatim copy** | random strings length L; sweep L past the conv receptive field | string-level acc vs L | tests Thm 2.3 / Cor 2.8 directly |
| 2.7 | **Extended commonsense** (OLMES 10-task, CF **and** MCF) | 5-shot curated; per-task normalization from OLMES Table 2 | acc + acc_norm + prob mass | ❌ **drop COPA (100 items, σ=2.15, CI=8.30)**; ⚠️ **WinoGrande is at chance <1B — report but don't decide on it** |
| 2.8 | **TriviaQA + SQuAD/`squad_completion`** | zero-shot | EM / F1 | Based's evidence: **25–70 %** at 360M ⇒ off both floor and ceiling; ⚠️ TriviaQA is *parametric*, not in-context, recall |

### TIER 3 — run ONCE at the largest scale
| # | Eval | Rationale |
|---|---|---|
| 3.1 | **Full RULER** at {4K, 8K, 16K}, 500 ex/task | Only if the largest model is ≥1B with ≥8K context. Reference points: **EXAONE-4.0-1.2B = 87.0 @ 4K**; threshold **85.6**; **Mamba-2.8B and RWKV-v5 are already below the 4K floor**. Report per-task. |
| 3.2 | **Length extrapolation** of every Tier-1/2 synthetic to 2×/4×/8× training length | ⭐ **Mandatory** — §4.3.4 says a shortcut otherwise passes every test. |
| 3.3 | **Scaling curves** — 3 sizes × 3 seeds, isoFLOP + param-matched panels; loss→downstream power law | §8.4. **The money plot: on-curve for commonsense, off-curve for recall.** |
| 3.4 | **Boxes / entity tracking** (Kim & Schuster, incl. implicit "move contents") | ⚠️ Expect floor; report the **"repeats initial state"** failure-signature rate rather than accuracy. |
| 3.5 | **BABILong at 4K/8K/16K, QA1–QA10** | full long-range picture incl. **QA7 counting** (LLM ceiling only 28 %). |
| 3.6 | **IFEval** | ⚠️ **only if an instruct/SFT variant exists.** Reference floors: **SmolLM2-135M-Instruct = 29.9**, **360M-Instruct = 41.0**. Base models will floor. |
| 3.7 | **Throughput / state-size Pareto** | Based's x-axis: recurrent-state bytes vs recall accuracy; plus prefill and generation throughput. This is the *point* of the architecture (§8.3). |

### Seed and noise budget (bottom line)
- **Pilot first: 3 seeds at the primary size**, measure your own σ per eval. Do not trust the 7B
  σ values as-is — **inflate by ~2×** for planning.
- **Tier 1 decisions (δ ≈ 2 pts): 3 seeds** suffices for HellaSwag/PIQA/MQAR/phonebook.
- **Headline claims (δ ≈ 1 pt): 5–10 seeds**, or switch to continuous metrics and get the same
  power with 3 (2–18× SNR gain).
- **Every synthetic: ≥ 1000 items** (Miller: n ≈ 969 for δ = 0.03 at 80 % power).
- **Always paired + clustered**; naive SEs are up to **3×** too small on template-generated items.

### The three plots that make the paper
1. **MQAR accuracy vs model dimension**, one panel per (N, D), all architectures — the Zoology
   Figure 2 format. Shows *how much dimension* our attention layers save.
2. **Phonebook / MQAR accuracy vs interaction distance**, with the conv receptive field marked —
   shows the cliff, and whether full attention removes Griffin's window ceiling.
3. **Downstream error vs loss** (Gadre et al. power law), with recall-intensive and
   non-recall tasks overlaid — shows the mostly-LIV hybrid **on** the curve for commonsense and
   **off** it for recall. This is the quantitative form of "average perplexity hides it."

### Open items to resolve before finalizing
1. ❓ **Heineman et al. "Signal and Noise" (AI2, 2025)** — arXiv id unconfirmed; it is the most
   directly relevant work for ranking benchmarks by SNR for small-scale ablation decisions.
2. ❓ **LFM2 Technical Report (arXiv:2511.23404) §6–9 + Appendix C** — could not retrieve; the
   phonebook question for LFM2 is unresolved (all other Liquid sources point to RULER).
3. ❓ **No published sub-1B RULER numbers** other than EXAONE-4.0-1.2B. We may have to produce
   the reference point ourselves.
4. ✅ **RESOLVED — repo URLs verified to exist (HTTP 200):**
   <https://github.com/allenai/olmes> · <https://github.com/allenai/OLMo-Eval> ·
   <https://github.com/nelson-liu/lost-in-the-middle> ·
   <https://github.com/sebschu/entity-tracking-lms> · <https://github.com/booydar/babilong> ·
   <https://github.com/princeton-nlp/HELMET>. ⚠️ Licenses not re-confirmed for these six (GitHub
   API rate limit) — check before redistributing derived data.
5. ⚠️ **Verify** the exact theorem in Sarrof et al. (2405.17394) before claiming an equality with
   star-free languages; and Grazzi et al. / DeltaNet statements.
6. ⚠️ **Measure the conv receptive field** of the actual LIV architecture (ℓ·k) — every distance
   sweep should be anchored to it.
