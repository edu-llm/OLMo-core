# factcrowd — does storing facts cost reasoning?

2026-08-03, revision 2. Build spec for the fact-crowding experiment. Everything needed to build it
and run it on the eduLLM platform is here.

Revision 2 responds to an adversarial review that ran seven independent reviewers against separate
failure axes. Most of it was right, and §16 records what changed, what I pushed back on, and what is
deferred. The short version: the axis was mis-defined, the manipulation was confounded with four
mechanisms that need no capacity at all, and the measurement had a quarter of the power it claimed.

---

## 1. The question

**Does storing facts consume model capacity that would otherwise serve reasoning?**

The programme was founded on that as a premise, and it has never been tested. It traces to two
unsupported sentences in the phi-3 technical report (`arXiv:2404.14219v1` — pin v1; the live v4
revises the numbers) — web pages are filtered out "to leave more model capacity for reasoning for the
mini size models," and "the model simply does not have the capacity to store too much factual
knowledge" — with no controlled ablation anywhere in the report. The report's own table contradicts
it: phi-3-small at 7B scores 59.1 on 5-shot TriviaQA against phi-3-mini at 3.8B scoring 64.0, while
MMLU rises 68.8 to 75.3. Nearly doubling parameters made factual recall worse. Microsoft dropped the
framing a generation later; phi-4 attributes the same deficit to hallucination and quantifies a
**token-budget** trade-off instead (Table 4: +6.9 TriviaQA, −0.7 MATH, −0.7 GSM8k, −4.3 HumanEval,
average 0.0), which is data competing for training tokens rather than facts competing for weights.

**Physics of Language Models 3.3** measures fact capacity exactly — ~2 bits/parameter at 1000
exposures per fact, ~1 bit at 100 — by summing autoregressive loss over exactly the knowledge tokens
and feeding it to a bit-complexity bound. It measures no reasoning. **Marek et al.**
(`arXiv:2605.26097`) have the saturation framing but state in their Limitations that they "rely on
proxies for measuring model capacity, such as model size and pretraining loss," and the competition
they demonstrate is old knowledge against new knowledge.

**Nobody has logged bit-counts and a controlled reasoning endpoint on the same checkpoints across a
swept oversubscription ratio.** That is what this does.

### The prior, sorted by what each paper manipulates

Revision 1 said "expect flat" and listed three papers. Sorting them by *manipulation* rather than by
conclusion reverses the reading, and this is the single most important correction in the review:

| Paper | Manipulates | Finds |
|---|---|---|
| Looped LMs (`2510.25741`), Mixture of Parrots, Johnston & Belrose | **architecture** at fixed load | dissociation — capacity and reasoning move independently |
| **`arXiv:2505.18091`** (NeurIPS'25 Spotlight), phi-4 Table 4 | **load** at fixed architecture | reasoning falls as fact load rises |

**This experiment manipulates load.** So "everything adjacent points the other way" does not survive
the sort — the papers that point away manipulate something else.

`arXiv:2505.18091` is the closest existing work to this design and revision 1 cited it only as a
variance anchor. It sweeps synthetic-biography-versus-web ratio × model size across Pythia 14M–6.9B,
finds a 2.09pp average zero-shot loss at ratio 0.3 on a 410M model, **attributes it to capacity
allocation, and explicitly rules out token displacement.** It must be cited as prior work, its design
contrasted with ours, and our contribution stated against it: it manipulates *ratio* and infers
capacity; we manipulate *bits* at fixed ratio and measure capacity directly.

One citation from revision 1 was wrong and is withdrawn: the "iGSM op15 46.3 → 70.7 with capacity
nearly unchanged" result is **not in the looped-LM paper**, which does not mention iGSM. That pair
traces to `arXiv:2506.18233` Table 1, whose ladder is **depth**, not width — and at depth 12, which
we fix, it is saturating. Cited for P5 it was evidence against.

### Design priority

**Correctness of the measurement, then adjustability, then speed.** Four reasoning endpoints have
produced uninterpretable nulls in this programme, every one an instrumentation bug: iGSM at chance
because the eval discarded the derivation and graded one mod-23 integer; a deduction eval *below* its
own 0.500 floor because truncated derivations parsed as wrong; reasoning-gym macro-averaging 14
families with floors from 0 to 0.5; two-hop composition at 2.3× the product of its parts, so it was
measuring fact access. And the previous sweep ran at 0.51% of the capacity ceiling.

---

## 2. Scope

**In.** Entity and fact corpus generation with exact bit accounting; two ways of sweeping the axis
(§3.1); related and unrelated reasoning slices; a width-scaled ladder at fixed depth; the Allen-Zhu
bit-counting probe; recall by generation and recognition; reasoning behind a code-enforced gate;
a pre-registered pooled regression and equivalence test; the reasoning-only control arm; a positive
control; one 32k BPE trained on the **mixture**; a config system where one file defines one cell.

**Out.** The split/masked arm — Experiment 4 owns it, though §7.5 keeps the seam open. Any retrieval
store. Distributed training beyond one node. A Zipf arm. An MoE control (P14 was withdrawn: Mixture
of Parrots fixes *active* parameters and grows total, so at equal total it predicts dense reasons
better — the prediction had the sign backwards, and MoE is out of scope anyway).

**Cut on evidence, not on cost.** iGSM: its cited parameter ladder appears in neither iGSM paper, and
the real from-scratch figure is op=15 at 99.1% for GPT-2 small — saturated. The **trained linear
probe**: over a 33,600-class date pool it is 8.6M parameters fit on 20k rows, ill-posed. The **Pythia
seed sweep** as a variance gate (§8.5). The **monotonicity re-run rule** (§8.5).

**Deferred.** Continuation from a pretrained checkpoint. If it happens it should be **Qwen3-0.6B**,
not Pythia: `olmo_core.nn.hf.convert` already supports qwen3, where GPT-NeoX needs a 600–900 line
port because OLMo-core has no parallel-residual block. Same science, no port.

---

## 3. The axis

**The independent variable is demanded fact bits per parameter.** Revision 1 used
ρ = demanded bits ÷ (R_E · P), which put an assumed constant on both sides of the equation: R_E
defined the x-axis *and* was the predicted outcome, so "the knee sits at ρ=1 by construction" was a
tautology and `CellSpec.check()` was comparing a quantity to itself. R_E is now used only for
*interpretation* — where to expect the knee — never for placing a cell.

Two reporting bases, because the literature is ambiguous and the choice matters:

| Basis | Value | Why |
|---|---|---|
| **non-embedding** | primary for cell placement | nothing in a tied embedding table is what Allen-Zhu's law is about |
| **total** | reported alongside | Physics 3.3 §9 says "P … total number of parameters," excluding only *unused* embedding rows |

At a tied 32k vocabulary the two differ by **1.650× at 13M falling to 1.217× at 113M — monotone in
model size**, so a design
that silently picks one loses cross-size comparability, which is what the size axis exists for.
Report both for every cell; `rho.py` takes the basis as an explicit argument.

**In this experiment the gap is much smaller than that, and the reason is worth stating.** §6.3 replaces
the 32k BPE with a *closed word-level vocabulary*, which comes out at 3,584 padded entries — so the tied
table is 0.9M parameters at 13M rather than 8.2M, and the real ratios are **1.073× / 1.049× / 1.032× /
1.024×**. The monotone shape survives; the magnitude does not. So the basis choice moves a cell's
reported x-coordinate by 7% at the bottom of the ladder and 2% at the top, not by 65%. Both are still
reported, because 7% is not nothing and because the ratio is still monotone in size — but the dual-basis
argument is a precaution here rather than the load-bearing correction it would be under a 32k BPE.

Two further corrections to demand accounting, both in `rho.py`:

- **The name term.** Physics 3.3's bioS demand is `N·[log2(N₀/N) + log2(S₀)]`, where N₀ is the size
  of the name universe. Revision 1 excluded names entirely on the grounds that a name is a key. The
  key/value distinction is right for *which pools carry values*, but knowing *that* a given name
  exists is itself information. The term is +16.4% of *attribute* bits at N = 714k and +9.8% at
  N = 6.4M — 14.1% and 8.9% of total demand — so on the count axis, where N varies, it changes the
  **shape** of the trend rather than just its offset. On the entropy axis, where N is fixed, it is a
  constant offset instead, which is why that axis's b=0 cell sits at 0.173 bits/param and not at zero — and without it achieved R(F) can exceed R^max,
  which Remark 4.2 forbids.
- **R_E is unknown to about 1.8×, not 8%.** There is no 200-exposure run in Physics 3.3; log
  interpolation between its anchors gives 1.30 and linear gives 1.11. Worse, the paper reports gated
  MLPs at **1.3× lower capacity** than GPT-2 at 100 exposures even with tuned LR — and `olmo2_*` is
  SwiGLU. Morris (`2505.24832`) reports 3.6 b/p. Plan for 0.9–1.3 and **measure it once in our own
  setup** before the grid places a cell (§12, M0).

### 3.1 Two ways to sweep it, and why we want both

**The count sweep (as designed).** Vary entity count at fixed exposures. Then tokens ∝ demand,
`corr(log2 demand, log2 tokens) = 0.9995`, VIF ≈ 1,076 — **the trend is not identified within a
row.** Concretely it is a **50× mixture-ratio sweep**: reasoning's share of the stream runs 38.6% at
13M/ρ=0.25 to 0.772% at 64M/ρ=4, and 12.7× within the 28M row alone. That is precisely the phi-4
token-budget effect this experiment exists to distinguish itself from, and "matched absolute
reasoning tokens" does not neutralise a ratio effect. Four mechanisms are monotone in demand and need
**no capacity at all**:

| Mechanism | Magnitude on the 28M row |
|---|---|
| **Adam second-moment dilution** — at high demand the fact slice sets `v` almost alone, so each reasoning gradient is divided by a denominator it barely contributed to | effective LR on reasoning directions falls with its gradient share |
| **Cumulative weight decay** | `exp(−Σ lr·wd)` runs 0.796 at ρ=0.25 to 0.055 at ρ=4 — **14.6×** |
| **Batch composition** | 64M/ρ=4 at a 1M-token batch delivers ~7.7k reasoning tokens/step ≈ 4 sequences |
| **Epoch asymmetry** | facts get 200 epochs, reasoning gets 1, in every cell |

**The entropy sweep (new, and the identified one).** Hold N, tokens, exposures, mixture ratio, steps,
schedule, weight decay and surface statistics *exactly* fixed. Sweep demand by **value-pool
entropy**: render every attribute value as exactly four sub-values drawn from a pool of size `2^(b/4)`,
so bits per attribute is `b` while the token count is invariant.

At 28M with N fixed at 611,184 — the ρ=1 entity count — every cell is 13.22B tokens and 0.57 h on
8×H100. Demand below **includes the name term**, which is a constant 0.173 bits/param here because N
is fixed:

| sub-pool | bits/attr | bits/entity | demanded b/p | ρ at R_E=1.2 |
|---|---|---|---|---|
| 1 | 0 | 0 | 0.173 | 0.14 |
| 2 | 4 | 24 | 0.691 | 0.58 |
| **4** | **8** | **48** | **1.209** | **1.01** |
| 16 | 16 | 96 | 2.244 | 1.87 |
| 64 | 24 | 144 | 3.280 | 2.73 |
| 256 | 32 | 192 | 4.315 | 3.60 |

**Both sweeps must be plotted on the same x-axis definition** — demand including the name term, on
the non-embedding basis. §3.1's "count-slope minus entropy-slope" subtraction is meaningless
otherwise, and an earlier draft of this document quoted the two tables on different definitions.

**The midpoint is bioS.** 6 × 8 = 48 bits/entity against bioS's 47.592 — a 0.9% match — so the axis
anchors to the literature exactly where the comparison is made. Six cells cost **2.4 h on 8×H100
(~$130)**, against 4.7 h for the 28M count row, and every one of the four confounds above is held
fixed by construction rather than argued away.

Two implementation notes. Values stay **natural**: four words from real word lists, so a 32-bit value
reads like "Northgate Rivermont Bellweather Ashcombe" rather than a random string — this is what
keeps §3.3's prose commitment. And holding **four** sub-values at every b, varying only the pool
size, is what makes the token count invariant; varying the number of sub-values would not.

**Keep both sweeps.** *Count-sweep slope minus entropy-sweep slope directly measures how much
"crowding" was tokens, steps and ratio.* That is a result in itself and it is the honest answer to
the referee who asks.

### 3.2 The first run

Per the cut decision: **omit the 113M row and the 64M ρ=4 cell.**

| | ρ=0.25 | ρ=0.5 | ρ=1 | ρ=2 | ρ=4 |
|---|---|---|---|---|---|
| **13M** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **28M** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **64M** | ✓ | ✓ | ✓ | ✓ | — |

Fourteen cells plus three reasoning-only controls, **187.2B tokens**. **15.1 h on 8×H100 ≈ $831**, or
48 h on 8×A100 ≈ $1,053 — roughly a third of the uncut grid. Largest single cell (`64m_d2p4`) 4.7 h,
comfortably inside `olmo-core-train`'s 24 h ceiling (§10).

**Submitted as three jobs, one per model size** (§10.3). Fully sequential the grid is 15.1 h; with
the three rows running concurrently the wall clock is the slowest row, **9.1 h**. The other gain is
independence: three approvals rather than one, and a failure in the 64M row does not strand the other
two.

Two notes on the cut. Dropping the 113M row is right for a reason better than cost: **width-scaling
cannot hold reasoning fixed** (§8.4), so that row could not do the job it was added for. And the
`P² → 81×` scaling in revision 1 is 81× only in non-embedding parameters; counting the FLOPs the
tied LM head actually performs it is ~53×, so small rows are relatively more expensive than they
looked.

**ρ=1 contributes exactly zero to a log-slope.** Leverage is `(x−x̄)²` = 4/1/0/1/4 across
log₂ρ ∈ {−2,−1,0,1,2}, and cost rises with ρ, so leverage-per-hour runs 16/2/0/0.5/1. Keep ρ=1 for
the hinge location and the bioS anchor, not for the slope — and note that the primary regression is
linear in demanded bits per parameter (§8.5), where ρ=1 is an ordinary interior point.

### 3.3 Corpus: four slices

**Facts.** Synthetic biographies in natural-language prose, bioS-style, ~100 tokens/bio, **≥20
templates per fact**. Prose over dense records because the representation has to match what we test.
Multi-template is mandatory: Physics 3.3 found diverse rendering does not hurt capacity and may help
it, and our own single-template corpus answered the same question at 83% under one phrasing and 1.3%
under another — the fact was stored as pattern-slot → value, not as (entity, attribute) → value.

**Exposures fixed at 200, uniform.** Fixing exposures is what decouples entity count from exposure
starvation; the previous sweep held tokens fixed while raising entity count, so exposures fell
196 → 49 → 12 and storage collapsed from 33.1 to 0.20 bits/entity with no way to attribute it.
Uniform rather than Zipfian because under Zipf at the same budget a large fraction of entities fall
below the exposure floor, demanded bits stop being computable, and the experiment loses its
independent variable. **The "~35-exposure floor" cited in revision 1 is unsourced** — it is in
neither Physics 3.3 nor 4.1, whose lowest useful-data exposure count is 100. The Zipf argument stands
on computability; the specific floor does not.

**Related reasoning.** Comparison over the same entities, covering a **fixed 25k-entity probe subset in
every cell** at constant absolute tokens and constant per-entity coverage. The probe must fit the
smallest cell — 13M at ρ=0.25 is **64,180** entities once the name term is counted, against 79,397
without it — so 25k leaves a **39k** non-probe comparison group there, still far above the n ≥ 2,000 the
eval needs. Built as `<compare>` (§8.3): given two people, the earlier of their birth years. Composition
and aggregation are specified but not built.

Its budget is **separate from and much smaller than** the unrelated slice's — 50M tokens against 1.0B —
and sized on per-entity coverage rather than on parity. The slice names two probe entities per item, so
at 1.0B each entity's birth-year rank would be supervised 4,211 times against the 200 exposures fixed
above: the slice would then teach the ranks outright, "needs two facts" would stop being true, and a
decline in its score would say nothing about fact *access*. 50M gives 211 mentions per entity, level with
the facts. One residual to report rather than fix: the slice injects ~216 kbit of fact demand outside the
demand axis (the 400-word birth-year pool has no ordinal structure, so its total order has to be stored),
constant in absolute bits across cells and absent from the control — so it shifts the x-axis rather than
tilting it.

**Unrelated reasoning.** The load-bearing measurement — see §8.3 for which endpoints, which changed.

**And a positive control, now mandatory.** Physics 3.3's **Result 11** says non-knowledge-dense
competitors do not interfere. If true, a semantics-free unrelated slice may be *incapable* of showing
interference **by the source paper's own finding**, and a null would be predicted regardless of
crowding. So one cell at ρ=2 carries a second, **disjoint bioS population** as the competitor. If a
second fact population does not crowd, the instrument cannot detect crowding at all and nothing else
in the grid means anything.

Relatedly, the domain token stays but its justification does not. The junk-data result is verbatim,
but the paper's junk is *unlearnable* (0.05–0.2 appearances per person) and **Fig 8(f) shows
structured, repetitive junk causing zero degradation** — which is the applicable panel, because our
reasoning slices are structured and learnable. Keep the token; it is harmless. Drop the "20×" claim.

### 3.4 Invariants

**Reasoning slices are constant in absolute tokens** and **identical across cells** — not merely
equal in volume. Revision 1 got this by materialising them once; the built version gets it by
generating each item from a fixed seed, which is stronger, because there is no file to drift and no
step that could shuffle two cells differently. Identity is of *items*, not of token ids: wherever the
schema differs the vocabulary differs, so on the entropy axis the same problems carry different ids.
A test asserts both halves.

**Every segment gets a domain token.**

**One frozen, checksummed evaluation set shared by every cell** (§8.5).

### 3.5 What revision 1 got wrong about the compute-matched reading

Revision 1 claimed the compute confound could be read around for free, by comparing the ρ=4 cell at
7.9% of training against the ρ=0.25 cell at 100%. **That is withdrawn.** Those two checkpoints sit at
**99.2% and 10% of peak learning rate**; the pre-cooldown loss deficit at this scale is 0.1–0.2 nats,
worth 4–16× in compute-equivalent — **5–20× the effect being hunted, with a sign pointing at
crowding.** The claim that it "costs nothing" was wrong.

The fix that does work: **WSD with an independent decay-to-zero branch from each compared
checkpoint**, ~1.5 h per row. Then both ends of the comparison are annealed and the reading is
legitimate. Budget it rather than assuming it.

---

## 4. Where the code lives

Branch `edullm/fact-crowding` of OLMo-core — the `edullm/**` name is **required**, because that is
what the image build workflow triggers on and a branch outside it publishes no image and warns
nobody. Experiment code under `src/scripts/train/factcrowd/`, tests under
`src/test/scripts/factcrowd/`, changes to `src/olmo_core/` only where genuinely needed. `mypy src/`
covers scripts; the image build runs `ruff check .` over the whole checkout, so ruff must be clean
repo-wide or there is no image.

```
src/scripts/train/factcrowd/
  README.md            how to run it; PRD.md is why
  PRD.md               this file
  train_cell.py        the platform entry point: one cell, one run
  cells.py             one cell, and everything derivable from it              [done]
  corpus/
    entities.py        entity table, closed pools, exact bit accounting          [done]
    values.py          attribute values as words; the bioS and entropy schemas   [done]
    vocab.py           the closed word-level vocabulary                          [done]
    render.py          entity -> biography over 32 templates; exact value spans  [done]
    stream.py          stream order, the token-offset index, token assembly      [done]
    tasks.py           the reasoning slices: Mano at L=10, related comparison    [done]
    source.py          the TokenSource adapters — 90 lines, all four methods     [done]
  ladder/
    rho.py             demand <-> entity count, both parameter bases            [done]
    sizes.py           width-scaled configs at fixed depth 12                   [done]
  measure/
    bits.py            Allen-Zhu estimator, with the name term
    recall.py          generation and recognition (post-hoc job, not a callback)
    reasoning.py       scorers; three counts and answer-token CE in bits
    registry.py        endpoint registration and gates G1-G8
  analysis/
    trend.py           pooled regression + the equivalence test
    figures.py
  configs/
    cells/count/       one YAML per cell; the grid is data, not code            [done]
    cells/entropy/     the identified axis                                      [done]
    cells/smoke/       the two-minute cell                                      [done]
```

Two modules revision 1 planned are absent by design. There is no `mixture.py`: OLMo-core's
`MixingInstanceSource` already mixes weighted sources, so `train_cell.py` converts the absolute token
targets of §3.4 into instance counts and hands them over — about twenty lines. And no `tokenize.py`:
§6.3 makes the tokenizer an assertion rather than a measurement, and a closed word-level vocabulary
discharges that assertion exactly instead of approximately, so there is no BPE to train. Likewise
`train/schedule.py` and `train/callbacks.py` — `WSD` and `ListCheckpointerCallback` are OLMo-core's.

---

## 5. Contracts

Unchanged from revision 1 except as noted: `EntityTable` (numpy index arrays, not `pa.Table` —
`pyarrow` is not a dependency), `SliceSpec`, `CellSpec`, `EndpointResult`, `Measurement`. See
`corpus/entities.py` and `ladder/rho.py` for the two that are built.

Three amendments:

```python
@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    model: ModelSpec
    sweep: Literal["count", "entropy"]   # NEW: which axis this cell is on
    demanded_bits_per_param: float       # NEW: replaces rho as the placed quantity
    bits_per_attribute: float            # NEW: b, for the entropy sweep
    exposures: int                       # 200 everywhere
    entity_table: EntityTableSpec
    reasoning: tuple[SliceSpec, ...]
    mask_spec: MaskSpec | None           # the Experiment-4 seam
    init_from: str | None


@dataclass(frozen=True)
class EndpointResult:
    correct: int
    incorrect: int
    unparseable: int                     # never folded into incorrect
    answer_token_ce_bits: float          # NEW: co-primary, continuous, parser-free
    graded_score: float | None
    degenerate_baseline: float           # measured, not assumed
    chance_floor: float
    n: int
```

`answer_token_ce_bits` is the important addition. It has no floor, no ceiling and no parser; it has
far lower variance than exact match; and — the point — it is **commensurate with the fact side**, so
reasoning and storage are finally measured in the same units.

`Measurement` gains `total_tokens`, `demanded_bits_per_param_total_basis`, and per-slice gradient-norm
and Adam second-moment statistics, so the mechanisms in §3.1 are *measured* rather than assumed away
(§16.2).

---

## 6. Corpus pipeline

### 6.1 Generation and rendering

`entities.py` produces an `EntityTable` from `(schema, n_entities, seed)`; pools are closed and
declared so `bits_per_entity` is arithmetic. The table is the only thing stored (~180 MB at 2.9M
entities); renders are generated from `(table, seed, exposure_idx)`. `InstanceSource` needs only
`__len__`, `__getitem__`, `num_tokens` and `fingerprint`, and `RandomInstanceSource` is the working
precedent for generating deterministically at access time.

**Render by concatenating precompiled token pieces.** Tokenize each template's fixed segments and
each pool value once at build time; a render is then a list concatenation. This makes the value-token
spans **exact by construction**, which is what `bits.py` requires, and removes the class of bug where
eval keys are built from a canonical name while prose uses a variant.

Two hard constraints from benchmarking:

- **Never construct a `np.random.Generator` per biography.** It costs 7.9 µs against 0.28 µs for
  splitmix64, which caps a worker at 13M tok/s before any rendering happens. `RandomInstanceSource`
  constructs one per *instance*, which is safe at its granularity and fatal at ours. A pure
  `splitmix64(seed, idx)` renderer benchmarks at **11.3M tok/s single-threaded, 41.7M at 8 workers**,
  against the 19.0M tok/s an 8-GPU node needs. Purity is free.
- One gate: assert `tokenize(prefix + value + suffix) == concat(tokenize(prefix), tokenize(value),
  tokenize(suffix))` across the full template × pool cross product. Failures fall back to real
  tokenization and are logged.

### 6.2 Exposure accounting

Every entity gets exactly **200 bio exposures**. The related slice adds a constant, measured number
of mentions for the 25k probe entities only. Both are recomputed from the rendered stream.

### 6.3 Tokenizer — an assertion, not a measurement

**A 32k BPE cannot be trained on biographies alone.** The bioS text has 265 word types (2,887 at
full-size pools), so the vocabulary saturates at 1,255 (7,509). A bio-only BPE inflates FLD by
**2.33×** (1,095 tokens/example, p99 2,138 → sequence length 2,560 → 6.25× attention cost).

**Adding 3% reasoning text to the BPE training corpus restores 0.94× GPT-2, and it is free.** So the
BPE is trained on the **mixture**, and `tokenize.py` asserts tokens-per-example ≤ 1.05× GPT-2 for
every slice rather than measuring it later and hoping.

Published as `tokenizer/factcrowd-bpe-32k/v1`. Note that OLMo-core's tokenizer map on `main` holds
dolma2 alone, so a custom tokenizer must be constructed in our own script — which `train_cell.py`
does — rather than named through the platform's dataset registry.

### 6.4 The bioS schema, and what is still provisional

Our reconstruction — four categorical pools at 200/300/100/263 plus a birth date over 12 × 28 × 400 —
gives 47.592 bits, pinning the published 47.6 to 0.01 of a bit. The review reports the real bioS set
as including gender and deriving working city from employer at 0 bits, reaching the same total by
another route; revision 1's stated pool list (which included a second 200) summed to 53–54 and was
wrong. Ours is arithmetically exact and is what we use; **the paper should be checked before
publication** since the comparison is the point.

**The attribute vocabulary is still placeholders and must be replaced before M1.** Bit accounting is
exact because it depends only on pool sizes, but `TOKENS_PER_BIO = 100` is unvalidated until the
strings are real, and every cell's token budget depends on it.

---

## 7. Training

### 7.1 The ladder

Fixed depth 12, sweeping `d_model`:

| Row | d_model | heads | d_ffn | non-emb | +tied 32k emb | N at ρ=1 |
|---|---|---|---|---|---|---|
| **13M** | 256 | 4 | 1024 | 12,595,456 | 20.8M | 266k |
| **28M** | 384 | 6 | 1536 | 28,330,368 | 40.6M | 611k |
| **64M** | 576 | 9 | 2304 | 63,729,216 | 82.2M | 1.41M |
| ~~113M~~ | 768 | 12 | 3072 | 113,283,840 | 137.9M | 2.54M |

The N column is the entity count at ρ=1 **with the name term included**, which is 11–16% below the
attribute-only figure an earlier draft quoted (318k / 714k / 1.61M / 2.86M). Since tokens scale with
N, that is also why the budget in §10 fell.

`d_model=256` at depth 12 is inside Physics 3.3's validated regime — their App. A lists GPT2-16-4
(d=256, 12.58M) as a "skinny and deep" anchor — and tying embeddings actively helps below 10M.

**Build through `llama_like`, never an `olmo2_*` factory.** `olmo2_190M` passes `d_model=768`
explicitly *and* splats `**kwargs`, popping `n_layers` and `n_heads` but not `d_model`, so supplying
a width raises `TypeError` for every row. The silent variant is worse: passing `n_heads` alone
succeeds and returns a 768-wide model with `head_dim=192` that trains and reports a loss curve.
`sizes.py` now calls `llama_like` directly and asserts the built non-embedding count.

### 7.2 Hyperparameters — specified, because they are not implementation details

Revision 1 named none of these. For an experiment whose result is a 2pp trend, and where cumulative
weight decay varies 14.6× across a row, they are part of the design.

| | Value | Note |
|---|---|---|
| sequence length | **512** | attention is +7.6% of FLOPs here against +30% at 2048 |
| global batch | **≥ 256k tokens** (512 sequences) | below this the small rows lose 20–25% MFU to batch shape |
| schedule | **WSD**, warmup **2% of each cell's own steps**, decay-to-zero over the last 10% | with independent decay branches per compared checkpoint (§3.5) |
| learning rate | tuned once at 28M, then **identical in every cell** | the invariant matters more than the value |
| weight decay | identical in every cell, and `Σ lr·wd` **logged per cell** | it varies 14.6× across a count row and cannot be argued away |
| intra-document masking | on (`generate_doc_lengths`) | otherwise packed biographies attend across each other and contaminate the bit count. Flash backends only; `pad == eos` explodes `max_docs`; no composable test coverage |

### 7.3 Checkpoints and packing

Ten snapshots, log-spaced at **0.5 / 1 / 2 / 4 / 8 / 16 / 32 / 50 / 75 / 100%**.
`ListCheckpointerCallback` already exists — but it inherits **`max_checkpoints=3`**, which deletes
7 of the 10, and on this platform the prune also deletes `.metadata.json`, a key the workload role is
denied by name, so the run dies with `OLMoNetworkError`. **`max_checkpoints=null` is mandatory** and
appears in §10's command.

**Packing was revision 1's largest unspecified task, and the entropy axis dissolves it.**
`InstanceSource.num_tokens ≡ len × sequence_length`, so ~100-token biographies at sequence length 512
give either 80% padding or a `TokenSource` with an O(1) global-offset → document map — and a
prefix-sum index over 1.29B documents is 10–20 GB, contradicting "never materialised." Under the
entropy axis every value is a fixed token count, so **biographies are fixed-length by construction**
and instance `i` simply holds biographies `[i·k, (i+1)·k)`. Packing becomes arithmetic. For the count
sweep, pad each biography to a fixed slot and log the padding fraction.

### 7.4 Data loading

Materialise the reasoning slices **once**: they are byte-identical in every cell, **4.0 GB at u32,
reused 17–29×**, a few core-hours offline. This is also the only way FLD would be affordable at all,
and it avoids 2.32 of the 2.41 TB the count grid would otherwise write. Facts stay on-the-fly.

Two traps: the composable mixer takes a **ratio** and `composable/utils.py:182-192` proportionally
rescales *every* slice if any source runs short, so pre-assert `num_tokens ≥ Nᵢ` per source; and
"uniform over the stream" comes from `shuffle_strategy=inter_source`, not from the mixer, which
concatenates. Always pass `dtype` explicitly — both `NumpyDatasetConfig.get_dtype` and the composable
`NumpyDocumentSource` fall back to the narrowest dtype the vocab fits and read a u32 corpus two bytes
at a time without raising.

### 7.5 The Experiment-4 seam

`Instance` is a TypedDict already carrying an optional `label_mask`, plumbed at `data_loader.py:486`.
`BioInstanceSource` emits one when `CellSpec.mask_spec` is set. One field, one branch, no sidecars.

---

## 8. Measurement

### 8.1 Bits

Allen-Zhu's estimator: **sum, never average**, the autoregressive loss over exactly the value tokens,
then the bit-complexity bound including the `1/ln2` conversion and the name term of §3. Per-token
loss extraction is first-class — `eval_batch` supports `loss_reduction="none"` →
`LMOutputWithLoss.ce_loss` — so the bit-counter is a callback. Bit-counting the largest remaining
cell is ~656M forward-pass tokens, 0.44% of budget.

Report **achieved** R(F) against **demanded**, plot reasoning against achieved, and log the
**per-entity distribution**. Assert `R ≤ R^max`.

### 8.2 Recall

Generation and recognition. **Generation cannot live in a training callback** — there is no precedent
in the repo and `TransformerGenerationModule.__init__` re-parallelises the model and mutates KV-cache
state — so it is a post-hoc job over saved checkpoints. The trained linear probe is cut (§2).

### 8.3 Reasoning endpoints — Mano promoted, FLD demoted

**Mano at L = 10 is the primary endpoint.** Revision 1 specified L = 13, where its degenerate
best-constant policy is **6.80%** (not 4.35%) and 13M–28M sits **1.4pp above degenerate** — failing
our own 20–80% gate. Physics 4.1 Fig 4 reports from scratch **at our exact 12 layers**:

| | 25.2M | 12L512D 37.7M | 56.6M | 12L768D 84.9M |
|---|---|---|---|---|
| **L=10** | 55.3 | **47.8** | 63.0 | **66.0** |
| L=13 | 8.2 | 19.4 | 30.6 | 36.7 |

In-band, +18.2pp across the parameter range, a 40.1pp task-difficulty dial, 16 tokens/example,
12,909 examples/sec/core, integer vocabulary so zero tokenizer risk. It is the only endpoint with
published from-scratch numbers at our configuration. Generator: `facebookresearch/PhysicsLM4`,
Apache-2.0.

**Mano is parametric** — it stores ~7.2 kbit of mod-23 tables in weights — so it is not a clean
unrelated probe alone. Pair it with **Brevo1 at N=110** from the same repo (39.9 → 53.2, 48.7pp dial,
verifier-scored, floor ≈ 0), which is *in-context*. Mano versus Brevo then separates "reasoning
crowded out" from "tables crowded out," which FLD cannot do. **Reasoning Core**
(`pip install reasoning-core`, MIT) is third: the only candidate whose own published run is a
random-init ~56M model on 0.5B + 0.5B tokens.

**FLD is conditional, not primary.** Its answer-accuracy floor is 33.3% only on the full split;
**every depth-stratified slice has a 51.1% floor** because all UNKNOWN examples carry `depth=None` —
which lands on the depth sweep, the test that proves the instrument responds, and is exactly the
"scored below its own floor" failure that killed a previous eval here. Three practical blockers too:
generation costs 5.69 core-seconds/example so 500M tokens is **~1,700 core-hours**;
`main`/`ICML_2023` does not run on Python ≥ 3.11 (21 `random.sample(<set>)` sites) while the NeurIPS
branch will not `pip install`; and `--depth-range` does not exist on the default NeurIPS_2024 HEAD.
Its 44.4/72.2 figures are arXiv v3 Appendix H, not PMLR Table 3's 37.7. Keep FLD only if the floor
and the core-hours are both resolved in M0.

**What is built.** Two slices, in `corpus/tasks.py`, both generated per example from a seed so they
hold nothing memorisable and cannot themselves compete for the capacity under measurement:

| | Width | Measured degenerate floor | Role |
|---|---|---|---|
| `<mano>` | 24 tokens at L=10 | **4.64%** against a 4.35% uniform baseline | primary, unrelated |
| `<compare>` | 19 tokens | **0.70%** over the 25k probe subset | related; needs two facts |

Both floors are the best of two policy families -- the best *constant* answer **and** the best *copy* of
a fixed span of the prompt. Searching constants alone is how an endpoint loses its floor: `<compare>`
originally answered with the earlier person's **name**, which is a span of its own prompt, so "always
name the first person" scored **50.2%** while the best constant name scored 0.02%. A binary endpoint
whose floor is quoted as 0% when it is 50% has half the dynamic range its admission gate assumes, and
any score under 50% is below its own floor -- which is the failure 1 lists for reasoning-gym. The task
now answers with the earlier **birth year**, a word that never appears in the prompt, so no copy policy
can reach it and the floor is 0.70%. The composition under test is unchanged: recall both years,
compare, emit the smaller.

The floors are measured, and measuring the first one changed the task. With a free choice of operand,
`× 0` is an absorbing state and the best constant answer was right **8.34%** of the time — worse than
the 6.80% this section cites for L=13, i.e. a weaker instrument than the one the section rejects.
Excluding zero as a multiplicand brings it to the uniform floor. This is the fourth instrumentation
bug of the kind §1 lists, and the only one caught before a run rather than after.

`<compare>` implements §3.3's related slice at its cheapest useful form — which of two people from the
fixed probe subset was born earlier. It is skipped on the entropy axis, whose six four-word composite
attributes contain no orderable field; a schema without one is refused rather than silently ordered on
an arbitrary sub-pool, so that axis carries `<mano>` alone. Composition and aggregation are not built.

**What the eval set has to avoid, when it is built.** Both tasks are generated from a seed, so an eval
set is a seed offset -- `+7` onward is unused (`+4` Mano, `+5` compare, `+6` the mixture). For Mano that
is enough on its own: the item space is ~2^54 and 1.0B tokens covers 2e-9 of it, so an eval item is new
with probability ~1. For `<compare>` the unit is the **unordered** pair, not the ordered one -- the
answer to "the earlier birth year of A and B" does not depend on the order, and the generator emits both
-- so there are 312M pairs over the 25,000-entity probe subset, not 625M. At the related slice's 50M
budget that is 2.63M items and **0.84% of eval pairs will have been seen in training**, which is small
enough to report and ignore. It was 14.1% at the 1.0B budget the slice originally carried, which was
not.

Brevo1 and Reasoning Core are the remaining gap in this section, and `<mano>` alone cannot separate
"reasoning crowded out" from "mod-23 tables crowded out" — that separation is the reason this section
asks for an in-context second endpoint, and until Brevo1 lands, a `<mano>` decline is ambiguous
between the two. `<compare>` versus `<mano>` separates a different pair (fact access versus capacity),
which is necessary but not the same contrast.

### 8.4 The contradiction at the centre, resolved

Revision 1 argued reasoning is **flat in width** and called that a feature, *and* predicted (P5) that
reasoning **improves with size**, "refuted if flat." Both cannot hold. **The flat-in-width half is
wrong**: the Mano anchor implies 16.1pp per parameter doubling (≈51pp across d=256→768) and moves
+18.2pp across our ladder at fixed depth 12.

So width-scaling does not hold reasoning fixed, the 113M row could not have broken the size confound,
and any width-swept axis needs the width response **measured and subtracted**. That measurement is
the reasoning-only control arm at **4 widths × 3 seeds, ~0.15 h**, which also returns a valid seed
distribution — and it should run **first**. If nothing brackets at 13M, delete that row.

### 8.5 Statistics — the measurement had 27% power, not 79%

Two errors stacked. The exact noncentral-t power at df=3 is 77.2%, not 79% (the docs paired a *t*
threshold with a *normal* statistic). The larger error is a category error: **the "0.5pp seed SD" is
eval sampling noise, not seed noise.** Both published anchors lie on exactly 21.27/√n
(0.21×√10,042 = 21.04; 2.15×√100 = 21.50) — the signature of a finite eval set — and the seed term
was never added. At p = 0.44 and n = 2,000 the binomial SE alone is 1.110pp, total SD 1.217pp,
**power 26.6%**. Eval noise is 83% of the variance. (A 3-seed pairwise test gives 98.9%, not "about
the same.")

Four fixes, all nearly free:

1. **n = 30,000 eval items**, one frozen checksummed set shared by every cell. Binomial SE → 0.287pp
   for 1.1B inference tokens, ~0.01% of budget. Freezing matters beyond the SE: because cell weights
   in a slope sum to zero, the shared item-difficulty term cancels **exactly**, worth another
   n ≈ 6,667 at item-correlation 0.7. "n ≥ 2,000" saved nothing and cost 2.4× in SD.
2. **Pool the rows** into one regression with size intercepts and replicates as observations rather
   than cell means. Power 26.5% → **67.7%** at the realistic SD; 77.2% → 99.97% at SD 0.5. Free — it
   is simply the right model. The regressor is **demanded bits per parameter**, linear, so the b=0
   cell is an interior point rather than a limit.
3. **Pre-register an equivalence test.** The expected result is a null, so this is an equivalence
   problem and revision 1 specified no equivalence test. With D = −4β and H₀: D ≥ 2.0pp, pooled at
   one seed with n = 30k gives **99.7% power to declare equivalence at 2pp** — turning the headline
   null from uninterpretable into publishable. Report the 90% CI on D and say "declines greater than
   X pp are excluded," never "no effect."
4. **Delete the monotonicity re-run rule.** It fires on **98.3% of rows by chance**, inflates type-I
   error from 5.0% to 16.7%, and shrinks the variance estimate to 0.57σ — which makes an equivalence
   test *falsely declare equivalence*. Replace with a pre-registered, outcome-blind rule.

Seeds are then keyed to a **measured** σ from §8.4's control arm: 1 if σ ≤ 0.99pp, 2 if ≤ 1.48,
3 if ≤ 1.85, re-scope above 1.9. Fix multiplicity with one named confirmatory test (uncorrected FWER
is 26.5% at k=6 rising to 84.2% at k=36). Drop P3's "3× the seed SD," whose implied α swings from
16.5% at n=2,000 to 2.0% at n=30k.

**M0 cannot be inference-only, and the Pythia seed sweep is cut.** With m=4 runs a "SD ≤ 1pp" gate
passes 27.9% of the time when the truth is 1.5pp, and an observed 0.5pp is compatible with 1.86pp.
Zero-shot Pythia-160M on FLD proof accuracy sits at the 0.0 floor where binomial SD is compressed
0.14× — biased *downward*. §8.3's own 20–80% rule disqualifies that instrument, and PolyPythias
itself excluded at-chance tasks from its seed analysis. The replacement is **replicate from-scratch
runs** (§8.4, §12).

### 8.6 The gate — resolution, not just dynamic range

Revision 1's gate tested whether an endpoint *responds*. It never tested whether it can *resolve* the
effect. From Allen-Zhu's own single-seed grid the worst parameter-order monotonicity violation is
median 27.1pp, max 56.0pp, with 8 of 12 rows ≥ 10pp; a one-seed 5-point trend needs run-level
σ ≤ 0.63pp to see 2pp at 80% power. **The design is 8–50× short and the old gate would not have
noticed** — and all four prior nulls are consistent with unmeasured σ.

| Gate | Requires |
|---|---|
| G1 | lower anchor (random init + best constant policy through the production parser), upper anchor, task-depth sweep; 20–80% of range; ≥15pp across depth |
| G2 | **label-permuted control** — a random-init score measures parser strictness, not the task |
| G3 | **hypothesis-only / premise-ablated probe** — FLD's own authors warn about this |
| G4 | headroom against the **achievable** ceiling from the b=0 arm, not an oracle range |
| G6 | **capacity responsiveness at fixed depth** — an endpoint flat in parameters cannot detect a capacity effect by construction |
| G7 | **resolution**: k ≥ 3 replicates, publish σ and MDE, require σ ≤ 0.65pp and unparseable ≤ 5% |
| G8 | a **calibrated** positive control by reasoning-token dilution (100/95/90/80/60%), finding the dose worth 2pp |

`grid.run()` raises on any endpoint that has not passed all of them. Bracket at **both mixture
extremes**, not only at no-facts.

---

## 9. Configs

One YAML per cell, fully resolved, no inheritance beyond a single `base.yaml`.

```yaml
# configs/cells/d384_b8.yaml    the bioS anchor on the entropy axis
extends: base.yaml
cell_id: d384_b8
sweep: entropy
model: {row: 28M, tie_word_embeddings: true, vocab: factcrowd-bpe-32k/v1}
bits_per_attribute: 8            # 4 sub-values from a pool of 4
entity_table: {schema: bioS, n_entities: 714_331, seed: 1234}
exposures: 200
reasoning:
  - {name: mano,    tokens: 500_000_000, length: 10}
  - {name: brevo1,  tokens: 250_000_000, n: 110}
  - {name: related, tokens: 250_000_000, probe_entities: 25_000}
init_from: null
```

On the entropy axis `n_entities` is **fixed** and `bits_per_attribute` is swept; on the count axis
`n_entities` is **derived** from `demanded_bits_per_param` by `rho.solve` and stating both is
refused. `CellSpec.check()` raises on any disagreement above 1%.

---

## 10. Compute and the platform

**There is no B200 and no H200.** Revision 1 budgeted in hardware this account cannot start. The
platform's dropdown tops out at `gpu-8xh100` ($55.04/hr) and `gpu-8xa100` ($21.96/hr); the largest
single card is `gpu-1xl40s` (48 GB). Anything at or above `gpu-8xa100` needs an **admin**, not a team
lead, because every profile over $20/hr routes that way.

| First run (§3.2), 14 cells + 3 controls, 187.2B tokens | 8×H100 | 8×A100 |
|---|---|---|
| wall clock, sequential | **15.1 h** | 48 h |
| wall clock, 3 jobs in parallel | **9.1 h** | 30 h |
| cost | **~$831** | ~$1,053 |

Per row: 13M 35.4B tokens / 1.3 h / $74, 28M 73.3B / 4.7 h / $259, 64M 78.5B / 9.1 h / $499. The 64M
row is three-fifths of the cost, so it is the one to re-budget after the first measurement. These are
FLOP estimates at 12/16/20% MFU per row, **not measurements** — read the real figure off the first
cell's first 50 steps.

At 20% MFU with the LM head counted. **Revision 1's FLOP count omitted the LM head**, which is
39/30/22% of the model across the rows — including it is +65/43/29% per row. And 8% MFU applied flat
across a 3× width span is structurally wrong: it inverts the cut decision by making small rows look
cheap. Plan on 12/16/20/24% per row and **measure MFU on M1's first 50 steps**, then re-budget.

The entropy sweep at 28M is **36.8B tokens, 2.4 h ≈ $130** for six cells. Its templates render 42
tokens per biography against the bioS axis's 69.2, so an estimate that reuses the bioS mean
overstates it by half — the figure above is measured from a dry run.

### 10.1 How a cell is submitted

One cell is one run. [Submit a run](https://github.com/edu-llm/platform/actions/workflows/submit-run.yml):

| Field | Value |
|---|---|
| `repository` | `OLMo-core` |
| `commit_sha` | full SHA of a commit on `edullm/fact-crowding` **that has a built image** |
| `workload_profile` | `olmo-core-check` for smoke, `olmo-core-train` for cells |
| `compute_profile` | `gpu-1xa10g` for smoke; `gpu-8xh100` or `gpu-8xa100` for cells |
| `team` | `memory-split` |
| `experiment` | `factcrowd-<milestone>` |
| `dataset_release` | **`none`** — the corpus is generated in-process |
| `wandb_project` | ours |

`dataset_release: none` is the quiet win of generating on the fly: nothing has to be published before
we can run, and the whole edullm-data airlock question moves out of the critical path.

**Nothing runs from source.** A commit becomes an image via the **Build eduLLM research image**
workflow, which fires on `edullm/**` and `main` only, takes 6–8 minutes, and is followed by a
registry scan of about 2 minutes. Submitting inside that window is refused as
`image_scan_findings_unreviewed`, which is usually not what happened. Budget 8–11 minutes from push
to submittable.

### 10.2 The command, and the settings it must carry

No shell wraps the command, so `bash -lc` is required for the environment variables to expand, and
the whole command plus environment must fit in 8,192 bytes.

```
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone
  src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID"
  --cell configs/cells/d384_b8.yaml
  --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

Every one of these is mandatory, and each is a run somebody has already lost:

| Setting | Value | Why |
|---|---|---|
| `--save-folder "$EDULLM_CHECKPOINT_DIR"` | literal on the command line | the platform reads the command text and cannot see inside the program; the OLMo-core default is `/tmp`, on a machine that stops existing — a 24 h run then exits zero having saved nothing and is **recorded as a success** |
| `checkpointer.max_checkpoints` | `null` | the prune deletes `.metadata.json` first and the workload role is denied that key by name → `OLMoNetworkError`, about an hour in. Also what would delete 7 of our 10 snapshots |
| `checkpointer.ephemeral_save_interval` | `null` | must be below `save_interval` or the config is refused in the first seconds |
| `lm_evaluator.enabled` | `false` | reads a C4 validation shard whose index was never published |
| `downstream_evaluator.enabled` | `false` | needs `ai2-olmo-eval`, which the image does not install |
| `trainer.max_duration` | set explicitly | defaults to one epoch |
| `--nproc-per-node` | = the device count | one process per device is **our** job; too few ranks is refused at submission, and used to bill for 8 while training on 1 |
| `train_module.compile_model` | `false` unless the image has a C compiler | Inductor dies without one |

A retry fires only for a **lost machine** — same run id, same checkpoint dir, `Trainer.fit()` resumes
itself. A crash in our own code exits, because the same traceback twice costs the budget twice.

### 10.3 Fan-out for the grid

The grid is 14 cells. `fanout_size: 14` with `fanout_index_parameter: cell` gives one submission,
priced and approved as one; each cell reads `AWS_BATCH_JOB_ARRAY_INDEX` and gets its own
`EDULLM_CHECKPOINT_DIR` with a `cell-<index>/` segment already in it. `train_cell.py` maps the index
to a config file. Concurrency is not settable.

---

## 11. The smoke run

Before any of the above. **Under $5 and under one hour auto-approves**, so this starts without a lead.

| Field | Value |
|---|---|
| `workload_profile` | `olmo-core-check` (1 h, 1 attempt, no checkpoint contract) |
| `compute_profile` | `gpu-1xa10g` ($1.01/hr) |
| `dataset_release` | `none` |
| `team` | `scratch` for the very first, `memory-split` after |

Four progressively less trivial smokes, each one able to fail cheaply. The first three have run
green; the configs are under `configs/cells/smoke/`.

1. **`--dry-run`** on `cpu-32vcpu`: builds the cell config, builds the model, asserts the
   non-embedding count, renders biographies, prints the token budget. Trains nothing. Catches every
   config and arithmetic error for cents. ✓
2. **A short run on `gpu-1xa10g`** at the 13M row on a small table: proves the data path feeds a GPU,
   the loss falls, checkpoints land in `$EDULLM_CHECKPOINT_DIR`, and W&B receives the run. ✓
3. **`smoke_13m_reason`**: facts and both reasoning slices through `MixingInstanceSource`, which is
   the path a real cell takes and the one the facts-only smokes never reach. Asserts the two reasoning
   slices land at equal absolute volume. ✓ (locally; not yet on the platform)
4. **`smoke_13m_ctrl`**: the reasoning-only control — no table, no renderer, no fact stream. A cell
   whose code had only ever been dry-run would be the one cell in the grid never executed. ✓
   (locally)

Then one bit-count and one Mano *eval* at the last step, which is still unbuilt (§8) and is where the
plan now has its most unexercised surface.

*Exit:* a checkpoint under `teams/scratch/runs/<run id>/checkpoints/`, a loss curve in W&B, a
`Measurement` row with a plausible achieved-bits number, and a measured MFU to re-budget from.

---

## 12. Milestones

**M0 — instruments. ~1 h GPU + CPU.** Reordered so the cheapest thing that can kill the design runs
first.

1. **The reasoning-only arm at 4 widths × 3 seeds**, ~0.15 h. Returns dReasoning/dWidth — settling
   §8.4 — and a real σ, which is what keys the seed count. Cheapest possible way to learn the grid is
   under-powered.
2. Bracket endpoints through **G1–G8 at both mixture extremes**. Retune Mano to L=10. Decide FLD
   in/out on the 51.1% floor and the 1,700 core-hours.
3. Train the BPE on bios **+ 3% reasoning text**; assert tokens/example ≤ 1.05× GPT-2.
4. Validate the bit estimator on planted bits, **including the name term and the 1/ln2 conversion**.
5. **Measure R_E** at exposures ∈ {50, 100, 200, 400, 1000} — one small model, one entity count.
6. Replace the placeholder vocabulary; re-derive tokens/bio and bits/entity.

*Exit:* a measured σ, a measured dReasoning/dWidth, a measured R_E, a bracketing report, and a
corrected demand table.

**M1 — one cell end to end, ~0.7 h.** 28M at b=8 (the bioS anchor) on WSD. Measure MFU and
re-budget. Then **3–6 replicates** to fix the seed count from data.

**M2 — the entropy sweep, ~2.4 h.** 28M, b ∈ {0, 4, 8, 16, 24, 32}. Frozen n=30k eval set,
pre-registered pooled regression and the 2pp equivalence test. *This is the identified axis and the
primary result.*

**M3 — the first run: the count grid, ~15.1 h in three jobs.** 13M/28M/64M minus 64M ρ=4, plus three controls, with
WSD cooldown branches so the compute-matched reading is legitimate. **M3 slope − M2 slope quantifies
how much of any "crowding" was tokens, steps and ratio.**

**M4 — placebos and mechanism.** Exposure placebo at fixed demand; the **disjoint-second-population
positive control** at ρ=2; domain-token-off; shuffled-value. The decisive cheap test: a post-hoc
reasoning-only fine-tune from the highest-b endpoint tracking recall and reasoning jointly — recovery
with recall intact means it was never capacity; recovery only as recall falls means it was, and
yields the bits↔accuracy exchange rate. Gradient-cosine, Fisher overlap and pruning curves are free
on saved checkpoints.

**M5 — scale.** Entropy axis at 64M, b ∈ {4, 8, 32}.

**Then decide** on Qwen3-0.6B continuation, the only arm addressing the parameter regime the phi-3
claim was actually about.

---

## 13. Testing

**Recompute, never trust.** Every corpus test recomputes from bytes.

Tests that encode specific past failures:

- key/prose surface agreement over 10k samples
- template × pool token-boundary agreement over the full cross product
- `unparseable` never folded into `incorrect`
- the bit-counter uses `sum` not `mean`, asserted on a planted-bits fixture, **and `R ≤ R^max`**
- every endpoint has a passing *and* a failing fixture for **each** of G1–G8
- `mixture.py` raises on a fractional reasoning slice, on a memorizable one, and when the composable
  mixer would rescale a short source
- `CellSpec.check()` raises when demand and `n_entities` disagree
- `sizes.py` raises when a built model misses its target by >1%, **and `build()` is exercised against
  real torch** — the factory-collision bug type-checked, passed a torch-free suite, and would have
  failed on the first GPU
- exposures recomputed from the stream equal exactly 200 for every entity
- **entropy-axis token invariance**: every b renders to an identical token count

---

## 14. Pre-registered predictions

Rewritten where revision 1 contradicted itself.

**P1.** *Measurement, not prediction.* Report where achieved R(F) saturates at 200 exposures, with a
CI. Revision 1's "pins at 2.0 ± 0.3, refuted below 1.5" imported the 1000-exposure frontier into a
200-exposure experiment while P9 said "closer to 1" and M2's exit said "~1.2" — **it was refuted by
construction.**
**P2.** Planted-bits recovery within 5%, and `R ≤ R^max` at every checkpoint. Revision 1's recall
hinge was an algebraic identity — under uniform residual `recall = R/R^max` exactly — so it could not
fail for the reason it was offered.
**P3.** On the **entropy axis**, reasoning is flat in demanded bits per parameter within the
equivalence bound. *Refuted if* the pooled slope's 90% CI excludes zero — the first identified
evidence for crowding.
**P4.** Related-reasoning accuracy declines past the hinge and tracks recall within 10 points.
**P5.** Reasoning improves with width at fixed demand, at the rate M0 measures. Revision 1 also
claimed reasoning is flat in width; that half is withdrawn (§8.4).
**P6.** The reasoning-only control shows no decline at any width.
**P7.** Recognition exceeds generation by ≥20 points at every demand level.
**P8.** Run-level σ from M0's replicates is ≤ 0.65pp. *Refuted if* larger, in which case the seed
count is re-scoped before M3 rather than after.
**P9.** Per-entity stored bits degrade roughly uniformly past saturation. *Refuted if* bimodal.
**P10.** **The count-sweep slope is more negative than the entropy-sweep slope.** The difference is
the token/step/ratio artifact, and predicting it is how we avoid claiming it as crowding.
**P11.** The disjoint-second-population control **does** crowd. *Refuted if* it does not, in which
case the instrument cannot detect crowding and nothing else in the grid is interpretable.

Effect size to power for is **~2pp**, from `arXiv:2505.18091` — which, note, is the paper that already
found the effect and attributed it to capacity.

---

## 15. What a result licenses

If reasoning is flat on the entropy axis with the count axis declining, the honest reading is that
the phi-3 conjecture fails at 13M–64M and the apparent effect is a data-mixture effect — which is
phi-4's own revised position and `2505.18091`'s rejected alternative.

If both decline together, that is the first identified evidence that fact storage and reasoning
compete for capacity, and it agrees with `2505.18091` by a different and better-controlled route.

**Neither licenses a claim about 7B**, and "Too Big to Think" (`2506.09099`) is a caution here: joint
training left *no* model able to extrapolate, which cuts against P5 and P6. Say all of this before a
referee says it.

Either way the experiment produces the first dataset with direct bit-counts and a bracketed reasoning
endpoint on the same checkpoints across a swept, **identified** demand axis.

---

## 16. What revision 2 changed, and what it did not

### 16.1 Adopted

The axis (demanded bits/parameter, both parameter bases, the name term, R_E measured not assumed);
the entropy sweep as the identified primary; the four named confounds; the withdrawal of the
compute-matched "costs nothing" claim; n=30k frozen eval, pooled regression, equivalence test, and
deletion of the monotonicity rule; Mano at L=10 with Brevo1 and Reasoning Core; answer-token CE in
bits; FLD demoted to conditional; iGSM, the linear probe and the Pythia seed sweep cut; gates G2–G8;
the mandatory positive control; specified hyperparameters; the LM head in the FLOP count; materialised
reasoning slices; `max_checkpoints=null`; generation moved to a post-hoc job; the `llama_like` fix;
the corrected prior; the withdrawn Ouro/iGSM citation; and `2505.18091` promoted to prior work.

### 16.2 Where I disagree, or where the fix is incomplete

**The entropy axis does not fully neutralise gradient-share variation, and the review's b=0 argument
overstates its case.** Fact-slice *loss magnitude* necessarily varies with b — that is the
manipulation — so Adam's second moment still sees a different fact-gradient scale in every cell. At
b=0 the fact loss collapses toward zero within a few thousand steps, so its gradient share collapses
too; that is a different regime, not a neutral one. The entropy axis fixes token share, step count,
schedule position and cumulative weight decay, which is four of the five, and it is a large
improvement. The residual is not arguable away, so it is **measured**: per-slice gradient norms and
Adam second-moment statistics are logged per cell (§5) and reported beside the slope.

**Total versus non-embedding parameters is not settled by the review's own evidence.** It cites
Physics 3.3 as saying "P = total number of parameters" but supports it with GPT2-small 124M → 88M,
which is total *minus* embeddings. The consistent reading is total minus *unused* embedding rows —
and for a 32k tied vocab with a BPE trained on our own corpus, almost every row is used. Hence:
report both, take the basis as an argument, and say which is which on every plot.

**`P² → 81×` was not wrong, it was a different quantity.** 81× is exact in non-embedding parameters;
~53× counts the LM head's FLOPs. Both are right under their own definition, and the review's framing
("the stated reason for cutting the 113M row is the wrong reason") is fair for the cost argument but
the row is cut on §8.4's grounds anyway.

**The entropy axis is not free to build.** Cheaper per row, yes, but it needs `values.py`, a
fixed-length renderer, and a natural-language scheme for high-entropy values. Call it 3–5
person-days, against a review estimate of 28–42 person-days for corpus plus measurement overall.

### 16.3 What a second adversarial pass found in revision 2's own code

Reviewers were run against `ladder/rho.py` and `corpus/entities.py` after they were written. Two real
bugs in the demand arithmetic, both now fixed and both pinned by tests:

**`solve()` bisected a function it never checked was monotone.** The derivative
`bits_per_entity + log2(N₀/N) − 1/ln2` is minimised at `N = N₀`, so **monotonicity requires
`bits_per_entity > 1/ln2 = 1.442695`**, and the docstring's "positive for any sane value" hid a hard
threshold. Below it demand rises, peaks near `N₀/e` and falls, and the bisection then both returns
non-closest answers and *refuses reachable targets* — at 1.0 bits/entity against a 160M name space,
a quarter of the range was broken. Any real schema clears it by an order of magnitude, but a
one-attribute debug schema would not.

**`solve()` silently clamped to one entity.** The bracket starts at `N = 1`, so the `n_entities < 1`
guard was unreachable on the name-term path: a demand of 1e-9 bits/param came back as one entity at
2.6e-6, a cell 2,600× off its own label, from the function documented as the only sanctioned way to
get an entity count. The linear path raised for the same input, so the two paths disagreed about
whether it was an error. `solve()` now verifies what the bisection achieved against a tolerance that
defaults to `check()`'s, which makes `check(solve(...))` hold by construction.

Also corrected: the basis ratio is **1.650×** at 13M, not 1.67× (the test asserted 1.650 while two
docstrings and this PRD said 1.67); `check()`'s error path could raise `solve()`'s error instead of
its own, losing the label and both numbers; `name_space` now has **no default**, because defaulting
it was the exact disagreement revision 2 exists to prevent; and "+16% of demand" is +16.4% of
*attribute* bits, 14.1% of total.

Six test weaknesses were found by mutation and closed. `demand()` could drop the name term entirely
with the suite green — shifting every reported x-coordinate 16.4% — because the decomposition
assertion is satisfied by `0 + a == a` and the basis test only checked a ratio. `EXPOSURES`,
`TOKENS_PER_BIO`, `R_E_BAND` and `capacity_bits` were never pinned: the token-budget test put
`EXPOSURES * TOKENS_PER_BIO` on both sides of its own equation, so the budget could change 37%
undetected, and `capacity_bits` returning `params / r_e` passed.

And the review recomputed three tables here that had been left on the attribute-only definition — the
N-at-ρ=1 column, the smallest cell's entity count, and the entropy sweep's demand column. All three
are corrected above. The most consequential was the last: the two sweeps were quoted on
differently-defined x-axes, which would have made §3.1's "count-slope minus entropy-slope"
subtraction meaningless.

### 16.4 What building the reasoning slices changed

**The reasoning budget is per slice, not per cell.** Revision 2 wrote "1.0B per cell" without saying
which, and the first implementation split a fixed total across whatever slices a cell carried. That gave
the reasoning-only control -- which carries only the unrelated slice -- **twice** the unrelated-slice
exposure of every cell it is the reference for, so its Mano score would have beaten theirs for a reason
with nothing to do with fact load. The control is the instrument for the single comparison this
experiment exists to make, and a fixed total quietly broke it. A cell's reasoning total is therefore
1.0B on the control and the entropy axis and 2.0B on a count-axis cell, and that asymmetry is the
correct consequence of the invariant rather than a violation of it.

**Mano excludes zero as a multiplicand.** With a free choice of operand, `× 0` is an absorbing state and
the best-constant policy scored **8.34%** -- worse than the 6.80% §8.3 cites as its reason for rejecting
L=13. The endpoint promoted for having a clean floor had a dirtier one than the endpoint it replaced,
and only measuring it showed that. Excluding zero brings it to 4.59% against a 4.35% uniform baseline.

**The control ran a different learning-rate schedule from every cell it anchors.** `warmup_steps` was
2,000 across the grid, on the principle that hyperparameters are held constant. For warmup that does the
opposite of what it intends: run length varies 37x, so 2,000 steps was 1.4% of the largest cell and
**52% of the control**. The control would have consumed its reasoning tokens at a mean 0.69 of peak
learning rate against 0.94 for the cell it is compared with -- systematically under-optimised, biasing
the crowding measurement toward a null. This is §3.5's withdrawn error reintroduced structurally rather
than by choice of checkpoint. Warmup is now a **fraction** (2%), so the schedule *shape* is identical
across cells, and a warmup that would consume a whole run is refused.

**The related slice's floor was 50%, quoted as 0.035%.** `degenerate_answer` searched constant answers
only. `<compare>`'s answer was the earlier person's *name* -- a span of its own prompt -- so "always name
the first person" is right **50.2%** of the time, needing no facts, no ordering and no arithmetic. The
best constant name managed 0.02%: a factor of 1,400. A binary endpoint whose floor is quoted as 0% when
it is 50% has half the dynamic range §8.6's admission band assumes, and any score under 50% is below its
own floor -- the reasoning-gym failure §1 lists. Two fixes: the task now answers with the earlier **birth
year**, a word absent from the prompt, dropping the floor to 0.70%; and `degenerate_baseline` searches
constant *and* copy families, taking a copy policy as the winner only when it beats the best constant by
more than three standard errors (the maximum over ~20 offsets is otherwise upward-biased by noise).

**The related slice out-supervised the facts it depends on by 21x.** At the unrelated slice's 1.0B it
would have named each of the 25,000 probe entities 4,211 times against the fact slice's 200 exposures,
teaching the birth-year ranks outright -- so "needs two facts to answer" would no longer have been true
and a decline would have said nothing about fact *access*. It also injects ~216 kbit of fact demand
outside the demand axis, present in every demand cell and absent from the control. The slice now has its
own budget, `related_reasoning_tokens` = 50M, sized to 211 mentions per entity. The uncounted demand term
remains, constant in absolute bits across cells, so it shifts the x-axis rather than tilting it.

**The zero-exclusion re-randomisation was a dead expression.** `1 + (residues[i] + position) % 22`
inside a branch that only fires when `residues[i] == 0` is the constant `1 + position`, so operator
position 0 always got `<n1>` and position 4 always `<n5>`, doubling the mass on one position-specific
operand. The floor survived it; the uniformity the comment claimed did not. The replacement is now
redrawn from the item's own mixer.

**`CompareTask.fingerprint` collided across tasks generating different items.** It hashed the schema, the
probe subset's *size* and the task seed, but neither the table's seed nor the probe ids -- so two tasks
over different entities shared a digest. That digest keys the cached instance index and is what a
"checkpoint versus corpus" audit compares.

**A repeated domain token is refused again.** An earlier fix let the vocabulary deduplicate a word
repeated within one role, which it must, because the task words and the biography literals both want
"and" and the caller concatenates those lists. Applied to `domain_tokens` too, it silently accepted two
slices sharing one label -- the labels that make the endpoints separable in the first place. The
deduplication is now narrowed to `literal_words`.

**The control's entity count is stated, not solved.** `rho.solve` refuses a zero target and keeps
refusing it: its linear path divides by `bits_per_entity` and its name-term path falls below the
monotonicity threshold of §16.3, so an answer would be an accident of the bracket. `name_bits`,
`demanded_bits` and `demand` accept zero entities and return exactly zero -- the limit written down
rather than evaluated -- and demand 0 with no reasoning tokens is refused as a run with no data.

### 16.5 Deferred, with reasons

FLD's 1,700 core-hours and 51.1% floor — decided in M0, not now. The exposure placebo and the
mechanism battery — M4. Qwen3-0.6B continuation — after M3. Retiring
`memory-split/docs/HANDOFF-factcrowd-dev.md`, which still points a fresh developer at a superseded
spec that disagrees with this one on architecture, materialisation, model sizes and corpus size: that
is a change in another repository and belongs to whoever owns it. **This file is the single source of
truth.**
