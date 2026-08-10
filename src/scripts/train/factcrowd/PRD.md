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
MMLU rises 68.8 to 75.3. Nearly doubling parameters made factual recall worse — though the two models
differ in recipe, data and architecture, so this is a **non-monotonicity the report leaves unexplained,
not a controlled refutation**. What is missing either way is the ablation, which is the gap this fills. Microsoft dropped the
framing a generation later; phi-4 attributes the same deficit to hallucination and quantifies a
**token-budget** trade-off instead (Table 4: +6.9 TriviaQA, −0.7 MATH, −0.7 GSM8k, −4.3 HumanEval,
average 0.0), which is data competing for training tokens rather than facts competing for weights.

**Physics of Language Models 3.3** measures fact capacity exactly — ~2 bits/parameter at 1000
exposures per fact, ~1 bit at 100 — by summing autoregressive loss over exactly the knowledge tokens
and feeding it to a bit-complexity bound. It measures no reasoning. **Marek et al.**
(`arXiv:2605.26097`) have the saturation framing but state in their Limitations that they "rely on
proxies for measuring model capacity, such as model size and pretraining loss," and the competition
they demonstrate is old knowledge against new knowledge.

Predecessors exist and must be cited rather than waved past: *Scaling Laws for Fact Memorization*
(`2406.15720`), *In-Tool Learning* (`2508.20755`), *Too Big to Think* (`2506.09099`), and on the
measurement side *Geometric Factual Recall* (`2605.12426`) and *Engram* (`2601.07372`). None combines
the three things this design needs at once, so the defensible claim is narrow and should be stated in
exactly this form:

> To our knowledge, the first fixed-architecture experiment varying persistent factual entropy at fixed
> entity count, exposure and token budget, while measuring a source-faithful factual-information lower
> bound and train-disjoint unrelated reasoning on the same checkpoints.

That sentence is true only once the defects in §16.5 are fixed; "nobody has ever tested this" is not,
and was too broad.

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
finds a 2.09pp average zero-shot loss at ratio 0.3 on a 410M model and attributes it to capacity
allocation. **That attribution is disputed and this PRD no longer leans on it:** an independent reading
of the same paper reports SynBio recall staying near zero while web tokens are displaced, which would
make it evidence that a mixture treatment can hurt *without any facts being stored*. Either way it is
the closest existing design and must be cited as prior work — read as a prior for stored-fact crowding
only if the recall figures support it. Our contribution stands against it regardless: it manipulates
*ratio* and infers capacity; we manipulate *bits* at fixed ratio and measure stored information
directly.

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
  SwiGLU. Morris (`2505.24832`) reports 3.6 b/p but measures **unintended memorisation of random
  token sequences**, not bio-fact storage at 200 exposures, so it is an upper anchor of a different
  quantity rather than a competing estimate of ours. Plan for 0.9–1.3 and **measure it once in our own
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

**One vocabulary across the sweep, and it is load-bearing.** Sizing each cell's pools at `2^(b/4)` made
the vocabulary a function of the treatment: **1,920 padded tokens at b=0 against 8,064 at b=32**, so the
high-entropy cell carried 8.1% more trainable parameters and a 4.2× wider softmax than the cell it is
compared against. Those biases run in opposite directions — more parameters can hide crowding, a wider
softmax can manufacture a reasoning decline — so the axis identified nothing. Every cell now shares one
union pool of `2^(32/4) = 256` words per slot and differs only in how many are reachable, giving
identical vocabulary, padded size, parameter count and token ids at 0 → 192 bits/entity.

One residual, stated: marginal token frequencies still differ across cells, because fewer reachable
values means each is emitted more often. That is inseparable from "vary the entropy" in any pool-based
scheme. The stronger design keeps all 256 words active at matched marginal frequency and varies only
*joint* entropy through correlated four-word codewords; it is specified here and not built.

**The midpoint is bioS.** 6 × 8 = 48 bits/entity against bioS's 47.592 — a 0.9% match — so the axis
anchors to the literature exactly where the comparison is made. Six cells cost **2.3 h on 8×H100
(~$127)**, against 4.6 h for the 28M count row, and every one of the four confounds above is held
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

Fourteen cells plus three reasoning-only controls, **182.8B tokens**. **14.7 h on 8×H100 ≈ $812**, or
47 h on 8×A100 ≈ $1,029 — roughly a third of the uncut grid. Largest single cell (`64m_d2p4`) 4.5 h,
comfortably inside `olmo-core-train`'s 24 h ceiling (§10).

**Submitted as three jobs, one per model size** (§10.3). Fully sequential the grid is 14.7 h; with
the three rows running concurrently the wall clock is the slowest row, **8.8 h**. The other gain is
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
| data parallelism | **FSDP**, bf16 parameters, **fp32 reductions** | `parallelize_model` gates every wrapper behind `if dp_config is not None`, so without one eight ranks train eight unsynchronised models on an eighth of the corpus each — at ~25 exposures, not 200. Nothing raises, and single-process smokes cannot see it |
| gradient clipping | **1.0** | an unclipped spike in a 150,000-step cell nobody is watching changes the achieved-bits curve this experiment reads |
| compilation | on with a GPU, off on CPU | the image carries a C compiler; compiling a CPU smoke costs more than it saves |
| model init seed | varies with the **replicate**, never with the corpus | `TransformerConfig.init_seed` defaults to 0 and was never set, so every cell and every notional replicate initialised the same network. A shared eval set reduces measurement noise; it does not create trained-model replicates |
| intra-document masking | **off, and this is a known deviation** | The PRD asked for `generate_doc_lengths`. Turning it on is not sufficient: documents render as `[domain, BOS, …, EOS]` while the detector wants adjacent `[EOS, BOS]`, and the packed-input path needs a Flash backend the CPU profile does not have. Left off deliberately, with the consequence stated in §7.3 |

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

**Two packing consequences, stated rather than left to be discovered.** Neither is a between-cell
confound — the streams are byte-identical across cells, so both are a uniform tax — but both bound what
the endpoints can show.

*Reasoning items are cut by instance boundaries.* `TaskStream` rounds down to whole items, which
protects the end of the stream and nothing else: the trainer requests 512-token windows and neither 24
nor 19 divides 512, so **3.1% of unrelated items and 3.5% of related ones** straddle a boundary, some
of them mid-answer. The eval must therefore generate items standalone and locate answers itself rather
than trusting `answer_start`, which is valid only before chunking. **No sequence length fixes this**:
504 is 24×21 but not a multiple of 19, so an earlier draft's "504-token fix" was arithmetically wrong.
Padding both tasks to 32 tokens with a label mask is the fix, and it is deferred rather than rushed in
ahead of the first run.

*Intra-document masking is off.* §7.2 records why: `[domain, BOS, …, EOS]` does not present the adjacent
`[EOS, BOS]` the detector looks for, and the packed path needs a Flash backend. So a biography can
attend to its predecessor's tokens. That inflates what the model can predict from context rather than
from weights, which means the achieved-bits estimate is an **upper** bound on stored information, and
the reported figure should say so until masking is on.


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

**Built, and the layering is in `README.md`.** `measure/` implements this section; §16.6 records what
building it corrected. What remains unbuilt is the gates' *evidence* — a label-permuted control, a
premise-ablated probe, a dilution ladder — and a gate whose evidence is missing returns false rather than
passing silently.

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

**THE IMPLEMENTED TASK IS NOT THE PUBLISHED MANO, AND ITS CALIBRATION DOES NOT TRANSFER.** This is the
most important correction in this section. Published Mano builds recursive prefix trees over `+`, `−`
and `×`, trains at variable lengths and evaluates at exact length. What `corpus/tasks.py` builds is a
flat infix expression of ten operands over `+` and `×` only, evaluated strictly left to right, at one
fixed training length, with zero excluded as a multiplicand. Ten *operands*, not ten operations.

So the 47.8 → 66.0 from-scratch figures above are **not** predictions for this task, and neither is its
task-memory estimate. They describe a harder generator. Two ways out, and the second is the plan:
vendor the upstream generator with golden fixtures, or keep the custom task and calibrate it
independently. Until that calibration exists the admission gate has to be measured on our own task, and
the paper's numbers may be cited only as the reason for choosing *a* mod-23 arithmetic task at this
depth — never as its expected accuracy. The class should be renamed before anything is written up; it is
still called `ManoTask` in this revision, which is a naming debt, and the docstring says so.

Even a faithful Mano stores its modular tables in weights, so an unrelated-reasoning claim cannot rest
on it alone — which is the same reason this section asks for an in-context second endpoint.

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

| First run (§3.2), 14 cells + 3 controls, 182.8B tokens | 8×H100 | 8×A100 |
|---|---|---|
| wall clock, sequential | **14.7 h** | 47 h |
| wall clock, 3 jobs in parallel | **8.8 h** | 29 h |
| cost | **~$812** | ~$1,029 |

Per row: 13M 34.7B tokens / 1.3 h / $73, 28M 71.5B / 4.6 h / $252, 64M 76.6B / 8.8 h / $487. The 64M
row is three-fifths of the cost, so it is the one to re-budget after the first measurement. These are
FLOP estimates at 12/16/20% MFU per row, **not measurements** — read the real figure off the first
cell's first 50 steps.

At 20% MFU with the LM head counted. **Revision 1's FLOP count omitted the LM head**, which is
39/30/22% of the model across the rows — including it is +65/43/29% per row. And 8% MFU applied flat
across a 3× width span is structurally wrong: it inverts the cut decision by making small rows look
cheap. Plan on 12/16/20/24% per row and **measure MFU on M1's first 50 steps**, then re-budget.

The entropy sweep at 28M is **36.0B tokens, 2.3 h ≈ $127** for six cells. Its templates render 42
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

**M2 — the entropy sweep, ~2.3 h.** 28M, b ∈ {0, 4, 8, 16, 24, 32}. Frozen n=30k eval set,
pre-registered pooled regression and the 2pp equivalence test. *This is the identified axis and the
primary result.*

**M3 — the first run: the count grid, ~14.7 h in three jobs.** 13M/28M/64M minus 64M ρ=4, plus three controls, with
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

### 16.5 What an independent expert review found, and what it changed

An external reviewer was given the executable system and returned a stop-ship verdict. Most of it was
right, and four findings were defects that would have invalidated every headline outcome. All four are
fixed, each pinned by a test that fails without the fix.

**Eight ranks trained eight independent models.** `train_cell.py` started `torchrun --nproc-per-node=8`
and supplied no `dp_config`, and `parallelize_model` gates FSDP and DDP behind `if dp_config is not
None`. So no wrapper was applied, gradients were never reduced, and the loader still sharded by rank:
eight models, an eighth of the corpus each, **~25 exposures instead of 200**. Nothing raised, and every
smoke run in this repo passed, because world size one is the only size at which the omission is
invisible. Now FSDP with bf16 parameters and fp32 reductions, gradient clipping at 1.0, and compilation
on a GPU — matching the platform reference. A two-rank test asserts the loss curve matches the
single-rank curve step for step, which unsynchronised gradients cannot do.

**The entropy sweep varied the model.** Pools sized per cell at `2^(b/4)` made the vocabulary a function
of the treatment: 1,920 padded tokens at b=0 against 8,064 at b=32, i.e. **8.1% more parameters and a
4.2× wider softmax on the high-entropy arm**, with the two biases running opposite ways. Fixed with one
union pool per slot and a reachable prefix per cell; §3.1 records the residual and the stronger design.

**The unrelated-reasoning eval was 100% leaked.** Items were keyed on `index ^ seed`, which makes the
item *set* independent of the seed: training at 1238 and evaluating at 1241 differ by 15, so eval item
`i` was training item `i ^ 15`. Reproduced at 2,000/2,000. Keying is now framed and domain-separated over
`(class, split, seed, index)` in two rounds, so the dependence on the seed is not a translation and
disjointness holds even at an identical seed — checked against 60,000 training items.

**Replicates would have shared one network.** `TransformerConfig.init_seed` defaults to 0 and was never
set, so varying the cell seed varied the corpus and not the initialisation. A `replicate` field now
drives initialisation and data order while the corpus, the reasoning items and their volumes stay fixed,
which is what makes a set of replicates a paired block. A shared eval set reduces measurement noise; it
never created trained-model replicates, and the review is right that the advertised power figures refer
to an obsolete grid.

Also corrected: the name term is now the exact `log2 C(N0, N)` rather than the `N·log2(N0/N)` proxy
Physics 3.3 writes, which understates it by 15.2% at 611k entities and shifted every cell's
x-coordinate; `solve`'s bracket is capped at `N0/2`, where the exact term is monotone for *any* positive
bits per entity, which removes the `1/ln2` schema threshold entirely. §8.3 now states plainly that the
implemented task is **not** the published Mano and that its calibration does not transfer. §7.2 and §7.3
record the FSDP, clipping, masking and item-splitting positions. §1's phi-3, `2505.18091`, Morris and
novelty claims are all narrowed.

**Where I disagree, or judged differently.**

*Item splitting is a tax, not a confound.* 3.1% of unrelated and 3.5% of related items straddle a
512-token boundary, and the review reads that as corruption exceeding the target effect size. It is
identical across cells — the streams are byte-identical — so it cannot bias a between-cell comparison;
it lowers the endpoint's ceiling uniformly. It also does not touch the measurement, because the eval
generates items standalone. Worth fixing by padding both tasks to 32 tokens with label masks; not worth blocking
a run for. Same for intra-document masking, with the consequence recorded in §7.3: the achieved-bits
figure is an upper bound until masking is on.

*The measurement layer does not block training.* It did when checkpoints recorded nothing about the cell
that produced them — that was a real defect and is fixed: every checkpoint now carries the cell config
plus schema, vocabulary, stream and task fingerprints, so a scorer written later can replay the exact
corpus and prove it. What remains is that the scorer is unwritten, which delays the *result*, not the
*runs*.

*Non-embedding parameters stay primary for placement.* The review is right that Physics 3.3 says total
used parameters, and that is now stated. But with the closed 3,584-word vocabulary the two bases differ
by 1.073× at 13M falling to 1.024× at 113M, so the choice moves a cell 2–7% and changes no conclusion —
and both are reported per cell. Re-placing the grid a third time for that is churn.

*The count sweep is already secondary.* §3.1 says the subtraction is not a causal decomposition, and the
review's stronger phrasing is adopted there rather than treated as new.

**Accepted and not yet done.** The four-arm storage-specific pilot (low-information, stable, balanced
deterministic, per-exposure resampled) and the ephemeral-mapping control; the three-seed 18-run design as
the primary rather than a single seed; a TOST interval instead of the one-sided rule, which is
non-inferiority and is now called that; the hinge sensitivity; disjoint auxiliary identities for the
related comparator with open-book, closed-book, constituent-recall and conditional reporting; a renamed
and independently calibrated arithmetic task; a Flash-backed packed path with masking; the checkpoint →
reconstruction → scoring smoke. §12 orders them.

### 16.6 What building the measurement half changed

`score_run.py` and `measure/` now exist: a checkpoint goes in, one row per (cell, replicate, step,
endpoint) comes out. Verified end to end on real checkpoints. Six things this section got wrong or left
underspecified, found by building it.

**The estimator needed one place to own the off-by-one.** OLMo-core builds labels as
`pad(input_ids[..., 1:])`, so `ce_loss[t]` scores `input_ids[t+1]` and the cost of token `p` is
`ce_loss[p-1]`. §8.1 says "sum the loss over exactly the value tokens" without saying which loss
positions those are, and getting it backwards is invisible — a bit count would charge a value token's
cost to the literal before it, and an endpoint would grade the token before the answer. Both produce
plausible numbers. `measure/spans.py` holds the rule, checked against a manually computed cross-entropy
from a real model, and every caller goes through it.

**The parse-status problem is structurally absent, not merely handled.** §8.3's endpoints both render a
single-token answer at a known position, so grading is a teacher-forced argmax — *identical* to greedy
decoding, with no continuation to truncate and no string to parse. That removes the failure mode §1
catalogues four times. `n_unparseable` is still reported, because G7 bounds it and a future multi-token
endpoint could reintroduce the problem.

**Achieved bits are an upper bound, and the report has to say so.** With intra-document masking off
(§7.3) a packed biography can attend to its neighbour, so part of what looks stored was read from
context. Every row carries `bits_is_upper_bound`, and a figure above the ~2 bits/param ceiling is
refused as a **measurement fault** rather than reported as a finding. Storage is also clamped at zero:
an untrained checkpoint's residual exceeds the prior — measured at 83–91 bits against a 47.59-bit prior
— and that is not negative storage.

**Rebuilding the corpus has to be verified, not assumed.** The corpus is generated, so scoring replays
the cell recorded beside the weights. §8 did not say to check that the replay is right. It is now, and
the guard has already refused a real checkpoint: one written before §3.1's union vocabulary rebuilds to a
different schema today, and scoring it would have produced entirely reasonable numbers about a corpus the
model never saw.

**§8.5's pooled regression is the wrong inferential unit, and the size of the error is now measured.**
It asks for one regression over all cells with size intercepts. That treats correlated observations as
independent: on the planned 3×6 design the cell-level standard error comes out **2.83× smaller** than
the blocked one, and its 90% interval declares equivalence — `[−0.74, +0.74]pp` — where the per-seed
interval `[−3.49, +3.49]pp` cannot, on identical data. `analysis/trend.py` uses the per-seed slope, runs
per ladder row, and provides both `tost` and `non_inferiority` so a null states which test it passed.
The margin is also end-to-end rather than per bit: §8.5 defines the effect as `D = −4β`, so reading 2pp
per bit/param would be permissive by 4.14×.

**G4 cannot be a two-sided band.** Under P3 the predicted result is every cell sitting *at* the b=0
arm's score, so a saturation check against that ceiling would refuse the pre-registered outcome for
being predicted. `measure/gates.py` checks the ceiling instead: the achievable range must be ≥10pp and a
cell must not out-score the no-facts arm. Three thresholds the PRD does not name (G3's 15pp drop, G4's
10pp range, G6's 2pp rise) are derived from ones it does and are labelled in their docstrings as the
module's choice. G7's MDE reconstructs to σ = 0.636pp for a 2pp effect at five points, matching this
section's 0.63pp — and only with a one-sided α, which is now stated. **G5 remains absent**, with a test
asserting it stays that way.

### 16.7 What an adversarial pass found in the measurement layer

Eight defects, all demonstrated by running code, all fixed. It also cleared the three things most worth
worrying about: `spans.predictor_slice` is correct and **every production caller uses the right
convention** (the off-by-one bugs were in the tests, not the code); padding is never charged, verified over
13,286 spans; and the bits prior corresponds exactly to the charged spans on both axes, including the
entropy axis's use of the *reachable* pool size rather than the union's.

**The regressor column was blank on the identified axis.** `collect` built its identity from the cell
record alone, and `CellSpec.to_dict()` drops `None` — but the two sweeps state disjoint halves of that
block: the count axis states a demand and derives an entity count, the entropy axis the reverse. So
`demand_bits_per_param`, which *is* `trend.SeedBlock.demands`, arrived empty for every entropy cell. The
resolved record carries both and was being discarded. It is now read first, which also replaces the
solver's target with the achieved value from the integer entity count.

**Recall reported an untrained model as 4.15× above chance.** Three defects compounding: recognition
concatenated an attribute's pools and took one argmax over the union, so a word only pool 2 contains could
beat the truth at position 1; unreachable words from the entropy axis's 256-word union competed despite
never being trained; and the pooled chance was `1/mean(n)` instead of `mean(1/n)`, understating it 3.8× on
bioS. Recognition is now restricted per position to that position's reachable pool, and chance is the
product over positions — measured 1.01× on the same checkpoint, i.e. at chance, and exactly 2⁻ᵇ on the
entropy axis.

**The bit estimator's headline disagreed with its own distribution.** `stored` was `max(0, prior −
mean(residual))` while the per-entity figures were `max(0, prior − residualᵢ)`. Those differ by Jensen
whenever any entity's residual exceeds the prior, which is the norm early in training. The headline said
"stored nothing" while the distribution §8.1 asks for said half the entities held 36 bits each. Now
per-entity throughout, which is the right unit: an entity the model has learned nothing about contributes
zero, not a negative offset against the ones it has.

**TOST declared equivalence from zero variance.** A dead endpoint (`n_correct = 0` everywhere) or a
saturated one gives identical integer counts across seeds, so the between-seed SD is exactly zero, every
t is infinite, and both one-sided nulls are rejected with p = 0 — the pre-registered headline arriving at
maximum strength from an instrument that measured nothing. A frozen shared eval set makes this *more*
likely, not less. Both tests now withhold the verdict below a degenerate standard error and say why,
while still reporting the interval and both p-values.

**Fingerprint verification missed the two things most worth checking.** Schema and vocabulary are blind to
the expression length and the renderer seed: a rebuild with `MANO_LENGTH = 13` or a different renderer
seed passed both digests while changing every item scored and every span charged. The recorded task
digest could not be used because it bakes in the split, so tasks now also publish a split-independent
`structure_fingerprint`, and the renderer publishes its own. Both are recorded and checked, and an
endpoint present at training but absent from the rebuild is refused.

**One checkpoint-less sibling aborted the whole fan-out.** `find_checkpoints` raises rather than returning
empty on a missing directory, so a `logs/`, a `wandb/`, or a cell that died before its first save took
scoring down for every other cell.

Also: `score_reasoning(floor=…)` was ignored unless `degenerate_answer` was passed too, silently
re-measuring the floor once per endpoint per checkpoint. And an `IndexError` when the model argmaxes into
the vocabulary padding, which the count axis had been lucky enough to miss.

**Seven silent mutation survivors, now all caught.** The bit-count and answer-CE span tests used a
*constant* loss, under which a one-position shift is arithmetically invisible; both now use a varying row.
`measure/recall` had no test importing it at all. `bits.score_checkpoint` and `checkpoint.load` were
untested drivers, so a doubled prior and a `split="train"` rebuild both passed the suite.

### 16.8 What a third expert audit changed, and where I judged differently

An independent audit of the measurement commit returned eighteen findings. Most were right. The four
that mattered most were not about arithmetic:

**The gates were unreachable.** §8.6 says a row is admitted only on gate evidence, `gates.py` implements
G1–G8, and `score_run` refuses to mark a row confirmatory without a report — but *nothing could produce
a report*. G8 in particular needs the same cell trained at 100/95/90/80/60% of its reasoning tokens, and
no committed config could express that. The design's central safeguard was a document sentence with an
unsatisfiable precondition: the only way to admit a row was to hand-write the JSON, which is asserting
the gates passed rather than measuring it.

Fixed by building both halves. `cells.dilution_ladder_cells` generates the ladder — five cells, 4.25B
tokens, 1.4 slot-hours, about $8, the cheapest item in the entire design — and `measure/evidence.py`
assembles a report from scored runs, recognising the ladder by cell id, the ceiling and the width sweep
from the controls, and σ from the most-replicated cell. Four gates (G4, G6, G7, G8) are now feedable from
configs that exist. **G1, G2 and G3 are not**, and they report as owed rather than being defaulted to
pass, so no row is admitted yet. That is the honest state and it is now visible in a file rather than
inferable from a docs cross-reference.

Two properties are load-bearing and tested. A report assembled from a run does not admit that run — the
gate cells would be admitting themselves. And an empty report does not pass: `GateReport.passed` requires
a non-empty result set, because "no gates run" must never read as "all gates passed".

**World size was about to become a second treatment.** On A10G, `28m_d4p8` needs eight devices (25.4 h on
four, past the 24 h per-cell cap) while the other five 28M cells fit on four. Only the top cell differs,
so world size would have correlated almost perfectly with demand and any effect of it would have landed
directly on the row's slope. FSDP holds the global batch fixed and a 2-rank curve reproduces a 1-rank one
exactly here, which is evidence but not a guarantee across 83,000 steps of bf16 reductions. The rule is
now explicit — one world size per confirmatory row — and the 28M row goes entirely to 8×A10G at
27.2 h and about $442, roughly $77 and half a day more than the mixed plan. Worth it: the 28M row also
carries the entropy sweep, and a confounded top cell there attacks the M3 − M2 subtraction that separates
crowding from tokens-and-steps.

**The run order was backwards.** The README handed a reader the count grid first. With admission
code-enforced, that produces 170 checkpoints of correctly-labelled descriptive data. M0's σ block and the
dilution ladder together cost 4.7 h and $53 — 7.5% of the A10G plan's $705 — and until they run nothing
downstream can be claimed rather than plotted. They now come first, then the entropy sweep, which is the
identified axis and costs a third of the 28M count row.

**Where I judged differently.** The audit proposed a sequence length of 504 to stop chunking truncating
~3% of reasoning items, on the grounds that it is a multiple of both task widths. It is not: 504 is 24×21
and the `<compare>` items are 19 tokens wide, which divides nothing here. No sequence length fixes this,
so the claim is removed rather than acted on; padding both tasks to 32 tokens with label masks is the
actual fix and remains deferred.

The audit also asked for a second switch separating the fact stream from the task streams in
`BuiltCorpus`. The measured problem was real — `with_streams=False` still built an offset index over
billions of tokens, 4.7 s against 0.3 s per checkpoint — but a second switch is dead surface: scoring
reads `tasks` and `renderer`, which are built either way, and no caller wants one volume without the
other. One switch now covers both, and a mutation test confirms it: the version with two switches passed
its own test with the fix reverted, because both flags were false at the only call site.

One further defect surfaced while checking the audit's cost figures rather than from the audit itself.
Every ceiling in the README is a `maximum_attempts=1` figure. The `gh` commands do pass that explicitly,
but the platform default is higher, and at two attempts the two 8xA10G submissions cross the $500 routine
bound ($652 and $554) and need admin release. Stated where the ceilings are, since a reader comparing them against a rejected
submission would otherwise have no way to see why.

**And one defect the audit's own fix introduced.** Renaming the recall columns to ``template_*`` -- the
audit's point, and correct, since this measures template reconstruction and not closed-book recall --
mangled every one of them. Three layers each handled the prefix: ``RecallResult.summary`` namespaced its
keys, ``score_run`` stripped a hardcoded ``recall_``, and ``collect`` added ``recall_`` back. They agreed
until one was renamed, after which ``"template_all_chance"[len("recall_"):]`` is ``e_all_chance`` and the
column shipped as ``recall_e_all_chance``. The bit and reasoning columns were unaffected; only the
template block was.

The names now have one owner and neither downstream layer renames. The more useful lesson is about the
test suite rather than the code: nothing faster than a 27-minute end-to-end run had ever looked at a
column name, so a rename could mangle the output and every fast test still passed. Two fast tests now
assert the exact column set through ``ScoredCheckpoint.rows()``, and they assert the set rather than a
pattern -- a test accepting "something containing chance" would have passed on the mangled names too.

### 16.9 M0 ran, and G4 refused the endpoint

The gate report exists. `run_019fdf85` scored the sigma block, the dilution ladder and the round-two
re-runs in seven minutes and wrote `gates-mano.json`. It does not pass, and the reason is not one of the
three unbuilt gates:

```
G8: ladder on row 13M with doses [100, 95, 90, 80, 60] (complete)
G6: controls at 3 width(s) [12595456, 28330368, 63729216], 3 replicates each
G4: ceiling from the row-13M control, 4.44% (mean of 3)
G7: 3 replicates of 13m_ctrl
does not pass: G1, G2, G3, G4, G6, G8
```

**The `<mano>` endpoint has no dynamic range.** Its degenerate floor -- the best constant policy, always
answering `<n16>` -- is **4.695%**. The reasoning-only control, which carries no facts and has every
parameter the ladder can give it, reaches **4.44%**: *at* the floor. (The 4.695% is the best-constant policy over the item
generator; the same policy on the frozen 30,000-item set is about 4.62%, so the control sits within the
baseline's own uncertainty of it rather than provably beneath it -- see §16.10.) The achievable range is
**~0 pp** where G4 requires 10 pp and the design needs to resolve 2 pp.

Three failures follow from that one fact rather than being independent:

- **G4** -- floor-to-ceiling is negative.
- **G6** -- accuracy cannot rise with width when every width sits at the floor.
- **G8** -- the ladder is *complete* at all five doses and still cannot produce a 2 pp decline, because
  there is nothing above the floor to decline from.

**G7 passes, and its pass is the most misleading number in the report.** It measures run-to-run sigma over
three replicates, and three runs pinned at a constant-policy floor agree with each other beautifully. A
resolution gate cannot distinguish "precise" from "stuck", which is an argument for reading G4 before G7
and not the reverse.

The training curves said this before the endpoint did, and nobody was reading them for it. The
reasoning-only control's final train CE is **1.6892 / 1.6891 / 1.6890** at 13M / 28M / 64M, and **five
times the parameters moves it by 0.0002 nats**. (An earlier draft converted that to "about 5.4
equiprobable answers". It cannot be: whole-sequence CE averages syntax, operands, operators and the
answer, so it describes the sequence and not the answer distribution. The invariance to width is the part
that needed no conversion and is the part that holds.) A task that
does not respond to width is a task the model is not learning; it is fitting the marginal distribution of
answers. §16.8's note that `train/CE` is not the endpoint remains true, and this is the one thing the
control's CE *was* telling us.

**What this does and does not invalidate.** The crowding hypothesis is untested rather than refuted: with
the endpoint at its floor, every cell's `<mano>` accuracy is noise about a constant, so a null on the count
axis or the entropy axis would mean nothing. That is exactly the failure PRD 1 catalogues four times in the
prior literature and exactly what §8.6 exists to catch, and it was caught before a null was published
rather than after. The storage half is unaffected: achieved bits against demanded is measured from value
spans and needs no reasoning endpoint at all.

**Before any more training.** `<mano>` at L=10 is unlearnable at these widths on 1.0B reasoning tokens, so
the design needs an endpoint that moves before a crowding claim is possible. In rough order of cost:

1. Read `<compare>` from the confirmatory scoring. It is a two-entity comparison answering a single birth
   year against a 0.70% floor, and it is absent from the controls only because a control has no facts to
   order -- so M0 never saw it. If it has range, the design has an endpoint today.
2. Retune `<mano>` down from L=10. PRD 12's M0 already contemplated this ("Retune Mano to L=10"); the
   measurement says go further, and the cost is one ladder rerun.
3. The in-context endpoints §8.3 names, which trade a memorised task for a read-and-answer one.

None of that is worth spending on until `<compare>` has been read, which the confirmatory scoring gives
for free.

### 16.10 What a re-audit of `a6e7074` found, and what it changes

An independent re-audit reached the same verdict this file reached in §16.9 -- crowding is untested, not
refuted -- and found several things §16.9 did not. Everything below was reproduced here before acting on
it.

**The gate report never existed.** `--write-gate-report` used `Path(...).write_text()`, and
`Path("s3://bucket/key")` collapses to `s3:/bucket/key`, so the M0 job created a **local directory named
`s3:`** in container scratch and wrote there. The log line said the report had been written to S3; nothing
was. `load_reports` read the same way, so the pending confirmatory job -- which passed that URI as
`--gate-report` -- would have died on `FileNotFoundError` after paying for the machine. `--out` had
branched on `is_url` and uploaded since it was written: the same concern, in the same file, handled two
ways. Both now go through `olmo_core.io`.

**`<compare>` cannot rescue the design, and the reason is a leak I introduced.** §16.9 named it the
cheapest next read. It is not readable. The answer is the earlier person's *birth-year value* -- changed
from a name in an earlier revision, to kill a 50.2% copy-policy floor -- so every training item states
`min(year(A), year(B)) = Y`. An entity's own year is therefore the **maximum answer over the items it
appears in**, exactly, whenever it is the earlier one at least once.

Measured on `13m_d0p3`, using 400,000 of the 2.63M training items the 50M-token slice buys (32 mentions
per entity against ~211 at full budget):

| | |
|---|---|
| entity years recovered by `max(answers)` | **97.09%** of 25,000 |
| eval accuracy from those alone, no biographies, no model | **99.72%** |

A model can score ~99.7% on `<compare>` without reading a single biography. It measures its own
supervision, not retrieval or composition. Promoting it after `<mano>` failed would also have been
outcome-contingent endpoint switching; the leak makes the question moot.

**The bit-measurement cohort is the contaminated one.** `CompareTask._probe_ids` is entities `0..24,999`,
and `--bit-entities` defaults to the first 25,000. So the storage sample is precisely the cohort receiving
~211 extra birth-year constraints each. Storage from that cohort must not be extrapolated to the corpus;
probe and non-probe cohorts have to be reported separately.

**Admission failed open in four ways**, all reproduced and all now refused: a report carrying a single
passing `G1` passed, because coverage against `GATES` was never checked; so did one carrying every gate
plus an invented `G99`; so did one carrying a gate twice; and `"passed": "false"` in JSON became Python
`True`, because `bool("false")` is truthy. `coverage_problem()` now distinguishes "G7 failed" from "G7 is
absent". The real M0 report contains every gate and fails, so none of this changes §16.9's verdict -- but
all four would have mattered for a positive one.

**Provenance was blank for a reason that is fixable.** The runtime image excludes `.git`, so every git
call inside a run fails and `train_commit` was empty on every checkpoint. The platform injects
`EDULLM_COMMIT_SHA`; `provenance.commit()` now prefers it and falls back to git on a laptop.

**Two table defects that would have biased an analysis rather than broken it.** Six prefixes carry 19
physical checkpoint directories for 18 logical cells, because crashed `13m_d0p6` and its re-run both
wrote -- and `--last-only` means "highest checkpoint in this prefix", which for the crashed one is a
partially-trained model in the same column as finished ones. `select_complete` now keeps one checkpoint
per `(cell_id, replicate)` and drops any that never reached the cell's own planned final step, and
`--expect-cells` refuses a short grid. Separately, storage is per checkpoint while the table is per
endpoint, so `achieved_bits_per_param` repeated on both rows of every two-endpoint cell; averaging that
column over rows double-weights exactly the fact-bearing count cells and none of the controls. One row per
checkpoint now carries `storage_row=True`.

**Where the audit overstates, slightly.** It says "below floor is slightly overstated -- the baseline is
estimated", and that is right: 4.695% is the best-constant policy computed over the item generator, and
the same policy on the frozen 30,000-item set is about 4.62%. The control's 4.44% is therefore at the
floor rather than provably beneath it. The conclusion does not move -- the achievable range is ~0 against
a required 10pp -- but §16.9's "a quarter of a point below the floor" should read "at the floor, within
the baseline's own uncertainty".

It is also right that whole-sequence training CE of 1.689 cannot be read as "5.4 possible answers": that
average covers syntax, operands, operators and the answer together, so the arithmetic in §16.9 describes
the sequence and not the answer distribution. The observation that survives is the one that needed no
conversion -- five times the parameters moves that CE by 0.0002 nats.

**The storage quantity needs its honest name.** It measures teacher-forced attribute-value CE reduction on
a fixed sampled cohort, conditioned on the gold biography prefix, at exposure 0, clamped at zero, over
non-embedding parameters. It is not Allen-Zhu R(F): the name term is absent, so at entropy `b=0` the
demand is ~0.200 b/param entirely from names while achieved storage is identically zero, and
`check_against_demand` cannot fire because value storage is capped below value-plus-name demand by
construction. Report it as a proxy, signed as well as clamped, with both denominators.

### 16.11 Phase 2: two endpoint defects, and the ordering that would have caught them

Phase 1 finished. `PHASE2.md` carries the plan; this records why the plan looks the way it does.

**Both endpoints were broken, and neither failure needed a GPU to find.**

`<mano>` at length 10 is a constant function. Eighteen scored cells span 4.127%–4.610% against a 4.695%
best-constant floor with **none above it**, and eleven of them scored *exactly* 1342/30000 — the number of
eval items answered `<n0>`. `MANO_LENGTH` was a module constant, so no depth sweep was expressible and the
unlearnability was discovered by training eighteen cells rather than by the two-hour, 14B-token calibration
that is now §4.A. The docstring had already recorded a reduction from 13 to 10 for the same reason; nobody
measured where the task becomes learnable, and the answer was "not at 10 either".

`<compare>` leaks the value it asks for. Its answer is the earlier person's birth-*year value*, so every
training item states ``min(year(A), year(B)) = Y`` and an entity's own year is the **maximum answer over
the items it appears in**, exactly, whenever it is the earlier one even once. Measured on `13m_d0p3` with
400,000 of the 2.63M available items: **97.09% of years recovered, and 99.72% eval accuracy with no
biographies and no model.** That is worse than the audit that raised it estimated.

The leak came from an earlier fix. The answer was a *name* until a copy policy scored 50.2% on it; moving
the answer to the year killed the copy floor and created a supervision channel. Both changes were locally
correct and the second undid the point of the task.

**Reducing exposure does not fix it**, which is worth recording because it was the obvious first idea. At
1.59 mentions per entity the reconstruction still recovers 58.5%, because a single item in which an entity
is the earlier one reveals that entity's year outright. The pools have to be disjoint, and now are:
`EntityTable.probe_ids_for(split)` splits the probe prefix in half, verified at 0 of 30,000 eval items
sharing an entity with training supervision.

**What phase 1's own models did, versus what the task allowed.** No phase-1 model exploited the leak — they
scored 0.5%–1.0% against a 0.605% floor, not the ~99.7% available. So the observed `<compare>` numbers were
never inflated; the task was simply unlearned, like `<mano>`. Both readings condemn the endpoint, and the
distinction matters for the write-up: the earlier claim that `<compare>` "measures its own supervision
shortcut" describes what a *competent* model would have done here, not what these did.

**The storage half survives, with a bound rather than a caveat.** Stored bits decompose additively over
attribute spans, so `birth_year` contributes at most its 8.644-bit prior. At demand 0.3 that leaves
13.0 bits (13M) and 15.9 bits (28M) of genuine multi-attribute storage; at demand ≥ 0.6 the entire signal
is consistent with `birth_year` alone. No re-run was needed to establish that, and the `--bit-offset` flag
now available would turn the bound into a direct measurement.

**`28m_b32`'s anomaly is resolved as a measurement artefact.** It claims 35.2% of prior stored while
reconstructing 0.097% of attributes, where `28m_b4` reconstructs 6.2% and claims 0.0%. Anti-correlated
measurements on the same axis mean the CE reduction is not retrievable knowledge. §16.9 left this open as
the discriminator to look for; the confirmatory scoring supplied it.

**Hardware, since phase 2 was expected to move to A100s.** 8×A10G returned **1.06×** of 4×A10G at 28M.
These sizes are communication- and launch-bound, not FLOP-bound, so a faster card does not return its FLOP
ratio and the case for an A100 cluster rests on running a *bigger model* rather than the same one faster.
That is an argument for the 113M rung, which has been defined in `ladder/sizes.py` since the beginning and
has never been trained. It is also why §7 asks for one paired measurement before any large commitment: the
README's 1.9× guess for eight devices was really 1.06×, a submission set its runtime bound from it, and the
run was killed at 72% after 13.8 hours.

**The ordering lesson, stated once.** Every defect above was findable without training: a depth sweep is
1.0B tokens per cell, a reconstruction attack on the compare stream is a hundred lines of numpy, and the
cohort overlap is two `range` calls compared. Phase 1 ran the confirmatory grid first because it was ready
first. §4 is ordered by what can invalidate what, not by what is ready.

### 16.12 Deferred, with reasons

FLD's 1,700 core-hours and 51.1% floor — decided in M0, not now. The exposure placebo and the
mechanism battery — M4. Qwen3-0.6B continuation — after M3. Retiring
`memory-split/docs/HANDOFF-factcrowd-dev.md`, which still points a fresh developer at a superseded
spec that disagrees with this one on architecture, materialisation, model sizes and corpus size: that
is a change in another repository and belongs to whoever owns it. **This file is the single source of
truth.**

Still owed for admission: **G1 is now expressible** — `configs/cells/calibration/` is its task-depth
sweep, and running it is §4.A of `PHASE2.md`. **G2 is nearly free** and was overlooked all along:
`CheckpointerCallback.pre_train_checkpoint` writes a step-0 checkpoint, which is the untrained control the
gate asks for. Only **G3's premise-ablated probe** still needs a corpus variant that does not exist. Each is cheap to run once
built — the expense is the variant, not the compute — and until they exist `score_run` will mark every
row `confirmatory=False`, which is the correct answer rather than a defect to work around.
