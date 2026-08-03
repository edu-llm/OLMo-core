# factcrowd — does storing facts cost reasoning?

2026-08-03. Build spec for the fact-crowding experiment. Everything needed to build it and run it is
here.

---

## 1. The question

**Does storing facts consume model capacity that would otherwise serve reasoning?**

The programme was founded on that as a premise, and it has never been tested. It traces to two
unsupported sentences in the phi-3 technical report — web pages are filtered out "to leave more model
capacity for reasoning for the mini size models," and "the model simply does not have the capacity to
store too much factual knowledge" — with no ablation anywhere in the report comparing a model trained
with that data against one without it. The report's own table contradicts it: phi-3-small at 7B
scores 59.1 on 5-shot TriviaQA against phi-3-mini at 3.8B scoring 64.0, while MMLU rises 68.8 to
75.3. Nearly doubling parameters made factual recall worse. Microsoft dropped the framing a
generation later; phi-4 attributes the same deficit to hallucination and quantifies a
**token-budget** trade-off instead, which is data competing for training tokens rather than facts
competing for weights.

Two papers come close and the gap between them is the opportunity. **Physics of Language Models 3.3**
measures fact capacity exactly — ~2 bits per parameter at 1000 exposures per fact, ~1 bit at 100,
architecture-universal — by summing autoregressive loss over exactly the knowledge tokens and feeding
it to a bit-complexity bound. It measures no reasoning at all. **Marek et al.** have the saturation
framing but state plainly that they "rely on proxies for measuring model capacity, such as model size
and pretraining loss," and the competition they demonstrate is old knowledge against new knowledge,
not facts against reasoning.

**Nobody has logged bit-counts and a controlled reasoning endpoint on the same checkpoints across a
swept oversubscription ratio.** That is what this does.

The independent variable is **ρ = demanded fact bits ÷ model capacity**. Not entity count, not token
count. The primary result is a **trend across five ρ values**, not a pairwise contrast. Flat refutes
crowding at this scale; a decline past ρ=1 supports it.

**Expect flat.** Everything adjacent points that way. Ouro runs Allen-Zhu's bit-counting protocol
verbatim on bioS and also measures reasoning: looping raises iGSM op15 accuracy from 46.3% to 70.7%
at identical parameter count while measured knowledge capacity is "nearly unchanged" — a
dissociation, and the strongest single piece of evidence against the premise. Mixture of Parrots
argues reasoning needs width while memorization needs total parameters, two different resources.
Johnston & Belrose find capacity pressure makes small models *memorize* two-hop answers rather than
compose them, which is the reverse causal direction. The experiment is worth running anyway: the
programme spent a year assuming the premise, and either answer is the first direct measurement.

**Design priority, in order: correctness of the measurement, then adjustability, then speed.** Four
reasoning endpoints have already produced uninterpretable nulls in this programme, every one an
instrumentation bug rather than a scientific result — iGSM scored at chance because the eval discarded
the derivation and graded a single mod-23 integer; a deduction eval scored *below* its own 0.500
floor because truncated derivations parsed as wrong; reasoning-gym macro-averaged 14 families with
chance floors from 0 to 0.5; two-hop composition ran at 2.3× the product of its parts, so it was
measuring fact access. And the whole previous sweep ran at 0.51% of the capacity ceiling, which says
nothing about a hypothesis whose mechanism requires the ceiling. The harness makes the checks that
would have caught these unavoidable rather than optional.

---

## 2. Scope

**In.** Entity and fact corpus generation with exact bit accounting; related and unrelated reasoning
slices; a width-scaled model ladder at fixed depth; the Allen-Zhu bit-counting probe; recall three
ways; reasoning evaluation behind a code-enforced bracketing gate; the trend regression; a
reasoning-only control arm per model size; one 32k domain BPE, trained once and published; a config
system where one file defines one cell.

**Out.** The split/masked arm — Experiment 4 owns it, though §7.4 keeps the seam open. Any retrieval
store. Distributed training beyond single-node multi-GPU. A Zipf arm. An MoE control.

**Deferred.** The Pythia continuation arm (§11, M4). Continuing training from a Pythia checkpoint
inside OLMo-core needs a GPT-NeoX architecture port — `olmo_core.nn.hf.convert` supports llama, qwen3
and qwen3_5 only — on the order of 400 lines plus tests, comparable to p3-math-split's
`nn/transformer/qwen.py`. That is a real cost and should be paid deliberately once the trend is in
hand rather than up front. **M0's seed-variance measurement is unaffected**: it is HF inference on
released checkpoints and needs no port.

---

## 3. The experiment

### 3.1 Grid

Rows are named by non-embedding parameters, which is what ρ is defined against (§7.1).

| | ρ=0.25 | ρ=0.5 | ρ=1 | ρ=2 | ρ=4 |
|---|---|---|---|---|---|
| **13M** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **28M** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **64M** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **113M** | | | ✓ | ✓ | |

Seventeen cells, plus one reasoning-only control per size. ρ is swept geometrically because the knee
sits at ρ=1 by construction and a fine sweep near the ceiling is wasted compute.

**One seed to start**, conditional on M0. A pairwise contrast at one seed is weak, but the trend pools
five points and reaches ~79% one-sided power for a 2pp total decline at a seed SD of 0.5pp — about
what a three-seed pairwise test would give. That 0.5pp is interpolated from published anchors of
0.21pp at n=10,042 eval items and 2.15pp at n=100, not measured. **M0 measures it directly on the
released `pythia-160m-seed1..4` variants plus base, inference only. If it lands above ~1pp, add seeds
at ρ=0.25 and ρ=4 before filling the middle.** Two things one seed does not buy: any claim about an
individual cell, and protection against an outlier run — the published pretraining outlier rate is
~4% of runs at >2 SD, so across 17 cells expect roughly one. Report the trend, not the cells, and
re-run any cell that breaks monotonicity.

Cost scales as **P²** at fixed ρ, because entity count scales with P and tokens with entity count.
Relative to the 13M row, 28M is 5×, 64M is 26×, 113M is 81×. That is why the top row is two cells and
why 250M is absent.

**Scale by width at fixed depth 12.** Reasoning capability tracks depth — 3-hop accuracy is
13/55/100/100 at 2/3/4/5 layers — while fact capacity tracks total parameters, which width supplies,
and reasoning is flat in width: GPT-2 Medium and Large from scratch give no gain over 124M on k-hop
QA. Those are the same fact from opposite ends and they make the two axes physically separable.
Width-scaling holds reasoning capability roughly fixed while occupancy varies, so a reasoning change
is attributable to fact load rather than architecture.

The risk is that reasoning then reads flat for a boring reason. The fix is to take dynamic range from
**task** depth instead of model depth:

| knob | role |
|---|---|
| model width | sweeps ρ — the hypothesis |
| model depth | fixed at 12, chosen so the endpoint sits mid-range |
| task depth | sweeps difficulty, proving the instrument responds |

### 3.2 Corpus: four slices

**Facts.** Synthetic biographies in natural-language prose, bioS-style: N people × 6 attributes from
closed pools, 47.6 bits/person by construction, ~100 tokens/bio, **≥20 templates per fact**. Prose
over dense records because the representation has to match what we test; it costs 6.6× more tokens per
bit (0.48 bits/token against a compact record's 3.12) and the hardware absorbs it. Multi-template is
mandatory: Physics 3.3 found diverse rendering does not hurt capacity and may help it, because
without diversity "the model wastes capacity memorizing sentence structures," and our own
single-template corpus answered the same question at 83% under one phrasing and 1.3% under another —
the fact was stored as pattern-slot → value, not as (entity, attribute) → value.

**Exposures fixed at 200, uniform.** Above our measured collapse threshold (between 49 and 196) and
above the ~35-exposure floor where capacity is roughly zero. Fixing exposures is what decouples the
axes: the previous sweep held total tokens fixed while raising entity count, so exposures fell
196 → 49 → 12, storage collapsed from 33.1 to 0.20 bits/entity, and nothing could be attributed to
entity count rather than exposure starvation.

Uniform rather than Zipfian is an instrument decision, not a simplification. Under Zipf at the same
total exposure budget, 60% of entities (α=1.0) or 89% (α=1.2) fall below the ~35-exposure floor. The
tail is then never learned, demanded bits are not N×47.6 but an unknown smaller number, and **ρ stops
being computable** — the experiment loses its independent variable.

**Related reasoning.** Composition, comparison and aggregation over the same entities — where crowding
shows up first if it operates through fact access. It covers a **fixed 25k-entity probe subset in
every cell**, at constant absolute tokens and constant per-entity coverage. If it covered all
entities instead, per-entity coverage would vary 20× across the ladder and confound P4. The probe
must fit inside the smallest cell — 13M at ρ=0.25 has 79k entities — and 25k leaves a 54k non-probe
comparison group there while staying far above the n≥2,000 the eval needs. The probe subset is also
the eval population for related reasoning, and comparing probe against non-probe recall is a free
internal check on whether QA mentions change storage.

**Unrelated reasoning.** The load-bearing measurement. If reasoning degrades here as fact load rises,
that is crowding proper; if only related reasoning degrades, the mechanism is fact access, which we
have already shown. The slice must contain **no memorizable facts** or it competes for the capacity
being measured.

### 3.3 Two invariants that make cells comparable

**Reasoning slices are constant in absolute tokens, not as a ratio.** 1.0B tokens per cell, every
cell: 500M FLD, 250M Mano, 250M related. Hold the ratio instead and reasoning data volume moves with
fact load, confounding the result in both directions. This also answers the catastrophic-forgetting
concern directly — reasoning is present in every cell at identical volume, so nothing is being
forgotten for lack of data. `mixture.py` takes absolute token counts and raises if a caller passes a
fraction.

**Every segment gets a domain token.** Without one, a 7/8-junk mixture costs up to 20× capacity at
these exposure counts; with one, 2×. Our corpus is a deliberate mixture, so this is not optional and
there is no flag to disable it.

### 3.4 ρ and training length are collinear, and how to read around it

With exposures fixed at 200 and bits-per-entity fixed at 47.6, tokens ∝ N ∝ ρ. On the 28M row the
ρ=4 cell sees 58.1B tokens against ρ=0.25's 4.6B — **12.7× the optimizer steps and 12.7× the
compute.** This is structural: ρ cannot be swept at fixed exposures without sweeping training length
with it.

Reasoning-token exposure is *not* confounded. The mixture is uniform over the stream, so at any
fraction of training every cell has consumed the same absolute number of reasoning tokens. What rises
with ρ is total compute — and more compute generally helps reasoning, so **the confound biases against
finding crowding.** A flat result could be crowding cancelled by longer training.

The fix costs nothing, because the checkpoint schedule already provides it. Two readings, both
pre-registered:

- **Reasoning-token-matched** — compare cells at equal *fraction* of training. This is the primary
  reading and the one the grid was designed for.
- **Compute-matched** — compare cells at equal *absolute* total tokens. The ρ=4 cell at 7.9% of
  training has consumed exactly what the ρ=0.25 cell consumed at 100%, and 8% is already on the
  log-spaced schedule (§7.2). Worth noticing: the spacing chosen for the bits curve happens to land on
  the compute-matched point.

**Crowding is supported only if the decline appears in both readings.** A decline in one alone names
which mechanism is operating, which is still worth having.

---

## 4. Where the code lives

Branch `edullm/fact-crowding` of OLMo-core, following the `edullm/p3-math-split` precedent: experiment
code under `src/scripts/train/factcrowd/`, tests under `src/test/scripts/factcrowd/` loaded by path,
and changes to `src/olmo_core/` only where genuinely needed. `mypy src/` covers scripts, so
`make checks` gates this code too.

A branch rather than a sibling repo because the experiment's whole data and measurement path is
`InstanceSource` subclasses and `Callback` subclasses. A separate package would be a thin shell around
OLMo-core internals, versioned separately from the internals it depends on.

```
src/scripts/train/factcrowd/
  README.md            how to run it; PRD.md is why
  PRD.md               this file
  corpus/
    entities.py        entity table generation, closed pools, exact bit accounting
    render.py          entity -> biography, ≥20 templates, precompiled token pieces
    reasoning.py       FLD / Mano / related adapters
    mixture.py         absolute-token mixing; the two invariants of §3.3
    source.py          the on-the-fly InstanceSource
    tokenize.py        the 32k BPE trainer, run once
  ladder/
    rho.py             (P, ρ) <-> (n_entities, fact_tokens). One function, both ways.
    sizes.py           width-scaled TransformerConfigs at fixed depth 12
  train/
    run.py             trainer wiring
    callbacks.py       BitCountCallback, ReasoningEvalCallback
  measure/
    bits.py            Allen-Zhu estimator
    recall.py          generation / recognition / linear probe
    reasoning.py       scorers; every one returns three counts
    registry.py        endpoint registration and the bracketing gate
  analysis/
    trend.py           the primary test: regression of reasoning on log2(ρ)
    figures.py
  configs/
    base.yaml
    cells/             one YAML per cell; the grid is data, not code

src/test/scripts/factcrowd/    mirrors the above, one-to-one
```

Match the house style: line length 100, isort + black, ruff, mypy, Sphinx-syntax docstrings on public
functions. Prefer a small number of well-named functions with exact contracts over clever
abstraction — teammates who did not write this will read it and adjust it often.

---

## 5. Contracts

Five types carry everything. Keeping them small is what makes cells swappable.

```python
@dataclass(frozen=True)
class EntityTable:
    """N entities x K attributes drawn from closed pools. Bits are exact by construction."""
    entities: pa.Table                  # entity_id, name, attr_1..attr_K
    pools: dict[str, tuple[str, ...]]
    probe_ids: frozenset[int]           # the fixed 25k related-reasoning subset
    seed: int

    @property
    def bits_per_entity(self) -> float:
        """sum(log2(len(pool)) for each attribute). No estimation anywhere."""

    @property
    def total_bits(self) -> float:
        return len(self.entities) * self.bits_per_entity


@dataclass(frozen=True)
class SliceSpec:
    """One component of the training mixture."""
    name: str                           # "facts" | "fld" | "mano" | "related"
    tokens: int                         # absolute, never a fraction
    domain_token: str                   # mandatory
    memorizable: bool                   # facts and related True; fld and mano MUST be False


@dataclass(frozen=True)
class CellSpec:
    """One grid cell. This is what a config file deserialises to."""
    cell_id: str
    model: ModelSpec                    # d_model, n_layers=12, vocab
    rho: float
    exposures: int                      # 200 everywhere
    entity_table: EntityTableSpec       # n_entities DERIVED from rho, never specified
    reasoning: tuple[SliceSpec, ...]    # constant in ABSOLUTE tokens across cells
    mask_spec: MaskSpec | None          # unused; the Experiment-4 seam
    init_from: str | None               # None = scratch

    def demanded_bits(self) -> float: ...
    def capacity_bits(self, r_e: float = 1.2) -> float:
        """MEASURED non-embedding params x R_E. Embeddings excluded; this is load-bearing."""
    def check(self) -> None:
        """Raise if rho as specified disagrees with demanded/capacity by >1%."""


@dataclass(frozen=True)
class EndpointResult:
    """Deliberately not a float."""
    correct: int
    incorrect: int
    unparseable: int                    # never folded into incorrect
    graded_score: float | None          # token accuracy or partial credit
    degenerate_baseline: float          # measured, not assumed
    chance_floor: float
    n: int


@dataclass(frozen=True)
class Measurement:
    """One row per checkpoint per cell."""
    cell_id: str
    step: int
    frac_trained: float
    total_tokens: int                   # for the compute-matched reading of §3.4
    achieved_bits: float
    achieved_r: float                   # achieved_bits / non_embedding_params
    per_entity_bits: np.ndarray         # the histogram, not just the mean
    recall_generation: float
    recall_recognition: float
    recall_probe: float
    reasoning: dict[str, EndpointResult]
```

`EndpointResult` is three counts and not one number because every past reasoning failure came from
collapsing it. `unparseable` is separate because a truncated derivation is not a wrong answer.
`degenerate_baseline` is measured because always-answering-"no" scored 0.507 on a task we read as
0.493.

---

## 6. Corpus pipeline

### 6.1 Generation and rendering

`entities.py` produces an `EntityTable` from `(n_entities, pools, seed)`. Pools are closed and
declared, so `bits_per_entity` is arithmetic, never estimated. The default follows Physics 3.3's
bioS — 6 attributes over pools of 200/300/100/263/200 plus dates, giving 47.6 bits/entity — chosen so
our bit-counts are directly comparable to theirs.

**The table is the only thing stored.** At 2.9M entities it is ~180 MB. Renders are generated from
`(table, seed, exposure_idx)`, so 200 exposures never touch disk.

Nothing about that requires new machinery. `InstanceSource` needs only `__len__`,
`__getitem__(idx) -> Instance`, `num_tokens`, and `fingerprint`, and
`olmo_core/data/composable/random_instance_source.py` is a working precedent: it generates instances
deterministically from `seed + idx` at access time and never touches disk. `BioInstanceSource` is the
same shape, mapping `idx -> (entity_id, exposure_idx) -> tokens`.

**Render by concatenating precompiled token pieces.** Tokenize each template's fixed segments and each
pool value once at build time; a render at train time is then a list concatenation, not a BPE call.
This matters three ways:

1. It is fast enough. The 13M cell needs ~2M tokens/s of corpus; a BPE call per bio is not, and a
   concat is.
2. **It makes the value-token spans exact by construction**, which is precisely what `bits.py`
   requires — no post-hoc span recovery and no alignment heuristic.
3. It removes a whole class of bug. The surface string in prose and the key used at eval are the same
   token arrays, so there is no way to build eval keys from a canonical name while prose uses a random
   variant — which trains recall-from-variant instead of copy-from-context.

One gate makes this safe. BPE can merge across a piece boundary, so a test asserts
`tokenize(prefix + value + suffix) == concat(tokenize(prefix), tokenize(value), tokenize(suffix))`
across the full template × pool cross product (~36k pairs, seconds to run). Pairs that fail fall back
to real tokenization and are logged; if more than a handful fail, the template set is wrong.

### 6.2 Exposure accounting

Every entity gets exactly **200 bio exposures** in every cell. The related-reasoning slice adds a
constant, measured number of additional mentions for the 25k probe entities only. Both numbers are
recomputed from the rendered stream and reported per cell; because the probe subset and its coverage
are constant across cells, this cannot confound the ρ trend, and probe-vs-non-probe recall measures
whether it matters at all.

`mixture.py` raises if a slice marked `memorizable=False` reuses content across examples above a reuse
threshold (default 1.05). FLD satisfies this by construction, which is why it is primary.

### 6.3 Tokenizer

**One 32k domain BPE**, trained once on a realistic sample and published as
`tokenizer/factcrowd-bpe-32k/v1`. ρ is defined against *non-embedding* capacity, and a 100k vocab at
d=256 would be 25.7M embedding parameters against 12.60M non-embedding — arithmetically fine and
indefensible to a reader. At 32k it is 8.2M. For an experiment about non-embedding capacity, spending
most of the model on an embedding table is self-defeating.

Open, cheap to settle before committing: FLD's WordNet-derived lexicon may tokenize badly under a BPE
trained mostly on biographies. Measure tokens-per-FLD-example under our BPE against GPT-2's during M0.

---

## 7. Training

### 7.1 The ladder

OLMo-core's presets vary depth *and* width, which would confound the two axes of §3.1. Fix depth at 12
and sweep `d_model`, taking the four widths the FFN's `multiple_of=256` quantum makes clean:

| Row | d_model | heads | head_dim | d_ffn | non-emb | N at ρ=1 | fact tokens at ρ=1 |
|---|---|---|---|---|---|---|---|
| **13M** | 256 | 4 | 64 | 1024 | 12.60M | 318k | 6.35B |
| **28M** | 384 | 6 | 64 | 1536 | 28.33M | 714k | 14.28B |
| **64M** | 576 | 9 | 64 | 2304 | 63.73M | 1.61M | 32.13B |
| **113M** | 768 | 12 | 64 | 3072 | 113.28M | 2.86M | 57.12B |

Built from `TransformerConfig.olmo2_190M(vocab_size, n_layers=12, d_model=..., n_heads=...)`. **Rows
are labelled by their measured non-embedding count, not by a round-number target**, because the labels
are cosmetic and the measurement is not. The ladder is a clean ~2.2× geometric progression, which is
what the design requires.

Note the FFN sizing, because it is easy to get wrong by a third. Every `olmo2_*` factory applies
`hidden_size_multiplier=1.5` on top of `8·d_model/3` and then rounds up to a multiple of 256
(`nn/transformer/config.py:1677-1681`), so `d_ffn` is 4× `d_model` here rather than the 2.67× a plain
SwiGLU would give. Read the parameter count off the built config, never off a hand formula:
`sizes.py` asserts the built model's non-embedding count matches the table to within 1%, and `rho.py`
derives N from that measured count — never from the label. That assertion is the difference between a
cell that sits at its stated ρ and one that only claims to.

**Tie the embeddings.** At a 32k vocab and d=256, untied embeddings are 16.4M against 12.6M
non-embedding — 57% of the model in a table that ρ explicitly excludes. Tied, it is 8.2M and 39%.
`TransformerConfig.tie_word_embeddings` (`config.py:333`) makes this a one-line change, and it is the
same argument that chose 32k over 100k. Embedding share by row, tied: 39% / 30% / 22% / 18%.

### 7.2 Checkpoints

Ten snapshots, log-spaced at **0.5 / 1 / 2 / 4 / 8 / 16 / 32 / 50 / 75 / 100%**. Capacity fills fast
then asymptotes, so linear spacing puts half its points on the flat part and under-samples both the
rising region and the double-descent bump at crossover. Log spacing puts seven points in the first
third where the bits curve is moving, and it supplies the compute-matched reading of §3.4 for free.

Bit-counting the largest cell (64M at ρ=4, 6.43M entities × ~20 value tokens) is ~130M forward-pass
tokens, well under a minute, so measurement cost is not the constraint. Subsample to 20k entities at
intermediate checkpoints, use the full table at the last one, and run the pricier reasoning evals at
four points.

### 7.3 Data loader

`BioInstanceSource` (§6.1) mixes with the reasoning sources through the composable API. Two OLMo-core
edges to respect, both of which have burned runs already:

- **Always pass `dtype` explicitly.** Both `NumpyDatasetConfig.get_dtype` and the composable
  `NumpyDocumentSource` fall back to the *narrowest* dtype the vocab fits, which silently reads a u32
  corpus two bytes at a time. Nothing raises; the only symptom is a loss curve that is merely worse
  than it should be.
- The composable `NumpyDocumentSource` derives document boundaries by scanning for EOS, **not** from a
  `.csv.gz` sidecar. That means it reads eduLLM `.u32le.bin` shards correctly, which the legacy
  `NumpyFSLDatasetMixture` path does not — worth telling the data team, whose consumer contract
  currently names `NumpyFSLDatasetConfig` as the only safe class.

### 7.4 The Experiment-4 seam

We are not building the split/masked arm, but do not close the door. `Instance` is a TypedDict that
already carries an optional `label_mask`, so `BioInstanceSource` emits one when `CellSpec.mask_spec`
is set, and `mask_spec` is otherwise unused. No mask sidecar files and no separate code path. Cost now
is one field and one branch; cost later of retrofitting is a rewrite.

---

## 8. Measurement

### 8.1 Bits

`bits.py` implements Allen-Zhu's estimator: **sum, never average**, the autoregressive loss over
exactly the value tokens, producing `loss_name` and `loss_value`, then convert via their
bit-complexity bound. Averaging is the single easiest way to get this silently wrong, so the function
takes token spans and asserts it received them — spans that §6.1 produces by construction.

Report **achieved** R(F) against **demanded** R^max(F), and plot reasoning against achieved. If the
x-axis is nominal rather than real, the experiment measures nothing.

Log the **per-entity distribution**, not just the mean. Whether degradation past saturation is uniform
or preferential is unmeasured in the literature, the two sources disagree, and the histogram costs
nothing.

### 8.2 Recall

Three measures, because they diverge: closed-book generation, MC recognition, and a trained linear
probe. Physics 3.1 found unaugmented facts cap extraction near 10% against ~96% augmented, so
generation alone understates storage.

### 8.3 Reasoning and the bracketing gate

Endpoints register through `registry.py` and **cannot run in the grid until bracketed**. Enforced in
code, not documentation.

`registry.bracket(endpoint, model)` runs and stores a **lower anchor** — a randomly-initialised model
plus the best constant policy, both through the production parser — an **upper anchor**, an oracle
handed the facts and derivation, and a **task-depth sweep** confirming the instrument responds.
Admission requires: sits between 20% and 80% of its range, moves ≥15 points across task depth, and
has a non-degenerate floor. `grid.run()` raises on any unbracketed endpoint.

Three endpoints ship. **FLD** is primary: a public Apache-2.0 generator (`hitachi-nlp/FLD-generator`),
so unbounded; real English syntax over a WordNet-derived pseudo-random lexicon; depth a dial from 1 to
8; proof accuracy with a **0.0 chance floor**, which removes the degenerate-baseline problem outright;
content words regenerated per example, so the slice carries no memorizable facts; and T5-base at 220M
reading 44.4 proof / 72.2 answer, so there is headroom in both directions at our scale. Score proof
accuracy primary, answer accuracy (33.3 floor) secondary. **Mano** is the parameter-sensitive symbolic
endpoint — mental mod-23 arithmetic, 8.2% at 25M to 36.7% at 85M at expression length 13 — with no
derivation parser to get wrong. **iGSM** third, and only behind a validated solution-step parser,
since its published accuracies are not computed from the final integer.

LSAT-family verbal reasoning is ruled out: AR-LSAT reads 21.5 and ReClor 33.7 for LLaMA-3.1-70B
against chance floors of 20.0 and 25.0. Nothing satisfies both "LSAT-like prose" and "trainable from
scratch at this scale"; FLD is the closest available compromise.

---

## 9. Configs

One YAML per cell, fully resolved, no inheritance beyond a single `base.yaml`. A cell config is the
unit a person edits and a run reproduces.

```yaml
# configs/cells/d576_rho2.yaml
extends: base.yaml
cell_id: d576_rho2
model: {d_model: 576, n_layers: 12, n_heads: 9, tie_word_embeddings: true,
        vocab: factcrowd-bpe-32k/v1}
rho: 2.0
exposures: 200
entity_table: {schema: bioS, seed: 1234}   # n_entities DERIVED from rho
reasoning:
  - {name: fld,     tokens: 500_000_000, depth_range: [1, 8]}
  - {name: mano,    tokens: 250_000_000, length: 13}
  - {name: related, tokens: 250_000_000, probe_entities: 25_000}
init_from: null
```

`ladder/rho.py` is the heart: `solve(P, rho, bits_per_entity=47.6, R_E=1.2) -> (n_entities,
fact_tokens)` and its inverse. Every config derives from it, so no cell can silently drift off its
intended ρ, and `CellSpec.check()` raises if the two disagree by more than 1%.

---

## 10. Compute and the run path

Train on a **capacity block via torchrun**, following the 8×B200 run this programme has already
provisioned once. **H200 is the likelier allocation**, so budget in H200-hours and treat B200 as the
upside.

| Component | H200-h | B200-h |
|---|---|---|
| 13M row, 5 cells | 14.4 | 6.3 |
| 28M row, 5 cells | 69.0 | 30.4 |
| 64M row, 5 cells | 340.8 | 149.9 |
| 113M at ρ=1 and ρ=2 | 413.5 | 181.8 |
| Reasoning-only controls, 4 sizes | 4.6 | 2.0 |
| **Total, 1 seed** | **~842** | **~370** |

On an 8-GPU node that is **~105 hours (4.4 days) on H200** against ~46 hours on B200. M0 is inference
only and needs neither.

FLOPs = 6·P·tokens at 8% MFU, calibrated against Physics 3.3's own anchor of 13,056 A100-hours for
GPT2-20-16 on bioS(10M) at 1000 exposures. The H200 column is the B200 column × 2.27, the ratio of
dense BF16 throughput (≈990 TFLOPS against ≈2,250) at equal MFU. That is the conservative reading: at
these widths MFU is poor on both, and a 256-wide model is *easier* to keep busy on the smaller
machine, so the real H200 penalty is likely under 2.27×. Measure MFU on the first 50 steps of M1 and
re-budget from that number rather than from this table.

**If 842 H200-hours is too much, cut from the top row.** The 113M cells are half the budget and their
job is to break the size confound, not to be the most extreme point. Dropping them leaves ~429
H200-hours and three complete rows, which still carries the trend. The floor is the 28M row alone at
**69 H200-hours** — 8.6 hours on an 8×H200 node — plus 13M and 64M at ρ=1 for another 46.

### The data path

Corpora are generated in-process and never materialised (§6.1). That is not only a speed decision:
materialising the grid is **597B tokens ≈ 2.4 TB** at the eduLLM standard's mandated u32, against the
587 GiB the entire `edullm-data` bucket holds today, and the largest single cell alone is 518 GB.

So publish what actually regenerates a cell: the **tokenizer, the entity tables, and the generator
configs** — under 2 GB for the whole grid, since `(table, seed, config)` reproduces any cell exactly.
Materialise and publish token shards only for cells that appear in the paper, if any.

Where we do touch the pipeline, the constraints are the data standard's, not ours: shards are
`.u32le.bin` with `tokens × 4 == bytes` exactly, the corpus points at a published tokenizer pinned by
checksum, names are kebab-case 2–5 words with no dates or version tokens, and versions are immutable.
Producers write **only** to `s3://edullm-landing`; the validator is the only principal that can write
`edullm-data`. Publishing is single-threaded by default and has timed out on a 218-shard corpus, so
pass `hash_workers`/`copy_workers` and `--timeout attemptDurationSeconds=7200`. Use the
`edullm-dataset-design` skill before the first publish, not after.

The eduLLM submission platform is not a candidate for the training runs. Its only provisioned GPU is a
single A10G — `gpu-8xa100` raises `UnprovisionedComputeProfileError` before a queue is chosen — and
842 H200-hours does not fit behind that. Larger instances are a budget call rather than a policy one,
so this may change; nothing in the design depends on which it is.

---

## 11. Milestones

**M0 — instruments before science. No training.** Not optional and not reorderable.

1. **Measure seed SD** on `pythia-160m-seed1..4` plus base, inference only. **This gates the one-seed
   decision and therefore everything.**
2. **Bracket every endpoint** against a random-init model and its best constant policy, through the
   production parser. If either matches the intended result, the endpoint is unusable.
3. **Validate the bit estimator** against a synthetic corpus with a hand-computable answer. Sum, not
   mean.
4. **Wire FLD** and confirm per-example content regeneration.
5. **Measure FLD tokenization** under our 32k BPE against GPT-2's (§6.3).

*Exit:* a bracketing report with each endpoint's anchors and depth sweep, and a measured seed SD.

**M1 — one cell end to end.** 28M at ρ=1, from scratch, ~9 H200-hours. *Exit:* a `Measurement` row at
every log-spaced checkpoint; exposures recomputed from the stream landing at exactly 200; mixture
token counts exact; a recall curve matching the predicted hinge within 10 points; and a measured MFU
to re-budget §10 from.

**M2 — the 28M row.** Five ρ values, ~69 H200-hours. *Exit:* achieved R(F) pins at ~1.2 past ρ=1, and
the first reasoning trend under both readings of §3.4.

**M3 — the full grid.** 13M/28M/64M rows, 113M at ρ=1 and ρ=2, plus the four controls, ~842
H200-hours. *Exit:* the reasoning-vs-ρ plot with a slope and CI.

**M4 — continuation arm.** Deferred; needs the GPT-NeoX port (§2). Decide after M3. Pythia ships 154
public checkpoints each at 70M/160M/410M, so prior occupancy becomes a measured variable rather than a
proxy — which is the upgrade over using tokens-per-param and pretraining loss as stand-ins. Prefer
160M if only one runs: a 70M Pythia has little reasoning ability to preserve, which is the whole
reason for starting from a pretrained checkpoint.

**M5 — seeds where it matters.** Add seeds at the noisiest cells, guided by M0's measured SD.

---

## 12. Testing

**Recompute, never trust.** A test that asserts a field is present is decoration. Every corpus test
recomputes from bytes: token count from file size, bits from the pool definitions, exposure counts
from the rendered stream.

Tests that encode specific past failures:

- key/prose surface agreement over 10k samples
- template × pool token-boundary agreement over the full cross product (§6.1)
- `unparseable` never folded into `incorrect`
- the bit-counter uses `sum` not `mean`, asserted on a planted-bits fixture
- every endpoint has a passing *and* a failing bracketing fixture
- `mixture.py` raises on a fractional reasoning slice, and on a memorizable one
- `CellSpec.check()` raises when ρ and `n_entities` disagree
- `sizes.py` raises when a built model's non-embedding count misses its target by >1%
- exposures recomputed from the stream equal exactly 200 for every entity

---

## 13. Pre-registered predictions

**P1.** Achieved R(F) rises linearly with demanded bits to ~1.8 b/p and pins at 2.0 ± 0.3. *Refuted
if* it pins below 1.5 or exceeds 2.5.
**P2.** Closed-book recall stays above 90% to the hinge, then falls as 2.0/demanded within ±10
points — 50% at ρ=2, 25% at ρ=4. *Refuted if* the decline starts below 50% occupancy, which would mean
the bit accounting is wrong and the experiment is measuring something else. This is a built-in
validity check: recall should be a hinge, not a slope.
**P3.** Unrelated-reasoning accuracy is flat within seed noise across ρ = 0.25 → 4, **under both
readings of §3.4**. *Refuted if* it declines by more than 3× the seed SD in both, which would be the
first direct evidence for crowding. A decline *before* the hinge indicates interference rather than
capacity.
**P4.** Related-reasoning accuracy declines past the hinge and tracks recall within 10 points, because
it depends on fact access.
**P5.** At fixed ρ, reasoning improves with model size. *Refuted if* flat, meaning 13M–113M is too
narrow a range.
**P6.** The reasoning-only control shows no decline at any size, isolating fact load as the cause of
any decline seen elsewhere.
**P7.** Recognition exceeds generation by ≥20 points at every ρ, per Physics 3.1's extraction gap.
**P8.** Seed SD on the reasoning endpoints is under 2 points at n ≥ 2,000 eval items. *Refuted if*
larger, in which case the grid needs more seeds than M5 plans.
**P9.** Storage at 200 exposures lands between the 1 and 2 bits/param lines, closer to 1. *Refuted if*
it exceeds 2, which would mean our bit accounting is inflated.
**P10.** Achieved R(F) plateaus before the token budget is exhausted at ρ ≥ 1, and does not at
ρ = 0.25.
**P11.** Depth-2 related reasoning exceeds depth-3 by ≥20 points at every ρ.
**P12.** Per-entity stored bits degrade roughly uniformly past ρ=1 rather than showing a rare-item
cliff, because exposure is uniform by construction. *Refuted if* the histogram is bimodal, which would
mean preferential forgetting and a different mechanism.

Effect size to power for is **~2pp**, the only comparable published number: a 410M Pythia on 32B
FineWeb-Edu tokens mixed with synthetic biographies at ratio 0.3, losing 2.09 points of average
zero-shot downstream accuracy. Our own previous split-vs-dense delta of 0.0006 is roughly 30× smaller
than that, which is a further sign the previous design was measuring nothing.

---

## 14. Still open

**The capacity block.** Everything downstream of M0 assumes one; nothing in M0 does. H200 versus B200
changes the wall clock by 2.3× and changes no decision in this document.

**Whether any token shards get published**, and if so which. Lean: none until a paper cell exists.

**M4's GPT-NeoX port.** Deferred, not descoped. If continuation matters more than the 113M row, the
port plus 160M at three checkpoints costs roughly what the 113M row costs, and it is the one arm that
supplies a model with real reasoning ability — so the unrelated-reasoning slice measures something
that exists outside our corpus.

**What a flat result licenses.** If reasoning is flat, the founding premise is refuted at 13M–113M by
direct measurement, and the pivot away from it is retroactively justified with evidence rather than
inference. It does not license a claim about 7B. Say so in the writeup before someone else does.

If reasoning declines past the hinge instead, it is the first demonstration that fact storage and
reasoning compete for capacity — a result in its own right, and one that revives the externalisation
argument on a proper footing. Either way this produces the first dataset with direct bit-counts and a
bracketed reasoning endpoint measured on the same checkpoints, which is the thing both anchor papers
left undone.
