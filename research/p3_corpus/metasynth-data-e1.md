# metasynth-data-e1.md — E1 data plan: synthetic named-lemma corpus + Metamath

**Written:** 2026-07-31. **Scope:** the two-corpus plan for E1 under the teacher-retriever design —
what gets generated, what gets extracted, how big each is, and what has actually been measured
versus estimated.

**One line:** train on a generated named-lemma rewriting corpus where the fact block is injected by
a perfect retriever, gate on whether the dense arm clears the floor, then re-run on Metamath `set.mm`
to show the finding survives on real verified mathematics.

> This is an alternative to `s3-data-e1.md`, not an extension of it. That plan offloads *coefficients*
> from arithmetic word problems; this one offloads *named equational lemmas* that must be applied.
> The two share no corpus.

---

## 1. Architecture

| Role | Source | Volume | State |
|---|---|---|---|
| Fact offloading (the experiment) | generated — named-lemma rewriting | unbounded; 566 bytes / 264 tok per example at depth 4 | generator written, `scripts/synth_generator.py` |
| Fact store | generated with the corpus | 20 k built; 100–200 k recommended | **to size** |
| Real-math replication | Metamath `set.mm` | 39,641 theorems / 8.15 M steps | **needs a verifier pass** |
| Held-out-fact eval | generated | 750 facts never cited in training | to build |
| Depth-OOD eval | generated at depth > train | follows the iGSM protocol | to build |

Everything the experiment turns on is generated. Metamath is the external-validity check, not the
primary instrument.

---

## 2. What "depth = 4" means

Depth is **the number of rewrite steps in the derivation**, which is also the number of lemma
citations, and therefore the number of entries in the fact block.

Problems are constructed **backwards**. Pick a ground term as the normal form, then un-apply a lemma
`k` times — each time finding a subterm matching the lemma's right-hand side and replacing it with
the corresponding left-hand side instance. The result is the start state. The forward derivation is
then guaranteed to exist and is replayed and checked before the example is emitted.

At depth 4 the model sees four lemmas and must produce four rewrite steps, each writing out the full
resulting term:

```
I know these mathematical statements:
lem_0f7dc4 : for all A, f(A, f(A, A)) = A
lem_05131d : for all A, f(h(u), h(A)) = f(c, A)
lem_19e7cb : for all A, f(f(A, d), g(c, d)) = A
lem_0a75d0 : for all A, f(f(a, A), b) = A
---
STATE
|- g(f(h(f(u, f(u, u))), h(b)), f(f(a, h(u)), f(f(b, d), g(c, d))))
STEP
rw [lem_0f7dc4]
RESULT
|- g(f(h(u), h(b)), f(f(a, h(u)), f(f(b, d), g(c, d))))
STEP
rw [lem_05131d]
RESULT
|- g(f(c, b), f(f(a, h(u)), f(f(b, d), g(c, d))))
STEP
rw [lem_19e7cb]
RESULT
|- g(f(c, b), f(f(a, h(u)), b))
STEP
rw [lem_0a75d0]
RESULT
|- g(f(c, b), h(u))
```

The property that matters: **no RESULT line is producible from the lemma name alone.** The model has
to read the statement, find where it matches, and carry out the substitution. That is the requirement
Lean tactic prediction failed — there, 100.0% of 337,314 premise references had the premise name
appearing verbatim in the target, and the elaborator did the derivation.

Depth is the difficulty dial. Measured cost per example:

| Depth | Bytes | dolma2 tokens | Fact-block share |
|---|---|---|---|
| 1 | ~150 | ~70 | 43% |
| 2 | ~290 | ~135 | 46% |
| 4 | 566 | 264 | 48% |
| 8 | ~1,100 | ~510 | 48% |

---

## 3. Expected accuracy, and the source

**Reference point: 99% in-distribution.** Ye, Xu, Li & Allen-Zhu, *Physics of Language Models:
Part 2.1, Grade-School Math and the Hidden Reasoning Process* (ICLR 2025, arXiv:2407.20311,
SSRN 5250629). Result 2: a GPT-2 with rotary embeddings, **pretrained from scratch** on the iGSM
synthetic corpus, "achieves 99% accuracy in solving math problems from the same distribution" and
additionally generalizes out-of-distribution to reasoning lengths never seen in training. Their
training difficulty is `op ≤ 15` (iGSM-med) or `op ≤ 21` (iGSM-hard), where `op` is the number of
solution operations; OOD evaluation runs at `op` of 20–23 and 28–32 respectively. Arithmetic is
mod 23 so that computation error does not contaminate the reasoning measurement. For calibration on
difficulty: GPT-4 few-shot **fails** on iGSM-med, against a ~32% guessing baseline.

99% is not a property of synthetic data. It is what a model reaches on *that* difficulty setting
after being trained on it — the same corpus defeats GPT-4 few-shot. The lesson is that a from-scratch
model saturates whatever fixed distribution you train it on, so difficulty has to be set deliberately.

### 3.1 Measured difficulty of this generator

Branching factor is the number of applicable `(lemma, position)` pairs at each state — the choices the
model faces per step. Its reciprocal is the per-step chance rate, and the product over a derivation is
the chance rate for a full trace. Measured over 400 problems per configuration:

| Configuration | Mean branching | Chance / step | Chance / full trace | Term size |
|---|---|---|---|---|
| depth 4, no distractors | 1.79 | 0.558 | **2.5 × 10⁻¹** | 96 chars |
| depth 8, no distractors | 2.81 | 0.356 | 4.0 × 10⁻³ | 141 chars |
| depth 4, +8 distractors | 2.00 | 0.501 | 2.1 × 10⁻¹ | 96 chars |
| depth 8, +16 distractors | 2.97 | 0.337 | 3.6 × 10⁻³ | 141 chars |
| depth 16 | — | — | — | generator fails, see below |

**Depth 4 is dead.** Chance on a full trace is 25%: guess among the applicable rewrites and you get a
quarter of derivations right. Dense will sit at or near 100% and there is no headroom whatsoever.

**Random distractors barely work.** Adding 8 distractor lemmas to a depth-4 problem moves mean
branching from 1.79 to 2.00; 16 distractors at depth 8 move it from 2.81 to 2.97. A lemma sampled at
random from the store almost never matches any subterm of the current state, so it is free to ignore.
To create real choice you need **confusable distractors** — lemmas that genuinely apply at some
position but lead away from the normal form.

**Depth 16 does not currently generate**, because backward construction hits the 220-character term
cap: terms grow from 96 chars at depth 4 to 141 at depth 8. Raising the cap, or biasing toward
contracting rules, is required before depth can be used as the difficulty dial.

### 3.2 Revised estimate

Error compounds across steps, so full-trace accuracy is roughly `p^depth`:

| Per-step p | depth 4 | depth 8 | depth 16 |
|---|---|---|---|
| 0.999 | 99.6% | 99.2% | 98.4% |
| 0.99 | 96.1% | 92.3% | 85.1% |
| 0.97 | 88.5% | 78.4% | 61.4% |
| 0.95 | 81.5% | 66.3% | 44.0% |

iGSM's 99% over `op ≤ 15` implies a per-step accuracy near 0.999. Our per-step operation — find the
matching position, substitute — is mechanically *simpler* than dependency resolution plus arithmetic,
so expect `p` at least that high once trained.

| Depth | Estimated dense accuracy | Verdict |
|---|---|---|
| 4 | >99% | ceiling, unusable |
| 8 | 90–97% | still too easy |
| 16 (once supported) | 75–90% | probably still above the band |

**Depth alone will not get you into 30–70%.** The reason is structural: at branching ~2 the derivation
is nearly forced, so there is no search and almost nothing to get wrong. iGSM is hard partly because
the model must *plan* — choose which parameter to compute next and avoid unnecessary work. This task
as drafted has no planning component. Getting into the band needs branching well above 2, which means
confusable lemma families and matching distractors, not longer chains.

> **The gate has two sides.** Dense must sit notably above the floor *and* notably below the ceiling.
> A dense arm at 99% leaves no room for a 3 pp effect. Calibrate on measured branching before
> spending GPU, not after.

---

## 4. Complexity relative to iGSM

The concern is overtuning: a model that learns *our template* rather than reasoning. Compared
against iGSM on each axis that bears on that:

| Axis | iGSM | This generator | Which is harder |
|---|---|---|---|
| Combinatorial space | >90 trillion solution templates (Prop 2.2) | lemma sequences alone are `store^depth`; 20 k store at depth 4 is 1.6×10¹⁷, a 200 k store is 1.6×10²¹, before goal terms and rewrite positions | **ours**, by orders of magnitude |
| Per-step output | one number, mod 23 (24 possibilities) | a full term, unbounded string | **ours** |
| Per-step operation | dependency lookup + arithmetic | pattern-match a LHS against a subterm at some position, then substitute | **ours** — matching is structural |
| Reasoning depth | trained at op ≤ 15 / ≤ 21, tested to 32 | depth 4 as drafted | **iGSM**, by roughly 4× |
| Surface variation | templated English over 4 hierarchical categorizations, 4 layers, 100 items/layer | one rigid template, no natural language | **iGSM**, substantially |
| Distractors | unused parameters present in problems | none — oracle block, exactly the lemmas needed | **iGSM** |
| Planning | must choose which parameters to compute and in what order; the model learns to emit shortest solutions | derivation order is fixed by the goal structure | **iGSM** |

**Verdict: combinatorially richer, structurally simpler.** Diversity of *instances* is not the
overtuning risk here — there are more distinct problems than iGSM has. The risk is that every
instance has the same shape, so the model can learn a fixed program (scan the block, match, rewrite,
repeat) that is correct but shallow, and generalizes to nothing outside this format.

### 4.1 Required changes before this is a fair test

- **Shuffle the fact block.** As drafted, `render_example` lists lemmas in first-use order, which
  leaks the derivation order outright. Blocking bug.
- **Raise and vary depth.** Sample depth per example rather than fixing it, and set the training
  ceiling from the 30–70% calibration.
- **Add distractor lemmas** to the block at a controlled rate. The oracle condition should mean
  "the needed lemmas are present," not "only the needed lemmas are present."
- **Vary the signature** across the corpus — operator arities, symbol inventory, variable naming —
  so the model cannot bind to one fixed alphabet.
- **Hold out a structurally different eval**: a fresh signature, unseen lemma shapes, and depths
  above the training ceiling, following iGSM's OOD protocol.

---

## 5. Sizing and fact saturation

The mechanism under test is that facts occupy capacity that could serve reasoning. If the dense arm
never memorizes the facts there is no occupied capacity to free, and both arms simply read the block.
Allen-Zhu & Li's knowledge-capacity work — cited by LMLM for the same reason — puts parametric
knowledge as under-trained below roughly a few hundred exposures, with ~1000 the point where capacity
estimates stabilize.

Exposures per fact = `examples × facts_per_example × epochs / store_size`. At depth 4 each example
cites 4 lemmas, which multiplies exposures fourfold at no repetition cost:

| Budget (264 tok/example) | Examples | 200 k store | 100 k store | 50 k store |
|---|---|---|---|---|
| 2 B | 7.6 M | 152 | 303 | 606 |
| 7.4 B (Chinchilla 20) | 28.0 M | **561** | **1,121** | 2,243 |

**Multiple epochs are not the right lever.** With a generator, fact exposure and data repetition are
decoupled: 28 M *distinct* examples all drawing from a 100 k store gives every fact ~1,100 exposures
with zero repeated examples. Metamath cannot do this — its corpus is fixed, so more exposure means
more epochs, and past roughly 4 they stop behaving like fresh data.

Practical costs at 28 M examples: **15.9 GB** on disk, and **~16 h single-core** to generate at the
measured 474 examples/s. Parallelize across cores.

### 5.1 The threshold is softer than "100–1000", and singletons are not waste

Two corrections to how the target above has been applied when screening corpora.

**80 exposures is workable; the 100 figure was being treated as a cliff it is not.** Allen-Zhu & Li's
capacity scaling work reports roughly 2 bits per parameter at 1000 exposures per fact, falling to
about 1 bit per parameter at 100. That is a smooth degradation, not a threshold. At 80 exposures a
370M model is near the 1 bit/param regime — on the order of 46 MB of storable knowledge, which is far
more than any store here requires. A corpus giving ~80 exposures is a weaker version of the same
experiment, not a broken one, and should be screened on other grounds.

**What matters is the citation-weighted distribution, not the per-fact count.** A fact cited once
cannot be memorized, so it contributes nothing to occupied capacity — but it also costs almost
nothing, because it appears in almost no examples. The right question is what share of premise *uses*
come from well-exposed facts. In Mizar, facts cited exactly once are **22.2% of the store but only
0.73% of all citations** (4,327 of 590,705). The store is heavy-tailed and the tail is nearly free.

Singleton facts are in fact **required by the design**, not tolerated by it. The 2×2×2 grid includes a
{used fact, new fact} axis, and rarely-cited facts are the natural population for the unseen-fact eval
condition. A corpus with no singletons could not populate that cell. Real mathematical citation
distributions are Zipfian, so a heavy tail is also what external validity looks like.

The screening rule that follows: reject on **coverage**, not on singleton share. FLD fails because
*zero* facts reach 10 citations, so no fraction of its uses is memorizable. Mizar passes because 93.0%
of uses come from the 6,108 facts cited ≥10 times. A corpus where 22% of facts are singletons and 93%
of uses are saturable is exactly the shape wanted.

---

## 6. Tokenizer

Measured bytes per token on realistic samples — 150 k generated examples (85 MB) and the full 42,318
Metamath statements:

| Tokenizer | Synthetic depth 4 | Metamath statements |
|---|---|---|
| bytes-utf8 | 1.00 | 1.00 |
| gpt2 (50,257) | 1.69 | 1.90 |
| gpt_neox_olmo (50,280) | 1.72 | 2.03 |
| dolma2 / dolma2_sigdig (100,278) | **2.14** | 2.05 |
| domain BPE 32 k | 1.78 | **2.16** |

Four findings. **`bytes-utf8` is the worst option available** and switching off it roughly halves the
token count for identical content — the single largest compute lever here. **`dolma2_sigdig` is
identical to `dolma2` on counts**; its right-to-left digit grouping changes which tokens represent a
number, not how many, so it buys representation quality for arithmetic and no compression. **The
domain BPE wins on Metamath and loses on synthetic**, because lemma names are 14.1% of bytes but ~17%
of tokens — a 10-character `lem_0f7dc4` costs 5.47 dolma2 tokens, and 20,000 random-hex names cannot
all get dedicated slots in a 32 k vocabulary. **Shorten the names**: a 200 k store needs only five hex
characters, and a structured scheme would tokenize to one or two tokens instead of five or six. Fix
that and re-measure before choosing.

The decision is settled by embedding cost regardless. At `d_model=1024` untied, a 100 k vocab is
**205 M parameters, 43% of a 370 M model**, against 65.5 M and 19.6% at 32 k. Holding total size
fixed, dolma2 leaves 165 M non-embedding parameters where 32 k leaves 305 M. For an experiment whose
hypothesis is about non-embedding capacity, spending 43% of the model on an embedding table
undermines the measurement.

---

## 7. Metamath as the replication corpus

Measured on `set.mm` (42.9 MB, parsed directly):

| Quantity | Value |
|---|---|
| Theorems with proofs | 39,641 |
| Axioms / syntax constructors | 2,677 |
| Total proof steps | 8,151,235 (mean 205.6 per theorem) |
| Premise references | 1,303,436 (mean 32.9 distinct per theorem) |
| — logical (`\|-`) | 766,899 (58.8%) |
| — syntax construction | 506,884 (38.9%) |
| — local hypotheses | 29,653 (2.3%) |
| Fact block + goal, rendered | **22.0 M GPT-2 tokens** |
| Mean essential steps per theorem | 40.4 |
| Mean tokens per essential step | 47.0 |
| **Full traces, essential (`\|-`) steps only** | **97 M tokens** |
| **Full traces, all steps incl. syntax** | **190 M tokens** |
| Fact-block share, essential rendering | **21.0%** |

Trace figures are now measured, not extrapolated: `scripts/mm_expand.py` implements enough of the
Metamath spec (scoping, mandatory hypotheses, compressed-proof decoding, substitution) to run the
verifier's stack machine and record the formula produced at every step. Expanding a 3,000-theorem
sample and scaling gives 190 M tokens for the all-steps rendering and 97 M for essential-only. **The
earlier ~296 M extrapolation was 56% too high**, and the fact-block share is 21.0% rather than the
~6.5% previously estimated — a materially stronger manipulation than assumed.

### 7.1 Expected accuracy

**Anchor: 29.22%.** Polu & Sutskever, *Generative Language Modeling for Automated Theorem Proving*
(arXiv:2009.03393). Their Table 5 baseline is a **160 M-parameter model trained from scratch** on the
raw Metamath dataset with a proofstep objective over 18 B tokens, closing **29.22%** of a held-out
test set. Their best model (larger, pretrained on WebMath) reaches 56.22%; the prior state of the art,
MetaGen-IL, was 21.16%.

That baseline is close to an exact analogue of this plan: from scratch, small, set.mm, whole-proof
success. Scaling 160 M → 370 M and adding an oracle fact block, which GPT-f did not have, should help
rather than hurt.

| Metric | Estimate for a 370 M from-scratch run |
|---|---|
| Whole-proof success via search | **30–40%** |
| Per-step result exact match | **70–85%** |
| Chance | effectively 0 (formula generation) |

**Both bracket the 30–70% band**, and unlike the synthetic task that estimate rests on a published
number for nearly this configuration rather than on extrapolation. Note the corpus/budget mismatch:
GPT-f used 18 B tokens against a 97–190 M corpus, so roughly 95 epochs.

Why Metamath rather than Lean: its design principle is the inverse of Lean's. Its own documentation
states that proofs "include every step, no exceptions... This is different from almost all other
computer-verifiable proof systems, which allow statements (like `simp`, `auto`, or `blast`) that
don't show the proof steps." That is exactly the property that disqualified LeanDojo. Verification is
seconds across five independent verifiers, and the ASCII surface tokenizes well.

### 7.2 Full worked trace — theorem `0ellim`

"If A is a limit ordinal then the empty set is an element of A." Lines marked `.` are syntax
construction; unmarked lines are the 8 essential logical steps.

```
I know these mathematical statements:
wlim      : wff Lim A
c0        : class (/)
wcel      : wff A e. B
wne       : wff A =/= B
wceq      : wff A = B
nlim0     : |- -. Lim (/)
limeq     : |- ( A = B -> ( Lim A <-> Lim B ) )
mtbiri    : |- ( ph -> -. ps )
necon2ai  : |- ( ph -> A =/= B )
word      : wff Ord A
wb        : wff ( ph <-> ps )
limord    : |- ( Lim A -> Ord A )
ord0eln0  : |- ( Ord A -> ( (/) e. A <-> A =/= (/) ) )
syl       : |- ( ph -> ch )
mpbird    : |- ( ph -> ps )
---
GOAL  |- ( Lim A -> (/) e. A )
DERIVATION
.  1  cA             class A
.  2  wlim           wff Lim A
.  3  c0             class (/)
.  4  cA             class A
.  5  wcel           wff (/) e. A
.  6  cA             class A
.  7  c0             class (/)
.  8  wne            wff A =/= (/)
.  9  (reuse)        wff Lim A
. 10  cA             class A
. 11  c0             class (/)
. 12  cA             class A
. 13  c0             class (/)
. 14  wceq           wff A = (/)
. 15  (reuse)        wff Lim A
. 16  c0             class (/)
. 17  wlim           wff Lim (/)
  18  nlim0          |- -. Lim (/)
. 19  cA             class A
. 20  c0             class (/)
  21  limeq          |- ( A = (/) -> ( Lim A <-> Lim (/) ) )
  22  mtbiri         |- ( A = (/) -> -. Lim A )
  23  necon2ai       |- ( Lim A -> A =/= (/) )
. 24  (reuse)        wff Lim A
. 25  cA             class A
. 26  word           wff Ord A
. 27  (reuse)        wff (/) e. A
. 28  (reuse)        wff A =/= (/)
. 29  wb             wff ( (/) e. A <-> A =/= (/) )
. 30  cA             class A
  31  limord         |- ( Lim A -> Ord A )
. 32  cA             class A
  33  ord0eln0       |- ( Ord A -> ( (/) e. A <-> A =/= (/) ) )
  34  syl            |- ( Lim A -> ( (/) e. A <-> A =/= (/) ) )
  35  mpbird         |- ( Lim A -> (/) e. A )
```

Read the essential spine on its own and it is a clean mathematical argument: the empty set is not a
limit ordinal (18); equal classes have equal limit-hood (21); so if A were empty it would not be a
limit (22); contrapositively a limit A is non-empty (23); limits are ordinals (31); for ordinals,
non-emptiness is equivalent to containing the empty set (33–34); therefore a limit contains the empty
set (35).

Two design consequences. **The syntax steps should be filtered out** — 27 of 35 lines here carry no
logical content, matching the 38.9% syntax share database-wide, and asking the model to reproduce
`class A` teaches nothing. **The `(reuse)` steps matter**: Metamath's compressed format lets a proof
reference an earlier result, so a step-level rendering must either inline those or expose the
reference, and that decision changes both the token count and what the model must track.

---

### 7.3 Scaling beyond set.mm

190 M tokens is not enough on its own — at a 7.4 B budget that is 39–76 epochs depending on the
rendering. Three ways to grow it, in increasing order of payoff.

**The other Metamath databases** add real but modest volume. Measured across all five:

| Database | Theorems | Steps | Premise refs |
|---|---|---|---|
| set.mm | 39,641 | 8,151,235 | 1,303,436 |
| iset.mm (intuitionistic) | 9,604 | 733,382 | 151,592 |
| nf.mm (New Foundations) | 5,966 | 383,389 | 76,696 |
| ql.mm (quantum logic) | 1,138 | 113,856 | 15,751 |
| hol.mm | 138 | 12,910 | 2,160 |
| **Total** | **56,487** | **9,394,772** | **1,549,635** |

That is **+42.5% theorems but only +15.3% steps** — their proofs are shorter — taking the corpus to
roughly **219 M all-steps / 112 M essential-only**.

*Re-measured against the current `develop` snapshot, counting `$p` assertions and compressed-proof
label references: set.mm 47,588 theorems and 1,543,440 steps, iset.mm 16,236 / 377,941, nf.mm
5,971 / 76,796, ql.mm 1,140 / 15,767, hol.mm 151 / 2,173. The theorem counts are higher than the table
because the library has grown; the step counts are lower because they count proof-step label
references rather than expanded stack operations. The rendered corpus that follows from set+iset+nf is
69,767 theorems and 101 M symbol tokens (§11.6), which is the number to plan against.*

**The "different logics sharing notation" worry does not survive measurement — include them.** The
concern was that iset.mm (intuitionistic), nf.mm (New Foundations) and ql.mm (quantum logic) reuse
set.mm's notation under incompatible axioms, so the same label would denote conflicting facts. Fetched
all five databases from `develop` and compared statements directly:

| | shared labels with set.mm | identical statement | genuinely conflicting |
|---|---|---|---|
| iset.mm | 12,979 | **12,895 (99.4%)** | 84 (0.6%) |
| nf.mm | 4,901 | **4,665 (95.2%)** | 236 (4.8%) |

The auxiliary databases are overwhelmingly **the same named facts with different proofs**, which is
precisely the "diversity of use" property §9.1 asks for — the same premise exercised in a second
derivation — rather than noise. Most of the 84 iset.mm conflicts are cosmetic bound-variable renaming
(`( A. x ph -> E. x ph )` versus `( A. x ph -> E. y ph )`); only a handful, like `2a1i`, differ
substantively. Namespace or drop those and the rest is free exposure on facts already in the store,
which is the scarce resource.

On the current `develop` snapshot the auxiliaries add 23,503 theorems and 472,677 proof steps against
set.mm's 47,588 and 1,543,440 — **+49% theorems and +31% steps for +30% bytes.** Constant-symbol
overlap with set.mm is 93% for iset.mm and 63% for nf.mm, but only 30% for ql.mm and its 1,140
theorems, so ql.mm and hol.mm (151 theorems) are marginal either way. They remain natural held-out
domains for the new-fact eval.

**Sub-proof extraction** multiplies examples without new mathematics: every intermediate goal in a
proof is itself a valid theorem with its own derivation, so a 205-step proof yields many training
targets rather than one.

**Forward-proving generation is the real answer, and it dissolves the volume problem.** Wang & Deng,
*Learning to Prove Theorems by Learning to Generate Theorems* (NeurIPS 2020, arXiv:2002.07019),
introduced MetaGen for exactly this: "the basic operation is generating a proof step — selecting an
existing theorem and constructing a substitution. From this single proof step we can derive a new
theorem. Now, we can treat this new theorem as an existing theorem and repeat." They note the space
of new theorems and proofs is **infinite**.

This is the combination the other options cannot offer:

| Property | Forward-proven Metamath | Synthetic named-lemma | Human set.mm |
|---|---|---|---|
| Volume | unbounded | unbounded | 219 M tokens |
| Premises are real, meaningful theorems | yes | **no** | yes |
| Diversity of use per premise | inherited from set.mm | low | 0.876 distinct-result ratio |
| Kernel-verifiable | yes | yes (generator) | yes |
| Contamination | none (training from scratch) | none | none |

The machinery already exists in `scripts/mm_expand.py` — substitution and mandatory-hypothesis
resolution are the same operations run generatively instead of as a check. MetaGen's neural component
exists only to make generated theorems *resemble human ones*; for corpus volume, unguided forward
proving is sufficient and is a few hundred lines.

**Measured caveat — unguided forward proving produces valid nonsense.** Implemented in
`scripts/mm_forward.py` and run against set.mm, it yields theorems like:

```
eleq1i     : |- ( A e. C <-> B e. C )
df-sgrOLD  : |- SemiGrp = ( Magma i^i Ass )
apply eleq1i to df-sgrOLD
|- ( SemiGrp e. ( 6 x. 3 ) = ; 1 8 <-> ( Magma i^i Ass ) e. ( 6 x. 3 ) = ; 1 8 )
```

Derivable and well-formed, but it asserts that SemiGrp is an element of "6 × 3 = 18". This is exactly
why MetaGen needed a *learned* generator with a GAN-style discriminator — their stated hypothesis is
that "generated data will be more useful if they are similar to human-written data."

What survives: the **premises** are still real set.mm theorems and each application is a distinct
instantiation, so the fact store keeps its meaning and its diversity of use. What breaks: the problem
distribution is far from human mathematics, so transfer to a human-theorem eval is unproven. Cost is
~100 GPT-2 tokens per generated example, so a 7.4 B budget is ~74 M examples.

Treat unguided forward proving as **augmentation at a modest mixing ratio**, never as the bulk of the
corpus, and keep the eval on human theorems only. Matching MetaGen's quality needs their learned
generator, which is a project in itself.

---

## 8. Evaluation

| Set | Cells | What it establishes |
|---|---|---|
| `e1_synth_seen_fact` | {split, dense} × {block on, off} | split-vs-dense where dense could have memorized (control) |
| `e1_synth_new_fact` | {split, dense} × {block on, off} | **the offloading claim** — 750 held-out facts dense cannot know |
| `e1_synth_ood_depth` | as above, depth above the training ceiling | does either arm generalize, per the iGSM protocol |
| `e1_metamath` | {split, dense} × {block on, off} | does it survive on real verified mathematics |

Metrics, reported separately rather than aggregated: **premise-name accuracy** (retrieval), **result
exact-match per step** (content application), **full-derivation exact match**, and **teacher-forced
NLL on result tokens with the block on versus off**. The last is continuous and moves even when exact
match is at the floor.

**Pre-flight, from one dense run, before split is trained at all:** dense-with-block must beat
dense-without-block, or the facts are not load-bearing and the contrast is vacuous. Dense must also
land inside 30–70%. Both checks come from the same run.

---

## 9. Controls

- **Fact-block dropout during training — required, and currently absent.** With a perfect retriever
  the fact is always in context, so the cheapest way for the *dense* arm to reduce loss is
  attend-and-copy, not internalize. Nothing pressures it to memorize, and dense ≈ split becomes a
  trivial null: neither arm stored anything. Both reference systems avoid this — ReProver sets
  `p_drop: 0.5` and drops each retrieved premise independently, and LMLM's fuzzy retriever genuinely
  misses. Drop the block on some fraction of training examples so the dense arm sometimes has to
  answer without it, while the split arm is masked either way and cannot benefit. Eval stays oracle.
- **Cooldown A / B** (matched total tokens / matched loss-bearing tokens). At ~48% masked, the B
  extension is ~1.9×, materially larger than the ~1.18× the Lean corpus needed.
- **Random-mask arm.** Cooldowns match *how many* tokens are masked, not *which*. Without an arm that
  masks an equal-sized random set of non-fact tokens, a dense win reads as "any masking hurts."
- **Store-size sweep.** Store size trades capacity occupied against exposures per fact; both bear on
  the effect size, in opposite directions.

### 9.1 Diversity of use — a corpus criterion, not a control

Exposures per fact is necessary but not sufficient. If a lemma is used 1,000 times and every use is
the same mechanical operation with a different variable binding, then "knowing" it is exactly
"reciting" it, and the split/dense contrast measures recall rather than reasoning. That is why
arbitrary lemmas are fine for LMLM — its claim *is* about recall — and not for this experiment.

The measurable criterion is the **distinct-result ratio**: across all applications of a premise, what
fraction produce different resulting formulas. Measured over 2,500 expanded set.mm proofs, the mean
is **0.876** and the median is **1.000**; for **62.7%** of premises every single application yields a
different formula. `syl` is used 2,400 times and produces 2,278 distinct results.

The design tension to hold: fact *content* should be arbitrary, since incompressible facts cost the
most bits and give the strongest capacity effect, while fact *applicability* should be non-trivial,
since that is where reasoning lives. Arbitrariness is not the problem; mechanical application is.
Report this ratio for whatever corpus is chosen, and treat a low value as disqualifying.

---

## 10. Open items

- [ ] **Add fact-block dropout to training** (§9) — without it the dense arm has no reason to
      memorize and the experiment can null trivially. Blocking.
- [ ] Build the forward-proving generator on top of `mm_expand.py` (§7.3) — this is what takes the
      corpus from 219 M tokens to unbounded while keeping real premises.
- [ ] Fix the fact-block ordering leak in `render_example` — blocking if synthetic is used.
- [ ] **Raise branching above ~2.** Confusable lemma families and matching distractors, not depth
      and not random distractors, which were measured to move branching by 0.2 (§3.1).
- [ ] Raise the 220-character term cap so depth >8 generates at all.
- [ ] Shorten the lemma-name scheme and re-run the tokenizer bake-off.
- [ ] Decide store size (100 k vs 200 k) against the saturation table in §5.
- [ ] Decide the `(reuse)` policy for Metamath step-level rendering (§7.2).
- [ ] Add signature variation and a structurally-held-out eval per §4.1.
- [x] ~~Verifier pass over set.mm~~ — done, `scripts/mm_expand.py`; 190 M / 97 M measured.
- [x] ~~Metamath fact-block share~~ — 21.0% under essential-only rendering.

---

## 11. The portfolio

**Superseded by §11.6.** The table below was the state before direct measurement. Every figure in it
turned out to be wrong in a way that mattered: IsarStep's token count and fact block were misdescribed,
TPTP ships no proofs at all, and NaturalProofs' per-step attribution is a third of what was claimed.
It is kept here because later sections refer back to it; §11.6 carries the measured replacement.

| Corpus | Examples | Tokens | Fact block | Target |
|---|---|---|---|---|
| ~~**IsarStep**~~ | 820 k train / 5 k val / 5 k test | ~~~410 M (estimated)~~ | ~~native — field F.5~~ | a proposition to synthesize |
| ~~**Metamath**, theorem-level~~ | 56,487 theorems | ~~112 M (all 5 DBs)~~ | native — cited labels | full derivation |
| ~~**TPTP** v9.3.0, math domains~~ | 11,105 problems | ~~80 M~~ | native — separate `.ax` files | ~~proof~~ — **none ship** |
| ~~**NaturalProofs**~~ | 32,511 theorems | ~~48 M~~ | ~~native — `refs` per proof~~ | prose derivation |
| ~~**Total**~~ | | ~~**~650 M**~~ | | |

### 11.1 One characteristic example from each

**IsarStep** — five fields per example; F.5 is the fact block, F.1 is what the model must produce.
Taken from the paper's √2-irrationality worked example:

```
F.5  (library lemmas — the fact block)
     even_mult_iff : even (?a * ?b) = (even ?a ∨ even ?b)

F.2  (local propositions used to derive the target)
     2 * b² = a²

F.3  (a local proposition derived FROM the target)
     ∃c ∈ ℤ. a = 2c

F.4  (other local propositions justifying F.3)

F.1  (TARGET — the model synthesizes this)
     a is even
```

The target is a mathematical proposition, not a lemma label. That is the property Lean tactic
prediction lacks.

**Metamath** — theorem `0ellim`, "if A is a limit ordinal then ∅ ∈ A". Full trace in §7.2; the
essential spine is:

```
I know these mathematical statements:
nlim0     : |- -. Lim (/)
limeq     : |- ( A = B -> ( Lim A <-> Lim B ) )
mtbiri    : |- ( ph -> -. ps )
necon2ai  : |- ( ph -> A =/= B )
limord    : |- ( Lim A -> Ord A )
ord0eln0  : |- ( Ord A -> ( (/) e. A <-> A =/= (/) ) )
---
GOAL  |- ( Lim A -> (/) e. A )
  18  nlim0      |- -. Lim (/)
  21  limeq      |- ( A = (/) -> ( Lim A <-> Lim (/) ) )
  22  mtbiri     |- ( A = (/) -> -. Lim A )
  23  necon2ai   |- ( Lim A -> A =/= (/) )
  31  limord     |- ( Lim A -> Ord A )
  33  ord0eln0   |- ( Ord A -> ( (/) e. A <-> A =/= (/) ) )
  34  syl        |- ( Lim A -> ( (/) e. A <-> A =/= (/) ) )
  35  mpbird     |- ( Lim A -> (/) e. A )
```

**TPTP** — problem `SET001-1`, Set Theory. The axioms live in a separate file already, and the
header carries a difficulty rating:

```
% File     : SET001-1 : TPTP v9.3.0
% Domain   : Set Theory
% Problem  : Set members are superset members
% Rating   : 0.00 v2.0.0            <-- difficulty label, 0 = every ATP solves it

include('Axioms/SET001-0.ax').      <-- the fact block, already a separate file

cnf(b_equals_bb,   hypothesis,          equal_sets(b,bb) ).
cnf(element_of_b,  hypothesis,          member(element_of_b,b) ).
cnf(prove_element_of_bb, negated_conjecture, ~ member(element_of_b,bb) ).
```

**NaturalProofs** — "Derivative of Exponential Function" from ProofWiki. Note that the `{{eqn}}`
template's `c =` field names the theorem used at **each individual step**, so per-step premise
attribution is native, not something to be inferred:

```
THEOREM  Derivative of Exponential Function
         d/dx (exp x) = exp x

REFS (the fact block)
  Exponential of Sum                   : exp(x + y) = exp x · exp y
  Multiple Rule for Limits of Functions: lim_{x→c} (λ f(x)) = λ l
  Derivative of Exponential at Zero    : lim_{h→0} (exp h − 1)/h = 1

PROOF
  d/dx (exp x) = lim_{h→0} (exp(x+h) − exp x)/h      c = {{Defof|Derivative}}
               = lim_{h→0} (exp x · exp h − exp x)/h  c = [[Exponential of Sum]]
               = lim_{h→0} (exp x (exp h − 1))/h
               = exp x · lim_{h→0} (exp h − 1)/h      c = [[Multiple Rule for Limits]]
               = exp x                                c = [[Derivative of Exponential at Zero]]
```

### 11.2 On the 10-epoch objection

The concern was that ~650–700 M tokens against a 7 B budget means ~10 repetitions, which is
"typically too much." That heuristic is right for general pretraining and **wrong for this
experiment**, for two reasons.

First, the source of the heuristic — data-constrained scaling work — finds repetition roughly free up
to about 4 epochs and decaying toward zero by around 16. Ten epochs sits in the diminishing-but-real
band, not the worthless one. It costs something; it is not disqualifying.

Second, and more importantly: **this experiment needs repetition.** The dense arm can only occupy
capacity with facts if it memorizes them, and memorization needs on the order of 100–1000 exposures
per fact (§5). A corpus that cites each fact ~20 times in one pass gives 200 exposures at 10 epochs —
which is the regime where the manipulation has something to bite on. Running a single epoch over 7 B
tokens of *fresh* data would leave the dense arm having seen most facts a handful of times and
memorizing nothing, producing a trivial null.

The thing to avoid is not repetition but **repetition of the reasoning portion without added fact
exposure**. That argues for more facts per example rather than more epochs, and for tracking
per-region loss on the fact block (§9) rather than epoch count as the health signal.

### 11.3 Hub-sweep candidates, verified

A sweep of ~8,300 HuggingFace datasets plus Zenodo, Kaggle and GitHub returned two headline
candidates. Both were measured directly against the criteria rather than accepted on their schema.
One is disqualified; the other is real but far more expensive than advertised.

**FLD (Formal Logic Deduction) — rejected.** On format it is a perfect match: a named fact block,
and a derivation target that writes out every intermediate formula
(`fact1 -> int1: ¬{A}{a}; int1 & int2 -> hypothesis;`). It fails on the property that actually
matters. Measured over 60 k examples from `hitachi-nlp/FLD.v2`:

| | `star` | `default` |
|---|---|---|
| distinct fact strings | 418,132 of 422,032 instances | 368,197 of 370,097 |
| reuse per distinct fact | **1.01** | **1.01** |
| facts appearing exactly once | **99.2%** | **99.5%** |
| facts appearing ≥10× | **0** | **0** |
| distinct fact *labels* | 32 (`sent1:`…`sent32:`) | 27 |

The labels are positional, not names — `sent6` denotes a different proposition in every example — and
the content behind them is generated fresh per example over pseudowords. There is no library to
memorize, so the dense arm cannot occupy capacity with these facts and masking them destroys
information rather than externalizing it. Split would lose by construction, which is a guaranteed but
uninformative null. This is the objection already raised against the home-grown generator and GSM8K,
in its purest form: the lemmas do not mean anything, so nothing makes the model learn them.

**`fumiyau/mathlib4-state-change` — real, but a multi-day job, not a drop-in.** 276,014 rows pairing
`state_before` / `tactic` / `state_after` with a list of named Lean declarations carrying full source.
That combination — a large persistent named store *and* a content target — is what every other Lean
corpus lacks, and the 154,853-declaration index is only 28 MB. Three measured problems stand between
it and use:

1. **The shipped block is a whole-file dump, not premise selection.** Across 1,305 files with ≥3
   sampled rows, **100%** have a byte-identical definition set on every row. It is "every declaration
   in this file" (mean 83.9, max 625), not the premises the step needs.
2. **The needed premise is usually absent.** Only 33.2% of rows have ≥1 tactic-cited identifier
   present in their own block, and 16.3% have all of them; a step touches **0.42** of the ~84 entries
   on average. Masking that block would remove almost nothing load-bearing, so split and dense would
   be near-indistinguishable for a trivial reason.
3. **40.1% of targets are the constant string `no goals`.**

Problem 1 is fixable, because the premises the tactic names do resolve against the *global* index even
when absent from the local file: 61.6% at the identifier level, 79.0% of rows resolving at least one,
38.2% resolving all. Only 15.6% of those resolved premises were in the file dump — confirming the dump
is nearly worthless as a retriever, and that a real oracle block has to be rebuilt as
{premises actually cited} + {sampled distractors}.

The saturation profile, measured over all 276,014 rows using true usage rather than file
co-occurrence. **These counts are inflated** — they come from the regex resolver whose false positives
are described below, and LeanDojo's ground-truth annotations give 6,904 facts at 66.1% instead. Kept
for the shape of the distribution, not the absolute numbers:

| min. true uses | declarations kept | share of all uses | exposures @10 epochs |
|---|---|---|---|
| 1 | 57,245 | 100.0% | 10+ |
| 3 | 27,098 | 92.2% | 30+ |
| **10** | **8,392** | **75.5%** | **100+** |
| 25 | 3,116 | 61.0% | 250+ |
| 100 | 702 | 40.5% | 1,000+ |

538,824 true premise uses spread over 57,245 declarations — **3.48 per declaration across the full
store**, against the 149.6 file-co-occurrences the raw field suggests. The usable configuration is the
≥10 row: about **8,400 facts covering three quarters of all premise uses**, which is the only band
that reaches §5's 100–1000 exposure target while keeping a store large enough to matter. Note the raw
top of the distribution is inflated by short-name collisions (`Lists.Equiv.symm` at 7,534,
`TypeVec.Arrow.mpr` at 4,684 are `symm` and `mpr` colliding), so proper name resolution — elaborating
against Mathlib rather than regex — is part of the work, not an optimization.

Realistic yield after requiring full premise resolution and a non-degenerate target is roughly
50–120 k rows. At the measured 837 bytes of state/tactic/result per row plus 195 bytes per rendered
premise, an 8-premise block puts that at **~55–130 M tokens** — not the ~571 M a naive read of the
field sizes gives.

**Superseded: use LeanDojo Benchmark 4 instead, which already ships the ground truth.** Heuristic
resolution plateaus — a namespace-aware resolver reaches 61.0% of identifiers at high precision
against the naive 78.9% at low precision (13.9% of short names are genuinely ambiguous), leaving
~40.6% of rows fully resolved. Closing that needs real Lean elaboration. But LeanDojo already did that
elaboration: its `annotated_tactic` field carries ground-truth premise annotations, and it ships
`state_before` / `state_after` alongside. Measured over the local `leandojo_benchmark_4` export:

| | measured |
|---|---|
| theorems / traced tactics | 122,517 / 259,580 |
| tactics with ≥1 **annotated** premise | **167,779 (64.6%)** — mean 1.61 |
| `state_after` present and ≠ `state_before` | **259,580 (100%)** |
| premise citations with a statement in `corpus.jsonl` | 360,836 of 418,704 (**86.2%**) |
| store | **66,603 premises**; 6,904 at ≥10 citations (66.1% coverage) |
| usable rows after dropping `no goals` | **95,727** |
| rendered size | **129.1 MB ≈ 59 M tokens**, fact block **22.6%** |

The 22.6% masked fraction lands near §3.4's ~17% target with no subsampling, against 95.1% for the
raw file dump. This removes the two-to-three-day resolution job entirely and replaces it with a
rendering pass. `mathlib4-state-change` remains interesting only for its larger 154,853-declaration
index, which is not worth the resolution cost when LeanDojo's 180,907-premise `corpus.jsonl` is
already annotated. Worth doing as a fifth corpus for the persistent-store property, at an
honest cost of two to three days.

The remaining sweep results do not change the picture: `tasksource/FOL-nli` (102,774 rows) shares FLD's
ephemeral-fact problem, and the large premise-selection and tactic-prediction corpora — LeanRank at
2.1 M rows, miniCTX at 614 k — fail the content-target criterion, as expected.

### 11.4 Reasoning Core and `procedural-pile` — volume without a store

A separate survey of *generators* proposed `reasoning-core/procedural-pile` (30,161,594 rows,
14.5 GB, CC-BY-4.0) as closing the token gap outright, with its `rewrite_system` and `planning` tasks
as the experimental arm. Volume is genuinely there. Measured over 1,000,000 sampled rows (3.3% of the
pile), the store is not.

**26.9% of all answers pile-wide are a bare label** — a letter, `True`/`False`, or a class name. That
includes `metamath_entailment` (2.9% of rows, 100% label answers) despite its named `r1:`/`r2:` rule
block, and `combinatorics_formula_selection` (3.1%, 100% labels).

**`rewrite_system` — the rules are unnamed in the prompt, and the pool is 63 rules.** The two surveys
disagreed on naming; the printed prompt settles it:

```
Rules:
- or(true,X) -> true
- and(true,X) -> X
- if(true,X,Y) -> X
- and(X,false) -> false
```

Bare `lhs -> rhs`, no labels. Names exist in generator metadata the model never sees, so emitting them
requires a fork. That is the smaller problem. Across 18,670 sampled rows carrying 128,572 rule
instances there are **63 distinct rules**, reused 2,041× each. The entire fact library is
**1,205 bytes — about 548 tokens.** A 370M model memorizing 548 tokens frees no measurable capacity,
which is the GenesisGeo failure (31 rules) reproduced with a number attached. The rules are also
self-evident identities: `or(true,X) -> true` states itself completely, so there is no latent
knowledge that the block is standing in for.

**`planning` — nine names carrying 8,101 meanings.** This was proposed as the one unbounded pinnable
named-fact library, at ~200k facts with ~15 citations each. In the shipped pile the action names are
positional placeholders, and the same name denotes a different schema in every randomized domain:

| name | occurrences | distinct definitions | reuse per definition |
|---|---|---|---|
| `action_0` | 17,901 | 1,987 | 9.01 |
| `action_1` | 16,422 | 1,850 | 8.88 |
| `action_2` | 13,454 | 1,555 | 8.65 |
| **all 9 names** | **64,269** | **8,101** | **7.93** |

A name is ambiguous across roughly 900 meanings, and 49.9% of full action blocks occur exactly once.
`action_0` identifies nothing, so the dense arm cannot memorize it and masking the block deletes
information available nowhere else — the FLD failure again, and again a guaranteed but uninformative
null.

This is repairable in principle: a fork that namespaces names per domain seed
(`seed17_action_0`) would make them globally unique. But the measured reuse of an actual
(name, definition) pair is **7.93**, not 15, giving ~80 exposures at 10 epochs — below §5's target —
and the resulting facts are randomly generated PDDL schemas, which is the meaningless-lemma objection
in its strongest form.

**Verdict: no task in the pile passes both tests as shipped.** The pile remains useful as generic
reasoning pretraining if that is ever wanted, and `rewrite_system`'s throughput (205–266 examples/s
per core, 3–4 core-hours per 3B tokens) is real and worth remembering if the home-grown generator of
§2 needs a faster engine. Neither substitutes for a corpus with a persistent named store.

### 11.5 Mizar clears both tests — and nothing else has

Two further surveys covered geometry/NL corpora and non-Lean ITP corpora. One produced the first
candidate to pass both the content-target and the persistent-store test.

**GenesisGeo — rejected on store size, despite being the largest thing found.** 21,799,134 examples,
**7.82 B GPT-2 tokens**, Apache-2.0, with a genuine content target (a step emits the derived
predicate `simtrir c d e c e d` before the rule name) and the strongest precedent available:
AlphaGeometry trained a **151M-parameter transformer from scratch** on this distribution to 25/30 on
IMO-AG-30. It fails for the same reason as `rewrite_system`. Newclid's `rules.txt` is **31 named
rules, 3,219 bytes, 1,276 tokens** — the entire library. Worse, roughly **47% of cited steps are
`a00`/`a01`**, Gaussian-elimination angle-chasing steps from the algebraic module that have no
theorem statement to place in a fact block at all. Volume cannot substitute for a store.

**`uw-math-ai/theorem-search-dataset` — a fact library with no derivations.** 2,892,053 rows /
631.3 MB, **1,341,083 named theorems** with full LaTeX bodies drawn from arXiv, ProofWiki, Stacks,
HoTT and others. It ships no proofs, so there is no target. Worth remembering only as raw material if
a named-statement store ever needs enlarging independently of its reasoning corpus.

**Mizar HTML export — passes both tests.** Downloaded `html2.tar.gz` (1,153 articles, 156.3 MB) and
measured it directly. The content-target property was already clear from the format: in

```
then  x in Y by A1, TARSKI:def_3;
```

the model must emit the proposition `x in Y`. Knowing the name `TARSKI:def_3` is worth nothing; you
must know that it says `X c= Y iff for x holds x in X implies x in Y` and instantiate it. This is the
cleanest content target in the survey. The measurements:

| | measured |
|---|---|
| justification steps (`by` / `from`) | **737,798** |
| steps citing ≥1 globally-named fact | **499,499 (67.7%)** |
| `::_thesis:` state annotations (a free second target) | **594,716** |
| named facts defined in corpus | **29,704** |
| distinct global names cited | **19,507** |
| total citations | **590,705** — mean **30.3** per fact |
| cited exactly once | 4,327 (22.2%) |

Store profile, which is the part that matters:

| min. citations | facts kept | share of all citations | exposures @10 epochs |
|---|---|---|---|
| 1 | 19,507 | 100.0% | 10+ |
| 3 | 12,345 | 98.3% | 30+ |
| **10** | **6,108** | **93.0%** | **100+** |
| 25 | 3,055 | 85.1% | 250+ |
| 100 | 877 | 67.6% | 1,000+ |

A **6,108-fact store covers 93.0% of all premise uses** at 100+ exposures per fact. The comparison
originally drawn here was against `mathlib4-state-change`'s 8,392 facts at 75.5%, but that figure came
from a resolver with false positives; the trustworthy comparison is LeanDojo's ground-truth 6,904
facts at 66.1% (§11.6). Mizar's citation distribution is still the strongest profile measured
alongside Metamath's — compact enough to saturate, large enough to occupy capacity, and concentrated
enough that a restricted store covers nearly all the reasoning.

The one gap is statement extraction, and it is bounded. A simple parser resolves 43.6% of citations
to a stored statement (a more careful one reached 64.2%), because theorem headers are easy but
`definition` blocks need separate handling. Definitions are **206,040 citations (34.9%) over just
3,652 distinct names**, so that is a small, high-value target. Critically, **all 924 referenced
articles are present among the 1,153 files, so the resolution ceiling is 100%** — the shortfall is
parser work, not missing data.

Precedent is unusually close: Urban & Jakubův trained **GPT-2 117M from scratch on exactly these
files** on a single GTX 1080. A 370M run is a scaled-up replication, not a leap.

Two notes. The snapshot is March 2020; current MML is 1,498 articles, so regenerating buys ~30% more.
And Mizar notation will fragment badly under a general BPE (`c=`, `\/`, `::_thesis:`), so the custom
tokenizer of §6 is required rather than optional — the ~45M raw-token estimate will not hold
otherwise.

**Scale-up path: `prf2.tar.gz`**, E/ENIGMA TPTP proofs over MPTP-Mizar — 25,060 proof files,
627.3 MB, ~188 M tokens, ~1.5 M derived clauses, and a **native fact block already at the top of every
file** (mean 10.0 named axioms with full statements). The target is the derived clause, which is
unification plus substitution over the parents' content, so it is the strongest content-target pass of
anything surveyed. The cost is machine-mangled notation (`k3_xboole_0`, `v1_relat_1`), further from
human mathematics. Only 25,060 of the 43,717 available ATP proofs are in this snapshot.

**Where this leaves the portfolio.** Of everything surveyed across four sweeps, exactly one corpus
passes both tests outright — Mizar — with `prf2` as its scale-up and `mathlib4-state-change` (§11.3)
as a distant third after two to three days of premise resolution. Metamath and IsarStep remain the
incumbents. The recurring failure is not format compliance, which is common, but a persistent named
store: FLD at 1.01 reuse, Reasoning Core at 63 rules, GenesisGeo at 31.

### 11.6 Measured portfolio — replaces §11

Everything below was computed locally from the artifact. Token counts are GPT-2-equivalent unless
noted, because the corpora use incompatible native tokenizations and mixing units caused three of the
errors listed at the end of this section.

| Corpus | Store | ≥10 citations | Coverage | Tokens | Passes all three tests |
|---|---|---|---|---|---|
| **Metamath** (set+iset+nf) | 45,881 | 9,835 | **91.8%** | 101 M symbol / 115 M GPT-2 | yes |
| **Mizar** html2 | 19,507 cited (29,704 defined) | 6,108 | **93.0%** | 23 M rendered, 71 M raw | yes |
| **MPTP / E prf2** | native axiom block | — | — | 170.5 M native / 299 M GPT-2 | yes, after filtering |
| **LeanDojo** Benchmark 4 | 66,603 | 6,904 | 66.1% | 59 M | yes |
| **IsarStep** (fact-bearing 28.1%) | 60,691 synthetic IDs | 7,483 | 64.4% | 79 M | yes, with hashed names |
| PISA / AFP extraction | 136,766 | 10,088 | 60.6% | 487 M (66 M content) | no — 13.4% content targets |
| TPTP v9.3.0 | native `.ax` files | — | — | 0 usable | no — ships no proofs |
| NaturalProofs | 3,907 per-step | 367 | 57.7% | 48 M mostly prose | no — 30.4% attribution |

**Maximum composition, ignoring notation compatibility: ~623 M GPT-2 tokens** (Metamath 115 + prf2 299
+ Mizar 71 + LeanDojo 59 + IsarStep 79). Relaxing the content-target rule to admit PISA raises the
ceiling to ~1,110 M. At 10 epochs the 623 M composition gives ~6.2 B training tokens.

Only four independent libraries exist, and several corpora are alternative renderings of one of them:

- **Mizar MML** — html2 and prf2 share 10,802 theorems by number. Verified by hand that matched names
  state the same result (`CARD_3:100` ↔ `t100_card_3`). Their proofs share no text, average 9.9 versus
  11.3 steps, and agree on 70.0% of Mizar's cited premises, so both are worth keeping.
- **Isabelle AFP** — IsarStep and PISA share 2,687 theory files, covering 73.6% of IsarStep's examples.
- **Lean mathlib** — LeanDojo contains 96.2% of `mathlib4-state-change`.
- **Metamath** — standalone.

**`prf2` needs a premise filter.** 62.7% of its 240,628 axiom citations are FOL-translation
bookkeeping — `dt_` type declarations (19.3%), `redefinition_` (11.9%), `cc_`/`fc_` cluster
registrations — not mathematical facts. Only 27.2% map onto a Mizar fact (17,266 distinct). Drop the
bookkeeping prefixes and keep the `t`/`d` axioms before treating its block as a fact store.

**IsarStep is usable after all.** The earlier rejection conflated it with PISA. It is a separate ICLR
2021 dataset, 821,933 train examples, and `IsarStep_ascii/train_g.src` renders its `<used_global_facts>`
block in readable Isabelle. Its facts ship unnamed, but hashing the term string gives stable
identifiers: 61,101 distinct facts raw against 60,691 after alpha-normalising binders, so naive
hashing over-splits by only **0.7%** and no normalisation pass is needed. Recovering the *real* names
is not practical — matching against 184,993 named lemma statements from the local AFP and Isabelle
sources resolves **0.3%**, because IsarStep's statements are elaborated rather than source text.

**Corrections to figures used earlier in this document.**

| Figure | Previously | Measured |
|---|---|---|
| Mizar rendered size | ~200–300 M tokens | 23 M rendered · ~40 M at full resolution · 71 M raw |
| IsarStep | native named F.5 block, ~410 M | block in 28.1% of examples, unnamed, 228 M full / 79 M usable |
| `mathlib4-state-change` store | 8,392 facts at 75.5% | superseded by LeanDojo's 6,904 at 66.1% |
| NaturalProofs attribution | 84.7% of steps | 30.4% of steps |
| TPTP | 80 M tokens | problem statements only; no `Solutions/` directory exists |
| prf2 vs Mizar | same mathematics, additive | 48.5% theorem overlap; fact stores 27.2% shared |

One recurring parser trap, hit three times: Mizar definition markers sit mid-line and carry a trailing
suffix — `:: deftheorem defines c= TARSKI:def_3_:_` — so a regex anchored after `::` or at the digit
silently returns zero definitions. Fixing it moved prf2's measured store overlap from 13.4% to 27.2%.

---

## 12A. Final selection

Four independent libraries, five artifacts. Everything below was measured locally.

### 12A.1 The criterion that reordered the list

How much of the target is already present in the input — the share of the answer a
model can produce by copying rather than deriving. This was never measured until now
and it changes two placements:

| corpus | target | paste share | verdict |
|---|---|---|---|
| **Mizar** | a fresh proposition | **0.332** | strongest content target measured |
| **Metamath** | the derived formula | **0.503** | strong |
| **Magnushammer** | resulting proof state | **0.688** | acceptable |
| ~~LeanDojo~~ | resulting proof state | **0.891** | **drop** — 36.5% of transitions carry ≥95% of the answer, and 41.2% of targets are the constant `no goals` |

LeanDojo is dropped. It has the weakest target, the smallest store of the Lean/Isabelle
group (6,904 facts at 66.1%), and only 59 M tokens. Magnushammer beats it on every axis.

### 12A.2 Magnushammer replaces IsarStep

Both are mined from the Archive of Formal Proofs plus the Isabelle standard library, so
they are renderings of one library, not two corpora. Magnushammer wins outright:

- **Named facts with statements.** Its `premises` dict maps a name to
  `[fully_qualified_name, statement]`, and the statements carry `fixes` (conditions) and
  `shows` (conclusion) — satisfying the "name, conditions, implications, formula"
  requirement directly. IsarStep's facts are anonymous and need hashed IDs.
- **Bigger store.** 133,235 module-qualified facts, 14,306 at ≥10 citations covering
  62.8%, against IsarStep's 7,483 at 64.4%.
- **Better coverage of its own corpus.** A fact block on 25.5% of 2,648,178 transitions,
  against IsarStep's 28.1% of 821,933 examples.

**Drop the `local.` premises.** 93.0% of the unqualified bucket is `local.`-prefixed —
Isabelle proof-local context, not global facts. `local.assms` alone appears in 16,863
proofs carrying 9,438 different statements, which is the FLD failure exactly. Keep the
~7% that is not `local.`-prefixed (`List.`, `Matrix.`, `axioms.`, ~30,000 uses): those
are genuine global names the module-qualified regex missed on capitalisation.

### 12A.3 The final list

| # | Corpus | Library | Tokens | Store at ≥10 | Coverage |
|---|---|---|---|---|---|
| 1 | **Metamath** set+iset+nf | Metamath | 115 M | 9,835 | 91.8% |
| 2 | **Mizar** html2 + thproofs | Mizar MML | 116 M | 9,023 | 94.8% |
| 3 | **MPTP prf2** | Mizar MML | 299 M | 2,952 | 73.9% |
| 4 | **ENIGMA** mzr01–09, deduplicated | Mizar MML | ~175 M | 1,387 (mzr01) | 66.1% |
| 5 | **Magnushammer** | Isabelle AFP | ~789 M | 14,306 | 62.8% |
| | **Total** | | **~1.49 B** | | |

**ENIGMA revised down from ~500 M.** The five archives are five separate ENIGMA/E runs
over the same 57,880-problem MPTP set, each succeeding on ~13,700. They duplicate each
other heavily: mzr01 and mzr03 share 10,705 theorems at mean Jaccard 0.763, with **26.9%
byte-identical clause sets** and 40.7% at ≥0.9 similarity. Only 16.5% are genuinely
different derivations. Deduplicating to ~19–20 k distinct theorems plus the genuinely
different alternatives gives ~332–440 MB, or **150–200 M tokens**.

ENIGMA is *not* redundant against Mizar html2+thproofs, though: only 47.3% of its proved
theorems appear in that snapshot, and 7,790 do not, because ENIGMA covers the full MML
while html2 is 1,153 articles and thproofs is theorem-proofs only. The text differs
completely in any case — declarative Mizar versus TPTP resolution.

**A correction.** Earlier notes claimed ENIGMA's derivations are 5.4× longer than prf2's
(61.5 clauses against 11.3). That is wrong: the prf2 count used a regex requiring `fof(`
with no space, and prf2 writes `fof ( c_0_8 ,`. Measured consistently the two are
comparable — **ENIGMA 50.3 clauses per proof, prf2 55.6**. ENIGMA's real advantages are
its lower bookkeeping fraction (34.6% against 62.7%) and the theorems html2 lacks.

Genuinely independent mathematics: **Metamath 115 M + Mizar 116 M + Isabelle 789 M ≈ 1.02 B**.

### 12A.5 Epochs

A fact cited C times reaches 100 exposures after ⌈100/C⌉ epochs, so "how many epochs" is
a coverage lookup at a shifted threshold:

| epochs | min citations | Mizar facts / coverage | Metamath facts / coverage |
|---|---|---|---|
| 2 | 50 | 2,703 — 81.3% | 2,976 — 79.4% |
| 3 | 34 | 3,815 — 85.7% | 3,958 — 82.9% |
| 4 | 25 | 4,859 — 88.6% | 5,021 — 85.5% |
| **5** | **20** | **5,718 — 90.5%** | **5,948 — 87.2%** |
| 10 | 10 | 9,023 — 94.8% | 9,835 — 91.8% |
| 20 | 5 | 13,254 — 97.6% | 16,162 — 95.3% |

**Five epochs is the right target.** At 1.49 B tokens that is ~7.5 B seen, which lands on
Chinchilla-optimal for a 370M model (~7.4 B), while still saturating the facts behind
87–90% of all premise uses. Doubling to 10 epochs costs 2× the compute and buys about
four percentage points of coverage. Even three epochs retains 83–86%; the curve is flat
above ~4.

### 12A.6 Magnushammer's paste share is filterable

Its 0.688 mean is the weakest target among the kept corpora, but the distribution is
broad rather than concentrated:

| paste band | transitions | share |
|---|---|---|
| below 0.5 | 456,083 | 20.1% |
| below 0.6 | 721,733 | 31.7% |
| below 0.7 | 1,046,374 | 46.0% |
| 0.9–1.0 | 465,439 | 20.5% |

Filtering to transitions below 0.6 keeps 721,733 examples at a paste share comparable to
Metamath's 0.503, at the cost of about two thirds of the corpus. That trade is worth
making if Magnushammer is used for the reasoning signal rather than for volume.

### 12A.4 Storage

| artifact | download | extracted |
|---|---|---|
| Metamath `.mm` sources | 0.07 GB | 0.07 GB |
| Mizar `html2` + `thproofs` | 0.05 GB | 0.26 GB |
| MPTP `prf2.tar.gz` | 0.05 GB | 0.66 GB |
| ENIGMA `mzr01–09` (5 archives) | 1.35 GB | 17.50 GB |
| Magnushammer `all_data.json` | 2.33 GB | 2.33 GB |
| **Total** | **3.85 GB** | **20.82 GB** |

ENIGMA dominates the extracted footprint: each archive expands ~12× because 76% of its
files are failed proof attempts with no derivation. Filtering to the 23.7% that contain a
complete SZS derivation before storing brings the total to roughly **6 GB extracted**, and
deduplicating across runs (26.9% of shared proofs are byte-identical) takes it to about
**4.5 GB**.

Optional additions: CASC-27 (1.22 GB / 9 GB) for a HOL4 vocabulary, and TPTP v9.3.0
(0.94 GB / 11 GB) only if the CASC route is taken.

---

## 12. Where this leaves the earlier two options

| | Synthetic | Metamath |
|---|---|---|
| Estimated dense accuracy | >99% at depth 4; 75–90% even at depth 16 | 30–40% whole-proof, 70–85% per-step |
| Basis for the estimate | extrapolated from iGSM plus measured branching | published 160 M from-scratch baseline at 29.22% |
| In the 30–70% band? | **not without redesigning for branching** | **yes, on the whole-proof metric** |
| Corpus | unbounded | 97–190 M tokens (~50–100 epochs at Chinchilla) |
| Fact-block share | 48% | 21% |
| Contamination | impossible | mathlib-adjacent text is widespread |

The ordering argued for earlier — synthetic first because its gate is near-certain — **no longer
holds.** The measurements invert it: synthetic is near-certain to sit at the *ceiling*, which fails
the gate just as surely as the floor, and fixing that needs generator work whose outcome is unknown.
Metamath has a directly comparable published number that lands in the band. Run Metamath first, and
treat synthetic as the controlled follow-up once branching is calibrated.
