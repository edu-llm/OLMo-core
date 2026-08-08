# Independent verification of the corpus survey

**Date:** 2026-08-01. **Method:** every figure below was produced by a second implementation
written from the source format, not by re-running `scripts/mm_*.py`, `scripts/mizar_*.py` or
`scripts/isarstep_*.py`. Scripts live in `scripts/verify/`.

---

## 0. Volume: can any of this reach 700 M tokens?

Yes, but only under a step-level rendering. Corpus size here is not a property of the corpus, it
is a property of how you render it — the same five Metamath databases span 43× depending on the
choice.

| Metamath ×5, rendering | gpt2 tokens | each example contains |
|---|---|---|
| theorem-level, essential steps | 133 M | one example per theorem |
| theorem-level, all steps | 215 M | adds syntax-construction steps |
| step-level, label + result only | 236 M | minimal framing |
| **GPT-f proofstep, essential steps** | **277 M** | goal + the cited fact in full + the substitution |
| **GPT-f proofstep, all steps** | **440 M** | same, keeping the 37% that is grammar |
| GPT-f proofstep + sub-goal extraction | 5,671 M | every intermediate result also becomes a goal |

The GPT-f rows matter because that format already inlines the fact block: each step states the
cited theorem's hypotheses and conclusion in full alongside the substitution. That is the shape the
experiment needs, and it is worth 2–3× the naive rendering.

**Published precedent confirms the upper end.** Polu & Sutskever (arXiv:2009.03393) report their
`set.mm` proofstep dataset at ~1 B tokens, and trained a 160 M model from scratch on 18 B tokens of
it over ~18 epochs, reaching **29.22%** whole-proof success against MetaGen-IL's 21.16%. Their
per-step cost exceeds mine because their goals carry accumulated subgoal hypotheses and their set
is augmented with synthetic arithmetic proofs. Verified against the paper text, not recalled.

| Route to 700 M | tokens | cost |
|---|---|---|
| TSTP scrape | 726 M floor, ~4 B central | ~350 k CGI fetches; fragmented store |
| **CoqStoq + PISA** | **1.00 B** | resolve CoqStoq premise IDs; PISA already local |
| PISA + prf2 | 832 M | both already on disk; two unrelated syntaxes |
| Metamath + Mizar + prf2 | 825 M | three notations in one 370 M model |
| Metamath alone, GPT-f format | 440 M | short of the floor |
| Metamath + sub-goal extraction | up to 5.67 B | quadratic redundancy, not new mathematics |
| Regenerate at full MPTP scale | 531 M (+328 M from TPTP) | ATP compute; filtering for named steps costs ~85% |

Note on the 825 M combination: prf2 is Mizar's mathematics in TPTP syntax, so it is not new
mathematics — but it is an entirely different token stream over the same theorems, which is the
"diversity of use" the design asks for. Their fact stores only join at 22.3% of prf2's uses,
because most prf2 axioms are MPTP-generated type-system artefacts with no Mizar counterpart.

### TSTP: the biggest lever, and why it is not the recommendation

The recorded verdict on TPTP — "no Solutions directory, producing a target needs running E or
Vampire over 11,105 problems" — was structurally right and factually wrong on every number. TPTP
v9.3.0 holds **26,990 problems**, not 11,105, and **20,415 of them already have machine proofs**
recorded in the TSTP solution library: 354,145 stored files, of which ~327,700 are real
derivations in exactly prf2's format. So the work is a scrape, not an ATP campaign. TSTP is
CGI-only — `TSTP.tgz` and `Solutions.tgz` both 404 — so it means roughly 350,000 fetches.

Two measurements argue against making it the primary corpus.

**Named-axiom density falls with proof length.** My own probe of 11 derivations averaging 34 steps
found **33.8%** of inference steps consume a named axiom; a sweep of 55 derivations averaging 185
steps found **4.6%**. Both are correct for their sample — ATP proofs consume their named axioms at
the leaves and then chain internal `c_0_N` labels, so density decays as the derivation grows. Since
TSTP's token mass sits in the long tail (median 2,215 tokens, mean 19,752), the corpus-weighted
figure is near the low end. Selecting short proofs raises density but discards most of the volume.

**The store fragments.** Measured on the local distribution: only **37.6% of TPTP problems include
a shared `.ax` axiom file** at all. The other 62.4% carry problem-local axiom names, which is the
FLD failure mode — a name that means something different in every problem. Where sharing does
exist it is good (1,989 axiom files, mean 93.3 problems each), but the namespace as a whole is
dominated by large ontologies rather than a curated mathematical library, and only 43.7% of
problems are in mathematical domains. This is exactly the property that makes prf2 better-behaved
than TSTP at large: prf2's names are all MPTP renderings of Mizar MML names, one global library.

CoqStoq (474 M tokens of derived goal states, ~36% of steps carrying premise IDs) and PISA/Isabelle
(526 M tokens of proof states, ~35% of Isar steps citing premises, already at
`/tmp/dscount/afp_ext`) reach 1.00 B with roughly seven times TSTP's citation density and no
scraping. That is the recommendation.

---

## 0.05 Two renderings that look like volume and are not

**GPT-f's proofstep format is unusable for this experiment, despite being the largest Metamath
rendering.** Its target field is verbatim
`[[ |- A = B  |- C = B ]] |- A = C \ {{ A : ... }} {{ B : ... }} {{ C : ... }}` — the cited
theorem's hypotheses and conclusion in full, plus the substitution, with **no premise name
anywhere**; the `proof_label` field names the theorem being *proved*, not the one being cited.
There is therefore no block to mask and no name to leave behind. Worse, the statement is the
prediction target, so masking it deletes supervision rather than hiding a fact. A head fact
averaging 98.9 citations would be seen unmasked 98.9 times per epoch by the split arm — identical
to dense — against 1 per epoch under a named-block rendering. **Strike the 440 M figure.** The
usable Metamath renderings are 133 M (theorem-level) and 236 M (step-level), both keeping
statements in a maskable block and names in the body.

The general rule: *the derivation body may cite a fact only by name.* Any rendering that restates
the fact inline nullifies the manipulation regardless of how many tokens it buys. This is the same
defect as IsarStep — the fact present as text but not as a name.

**set.mm's git history is mostly minimisation churn, not alternative proofs.** The idea was that
30 years of revisions hold genuinely different derivations over an unchanged store. Sampling 19 of
the 8,222 commits touching `set.mm` (0.2% of history) gives 57,151 labels ever seen and **124,113
distinct proof strings** against 47,611 theorems in the current file — apparently 2.6×, and still
climbing linearly. But comparing consecutive versions of the same theorem:

| | |
|---|---|
| consecutive distinct-proof pairs | 66,962 |
| mean Jaccard of the cited-premise sets | **0.878** (median 0.939) |
| pairs sharing ≥80% of premises | **85.9%** |
| mean proof-length ratio | 0.951 |
| pairs differing in >20% of premises | **14.1%** |

So the honest yield is about **+20% of genuinely different derivations**, not 2.6×. The rest is
near-duplicate data that would raise fact exposure without adding reasoning diversity — which in
this design also worsens the split arm's goal-line leak. Worth harvesting with a Jaccard filter;
dangerous to include wholesale.

**The local MPTP release is MPTP2078, not the full library.** `/tmp/dscount/mptp` holds 2,078
`bushy` problems (9 MB, mean 30.2 named axioms) and the same 2,078 `chainy` problems (45 MB, mean
85.1 axioms). Axiom names are MML names — `t6_boole`, `d4_relat_1`, `cc1_relat_1` — so the store is
one global library, confirming the property that makes prf2 better behaved than TSTP. But the
regeneration route needs the full ~43,717-problem MPTP, which is not local.

---

## 0.06 Named-premise density: the axis that decides everything, measured across the survey

The share of derivation steps that cite a named library fact is what the split manipulation
actually operates on. Volume without it is worthless. Measured across everything surveyed:

| corpus | steps citing a named library fact | |
|---|---|---|
| **Metamath, theorem-application steps** | **100%** | by construction — every derivation step is a library application |
| Mizar, `by` justifications | **69.7%** | human declarative |
| CoqStoq / PISA proof states | ~35–36% | |
| TSTP, short derivations | 33.8% | my probe, mean 34 steps |
| AFP Isar, proposition-stating steps | 15.5% | |
| prf2, ATP over Mizar | 14.1% | |
| TSTP, long derivations | 4.6% | mean 185 steps |
| SMT-LIB / Alethe | 2.2% ceiling | 97.79% of 10.5 M references are intra-proof step ids |
| OpenTheory elaborated to Dedukti | **0.1%** | citations are hash-cons registers, not library names |
| CPF / CeTA termination certificates | 0% | rules are inlined at every use and never named |
| OpenTheory articles, raw | 0% | a numeric register file — 7,153,822 `def` against exactly 7,153,822 `remove` |

### Elaborated HOL proof terms: the predicted counterexample, and why it fails

The one route that should have broken the anti-correlation was taking a large *procedural* library
and elaborating its proofs into explicit declarative form. OpenTheory pushed through Holide into
Dedukti does exactly that, at real scale — 676 M gpt2 tokens measured from 17% of the corpus,
projecting to 3.4–4.0 B (~1.5 B excluding the 54.5% that is ARM/M0 machine-code verification), with
every step carrying an explicit statement and `dk check` verifying a 387-theorem prefix clean.

Measured directly on the generated `.dk` files, it fails the store test outright:

| | measured across five generated files |
|---|---|
| declarations that are auto-generated per-file names (`type_N`, `term_N`, `thm_N`) | **~100%** |
| citations pointing at a HOL4 library name | **0.1%** |
| citations pointing at an auto-generated per-file name | ~25% |
| citations pointing at a kernel primitive | ~24%, over **6 distinct** primitives |
| citations pointing at a local lambda binder | ~50% |
| auto names shared between `hol4-ring` and `hol4-sort` | **15,933** — the same name, different statements |
| HOL4 library names shared between those files | ~50 |

`thm_500` exists in most files and means something different in each. That is the FLD failure — a
positional label denoting a fresh proposition every time — reproduced at a scale of tens of
thousands of names per file. HOL4's real 52,790-name library survives only as *constants*; the
theorem-level citations have become hash-cons registers. hol2dk inherits the same defect by
construction: its emitted names are `lem<k>`, generated per file.

**This is not bad luck, it is the mechanism.** Elaboration is what produces the content target —
you get an explicit statement at every step precisely by expanding proofs down to primitive
inferences — and primitive inference chains have no library names to cite, because the library
lives at a coarser granularity than the steps. Getting the target destroys the store.

**The ordering below is therefore monotone in step granularity, not just in how machine-generated
the proof is.** Metamath sits at the top for a structural reason: it defines its primitive step to
*be* a library-theorem application, so there is no finer level to expand into and no way for the
names to disappear. That is the property to select on, and only Metamath and Mizar have it.

**The ordering has a mechanism, and it is monotone in how machine-generated the proof is.** Human
declarative proofs cite the library because that is how a person writes a proof. Machine proofs
derive a long chain of internal lemmas and touch named axioms only at the leaves, so density falls
as derivation length grows — which is why TSTP measures 33.8% on short proofs and 4.6% on long
ones. The consequence for this project: machine provers make volume cheap and manipulation surface
expensive, and the two cannot be bought together. SMT-LIB is the extreme case — roughly 25 B tokens
available from one logic, with a hard ~2.2% ceiling that no amount of `:named` annotation can lift,
because the ratio is a property of the proof format rather than of the benchmarks.

### The `(reuse)` policy is a bigger decision than the notes assumed

Metamath's 100% holds only for steps that *apply a theorem and produce a new formula*. Decoding all
of `set.mm` gives 9,710,752 stack operations:

| operation | count | share |
|---|---|---|
| global named fact applied | 3,658,251 | 37.7% |
| backreference to an earlier step | 3,652,862 | 37.6% |
| local hypothesis pushed | 2,198,903 | 22.6% |
| optional `$f` variable | 200,736 | 2.1% |

Restricted to operations carrying a `|-` formula, **68.9% are backreferences** and only 25.8% are
named applications. Backreferences produce no new formula — they re-push something already derived
— so they are bookkeeping, and the token measurements in §0 render only the 1,365,902 theorem
applications, which are 100% named. But the open item in `metasynth-data-e1.md` §7.2, "decide the
`(reuse)` policy", is therefore load-bearing: emit reuses as `reuse line N` and you add unnamed
lines at 2.7× the count of the named ones, diluting Metamath's density from 100% to 25.8% and
handing the model a large, easy, unnamed sub-task. Inlining the reused formula instead keeps
density at 100% at the cost of a larger corpus.

---

## 0.1 Corrections to my own first-pass numbers

Every claim was re-derived by a different method. Six changed.

| Figure | First pass | Corrected | Why it was wrong |
|---|---|---|---|
| prf2 steps consuming a named axiom | 13.0% | **14.1%** | a non-greedy regex truncated nested `inference()` terms |
| prf2 tokens | 306 M | **328 M** | bytes/token taken from one file (2.15); over 40 files it is 2.00 |
| AFP tokens | 111 M | **141 M** | same single-file error; the real rate is 2.10, not 2.66 |
| Mizar proofs / body tokens | 89,706 · 43 M | **79,377 · 48 M** | the extraction regex did not handle nested `proof … end;` |
| Metamath copy fraction | 0.581 | 0.581 set / **0.405 LCS** | the set measure ignores order and repetition |
| IsarStep name recovery | "essentially 0%" | **12.4% of names, 14.4% of citations** | the probe searched only the AFP and used a normaliser that ate binder variables |
| NaturalProofs store "wrong by 49×" | 367 is wrong | **367 is exact but narrowly scoped** | the recorded 367 counts per-step `c =` targets at ≥10 citations; my 18,067 counts every in-proof link. Both correct, different definitions |

Everything else was reproduced. In particular the Metamath store figures were confirmed to the
digit, and the two independent Metamath parsers disagreed on **zero** of 50,588 assertions.

### A disagreement I could not resolve

Two independent re-measurements of LeanDojo's `state_after` novelty differ by roughly 4× — 20.6% of
target tokens new under an order-preserving alignment against 5.4% under a token-count ratio. The
gap is a difference of measure, not of fact: both agree the target is overwhelmingly a copy, and
both agree on the decisive number, that only about **1.5% of target tokens are attributable to the
fact block**. I did not spend further compute resolving it because LeanDojo cannot reach the volume
floor under any rendering.

**Validation of the instrument.** The Metamath measurements rest on a proof decoder and
substitution engine written from the spec. It was checked by running the full stack machine over
3,000 randomly sampled `set.mm` proofs and requiring each to reduce to its own statement:
**3,000 of 3,000 verified, 0 failures.** Every non-assertion reference it decodes is an optional
`$f` variable declaration, and no local `$e` hypothesis is ever miscounted as a global premise.

---

## 1. The criteria, as testable predicates

Restated from the experiment description so each corpus can be scored mechanically.

| | Test | Fails if |
|---|---|---|
| T1 | **Content target** — the model emits a derived formula | the target is a name, label, tactic, or True/False |
| T2 | **Named citation** — the derivation cites facts by name mid-proof | facts are anonymous, so there is nothing to leave behind when the block is masked |
| T3 | **Persistent store** — one name means one statement corpus-wide | names are positional or regenerated per example |
| T4 | **Saturable** — enough citations per fact that dense would memorise it | coverage at ≥10 citations is low |
| T5 | **Self-contained facts** — name, conditions, implications and formula | the block entry is meaningless without hidden context |
| T6 | **Held-out facts are constructible** — a fact can be removed with its dependents and still leave eval problems | removal destroys the corpus, or the statement is guessable from the name |
| T7 | **Volume** — enough tokens for a from-scratch 370M run | under ~100M tokens means dozens of epochs |

T5 and T6 are new. T6 is the `{used fact, new fact}` axis of the 2×2×2 design and nothing in the
prior survey tested it.

---

## 2. Metamath — figures reproduce exactly

Merged `set.mm + iset.mm + nf.mm`, block composition counted as distinct premises per theorem:

| Quantity | Recorded | Measured | |
|---|---|---|---|
| distinct cited logical facts | 45,881 | **45,881** | exact |
| block citations | 1,159,859 | **1,159,859** | exact |
| facts at ≥10 citations | 9,835 | **9,835** | exact |
| coverage at ≥10 | 91.8% | **91.8%** | exact |
| coverage curve | 100 / 97.4 / 95.3 / 91.8 / 85.5 / 71.5 | identical | exact |
| fact-block share | 21.0% | **20.7%** | confirmed |
| essential-step corpus | 101 M (3 DBs) | **98 M for set.mm alone**, ~123 M for three | low by ~20% |

There is a second, better convention the survey did not report. Counting every *use* rather than
distinct premises per theorem gives 1,719,431 citations and **10,949 facts at ≥10 covering 94.4%**.
Both are defensible; distinct-per-theorem is the right one for sizing the block, every-use is the
right one for counting how often the model reads the name.

### 2.1 New: 58.4% of Metamath facts are not self-contained (T5)

24,125 of the 41,326 cited facts in `set.mm` carry `$e` hypotheses, and those facts account for
**68.8% of all premise uses**. The rendering used in the plan prints only the conclusion, which for
an inference rule says nothing:

```
plan            syl : |- ( ph -> ch )
complete        syl : |- ( ph -> ps ) & |- ( ps -> ch )  =>  |- ( ph -> ch )
```

Rendering complete facts costs **1.68×** the block, moving the fact-block share from 20.7% to
**30.5%** and the corpus from 98 M to 112 M tokens. This is a stronger manipulation, not a weaker
one, but it invalidates the worked examples as drafted.

### 2.2 New: a quarter of Metamath facts are guessable from their names (T6)

Metamath labels are systematic — `eqtr`, `eqtri`, `eqtrd`, `3eqtri`. For 800 held-out facts, taking
the nearest surviving name-neighbour and comparing statements token-wise:

| similarity | share of held-out facts |
|---|---|
| ≥0.50 | 62.2% |
| ≥0.80 | 25.1% |
| ≥0.90 | 9.0% |

A quarter of held-out facts have a surviving neighbour that is 80% of the answer, so the new-fact
eval must screen for name-novelty rather than sample at random.

### 2.3 New: held-out facts are affordable if chosen from the tail

Holding out a fact means dropping its own proof and every proof citing it.

| selection | proofs lost | eval problems |
|---|---|---|
| 750 sampled uniformly from facts cited ≥3 | **48.1%** | — |
| 250 sampled uniformly | 20.3% | — |
| **750 least-guessable facts cited 2–8 times** | **6.2%** | **2,234** |

Uniform sampling is unaffordable; tail-plus-novelty selection is cheap. Mean guessability of the
chosen 750 is 0.169.

### 2.4 New: the content target is real but partly a paste

Over 85,627 logical steps from 3,000 verified proofs, the mean share of result symbols already
present in the cited fact is **0.581**, and **9.4% of steps produce a result containing no symbol
that was not already in the fact**. Real work, but easier than Mizar.

---

## 3. Mizar — headline numbers hold, several details are off

| Quantity | Recorded | Measured | |
|---|---|---|---|
| justification steps | 737,798 | **737,798** | exact |
| `::_thesis:` annotations | 594,716 | **594,716** | exact |
| steps citing ≥1 global fact | 499,499 (67.7%) | 514,146 (**69.7%**) | close |
| total citations | 590,705 | 587,869 | close |
| distinct names cited | 19,507 | **22,874** | +17% |
| facts defined | 29,704 | **33,543** | +13% |
| facts at ≥10 | 6,108 (93.0%) | **6,203 (92.0%)** | confirmed |
| definitions' share of citations | 34.9% | 34.6% | confirmed |
| referenced articles all present | yes, 924 | yes, **1,064** | confirmed |
| rendered size | 23 M, ~100 M ceiling | **52 M proof-level, 71 M raw ceiling** | both wrong |

### 3.1 Mizar beats Metamath on every design axis that was not being measured

| | Metamath | Mizar |
|---|---|---|
| facts that are closed, self-contained statements (T5) | 41.6% | **96.4%** |
| target words already present in the cited fact (T1) | 0.581 | **0.332** |
| steps that are ≥90% paste | 12.0% | **0.6%** |
| held-out facts guessable from a name-neighbour at ≥0.8 (T6) | 25.1% | **3.9%** |
| cost of holding out 750 facts (T6) | 6.2% of proofs | **0.56% of steps** |
| corpus volume (T7) | 98–112 M | 52–61 M |

Mizar names are `TARSKI:def_3` — opaque numbering, so a held-out statement cannot be inferred from
the name. Metamath names are compositional, which leaks. The one place Metamath wins is volume, and
that it ships a verifier.

The open cost on Mizar is statement extraction: my parser attaches a statement to **55.9%** of
citations (the survey reported 43.6% simple / 64.2% careful). All 1,064 referenced articles are
present, so the ceiling is genuinely 100% and the shortfall is parser work.

---

## 4. IsarStep — rejected for the right reason, but the recorded store is wrong

| Quantity | Recorded | Measured | |
|---|---|---|---|
| examples | 821,933 | **821,933** | exact |
| global block populated | 28.1% | **28.1%** | exact |
| distinct facts | ~114,000 | **61,102** | recorded figure ~1.9× too high |
| facts at ≥10 | 1,816 | **7,489** | recorded figure 4× too low |
| coverage at ≥10 | 41.5% | **64.2%** | recorded figure 23 pp too low |
| tokens | 322 M | **256 M** | high by 26% |
| facts are unnamed | claimed | **confirmed: 6 of 374,858 instances are identifiers (0.00%)** | confirmed |

The coverage error has a specific cause: the curve was computed on a 30% sample of `train_g`, which
divides every fact's citation count by about 3.3 and therefore pushes facts below the ≥10 threshold
that belong above it. Subsampling is not valid for a saturation curve.

The rejection still stands, and on the decisive ground: the facts are serialised Isabelle terms with
no names, so the split manipulation cannot be expressed. `train.meta` field 2 does carry a name
(`thm_yun_factorization_main2125`) but it names the *enclosing* theorem being proved, not the cited
premises, and it is auto-generated positional text.

---

## 5. prf2 / MPTP — a new disqualifying measurement

"Same maths as Mizar" is **confirmed exactly**: 1,111 of 1,111 prf2 problem articles sit inside
Mizar's 1,153.

Two things the survey never measured:

- **Only 14.1% of inference steps consume a named library axiom.** Of 205,750 derived terms in
  3,000 proofs, 176,689 operate purely on internal clause numbers (`c_0_13`, `c_0_14`). The fact
  block is 10.7% of bytes. The split manipulation would touch about one step in seven.
- **Full-corpus store**: 31,481 named axioms, 240,628 uses, 2,952 facts at ≥10 covering 73.9%.
  Corpus is **328 M gpt2 tokens**, not the recorded 171 M.

The shared-store idea is weaker than it looks. 61.2% of prf2 axiom names have the MPTP `t*`/`d*`
shape, but only **22.3% of prf2's fact uses resolve to a Mizar statement**; the rest are
MPTP-generated type-system axioms (`dt_`, `fc_`, `cc_`, `redefinition_`, `fraenkel_`) with no Mizar
counterpart. Joining the two stores gives 6,776 facts at ≥10 covering 91.6% — essentially unchanged
from Mizar alone.

---

## 6. AFP / PISA — rejection confirmed, recorded size badly wrong

Scanned 2,500 of 10,351 AFP `.thy` files directly.

- **32.8%** of proof steps state a proposition; **67.2%** are bare tactic/apply lines. "Target
  mostly a tactic" is confirmed.
- Only **15.5%** of the proposition-stating steps carry a named lemma citation. The intersection —
  steps that are both a content target and a named citation — is about **5.1%** of all steps.
- AFP source is **141 M gpt2 tokens**, not the recorded 487 M.

---

## 6.1 NaturalProofs — the recorded store is wrong by a factor of 49, but volume kills it anyway

| Quantity | Recorded | Measured | |
|---|---|---|---|
| store | 367 facts | **18,067 in-proof link targets** (5,616 via `ref_ids`) | 49× larger |
| facts at ≥10 citations | — | **2,569 covering 74.4% of uses** | better than LeanDojo's 66.1% |
| per-step attribution | 30.4% | **35.4%** of `c =` fields carry a wiki link | confirmed |
| volume | 48 M | **11 M gpt2 tokens** | 4× smaller |

The `{{eqn}}` template genuinely gives per-step attribution — `| r = <formula> | c = [[Theorem
Name]]` is the target and the named citation side by side, which is the cleanest surface format in
the whole survey. It fails only on volume, and by two orders of magnitude against a 700 M floor.

---

## 6.2 IsarStep name recovery — 12.4%, and the rest needs Isabelle

IsarStep is the largest structurally-close corpus at 256 M tokens and fails only on naming, so
recovery is worth costing. Matching its anonymous fact strings against named lemmas harvested from
the local Isabelle distribution (1,851 `.thy` files) plus the AFP (10,351 files) — 188,656 named
statements in total, after stripping the outer `\<And>` binder and type annotations:

| group | names recovered | citations covered |
|---|---|---|
| 200 most-cited facts | 18.0% | 18.3% |
| 2,000 most-cited | 15.5% | 16.7% |
| whole store | **12.4%** (7,559 of 61,102) | **14.4%** |

Two reasons the rest fail, both visible in the misses. Some are `definition`/`abbreviation` rather
than `lemma` (`Let s f == f s` is `Let_def`). Others are locale-local facts carrying anonymised
parameters (`LLL.LLL_invariant <X0> <X1> …`), which have no global name to recover. A real recovery
needs to run Isabelle and query its theorem database across 800+ AFP sessions — days of compute
plus a working Isabelle install — and would still leave the locale-local fraction nameless.

---

## 7. A structural finding that applies to every library corpus

**A theorem's statement appears as the goal of its own proof.** Masking the fact block does not hide
a fact from the split arm; it reduces the fact's exposure from *many* to *one*. 97% of Metamath's
cited facts and 59% of Mizar's are themselves proved inside the corpus.

The manipulation is therefore a dosage difference of roughly **90–100×** on head facts, which is
still large — but it sets a ceiling on the token budget:

| corpus | tokens | epochs at 7.4 B | split-arm exposures per head fact via the goal-line leak |
|---|---|---|---|
| Metamath, essential-step | 98 M | 75 | 76 |
| Mizar, proof-level | 52 M | 148 | **148** |

Past roughly 100 exposures the split arm starts memorising the store through the leak alone, which
erodes the contrast. Metamath at a 7.4 B budget sits just under that line; Mizar at 7.4 B is over it
and should be run at ~4 B, or mixed with other text.

For held-out facts the leak is fatal rather than dilutive, which is why §2.3 removes each held-out
fact's own proof as well as its dependents.
