# 05 — Deliverable, competing tracks, original intent, and surviving framing

**Author:** reassessment team member 05. **Date:** 2026-08-01.
**Labels used throughout:** MEASURED (a number someone actually observed), INFERRED (derived from
measured things or from documents), ASSUMED (a guess I am flagging as a guess).

**Status:** COMPLETE. Sections in write-order (§3 first, then §1, §2, §4, §5, §6) — read §6 for the
bottom line.

**Execution constraint honoured:** no code was run on the local Mac. Everything here is reading,
grepping, PDF extraction, `git log`/`git diff --stat` on the local OLMo-core checkout, and arithmetic
done by hand. Web search was unavailable (403 from the search backend), so the prior-art audit in §4.0
relies on the repo's own primary-source quotations in `02_lowrank_gates.md` §5B rather than a fresh
fetch — **flagged as a limitation: the FLAR-SVD / llama.cpp #956 / ARM-KleidiAI citations were not
independently re-verified by me.**

---

## §3. What Eric actually claimed and cared about (the original PDF)

Source: `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/Eric_LIV_Brainlift-1.pdf`.
**7 pages total: ~5.5 pages of body + 1.5 pages of references (21 refs).** Title: *"LIV Hybrid
Language Models," By: Eric Wu.* Structure: Abstract; §1 Linear Input Varying (LIV) Layer (+1 figure);
§2 Learning Science as an Inspiration; §3 Improvements Beyond the Stock LIV Layer (3.1 factorize
gates / 3.2 share KV / 3.3 test local spans); §4 Advantages, Costs, and Competing Approaches;
§5 Four Questions That Should Decide Whether the Idea Continues; References.

### 3.1 The five things in the PDF the downstream design doc got wrong or dropped

**(a) MEASURED (from the text) — P1 was NEVER primarily a latency claim. It was a parameter-count
claim, and Eric himself made the latency claim conditional.**
p.4 §3.1 opens: *"The first proposed change concerns **parameter count**."* The whole section is
param arithmetic (16.783M stock → 9.443M at r=128; "7.340 million fewer than the stock LIV mixer and
1.043 million fewer than the comparable GQA mixer"). The only latency sentence is the **last line of
his own caveat paragraph** (p.4, italics): *"Two smaller matrix multiplications also need an
efficient fused implementation before the parameter reduction becomes a latency reduction."*
→ The HANDOFF's headline "**P1's latency claim is dead**" overstates the damage. Eric wrote a
*conditional*; the L40S benchmark **discharged the condition and found it false even when satisfied**
(fused + CUDA-graphed is still 8.2% slower). That is Eric's own hypothesis being *answered*, not his
proposal being *refuted*. This is a materially more favourable framing and it is textually supported.

**(b) MEASURED — Eric pre-wrote most of the "adverse findings" the research swarm later "discovered."**
- p.4 caveat: *"A rapidly decaying spectrum would therefore support factorization, but it would not
  guarantee preserved language-model quality… **Rank 128 should consequently be treated as a
  hypothesis, tested against several ranks and a full-rank control** through singular-value analysis,
  post-training compression and fine-tuning, and from-scratch training across multiple seeds."*
  → The spectra probe finding ("gates are NOT low-rank, effective rank 771-790/1024") falsifies a
  premise **Eric never asserted**. He asserted the opposite: that it needed testing.
- p.4 §3.2: *"However, this does not cut the entire attention workload in half. All six calls still
  form queries, compute scores, apply a softmax, read KV data, and form weighted sums."* and *"That
  shared representation may be too stale or too restrictive for some tasks."*
  → P2's "saves capacity not bandwidth → latency ≈ 0 by construction" is **in the original PDF**.
- p.5 §3.3: *"Existing Transformer evidence favored short dynamic widths near three or four in the
  setting tested and reported an accompanying training-throughput cost (**Sieberling et al., 2026**).
  Wider spans should therefore be treated as an experimental hypothesis and retained only when their
  measured benefits justify the additional computation."*
  → **Eric cited Sieberling himself.** The prior-art verdict "published width sweep is flat past k=3"
  is not new information to the author; it is his own stated prior. The section title is literally
  *"Test several local spans **without assuming they are useful**."*
- p.5 §3.3: *"the router does not automatically reduce computation… Conditional-compute savings require
  hard or top-k routing"* and *"Router weights also should not be interpreted as direct explanations of
  the context span used by a token."* → the router hazards are his too.

**INFERRED, and this is the single most important finding of §3:** roughly 70-80% of the "adverse
evidence" the design doc treats as reasons to retreat is **quoted or paraphrased from the author's own
caveats**. The research phase largely *confirmed the author's own skepticism with measurements*. That
is a good outcome for the project and a bad reason to abandon it.

**(c) MEASURED — the design doc INVERTED Eric's hardware priority.**
p.6 §5 Q3: *"Does the model improve time to first token, prefill throughput, token-by-token latency,
or energy **on the hardware it is meant to serve**? Results should name **the CPU or edge device**,
batch size, context length, precision, and kernel implementation, **with a modern GPU included as a
control**."*
Eric: edge/CPU **primary**, GPU **control**. HANDOFF decision 5 + §0: *"Edge/energy: **Cut.** GPU-only."*
That is a 180° inversion of the author's stated target, justified in the HANDOFF by *methodology*
(clock variance on mobile). The justification is defensible but it should be **named as a scope
change against the author's intent**, not buried in a decision table. It also silently removes the
one platform on which P1's parameter reduction is most likely to pay (memory-constrained edge, where
weight bytes bound the model, unlike an L40S with 46 GB).

**(d) MEASURED — Eric's preferred P3 variant was the CHEAP DILATED one; the design doc runs the
EXPENSIVE DENSE one.**
p.5: *"One option is to run **dense** kernels with widths 3, 5, 9, and 15… requiring 3+5+9+15=32
learned taps per channel. **A cheaper comparison** uses four three-tap kernels with dilation values
1, 2, 4, and 7… giving effective spans of 3, 5, 9, and 15 with **only twelve taps in total**."*
The frozen P3 scope is "widths (`k5/k9/k15`) inside the real gated block first" — i.e. dense single
widths. Eric's dilated 4-branch design (12 taps, spans to 15) is the variant that the earlier
retracted claim ("edge runtimes can't run the variants") was *disproven* for — someone exported it
and all 4 convs delegated to XNNPACK with zero CPU fallback. **The cheap variant is the one shown to
be deployable and the one the author preferred, and it is currently deprioritized.**

**(e) MEASURED, and the biggest structural drop — §2 "Learning Science as an Inspiration" is ~1.2 of
5.5 body pages (≈22% of the document) and has essentially no counterpart anywhere in the 1,330-line
design doc or the ~14,600 lines of research.**
The PDF's intellectual spine is a *functional-decomposition* argument grounded in memory research:
Cowan 2001 / Vogel 2005 / Oberauer 2016 (working-memory *filtering* → LIV's input-dependent gates
"strengthen useful nearby features and suppress irrelevant or stale ones"); Tulving & Thomson 1973
(*encoding specificity* → attention's content-addressed retrieval); McClelland 1995 / Kumaran 2016
(*complementary learning systems* → two interacting mechanisms rather than one uniform store).
His actual thesis (p.3): *"A mostly-LIV hybrid therefore **separates two jobs**. LIV layers handle
frequent local filtering and composition with a bounded state. Attention layers are retained at
selected depths for distant, cue-dependent retrieval."*
→ **The project is a division-of-labour hypothesis, not an efficiency paper.** P1/P2/P3 are three
*implementations* of "make the local mechanism cheap so you can afford to keep the retrieval
mechanism sparse." The design doc reduced the project to its three tactics and lost the thesis.
This matters enormously for framing (§4 below): the strongest surviving story is a *division-of-labour
measurement*, and the repo already has the instrument for it (`L0` vs `A16-P`, MQAR, the 12 KiB/token
KV accounting).

**(f) MEASURED — Eric pre-registered a falsification discipline, and it licenses negative results.**
p.6 §5, final line: *"**If one of the four questions fails, the model should be simplified rather than
defended through the learning-science analogy.**"*
And p.6 Q3: *"**FLOP formulas cannot answer this question by themselves.**"*
→ The 695 GB/s vs 161 GB/s iso-byte result is **the direct, affirmative answer to Question 3 as the
author posed it**, and it vindicates the exact skepticism he wrote down. Reporting it is *executing
the proposal*, not abandoning it. Any framing that treats the negative latency result as a failure is
misreading the source document.

### 3.2 Which proposal was Eric most invested in? — INFERRED

Ranked by page-space, specificity of arithmetic, and depth of self-caveat:
| | space | numeric specificity | self-caveat depth | read |
|---|---|---|---|---|
| **P1 gates** | ~1 page (p.3 end–p.4) | highest — 4 exact param counts, r=128 named | longest caveat in the doc (a full italic paragraph) | **most invested, and most self-doubting** |
| **P3 spans** | ~0.9 page (p.5) | medium — tap counts, dilations | high; title says "without assuming they are useful" | genuinely curious, explicitly agnostic |
| **P2 KV** | ~0.5 page (p.4) | lowest — "six → three banks," no bytes | medium | **least invested** — shortest, cited to Brandon et al. as existing work |

P1 is the flagship. It is also the one he hedged hardest. P2 reads as the obligatory third leg. Note
that the HANDOFF's ordering (P1 first, P2 second, P3 reframed) matches this — that part is faithful.

### 3.3 What the PDF says the OUTPUT should be — MEASURED

p.6 §5 Q2: *"**The report** should separate weight bytes, KV capacity, KV writes, KV reads,
convolution state, peak allocations, and attention computation."* p.6 §5 closing: *"These tests
should begin at **smaller scales before a one-billion-parameter run**. They should use **multiple
seeds** and keep the tokenizer, training corpus, data order, optimizer, context length, and evaluation
harness **fixed**."*
→ The author's own target artifact is **a report answering four questions with a seven-line byte
ledger and multiple seeds, at sub-1B scale.** The 350M decision and the seed emphasis in the HANDOFF
are faithful to this. There is no page limit, venue, or date anywhere in the PDF.

---

## §1. The deliverable and the deadline

### 1.1 THE HEADLINE FINDING: there is no deadline anywhere. MEASURED.

I searched the entire repo (excluding `node_modules/`, `OLMo-core/` vendor code, and `.venv`s) for
every form of due-date or scope-constraint language: `deadline`, `due date/on/by`, `submission`,
`submit by`, `page limit`, `word limit`, `word count`, `advisor`, `professor`, `instructor`,
`grading`, `rubric`, `course requirement`, `program requirement`, `defense`, `thesis`,
`camera-ready`, `abstract deadline`, plus every major venue name (NeurIPS/ICLR/ICML/ACL/EMNLP),
plus date patterns `2026-(08|09|10|11|12)-DD`, `by August`, `end of the week/month/summer/quarter`,
`N weeks left/remaining`.

**Result: ZERO hits that constitute a commitment to anyone.** Specifically:

| I checked | what I found |
|---|---|
| `/Users/ericwu/Developer/Capstone_LLM/HANDOFF.md` (33.7 KB) | no date beyond "Last updated 2026-07-31" |
| `/Users/ericwu/Developer/Capstone_LLM/KDA/HANDOFF.md` | none |
| `/Users/ericwu/Developer/Capstone_LLM/quant_research/HANDOFF.md` | none |
| `/Users/ericwu/Developer/Capstone_LLM/KDA-LIV/HANDOFF.md` | "Week-1 gate", "8.6 eng-days" — **internal effort estimates, not calendar commitments** |
| `/Users/ericwu/Developer/Capstone_LLM/edullm-data/HANDOFF.md` | one real date: **S3 Object Lock expires 2026-08-24** — an infra expiry, unrelated to any deliverable |
| `docs/` (14 files) | design-doc "~15-16 days" / "~8 days" **compute** estimates only |
| `handoffs/` | one stale subdir, `pedagogical-sft-farmshare-auto-pilot`, no dates |
| repo-root `.docx`/`.pdf` | see §1.2 — these reveal the *context* but state no date |
| `.claude/` | `settings.local.json` is 4 lines (one permission). Two skills. **No CLAUDE.md at repo root at all** |
| `/Users/ericwu/.claude/projects/-Users-ericwu-Developer-Capstone-LLM/memory/` (16 files) | no deadline in any of them |

**INFERRED consequence, and it is the single most important governance fact in this reassessment:**
**nothing external is forcing the schedule.** Every "we must decide now" pressure in the LIV
HANDOFF is self-generated. This cuts both ways and both cut against the current plan:
- It removes any justification for launching an expensive multi-arm training program *because time
  is short*. There is no clock.
- It also removes any justification for *not stopping*. There is no submission that would be
  incomplete if the LIV track ended today.
The binding constraint is **the human's attention and money**, not a calendar. Every proposal must
therefore be justified on value-per-dollar and value-per-day-of-attention, with no deadline
discount available to either side of the argument.

**A second, related MEASURED absence: no venue and no audience is named anywhere.** No arXiv target,
no workshop, no course, no committee. The word "publishable" appears in the HANDOFF (decision §0
rationale: *"improving its gates stays publishable"*) but publishable *where* is never stated. A
project whose framing decisions are being made on "what is publishable" while the venue is undefined
is optimizing against an unspecified objective. **This should be resolved before any further design
work** — the answer changes the framing ranking in §4 materially (a workshop negative-results paper,
an arXiv note, and a course capstone report have very different bars).

### 1.2 What the deliverable actually is — INFERRED from artifacts, not stated anywhere

The repo root reveals the surrounding context, which nobody wrote down:
- `/Users/ericwu/Developer/Capstone_LLM/Qwen3_Project_Recommendation.docx` — a **group** memo
  ("our group's later-agreed upon intended project domain", "**we** should train a smaller
  dictionary", "**we** should first wrap the whole attention bundle"). So Eric operates inside a
  team called **eduLLM**, and at some point wrote an architecture recommendation for it.
- `/Users/ericwu/Developer/Capstone_LLM/P4 Validating Learning Science - *.docx` (5 versions,
  Jul 20-21) — a **completed, typeset, multi-author final report**: *"FINAL REPORT / P4 Validating
  Learning Science for Machine Pretraining."* It carries per-test **author bylines** (`Test 01
  Authors: William, Anshul`) and a table of contents. **This is direct evidence of the actual
  deliverable format the program uses: a multi-author typeset report with per-section authors,
  preregistered tests, and prior-work sections.**
- `OLMo-core` git history shows an `edu-llm` GitHub org with a roster, operators ("Enable Meric and
  Amy as eduLLM operators"), CI, and a job-approval workflow. This is a **shared lab**, not a solo
  project.

**INFERRED deliverable format, with moderate confidence:** a written report section — comparable in
scope and format to one of the P4 tests (each is ~3-6 pages: Prior work / Question / Design /
Results) — contributed to a shared eduLLM report, plus the code landed in the shared OLMo-core fork.
**ASSUMED (flagged): I have no direct evidence of length, venue, or grading.** The P4 document is
the best available proxy and it implies **a few pages per test, not a 20-page paper.**

**This matters enormously and it contradicts the scale of the current plan.** The LIV design doc is
**1,330 lines / 101 KB** — roughly **8× longer than the entire P4 final report's per-test sections
combined** — and it is a *plan*, not a result. The research dossier is another **~14,600 lines**.
**MEASURED ratio: the project has produced ~16,000 lines of planning and zero lines of the
deliverable.** If the target artifact really is a P4-style few-page test writeup, then the planning
is over-produced by more than an order of magnitude relative to the output format, and the marginal
value of more design work is close to zero.

---

## §2. What else the human is already sitting on — the competing tracks

Five parallel tracks. Below, each is scored on **(a) is there a finished, defensible RESULT?**,
**(b) how far from a writeup?**, **(c) does it compete with LIV for the same resource?**

### 2.1 Scorecard (all MEASURED from the HANDOFFs and git unless marked)

| track | HANDOFF status | scientific result in hand? | code state | distance to a writeup |
|---|---|---|---|---|
| **KDA-Householder** (`/Users/ericwu/Developer/Capstone_LLM/KDA/HANDOFF.md`) | *"COMPLETE. 278 tests pass on GPU, 98/98 probe runs, 3 audits closed, 0 failures. **The science is done. Only committing and write-up remain.**"* | **YES — four of them, all n≥5, all with CIs** | novel Triton fwd+bwd kernel, 6-level verification chain, 18/18 mutation-test negative control. **Uncommitted** (~2,900 lines on `dp2-kda-phase-0-prep`, pushed) | **DAYS.** Results, controls, and stats already exist |
| **quant_research** (`.../quant_research/`) | *"Loop status: **CAPSTONE REACHED**"*; 9 waves; `final/` has 4 deliverable docs | **YES — a full arc on real Qwen3-0.6B** | 169 experiment scripts + sbatch + preregistration doc | **DAYS-TO-A-WEEK.** `final/00-executive-verdict…md` already exists |
| **LIV brainlift** (root `HANDOFF.md`) | *"design complete… **no model trained, nothing running**"* | **Partially — 3 real negative/mechanistic probe results, 0 LM results** | 1,548 lines new OLMo-core code on a branch, **unmerged**; 55 tests | **WEEKS-TO-MONTHS** for the designed study; **days** for the negative-results paper (§5) |
| **KDA-LIV / CORE-6** (`.../KDA-LIV/HANDOFF.md`) | audit complete (24 agents, ~25k lines), design locked, *"**No model trained. Nothing running.**"* Needs **$965** + a not-yet-known amplification factor `A` | **NO LM result.** Two real methodological findings (the 19×-short slice gate; `TRITON_F32_DEFAULT=ieee` = 166× accuracy) | 6,174 lines on `core6-integration`, GPU gate passed | **MONTHS + $965**, and gated on a measurement not yet returned |
| **edullm-data** (`.../edullm-data/HANDOFF.md`) | live, deployed, 734 tests passing, 22 commits last session | **infra, not science** | shipped airlock in sbsandbox | N/A — it is a service, not a paper |

### 2.2 The adversarial read: KDA-Householder is a finished paper that nobody is writing

**This is the strongest single finding of §2.** MEASURED, from `KDA/HANDOFF.md`:

1. **A novel operator that exists in no library.** KDA's per-channel forget gate × DeltaProduct's R
   Householder factors. Both `fla` ops structurally refuse the combination (`chunk_kda` is R=1 only;
   `chunk_gated_delta_product` hard-asserts a per-head gate). So the kernel had to be written, and it
   was — forward *and* backward, in Triton.
2. **A verification chain most published kernel papers do not have:** 6 levels, bit-exact 0.0 vs
   `fla` at R=1, fp64 `gradcheck` PASS, 3.6e-15 manual-backward agreement, 7.1e-15 emulator
   agreement, 44/44 Triton on L40S, **and an 18/18 mutation-testing negative control** proving the
   clean result is coverage rather than a blind spot.
3. **A headline scientific result with a difficulty confound killed by design.** R=4 − R=1 on S5
   (non-solvable group): **+55.74pp SIG [+47.5, +63.9] at length 128, n=8, SIG at all 7 lengths out
   to 2048.** The dissociation is carried by a **paired interaction** against parity. And the
   **solvability control** is the part that makes it a *theory* result rather than a *capacity*
   result: S3 and S4 are solvable groups that are far harder than parity (13-22% vs 55% accuracy at
   length 2048) and R buys them **exactly nothing (ns at every length)**, while non-solvable S5 is
   SIG at every length. **The effect tracks solvability, not difficulty.** That is a clean
   theoretical prediction confirmed with the confound explicitly excluded.
4. **A second, previously-unpublished result thrown in:** depth × R substitution, SIG at all 7
   lengths (n=5), with the honest nuance that substitution is *complete only at short lengths*.
5. **A systems number:** 406× over the reference at B2/T8192/R4, with *less* memory (2.09 vs 5.65
   GiB). A 2020-step run goes 31.3 hours → 5 minutes.
6. **Self-refutations already performed** (which is what makes it credible): the n=3 "+8.92pp KDA >
   GDN" collapsed to **+2.01pp ns at n=8** and the HANDOFF says so; the n=2 "systematic backend gap"
   was noise and the HANDOFF says so.

**Assessment: this is a complete, publishable result with a novel artifact, a killed confound, real
sample sizes, confidence intervals, and self-corrections already in the record.** The HANDOFF's own
Next Steps §2 even *lists the four results in descending order of how much they carry the argument*.
The writeup outline is already written. The remaining work is **transcription plus `git add`**.

**Comparison to LIV, quantified:**

| | KDA-Householder | LIV brainlift |
|---|---|---|
| LM/probe results with n≥5 and CIs | **4 result families, 98/98 runs, 0 failures** | **0** |
| GPU-hours already spent on science | **<2 GPU-hours for the whole probe program** | ~minutes (5 microbenchmarks) |
| GPU-hours still needed for the designed study | **0** | **~3,000 A100-hours** (design doc §8) at 350M, or ~8 days on 8×A100 |
| novel artifact | a Triton kernel that exists nowhere | a mixer port of released Apache-2.0 code (parity 0.0 — i.e. deliberately *not* novel) |
| confounds killed | difficulty (S3/S4), capacity (parity), backend (n=8 ns) | none yet — no arms trained |
| distance to submittable | **days** | **weeks-to-months** |

### 2.3 So is the honest recommendation "the LIV track is the wrong horse"?

**Partially yes, and I will state it plainly: on a pure expected-value basis, writing up
KDA-Householder dominates every LIV option currently on the table.** It is finished, it is novel,
it has the sample sizes, and it costs approximately zero additional compute. Any hour spent
designing LIV arms is an hour not spent converting a finished result into a document. There is
**no deadline** (§1.1) forcing LIV forward and **no venue** requiring it.

But three honest qualifications, because "abandon LIV" is too crude:

1. **They are not fully in competition for the same resource.** KDA writeup needs *writing time*
   and zero compute. LIV needs *compute + money*. If the human can write in the morning and let a
   FarmShare job run, they are not exclusive. The exclusive resource is **decision attention**, and
   *that* is genuinely scarce — this repo currently has **five** open tracks and **three** of them
   are in "designed, nothing running" state (LIV, KDA-LIV, and parts of quant_research's next loop).
   **The real pathology is not LIV specifically; it is that the human has three completed-design,
   zero-result tracks and one completed-result, zero-writeup track.** That ratio is backwards.
2. **KDA-LIV/CORE-6 is a stronger candidate for the axe than LIV.** It needs **$965**, its gating
   amplification factor `A` was still unmeasured as of 2026-07-31, one of its gates already **failed
   by 19×** and required a mid-flight protocol redefinition, and it produces a result about *KDA
   insertion* — a mechanism whose own dedicated track is already **COMPLETE**. LIV is cheaper, is not
   gated on an unknown, and has already produced three real measurements.
3. **LIV has one asset KDA does not: a genuinely surprising, generalizable negative result that
   nobody has written down** (the 695 vs 161 GB/s iso-byte control). See §4/§5.

**Recommended ordering, stated adversarially:** (1) write up KDA-Householder — days, free, highest
credibility-per-hour in the repo. (2) Write up the LIV *systems* negative result as a short
companion piece — days, free, uses only measurements already in hand. (3) Only then decide whether
any LIV training arm is worth thousands of GPU-hours, with the two documents already banked so the
decision is not load-bearing. (4) **Cancel or indefinitely defer KDA-LIV/CORE-6's $965** until
something forces it.

---

## §4. The strongest surviving framings

### 4.0 First: an honest novelty audit of the negative result, before I build a framing on it

The candidate headline handed to me was *"bytes saved ≠ time saved: a measured refutation of roofline
reasoning for decode-time factorization."* Before ranking it I checked whether it is new. **It is
partly not, and the repo's own research already says so** —
`/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/02_lowrank_gates.md` §5B.1-5B.3:

**Prior art that already exists (MEASURED, from that file's quotations of primary sources):**
- **FLAR-SVD** (Thoma et al., CVPR 2025 Mobile AI Workshop) measured **FW-SVD and ASVD cutting params
  and FLOPs ~2× and becoming SLOWER than uncompressed** on Snapdragon 8 Gen 2 INT8 (8.8 / 9.0 ms vs
  **8.0 ms** baseline) and **67% slower on Jetson FP16** (24.4 / 24.5 vs **14.6 ms**). It states the
  mechanism in words very close to ours: *"having a disproportion between input and weights sizes can
  introduce overheads in inference even at low rank ratios,"* and *"the projection matrix (sized
  128 × 128) is not even achieving this inflection point at 10%, a very low rank."*
- **llama.cpp #956**: a LoRA-shaped skinny matmul at **120 ms vs ~5 ms expected — 24×** — diagnosed as
  *"tall and skinny matrices essentially get 0 vectorization."*
- **ARM Compute Library** hard-dispatches `M == 1` to separate GEMV kernels; **KleidiAI** pads K to a
  multiple of 32; **ggml** uses `QK_K = 256`. All three are structural reasons a rank-128 reduction
  dimension does not pay.

**So "low-rank factorization can be slower than dense" is PUBLISHED and is not a novel claim.**
Any framing that presents it as a discovery will be correctly rejected. **This is the single most
important correction I can make to the brief I was given.**

**What IS new here (INFERRED, by differencing against the above):**

| increment | why prior art does not cover it |
|---|---|
| **A datacenter GPU with CUDA graphs on.** L40S, graphed, 3 trials, ≤0.34% spread | FLAR-SVD is mobile NPU/Jetson. Graphing **removes launch overhead as the explanation** — the usual dismissal ("just dispatch overhead, use graphs") is pre-empted by construction |
| **A true ISO-BYTE control.** `dense` 40 MiB @ 56.2 µs vs `lowrank_fused r=512` 40 MiB @ **90.0 µs** | Nobody in the cited prior art holds bytes fixed and varies only structure. This isolates *shape* from *volume* — the actual scientific move |
| **Achieved-bandwidth attribution:** dense **695 GB/s** = 80% of L40S peak (genuinely bandwidth-bound) vs `lowrank_fused r=128` **161 GB/s** = 19% | Converts "it was slower" into "the roofline's *denominator assumption* — that you achieve peak — fails, and by 4.3×" |
| **The `grouped` g=2 vs g=4 datapoint** — 20 MiB @ 47.680 µs vs 10 MiB @ 47.600 µs: **halving bytes buys 0.17%** | This is the cleanest single refutation in the dataset and it is *not* about low-rank at all. It is a general statement about this decode regime |
| **Per-kernel cost measured under graphs: ~3.4 µs** | A number, not an adjective |

**Honest verdict on novelty: the phenomenon is known; the CONTROL is not.** The defensible
contribution is **methodological** — *"here is the control you must run, here is what it costs to
skip it, and here is the attribution it gives you that a before/after latency table cannot."*
That is a real contribution but it is a **short-paper / workshop-note** contribution, not a headline
result. Any framing must be scoped accordingly. This downgrades the candidate headline from
"refutation" to "the missing control," which is smaller but true.

---

### 4.1 FRAMING A — "The control you didn't run" *(rank 1)*

**Title:** *Bytes Are Not Time: An Iso-Byte Control for Decode-Time Weight Factorization*

**Abstract (one paragraph).** Roofline reasoning is the standard justification for compressing decode
weights: decode is memory-bound, so a 4× reduction in weight bytes should approach a 4× reduction in
time. We show this inference is unsound in a way that a conventional before/after latency table
cannot detect, and we give the control that detects it. On an NVIDIA L40S, with CUDA graphs enabled
and kernel counts measured rather than assumed, we replace the two full-width gate projections of an
LFM2-style gated short-convolution mixer (d=1024) with rank-r factorizations. The factorized mixer
reads **4× fewer bytes** and runs **8.2% slower** (60.83 vs 56.22 µs, 3 trials, spread ≤0.34%). The
diagnosis comes from two controls that the compression literature does not run. First, an
**iso-byte** control: at 40 MiB/token held constant, dense takes 56.2 µs and a rank-512 factorization
takes **90.0 µs**, isolating ~3.4 µs of cost per additional kernel *under graphs*. Second, an
**achieved-bandwidth** measurement: dense reaches **695 GB/s** (80% of peak — genuinely
bandwidth-bound, so the roofline's premise holds for it), while the factorized form reaches only
**161 GB/s** (19%). The roofline is not wrong about bytes; it is wrong about the *rate*, and skinny
GEMVs miss that rate by 4.3×. A third arm makes the point independent of low-rank entirely: a
block-diagonal mixer at 20 MiB and one at 10 MiB differ by **0.17%** — in this regime, halving weight
traffic buys nothing measurable. We recommend the iso-byte control as a standing prerequisite for any
decode-time compression claim, and we quantify what published work (FLAR-SVD, which reports
Snapdragon slowdowns without isolating the mechanism) leaves unattributed.

**Minimum result set required:**
| result | status |
|---|---|
| 6-arm graphed latency table with measured kernel counts | ✅ **IN HAND** — `probes/p1_verify_results.json` |
| Iso-byte control (r=512 at 40 MiB) | ✅ **IN HAND** — same file |
| Achieved bandwidth per arm | ✅ **IN HAND** — `probes/README.md` |
| grouped g=2 vs g=4 (byte-halving null) | ✅ **IN HAND** |
| Prior-art positioning vs FLAR-SVD / llama.cpp #956 / ARM+ggml kernel constraints | ✅ **IN HAND** — `02_lowrank_gates.md` §5B |
| **Generality: does it hold at a second shape or a second card?** | ❌ **MISSING.** This is the one real gap |
| Analytic model + where it fails | ✅ **IN HAND** — `probes/l40s_breakeven.py` |

**Remaining cost:** ONE FarmShare L40S job re-running the existing `p1_verify.py` at 2-3 additional
`d` values (e.g. 768 / 1536 / 2048) and, if a second card is ever reachable, one repeat. **Est. <1
GPU-hour, ~half a day of work, $0.** Everything else is already on disk.
**Credibility given evidence in hand: HIGH.** Every number is measured, replicated, and the spread is
25-200× smaller than the effects. **Remaining cost: LOWEST of any framing.**
**Weakness to state honestly:** it is a *methods note*, and one card / one shape is thin. The
generality run is not optional.

---

### 4.2 FRAMING B — "What Liquid never published" *(rank 2)*

**Title:** *Measuring What Ships: Ratio, Width, and Recall Ablations for a Deployed Hybrid
Convolution–Attention Architecture*

**Abstract.** LFM2/LFM2.5 ship a 10-convolution / 6-attention hybrid to real devices, and the
released record contains no conv:attention ratio ablation, no kernel-width ablation, and no
retrieval benchmark. The cross-layer-KV literature that motivates the same efficiency story (CLA,
Hymba, Character.AI, Gemma 3n) reports perplexity and averaged task scores and **no needle, passkey,
RULER, or MQAR result at all**. We supply the missing measurements on a 350M-parameter
reimplementation verified to **exact float64 parity** with the released operator, using a declarative
arm builder in which a parameter ledger reconciles component-by-component to the released
architecture (354,483,968 parameters, exact). We report recall, length extrapolation, and
retrieval-sliced perplexity — with seed counts chosen from a measured per-endpoint standard
deviation rather than assumed — for the released topology, a parameter-matched all-attention control
(matched to within 0.03% on parameters and diverging to **1.96× FLOPs at 32K**, which we report
rather than hide), a reduced-attention arm, and kernel widths k ∈ {3,5,9,15} evaluated *inside* the
real double-gated block, where the one published width sweep (which is flat past k=3) does not apply
because that sweep used an ungated residual convolution.

**Minimum result set required:**
| result | status |
|---|---|
| Parity-verified mixer + exact ledger + arm builder | ✅ **IN HAND** — 1,548 lines, 55 tests, parity 0.0 |
| MQAR harness + calibrated operating point + the 1/D floor | ✅ **IN HAND** on a 4-layer proxy — **but the HANDOFF explicitly says the numbers do not transfer to real `L0`** |
| Trained arms `L0`, `A16-P`, `A-fewer3`, `W-k5/k9/k15` at ≥5 paired seeds | ❌ **NONE.** Zero models trained |
| Measured `s_δ` per endpoint → required n | ❌ MISSING |
| 32K-matched training stage for any long-context claim | ❌ MISSING |

**Remaining cost:** the design doc's own estimate is **~3,000 A100-hours / ~8 days on an 8×A100 node**
for the 350M-headline version, plus the un-costed re-calibration of MQAR on real `L0`, plus a Phase-0
decode/conv-state path that **does not yet exist**. Real money on SB-AWS. **HIGHEST cost of any
framing by two orders of magnitude.**
**Credibility given evidence in hand: currently ZERO** — every load-bearing number is unmeasured.
Credibility *if completed* is high; the framing is genuinely outcome-independent, which is its real
virtue. But the product of (credibility now) × (1/cost) is the worst on this list.

---

### 4.3 FRAMING C — "The division of labour" *(rank 3, and the one closest to what Eric actually wrote)*

**Title:** *Local Filtering and Cued Retrieval Are Separable: A Controlled Test of the Mostly-LIV
Hybrid Hypothesis*

**Abstract.** The mostly-convolution hybrid rests on a division-of-labour claim: that most sequence
mixing is local filtering, which a bounded-state gated convolution can do, and that content-addressed
retrieval — the operation a finite convolution provably cannot perform — is needed at only a few
depths. This is an empirical claim about *how much* of each job a language model needs, and it has
never been measured directly on a deployed hybrid. We measure it three ways on a 350M
LFM2-architecture reimplementation verified to exact parity with the released operator. First, a
**dose-response**: sweep the number of global-attention layers a ∈ {0, 2, 4, 6} at fixed parameters
and read the retrieval endpoint, locating the depth at which retrieval saturates rather than assuming
6 is correct. Second, an **exact-byte accounting** of what the division buys, separating weight
bytes, KV capacity, KV writes, KV reads, and convolution state — where we show the effect is a
*long-context* effect that is largely invisible at trainable contexts (KV is **6.6%** of decode
traffic at 4K and **36.2%** at 32K), which is itself the finding most likely to change how these
architectures are evaluated. Third, a **measured refutation of the cheap version of the efficiency
story**: making the local mechanism smaller does not make it faster (§Framing A).

**Minimum result set required:**
| result | status |
|---|---|
| The byte accounting (6.6% / 36.2%, 12 KiB/token scale-invariance) | ✅ **IN HAND** — though sibling doc 02 found `crossover.py` is parameterized at 1.2B and prints **2.1% / 14.7%**, so the script must be fixed before publishing |
| The latency refutation | ✅ **IN HAND** |
| The `a`-sweep dose-response, ≥5 seeds | ❌ **MISSING** — but this is the *cheapest* possible training program: one axis, one endpoint, no matched controls needed because `a` is the only thing varying |
| Retrieval endpoint calibrated on the real model | ❌ MISSING |

**Remaining cost:** materially less than Framing B — a single-axis sweep, and sibling doc 05 in the
KDA-LIV track costed a comparable dose-response as **free on FarmShare** (1B tokens, 1 GPU/job).
**Est. days of GPU, not an 8-GPU node for 8 days.**
**Credibility: MEDIUM-HIGH.** Two of three legs are in hand; the third is the cheapest experiment in
the entire program. **This is the framing that honours the original PDF** (§3.1e) and it is the only
one whose thesis statement is a sentence Eric actually wrote.
**Weakness:** the dose-response is close to what the sibling KDA-LIV track already designed, so the
two tracks should be merged rather than run twice.

---

### 4.4 Ranking: (credibility now) × (1/remaining cost)

| rank | framing | credibility with evidence in hand | remaining cost | product |
|---|---|---|---|---|
| **1** | **A — the iso-byte control** | **HIGH** (100% of load-bearing numbers measured, replicated, spread ≤0.34%) | **<1 GPU-hour + ~0.5 day** | **best by a wide margin** |
| **2** | **B — what Liquid never published** | **ZERO now** / high if completed | **~3,000 A100-hours + real money + un-built decode path** | worst |
| **3** | **C — division of labour** | MEDIUM-HIGH (2 of 3 legs in hand) | days of single-GPU time, plausibly free on FarmShare | **strong second** |

**Note the ranking is 1, 3, 2 in outcome terms:** A and C are both worth doing and together they are
a coherent document; B is the expensive one and it should be entered only after A and C are banked.
**A and C compose:** A is the systems section of C. A single report can be *"the mostly-LIV hybrid's
division of labour, measured — including why the obvious efficiency shortcut does not work."* That
composite is my actual recommendation.

---

## §5. The minimum submittable unit — the floor

**The question:** if the human stopped collecting data TODAY (2026-08-01) and wrote up only what
exists, what document could they honestly write, and how good would it be?

### 5.1 Exact inventory of what exists, as of today

**MEASURED results (a number someone observed, with an artifact on disk):**

| # | result | artifact | strength |
|---|---|---|---|
| R1 | 6-arm CUDA-graphed decode latency, kernel counts profiler-measured, 3 trials, spread ≤0.34% | `probes/p1_verify_results.json` (FarmShare 1670884) | **strong** — effects 25-200× the spread |
| R2 | Iso-byte control: 40 MiB dense 56.2 µs vs 40 MiB `lowrank_fused r=512` 90.0 µs | same file | **strong, and it is the novel bit** |
| R3 | Achieved bandwidth 695 GB/s (dense, 80% of peak) vs 161 GB/s (r=128, 19%) | `probes/README.md` | **strong** |
| R4 | Byte-halving null: `grouped` g=2 (20 MiB) 47.680 µs vs g=4 (10 MiB) 47.600 µs = **0.17%** | `p1_verify_results.json` | **strong**, underused |
| R5 | Gate spectra: effective rank 771-790 / 1024, activation-aware 493.3 vs value-stream control 507.8 — the collapse belongs to the input distribution, not the gates. 32,768 tokens, `rank(Σ_x)=1024` reported | `probes/spectra_v2_results.json` | **strong**, with a documented near-miss (568 tokens gave a spurious 3.0× collapse) |
| R6 | rank-128 retains **92.6%** of activation-weighted energy vs **45.8%** plain Frobenius | same | strong |
| R7 | Structure energy: `lowrank r=128` **0.929** vs `grouped g=4` **0.130**, the latter identical to a random mask of equal density; random channel permutations [0.125, 0.133] | `probes/structure_energy_results.json` | **strong as a prior, weak as a verdict** — the metric is Eckart-Young-favourable to low-rank by construction, and the HANDOFF says so |
| R8 | Real LFM2.5-350M ONNX q4 per-op decode profile: `MatMulNBits` **91.2%**, `Conv` **1.0%** | HANDOFF §"finding that reorganized the design" | strong, and it is the number that ranks the three proposals |
| R9 | MQAR harness calibration + the **1/D degenerate floor** and its three legible loss plateaus (`ln(vocab/2)`, `ln(D)`, 0), all three observed to 2 dp; bimodality holds at low load and **breaks** at high load | `probes/mqar/mqar_calibration.json` (jobs 1670928, 1670987) | **methodologically strong, scientifically preliminary** — on a 4-layer proxy, and the HANDOFF says the numbers do not transfer |
| R10 | Exact parity: `ShortConv` diffed to **0.0 in float64** vs released `Lfm2ShortConv` at k=3/5/9; `L0` ledger **354,483,968 exact**; two geometry omissions caught (untied embeddings +67,108,864; per-head QK-norm +768) | branch `agent/claude-01/liv-short-conv-mixer`, 55 tests | **strong** — this is real, checkable engineering |
| R11 | Analytic break-even model + the correction that the widely-quoted 4.72 µs was computed at d=2048, not the frozen d=1024 (true values 1.35 µs A100 / 2.43 µs L40S) | `probes/l40s_breakeven.py` | moderate |

**MEASURED negative/process results (worth reporting, often more useful than R1-R11):**
- Four architecture-integration traps, each of which produced a **working, backpropagating,
  test-passing model that was the wrong model** — most sharply: setting `block.attention` instead of
  `block.sequence_mixer` silently creates a new attribute, and the resulting 16-layer model trains
  cleanly with **zero** ShortConv layers.
- The receptive-field test that only works against a **nonzero** background (with zeros, the
  multiplicative gates make every lag read "no reach" — a false pass).
- The 568-token covariance near-miss (R5).
- Two MQAR process failures: a difficulty sweep run *before* a positive control (0.000 everywhere,
  uninterpretable) and an under-trained rerun from a stale sbatch.

**What does NOT exist: any language-model result. Zero models trained. Zero seeds. No CE, no recall,
no extrapolation, no arm comparison.** This is the floor's hard boundary.

### 5.2 The document that could honestly be written TODAY

**Title:** *Five Cheap Measurements That Reshaped an Architecture Study — and the Controls That
Made Them Trustworthy*
(Or, in the composite form recommended in §4.4, as the systems + methods half of the
division-of-labour report.)

**Shape (INFERRED, from the P4 report's per-test format — see §1.2):** ~6-10 pages.
1. **The claim under test** — a mostly-convolution hybrid saves memory and time; three proposed
   improvements sharpen the saving.
2. **The instrument** — a parity-verified reimplementation (float64 diff 0.0) and an exact parameter
   ledger, with the four traps that produce a silently-wrong model documented as a checklist.
3. **Result 1: bytes are not time** (R1-R4, §4.1). The headline.
4. **Result 2: the premise was about the input distribution, not the gates** (R5-R6), with the
   568-token near-miss reported as a methodological warning.
5. **Result 3: which structure to prefer once latency stops deciding** (R7), stated as a prior with
   its Eckart-Young caveat intact.
6. **Result 4: where the decode time actually is** (R8) — and the observation that this ranks the
   three proposals before any of them is run.
7. **Result 5: an evaluation instrument with a degenerate-strategy floor** (R9) — the 1/D floor,
   which is a genuine trap in the recall-benchmark literature.
8. **What this does not show** — no LM result, one card, one shape.

### 5.3 How good would it be? — an adversarial answer

**Good, but only if it is honest about its genre.** Concretely:

**What it is:** a competent, unusually well-controlled *pre-registration-and-de-risking* report. Every
number is measured, replicated where it matters, and the document contains at least six places where
the author's own earlier claim is retracted with the reason. That last property is rarer than it
should be and is the strongest thing about the whole corpus.

**What it is NOT, and must not be dressed up as:**
- It is **not** an architecture result. It compares no trained models.
- It is **not** a novelty claim about low-rank being slow (§4.0 — FLAR-SVD published that).
- Its single most-cited framing (*"Liquid published no ratio ablation, no width ablation, no recall
  benchmark"*) is a statement about a **gap**, and a paper that identifies a gap without filling it
  is a proposal, not a result.

**Grade, stated plainly (INFERRED, calibrated against the P4 report as the format proxy):** as a
standalone document, this is a **solid B+ / weak-A capstone artifact and a plausible workshop
short-paper** if and only if Framing A is the headline and the generality run (one FarmShare job) is
added. Without the generality run it is one card, one shape, and a reviewer will say so. As a
*conference* paper it does not clear the bar, and nothing short of Framing B's ~3,000 A100-hours
would change that.

### 5.4 The floor, stated as a decision rule

**This is the number every proposed experiment must beat:**

> **Floor = a 6-10 page methods-and-measurement report, available in ~2-5 days of writing, at ~$0
> and <1 additional GPU-hour, with 11 measured results and zero trained models.**

Therefore:
1. **Any proposed experiment costing more than ~1 GPU-day must argue it moves the document from
   "solid B+ methods note" to something categorically different.** Adding a 12th measured probe does
   not; it makes a long methods note longer.
2. **The only expenditure that changes the document's CATEGORY is a trained-arm comparison** — i.e.
   Framing C's single-axis dose-response is the cheapest thing that crosses the category boundary,
   and Framing B is the expensive way to cross the same boundary.
3. **Adversarial corollary the human will not like:** measured against this floor, the
   `KDA/HANDOFF.md` track is *already above it* — it has 4 result families at n≥5-8 with CIs, a
   killed confound, a novel kernel, and 98/98 runs. **The LIV floor document is worse than a
   document that could already be written from a different directory in the same repo, for the same
   zero dollars.** If only one thing gets written this month, it should not be the LIV one.

---

## §6. Bottom line

1. **No deadline exists anywhere in the repo, and no venue is named.** MEASURED (exhaustive grep,
   §1.1). Every schedule pressure in the LIV HANDOFF is self-generated. The binding constraint is
   the human's attention, not a calendar — which removes the time-pressure argument from *both* the
   "launch now" and "we can afford to plan more" sides.
2. **The deliverable format is INFERRED, not stated:** a P4-style multi-author typeset report with
   per-section bylines, few pages per test. Against that format, the project has produced ~16,000
   lines of planning and zero lines of deliverable — over-produced by more than an order of
   magnitude.
3. **KDA-Householder is a finished, novel, statistically-honest result with a written outline and
   zero remaining compute.** It should be written up first. quant_research is second. The LIV track
   is genuinely third — though its *negative systems result* can be written in parallel for free.
4. **KDA-LIV/CORE-6, not LIV, is the right thing to cut.** $965, a gate that already failed by 19×,
   an unmeasured amplification factor it is entirely contingent on, and a question about a mechanism
   whose own track is already COMPLETE.
5. **The original PDF is more sophisticated than the design doc gives it credit for.** Eric
   pre-registered ~70-80% of the "adverse findings," made P1's latency claim explicitly conditional,
   cited Sieberling himself, and wrote *"if one of the four questions fails, the model should be
   simplified rather than defended through the learning-science analogy."* The measurements
   **executed** his proposal; they did not refute it. Three things were dropped against his intent:
   the edge/CPU primary target, the cheap dilated P3 variant, and — biggest — the entire
   division-of-labour thesis that was 22% of his document.
6. **The strongest headline is Framing A (the iso-byte control), but scoped as "the missing control,"
   not as a refutation** — FLAR-SVD already published low-rank slowdowns. What is new is the control
   (iso-byte + achieved-bandwidth + graphs-on), not the phenomenon. It needs one FarmShare job at a
   second shape to not be a single-point claim.
7. **Recommended composite:** one report = Framing C's thesis (division of labour, honouring the PDF)
   with Framing A as its systems section, and Framing C's single-axis dose-response as the only new
   training spend. Framing B's 3,000 A100-hours stays unfunded until A and C are banked.
