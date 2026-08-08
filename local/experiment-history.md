# P4 Experiment History & Onboarding

**Project: Validating Learning Science for Machine Pretraining**

*A running log of what we've tried and what we found, so new members can get up to speed quickly. Last updated late July 2026.*

---

## What this project is

Human learning science has three celebrated results — **spacing** (review spread over time beats cramming), **interleaving** (mixing subjects beats blocking them), and **mastery gating** (advance only once the current skill is learned). They're well established for people; nobody has cleanly shown they transfer to *machine* pretraining. This project runs controlled, pre-registered experiments to find out which principles transfer, which weaken, and which should be dropped.

A recurring theme in our findings: **the human effect often shows up only weakly, or turns out to be explained by something simpler** (reviewing at all, rather than *how* you space it; replay, rather than the mastery gate itself).

## Status at a glance

| # | Experiment | Status | One-line result |
|---|---|---|---|
| 01 | Spaced vs. uniform review | ✅ **Done** | Review helps (~4.3%); *how* you space it does not. |
| 02 | Blocked vs. interleaved subjects | 🟡 Toy pilot only | Big effect in a toy MLP; LM-scale run designed, not yet run. |
| 03 | Mastery-gated curriculum | 🟡 Toy done → ❌ theory says null | Toy gate+replay wins; follow-up analysis concludes it won't help LLM pretraining. |
| 04 | Synthetic student (classroom proxy) | 📋 Assessment only | Recommend bounded validation; likely won't be a valid human proxy. |
| 05 | Four-model peer distillation | 📋 Designed, not run | Protocol locked; no results yet. |

Legend: ✅ complete · 🟡 partial/pilot · ❌ negative/abandoned · 📋 design/assessment only

---

## 01 · Spaced vs. Uniform Review — ✅ Done

**Question.** Holding the number of exposures *and* the total token budget fixed, does an **expanding-interval** review schedule retain facts better than a **uniform** one? (The competing null: only exposure count and recency matter, and spacing adds nothing.)

**Preliminary pilot.** 162M Pythia, 20 paired seeds. Model learned Shakespeare, then trained on WikiText (interference). Compared no review / 7 uniform reviews / 7 expanding reviews, matched budget and first+last review times.
- Expanding held higher retention *across the training trajectory* — AUC 0.325 vs 0.221 (diff 0.104, 95% CI [0.011, 0.194]).
- But the two were **tied at the final delayed checkpoint**. Both beat no-review.
- Caveat: this measured next-token loss on natural text, not controlled facts — motivating the main experiment.

**Main experiment (Anshul & Will).** OLMo DataDecide 300M (~377M params), LoRA (5.24M trainable), **FictionalQA** (invented facts, so no pretraining contamination), seeds 17/23/42. Phases: Stage 1 teach old facts (60 updates) → Stage 2 new facts + 12 interspersed reviews (180) → **buffer** of 180 new-fact updates with no review (the delayed-retention test). Metric: answer-token cross-entropy on held-out paraphrases (lower = better).

| Comparison (final old-fact loss) | Difference | 95% paired CI |
|---|---|---|
| Uniform − no review | **−0.173** | [−0.214, −0.131] |
| Expanding − no review | **−0.173** | [−0.210, −0.137] |
| Expanding − uniform | −0.001 | [−0.005, +0.004] |

**Results.**
- **Review works:** both schedules cut delayed old-fact loss ~4.3% vs no review, consistent across all three seeds.
- **Spacing schedule does not matter:** expanding vs uniform is a dead heat (CI hugs zero).
- **Mechanism:** review lowers loss *before* the buffer; it does **not** slow the forgetting rate (all arms forgot ~0.15–0.16 during the buffer). It's a better starting point, not a more durable memory.
- Small plasticity cost on new-fact learning (CI crosses zero); on the joint metric, review still wins.

**Takeaway.** Under matched budget, **reviewing helps but the schedule doesn't** — expanding's apparent edge was a front-loading artifact that washed out after the buffer. Use uniform review as the simpler default. Repo: `github.com/anshulmago1/olmo-fictionalqa-review`.

---

## 02 · Blocked vs. Interleaved Subject Training — 🟡 Toy pilot only

**Question.** Does interleaving subjects (arithmetic, logic, geometry, statistics) instead of training them in blocks produce higher **post-interference accuracy** and less forgetting? Only subject *order* changes; everything else (data, tokens, optimizer, compute) is matched.

**What's done (Andrew & Tejas).** They built the continued-pretraining pipeline and ran it end to end on a toy shared-network pilot: terminal accuracy **52.8% blocked → 79.2% interleaved**, while *independent* per-subject models showed exactly zero order effect (the zero-effect control confirms schedule-dependent interference *within a shared network*, not a data or evaluation artifact).

**What's designed but not yet run.** The LM-scale study: ~195M-param OLMo-core decoder (24 layers, width 768), fork one checkpoint into blocked (80 consecutive updates/subject) vs interleaved (switch every update); 1,280 curriculum updates + 160 shared interference updates; primary metric = post-interference macro accuracy across the 4 subjects. 12 paired runs (4 counterbalanced orders × 3 seeds), ≈1.37×10¹⁸ FLOPs (~3–15 H100-hours, ~$10–150).

**Takeaway.** Promising toy signal, but **no language-model result yet**. A bigger pretrained model may show a smaller, larger, or null effect — treat the toy number as motivation only. Framed as a *continued-pretraining* screen, not a claim about from-scratch pretraining.

---

## 03 · Mastery-Gated Curriculum — 🟡 Toy done → ❌ theory concludes null

**Question.** Does advancing only when the model proves **mastery** on held-out probes (optionally with **replay** of mastered skills) beat a fixed-clock curriculum or a shuffle baseline, at matched token budget?

**Toy Experiment I — from-scratch decoder, 4 arms.**
- *Multi-digit addition:* shuffle **matched** mastery and was more token-efficient — an honest negative (addition skills are independent, no prerequisite ladder to exploit).
- *LEGO-style variable-chain task* (resolve a depth-k chain of scrambled definitions; genuine prerequisite structure):

  | Arm | In-dist | OOD | Per-depth (1→5) |
  |---|---|---|---|
  | **Mastery-gated + replay** | **0.81** | **0.75** | 0.97 / 0.99 / 0.97 / 0.95 / 0.18 |
  | Shuffle | 0.47 | 0.37 | 1.00 / 0.80 / 0.36 / 0.12 / 0.09 |
  | Mastery, no replay | 0.32 | 0.33 | 0.15 / 0.10 / 0.13 / 0.28 / 0.96 |
  | Fixed clock | 0.14 | 0.12 | 0.15 / 0.05 / 0.06 / 0.14 / 0.28 |

  The naive gate *without* replay shows textbook **catastrophic forgetting** — it reaches depth 5 (0.96) but wipes out depths 1–4. Gate **+ replay** climbs the whole ladder.

**Toy Experiment II — teach addition, then repeated addition.** Mastery+replay best (multi-add 0.97, addition retention 0.99, efficiency AUC 0.318). Key trend: **addition collapses without replay** (drops to ~0.55–0.67) even with mastery learning; every arm *with* replay retains addition (~0.99–1.00). Mastery arms also move on early, wasting fewer tokens.

**Follow-up analysis (Adam & Katie) — the deflating result.** A theoretical writeup concluded mastery gating will give a **negligible speedup for LLM pretraining**, and that any observed gains are attributable to *non-mastery* design choices (scheduling, adaptive difficulty). Training on strictly less data is ruled out by the **Data Processing Inequality**. Large-scale runs are also impractical: mastery is ill-defined in pretraining, the skill graph is expensive to build, and held-out eval sets leak signal into the model. **Verdict: side with the null hypothesis.** (Overleaf: `overleaf.com/read/fpdgpzbqkswg`.)

**Takeaway.** The real lever in the toy results was **replay (forgetting prevention), not the mastery gate itself** — and the follow-up argues the gate adds little at pretraining scale. Mastery gating still genuinely helps in narrow, structured settings (e.g. sparse-reward RL), but that's not novel. Consider this thread **de-prioritized**.

---

## 04 · Synthetic Student (classroom proxy) — 📋 Assessment only

**Idea.** Build small from-scratch models that behave like ≈grade-4 learners, instantiate a "classroom" with a realistic ability spread, and use it to evaluate whether a curriculum is good.

**Recommendation.** Proceed only to a **bounded Phase I**: test whether the synthetic classroom can *detect controlled curriculum defects* (shuffled order, removed prerequisites, planted wrong worked-example) above cheap baselines. **Do not** claim it predicts real human learning.

**Why it may fail (each threat is individually fatal, and research-backed):**
- **R2 — no human forgetting curve.** Humans forget as a power law of *time*; a frozen LLM forgets nothing over time and only via gradient *interference*. Worse, LLM catastrophic forgetting *intensifies with scale* — the opposite of a stable, calibratable curve.
- **R3 — LLMs don't learn like humans.** Alien sample efficiency, grokking / step-like acquisition, the Reversal Curse, and memorization-vs-understanding all break curve/error fidelity.
- **R4 — "grade level" isn't coherent** for a jagged-frontier model.
- **R5 — curriculum→training-data conversion** introduces a teaching-modality gap (SFT ≠ practice-with-feedback).
- **R6 — sim-to-real gap** may simply not close.

**Takeaway.** The defensible claim is a *defect detector*, not a human predictor. The honest likely finding — "a synthetic-LLM classroom is not a valid model of a human one" — is itself worth publishing. No experiment run yet.

---

## 05 · Four-Model Peer Distillation — 📋 Designed, not run

**Question.** Can four complementary same-size 400M **peers** train a better single deployable 400M model than a 400M student distilled from one stronger larger teacher? Primary comparison: selected `peer_frr_onpolicy` minus selected `large_teacher_diverse`, under matched starts/data/updates/selection/eval, with a **one-model** (not ensemble) deployment endpoint.

**Status.** Full pre-registered protocol exists (`local/four-model-peer-distillation-protocol.md`) with locked arms, safety gates, fail-safe run modes, and a compressed 2–4× B200 "large-effect screen." **No results yet.** See also `local/verifiable-peer-distillation-400m-prd.md` for a redesigned executable version that fixes the main risk (choosing a *verifiable* task calibrated to a 30–60% base-accuracy sweet spot so the peer-rescue signal actually exists).

**Takeaway.** Highest-risk part is the signal source: peer distillation only transfers capability through *verified rescues*, which need (a) a task the 400M peers can sometimes get right and (b) decorrelated peers. Watch for the "all models too weak → no signal" failure mode.

---

## Cross-cutting lessons for new members

1. **Matched budgets are everything.** Every result here hinges on holding exposures, tokens, updates, and compute fixed so only the variable of interest moves. Design the control before the treatment.
2. **The human effect usually shrinks to something simpler.** Spacing → "just review." Mastery gating → "just replay." Expect to isolate and name the real mechanism, not to confirm the headline.
3. **Use contamination-free probes.** FictionalQA, invented rule systems, and synthetic skill tasks exist because you can't measure retention on text whose first-exposure time you don't know.
4. **A negative/null result is a real deliverable.** Two of five threads landed on "no" or "not the way you think," and those are written up as findings, not buried.
5. **Toy pilots motivate; they don't predict.** The MLP and small-decoder pilots detect *whether a pipeline works*, not what a real LM will do. Don't over-read them.
6. **Watch the learning-rate-decay confound.** Data fed late in training (under a decaying LR) barely moves the model — any "late-hard curriculum loses" result can be a decay artifact. Hold the LR schedule identical across arms.
