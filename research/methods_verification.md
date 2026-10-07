# Verification of P4 papers against the team's code

*All checks compare our writeups against the public repository `github.com/edu-llm/OLMo-core`. Files were read remotely through the GitHub API; nothing was cloned or saved locally.*
- **Part I** (2026-10-02): the spacing whitepaper, [P4_whitepaper_writeup.md](../paper/source/P4_whitepaper_writeup.md), against branch `anshulm/fictionalqa-review`.
- **Parts II–IV** (2026-10-04): the interleaving and mastery-gating sections of [P4_final_report.md](../paper/source/P4_final_report.md), against `p4/blocked-vs-interleaved` and a scan of all 152 branches.

## TL;DR

**Spacing whitepaper (Part I)**
- **The experiment and its headline numbers are real; this is not a hallucinated report.** The config matches the paper. The final losses (old 3.981 / 3.809 / 3.808, plus new-fact loss and buffer increase) match the team's committed `REVIEW_LAB.md` exactly.
- **Four descriptions are wrong or misleading:**
  1. A "review" is one update of 16 random statements from a single document style; about half the evaluated facts are never reviewed.
  2. The "held-out paraphrased" evaluation set is the question form of *trained* statements (answer verbatim 93% of the time); only 80 items are scored, and EOS is included.
  3. "Trajectory change" includes 3 buffer checkpoints.
  4. "377M" includes LoRA, and the Stage-2 optimizer/warmup reset is not mentioned.
- **Not verifiable:** the CIs (no code computes them) and the secondary numbers and Figure 1 (run outputs not committed). Exact match is recorded but not reported.

**Blocked vs. interleaved (Part II)**
- **The report describes a different study from the one that was run.** It describes a proposed 195M pretrained-checkpoint study (4 academic subjects, sessions, an interference phase, 12 paired runs). What was run is a **randomly initialised 162M model on 4 synthetic digit-manipulation skills, 512 updates, one order, one run**. The report doesn't mention that run.
- **Committed result:** blocked 27.4% vs. interleaved 61.6%. Blocked only knows the last skill (94.5%), which is recency and catastrophic forgetting. Interleaved hadn't converged (training loss 0.21 vs. 0.02).
- **The report's "toy pilot" numbers (52.8% → 79.2%) have no code on any branch.**

**Mastery gating (Part III)**
- **You were nearly right.** The subteam ran no full experiments, but the report does include **two small preliminary experiments**: a variable-chain depth task, and addition → repeated addition. **No code for them exists on any of the 152 branches**, so the numbers can't be verified. Model size, seed count and budgets aren't reported either.
- **Inconsistencies:**
  - the Preliminary Experiment II text says "(0.59, 0.67)" but the table shows 0.67 / 0.55;
  - the one-pager labels the mastery panel "Prerequisite-chain task" but plots Preliminary Experiment II numbers;
  - there is an informal sentence in Limitations that must be removed.

**Other (Part IV)**
- The Pythia 20-seed spacing pilot also has no code anywhere.

**Ask the team for:**
- the spacing `runs/` folder and CI script;
- the code and logs for the toy interleaving pilot and both mastery preliminary experiments;
- the Pythia pilot code.

# Part I: Spacing whitepaper

| Branch | Head commit | Contents | Relevant to the whitepaper? |
|---|---|---|---|
| `anshulm/fictionalqa-review` | `2d04968` (2026-07-24) | The spacing experiment: `src/olmo_core/review_lab/`, `configs/review_lab/olmo370m_fictionalqa_3seeds_buffer_t1000.yaml`, `REVIEW_LAB.md` (design and headline results) | **Yes. This is the experiment the paper reports.** |
| `p4/blocked-vs-interleaved` | `d6a71ab` (2026-07-24) | Blocked vs. interleaved training on synthetic digit-manipulation skills, with one completed run | No |
| `andrew/peer-distillation` | `3be92d8` (2026-07-28) | Four-model peer-distillation protocol (notebook, PRD, research brief). It also contains the blocked-vs-interleaved commit. No results. | No |

## Bottom line

**The whitepaper describes a real experiment and reports its real headline numbers. It is not a hallucinated report.**

- The design parameters in §2 match the committed final config exactly:
  - model and revision, LoRA settings, batch size, sequence length, learning rate and warmup;
  - stage lengths, buffer length, 12 review events, final review at Stage-2 step 165;
  - seeds 17, 23 and 42.
- The headline results match `REVIEW_LAB.md` exactly: final old-fact loss 3.981, 3.809 and 3.808; final new-fact loss 2.993, 3.018 and 3.015; buffer increase +0.151, +0.161 and +0.158.

Some descriptions in the paper are inaccurate or incomplete, though, and several numbers cannot be checked from GitHub because the run outputs (`runs/`) are not committed. Both are listed below.

## Claim-by-claim check (spacing experiment)

✅ = matches the code or committed results · ⚠️ = partly accurate or incomplete · ❌ = inaccurate · ❓ = not verifiable from GitHub

### Methods

| Whitepaper claim | Status | What the code / config says |
|---|---|---|
| OLMo DataDecide 300M, "approximately 377M total parameters" | ⚠️ | `allenai/DataDecide-dolma1_7-300M`, rev `4b1b42ff`. The base model has 371.5M parameters; 377M includes the 5.24M LoRA adapter. `REVIEW_LAB.md` itself says "~370M". The exact model ID and checkpoint should be stated. |
| LoRA rank 16, α = 32, all linear layers, 5.24M trainable | ✅ | `LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")`. Dropout 0.05 is not mentioned in the paper. 5.24M checks out arithmetically. |
| Batch 16, sequence length 64, LR 1×10⁻⁴, 10-step linear warmup, then constant | ⚠️ | Correct, but incomplete. AdamW (β = 0.9/0.95, weight decay 0.1), gradient clipping at 1.0 and fp16 autocast are not reported. **The optimizer and its 10-step warmup are re-created at the start of Stage 2**: AdamW moments reset and the LR re-warms. The first review (step 8, shared by both arms) therefore falls inside the warmup. |
| FictionalQA facts | ✅ | `jwkirchenbauer/fictionalqa`, config `fict_qa`, pinned revision `131cb74`; 20 old and 40 new events, split at the event level. The event counts and split are not stated in the paper. |
| Three seeds (17, 23, 42); means ± SD; comparisons paired by seed | ✅ | `seeds: [17, 23, 42]`; `statistics.stdev`. Every condition in a seed starts from the same Stage-1 checkpoint. The seeds do *not* change the data split or the evaluated items; the paper should say so. |
| Stage 1: 60 updates; Stage 2: 180; buffer: 180; all arms run the same number of steps | ✅ | Matches. Review updates *replace* new-fact updates. |
| 12 review events per review arm | ✅ | `review.events: 12`. |
| Uniform: reviews at roughly equal intervals; expanding: close together, then spreading out | ✅ (but under-specified) | Uniform steps: 8, 22, 37, 51, 65, 79, 94, 108, 122, 136, 151, 165. Expanding (ratio 1.35): 8, 10, 13, 17, 22, 29, 38, 51, 68, 91, 123, 165. The paper should list these. |
| Final review at Stage-2 step 165; about 194 updates before the final test | ✅ | Matches (0-indexed). |
| "Reviews of the Stage 1 facts" (each review re-exposes the old facts) | ❌ (misleading) | **Each review event is one optimizer update containing 16 old statements drawn at random, with replacement, from a single FictionalQA document style.** Styles rotate (blog, corporate, encyclopedia, news, social). Reconstructing the sampling shows that only about 31–40 of the 80 evaluated old facts are ever reviewed, and only 11–17 are reviewed twice or more. The schedule therefore operates on style-level batches, not on individual facts. |
| Evaluation: "held-out evaluation set—paraphrased questions about the facts that never appear in training … rather than memorized training text" | ❌ (misleading) | Training text is `"Fictional fact: <statement>"`. Evaluation is `"Question: <question>\nAnswer:"`, scored on the answer tokens **plus EOS**. Each question is the dataset's question form of a statement the model *was* trained on, and the answer string appears verbatim in that statement for 93% of facts. There is one question per fact (duplicates reduced to their canonical root). **Only 16 items per style (80 old and 80 new) are evaluated**, the same items for every seed and condition. Loss is averaged within each style, then across the five styles. |
| Pre-buffer old loss = loss just before the buffer | ✅ | Evaluation at Stage-2 step 179. |
| Final old loss; buffer forgetting = final − pre-buffer | ✅ | Matches `mean_final_old_loss` and `mean_buffer_old_loss_delta`. |
| Old-loss change = final − end-of-Stage-1 loss | ✅ | `mean_old_loss_delta` uses the shared Stage-1 checkpoint as the baseline. |
| **Trajectory change** = "the average old-loss change across Stage 2 … *during* training rather than only at the end" | ❌ | The code (`_retention_auc`) averages over **every evaluation tagged `stage: 2`, and that includes the buffer.** The nine checkpoints are Stage-2 steps 29, 59, 89, 119, 149 and 179, plus buffer delays 15, 60 and 180. A third of the average is post-review buffer time. §2.4 and the §3.5 interpretation both need correcting. |
| New loss; joint loss = equal-weight mean of final old and new | ✅ | `joint_loss = 0.5 * (mean_final_old + mean_new)`. |
| "Around the same FLOPs" | ✅ (plausible) | Identical step counts. Token counts per arm are logged (`old_content_tokens`, `new_content_tokens`) and could be reported exactly. |

### Results

| Whitepaper number | Status | Note |
|---|---|---|
| Final old loss 3.981 / 3.809 / 3.808; new loss 2.993 / 3.018 / 3.015; buffer increase +0.151 / +0.161 / +0.158 | ✅ | Identical to the results table in `REVIEW_LAB.md`. |
| "~4.3% relative reduction" | ✅ | Stated in `REVIEW_LAB.md`; arithmetic checks out (0.1727 / 3.981 = 4.34%). |
| Pre-buffer loss, old-loss change, trajectory change, joint loss, and their SDs (Table 1) | ❓ | Not committed. They are internally consistent: pre-buffer + buffer increase = final, and the implied end-of-Stage-1 loss is 4.204 in all arms. Confirm against `runs/.../results_aggregate.csv`. |
| Paired 95% CIs (Tables 2–3 and §3.4) | ❓ | **No code in the repository computes confidence intervals.** They were computed elsewhere. The values are consistent with paired *t*-intervals (df = 2). The script or notebook that produced them should be committed, and the method stated. |
| Buffer checkpoints at 15 / 60 / 180 updates (Table, §3.2) | ❓ | The evaluation schedule matches (delays 15, 60, 180), but the values are not committed. |
| Figure 1 | ❓ / ✅ | Its x-axis checkpoints (0, 30, …, 180, 195, 240, 360) match the code's evaluation schedule. The values cannot be checked without `metrics.jsonl`. |
| Exact match | Not reported | Each run summary records `final_old_exact_match` and `final_new_exact_match`. They should be reported. |

### Tone of the claims

The team's own `REVIEW_LAB.md` is more cautious than the whitepaper. It calls the work "a LoRA pilot" and says that "larger samples and full-parameter training would be needed" for broader claims, and its report generator labels the result "a directional pilot rather than a decisive spacing result". The whitepaper's conclusion ("review helps, and how you space it does not"; "uniform review is the sensible default … for a production system") goes further than the evidence the team itself recorded.

## The other two branches

- **`p4/blocked-vs-interleaved`.** One deterministic run on a randomly initialized 162M OLMo-architecture model, using four synthetic digit-manipulation skills, 512 updates per arm and 1,024 held-out items. Held-out exact match was 27.44% (blocked) vs. 61.62% (interleaved). The authors note that it is a single run with no uncertainty estimate.
  - This is **not** the interleaving number in the older Final Report and the one-pager, which say 52.8% → 79.2% from a "toy MLP pilot". If interleaving is ever mentioned in the paper, use the branch's numbers and its caveats.
- **`andrew/peer-distillation`.** A protocol and PRD for a 400M four-peer distillation experiment. There are no results. It is unrelated to the spacing paper.
- **Earlier spacing pilot.** The one-pager reports a Pythia "162M, 20 paired seeds" pilot (Shakespeare → WikiText). Its code is not on any of these three branches, so it cannot be verified here.

## What to ask the team for

1. The `runs/review_lab/olmo370m-fictionalqa-3seeds-buffer180/` directory: `REPORT.md`, `results_by_seed.csv`, `results_aggregate.csv`, and the per-condition `metrics.jsonl`. With these we can verify Table 1, the buffer table and Figure 1, and run the item-level reanalyses in research.md §10B.
2. The script or notebook that computed the paired 95% CIs.
3. Confirmation of whether any hyperparameters (Stage-1 length, LR, expansion ratio 1.35, buffer length, 12 events) were chosen after looking at results.

---

# Part II: Final Report, "Blocked vs. Interleaved Subject Training" (Test 02)

*Checked 2026-10-04 against branch `p4/blocked-vs-interleaved` (commit `d6a71ab`, 2026-07-24). The files read were `generate_data.py`, `train_compare.py`, `make_random_olmo.py`, `run_farmshare.sbatch`, `README.md` and `results/farmshare-olmo-162m-job1654870/{RESULTS.md,result.json}`. Everything was read remotely; nothing was stored. The same commit is also on `andrew/peer-distillation`.*

## Bottom line

**The report's design and the experiment actually run are two different studies.** The report describes a *proposed* 195M pretrained-checkpoint study: four academic subjects, four sessions, an interference phase, and 12 paired runs. The committed run is a much smaller screening test: a randomly initialised 162M model, four synthetic digit-manipulation skills, one pass through the data, and **one run in one order**. The report never mentions this run. Instead it cites a "toy shared-network pilot" (52.8% → 79.2%) for which no code exists on any of the repository's 152 branches.

The committed run itself is internally sound. The evaluation is correct (greedy exact match, left padding, held-out digit strings disjoint from training), and its numbers match the result file exactly.

## Proposed design (report) vs. what was run (code)

| Aspect | Final Report (Test 02) | Committed run | Match? |
|---|---|---|---|
| Content | Arithmetic, logic, geometry, statistics; natural-language 4-choice questions | Four transformations of 6-digit strings: rotate left (A), rotate right (B), reverse (C), swap adjacent pairs (D). Each skill is cued by its own delimiter pair: `< >`, `[ ]`, `{ }`, `( )` | ❌ |
| Model | ~195M decoder: 24 layers, width 768, 12 heads, tied 32k embeddings, 512 context, OLMo-core | Randomly initialised HF `OlmoForCausalLM`: **162.16M** parameters, 12 layers, width 768, SwiGLU 2048, *untied* embeddings, OLMo-1B tokenizer, max length 64 | ❌ |
| Starting point | Common *pretrained* checkpoint plus a shared generic burn-in, with optimizer state cloned into both arms | Same *random* initialisation and seed in both arms; a fresh AdamW per arm | ⚠️ (paired, but not pretrained) |
| Training length | 1,280 updates; 32,768 tokens per batch; ~10.5M tokens per subject | **512 updates**; batch of 32 sequences (≤64 tokens); 4,096 examples per skill; one epoch | ❌ |
| Blocked arm | 80 consecutive updates per subject, in each of 4 sessions | 128 consecutive updates per skill, in a single session (A×128 → B×128 → C×128 → D×128) | ❌ |
| Interleaved arm | Subject changes every update | Subject changes every update (A, B, C, D, A, …) | ✅ |
| Single-subject batches; same examples and within-subject order in both arms | Yes | Yes. Enforced by an assertion; both schedules are built from the same per-skill lists | ✅ |
| Optimizer | AdamW, constant LR, matched across arms | AdamW (β = 0.9/0.95), weight decay 0.1, constant LR 1×10⁻⁴ with **no warmup**, gradient clipping 1.0, bf16 | ✅ |
| Counterbalanced subject orders | 4 orders × 3 data seeds = 12 paired comparisons | The code builds the 4 rotated orders, but **only order 0 (A→B→C→D) was run, once, with one seed** | ❌ |
| Interference phase | 160 generic-data updates, then a post-interference audit | None. There is a single evaluation at the end of training | ❌ |
| Primary outcome | Post-interference macro accuracy, with immediate accuracy and forgetting as secondary outcomes | End-of-training exact-match accuracy, overall and per skill | ❌ |
| Calibration to 65–85% IID accuracy | Planned | Not done | ❌ |
| Pilot evidence cited | "Toy shared-network pilot": 52.8% blocked vs. 79.2% interleaved; independent subject models show no order effect | **No code or results for this pilot on any branch.** The committed result is 27.44% vs. 61.62% | ❌ (unverifiable) |

## The committed result, read critically

| | A (rotate L) | B (rotate R) | C (reverse) | D (swap) | Overall |
|---|---:|---:|---:|---:|---:|
| Base (random init) | 0% | 0% | 0% | 0% | 0/1,024 |
| Blocked (A→B→C→D) | 0.0% | 15.2% | 0.0% | **94.5%** | 27.44% |
| Interleaved | 81.6% | 95.3% | 27.3% | 42.2% | **61.62%** |

- **The blocked arm is almost pure recency.** It has mastered only the last block (D). This is textbook catastrophic forgetting, and in ML it is the expected result rather than a surprise. To separate an interleaving benefit from "the last block wins", the other three orders need to be run and averaged.
- **The interleaved arm has not converged.** Its final training loss is 0.208, against 0.017 for blocked, and it does worse than blocked on D (42% vs. 95%). With one epoch, the comparison measures retention for blocked and incomplete learning for interleaved. Training to convergence, or reporting learning curves, would make the two arms comparable.
- **n = 1.** The result is a single deterministic run with no uncertainty estimate. The team's own `RESULTS.md` says so.

## What to fix in the report text
1. Describe the experiment that was actually run (162M, random init, digit skills, 512 updates, one order, one run), or clearly label the 195M design as *proposed*.
2. Remove or substantiate the 52.8% → 79.2% toy-pilot numbers. No code exists for them. Also fix the one-pager, which repeats them.
3. Do not call the four digit transformations "subjects". They are closely related sequence-to-sequence skills.

---

# Part III: Final Report, "On Mastery-Gated Curriculum Training" (Test 03)

## Bottom line

- **The subteam ran two small preliminary experiments, but no code exists for them on GitHub.** Neither appears on the three P4 branches, and a scan of all 152 repository branches found no mastery-gating, variable-chain or repeated-addition code. Other teams' curriculum branches (`edullm/curriculum-370m`, `curriculum-new`, `edullm/skillit-370m`, `hypothesis/smoke-skilldag-cl`) are readability- or Skill-it-based, not the P4 mastery gate. **The preliminary numbers cannot be verified.**
- The two experiments in the report:
  - **Preliminary Experiment I** (variable-chain "depth" task, 4 arms). Results are given only as a screenshot of a table (Figure 2).
  - **Preliminary Experiment II** (addition → repeated addition, 5 arms). There is a results table, plus a curve figure (Figure 3) that was embedded inside the table.
- Neither reports model size, number of seeds, token budget, mastery thresholds, or how the probes and sealed exams were built.

## Internal inconsistencies found
| Location | Issue |
|---|---|
| Preliminary Experiment II text vs. table | The text says addition degrades without replay to "(0.59, 0.67)". The table shows **0.67** (mastery, no replay) and **0.55** (time-gated, no replay). |
| Preliminary Experiment II table | There is an extra empty fifth column, which is a formatting artefact. |
| Preliminary Experiment II text | "Both mastery models move on to multi-addition before the halfway point" is consistent with Figure 3: both rise at about 2×10⁷ of 5×10⁷ tokens. ✅ |
| One-pager ([P4_onepager.md](../original/onepager/P4_onepager.md) and its figure) | The mastery panel is labelled "Prerequisite-chain task" (Preliminary Experiment I), but its bars (0.67 / 0.85 / 0.97) are **Preliminary Experiment II** multi-add OOD numbers. On the actual prerequisite-chain task (Preliminary Experiment I), OOD accuracy for shuffle / gate without replay / gate + replay is **0.37 / 0.33 / 0.75** (in-distribution 0.47 / 0.32 / 0.81), so the gate *without* replay does worse than shuffle. `P4_onepager.html` also mixes the two experiments in one sentence. |
| Preliminary Experiment I text | "Depth five low only because it was not unlocked within the budget". The table shows 0.18 at depth 5 for the gated-with-replay arm, so it was partly trained. The text should say "unlocked late". |
| Limitations section | The informal sentence ("constructing your dataset will not be fun lmao good luck with that") must be removed before any external use. |

Citation accuracy for Tests 02 and 03 is covered in [research/notes/09_final_report_citation_verification.md](notes/09_final_report_citation_verification.md).

---

# Part IV: Other P4 claims not backed by code
- **Test 01 spacing pilot (Final Report and one-pager):** "162M Pythia, 20 paired seeds, Shakespeare → WikiText, AUC 0.325 vs 0.221". No code on any branch. The whitepaper's FictionalQA experiment supersedes it; if it is cited, the code is needed.
- **Test 02 toy pilot (52.8% → 79.2%):** no code (see Part II).
- **Test 03 preliminary experiments I and II:** no code (see Part III).
