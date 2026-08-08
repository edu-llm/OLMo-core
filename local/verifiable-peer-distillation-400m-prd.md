# Verifiable-Task Peer Distillation at 400M: An Executable PRD

**Can complementary same-size peers, trained on programmatically verifiable tasks, produce a better single deployable 400M model than distillation from a single larger teacher — and does it beat the cheap ensemble-teacher baseline?**

P4 · Verifiable Peer Distillation (400M-locked) · Execution PRD v1

> **Status:** Pre-registered experimental design and execution protocol. It fixes the tasks, the arms, the matched budgets, the checkers, the analysis plan, and the artifacts *before* any outcome is examined. It is written to be executed by an operator or an agent with no further clarification. Model weights, private corpora, and run outputs are never committed; all work stays under `local/` and nothing is pushed to GitHub.

---

## 0. Why this PRD exists (what changed from the prior peer-distillation protocol)

The earlier four-model peer-distillation protocol (`local/four-model-peer-distillation-protocol.md`) was flat: no measurable improvement from peer distillation. This PRD diagnoses that as a **task-regime failure, not an algorithm failure**, and redesigns around three fixes:

1. **The signal source must exist.** Peer distillation transfers capability only through *verified rescues* — cases where one peer produces a checkably-correct answer another peer missed. If base accuracy is near the floor, there are almost no correct rollouts to rescue with, so nothing transfers. Multi-digit arithmetic at 400M is a floor task (≈0% accuracy), which is why it produced no signal. **We keep 400M and instead move to verifiable tasks calibrated to a 30–60% base-accuracy sweet spot**, where rescues are frequent enough to train on and there is still headroom to improve.

2. **Best-of-N throws away the ensemble mechanism.** Selecting "the best peer at step *t*" as the teacher copies that peer's blind spots instead of averaging them out. We add an **ensemble-teacher arm** (distill from the aggregated distribution / verified union of all four peers), which is the cheapest way to capture variance reduction and error cancellation, and which doubles as the honest "is the gain just ensembling?" control.

3. **Peers must be decorrelated to be complementary.** Four peers trained on identical data are correlated and cannot rescue each other. We force decorrelation via **slices** — each peer specializes on a different verifiable task family — so the union of their verified-correct answers covers more than any single 400M teacher produced.

Everything else (single deployable endpoint, matched budgets, paired-seed analysis, fail-safe run modes, sealed audit) is inherited.

---

## 1. Literature review

Knowledge distillation trains a small *student* to imitate a *teacher*'s soft output distribution rather than only hard labels; the "dark knowledge" in the relative probabilities is the training signal ([Hinton, Vinyals & Dean, 2015, arXiv:1503.02531](https://arxiv.org/abs/1503.02531)). Four threads in this literature bear directly on the design.

**Capacity gap and the teacher-assistant result.** A *more accurate* teacher can produce a *worse* student when the teacher is too large relative to the student: the student cannot represent the teacher's distribution and the distillation loss pulls it toward an unreachable target ([Cho & Hariharan, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf)). Teacher-Assistant Knowledge Distillation (TAKD) inserts intermediate-sized networks so each hop bridges a *manageable* capacity gap ([Mirzadeh et al., AAAI 2020, arXiv:1902.03393](https://arxiv.org/abs/1902.03393)). The crucial implication for us: TAKD's benefit comes from the *size ratio*, so a same-size hop (400M→400M) gets TAKD's documented costs (extra compute, error accumulation down the chain) with none of its benefit. TAKD is therefore only a legitimate arm when the teacher is *much* larger than the student (our 7B stress test), not for same-tier chaining.

**Ensemble / multi-teacher distillation.** Distilling from multiple teachers beats distilling from the single best one through three mechanisms: variance reduction from averaging, cancellation of *uncorrelated* systematic errors (blind-spot cancellation), and richer, better-calibrated soft targets. Formally, a single-teacher student can at best match its teacher, whereas an ensemble-distilled student can provably exceed the *average* teacher ([Multi-Teacher Ensemble Distillation: A Mathematical Framework, arXiv:2601.09165](https://arxiv.org/pdf/2601.09165); survey and bounds in [Ensemble Knowledge Distillation — Emergent Mind](https://www.emergentmind.com/topics/ensemble-knowledge-distillation)). The canonical early result is [Fukuda et al., "Efficient Knowledge Distillation from an Ensemble of Teachers," Interspeech 2017](https://www.semanticscholar.org/paper/Efficient-Knowledge-Distillation-from-an-Ensemble-Fukuda-Suzuki/86dc692fc0b6ee97077ae4132517cb8538802bcc); reported gains of ~3–5% on CIFAR with *diminishing returns* as teachers are added (1T→3T→6T) appear in [Ensemble KD for CTR Prediction, arXiv:2011.04106](https://arxiv.org/pdf/2011.04106). The intuition that a student "fuses" diverse teacher predictions into a more robust understanding is laid out in the [Gou et al. KD survey, arXiv:2004.05937](https://arxiv.org/pdf/2004.05937). Caveat we must respect: **teacher accuracy poorly predicts student quality** — "distillability" must be measured, not assumed ([Cho & Hariharan, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf)).

**Peer / mutual / online distillation.** Training several networks that teach each other (Deep Mutual Learning, [Zhang et al., CVPR 2018, arXiv:1706.00384](https://arxiv.org/abs/1706.00384); online codistillation, [Anil et al., 2018, arXiv:1804.03235](https://arxiv.org/abs/1804.03235)) improves generalization primarily as a *regularizer/ensembling* effect. It does not create capability absent from all peers, and its benefit collapses when peers are correlated — the empirical reason our prior run was flat.

**On-policy and sequence-level distillation.** Off-policy KD trains on the teacher's own text, creating a train/deploy mismatch. Sequence-level KD distills on generated sequences ([Kim & Rush, EMNLP 2016, arXiv:1606.07947](https://arxiv.org/abs/1606.07947)); on-policy variants train on the *student's own* rollouts scored/corrected by the teacher and are more effective for small LMs ([MiniLLM: Gu et al., 2023, arXiv:2306.08543](https://arxiv.org/abs/2306.08543); Generalized KD / on-policy distillation, [Agarwal et al., 2023, arXiv:2306.13649](https://arxiv.org/abs/2306.13649)). This motivates our `_onpolicy` arms.

**Verifier-gated self-improvement.** When answers are checkable, sampling many completions and training only on the *verified-correct* ones injects genuine information via the verifier — STaR ([Zelikman et al., 2022, arXiv:2203.14465](https://arxiv.org/abs/2203.14465)) and rejection fine-tuning ([Yuan et al., 2023, arXiv:2308.01825](https://arxiv.org/abs/2308.01825)). This is the mechanism that makes "verified rescue" a real signal rather than mutual averaging, and it is why the task must be programmatically checkable.

**Verifiable task substrates suitable at 400M.** Instruction/format-following with programmatic checkers ([IFEval: Zhou et al., 2023, arXiv:2311.07911](https://arxiv.org/abs/2311.07911)) and small-state synthetic reasoning ([bAbI: Weston et al., 2015, arXiv:1502.05698](https://arxiv.org/abs/1502.05698)) are both learnable at GPT-2-medium scale and exactly checkable, unlike multi-digit arithmetic which is a floor task at this size.

**Synthesis for this experiment.** (i) Stay at 400M but choose verifiable tasks in a 30–60% base-accuracy band so rescues exist. (ii) The primary hypothesis — complementary peers beat a single larger teacher into the same class — is only mechanistically plausible when peers are *decorrelated specialists* and a *verifier* gates the rescues. (iii) The ensemble-teacher arm is the cheap, strong baseline the peer arm must beat to matter; a same-size chained-TA arm is expected to *underperform* direct distillation and is included only as a predicted-negative control, with a genuine TAKD chain reserved for the much-larger-teacher stress test.

---

## 2. Objectives

**Primary objective.** Under matched starts, data, student updates, decode budget, selection policy, and sealed evaluation, is the selected `peer_frr_onpolicy` 400M model better than the selected `single_teacher_diverse` 400M model on the held-out verifiable-task suite?
Primary estimand: paired-seed mean of `acc(peer_frr_onpolicy) − acc(single_teacher_diverse)`, with a lower-bound superiority gate.

**Secondary objectives.**

- Does `peer_frr_onpolicy` also beat the cheap `ensemble_teacher` baseline? (If not, the expensive peer machinery is not justified.)
- Isolate the contribution of on-policy training (`self_op` vs `base`), of the verifier (`gold_sft` vs `single_teacher`), and of teacher diversity (`single_teacher_diverse` vs `single_teacher`).
- Preserve a **single-model deployment endpoint**, not an ensemble or router.
- Keep retention (general text loss) and per-family specialty from collapsing.

**Non-goals.** General-intelligence claims; treating ensemble/router/best-of-four as the deployment claim; using the 7B teacher as the primary comparator (secondary stress test only); running training without an explicit operator unlock.

---

## 3. Model and environment (fixed defaults, override-able)

| Role | Default checkpoint | Params | Notes |
|---|---|---|---|
| Student (×4 peers) | `allenai/OLMo-2-0425-1B` DataDecide **~400M** sibling, or OLMo DataDecide 300M/400M checkpoint | ~370–400M | Same architecture and tokenizer across all four peers. Supplied via `OLMO_400M_MODEL`. |
| Larger teacher (primary) | `allenai/OLMo-2-0425-1B` | ~1B | Supplied via `OLMO_LARGER_TEACHER_MODEL`. |
| Extreme teacher (stress test only) | OLMo-2 7B | ~7B | Optional; only arm where a TAKD chain is legitimate. |

- If the operator has an internal 400M checkpoint, substitute its path; the four peers **must** all start from the *same* warmed checkpoint so arms differ only in training signal.
- Environment: Python 3.10+, CUDA, PyTorch, Transformers, NumPy. Compressed screen targets 2–4× B200, ~10 h wall-clock.
- **Locked by default.** Training will not run unless `ALLOW_OLMO400M_TRAINING=I_UNDERSTAND_THIS_RUNS_OPTIMIZATION` and an explicit `OLMO400M_RUN_MODE` are set.

---

## 4. Verifiable task suite (the data manifest)

Four synthetic task families, each with an exact programmatic checker and a difficulty knob. All examples are generated with a fixed seed into `local/<expdir>/data/manifest.json`; ground truth is produced by the generator, so verification is free and exact. Each family is one **peer slice** (see §6.2). A fifth family is a deliberate floor control.

For every family the operator runs a **calibration pass** (§4.6) that tunes the difficulty knob until the *base* (undistilled) 400M student lands in **30–60% exact-match accuracy**. This band is a hard requirement — it guarantees rescues exist and headroom remains.

### 4.1 Family A — Instruction / format-following (IFEval-style)
- **Input:** a short prompt plus 1–3 composable constraints, e.g. *"Write about rain. Constraints: exactly 2 sentences; contain the word 'petrichor'; all lowercase."*
- **Checker (deterministic):** for each constraint, a Python predicate — sentence count via a segmenter, substring/word-count checks, `text == text.lower()`, JSON `json.loads` + key-set equality, regex for line endings, etc. Example passes iff **all** constraints hold.
- **Difficulty knob:** number of simultaneous constraints (1 → easy, 4 → hard) and constraint types.
- **Why 400M-appropriate:** partial compliance is common; huge headroom; 100% checkable.

### 4.2 Family B — Structured transduction / normalization
- **Input:** a string plus a named transform, e.g. date normalization (`"March 3, 2020" → "2020-03-03"`), list sort/dedup, case conversion (`snake_case ↔ camelCase`), bracket/JSON well-formedness repair.
- **Checker:** apply the reference transform in Python; exact-match the model output. For JSON, `json.loads` + canonical dump comparison.
- **Difficulty knob:** input length, transform composition depth (1 vs 2 chained transforms).
- **Why appropriate:** deterministic, unlimited data, clean per-transform sub-slices.

### 4.3 Family C — Small-state symbolic reasoning (bAbI-style)
- **Input:** 2–5 supporting facts + a question, e.g. entity/state tracking ("*The key is in box A. The key is moved to box C. Where is the key?*"), boolean-expression evaluation over 2–3 variables, set membership.
- **Checker:** exact-match against the generator's tracked ground-truth answer.
- **Difficulty knob:** number of supporting facts and distractors.
- **Why appropriate:** bAbI-class tasks are learnable at small scale; exact-match verifiable.

### 4.4 Family D — Extractive QA / span copying
- **Input:** a short passage + a question whose answer is a verbatim span of the passage.
- **Checker:** exact-match (and span-F1 as a soft secondary) against the gold span.
- **Difficulty knob:** passage length and number of near-duplicate distractor spans.
- **Why appropriate:** tests retrieval/attention (a 400M strength) rather than reasoning; checkable.

### 4.5 Family E — Arithmetic floor control (predicted negative)
- **Input:** 3–4 digit multiplication.
- **Checker:** exact integer match.
- **Role:** **negative control.** We *expect* ≈0% base accuracy and no distillation signal. If any arm shows a jump here, treat it as a leakage/bug signal, not success.

### 4.6 Calibration and manifest build (`manifest_only` run mode)
1. Generate a 2,000-example probe set per family across a grid of the difficulty knob.
2. Evaluate the base 400M student (greedy) on each grid point.
3. Select, per family, the knob value whose base accuracy is closest to the center of [30%, 60%]. Record it in `data/manifest.json`.
4. Generate the frozen splits from the selected knob: **train pool 40k / dev 2k / sealed test 4k** per family, disjoint by construction.
5. The manifest records: generator version, seed, per-family knob, split sizes, and the checker module hash. `championship_seed` refuses to run if this manifest is missing.

---

## 5. Distillation mechanics (shared across arms)

- **Verified rollout:** sample *k* completions from a source model at temperature 0.8; a completion is **accepted** iff its family checker returns pass. Accepted completions are the training targets.
- **On-policy:** rollouts are sampled from the *student being trained* (`_op`/`_onpolicy` arms); the teacher/peers only *score* (via checker) and, where valid, provide token-level soft targets.
- **Token-level KL vs sequence fallback:** token-level KL is used **only** when teacher and student share tokenizer and stage compatibility. When they do not (e.g. teacher not stage-matched), arms fall back to **sequence-level distillation** on accepted completions and log `distill_mode="sequence_fallback"` explicitly. Never silently apply invalid token-KL.
- **Teacher superiority gate:** before any teacher arm trains, the larger teacher must exceed the base student by ≥10 absolute points of suite accuracy. A teacher that fails this cannot serve as a comparator; the run records the gate result and skips teacher arms if it fails.
- **Selection policy (identical for all arms):** train each arm's four candidates, select the single candidate with the best **dev** suite accuracy (ties broken by lower general-text loss). The selected candidate is the deployable endpoint; sealed test is touched only after freezing.

---

## 6. Arms and the matched budget

### 6.1 Arms
All arms start from the same four warmed peer checkpoints and are matched on the budget in §6.3.

| Arm | Signal source | Role |
|---|---|---|
| `base` | none (eval only) | Floor reference. |
| `gold_sft` | SFT on generator ground-truth answers | Isolates "does a verifier/label help at all." |
| `self_op` | student's own verified-correct on-policy rollouts | Isolates on-policy self-improvement (STaR/RFT-style). |
| `single_teacher` | 1B teacher, greedy, off-policy | Vanilla single-teacher KD. |
| `single_teacher_diverse` | 1B teacher, diverse sampling (*k*=8) + verifier-gated | **Primary comparator.** |
| `ensemble_teacher` | aggregated distribution / verified union of all 4 peers | **Cheap strong baseline** the peer arm must beat. |
| `peer_frr_onpolicy` | complementary specialist peers + verifier-gated on-policy rescue | **Hypothesis under test.** |
| `chained_ta_samesize` *(optional)* | 1B→peer A(400M)→peers B/C/D(400M) | **Predicted-negative control** (same-size hop). |
| `takd_chain_7b` *(optional stress test)* | 7B→1B→400M genuine capacity chain | Only legitimate TAKD chain. |

**Primary comparison:** `peer_frr_onpolicy − single_teacher_diverse`.
**Secondary comparison:** `peer_frr_onpolicy − ensemble_teacher`.

### 6.2 Slices for the peer arm
For `peer_frr_onpolicy`, assign one family per peer as its **specialty slice**:

| Peer | Specialty slice (majority of its warmup) |
|---|---|
| peer 0 | Family A (format-following) |
| peer 1 | Family B (transduction) |
| peer 2 | Family C (symbolic reasoning) |
| peer 3 | Family D (extractive QA) |

Each peer is first specialized (distilled from the 1B teacher restricted to its slice), producing decorrelated experts. Then verifier-gated on-policy rescue runs across all four families: for each family, each peer samples on-policy; whenever another peer has a *verified-correct* answer the current peer missed, that answer becomes a training target ("failure→rescue→replay", FRR). Every peer is evaluated on **all** families, so success requires absorbing others' specialties, not just keeping its own.

### 6.3 Matched budget (per seed, per arm)
Budgets are equalized so no arm wins by spending more. Concrete defaults for the B200 screen:

- **Student updates:** 3,000 optimizer steps, batch 64, seq len 512 — identical for every arm.
- **Decode budget (target generation):** ≤ 200k accepted-target tokens per arm. The peer population's *combined* decode budget equals the single-teacher arm's decode budget — this is the "fair attempt budget" guarantee; `single_teacher_diverse` gets the same *k*×tokens the four peers collectively get.
- **Verifier calls:** ≤ 1.2M checker invocations per arm (cheap; logged).
- **Retention regularizer:** every arm interleaves a fixed 5% of general-text LM loss (from `OLMO400M_RETENTION_JSONL`) to keep retention comparable.

All of the above are recorded per arm in `cost_events.jsonl`: student updates, processed tokens, auxiliary tokens, attempted outputs, decoded tokens, accepted targets, device time, evaluation counts.

---

## 7. Execution profile and run modes

Environment (compressed screen):

```bash
export ALLOW_OLMO400M_TRAINING=I_UNDERSTAND_THIS_RUNS_OPTIMIZATION
export OLMO400M_BUDGET_PROFILE=b200_10h
export OLMO400M_B200_GPUS=4
export OLMO_400M_MODEL=/path/to/olmo_400m_student
export OLMO_400M_STAGE=<student_stage_label>
export OLMO_LARGER_TEACHER_MODEL=/path/to/olmo_1b_teacher
export OLMO_LARGER_TEACHER_STAGE=<teacher_stage_label>
export OLMO400M_RETENTION_JSONL=/path/to/retention_general_text.jsonl
export OLMO400M_EXPERIMENT_DIR=/path/to/output_dir   # must be under local/ for artifacts kept in-repo
```

Run order (fail-safe):

1. `OLMO400M_RUN_MODE=manifest_only` — once. Calibrates difficulty (§4.6), builds `data/manifest.json`, runs the teacher superiority gate. Refuses if difficulty calibration cannot place a family in [30%, 60%] (report which family failed and stop).
2. `OLMO400M_RUN_MODE=championship_seed` — one process per B200. Trains all arms for that seed into `seed_<n>/`. Refuses under `b200_10h` if the shared manifest is missing.
3. Use seeds **13, 29, 47, 71** for the screen (assign by `OLMO400M_B200_GPUS`). Confirmatory 16-seed list: `13,29,47,71,101,113,131,149,163,181,199,211,229,241,263,281`.
4. `OLMO400M_RUN_MODE=summarize` — after all configured seeds finish. Refuses if any configured seed is missing. Produces `confirmatory_summary.json`.

The screen is a **large-effect screen**, not confirmatory. A positive mean with a failing lower-bound gate at 2–4 seeds reflects limited power and must not be read as proof of failure. The profile does not enforce a wall-clock abort; use scheduler/process timeouts and sync ephemeral NVMe before shutdown.

---

## 8. Analysis plan and success metrics

**Primary metric.** `peer_vs_single_teacher_diverse_primary_gate_passed` in `confirmatory_summary.json`, defined as: paired-seed mean of suite exact-match accuracy difference `peer_frr_onpolicy − single_teacher_diverse` with a **BCa bootstrap 95% lower bound > 0** (10k resamples, paired by seed).

**Secondary gates.**
- `peer_vs_ensemble_teacher`: same paired test against `ensemble_teacher`. (Interpretation: peer machinery is justified only if this is also > 0.)
- Ablation ladder (report effect + CI, no gate): `self_op − base`, `gold_sft − single_teacher`, `single_teacher_diverse − single_teacher`, `chained_ta_samesize − single_teacher` (predicted ≤ 0).

**Safety / integrity gates (must all pass for a valid claim).**
- **Retention:** selected model's general-text loss ≤ base + 2% (no capability narrowing).
- **Specialty preservation:** no per-family accuracy drops below base for the peer arm (complementarity not bought by forgetting a slice).
- **Floor control:** Family E shows no significant accuracy change in any arm (else suspect leakage/bug).
- **Decorrelation check:** peers' per-example error vectors have mean pairwise correlation < 0.7 after specialization (else "complementary peers" is not actually complementary — report and treat peer result as suspect).

**Reporting.** Per-seed and pooled tables of suite accuracy by arm; paired differences with 95% CIs; per-family breakdown; the decorrelation matrix; and the all-in cost ledger by seed and arm. Sealed test is reported once, after all arms for all configured seeds are frozen.

---

## 9. Outputs

Required artifacts (all under `OLMO400M_EXPERIMENT_DIR`):

- `data/manifest.json` — calibrated difficulty knobs, splits, checker hash, generator seed.
- `model_manifest.json`, `larger_teacher_manifest.json` — resolved checkpoints, stages, tokenizer compatibility, `distill_mode`.
- `teacher_superiority_gate.json` — gate result.
- `seed_*/seed_results.json` — per-arm dev/test accuracy, per-family breakdown, selected candidate.
- `seed_*/sealed_audit.json` — sealed test results (written only after freeze).
- `seed_*/*/preaudit_result.json` — per-arm pre-freeze dev metrics.
- `seed_*/*/cost_events.jsonl` — full cost ledger.
- `seed_*/decorrelation.json` — peer error-correlation matrix.
- `confirmatory_summary.json` — primary + secondary gates, ablation ladder, safety gates.

Optional: `seed_*/per_example_traces.jsonl` (rescue provenance for audit).

---

## 10. Acceptance criteria

- `manifest_only` places every non-control family in [30%, 60%] base accuracy, or fails loudly naming the family.
- `b200_10h` fails fast without an explicit `OLMO400M_RUN_MODE`.
- `championship_seed` fails fast if the shared manifest is missing.
- `summarize` refuses unless all configured seeds are present.
- Teacher arms do not train unless the superiority gate passes; `distill_mode` is recorded (token-KL or sequence fallback).
- All four peers verifiably start from the same checkpoint hash.
- Every arm's cost ledger shows equal student updates and decode budget within 5%.
- No model weights, private retention corpus, or run outputs are committed; all authored files live under `local/`; nothing is pushed to GitHub.

---

## 11. Risks and open questions

**Risks.**
- No 400M checkpoint lands any family in the sweet spot → the whole premise fails; mitigate by widening the difficulty grid before abandoning.
- Teacher not stage-matched → sequence-distillation fallback (report explicitly; token-KL comparison then unavailable).
- 2–4 seeds underpower the lower-bound gate despite a positive mean (screen, not proof).
- Synthetic generators leak answer structure into inputs → inflated, meaningless accuracy; the floor control (Family E) and a held-out generator seed guard against this.
- Peers fail the decorrelation check → "complementary" claim void; report as such.
- Ephemeral NVMe loss on shutdown; no built-in wall-clock abort.

**Open questions.**
- Exact 400M checkpoint and stage label?
- Is the 1B teacher stage-matched, or is sequence fallback expected?
- General-text source for the retention regularizer?
- Run the optional `chained_ta_samesize` and `takd_chain_7b` arms in the screen, or defer to the confirmatory study?
- Confirmatory study seed count: 16 as listed, or expand for the secondary (peer vs ensemble) gate?
