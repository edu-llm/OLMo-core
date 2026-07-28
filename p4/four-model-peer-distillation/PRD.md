# PRD: P4 four-model peer distillation

## Summary

This project delivers a controlled experiment for testing whether four
complementary OLMo-compatible 400M peers can train a better single deployable
400M model than matched 400M students distilled from a stronger larger teacher.
The primary outcome is selected `peer_frr_onpolicy` 400M minus selected
`large_teacher_diverse` 400M under matched starts, data, student updates,
selection policy, and sealed evaluation.

The 2-4 B200 profile is a compressed large-effect screen designed for a roughly
10-hour allocation. It is not the full 16-seed confirmatory study.

## P4 report alignment

This is the peer-distillation section of the broader P4 learning-science agenda:
testing which human-learning-inspired principles transfer to machine
pretraining, which weaken, and which should be discarded. The final report's
discipline applies here: the paired seed is the experimental unit; audit items
are measurements, not independent replications; null results should be reported
as nulls rather than rescued by secondary contrasts; and any short-run B200 result
must be framed as a large-effect screen unless it is later replicated at the
planned confirmatory seed count.

This section does not claim to validate human learning science directly. It asks
a machine-pretraining question: whether a population-training topology can
transfer useful peer-discovered signal into a single 400M model better than a
larger-teacher distillation topology under matched controls.

## Problem

Ordinary teacher-student distillation can compress capability, but it may also
overfit the student to one teacher distribution. The hypothesis is that a small
population of complementary peers can expose each student to verified rescues
from other peers, creating useful training signal that is harder for a single
larger teacher to transfer into the same 400M deployment class.

The experiment must separate that hypothesis from easier but weaker claims, such
as best-of-four ensembling, router performance, or a single greedy teacher miss.

## Goals

- Test the primary peer-learning versus larger-teacher-distillation comparison
  with symmetric four-student selection.
- Preserve a one-model deployment endpoint, not an ensemble endpoint.
- Use the same four warmed student checkpoints across all championship arms.
- Give `large_teacher_diverse` a fair attempt budget against the peer population.
- Keep safety gates for retention, specialty preservation, shift behavior, and
  complementarity collapse.
- Produce auditable artifacts: manifests, per-seed results, sealed audit bundles,
  preaudit results, and cost ledgers.
- Support a compressed 2-4 B200 execution profile without changing the scientific
  arms after outcomes are visible.
- Report pair-level outcomes and uncertainty honestly, including positive means
  that fail the lower-bound gate at small seed count.

## Non-goals

- Proving broad general intelligence improvement.
- Claiming that the result predicts child learning, classroom outcomes, or human
  forgetting curves.
- Claiming open-ended creativity from lexical novelty or embedding distance.
- Treating a router, ensemble, or best-of-four population result as the primary
  deployment claim.
- Using a 7B teacher as the primary comparator. Extreme-capacity teachers are
  secondary stress tests only.
- Running training without explicit operator unlock and run-mode selection.

## Users and stakeholders

- Research operator: runs the notebook, supplies model and data paths, and
  preserves outputs.
- Team lead: approves compute spend, verifies protocol integrity, and interprets
  whether the result is a large-effect screen or confirmatory evidence.
- Research reviewer: inspects manifests, comparisons, safety gates, and cost
  accounting before accepting claims.

## Required inputs

- HF-loadable OLMo-compatible 400M student checkpoint and tokenizer.
- Student checkpoint stage label via `OLMO_400M_STAGE`.
- Approximately 1B OLMo-compatible larger teacher checkpoint and tokenizer.
- Teacher stage label via `OLMO_LARGER_TEACHER_STAGE`.
- Held-out retention JSONL with one `{"text": "..."}` record per line.
- Fast writable output directory for manifests, checkpoints, and ledgers.
- Python 3.10+ environment with CUDA, PyTorch, Transformers, and NumPy.

The retention corpus must be held-out general text, not the synthetic arithmetic
items, benchmark answer keys, or reshaped multiple-choice prompts. If retention
is weak or absent, the result can still screen the peer-learning mechanism but
cannot support an "improved without narrowing" claim.

## Core arms

The championship run must include exactly these arms:

- `gold_private_equal_cost`
- `self_snapshot_op`
- `peer_frr_onpolicy`
- `large_teacher_single`
- `large_teacher_diverse`

The primary comparator is `large_teacher_diverse`, not a single greedy teacher
trace.

## Functional requirements

1. The notebook must remain locked unless
   `ALLOW_OLMO400M_TRAINING=I_UNDERSTAND_THIS_RUNS_OPTIMIZATION`.
2. The `b200_10h` profile must require an explicit `OLMO400M_RUN_MODE`.
3. `manifest_only` must create the shared deterministic data manifest before
   parallel seed jobs begin.
4. `championship_seed` must refuse to run under `b200_10h` if the shared manifest
   is missing.
5. Parallel seed jobs must write to separate `seed_<n>` directories.
6. `summarize` must refuse to summarize if any configured confirmatory seed is
   missing.
7. The larger teacher must pass the superiority gate before championship teacher
   arms train.
8. Token-level KL must be used only when tokenizer and stage compatibility make it
   valid; otherwise teacher arms must use verified sequence-distillation fallback.
9. The final audit must open only after all requested arms are frozen.
10. Cost ledgers must record student updates, processed tokens, auxiliary tokens,
    attempted outputs, decoded tokens, accepted targets, device time, and
    evaluation counts.
11. Analysis must treat the population-training seed as the replicate. Thousands
    of generated audit items must not be used to manufacture narrow confidence
    intervals.
12. If the lower-bound gate fails at two to four seeds while the mean is positive,
    the result must be reported as underpowered or inconclusive, not as evidence
    that peer learning failed.

## B200 compressed execution

Use this profile for a 2-4 B200, roughly 10-hour run:

```bash
export ALLOW_OLMO400M_TRAINING=I_UNDERSTAND_THIS_RUNS_OPTIMIZATION
export OLMO400M_BUDGET_PROFILE=b200_10h
export OLMO400M_B200_GPUS=4
export OLMO_400M_MODEL=/path/to/olmo_400m_student
export OLMO_400M_STAGE=<student_stage_label>
export OLMO_LARGER_TEACHER_MODEL=/path/to/olmo_1b_teacher
export OLMO_LARGER_TEACHER_STAGE=<teacher_stage_label>
export OLMO400M_RETENTION_JSONL=/path/to/retention_general_text.jsonl
export OLMO400M_EXPERIMENT_DIR=/path/to/output_dir
```

Run order:

1. Run `OLMO400M_RUN_MODE=manifest_only` once.
2. Run one `OLMO400M_RUN_MODE=championship_seed` process per B200.
3. Use seeds `13`, `29`, `47`, and `71` according to `OLMO400M_B200_GPUS`.
4. Run `OLMO400M_RUN_MODE=summarize` after all configured seed jobs finish.

The compressed profile should be reported as a large-effect screen. A positive
mean with a failing lower-bound gate at two to four seeds should not be
overinterpreted as proof of failure.

The notebook does not enforce the 10-hour wall-clock budget by itself. Operators
should use scheduler, process-level timeout, or capacity-reservation controls and
must sync outputs off ephemeral storage before instance shutdown.

## Outputs

Required output artifacts:

- `data/manifest.json`
- `model_manifest.json`
- `larger_teacher_manifest.json`
- `seed_*/seed_results.json`
- `seed_*/sealed_audit.json`
- `seed_*/*/preaudit_result.json`
- `seed_*/*/cost_events.jsonl`
- `confirmatory_summary.json`

Optional output artifact:

- `seed_*/fixed_k_novelty.json`

If the fixed-k novelty audit is omitted, success level 4 remains incomplete by
design. The 10-hour core claim is the level-3 peer-vs-larger-teacher comparison.

## Success metrics

Primary metric:

- `peer_vs_larger_teacher_primary_gate_passed` in `confirmatory_summary.json`.

Supporting metrics:

- paired seed analysis for `peer_frr_onpolicy` versus `large_teacher_diverse`;
- mean paired percentage-point difference and confidence interval, even when the
  boolean primary gate is false;
- safety gate status;
- general retention margin buffers;
- expertise retention by peer;
- shift noninferiority;
- raw larger-teacher moonshot result;
- all-in cost ledger by seed and arm.

Interpretation ladder:

- Level 1: peer learning beats ordinary/private and self-snapshot controls.
- Level 2: peer learning beats `large_teacher_single`.
- Level 3: peer learning beats `large_teacher_diverse`; this is the 10-hour core
  claim.
- Level 4: teacher-external fixed-k novelty passes, if the optional novelty audit
  is run.
- Level 5: selected peer-trained 400M beats the raw larger teacher.

## Acceptance criteria

- The notebook hash matches the recorded expected SHA-256.
- All notebook code cells parse and contain no embedded execution outputs.
- The generator compiles.
- `b200_10h` fails fast without explicit `OLMO400M_RUN_MODE`.
- `championship_seed` fails fast if the shared manifest is missing.
- `summarize` requires all configured confirmatory seeds.
- The branch contains no model weights, private retention corpus, local output
  directories, or generated checkpoints.

## Risks

- No valid HF-loadable 400M student checkpoint is available.
- The public 1B teacher is not stage-matched to the internal 400M student,
  forcing sequence-distillation fallback.
- The 2-4 seed B200 run has limited statistical power and may fail the lower-bound
  superiority gate despite a positive mean.
- The synthetic arithmetic task family supports a controlled mechanism test, not
  a broad capability claim.
- A single checkpoint or single seed is not a valid experimental unit; the
  configured paired seed set must be completed or explicitly reframed.
- A poor retention corpus can weaken the "improved without narrowing" claim.
- The B200 profile does not enforce a wall-clock abort by itself; operators should
  use scheduler or process-level timeout controls.
- Outputs on ephemeral NVMe must be synced before instance shutdown.

## Open questions

- What exact HF-loadable 400M checkpoint will be used?
- What stage label should be assigned to the student checkpoint?
- Is the primary teacher truly stage-matched, or should the run explicitly report
  sequence-distillation fallback?
- What held-out general text source will be used for retention?
- Will the optional fixed-k novelty audit be run after the 10-hour core?
