# Four-Model Peer Distillation: An Experimental Protocol

**Can a small population of complementary peers train a better deployable model than distillation from a single larger teacher?**

Andrew · P4 Four-Model Peer Distillation

> **Status:** This is a pre-registered experimental design and execution protocol, not a results writeup. It specifies the arms, the primary comparison, the safety gates, and the artifacts the run must produce before any outcomes are examined. Results are reported separately once the run completes.

---

## 1. Introduction and motivation

Knowledge distillation is the standard route to compressing capability into a smaller, cheaper-to-serve model: a student learns to imitate a stronger teacher. It works, but it carries a known failure mode. A single teacher exposes the student to a single distribution of behavior, and the student can end up overfit to that teacher's particular way of solving problems rather than acquiring robust, transferable competence.

This project tests an alternative. Instead of one large teacher, we use a small population of complementary same-size peers. The hypothesis is that peers with different strengths can surface verified "rescues"—cases where one peer succeeds on inputs another fails—and that this cross-peer signal is a richer, harder-to-fake training source than anything a single larger teacher can transfer into the same small deployment class.

The design is built to keep that hypothesis honest. Peer-learning setups are easy to confuse with weaker claims that would look like success without supporting the actual mechanism: best-of-four ensembling, a routing policy that picks the right model per input, or simply catching a single greedy teacher on a bad day. The protocol is structured to separate the peer-learning hypothesis from all of these.

Two boundaries are worth stating up front. First, the deployment endpoint is one model, not an ensemble or a router—any population-level or best-of-four result is explicitly *not* the primary claim. Second, the primary comparator is a *diverse* larger-teacher arm given a fair attempt budget, not a single greedy teacher trace, because beating a deliberately weak baseline would prove nothing.

## 2. Objectives

**Primary objective.** Test the peer-learning-versus-larger-teacher comparison directly, with symmetric four-student selection: does the selected `peer_frr_onpolicy` 400M model outperform the selected `large_teacher_diverse` 400M model under matched starts, data, student updates, selection policy, and sealed evaluation?

**Supporting objectives.** The run preserves a single-model deployment endpoint rather than an ensemble; uses the same four warmed student checkpoints across every championship arm so arms differ only in training signal; gives the `large_teacher_diverse` arm a fair attempt budget against the peer population; maintains safety gates covering retention, specialty preservation, distribution-shift behavior, and complementarity collapse; and produces a fully auditable trail including manifests, per-seed results, sealed audit bundles, pre-audit results, and cost ledgers.

**Explicit non-goals.** This protocol does not aim to prove broad general-intelligence improvement, claim open-ended creativity from lexical novelty or embedding distance, treat a router, ensemble, or best-of-four population result as the primary deployment claim, use a 7B teacher as the primary comparator except as a secondary stress test, or run any training without an explicit operator unlock and run-mode selection.

## 3. Scope and stakeholders

The protocol assumes three roles. A **research operator** runs the notebook, supplies model and data paths, and preserves outputs. A **team lead** approves compute spend, verifies protocol integrity, and judges whether a given run counts as a large-effect screen or as confirmatory evidence. A **research reviewer** inspects manifests, comparisons, safety gates, and cost accounting before any claim is accepted.

## 4. Required inputs

The run requires an HF-loadable, OLMo-compatible 400M student checkpoint and tokenizer, with a stage label supplied via `OLMO_400M_STAGE`; an approximately 1B OLMo-compatible larger-teacher checkpoint and tokenizer, with a stage label supplied via `OLMO_LARGER_TEACHER_STAGE`; a held-out retention corpus as JSONL with one `{"text": "..."}` record per line; a fast, writable output directory for manifests, checkpoints, and ledgers; and a Python 3.10+ environment with CUDA, PyTorch, Transformers, and NumPy.

## 5. Experimental design

### 5.1 Arms

The championship run must include exactly five arms: `gold_private_equal_cost`, `self_snapshot_op`, `peer_frr_onpolicy`, `large_teacher_single`, and `large_teacher_diverse`. The `peer_frr_onpolicy` arm is the peer-learning arm under test, and `large_teacher_diverse` is the primary comparator.

The primary comparison is `peer_frr_onpolicy` minus `large_teacher_diverse`. All arms start from the same four warmed student checkpoints and are matched on data, student updates, selection policy, and sealed evaluation, so any measured difference is attributable to the training signal rather than to a difference in starting point or budget.

### 5.2 Distillation validity

Token-level KL distillation is only valid when tokenizer and stage compatibility actually hold between teacher and student. When they do not, the teacher arms must fall back to a verified sequence-distillation procedure rather than silently applying an invalid token-level objective. Because the public 1B teacher may not be stage-matched to the internal 400M student, the run should be prepared to report a sequence-distillation fallback explicitly.

### 5.3 Superiority gate on the teacher

Before any championship teacher arm trains, the larger teacher must pass a superiority gate. A teacher that is not demonstrably stronger cannot serve as a meaningful comparator, so this gate protects the primary comparison from being run against an inadequate teacher.

## 6. Execution controls and safeguards

The notebook is designed to fail safe rather than to run optimization by accident. The notebook stays locked unless `ALLOW_OLMO400M_TRAINING=I_UNDERSTAND_THIS_RUNS_OPTIMIZATION`, and the `b200_10h` profile requires an explicit `OLMO400M_RUN_MODE`. The `manifest_only` mode must create the shared deterministic data manifest before any parallel seed jobs begin, `championship_seed` must refuse to run under `b200_10h` if the shared manifest is missing, parallel seed jobs must write to separate `seed_<n>` directories, and `summarize` must refuse to run if any configured confirmatory seed is missing.

The larger teacher must pass the superiority gate before championship teacher arms train. Token-level KL is used only when tokenizer and stage compatibility make it valid; otherwise, teacher arms use the verified sequence-distillation fallback. The final audit opens only after all requested arms are frozen. Cost ledgers record student updates, processed tokens, auxiliary tokens, attempted outputs, decoded tokens, accepted targets, device time, and evaluation counts.

## 7. Compressed B200 execution profile

The `b200_10h` profile is a compressed large-effect screen designed for a roughly 10-hour allocation on 2–4 B200 GPUs. It is deliberately *not* the full 16-seed confirmatory study, and it should be reported as a screen: a positive mean paired with a failing lower-bound gate at two-to-four seeds is a matter of limited statistical power and should not be overinterpreted as proof of failure.

**Environment.**

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

**Run order.** Run `OLMO400M_RUN_MODE=manifest_only` once to build the shared manifest. Then run one `OLMO400M_RUN_MODE=championship_seed` process per B200, using seeds 13, 29, 47, and 71 according to `OLMO400M_B200_GPUS`. After all configured seed jobs finish, run `OLMO400M_RUN_MODE=summarize`.

## 8. Analysis plan and success metrics

**Primary metric.** The primary metric is `peer_vs_larger_teacher_primary_gate_passed` in `confirmatory_summary.json`.

**Supporting metrics.** Supporting metrics include paired-seed analysis of `peer_frr_onpolicy` versus `large_teacher_diverse`, safety gate status, general retention margin buffers, expertise retention by peer, shift noninferiority, raw larger-teacher moonshot result, and the all-in cost ledger by seed and arm.

## 9. Expected outputs

**Required artifacts.** The required artifacts are `data/manifest.json`, `model_manifest.json`, `larger_teacher_manifest.json`, `seed_*/seed_results.json`, `seed_*/sealed_audit.json`, `seed_*/*/preaudit_result.json`, `seed_*/*/cost_events.jsonl`, and `confirmatory_summary.json`.

**Optional artifact.** The optional artifact is `seed_*/fixed_k_novelty.json`. If the fixed-k novelty audit is omitted, success level 4 remains incomplete by design. The 10-hour core claim is the level-3 peer-versus-larger-teacher comparison.

## 10. Acceptance criteria

The acceptance criteria are that the notebook hash matches the recorded expected SHA-256; all notebook code cells parse and contain no embedded execution outputs; the generator compiles; `b200_10h` fails fast without an explicit `OLMO400M_RUN_MODE`; `championship_seed` fails fast if the shared manifest is missing; `summarize` requires all configured confirmatory seeds; and the branch contains no model weights, private retention corpus, local output directories, or generated checkpoints.

## 11. Risks and open questions

**Risks.** Key risks are that no valid HF-loadable 400M student checkpoint may be available; the public 1B teacher may not be stage-matched to the internal 400M student, forcing the sequence-distillation fallback; the 2–4 seed B200 run has limited statistical power and may fail the lower-bound superiority gate despite a positive mean; a poor retention corpus can weaken the "improved without narrowing" claim; the B200 profile does not enforce a wall-clock abort on its own, so operators should rely on scheduler- or process-level timeout controls; and outputs on ephemeral NVMe must be synced before instance shutdown.

**Open questions.** The open questions are which exact HF-loadable 400M checkpoint will be used, what stage label should be assigned to the student checkpoint, whether the primary teacher is truly stage-matched or the run should explicitly report a sequence-distillation fallback, what held-out general-text source will be used for retention, and whether the optional fixed-k novelty audit will be run after the 10-hour core.
