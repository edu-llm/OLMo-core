# P4 four-model peer distillation

This bundle contains the OLMo-compatible 400M four-peer peer-learning protocol.
It tests whether four complementary 400M peers can produce a better single
deployable 400M model than matched 400M students distilled from a demonstrably
stronger larger teacher. The primary comparison is selected `peer_frr_onpolicy`
400M versus selected `large_teacher_diverse` 400M under matched starts, data,
student updates, selection policy, and sealed evaluation.

## Files

- `OLMo400M_four_peer_peer_learning.ipynb` - runnable notebook artifact.
- `build_olmo400m_champion_notebook.py` - source generator for the notebook.
- `research_brief.md` - research and protocol rationale.

Expected notebook SHA-256:

```text
1b151959aa4593fef9b72e411df85b5fb6a3592e688016ab7f036cfe652cdc7e
```

## Required inputs

The notebook does not include model weights or retention data. Operators must
supply:

- an HF-loadable OLMo-compatible 400M student checkpoint and tokenizer;
- a stage-labeled approximately 1B OLMo-compatible larger teacher checkpoint and tokenizer;
- a held-out retention JSONL with one `{"text": "..."}` record per line;
- a writable output directory on fast local storage.

## B200 10-hour profile

For a 2-4 B200 allocation, use the compressed large-effect screen profile:

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

Create the shared manifest once:

```bash
export OLMO400M_RUN_MODE=manifest_only
```

Then launch one `championship_seed` process per B200, using seeds `13`, `29`,
`47`, and `71` according to `OLMO400M_B200_GPUS`. After all configured seeds
finish, run:

```bash
export OLMO400M_RUN_MODE=summarize
```

The compressed profile is not the full 16-seed confirmatory study. If the optional
fixed-k novelty audit is not run separately, success level 4 will remain incomplete;
the 10-hour core claim is the level-3 peer-vs-larger-teacher comparison.
