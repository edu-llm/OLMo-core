# Curriculum-new 370M on FarmShare (4 x L40S)

Seven pacings, three difficulty metrics, two LR schedules -- see plan
section 6b/6c for the full campaign design. A run is fully specified by
`PACING`, `LR_SCHEDULE`, `SEED`, and (for non-control pacings) `METRIC`:

- `PACING`: `control`, `linear_n10`, `anti_linear_n10`, `quadratic_n10`,
  `warmup_ramp_1000`, `warmup_quadratic_n10_1000`, `interleave_i10_linear`
- `METRIC` (required unless `PACING=control`): `compression_ratio`, `mtld`,
  `flesch`
- `CONTROL_METRIC` (required when `PACING=control`): which metric's
  materialized parent to train control on
- `LR_SCHEDULE`: `constant` or `cosine`
- `SEED`: any integer

The corpus rebuild (plan section 4/5) must have already materialized every
metric's parent under `CORPUS_ROOT` (default
`/scratch/users/<sunet>/curriculum-new-corpus/<metric>/manifest.json`)
before any training submission -- there is no per-run staging step.

```bash
cd /mnt/c/alpha_ai/OLMo-core-curriculum-new
PACING=linear_n10 METRIC=mtld LR_SCHEDULE=constant SEED=0 \
  bash .edullm/farmshare/submit_from_laptop.sh

PACING=control CONTROL_METRIC=mtld LR_SCHEDULE=constant SEED=0 \
  bash .edullm/farmshare/submit_from_laptop.sh
```

No AWS credential is used anywhere in this path -- only the W&B session
(`push_wandb_session_to_farmshare.sh`) is pushed to the run directory.
