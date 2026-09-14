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

No AWS credential is used anywhere in this path. The only secret involved is
the W&B API key, pushed to the run directory by `push_wandb_key.sh` as
`wandb-session.env` (mode 600, on scratch only). Supply it either as
`WANDB_API_KEY` in the environment -- preferred, since it never touches the
filesystem -- or on one line in `~/.wandb_api_key`, or point
`WANDB_KEY_FILE` at a file. The key is never echoed, never passed as a
command argument, and never written inside this repository; the submit
script reports only a byte count.

`submit_from_laptop.sh` needs both git (to archive the commit) and ssh (to
reach scratch). Where those live in different environments -- git on
Windows, the control socket inside WSL, for instance -- run the phases
separately: `PHASE=build` with a shared `STAGING` directory where git works,
then `PHASE=ship` plus `push_wandb_key.sh` where the socket works. Both of
the latter need ssh only.
