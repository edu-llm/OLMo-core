# Curriculum 370M on FarmShare (8 × L40S)

Five approved arms via `ARM_INDEX` 0–4:
`linear10-flesch`, `linear10-mtld`, `linear10-learn`, `warmup-mtld`,
`interleave-mtld`.

```bash
cd /mnt/c/alpha_ai/OLMo-core-curriculum-370m
ARM_INDEX=0 bash .edullm/farmshare/submit_from_laptop.sh
```

Each submit stages parent RegMix plus the arm's order manifest before training.
Restage when switching arms (`SKIP_TRAIN=1` to stage only).
