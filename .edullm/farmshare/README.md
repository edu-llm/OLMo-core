# Curriculum 370M on FarmShare (8 × L40S)

Nine approved arms via `ARM_INDEX` 0–8:
`linear10-flesch`, `linear10-mtld`, `linear10-learn`, `warmup-flesch`,
`interleave-flesch`, `control`, `quadratic10-mtld`, `warmup-mtld`,
`warmup-linear10-mtld`.

```bash
cd /mnt/c/alpha_ai/OLMo-core-curriculum-370m
ARM_INDEX=0 bash .edullm/farmshare/submit_from_laptop.sh
```

Each submit stages parent RegMix plus the arm's order manifest before training
(control stages only the parent). Restage when switching arms (`SKIP_TRAIN=1`
to stage only).
